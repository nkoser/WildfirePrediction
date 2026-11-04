"""Bekaempfungs-Auswertung (suppression_eval Teil B) auf allen Test-Stichproben aller vier Splits, GPU.

Je Split (cache_G zeitlich, cache_G_stratified zufaellig, cache_G_yearstrat Jahre, cache_G_blockstrat Bloecke)
und Modell werden Val und Test bewertet (Ausschnittskopf, Score F(dt + 1) bei Stichtag e = D - dt, Ziel Feuer im
Ausschnitt an D + 1, wie focus_eval), je Seed und dt per Platt auf dem Val-Split kalibriert und dann ueber die
Seeds gemittelt. Modelle:
  STS-LH / STS-Contagion (cache)  sts_<cache>_next_cache_splitcmp_{lh,cg}_s<s>_s<s>.pt, alle Splits (von null,
                                  nur Cache-Stichproben des Train-Splits); Kopf-Bias = Ereignisraten des Train-Splits
  STS-LH / STS-Contagion (Wuerfel) sts_extreme_[<sp>_]{lh,cg}_ref_s<s>_s<s>.pt, nicht beim Zufallssplit (Wuerfel-
                                  Vortraining wuerde dort in den Test lecken); Kopf-Bias aus dem Checkpoint
  3D U-Net                        predictions/paper/rec_<cache>_cache_next_paperhp_w0.npz (gespeicherte Val/Test-Scores)
Gruppen wie suppression_eval: Bevoelkerungs-Terzile des Ausschnitts (je Split auf den Test-Stichproben mit Feuer im
Fenster), FWI-Haelften, Bekaempfungsklasse des groessten NFDB-Brandes (>= 1 ha) mit Zuendung im Ausschnitt (+ dp)
in [D - 30, D]. Kennzahl: Quotient mean(p) / mean(y) (roh und Platt) mit Bootstrap-Intervall, PR-AUC.

Ausgabe NEU: metrics/suppression_splits<tag>/{summary.md, groups.csv, scores.npz}, Manifest p55_suppression_splits<tag>.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import sts_extreme, targets
from wildfire.paper.compare import DTS
from wildfire.paper.paper_metrics_ckpt import platt
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import CENTRE, doy_of
from wildfire.paper.suppression_eval import PCLS, RCLS, X_FWI, X_POP, boot_ratio, fires, fmt, pop_bins

SPLITS = {"t": ("cache_G", "zeitlich", ""), "rs": ("cache_G_stratified", "zufaellig", None),
          "ys": ("cache_G_yearstrat", "Jahre", "ys_"), "bs": ("cache_G_blockstrat", "Bloecke", "bs_")}


def score_nets(nets, Xs, doy, LC, norm, device, bs) -> np.ndarray:
    """-> (seeds, len(DTS), n) Ausschnitts-Score F(dt + 1)."""
    F = np.zeros((len(nets), len(DTS), Xs.shape[0]), np.float32)
    loc = np.arange(Xs.shape[0])
    with torch.no_grad():
        for k, dt_ in enumerate(DTS):
            e = LC - 1 - dt_
            tau = torch.tensor([float(dt_ + 1)], device=device)
            for a in range(0, loc.size, bs):
                j = slice(a, a + bs)
                x = norm(windows(Xs, loc[j], np.full(loc[j].size, e)))
                d = torch.from_numpy((doy[j] - dt_).astype(np.float32)).to(device)
                for s, net in enumerate(nets):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        raw_pa = net(x, d)[1]
                    F[s, k, j] = (-torch.expm1(net.pa_head.log_survival(raw_pa.float(), tau))[:, 0]).cpu().numpy()
    return F


def nfdb_class(meta: np.ndarray, by_day: dict, dp: int) -> np.ndarray:
    rc = []
    for D, r0, c0 in meta:
        d = dt.date(int(D // 10000), int(D // 100 % 100), int(D % 100)).toordinal()
        cand = [f for k in range(d - 30, d + 1) for f in by_day.get(k, ())
                if abs(f["row"] - r0) <= 6 + dp and abs(f["col"] - c0) <= 6 + dp]
        rc.append(max(cand, key=lambda f: f["size_ha"])["rcls"] if cand else "kein NFDB-Brand")
    return np.array(rc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nfdb", default="/data/Wildfire/artifacts/NFDB_point_txt.zip")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="Rauchtest: so viele Val- und Test-Zeilen")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"suppression_splits{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --tag waehlen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = exp / "checkpoints" / "paper"
    t0 = time.time()

    F_all = fires(exp, Path(args.nfdb))
    by_day = defaultdict(list)
    for f in F_all:
        if f["size_ha"] >= 1:
            by_day[f["date"].toordinal()].append(f)

    rows, saved, intro = [], {}, []
    rng = np.random.default_rng(0)
    for sp in args.splits:
        cache, sname, ref = SPLITS[sp]
        cd = exp / "derived_data" / "paper" / f"{cache}_L61"
        Xp = np.load(cd / "X.npy", mmap_mode="r")
        LC = Xp.shape[1]
        split, meta = np.load(cd / "split.npy"), np.load(cd / "meta.npy")
        idx = {p: np.where(split == p)[0] for p in ("val", "test")}
        if args.limit:
            idx = {p: v[np.linspace(0, v.size - 1, min(args.limit, v.size)).astype(int)] for p, v in idx.items()}
        y_all = targets.labels(cd, "next_patch")
        yc_all = targets.fire_map(cd, Xp, "next_patch")[:, CENTRE, CENTRE].astype(int)
        y = {p: y_all[v] for p, v in idx.items()}
        doy = {p: doy_of(meta[v, 0]) for p, v in idx.items()}
        X = {p: np.asarray(Xp[v]) for p, v in idx.items()}
        mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / cache / "norm_stats.npz")
        m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
        s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
        norm = lambda x: (torch.from_numpy(np.ascontiguousarray(x)).to(device).float() - m_t) / s_t  # noqa: E731
        tr = split == "train"
        rate_pa, rate_px = float(y_all[tr].mean()), float(yc_all[tr].mean())
        print(f"[ssp] {sp} {cache}: val {idx['val'].size:,}, test {idx['test'].size:,} "
              f"({time.time() - t0:.0f} s)", flush=True)

        # ---- Scores (val, test): name -> (seeds, len(DTS), n)
        S = {}
        for head, hname in (("lh", "STS-LH"), ("cg", "STS-Contagion")):
            paths = [ck / f"sts_{cache}_next_cache_splitcmp_{head}_s{s}_s{s}.pt" for s in args.seeds]
            if all(p.exists() for p in paths):
                nets = [sts_extreme.load_any(str(p), rate_px, rate_pa, device)[0] for p in paths]
                S[f"{hname} (Cache)"] = {p: score_nets(nets, X[p], doy[p], LC, norm, device, args.bs) for p in X}
            if ref is not None:
                paths = [ck / f"sts_extreme_{ref}{head}_ref_s{s}_s{s}.pt" for s in args.seeds]
                if all(p.exists() for p in paths):
                    nets = []
                    for p in paths:
                        c = torch.load(p, map_location="cpu")
                        nets.append(sts_extreme.load_any(str(p), c["init_px"], c["init_pa"], device)[0])
                    S[f"{hname} (Wuerfel)"] = {p: score_nets(nets, X[p], doy[p], LC, norm, device, args.bs) for p in X}
            torch.cuda.empty_cache()
            print(f"[ssp] {sp} {head}: {sorted(S)} ({time.time() - t0:.0f} s)", flush=True)
        rp = exp / "predictions" / "paper" / f"rec_{cache}_cache_next_paperhp_w0.npz"
        if rp.exists():
            z = np.load(rp)
            U = {}
            for p in X:
                pos = {int(i): n for n, i in enumerate(z[f"idx_{p}"])}
                if all(int(i) in pos for i in idx[p]):
                    sel = np.array([pos[int(i)] for i in idx[p]])
                    U[p] = np.stack([np.stack([z[f"paper55_T30_end{d}_s{s}_{p}"][sel] for d in DTS])
                                     for s in args.seeds if f"paper55_T30_end1_s{s}_{p}" in z.files]).astype(np.float32)
            if len(U) == 2:
                S["3D U-Net"] = U

        # ---- Gruppen auf dem Test
        te = idx["test"]
        Xt = X["test"]
        pop = Xt[:, LC - 1, X_POP].astype(np.float32).reshape(te.size, -1).mean(1)
        fr = Xt[:, LC - 1, X_FWI].astype(np.float32).reshape(te.size, -1)
        fr = np.where(fr == 255, np.nan, fr)
        fwi = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in fr])
        rc = nfdb_class(meta[te], by_day, args.dp)
        yt = y["test"]
        intro.append(f"- {sp} = {cache} ({sname}): Test {te.size:,}, Val {idx['val'].size:,}, Modelle "
                     f"{', '.join(S)}; NFDB-Verknuepfung {dict(Counter(rc.tolist()))}")
        for name, sc in S.items():
            for k, dt_ in enumerate(DTS):
                raw = sc["test"][:, k].astype(np.float64)
                cal = np.stack([platt(sc["val"][s, k], y["val"], sc["test"][s, k]) for s in range(raw.shape[0])])
                P = {"roh": raw.mean(0), "platt": cal.mean(0)}
                saved[f"{sp}__{name}__dt{dt_}__platt"] = P["platt"].astype(np.float32)
                e = LC - 1 - dt_
                fw = (Xt[:, e - 29:e + 1, FIRE_CH] > 0).reshape(te.size, -1).any(1)
                pb = pop_bins(pop, pop[fw])
                fh = np.where(np.isnan(fwi), "nan", np.where(fwi > np.nanmedian(fwi[fw]), "hoch", "niedrig"))
                groups = [("alle", np.ones(te.size, bool)), ("Feuer im Fenster", fw), ("kein Feuer im Fenster", ~fw)]
                groups += [(f"Feuer im Fenster, Bev. {b}", fw & (pb == b)) for b in PCLS[1:]]
                groups += [(f"kein Feuer im Fenster, Bev. {b}", ~fw & (pb == b)) for b in PCLS[1:]]
                groups += [(f"Feuer im Fenster, FWI {h}, Bev. {b}", fw & (fh == h) & (pb == b))
                           for h in ("niedrig", "hoch") for b in PCLS[1:]]
                groups += [(f"NFDB {c}", rc == c) for c in RCLS + ("kein NFDB-Brand",)]
                groups += [(f"NFDB {c}, Bev. {b}", (rc == c) & (pb == b)) for c in RCLS[:2] for b in PCLS[1:]]
                for g, m in groups:
                    r = {"split": sp, "model": name, "dt": dt_, "group": g, "n": int(m.sum()), "pos": int(yt[m].sum()),
                         "mean_y": float(yt[m].mean()) if m.any() else np.nan}
                    for tag, p in P.items():
                        pm = p[m]
                        r[f"mean_p_{tag}"] = float(pm.mean()) if m.any() else np.nan
                        r[f"ratio_{tag}"] = float(pm.mean() / yt[m].mean()) if m.any() and yt[m].sum() else np.nan
                        r[f"ratio_{tag}_lo"], r[f"ratio_{tag}_hi"] = boot_ratio(pm, yt[m].astype(np.float64), rng) \
                            if m.any() else (np.nan, np.nan)
                    ok = 5 <= yt[m].sum() <= m.sum() - 5
                    r["prauc"] = float(average_precision_score(yt[m], P["platt"][m])) if ok else np.nan
                    rows.append(r)
        print(f"[ssp] {sp} fertig ({time.time() - t0:.0f} s)", flush=True)
        del X, Xt

    # ---- Bericht
    L = ["# Brandbekaempfung und Bevoelkerung: Kalibrierung auf allen Test-Stichproben aller Splits", "",
         "Quotient = mean(p) / mean(y), p per Platt auf dem Val-Split des jeweiligen Splits kalibriert (je Seed und dt, "
         "dann Seed-Mittel), [95 %-Bootstrap]; > 1 = Ueberschaetzung. 'roh' = unkalibrierter Score. Bevoelkerungs-"
         "Terzile je Split auf den Test-Stichproben mit Feuer im Fenster. Ziel Feuer im Ausschnitt an D + 1.", ""] + intro + [""]
    show = ["alle", "Feuer im Fenster", "Feuer im Fenster, Bev. niedrig", "Feuer im Fenster, Bev. mittel",
            "Feuer im Fenster, Bev. hoch", "kein Feuer im Fenster, Bev. niedrig", "kein Feuer im Fenster, Bev. mittel",
            "kein Feuer im Fenster, Bev. hoch"] + \
           [f"Feuer im Fenster, FWI {h}, Bev. {b}" for h in ("niedrig", "hoch") for b in PCLS[1:]] + \
           [f"NFDB {c}" for c in RCLS[:2]] + [f"NFDB {c}, Bev. {b}" for c in RCLS[:2] for b in PCLS[1:]]
    for dt_ in (1, 15):
        L += [f"## dt {dt_}: Quotient nach Platt", ""]
        for sp in args.splits:
            models = sorted({r["model"] for r in rows if r["split"] == sp})
            if not models:
                continue
            L += [f"### {sp} = {SPLITS[sp][0]} ({SPLITS[sp][1]})", "",
                  "| Gruppe | n | Pos. | " + " | ".join(models) + " |", "|---|---|---|" + "---|" * len(models)]
            for g in show:
                cell = {r["model"]: r for r in rows if r["split"] == sp and r["dt"] == dt_ and r["group"] == g}
                if not cell:
                    continue
                any_r = next(iter(cell.values()))
                L.append(f"| {g} | {any_r['n']:,} | {any_r['pos']:,} | " + " | ".join(
                    f"{fmt(cell[mn]['ratio_platt'])} [{fmt(cell[mn]['ratio_platt_lo'])}, {fmt(cell[mn]['ratio_platt_hi'])}]"
                    if mn in cell else "–" for mn in models) + " |")
            L += [""]
    out.mkdir(parents=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    with (out / "groups.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "scores.npz", **saved)
    write_manifest(cfg, f"p55_suppression_splits{args.tag}", args.config,
                   {"args": vars(args), "splits": {sp: SPLITS[sp][0] for sp in args.splits}, "intro": intro,
                    "outputs": {k: str(out / k) for k in ("summary.md", "groups.csv", "scores.npz")}})
    print(f"[ssp] {out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

"""Status-Versuch ab 2011: STS mit / ohne Status, trainiert nur auf Wuerfel-Stichtagen ab 2011 (slurm 271), gegen die
Laeufe auf allen Jahren (Referenz *_ref, Status ctrl aus 255), GPU. Protokoll und Kennzahlen wie ctrl_eval.

Je Split und Kopf vier Modelle:
  alle Jahre, ohne Status   sts_extreme_[<sp>_]<h>_ref       (kein Kontext)
  alle Jahre, mit Status    sts_extreme_[<sp>_]ctrl_<h>      (ctx_ctrl*/cache_px)
  ab 2011, ohne Status      sts_extreme_[<sp>_]zero11_<h>    (ctx_zero11*/cache_px = Nullen)
  ab 2011, mit Status       sts_extreme_[<sp>_]ctrl11_<h>    (ctx_ctrl11*/cache_px)
dPR-AUC mit gepaartem Bootstrap: ab 2011 mit - ab 2011 ohne Status (der saubere Vergleich), alle Jahre mit - ohne,
ab 2011 mit Status - alle Jahre ohne Status (Referenz). Dazu der 3D U-Net.

Ausgabe NEU: metrics/ctrl11_eval<tag>/{summary.md, metrics.csv, groups.csv, scores.npz}, Manifest p55_ctrl11_eval<tag>.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import sts_extreme, targets
from wildfire.paper.compare import DTS
from wildfire.paper.ctrl_eval import GROUPS, HEADS, PAPER_UNET, SHOW, score, status_group
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.paper_metrics_ckpt import platt
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import doy_of
from wildfire.paper.suppression_eval import boot_ratio, fmt

SPLITS = {"t": ("cache_G", "", ""), "ys": ("cache_G_yearstrat", "ys_", "_ys"), "bs": ("cache_G_blockstrat", "bs_", "_bs")}
# Name -> (Checkpoint-Stamm mit {pre}/{h}, Kontextordner mit {sfx} oder None)
MODELS = {"alle Jahre ohne Status": ("{pre}{h}_ref", None),
          "alle Jahre mit Status": ("{pre}ctrl_{h}", "ctx_ctrl{sfx}"),
          "ab 2011 ohne Status": ("{pre}zero11_{h}", "ctx_zero11{sfx}"),
          "ab 2011 mit Status": ("{pre}ctrl11_{h}", "ctx_ctrl11{sfx}")}
PAIRS = (("ab 2011 mit Status", "ab 2011 ohne Status"), ("alle Jahre mit Status", "alle Jahre ohne Status"),
         ("ab 2011 mit Status", "alle Jahre ohne Status"), ("ab 2011 ohne Status", "alle Jahre ohne Status"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sfx", default="", help="Tag-Zusatz der ab-2011-Laeufe (Rauchtest: _smoke)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="Rauchtest: so viele Val- und Test-Zeilen")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"ctrl11_eval{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --tag waehlen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = exp / "checkpoints" / "paper"
    rng = np.random.default_rng(0)
    t0 = time.time()
    mrows, grows, saved, intro = [], [], {}, []
    for sp in args.splits:
        cache, pre, sfx = SPLITS[sp]
        cd = exp / "derived_data" / "paper" / f"{cache}_L61"
        Xp = np.load(cd / "X.npy", mmap_mode="r")
        LC = Xp.shape[1]
        split, meta = np.load(cd / "split.npy"), np.load(cd / "meta.npy")
        idx = {p: np.where(split == p)[0] for p in ("val", "test")}
        if args.limit:
            idx = {p: v[np.linspace(0, v.size - 1, min(args.limit, v.size)).astype(int)] for p, v in idx.items()}
        y_all = targets.labels(cd, "next_patch")
        y = {p: y_all[v] for p, v in idx.items()}
        doy = {p: doy_of(meta[v, 0]) for p, v in idx.items()}
        X = {p: np.asarray(Xp[v]) for p, v in idx.items()}
        mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / cache / "norm_stats.npz")
        m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
        s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
        norm = lambda x: (torch.from_numpy(np.ascontiguousarray(x)).to(device).float() - m_t) / s_t  # noqa: E731
        print(f"[c11] {sp} {cache}: val {idx['val'].size:,}, test {idx['test'].size:,} ({time.time() - t0:.0f} s)",
              flush=True)

        S, ctx_cache = {}, {}
        for h, hname in HEADS.items():
            for mname, (stem_f, ctx_f) in MODELS.items():
                stem = stem_f.format(pre=pre, h=h)
                if "11_" in stem:
                    stem = stem.replace("11_", f"11{args.sfx}_")
                paths = [ck / f"sts_extreme_{stem}_s{s}_s{s}.pt" for s in args.seeds]
                if not all(p.exists() for p in paths):
                    print(f"[c11] {sp} {h} {mname}: Checkpoints fehlen ({paths[0].name}), uebersprungen", flush=True)
                    continue
                cpx = None
                if ctx_f is not None:
                    cdir = ctx_f.format(sfx=sfx)
                    if cdir not in ctx_cache:
                        a = np.load(exp / "derived_data" / cdir / "cache_px.npy", mmap_mode="r")
                        ctx_cache[cdir] = {p: np.asarray(a[v]) for p, v in idx.items()}
                    cpx = ctx_cache[cdir]
                nets = []
                for p in paths:
                    c = torch.load(p, map_location="cpu")
                    nets.append(sts_extreme.load_any(str(p), c["init_px"], c["init_pa"], device)[0])
                S[f"{hname} {mname}"] = {p: score(nets, X[p], doy[p], LC, norm, None if cpx is None else cpx[p],
                                                  device, args.bs) for p in X}
                del nets
                torch.cuda.empty_cache()
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

        yt = y["test"]
        st = ctx_cache.get(f"ctx_ctrl{sfx}")
        if st is None:
            a = np.load(exp / "derived_data" / f"ctx_ctrl{sfx}" / "cache_px.npy", mmap_mode="r")
            st = {"test": np.asarray(a[idx["test"]])}
        avail = st["test"][:, 0, 10].reshape(len(yt), -1).max(-1) > 0
        years = meta[idx["test"], 0] // 10000
        intro.append(f"- {sp} = {cache}: Test {yt.size:,} (davon mit Lageberichten {int(avail.sum()):,}, Testjahre "
                     f"{sorted(set(years.tolist()))}); Modelle {', '.join(S)}")
        P = {}
        for name, sc in S.items():
            P[name] = np.stack([np.stack([platt(sc["val"][s, k], y["val"], sc["test"][s, k])
                                          for s in range(sc["test"].shape[0])]).mean(0) for k in range(len(DTS))])
            for k, dt_ in enumerate(DTS):
                saved[f"{sp}__{name}__dt{dt_}"] = P[name][k].astype(np.float32)
        for k, dt_ in enumerate(DTS):
            grp = status_group(st["test"][:, k])
            for name, pk in P.items():
                p = pk[k]
                r = {"split": sp, "model": name, "dt": dt_, **{m: paper_metrics(p, yt)[m] for m in SHOW}}
                r["pr_auc_avail"] = float(average_precision_score(yt[avail], p[avail])) if yt[avail].any() else np.nan
                mrows.append(r)
                for g, m in [(f"Status {s}", grp == s) for s in GROUPS] + \
                            [("mit Lageberichten", avail), ("ohne Lageberichte", ~avail)]:
                    gr = {"split": sp, "model": name, "dt": dt_, "group": g, "n": int(m.sum()),
                          "pos": int(yt[m].sum()), "mean_y": float(yt[m].mean()) if m.any() else np.nan,
                          "ratio": float(p[m].mean() / yt[m].mean()) if m.any() and yt[m].sum() else np.nan}
                    gr["ratio_lo"], gr["ratio_hi"] = boot_ratio(p[m], yt[m].astype(np.float64), rng) \
                        if m.any() else (np.nan, np.nan)
                    ok = 5 <= yt[m].sum() <= m.sum() - 5
                    gr["prauc"] = float(average_precision_score(yt[m], p[m])) if ok else np.nan
                    grows.append(gr)
            for hname in HEADS.values():
                for a_, b_ in PAIRS:
                    na, nb = f"{hname} {a_}", f"{hname} {b_}"
                    if na not in P or nb not in P:
                        continue
                    p, q = P[na][k], P[nb][k]
                    d = []
                    for _ in range(args.boot):
                        i = rng.integers(0, yt.size, yt.size)
                        if 0 < yt[i].sum() < yt.size:
                            d.append(average_precision_score(yt[i], p[i]) - average_precision_score(yt[i], q[i]))
                    mrows.append({"split": sp, "model": f"{na} - {b_}", "dt": dt_, "pair": True,
                                  "d_pr_auc": float(average_precision_score(yt, p) - average_precision_score(yt, q)),
                                  "d_pr_auc_lo": float(np.quantile(d, 0.025)),
                                  "d_pr_auc_hi": float(np.quantile(d, 0.975))})
        print(f"[c11] {sp} fertig ({time.time() - t0:.0f} s)", flush=True)
        del X, ctx_cache

    # ---- Bericht
    L = ["# Bekaempfungsstatus als Eingabe, Training nur ab 2011", "",
         "Paper-Protokoll (Ziel Feuer im Ausschnitt an D + 1, Stichtag D - dt), Platt auf dem Val-Split je Seed und dt, "
         "Seed-Mittel. 'ab 2011' = nur Wuerfel-Stichtage ab 2011 im Training (sonst gleiches Protokoll, gleiche "
         "Schrittzahl). Werte in %. dPR-AUC mit 95 %-Bootstrap (gepaart, nur Teststreuung).", ""] + intro + [""]
    for sp in args.splits:
        rs = [r for r in mrows if r["split"] == sp and not r.get("pair")]
        if not rs:
            continue
        L += [f"## {sp} = {SPLITS[sp][0]}", "",
              "| dt | Modell | Prec | Rec | F1 | PR-AUC | ROC-AUC | ECE | Brier | F1 max | P@R.91 | P@R.81 | "
              "PR-AUC (Jahre mit Berichten) |", "|---" * 13 + "|"]
        for dt_ in DTS:
            if sp == "t":
                f1p, prp = PAPER_UNET[dt_]
                L.append(f"| {dt_} | 3D U-Net (Paper) | – | – | {f1p:.2f} | {prp:.2f} | – | – | – | – | – | – | – |")
            for r in [r for r in rs if r["dt"] == dt_]:
                L.append(f"| {dt_} | {r['model']} | " + " | ".join(
                    fmt(100 * r[m]) for m in ("precision", "recall", "f1", "pr_auc", "roc_auc", "ece", "brier",
                                              "f1_max", "prec_at_rec_0.91", "prec_at_rec_0.81", "pr_auc_avail")) + " |")
        L += ["", f"### {sp}: dPR-AUC", "", "| Vergleich | " + " | ".join(f"dt {d}" for d in DTS) + " |",
              "|---|" + "---|" * len(DTS)]
        prs = [r for r in mrows if r["split"] == sp and r.get("pair")]
        for nm in dict.fromkeys(r["model"] for r in prs):
            cells = {r["dt"]: r for r in prs if r["model"] == nm}
            L.append(f"| {nm} | " + " | ".join(
                f"{100 * cells[d]['d_pr_auc']:+.2f} [{100 * cells[d]['d_pr_auc_lo']:+.2f}, "
                f"{100 * cells[d]['d_pr_auc_hi']:+.2f}]" if d in cells else "–" for d in DTS) + " |")
        L += [""]
        for dt_ in (1, 15):
            models = list(dict.fromkeys(r["model"] for r in grows if r["split"] == sp))
            L += [f"### {sp}, dt {dt_}: mean(p) / mean(y) nach Status am Vorhersagetag", "",
                  "| Gruppe | n | Pos. | mean(y) | " + " | ".join(models) + " |", "|---|---|---|---|" + "---|" * len(models)]
            for g in [f"Status {s}" for s in GROUPS] + ["mit Lageberichten", "ohne Lageberichte"]:
                cell = {r["model"]: r for r in grows if r["split"] == sp and r["dt"] == dt_ and r["group"] == g}
                a = next(iter(cell.values()), None)
                if a is None or a["n"] == 0:
                    continue
                L.append(f"| {g} | {a['n']:,} | {a['pos']:,} | {fmt(a['mean_y'])} | " + " | ".join(
                    f"{fmt(cell[mn]['ratio'])} [{fmt(cell[mn]['ratio_lo'])}, {fmt(cell[mn]['ratio_hi'])}]"
                    if mn in cell else "–" for mn in models) + " |")
            L += [""]
    out.mkdir(parents=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    for fn, data in (("metrics.csv", mrows), ("groups.csv", grows)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with (out / fn).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
    np.savez_compressed(out / "scores.npz", **saved)
    write_manifest(cfg, f"p55_ctrl11_eval{args.tag}", args.config,
                   {"args": vars(args), "intro": intro,
                    "outputs": {k: str(out / k) for k in ("summary.md", "metrics.csv", "groups.csv", "scores.npz")}})
    print(f"[c11] {out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

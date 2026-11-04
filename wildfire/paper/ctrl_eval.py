"""Bringt der Bekaempfungsstatus als Eingabe etwas? Vergleich sts_extreme mit --ctx px (ctx_ctrl*, slurm 255) gegen
die Referenzen ohne Zusatz (*_ref, gleiches Protokoll), GPU.

Je Split (t = cache_G, ys = cache_G_yearstrat, bs = cache_G_blockstrat) und Kopf (LH, Contagion): Val- und
Test-Stichproben des Caches im Paper-Protokoll (Ausschnittskopf, Score F(dt + 1) bei Stichtag e = D - dt, Ziel Feuer
im Ausschnitt an D + 1), Status-Kontext aus <ctx-dir>/cache_px.npy, je Seed und dt Platt auf Val, dann Seed-Mittel.
Dazu der 3D U-Net (gespeicherte Scores) als Einordnung.

Kennzahlen je dt: die Paper-Metriken (metrics.paper_metrics: Precision, Recall, F1 bei 0,5, PR-AUC, ROC-AUC, ECE,
MCE, Brier) plus bester F1 und Precision bei Recall 0,91 / 0,81; Differenz Status - Referenz in PR-AUC mit gepaartem
Bootstrap-Intervall. Gruppen: dominanter Status im Ausschnitt am Vorhersagetag (aus dem Kontext: OC > BH > UC >
EX30 > kein Bericht), Testjahre mit / ohne Lageberichte; Kennzahl mean(p) / mean(y) und PR-AUC.

Ausgabe NEU: metrics/ctrl_eval<tag>/{summary.md, metrics.csv, groups.csv, scores.npz}, Manifest p55_ctrl_eval<tag>.
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
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.paper_metrics_ckpt import platt
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import doy_of
from wildfire.paper.suppression_eval import boot_ratio, fmt

SPLITS = {"t": ("cache_G", "", "ctx_ctrl"), "ys": ("cache_G_yearstrat", "ys_", "ctx_ctrl_ys"),
          "bs": ("cache_G_blockstrat", "bs_", "ctx_ctrl_bs")}
HEADS = {"lh": "STS-LH", "cg": "STS-Contagion"}
PAPER_UNET = {1: (71.42, 67.82), 7: (65.09, 59.84), 15: (62.31, 55.07), 30: (54.07, 44.85)}   # F1, PR-AUC (MPCFire)
SHOW = ("precision", "recall", "f1", "pr_auc", "roc_auc", "ece", "mce", "brier", "f1_max", "prec_at_rec_0.91",
        "prec_at_rec_0.81")
GROUPS = ("OC", "BH", "UC", "EX30", "kein Bericht")


def score(nets, X, doy, LC, norm, cpx, device, bs) -> np.ndarray:
    """-> (seeds, len(DTS), n); cpx (n, len(DTS), 14, 13, 13) oder None (Referenz)."""
    F = np.zeros((len(nets), len(DTS), X.shape[0]), np.float32)
    loc = np.arange(X.shape[0])
    with torch.no_grad():
        for k, dt_ in enumerate(DTS):
            e = LC - 1 - dt_
            tau = torch.tensor([float(dt_ + 1)], device=device)
            for a in range(0, loc.size, bs):
                j = slice(a, a + bs)
                x = norm(windows(X, loc[j], np.full(loc[j].size, e)))
                d = torch.from_numpy((doy[j] - dt_).astype(np.float32)).to(device)
                px = None if cpx is None else torch.from_numpy(cpx[j, k].astype(np.float32)).to(device)
                for s, net in enumerate(nets):
                    if hasattr(net, "set_ctx"):
                        net.set_ctx(px, None)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        raw_pa = net(x, d)[1]
                    F[s, k, j] = (-torch.expm1(net.pa_head.log_survival(raw_pa.float(), tau))[:, 0]).cpu().numpy()
    return F


def status_group(cpx: np.ndarray) -> np.ndarray:
    """(n, 14, 13, 13) -> dominanter Status im Ausschnitt."""
    present = cpx[:, :4].reshape(len(cpx), 4, -1).max(-1) > 0
    return np.array([GROUPS[int(np.argmax(p))] if p.any() else GROUPS[-1] for p in present])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variant", default="ctrl", help="Tag-Teil der Status-Laeufe: sts_extreme_<sp>_<variant>_<kopf>")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="Rauchtest: so viele Val- und Test-Zeilen")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"ctrl_eval{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --tag waehlen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = exp / "checkpoints" / "paper"
    rng = np.random.default_rng(0)
    t0 = time.time()
    mrows, grows, saved, intro = [], [], {}, []
    for sp in args.splits:
        cache, pre, ctxd = SPLITS[sp]
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
        cpx_all = np.load(exp / "derived_data" / ctxd / "cache_px.npy", mmap_mode="r")
        cpx = {p: np.asarray(cpx_all[v]) for p, v in idx.items()}
        mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / cache / "norm_stats.npz")
        m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
        s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
        norm = lambda x: (torch.from_numpy(np.ascontiguousarray(x)).to(device).float() - m_t) / s_t  # noqa: E731
        print(f"[ce] {sp} {cache}: val {idx['val'].size:,}, test {idx['test'].size:,} ({time.time() - t0:.0f} s)",
              flush=True)

        S = {}
        for h, hname in HEADS.items():
            for var, vname, use_ctx in (("ref", "Referenz", False), (args.variant, "mit Status", True)):
                stem = f"{pre}{h}_ref" if var == "ref" else f"{pre}{var}_{h}"     # Referenzen: <sp>_lh_ref
                paths = [ck / f"sts_extreme_{stem}_s{s}_s{s}.pt" for s in args.seeds]
                if not all(p.exists() for p in paths):
                    print(f"[ce] {sp} {h} {var}: Checkpoints fehlen, uebersprungen", flush=True)
                    continue
                nets = []
                for p in paths:
                    c = torch.load(p, map_location="cpu")
                    nets.append(sts_extreme.load_any(str(p), c["init_px"], c["init_pa"], device)[0])
                S[f"{hname} {vname}"] = {p: score(nets, X[p], doy[p], LC, norm, cpx[p] if use_ctx else None, device,
                                                  args.bs) for p in X}
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
        years = meta[idx["test"], 0] // 10000
        avail = cpx["test"][:, 0, 10].reshape(len(yt), -1).max(-1) > 0
        intro.append(f"- {sp} = {cache}: Test {yt.size:,} (davon mit Lageberichten {int(avail.sum()):,}, Testjahre "
                     f"{sorted(set(years.tolist()))}); Modelle {', '.join(S)}")
        P = {}
        for name, sc in S.items():
            P[name] = np.stack([np.stack([platt(sc["val"][s, k], y["val"], sc["test"][s, k])
                                          for s in range(sc["test"].shape[0])]).mean(0) for k in range(len(DTS))])
            for k, dt_ in enumerate(DTS):
                saved[f"{sp}__{name}__dt{dt_}"] = P[name][k].astype(np.float32)
        for k, dt_ in enumerate(DTS):
            grp = status_group(cpx["test"][:, k])
            for name, pk in P.items():
                p = pk[k]
                r = {"split": sp, "model": name, "dt": dt_, **{m: paper_metrics(p, yt)[m] for m in SHOW}}
                r["pr_auc_avail"] = float(average_precision_score(yt[avail], p[avail])) if yt[avail].any() else np.nan
                ref = name.replace("mit Status", "Referenz")
                if name.endswith("mit Status") and ref in P:
                    q = P[ref][k]
                    d = []
                    for _ in range(args.boot):
                        i = rng.integers(0, yt.size, yt.size)
                        if 0 < yt[i].sum() < yt.size:
                            d.append(average_precision_score(yt[i], p[i]) - average_precision_score(yt[i], q[i]))
                    r["d_pr_auc"] = r["pr_auc"] - float(average_precision_score(yt, q))
                    r["d_pr_auc_lo"], r["d_pr_auc_hi"] = (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
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
        print(f"[ce] {sp} fertig ({time.time() - t0:.0f} s)", flush=True)
        del X

    # ---- Bericht
    L = ["# Bekaempfungsstatus als Eingabe: STS mit Status gegen Referenz", "",
         "Paper-Protokoll (Ziel Feuer im Ausschnitt an D + 1, Stichtag D - dt), Platt auf dem Val-Split je Seed und dt, "
         "Seed-Mittel. Status = Lageberichte der Feuerbehoerden am Vorhersagetag (ctx_ctrl*). Werte in %. "
         "dPR-AUC = Status - Referenz [95 %-Bootstrap, gepaart].", ""] + intro + [""]
    for sp in args.splits:
        rs = [r for r in mrows if r["split"] == sp]
        if not rs:
            continue
        L += [f"## {sp} = {SPLITS[sp][0]}", "",
              "| dt | Modell | Prec | Rec | F1 | PR-AUC | ROC-AUC | ECE | Brier | F1 max | P@R.91 | P@R.81 | "
              "PR-AUC (Jahre mit Berichten) | dPR-AUC |", "|---" * 14 + "|"]
        for dt_ in DTS:
            if sp == "t":
                f1p, prp = PAPER_UNET[dt_]
                L.append(f"| {dt_} | 3D U-Net (Paper) | – | – | {f1p:.2f} | {prp:.2f} | – | – | – | – | – | – | – | – |")
            for r in [r for r in rs if r["dt"] == dt_]:
                dd = (f"{100 * r['d_pr_auc']:+.2f} [{100 * r['d_pr_auc_lo']:+.2f}, {100 * r['d_pr_auc_hi']:+.2f}]"
                      if "d_pr_auc" in r else "–")
                L.append(f"| {dt_} | {r['model']} | " + " | ".join(
                    fmt(100 * r[m]) for m in ("precision", "recall", "f1", "pr_auc", "roc_auc", "ece", "brier",
                                              "f1_max", "prec_at_rec_0.91", "prec_at_rec_0.81", "pr_auc_avail"))
                    + f" | {dd} |")
        L += [""]
        for dt_ in (1, 15):
            models = sorted({r["model"] for r in grows if r["split"] == sp})
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
    write_manifest(cfg, f"p55_ctrl_eval{args.tag}", args.config,
                   {"args": vars(args), "intro": intro,
                    "outputs": {k: str(out / k) for k in ("summary.md", "metrics.csv", "groups.csv", "scores.npz")}})
    print(f"[ce] {out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

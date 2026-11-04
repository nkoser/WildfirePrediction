"""Welche Variablen tragen die Vorhersage? Permutation Importance im Protokoll des Papers.

Fuer einen trainierten Checkpoint (STS-DCTM oder ST-DCTM pro Pixel) wird je
Variable bzw. Variablengruppe deren Eingabe zwischen den Test-Stichproben
vertauscht: Stichprobe i bekommt fuer die betroffenen Kanaele das komplette
Fenster (30 Tage x 13 x 13) von Stichprobe perm(i), alle anderen Kanaele
bleiben. Die Verbindung zwischen Variable und Label ist damit zerstoert, ihre
Randverteilung bleibt erhalten - anders als beim Nullsetzen sieht das Modell
keine Werte, die es nie gesehen hat. Wichtigkeit = Metrik ohne Vertauschung
minus Metrik mit Vertauschung (PR-AUC, ROC-AUC; beide nur von der Reihenfolge
abhaengig, also ohne Kalibrierung vergleichbar).

Gemessen wird ohne neues Training: Die Frage ist, worauf DIESES Modell sich
stuetzt, nicht, welche Information ersetzbar ist. Korrelierte Variablen (t2m und
t2m_max, FFMC und FWI) koennen sich gegenseitig vertreten und erscheinen einzeln
zu klein; dafuer gibt es die Gruppen. Die Neutrainings ohne Feuer bzw. ohne
MODIS (channels.py) beantworten die zweite Frage.

Score wie paper_metrics_ckpt: F(dt + 1) bei Stichtag e = Zieltag - dt - 1;
Ausschnitt sts = Kopf, pxst = Maximum ueber die Pixel. Beim STS-DCTM ist der
Kalendertag eine eigene Eingabe ("season"); er wird wie eine Variable vertauscht.
Alle Variablen sehen dieselbe Permutation, Unterschiede zwischen ihnen enthalten
so kein Rauschen der Ziehung. Mit --repeats > 1 kommen weitere Permutationen dazu.

Mit --aggregate werden die Manifeste aller Checkpoints zusammengefasst
(Mittel und Streuung ueber Seeds und Wiederholungen), als CSV und Abbildung.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.pixel_stdctm import PixelSTDCTM, windows
from wildfire.paper.prepare import CHANNEL_NAMES
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import doy_of, load_sts

CENTRE = 6
SEASON = "season"
GROUPS = {
    "G_fire_history": [0],
    "G_era5_weather": list(range(1, 19)),
    "G_fire_danger": list(range(19, 26)),          # FFMC, DMC, DC, ISI, BUI, FWI, VPD
    "G_infrastructure": list(range(26, 31)),
    "G_terrain": list(range(31, 38)),
    "G_vegetation": list(range(38, 42)),
    "G_reflectance": list(range(42, 46)),
    "G_lst_emissivity": list(range(46, 50)),
    "G_brightness_temp": list(range(50, 54)),
    "G_landcover": [54],
    "G_all_modis": list(range(38, 54)),
    "G_all_but_fire": list(range(1, 55)),
}
SPECS = {"sts": "/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_d128_s*_s*.pt",
         "pxst": "/data/Wildfire/checkpoints/paper/pxst_cache_G_cubeP_s*_s*.pt"}
METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}


def variables(kind: str) -> dict[str, list[int]]:
    """Name -> Kanaele; Einzelkanaele, Gruppen und beim STS der Kalendertag (leere Liste)."""
    v = {n: [c] for c, n in enumerate(CHANNEL_NAMES)}
    v.update(GROUPS)
    if kind == "sts":
        v[SEASON] = []
    return v


def load_net(kind: str, path: str, device):
    if kind == "sts":
        return load_sts(path, device)
    c = torch.load(path, map_location="cpu")
    net = PixelSTDCTM(degree=c["degree"], tau_max=c["tau_max"])
    net.load_state_dict(c["state"])
    return net.to(device).eval()


def run(args, cfg) -> None:
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(args.pattern or SPECS[args.kind]))
    if args.index >= len(files):
        raise SystemExit(f"[imp] Index {args.index} ausserhalb von {len(files)} Checkpoints")
    ckpt = files[args.index]

    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")
    split = np.load(d / "split.npy")
    rows = np.where(split == "test")[0]
    doyD = doy_of(np.load(d / "meta.npy")[:, 0])
    LC = X.shape[1]
    L = LC + targets.shift(args.target)
    lab = {"patch": targets.labels(d, args.target)[rows],
           "centre": targets.centre_labels(d, X, args.target)[rows]}
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    net = load_net(args.kind, ckpt, device)
    var = variables(args.kind)
    print(f"[imp] {args.kind} {Path(ckpt).name}: {rows.size} Test-Stichproben, {len(var)} Variablen, "
          f"dt {list(args.dts)}, {args.repeats} Wiederholung(en)", flush=True)

    @torch.no_grad()
    def scores(dt_, chs=None, src=None, bs=64):
        """F(dt + 1) fuer Ausschnitt und Mitte; chs/src: Kanaele, die aus den Zeilen src kommen."""
        e = L - dt_ - 2
        k = torch.tensor([float(dt_ + 1)], device=device)
        sp, sc = [], []
        for a in range(0, rows.size, bs):
            r = rows[a:a + bs]
            x = windows(Xg, r, np.full(r.size, e))
            doy = doyD[r]
            if src is not None:
                rs = src[a:a + bs]
                if chs:
                    ci = torch.tensor(chs, device=device)
                    x[:, :, ci] = windows(Xg, rs, np.full(rs.size, e))[:, :, ci]
                else:                                   # Kalendertag
                    doy = doyD[rs]
            x = (x.float() - m_t) / s_t
            if args.kind == "sts":
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw_px, raw_pa = net(x, torch.from_numpy(doy - (LC - 1 - e)).to(device))
                fc = -torch.expm1(net.px_head.log_survival(raw_px[:, CENTRE, CENTRE], k))[:, 0]
                fp = -torch.expm1(net.pa_head.log_survival(raw_pa, k))[:, 0]
            else:
                raw = net(x)
                f = -torch.expm1(net.dctm.log_survival(raw.reshape(-1, raw.shape[-1]), k))[:, 0].view(len(r), -1)
                fc, fp = f[:, (13 * 13) // 2], f.max(1).values
            sp.append(fp.float().cpu().numpy()); sc.append(fc.float().cpu().numpy())
        return {"patch": np.concatenate(sp), "centre": np.concatenate(sc)}

    def evaluate(s):
        return {tg: {m: float(fn(lab[tg], s[tg])) for m, fn in METRICS.items()} for tg in lab}

    rng = np.random.default_rng(args.seed)
    perms = [rows[rng.permutation(rows.size)] for _ in range(args.repeats)]
    res = {}
    for dt_ in args.dts:
        t0 = time.time()
        base = evaluate(scores(dt_))
        drops = {}
        for name, chs in var.items():
            ms = [evaluate(scores(dt_, chs, src)) for src in perms]
            drops[name] = {tg: {m: [base[tg][m] - x[tg][m] for x in ms] for m in METRICS} for tg in lab}
        res[str(dt_)] = {"base": base, "drop": drops}
        top = sorted(drops, key=lambda n: -np.mean(drops[n]["patch"]["pr_auc"]))[:8]
        print(f"[imp] dt={dt_:2d} Basis PR-AUC {base['patch']['pr_auc']:.3f} in {time.time() - t0:.0f} s; groesster "
              "Abfall: " + ", ".join(f"{n} {np.mean(drops[n]['patch']['pr_auc']):+.3f}" for n in top), flush=True)

    write_manifest(cfg, f"p18_importance_{args.cache}{targets.tag(args.target)}_{Path(ckpt).stem}", args.config, {
        "cache": args.cache, "target": args.target, "kind": args.kind, "checkpoint": ckpt,
        "n_test": int(rows.size), "n_pos": int(lab["patch"].sum()), "repeats": args.repeats, "seed": args.seed,
        "method": "Kanalfenster zwischen Test-Stichproben vertauscht; drop = Basis - vertauscht",
        "score": "F(dt + 1) bei Stichtag e = Zieltag - dt - 1; Ausschnitt sts = Kopf, pxst = Maximum",
        "variables": {n: [CHANNEL_NAMES[c] for c in chs] for n, chs in var.items()}, "results": res,
    })


def aggregate(args, cfg) -> None:
    """Alle p18-Manifeste eines Caches und Ziels -> CSV und Abbildung."""
    exp = Path(cfg["experiment_root"])
    pre = f"p18_importance_{args.cache}{targets.tag(args.target)}_"
    docs = [json.loads(p.read_text()) for p in sorted((exp / "data_manifests").glob(pre + "*.json"))]
    if not docs:
        raise SystemExit(f"[imp] keine Manifeste {pre}*")
    rows_out = []
    for kind in sorted({doc["kind"] for doc in docs}):
        ds = [doc for doc in docs if doc["kind"] == kind]
        for dt_ in ds[0]["results"]:
            for tg in ("patch", "centre"):
                base = {m: [doc["results"][dt_]["base"][tg][m] for doc in ds] for m in METRICS}
                for name in ds[0]["results"][dt_]["drop"]:
                    r = {"model": kind, "dt": int(dt_), "label": tg, "variable": name,
                         "group": name.startswith("G_") or name == SEASON, "n_ckpt": len(ds)}
                    for m in METRICS:
                        v = np.concatenate([doc["results"][dt_]["drop"][name][tg][m] for doc in ds])
                        r[f"base_{m}"] = round(float(np.mean(base[m])), 4)
                        r[f"drop_{m}"] = round(float(v.mean()), 4)
                        r[f"drop_{m}_std"] = round(float(v.std()), 4)
                    rows_out.append(r)
    out = exp / "reports" / f"importance_{args.cache}{targets.tag(args.target)}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    print(f"[imp] {out} ({len(rows_out)} Zeilen)", flush=True)
    plot(rows_out, exp / "figures" / f"importance_{args.cache}{targets.tag(args.target)}.png")


def plot(rows_out: list[dict], path: Path) -> None:
    """Je Modell eine Zeile, links Gruppen, rechts die 20 wichtigsten Einzelkanaele; Balken je dt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [r for r in rows_out if r["label"] == "patch"]
    kinds = sorted({r["model"] for r in sel})
    dts = sorted({r["dt"] for r in sel})
    colors = ["#2a6fdb", "#3fa34d", "#e8a33d", "#c4413a"]
    fig, axes = plt.subplots(len(kinds), 2, figsize=(15, 6.5 * len(kinds)), squeeze=False)
    for i, kind in enumerate(kinds):
        for j, grp in enumerate((True, False)):
            rs = [r for r in sel if r["model"] == kind and r["group"] == grp]
            score = {}
            for r in rs:
                score[r["variable"]] = score.get(r["variable"], 0.0) + r["drop_pr_auc"]
            names = sorted(score, key=lambda n: -score[n])[: (None if grp else 20)][::-1]
            ax, h = axes[i, j], 0.8 / len(dts)
            for k, dt_ in enumerate(dts):
                val = {r["variable"]: r for r in rs if r["dt"] == dt_}
                y = np.arange(len(names)) + (k - (len(dts) - 1) / 2) * h
                ax.barh(y, [val[n]["drop_pr_auc"] for n in names], h, xerr=[val[n]["drop_pr_auc_std"] for n in names],
                        color=colors[k % len(colors)], label=f"dt = {dt_}", error_kw={"lw": 0.6})
            ax.set_yticks(np.arange(len(names)), names, fontsize=8)
            ax.axvline(0, color="#555", lw=0.6)
            ax.set_xlabel("Abfall der PR-AUC (Ausschnitt) bei Vertauschung")
            ax.set_title(f"{kind}: {'Gruppen' if grp else '20 wichtigste Einzelkanaele'}")
            ax.grid(axis="x", alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"[imp] {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap.add_argument("--kind", choices=sorted(SPECS), default="sts")
    ap.add_argument("--pattern", default="", help="Glob der Checkpoints; leer = Vorgabe je Modell")
    ap.add_argument("--index", type=int, default=0, help="welcher Checkpoint des Globs (Array-Index)")
    ap.add_argument("--dts", type=int, nargs="+", default=list(DTS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    (aggregate if args.aggregate else run)(args, cfg)


if __name__ == "__main__":
    main()

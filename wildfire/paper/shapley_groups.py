"""Exakte Shapley-Werte je Variablengruppe fuer Netze mit Gruppen-Dropout (sts_groupdrop.py), Paper-Protokoll D+1.

Werte-Funktion v(S) = PR-AUC (bzw. ROC-AUC) des Netzes, wenn nur die Gruppen in S sichtbar sind (die uebrigen
normiert auf 0 wie im Training). Das Netz hat jede Teilmenge im Training gesehen, ausgeblendete Gruppen sind fuer
es ein bekannter Zustand, keine fremden Werte. Mit G = 10 Gruppen werden alle 2^G = 1024 Teilmengen ausgewertet:

    phi_i = sum_{S ohne i} |S|! (G - |S| - 1)! / G! * (v(S + i) - v(S)),   sum_i phi_i = v(alle) - v(leer)

Score wie im Paper-Protokoll: Eingabe endet bei D - dt (Index LC-1-dt), F(dt + 1), Ziel y_next_patch;
Ausschnitt = Ausschnittskopf, Mitte = Pixel (6, 6). Teilmengen der Stichproben: all und ign30 (kein Feuer im
Ausschnitt in den 30 Fenstertagen, Positive = neue Zuendungen; dort ist die Feuergruppe per Konstruktion 0).

Modus --index i: ein Checkpoint (NAME:GLOB-Liste wie --models), speichert alle Scores
  metrics/shapley_groups<tag>/scores_<name>_s<seed>.npz  (Teilmengen x dt x Stichproben, float32)
Modus --aggregate: Shapley je Modell/Seed/dt/Kopf/Teilmenge, Mittel ueber Seeds, Gegenprobe gegen die Retrain-
Ablationen (sts_inputs-Checkpoints auf denselben Stichproben: alle, ohne Feuer, nur Feuer, nur Wetter, ohne MODIS
-> v(S) des Dropout-Netzes fuer genau diese Teilmenge), summary.md, shapley.csv, shapley.png, Manifest
p50_shapley_groups<tag>.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import sts_extreme, targets
from wildfire.paper.compare import DTS
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import CENTRE, doy_of
from wildfire.paper.sts_groupdrop import G_OF_CH, SHAP_GROUPS
from wildfire.paper.sts_inputs import INPUTS as _INPUTS
from wildfire.paper.sts_inputs_weather import EXTRA_INPUTS

GN = list(SHAP_GROUPS)
G = len(GN)
SUBSETS = np.array([[(m >> g) & 1 for g in range(G)] for m in range(2 ** G)], bool)    # Bitmaske m -> Gruppen
DEFAULT_MODELS = ["lh_gdrop=/data/Wildfire/checkpoints/paper/sts_extreme_lh_gdrop_s[0-9]_s[0-9].pt",
                  "cg_gdrop=/data/Wildfire/checkpoints/paper/sts_extreme_cg_gdrop_s[0-9]_s[0-9].pt"]
# Gegenprobe: Retrain-Ablation (sts_inputs) -> sichtbare Gruppen
CHECKS = {"ref": ("all", GN), "nofire": ("nofire", [g for g in GN if g != "fire"]), "fireonly": ("fireonly", ["fire"]),
          "weather": ("weather", ["era5_weather", "fire_danger"]),
          "nomodis": ("nomodis", ["era5_weather", "fire_danger", "infrastructure", "terrain", "landcover"])}
INPUTS = {**_INPUTS, **EXTRA_INPUTS}
METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}


def mask_of(groups) -> int:
    return sum(1 << GN.index(g) for g in groups)


def shapley(v: np.ndarray) -> np.ndarray:
    """v (2^G,) -> phi (G,)."""
    size = SUBSETS.sum(1)
    w = np.array([math.factorial(s) * math.factorial(G - s - 1) / math.factorial(G) for s in range(G)])
    phi = np.zeros(G)
    for i in range(G):
        without = ~SUBSETS[:, i]
        m = np.flatnonzero(without)
        phi[i] = np.sum(w[size[m]] * (v[m | (1 << i)] - v[m]))
    return phi


def setup(cfg, cache: str, n_eval: int, device):
    exp = Path(cfg["experiment_root"])
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / cache / "norm_stats.npz")
    cd = exp / "derived_data" / "paper" / f"{cache}_L61"
    Xp = np.load(cd / "X.npy", mmap_mode="r")
    split = np.load(cd / "split.npy")
    te = np.where(split == "test")[0]
    if n_eval:
        te = te[np.linspace(0, te.size - 1, min(n_eval, te.size)).astype(int)]  # gleichmaessig, Cache nach Label sortiert
    return {"exp": exp, "Xp": Xp, "LC": Xp.shape[1], "te": te,
            "m": torch.from_numpy(mean).float().to(device)[None, None, :, None, None],
            "s": torch.from_numpy(std).float().to(device)[None, None, :, None, None],
            "doyD": doy_of(np.load(cd / "meta.npy")[:, 0])[te],
            "y": {"patch_head": targets.labels(cd, "next_patch")[te],
                  "centre": targets.centre_labels(cd, Xp, "next_patch")[te]}}


@torch.no_grad()
def scores(net, d: dict, masks: np.ndarray, device, bs: int = 128) -> dict:
    """masks (M, C) bool -> {'patch_head','centre': (M, len(DTS), n) float32}."""
    M, n = masks.shape[0], d["te"].size
    out = {h: np.zeros((M, len(DTS), n), np.float32) for h in ("patch_head", "centre")}
    km = torch.from_numpy(masks.astype(np.float32)).to(device)[:, None, :, None, None]
    for j, dt_ in enumerate(DTS):
        e = d["LC"] - 1 - dt_
        tau = torch.tensor([float(dt_ + 1)], device=device)
        for a in range(0, n, bs):
            r = d["te"][a:a + bs]
            x = (torch.as_tensor(windows(d["Xp"], r, np.full(r.size, e))).to(device).float() - d["m"]) / d["s"]
            doy = torch.from_numpy(d["doyD"][a:a + bs] - (d["LC"] - 1 - e)).to(device)
            for i in range(M):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw_px, raw_pa = net(x * km[i], doy)
                out["centre"][i, j, a:a + r.size] = (-torch.expm1(net.px_head.log_survival(
                    raw_px[:, CENTRE, CENTRE], tau))[:, 0]).float().cpu().numpy()
                out["patch_head"][i, j, a:a + r.size] = (-torch.expm1(net.pa_head.log_survival(
                    raw_pa, tau))[:, 0]).float().cpu().numpy()
    return out


def ign30(d: dict) -> np.ndarray:
    """(len(DTS), n) bool: kein Feuer im Ausschnitt in den 30 Fenstertagen bis e."""
    out = []
    for dt_ in DTS:
        e = d["LC"] - 1 - dt_
        f = np.asarray(d["Xp"][d["te"], e - 29:e + 1, FIRE_CH]) > 0.5
        out.append(~f.reshape(f.shape[0], -1).any(1))
    return np.stack(out)


def expand(spec: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in spec:
        name, pat = s.split("=", 1)
        out += [(name, p) for p in sorted(glob.glob(pat))]
    return out


def seed_of(path: str) -> str:
    return Path(path).stem.rsplit("_", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--index", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--checks", nargs="*", default=["lh", "cg"], help="Praefixe der sts_inputs-Retrains (Gegenprobe)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    d = setup(cfg, args.cache, args.n_eval, device)
    out_dir = d["exp"] / "metrics" / f"shapley_groups{args.tag}"
    man_dir = d["exp"] / "data_manifests"
    stage = f"p50_shapley_groups{args.tag}"
    all_masks = SUBSETS[:, G_OF_CH]                                                     # (1024, 55)

    if not args.aggregate:
        name, path = expand(args.models)[args.index]
        f = out_dir / f"scores_{name}_{seed_of(path)}.npz"
        if f.exists():
            raise SystemExit(f"ABBRUCH: {f} existiert schon")
        out_dir.mkdir(parents=True, exist_ok=True)
        c = torch.load(path, map_location="cpu")
        net, _ = sts_extreme.load_any(path, c["init_px"], c["init_pa"], device)
        print(f"[shg] {name} {seed_of(path)}: {len(all_masks)} Teilmengen x {len(DTS)} dt x {d['te'].size} Stichproben",
              flush=True)
        sc = scores(net, d, all_masks, device)
        np.savez_compressed(f, rows=d["te"], **sc)
        full = mask_of(GN)
        print(f"[shg] gespeichert {f.name} ({time.time() - t0:.0f} s); PR-AUC Ausschnitt alle Gruppen: "
              + " ".join(f"{average_precision_score(d['y']['patch_head'], sc['patch_head'][full, j]):.3f}"
                         for j in range(len(DTS))), flush=True)
        return

    # ------------------------------------------------------------------ Aggregation
    if (man_dir / f"{stage}.json").exists():
        raise SystemExit(f"ABBRUCH: {stage} existiert schon")
    ign = ign30(d)
    subsets = {"all": np.ones_like(ign), "ign30": ign}
    rows, V = [], {}
    for fpath in sorted(out_dir.glob("scores_*.npz")):
        name, seed = fpath.stem[len("scores_"):].rsplit("_", 1)
        z = np.load(fpath)
        if not np.array_equal(z["rows"], d["te"]):
            raise SystemExit(f"ABBRUCH: {fpath.name} mit anderen Stichproben erzeugt")
        for h in ("patch_head", "centre"):
            for j, dt_ in enumerate(DTS):
                for sub, sel in subsets.items():
                    y = d["y"][h][sel[j]]
                    if y.min() == y.max():
                        continue
                    for met, fn in METRICS.items():
                        v = np.array([fn(y, z[h][m, j][sel[j]].astype(np.float64)) for m in range(len(SUBSETS))])
                        V[(name, seed, h, dt_, sub, met)] = v
                        phi = shapley(v)
                        rows += [{"model": name, "seed": seed, "head": h, "dt": dt_, "subset": sub, "metric": met,
                                  "group": g, "phi": float(phi[i]), "v_full": float(v[-1]), "v_empty": float(v[0]),
                                  "v_only": float(v[1 << i]), "v_without": float(v[(2 ** G - 1) ^ (1 << i)])}
                                 for i, g in enumerate(GN)]
        print(f"[shg] {fpath.name} ausgewertet ({time.time() - t0:.0f} s)", flush=True)
    with open(out_dir / "shapley.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # Gegenprobe: Retrain-Ablation auf denselben Stichproben
    checks = {}
    ckpt_dir = d["exp"] / "checkpoints" / "paper"
    for pre in args.checks:
        for cname, (inp, groups) in CHECKS.items():
            vals = []
            for s in args.seeds:
                cp = ckpt_dir / f"sts_extreme_{pre}_{cname}_s{s}_s{s}.pt"
                if not cp.exists():
                    continue
                c = torch.load(cp, map_location="cpu")
                net, _ = sts_extreme.load_any(str(cp), c["init_px"], c["init_pa"], device)
                sc = scores(net, d, INPUTS[inp][None], device)
                vals.append([average_precision_score(d["y"]["patch_head"], sc["patch_head"][0, j].astype(np.float64))
                             for j in range(len(DTS))])
                del net
            if vals:
                checks[(pre, cname)] = {"retrain": np.mean(vals, 0).tolist(), "mask": mask_of(groups), "n": len(vals)}
                print(f"[shg] Gegenprobe {pre}_{cname}: {len(vals)} Seeds ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------------------------ Bericht
    models = sorted({r["model"] for r in rows})

    def mean_phi(mdl, h, dt_, sub, met):
        ph = [r["phi"] for r in rows if (r["model"], r["head"], r["dt"], r["subset"], r["metric"]) == (mdl, h, dt_, sub, met)]
        return np.array(ph).reshape(-1, G) if ph else None

    def mean_v(mdl, h, dt_, sub, met, m):
        vs = [v[m] for k, v in V.items() if k[0] == mdl and k[2:] == (h, dt_, sub, met)]
        return (float(np.mean(vs)), float(np.std(vs))) if vs else (np.nan, np.nan)

    L = ["# Shapley-Werte je Variablengruppe (Gruppen-Dropout-Netze, Paper-Protokoll D+1)", "",
         f"Cache {args.cache}, Test n = {d['te'].size:,} (gleichmaessig), 2^{G} = {2 ** G} Teilmengen exakt, "
         "Mittel ueber Seeds. phi = Anteil an v(alle) - v(leer); v(leer) = nur Kalendertag.", ""]
    for sub in subsets:
        for h in ("patch_head", "centre"):
            for mdl in models:
                L += [f"## {mdl}, {'Ausschnitt' if h == 'patch_head' else 'Mitte'}, Stichproben {sub}, PR-AUC", "",
                      "| Gruppe | " + " | ".join(f"dt {t}" for t in DTS) + " |", "|---|" + "---|" * len(DTS)]
                P = {t: mean_phi(mdl, h, t, sub, "pr_auc") for t in DTS}
                if all(p is None for p in P.values()):
                    continue
                for i, g in enumerate(GN):
                    L.append(f"| {g} | " + " | ".join(
                        f"{P[t][:, i].mean():+.4f} ± {P[t][:, i].std():.4f}" if P[t] is not None else "–" for t in DTS) + " |")
                L.append("| v(alle) | " + " | ".join(f"{mean_v(mdl, h, t, sub, 'pr_auc', -1)[0]:.3f}" for t in DTS) + " |")
                L.append("| v(leer) | " + " | ".join(f"{mean_v(mdl, h, t, sub, 'pr_auc', 0)[0]:.3f}" for t in DTS) + " |")
                L.append("")
    L += ["## Gegenprobe: Dropout-Netz mit Maske gegen echtes Retrain ohne diese Gruppen (Ausschnitt, PR-AUC, all)", "",
          "| Retrain | sichtbare Gruppen | Retrain dt 1/7/15/30 | Dropout-Netz v(S) | max |Diff| |", "|---|---|---|---|---|"]
    for (pre, cname), cval in checks.items():
        mdl = f"{pre}_gdrop"
        vs = [mean_v(mdl, "patch_head", t, "all", "pr_auc", cval["mask"])[0] for t in DTS]
        diff = np.nanmax(np.abs(np.array(vs) - np.array(cval["retrain"])))
        L.append(f"| {pre}_{cname} | {', '.join(CHECKS[cname][1])} | " + "/".join(f"{x:.3f}" for x in cval["retrain"])
                 + " | " + "/".join(f"{x:.3f}" for x in vs) + f" | {diff:.3f} |")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.6 * len(models)), squeeze=False)
    colors = ["#c92a2a", "#1971c2", "#4dabf7", "#868e96", "#8d6e63", "#2f9e44", "#94d82d", "#f08c00", "#e8590c", "#7048e8"]
    for r_, mdl in enumerate(models):
        for c_, sub in enumerate(subsets):
            ax = axes[r_, c_]
            P = {t: mean_phi(mdl, "patch_head", t, sub, "pr_auc") for t in DTS}
            x = np.arange(len(DTS))
            wbar = 0.8 / G
            for i, g in enumerate(GN):
                ys = [P[t][:, i].mean() if P[t] is not None else np.nan for t in DTS]
                ax.bar(x + (i - G / 2 + 0.5) * wbar, ys, wbar, color=colors[i], label=g)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x, [f"dt {t}" for t in DTS])
            ax.set_title(f"{mdl}, {sub}: Shapley share of patch PR-AUC", fontsize=10, loc="left")
            ax.grid(axis="y", alpha=0.3)
    axes[0, 1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "shapley.png", dpi=150)
    write_manifest(cfg, stage, args.config, {
        "groups": SHAP_GROUPS, "n_eval": int(d["te"].size), "models": args.models,
        "checks": {f"{k[0]}_{k[1]}": v for k, v in checks.items()}, "out_dir": str(out_dir)})
    print("\n".join(L), flush=True)
    print(f"[shg] fertig ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

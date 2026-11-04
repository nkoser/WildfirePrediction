"""Zeitlicher Split (cache_G) gegen stratifizierten Split (cache_G_stratified) im Paper-Protokoll.

Liest nur Manifeste, rechnet nichts neu:
  3D U-Net   p2_reconstruct_<cache>_cache_paperhp (Ziel D) und _next_paperhp_w0 (Ziel D+1), 3 Seeds
             (cache_G: die vorhandenen Laeufe aus 42/211; cache_G_stratified: slurm 221)
  STS        p15_sts_dctm_<cache>{,_next}_cache<tag>_<head>_s<seed>, nur Cache-Stichproben, von null
             (slurm 222, beide Splits mit demselben Protokoll)
Je Modell Seed-Mittel von PR-AUC, ROC-AUC und F1 (Schwelle 0.5) je dt, Ausschnitt; STS auch die Mitte.
Ausgabe NEU: metrics/split_compare<tag>/summary.md, Manifest p47_split_compare<tag>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.audit.common import load_config, write_manifest
from wildfire.paper.compare import DTS

CACHES = ("cache_G", "cache_G_stratified")
TARGETS = {"cache": ("", "_paperhp", 1), "next_patch": ("_next", "_next_paperhp_w0", 0)}   # STS-, U-Net-Tag, end - dt
PAPER_D1 = "0.718/0.598/0.551/0.481"
METRICS = ("pr_auc", "roc_auc", "f1")


def unet(man: Path, cache: str, target: str):
    _, rn, off = TARGETS[target]
    p = man / f"p2_reconstruct_{cache}_cache{rn}.json"
    if not p.exists():
        return None
    runs = [r for r in json.load(open(p))["runs"] if r["channels"] == "paper55" and r["T"] == 30]
    out = {}
    for d in DTS:
        rs = [r["test"] for r in runs if r["end"] == d + off]
        out[str(d)] = {"patch": {k: float(np.mean([r[k] for r in rs])) for k in METRICS}, "n_seeds": len(rs)}
    return out


def sts(man: Path, cache: str, target: str, tag: str, head: str, seeds):
    fs = [man / f"p15_sts_dctm_{cache}{TARGETS[target][0]}_cache{tag}_{head}_s{s}.json" for s in seeds]
    fs = [f for f in fs if f.exists()]
    if not fs:
        return None
    res = [json.load(open(f))["results"] for f in fs]
    return {str(d): {"patch": {k: float(np.mean([r[str(d)]["patch_head"]["summary"][k]["mean"] for r in res])) for k in METRICS},
                     "centre": {k: float(np.mean([r[str(d)]["centre"]["summary"][k]["mean"] for r in res])) for k in METRICS},
                     "n_seeds": len(res)} for d in DTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sts-tag", default="_splitcmp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])
    man = exp / "data_manifests"
    out_dir = exp / "metrics" / f"split_compare{args.tag}"
    if out_dir.exists():
        raise SystemExit(f"ABBRUCH: {out_dir} existiert schon")

    models = {"3D U-Net": lambda c, t: unet(man, c, t),
              "STS-LH": lambda c, t: sts(man, c, t, args.sts_tag, "lh", args.seeds),
              "STS-Contagion": lambda c, t: sts(man, c, t, args.sts_tag, "cg", args.seeds)}
    res = {m: {c: {t: f(c, t) for t in TARGETS} for c in CACHES} for m, f in models.items()}
    splits = {}
    for c in CACHES:
        d = exp / "derived_data" / "paper" / f"{c}_L61"
        split = np.load(d / "split.npy")
        te = split == "test"
        splits[c] = {"n": {s: int((split == s).sum()) for s in ("train", "val", "test")},
                     "pos_test_D": float(np.load(d / "y.npy")[te].mean()),
                     "pos_test_D1": float(np.load(d / "y_next_patch.npy")[te].mean())}

    fmt = lambda v, h, k: "/".join(f"{v[str(d)][h][k]:.3f}" for d in DTS) if v else "–"  # noqa: E731
    lines = ["# Zeitlicher gegen stratifizierten Split (Paper-Protokoll)", "",
             "Seed-Mittel, Werte je dt " + "/".join(map(str, DTS)) + ". STS hier nur auf den Cache-Stichproben des "
             "jeweiligen Train-Splits trainiert (von null, Auswahl Val-PR-AUC), nicht auf Wuerfel-Stichtagen: die "
             "Wuerfel-Stichtage 2000-2019 ueberlappen den stratifizierten Test.", ""]
    for c, s in splits.items():
        lines.append(f"- {c}: Train {s['n']['train']:,} / Val {s['n']['val']:,} / Test {s['n']['test']:,}; "
                     f"Positivrate Test D {s['pos_test_D']:.3f}, D+1 {s['pos_test_D1']:.3f}")
    for k in METRICS:
        for h in ("patch", "centre"):
            if h == "centre" and k != "pr_auc":
                continue
            lines += ["", f"## {'Ausschnitt' if h == 'patch' else 'Mitte'}: {k}", "",
                      "| Modell | zeitlich D | stratifiziert D | zeitlich D+1 | stratifiziert D+1 |", "|---|---|---|---|---|"]
            for m, r in res.items():
                if h == "centre" and m == "3D U-Net":
                    continue
                lines.append(f"| {m} | {fmt(r['cache_G']['cache'], h, k)} | {fmt(r['cache_G_stratified']['cache'], h, k)} | "
                             f"{fmt(r['cache_G']['next_patch'], h, k)} | {fmt(r['cache_G_stratified']['next_patch'], h, k)} |")
            if h == "patch" and k == "pr_auc":
                lines.append(f"| 3D U-Net (Paper-Werte) | – | – | {PAPER_D1} | – |")
    out_dir.mkdir(parents=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    write_manifest(cfg, f"p47_split_compare{args.tag}", args.config,
                   {"sts_tag": args.sts_tag, "seeds": args.seeds, "splits": splits, "results": res})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

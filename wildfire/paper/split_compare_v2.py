"""Zeitlich (cache_G) gegen zufaellig stratifiziert (cache_G_stratified) gegen jahres-stratifiziert (cache_G_yearstrat).

Wie split_compare.py (dieselben Leser unet/sts), nur mit dem dritten Split aus split_audit (year_strat: ganze Jahre
je Split, Black Swans 70/10/20 verteilt, kein Raum-Zeit-Leakage). Liest nur Manifeste:
  3D U-Net   p2_reconstruct_<cache>_cache_paperhp / _next_paperhp_w0 (cache_G: 42/211, stratified: 221, yearstrat: 226)
  STS        p15_sts_dctm_<cache>{,_next}_cache_splitcmp_<head>_s<seed> (222 fuer cache_G und stratified, 227 yearstrat)
Ausgabe NEU: metrics/split_compare_v2<tag>/summary.md, Manifest p48_split_compare_v2<tag>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildfire.audit.common import load_config, write_manifest
from wildfire.paper.compare import DTS
from wildfire.paper.split_compare import METRICS, PAPER_D1, TARGETS, sts, unet

CACHES = {"cache_G": "zeitlich", "cache_G_stratified": "zufaellig", "cache_G_yearstrat": "Jahre strat."}
LEAK = {"cache_G": "0 %", "cache_G_stratified": "78 %", "cache_G_yearstrat": "0 %"}   # metrics/split_audit (Test overlap)


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
    out_dir = exp / "metrics" / f"split_compare_v2{args.tag}"
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
    lines = ["# Drei Splits im Paper-Protokoll: zeitlich, zufaellig stratifiziert, jahres-stratifiziert", "",
             "Seed-Mittel, Werte je dt " + "/".join(map(str, DTS)) + ". STS nur auf den Cache-Stichproben des "
             "jeweiligen Train-Splits (von null, Auswahl Val-PR-AUC). Leakage = Anteil Test-Stichproben, deren "
             "Eingabefenster Pixel-Tage mit Train teilen (metrics/split_audit).", ""]
    for c, s in splits.items():
        lines.append(f"- {c} ({CACHES[c]}, Leakage {LEAK[c]}): Train {s['n']['train']:,} / Val {s['n']['val']:,} / "
                     f"Test {s['n']['test']:,}; Positivrate Test D {s['pos_test_D']:.3f}, D+1 {s['pos_test_D1']:.3f}")
    head = " | ".join(f"{CACHES[c]} {t}" for t in ("D", "D+1") for c in CACHES)
    for k in METRICS:
        for h in ("patch", "centre"):
            if h == "centre" and k != "pr_auc":
                continue
            lines += ["", f"## {'Ausschnitt' if h == 'patch' else 'Mitte'}: {k}", "",
                      f"| Modell | {head} |", "|---" * (1 + 2 * len(CACHES)) + "|"]
            for m, r in res.items():
                if h == "centre" and m == "3D U-Net":
                    continue
                cells = [fmt(r[c][t], h, k) for t in TARGETS for c in CACHES]
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            if h == "patch" and k == "pr_auc":
                lines.append("| 3D U-Net (Paper-Werte) | – | – | – | " + PAPER_D1 + " | – | – |")
    out_dir.mkdir(parents=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    write_manifest(cfg, f"p48_split_compare_v2{args.tag}", args.config,
                   {"sts_tag": args.sts_tag, "seeds": args.seeds, "splits": splits, "results": res})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

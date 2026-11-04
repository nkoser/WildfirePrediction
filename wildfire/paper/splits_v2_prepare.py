"""Kandidaten-Split aus split_audit (derived_data/splits_v2/<name>/split.npy) als Cache-Variante fuer reconstruct/sts_dctm.

Anders als stratified_split.py liest dieser Schritt split.npy direkt (Zeilen wie cache_G_L61), weil block_strat
gepurgte Stichproben enthaelt (split = "drop"), die in der JSON fehlen. reconstruct und sts_dctm waehlen nur
train/val/test aus, "drop"-Zeilen werden also nie benutzt.

Ausgabe NEU unter $EXP/derived_data/paper/<cache>_L61/: split.npy (Kopie), X/y/meta/... als Symlinks auf cache_G_L61.
clim.npz wird nicht verlinkt (auf den Trainingsjahren von cache_G gerechnet).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.stratified_split import LINKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True, help="Unterordner in derived_data/splits_v2<split-tag>")
    ap.add_argument("--split-tag", default="")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--src", default="cache_G")
    ap.add_argument("--tag", default="_L61")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    base = exp / "derived_data" / "paper"
    src, out = base / f"{args.src}{args.tag}", base / f"{args.cache}{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon")
    split = np.load(exp / "derived_data" / f"splits_v2{args.split_tag}" / args.name / "split.npy")
    meta, y = np.load(src / "meta.npy"), np.load(src / "y.npy")
    if len(split) != len(meta):
        raise SystemExit(f"ABBRUCH: split.npy hat {len(split)} Zeilen, {src.name} {len(meta)}")
    bad = set(split.tolist()) - {"train", "val", "test", "drop"}
    if bad:
        raise SystemExit(f"ABBRUCH: unbekannte Split-Werte {bad}")

    out.mkdir(parents=True)
    for f in LINKS:
        if (src / f).exists():
            (out / f).symlink_to(Path("..") / src.name / f)
    np.save(out / "split.npy", split)

    tab = {s: {"positive": int(((split == s) & (y == 1)).sum()), "negative": int(((split == s) & (y == 0)).sum())}
           for s in ("train", "val", "test", "drop")}
    years = {s: dict(sorted(Counter(int(m // 10000) for m in meta[split == s, 0]).items())) for s in tab}
    print(f"[v2] {args.name} -> {args.cache}: {tab}")
    for s, v in years.items():
        print(f"[v2] Jahre {s}: {v}")
    write_manifest(cfg, f"p0_paper_{args.cache}{args.tag}", args.config, {
        "cache": args.cache, "source": str(src), "split_from": args.name, "split_table": tab, "years": years,
        "linked": [f for f in LINKS if (out / f).is_symlink()], "outputs": {"split": str(out / "split.npy")}})


if __name__ == "__main__":
    main()

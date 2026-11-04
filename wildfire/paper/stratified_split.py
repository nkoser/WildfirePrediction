"""Stratifizierter Split cache_G_stratified als eigene Cache-Variante fuer reconstruct und sts_dctm.

/firedata/MPCFire/cache_G_stratified enthaelt dieselben Stichproben wie cache_G (HDF5 und norm_stats
sind Symlinks dorthin); nur samples_variant_G.json teilt sie anders auf: zufaellig ueber alle 26 Jahre
statt Train 2000-2019 / Val 2020-2022 / Test 2023-2025.

Ausgabe NEU unter $EXP/derived_data/paper/cache_G_stratified_L61/:
  split.npy    (N,) <U5  train / val / test aus der stratifizierten JSON
  X.npy, y.npy, meta.npy, y_next_patch.npy, y_next_centre.npy, fire_next.npy
               relative Symlinks auf cache_G_L61 (gleiche Zeilen, gleiche Labels)
clim.npz wird bewusst NICHT verlinkt: die Klimatologie ist auf den Trainingsjahren von cache_G gerechnet.

Jede Zeile der JSON wird gegen meta.npy und y.npy von cache_G_L61 geprueft (Datum, Zeile, Spalte,
Label); bei der kleinsten Abweichung oder einem schon vorhandenen Zielordner bricht der Schritt ab.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest

LINKS = ("X.npy", "y.npy", "meta.npy", "y_next_patch.npy", "y_next_centre.npy", "fire_next.npy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", default="cache_G")
    ap.add_argument("--cache", default="cache_G_stratified")
    ap.add_argument("--tag", default="_L61")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    base = exp / "derived_data" / "paper"
    src, out = base / f"{args.src}{args.tag}", base / f"{args.cache}{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon")

    doc = json.loads(next((Path(cfg["dataset_root"]) / args.cache).glob("samples_variant_*.json")).read_text())
    records = doc["positive_samples"] + doc["negative_samples"]
    meta, y = np.load(src / "meta.npy"), np.load(src / "y.npy")
    assert len(records) == len(meta) == len(y), (len(records), len(meta), len(y))
    bad = Counter()
    for i, r in enumerate(records):
        if (int(r["date"].replace("-", "")), int(r["pixel_row"]), int(r["pixel_col"])) != tuple(meta[i]):
            bad["meta"] += 1
        if y[i] != (1 if r["sample_type"] == "positive" else 0):
            bad["label"] += 1
    if bad:
        raise SystemExit(f"ABBRUCH: JSON und {src.name} passen nicht zusammen: {dict(bad)}")
    print(f"[strat] {len(records)} Zeilen gegen {src.name} geprueft: Datum, Zeile, Spalte, Label stimmen")

    split = np.array([r["split"] for r in records])
    old = np.load(src / "split.npy")
    out.mkdir(parents=True)
    for f in LINKS:
        if (src / f).exists():
            (out / f).symlink_to(Path("..") / src.name / f)
    np.save(out / "split.npy", split)

    tab = {s: {"positive": int(((split == s) & (y == 1)).sum()), "negative": int(((split == s) & (y == 0)).sum())}
           for s in ("train", "val", "test")}
    years = {s: dict(sorted(Counter(int(m // 10000) for m in meta[split == s, 0]).items())) for s in tab}
    moved = {f"{a}->{b}": int(((old == a) & (split == b)).sum()) for a in tab for b in tab}
    print(f"[strat] Splits: {tab}")
    print(f"[strat] alt -> neu: {moved}")
    for s, v in years.items():
        print(f"[strat] Jahre {s}: {v}")

    write_manifest(cfg, f"p0_paper_{args.cache}{args.tag}", args.config, {
        "cache": args.cache, "source": str(src), "linked": [f for f in LINKS if (out / f).is_symlink()],
        "split_table": tab, "years": years, "moved_from_temporal": moved,
        "outputs": {"split": str(out / "split.npy")},
    })


if __name__ == "__main__":
    main()

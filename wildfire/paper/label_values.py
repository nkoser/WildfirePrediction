"""Welche Werte stehen in den MCD14A1-Rastern, aus denen die D+1-Labels kommen?

label_variants.raw_fire (und damit next_labels, y_next_patch) zaehlt jeden Wert > 0 als Feuer.
Laut Paper (Anhang C.1, Tabelle 6) ist nur firemask 9 (hohe Konfidenz) positiv; 7/8 (niedrig,
nominal) und 4 (Wolke) sind weder positiv noch negativ. Sind die Raster schon binaer, aendert das
nichts; stehen dort die Klassen 0-9, waeren unsere D+1-Labels zu grosszuegig.

Fuer alle Tage D+1 der Stichproben des Caches:
  - Histogramm aller Rasterwerte (ganzes Kanada, je Jahr),
  - Werte innerhalb der 13x13-Ausschnitte,
  - y_next_patch unter mehreren Regeln (> 0, >= 7, >= 8, == 9, falls vorhanden) mit Basisrate je
    Split und Abgleich gegen das gespeicherte y_next_patch.npy.
Schreibt metrics/label_values/summary.md und ein Manifest; veraendert keine vorhandenen Dateien.
"""

from __future__ import annotations

import argparse
import datetime as dt
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.label_variants import HALF, patch

RULES = {"gt0": lambda a: a > 0, "ge7": lambda a: a >= 7, "ge8": lambda a: a >= 8, "eq9": lambda a: a == 9}


def raw_values(root: Path, dates: set[dt.date]):
    """Rohwerte (H, W) je Tag aus Entire_Canada_Maps/MCD14A1 plus Werthistogramm je Jahr."""
    out, hist, meta = {}, defaultdict(Counter), {}
    by_year = defaultdict(set)
    for d in dates:
        by_year[d.year].add(f"{d.year}_{d.month:02d}_{d.day:02d}.tif")
    for year, names in sorted(by_year.items()):
        with tarfile.open(root / "MCD14A1" / f"MCD14A1__{year}.tar", "r:") as tf:
            for m in tf:
                base = Path(m.name).name
                if m.isfile() and base in names:
                    with MemoryFile(tf.extractfile(m).read()) as mem, mem.open() as src:
                        a = src.read(1)
                        meta.setdefault("dtype", str(a.dtype))
                        meta.setdefault("nodata", src.nodata)
                        meta.setdefault("shape", list(a.shape))
                        v, c = np.unique(a, return_counts=True)
                        hist[year].update(dict(zip(v.tolist(), c.tolist())))
                        out[dt.date(*map(int, base[:10].split("_")))] = a
        print(f"[lv] {year}: {sum(1 for d in out if d.year == year)} Tage, Werte {sorted(hist[year])[:20]}",
              flush=True)
    return out, hist, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--out-name", default="label_values")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp, raw = Path(cfg["experiment_root"]), Path(cfg["dataset_root"]) / "Entire_Canada_Maps"
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    meta = np.load(d / "meta.npy")
    split = np.load(d / "split.npy")
    y_saved = np.load(d / "y_next_patch.npy").astype(int)

    to_date = lambda m: dt.date(int(m // 10000), int(m // 100 % 100), int(m % 100))  # noqa: E731
    need = {to_date(m) + dt.timedelta(days=1) for m in meta[:, 0]}
    fire, hist, rmeta = raw_values(raw, need)
    missing = sorted(x for x in need if x not in fire)
    if missing:
        raise SystemExit(f"ABBRUCH: {len(missing)} Tage fehlen, z. B. {missing[:3]}")

    ys = {k: np.zeros(len(meta), np.int8) for k in RULES}
    in_patch = Counter()
    for i, (m, r, c) in enumerate(meta):
        p = patch(fire[to_date(m) + dt.timedelta(days=1)], r, c)
        v, n = np.unique(p, return_counts=True)
        in_patch.update(dict(zip(v.tolist(), n.tolist())))
        for k, f in RULES.items():
            ys[k][i] = f(p).any()

    total = Counter()
    for h in hist.values():
        total.update(h)
    res = {"raster": rmeta, "values_all_days": {str(k): int(v) for k, v in sorted(total.items())},
           "values_in_patches": {str(k): int(v) for k, v in sorted(in_patch.items())},
           "values_per_year": {str(y): {str(k): int(v) for k, v in sorted(h.items())} for y, h in hist.items()},
           "rules": {}}
    lines = []
    for k, yk in ys.items():
        row = {"agree_with_saved": float((yk == y_saved).mean())}
        for s in ("train", "val", "test"):
            sel = split == s
            row[f"rate_{s}"] = float(yk[sel].mean())
            row[f"n_pos_{s}"] = int(yk[sel].sum())
        res["rules"][k] = row
        lines.append(f"| {k} | {row['agree_with_saved']:.4f} | " + " | ".join(
            f"{row[f'n_pos_{s}']} ({row[f'rate_{s}']:.3f})" for s in ("train", "val", "test")) + " |")
        print(lines[-1], flush=True)

    out = exp / "metrics" / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(
        f"# MCD14A1-Rasterwerte hinter den D+1-Labels ({args.cache})\n\n"
        f"Raster: {rmeta}\n\n"
        f"Werte ueber alle {len(need)} Tage D+1 (ganzes Raster): {res['values_all_days']}\n\n"
        f"Werte in den 13x13-Ausschnitten: {res['values_in_patches']}\n\n"
        "Paper (Anhang C.1): nur 9 = positiv; 7/8 und 4 (Wolke) keine Negative.\n\n"
        "| Regel | Uebereinstimmung mit y_next_patch.npy | Train pos (Rate) | Val pos (Rate) | Test pos (Rate) |\n"
        "|---|---|---|---|---|\n" + "\n".join(lines) + "\n")
    write_manifest(cfg, f"p12_label_values_{args.cache}", args.config, res)


if __name__ == "__main__":
    main()

"""Stufe 6: Rekonstruktion der Labeldefinition der Fenster-Caches.

Stufe 5 hat gezeigt: das Eingabefenster endet exakt am Sample-Datum D und
enthaelt die Feuermaske als Kanal 0. Das HDF5-Attribut ``forecast=1``
legt nahe, dass das Label sich auf D+1 bezieht - am Zentrumspixel trifft
das aber nicht zu. Hier wird die Regel daher empirisch bestimmt:

Fuer viele Samples wird die rohe Feuermaske im 13x13-Patch um (r, c) fuer
D-k .. D+k gelesen und geprueft, welche Kandidatenregel die Labels
exakt reproduziert:

  * Zentrumspixel brennt an D+h
  * irgendein Pixel im Patch brennt an D+h
  * Anteil brennender Patchpixel >= ``min_fire`` an D+h
  * dieselben Regeln kumulativ ueber ein Horizontfenster

Zusaetzlich wird quantifiziert, wie stark der Feuerkanal im EINGABEfenster
mit dem Label korreliert - das ist das eigentliche Shortcut-Risiko.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from rasterio.io import MemoryFile
from rasterio.windows import Window

from wildfire.audit.common import assert_readonly, load_config, write_manifest

PATCH = 13
HALF = PATCH // 2


def read_fire_patches(
    dataset_root: Path, dates: list[dt.date], row: int, col: int, grid=(928, 1968)
) -> dict[str, np.ndarray]:
    """13x13-Feuerpatches fuer eine Liste von Tagen (randsicher)."""
    out: dict[str, np.ndarray] = {}
    by_year: dict[int, list[dt.date]] = defaultdict(list)
    for d in dates:
        by_year[d.year].append(d)
    r0, c0 = row - HALF, col - HALF
    for year, ds_list in by_year.items():
        tar = dataset_root / "Entire_Canada_Maps" / "MCD14A1" / f"MCD14A1__{year}.tar"
        if not tar.exists():
            continue
        wanted = {f"{d.year}_{d.month:02d}_{d.day:02d}.tif": d for d in ds_list}
        with tarfile.open(tar, "r:") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                base = Path(m.name).name
                if base not in wanted:
                    continue
                blob = tf.extractfile(m).read()
                with MemoryFile(blob) as mem, mem.open() as src:
                    full = np.zeros((PATCH, PATCH), dtype=np.uint8)
                    rr0, cc0 = max(0, r0), max(0, c0)
                    rr1, cc1 = min(grid[0], r0 + PATCH), min(grid[1], c0 + PATCH)
                    if rr1 > rr0 and cc1 > cc0:
                        sub = src.read(
                            1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0)
                        )
                        full[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] = sub
                out[wanted[base].isoformat()] = full
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    root = Path(cfg["dataset_root"])
    cdir = root / args.cache
    doc = json.loads(next(cdir.glob("samples_variant_*.json")).read_text())
    records = doc["positive_samples"] + doc["negative_samples"]
    h5_path = next(cdir.glob("*.h5"))

    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][...]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        n_pos = int((labels == 1).sum())
        k = max(1, args.n_samples // 2)
        idx = np.concatenate(
            [
                rng.choice(np.where(labels == 1)[0], size=k, replace=False),
                rng.choice(np.where(labels == 0)[0], size=k, replace=False),
            ]
        )
        idx = np.sort(idx)
        print(f"[s6] {len(idx)} Samples aus {args.cache} (min_fire={attrs.get('min_fire')}, "
              f"forecast={attrs.get('forecast')}, lookback={attrs.get('lookback')})")

        H = args.horizon
        offsets = list(range(-3, H + 1))
        rows = []
        fire_in_window: list[dict] = []
        for n, si in enumerate(idx.tolist()):
            rec = records[si]
            D = dt.date.fromisoformat(rec["date"])
            r, c = int(rec["pixel_row"]), int(rec["pixel_col"])
            patches = read_fire_patches(root, [D + dt.timedelta(days=o) for o in offsets], r, c)
            entry = {
                "sample_index": si,
                "label": int(labels[si]),
                "date": rec["date"],
                "row": r,
                "col": c,
                "split": rec.get("split"),
                "center": {},
                "patch_count": {},
                "patch_fraction": {},
            }
            for o in offsets:
                key = (D + dt.timedelta(days=o)).isoformat()
                p = patches.get(key)
                if p is None:
                    continue
                entry["center"][f"D{o:+d}"] = int(p[HALF, HALF])
                entry["patch_count"][f"D{o:+d}"] = int(p.sum())
                entry["patch_fraction"][f"D{o:+d}"] = float(p.mean())
            # Feuer im EINGABEfenster (Kanal 0, D-60..D) am Zentrum und im Patch
            w = np.asarray(f["windows"][si, 0])  # (13,13,61)
            fire_in_window.append(
                {
                    "sample_index": si,
                    "label": int(labels[si]),
                    "center_any_fire_in_input": bool(w[HALF, HALF].max() > 0),
                    "patch_any_fire_in_input": bool(w.max() > 0),
                    "center_fire_at_last_step": int(w[HALF, HALF, -1]),
                    "patch_fire_count_last_step": int((w[:, :, -1] > 0).sum()),
                    "n_input_days_with_patch_fire": int((w.reshape(-1, w.shape[-1]).max(0) > 0).sum()),
                }
            )
            rows.append(entry)
            if n % 25 == 0:
                print(f"    {n}/{len(idx)} ...", flush=True)

    # -------------------------------------------------- Regelkandidaten pruefen
    def rule_center(e, h):
        return e["center"].get(f"D{h:+d}", 0) > 0

    def rule_patch_any(e, h):
        return e["patch_count"].get(f"D{h:+d}", 0) > 0

    def rule_patch_frac(e, h, thr):
        return e["patch_fraction"].get(f"D{h:+d}", 0.0) >= thr

    def rule_center_cum(e, h):
        return any(e["center"].get(f"D{o:+d}", 0) > 0 for o in range(1, h + 1))

    def rule_patch_cum(e, h):
        return any(e["patch_count"].get(f"D{o:+d}", 0) > 0 for o in range(1, h + 1))

    thr = float(attrs.get("min_fire", 0.001))
    candidates = {}
    for h in range(-3, H + 1):
        candidates[f"center@D{h:+d}"] = lambda e, h=h: rule_center(e, h)
        candidates[f"patch_any@D{h:+d}"] = lambda e, h=h: rule_patch_any(e, h)
        candidates[f"patch_frac>={thr}@D{h:+d}"] = lambda e, h=h: rule_patch_frac(e, h, thr)
    for h in range(1, H + 1):
        candidates[f"center_cum@D+1..D{h:+d}"] = lambda e, h=h: rule_center_cum(e, h)
        candidates[f"patch_cum@D+1..D{h:+d}"] = lambda e, h=h: rule_patch_cum(e, h)

    scored = []
    y = np.array([e["label"] for e in rows], dtype=bool)
    for name, fn in candidates.items():
        pred = np.array([bool(fn(e)) for e in rows], dtype=bool)
        acc = float((pred == y).mean())
        scored.append(
            {
                "rule": name,
                "agreement": acc,
                "tp": int((pred & y).sum()),
                "fp": int((pred & ~y).sum()),
                "fn": int((~pred & y).sum()),
                "tn": int((~pred & ~y).sum()),
            }
        )
    scored.sort(key=lambda d: -d["agreement"])
    print("[s6] beste Regelkandidaten (Uebereinstimmung mit dem gespeicherten Label):")
    for s in scored[:10]:
        print(f"    {s['agreement']*100:6.2f} %  {s['rule']:32s} tp={s['tp']} fp={s['fp']} fn={s['fn']} tn={s['tn']}")

    # -------------------------------------------------- Shortcut-Quantifizierung
    fw = fire_in_window
    pos = [e for e in fw if e["label"] == 1]
    neg = [e for e in fw if e["label"] == 0]
    shortcut = {
        "positives_with_fire_in_input_patch": sum(e["patch_any_fire_in_input"] for e in pos) / max(1, len(pos)),
        "negatives_with_fire_in_input_patch": sum(e["patch_any_fire_in_input"] for e in neg) / max(1, len(neg)),
        "positives_with_fire_at_last_input_step": sum(e["center_fire_at_last_step"] > 0 for e in pos) / max(1, len(pos)),
        "negatives_with_fire_at_last_input_step": sum(e["center_fire_at_last_step"] > 0 for e in neg) / max(1, len(neg)),
        "positives_mean_input_days_with_patch_fire": float(np.mean([e["n_input_days_with_patch_fire"] for e in pos])),
        "negatives_mean_input_days_with_patch_fire": float(np.mean([e["n_input_days_with_patch_fire"] for e in neg])),
    }
    # Trivialklassifikator "Feuer im Eingabepatch -> positiv"
    pred = np.array([e["patch_any_fire_in_input"] for e in fw], dtype=bool)
    yy = np.array([e["label"] == 1 for e in fw], dtype=bool)
    shortcut["trivial_input_fire_classifier"] = {
        "accuracy": float((pred == yy).mean()),
        "recall": float((pred & yy).sum() / max(1, yy.sum())),
        "precision": float((pred & yy).sum() / max(1, pred.sum())),
        "balanced_accuracy": float(
            0.5 * ((pred & yy).sum() / max(1, yy.sum()) + (~pred & ~yy).sum() / max(1, (~yy).sum()))
        ),
    }
    print(f"[s6] Shortcut ueber den Feuerkanal im Eingabefenster: {json.dumps(shortcut, indent=2)}")

    # cache_G behaelt den alten Namen (Bericht s7 liest ihn); andere Varianten
    # bekommen eine eigene Datei statt den G-Befund zu ueberschreiben.
    write_manifest(
        cfg,
        "s6_label_def" if args.cache == "cache_G" else f"s6_label_def_{args.cache}",
        args.config,
        {
            "cache": args.cache,
            "h5_attrs": attrs,
            "n_samples_checked": len(rows),
            "rule_candidates": scored,
            "best_rule": scored[0] if scored else None,
            "input_fire_shortcut": shortcut,
            "samples": rows,
            "fire_in_input_window": fire_in_window,
        },
    )


if __name__ == "__main__":
    main()

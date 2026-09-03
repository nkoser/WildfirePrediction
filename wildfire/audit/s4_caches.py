"""Stufe 4: Audit der vier vorgebauten Fenster-Caches.

Untersucht je Variante:
  * HDF5-Struktur: Datasets, Shapes, dtypes, Chunking, Kompression,
    Attribute (Kanalnamen, Fensterlaenge, Patchgroesse, Hash-Konfiguration);
  * ``samples_variant_<X>.json``: Split-Zuordnung, Sample-IDs, Metadaten,
    Klassenverteilung je Split, Jahresverteilung, raeumliche Verteilung;
  * ``norm_stats.npz``: Kanalzahl, welche Kanaele ausgenommen sind;
  * Wertestatistik auf einer deterministischen Zufallsstichprobe von
    Fenstern (NaN-Anteil, konstante Kanaele, Wertebereiche);
  * Ueberschneidungen der Sample-IDs zwischen den Varianten und zwischen
    Train/Val/Test (Leakage-Vorpruefung).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest


def walk_h5(obj, prefix="", out=None, depth=0):
    out = {} if out is None else out
    if depth > 4:
        return out
    for key, item in obj.items():
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Dataset):
            out[path] = {
                "kind": "dataset",
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "chunks": list(item.chunks) if item.chunks else None,
                "compression": item.compression,
                "compression_opts": item.compression_opts,
                "nbytes": int(item.nbytes),
                "nbytes_human": human_bytes(item.nbytes),
                "attrs": {k: _jsonable(v) for k, v in item.attrs.items()},
            }
        else:
            out[path] = {
                "kind": "group",
                "attrs": {k: _jsonable(v) for k, v in item.attrs.items()},
                "n_children": len(item),
            }
            walk_h5(item, path, out, depth + 1)
    return out


def _jsonable(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 200 else {"shape": list(v.shape), "head": v.ravel()[:20].tolist()}
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def describe_samples(doc) -> dict:
    """Robuste Beschreibung des Sample-Index, ohne dessen Schema zu raten."""
    info: dict = {"top_level_type": type(doc).__name__}
    if isinstance(doc, dict):
        info["top_level_keys"] = list(doc.keys())[:60]
        info["key_types"] = {k: type(v).__name__ for k, v in list(doc.items())[:60]}
        info["key_lengths"] = {
            k: (len(v) if hasattr(v, "__len__") else None) for k, v in list(doc.items())[:60]
        }
        for k, v in doc.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                info.setdefault("record_lists", {})[k] = {
                    "n": len(v),
                    "fields": sorted(v[0].keys()),
                    "example": {kk: _jsonable(vv) for kk, vv in list(v[0].items())[:30]},
                }
            elif isinstance(v, list) and v:
                info.setdefault("scalar_lists", {})[k] = {
                    "n": len(v),
                    "example": _jsonable(v[0]),
                    "dtype": type(v[0]).__name__,
                }
            elif isinstance(v, dict):
                info.setdefault("nested", {})[k] = {
                    "keys": list(v.keys())[:40],
                    "lengths": {kk: (len(vv) if hasattr(vv, "__len__") else None)
                                for kk, vv in list(v.items())[:40]},
                }
    elif isinstance(doc, list):
        info["n"] = len(doc)
        if doc and isinstance(doc[0], dict):
            info["fields"] = sorted(doc[0].keys())
            info["example"] = {k: _jsonable(v) for k, v in list(doc[0].items())[:30]}
    return info


def collect_records(doc) -> tuple[list[dict], str]:
    """Findet die eigentliche Sample-Liste, egal wie sie verschachtelt ist."""
    if isinstance(doc, list) and doc and isinstance(doc[0], dict):
        return doc, "root-list"
    if isinstance(doc, dict):
        # ALLE Record-Listen zusammenfuehren (z. B. positive_samples und
        # negative_samples), sonst faellt eine Klasse aus der Statistik.
        merged: list[dict] = []
        keys: list[str] = []
        for k, v in doc.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                keys.append(f"{k}({len(v)})")
                for r in v:
                    r = dict(r)
                    r.setdefault("_list_key", k)
                    merged.append(r)
        if merged:
            return merged, "+".join(keys)
    return [], ""


def field_stats(records: list[dict]) -> dict:
    stats: dict = {}
    if not records:
        return stats
    fields = sorted({k for r in records[:5000] for k in r})
    for f in fields:
        vals = [r.get(f) for r in records]
        non_null = [v for v in vals if v is not None]
        entry: dict = {"n_non_null": len(non_null), "type": type(non_null[0]).__name__ if non_null else None}
        if non_null and isinstance(non_null[0], (int, float, bool, str)):
            c = Counter(non_null)
            entry["n_unique"] = len(c)
            if len(c) <= 40:
                entry["histogram"] = {str(k): v for k, v in c.most_common()}
            else:
                entry["most_common"] = {str(k): v for k, v in c.most_common(10)}
                if isinstance(non_null[0], (int, float)) and not isinstance(non_null[0], bool):
                    a = np.asarray(non_null, dtype=float)
                    entry.update(min=float(a.min()), max=float(a.max()), mean=float(a.mean()))
        stats[f] = entry
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    root = Path(cfg["dataset_root"])
    out: dict[str, dict] = {}
    ids_by_variant: dict[str, set] = {}

    for cache in cfg["caches"]:
        cdir = root / cache
        print(f"[s4] === {cache} ===", flush=True)
        entry: dict = {"cache": cache, "dir": str(cdir), "files": {}}

        for p in sorted(cdir.iterdir()):
            entry["files"][p.name] = {"size_bytes": p.stat().st_size, "size_human": human_bytes(p.stat().st_size)}

        # --- norm_stats.npz
        npz_path = next(cdir.glob("norm_stats.npz"), None)
        if npz_path:
            with np.load(npz_path, allow_pickle=True) as z:
                entry["norm_stats"] = {
                    k: {
                        "shape": list(np.asarray(z[k]).shape),
                        "dtype": str(np.asarray(z[k]).dtype),
                        "values": np.asarray(z[k]).ravel()[:80].tolist()
                        if np.asarray(z[k]).size <= 200 else None,
                    }
                    for k in z.files
                }
            print(f"    norm_stats keys: {list(entry['norm_stats'])}")

        # --- samples_variant_X.json
        sjson = next(cdir.glob("samples_variant_*.json"), None)
        if sjson:
            doc = json.loads(sjson.read_text())
            entry["samples_schema"] = describe_samples(doc)
            records, where = collect_records(doc)
            entry["samples_records_from"] = where
            entry["n_records"] = len(records)
            entry["record_field_stats"] = field_stats(records)
            print(f"    {sjson.name}: {len(records)} Records aus '{where}'")

            # Split x Label Kreuztabelle, sofern erkennbar
            split_keys = [k for k in ("split", "_split_from_key", "subset", "partition") ]
            label_keys = [k for k in ("sample_type", "label", "y", "is_fire", "positive", "target", "class")]
            sk = next((k for k in split_keys if records and k in records[0]), None)
            # Jahr x Split, um den zeitlichen Hold-out zu verifizieren
            if records and "year" in records[0] and sk:
                ys: dict = defaultdict(Counter)
                for r in records:
                    ys[str(r.get(sk))][str(r.get("year"))] += 1
                entry["split_x_year"] = {k: dict(sorted(v.items())) for k, v in ys.items()}
            lk = next((k for k in label_keys if records and k in records[0]), None)
            if sk:
                cross: dict = defaultdict(Counter)
                for r in records:
                    cross[str(r.get(sk))][str(r.get(lk)) if lk else "n/a"] += 1
                entry["split_x_label"] = {k: dict(v) for k, v in cross.items()}
                print(f"    Split x Label ({sk} x {lk}): "
                      f"{ {k: dict(v) for k, v in cross.items()} }")

            # Sample-IDs fuer den Varianten-Vergleich
            # Es gibt kein ID-Feld; die fachliche Identitaet eines Samples ist
            # (Datum, Pixelzeile, Pixelspalte, Typ). Genau darauf muessen die
            # Leakage-Pruefungen aufsetzen.
            def sid(r: dict) -> str:
                return f"{r.get('date')}|{r.get('pixel_row')}|{r.get('pixel_col')}|{r.get('sample_type')}"

            entry["id_key"] = "date|pixel_row|pixel_col|sample_type (synthetisch)"
            ids_by_variant[cache] = {sid(r) for r in records}
            entry["n_unique_sample_ids"] = len(ids_by_variant[cache])
            entry["n_duplicate_sample_ids"] = len(records) - len(ids_by_variant[cache])

            # --- Leakage innerhalb einer Variante: teilen sich die Splits
            #     Pixelorte oder Kalendertage?
            by_split_px: dict[str, set] = defaultdict(set)
            by_split_date: dict[str, set] = defaultdict(set)
            by_split_year: dict[str, set] = defaultdict(set)
            for r in records:
                sp = str(r.get("split"))
                by_split_px[sp].add((r.get("pixel_row"), r.get("pixel_col")))
                by_split_date[sp].add(r.get("date"))
                by_split_year[sp].add(r.get("year"))
            splits = sorted(by_split_px)
            cross: dict = {}
            for i, a in enumerate(splits):
                for b in splits[i + 1:]:
                    cross[f"{a}|{b}"] = {
                        "shared_pixels": len(by_split_px[a] & by_split_px[b]),
                        "n_pixels_a": len(by_split_px[a]),
                        "n_pixels_b": len(by_split_px[b]),
                        "shared_dates": len(by_split_date[a] & by_split_date[b]),
                        "shared_years": sorted(str(y) for y in (by_split_year[a] & by_split_year[b])),
                    }
            entry["split_leakage"] = cross
            entry["years_per_split"] = {k: sorted(str(y) for y in v) for k, v in by_split_year.items()}
            print(f"    Split-Leakage: {cross}")

        # --- windows_*.h5
        h5_path = next(cdir.glob("*.h5"), None)
        if h5_path:
            with h5py.File(h5_path, "r") as f:
                entry["h5"] = {
                    "file": h5_path.name,
                    "size_bytes": h5_path.stat().st_size,
                    "root_attrs": {k: _jsonable(v) for k, v in f.attrs.items()},
                    "objects": walk_h5(f),
                }
                # groesstes Dataset = die Fenster
                dsets = {k: v for k, v in entry["h5"]["objects"].items() if v["kind"] == "dataset"}
                if dsets:
                    main_key = max(dsets, key=lambda k: dsets[k]["nbytes"])
                    entry["h5"]["main_dataset"] = main_key
                    d = f[main_key]
                    n = d.shape[0]
                    k = min(int(cfg["sampling"]["h5_sample_windows"]), n)
                    idx = np.sort(rng.choice(n, size=k, replace=False))
                    print(f"    {main_key} {d.shape} {d.dtype}: ziehe {k} Fenster ...", flush=True)
                    chan_axis = int(np.argmin([s for s in d.shape[1:]])) + 1 if d.ndim >= 3 else None
                    sample = np.stack([np.asarray(d[i]) for i in idx])
                    entry["h5"]["sample_stats"] = summarize_windows(sample, d.shape)
                for key, meta in dsets.items():
                    if meta["nbytes"] < 5e7 and meta["shape"] and meta["shape"][0] < 5_000_000:
                        arr = np.asarray(f[key][...])
                        if arr.dtype.kind in "iufb" and arr.size:
                            vals, counts = np.unique(arr, return_counts=True)
                            meta["value_histogram"] = (
                                {str(v): int(c) for v, c in zip(vals[:40], counts[:40])}
                                if vals.size <= 40 else
                                {"min": float(arr.min()), "max": float(arr.max()),
                                 "mean": float(arr.mean()), "n_unique": int(vals.size)}
                            )
        out[cache] = entry

    # --- Varianten-Vergleich / Leakage-Vorpruefung
    overlaps = {}
    variants = [c for c in cfg["caches"] if c in ids_by_variant]
    for i, a in enumerate(variants):
        for b in variants[i + 1:]:
            inter = ids_by_variant[a] & ids_by_variant[b]
            overlaps[f"{a}|{b}"] = {
                "n_a": len(ids_by_variant[a]),
                "n_b": len(ids_by_variant[b]),
                "n_intersection": len(inter),
                "jaccard": len(inter) / max(1, len(ids_by_variant[a] | ids_by_variant[b])),
            }
    print(f"[s4] ID-Ueberschneidungen zwischen Varianten: {overlaps}")

    write_manifest(cfg, "s4_caches", args.config, {"caches": out, "variant_id_overlaps": overlaps})


def summarize_windows(sample: np.ndarray, full_shape) -> dict:
    """Kanalweise Statistik auf der gezogenen Fenster-Stichprobe."""
    info: dict = {
        "sample_shape": list(sample.shape),
        "full_shape": list(full_shape),
        "dtype": str(sample.dtype),
        "global": {},
        "per_axis_candidates": {},
    }
    a = sample.astype(np.float64, copy=False)
    finite = np.isfinite(a)
    info["global"] = {
        "min": float(a[finite].min()) if finite.any() else None,
        "max": float(a[finite].max()) if finite.any() else None,
        "mean": float(a[finite].mean()) if finite.any() else None,
        "nan_fraction": float(1.0 - finite.mean()),
        "zero_fraction": float((a == 0).mean()),
    }
    # Kanalachse = die Achse mit einer plausiblen Kanalzahl (<= 128)
    for ax in range(1, sample.ndim):
        if sample.shape[ax] <= 128:
            moved = np.moveaxis(a, ax, 0).reshape(sample.shape[ax], -1)
            fin = np.isfinite(moved)
            info["per_axis_candidates"][f"axis{ax}"] = {
                "n": int(sample.shape[ax]),
                "min": np.where(fin.any(1), np.nanmin(np.where(fin, moved, np.nan), axis=1), np.nan).tolist(),
                "max": np.where(fin.any(1), np.nanmax(np.where(fin, moved, np.nan), axis=1), np.nan).tolist(),
                "mean": np.where(fin.any(1), np.nanmean(np.where(fin, moved, np.nan), axis=1), np.nan).tolist(),
                "std": np.where(fin.any(1), np.nanstd(np.where(fin, moved, np.nan), axis=1), np.nan).tolist(),
                "nan_fraction": (1.0 - fin.mean(axis=1)).tolist(),
                "n_constant_channels": int(
                    sum(1 for i in range(moved.shape[0]) if np.nanstd(np.where(fin[i], moved[i], np.nan)) == 0)
                ),
            }
    return info


if __name__ == "__main__":
    main()

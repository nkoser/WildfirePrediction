"""Stufe 5: Kanal-Identifikation, Zeitausrichtung und Leakage-Test.

Die Fenster-Caches liefern einen uint8-Tensor (N, 55, 13, 13, 61) ohne
Kanalnamen. Weder die Kanalzuordnung noch die Zeitausrichtung noch die
Labeldefinition sind dokumentiert - sie werden hier NICHT angenommen,
sondern durch exakten Abgleich mit den Rohrastern rekonstruiert:

1. Fuer ein Sample (Datum D, Zeile r, Spalte c) wird aus den tar-Archiven
   die Rohzeitreihe jedes Modalitaets-Bandes am Pixel (r, c) fuer
   D-``span`` .. D+``lead`` gelesen.
2. Jeder der 55 Cache-Kanaele wird als 61-Schritt-Reihe am Patchzentrum
   entnommen und gegen alle Rohreihen bei allen Zeitversaetzen exakt
   verglichen (uint8 == uint8, keine Toleranz).
3. Daraus folgen: Kanal -> (Modalitaet, Band), Zeitindex -> Kalendertag,
   und ob ein Zeitindex NACH dem Vorhersagezeitpunkt liegt.
4. Das Label wird gegen die Feuermaske an D-1, D, D+1, ... geprueft.

Das ist der entscheidende Leakage-Test: enthaelt das Eingabefenster den
Zielzeitpunkt oder den Feuerkanal am Zielzeitpunkt?
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

from wildfire.audit.common import assert_readonly, load_config, write_manifest

# Reihenfolge und Bandzahl je Modalitaet laut README-Tabelle.
MODALITY_BANDS = {
    "DEMs": 7,
    "ERA5": 18,
    "FWI": 6,
    "MCD09CMG": 4,
    "MCD09GA": 4,
    "MCD11A1": 4,
    "MCD12Q1": 1,
    "MCD14A1": 1,
    "MCD15A3H": 2,
    "NDVI_EVI": 2,
    "OSMs": 4,
    "VPD": 1,
    "Worldpop": 1,
}


def read_pixel_series(
    dataset_root: Path, modality: str, dates: list[dt.date], row: int, col: int
) -> np.ndarray:
    """Rohwerte aller Baender am Pixel (row, col) fuer die gegebenen Tage.

    Rueckgabe: (n_bands, n_dates) uint8, 255-Sentinel fuer fehlende Tage
    wird nicht verwendet - fehlende Tage bleiben als -1 markiert.
    """
    n_b = MODALITY_BANDS[modality]
    out = np.full((n_b, len(dates)), -1, dtype=np.int16)
    by_year: dict[int, list[tuple[int, dt.date]]] = defaultdict(list)
    for i, d in enumerate(dates):
        by_year[d.year].append((i, d))

    for year, items in by_year.items():
        tar_path = dataset_root / "Entire_Canada_Maps" / modality / f"{modality}__{year}.tar"
        if not tar_path.exists():
            continue
        wanted = {f"{d.year}_{d.month:02d}_{d.day:02d}.tif": i for i, d in items}
        with tarfile.open(tar_path, "r:") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                base = Path(m.name).name
                if base not in wanted:
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                blob = fh.read()
                with MemoryFile(blob) as mem, mem.open() as ds:
                    win = ((row, row + 1), (col, col + 1))
                    vals = ds.read(window=win).reshape(ds.count)
                out[:, wanted[base]] = vals[:n_b]
    return out


def identify(cache_dir: Path, dataset_root: Path, sample_indices: list[int], span: int, lead: int) -> dict:
    h5_path = next(cache_dir.glob("*.h5"))
    sjson = next(cache_dir.glob("samples_variant_*.json"))
    doc = json.loads(sjson.read_text())
    records = doc["positive_samples"] + doc["negative_samples"]

    results: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        n_t = f["windows"].shape[-1]
        centre = f["windows"].shape[2] // 2
        for si in sample_indices:
            meta = f["meta"][si]
            label = int(f["labels"][si])
            rec = records[si]
            date = dt.date.fromisoformat(rec["date"])
            row, col = int(rec["pixel_row"]), int(rec["pixel_col"])
            assert int(meta[0]) == int(date.strftime("%Y%m%d")), "meta != samples.json"
            assert (int(meta[1]), int(meta[2])) == (row, col), "meta-Pixel != samples.json"

            win = np.asarray(f["windows"][si])  # (55, 13, 13, 61)
            series = win[:, centre, centre, :].astype(np.int16)  # (55, 61)

            dates = [date + dt.timedelta(days=k) for k in range(-span, lead + 1)]
            raw: dict[str, np.ndarray] = {}
            for mod in MODALITY_BANDS:
                raw[mod] = read_pixel_series(dataset_root, mod, dates, row, col)

            matches: list[dict] = []
            for ch in range(series.shape[0]):
                s = series[ch]
                hits = []
                for mod, arr in raw.items():
                    for b in range(arr.shape[0]):
                        ref = arr[b]
                        for off in range(0, len(dates) - n_t + 1):
                            seg = ref[off : off + n_t]
                            if np.any(seg < 0):
                                continue
                            if np.array_equal(seg, s):
                                first = dates[off]
                                last = dates[off + n_t - 1]
                                hits.append(
                                    {
                                        "modality": mod,
                                        "band": b,
                                        "t0_date": first.isoformat(),
                                        "t_last_date": last.isoformat(),
                                        "days_after_sample_date": (last - date).days,
                                    }
                                )
                matches.append(
                    {
                        "channel": ch,
                        "n_hits": len(hits),
                        "hits": hits[:4],
                        # ALLE Kandidaten, nicht nur die ersten - die
                        # Schnittmenge ueber Samples braucht sie vollstaendig.
                        "hit_names": sorted({f"{h['modality']}[{h['band']}]" for h in hits}),
                        "offsets": sorted({h["days_after_sample_date"] for h in hits}),
                        "constant": bool(np.all(s == s[0])),
                        "value_range": [int(s.min()), int(s.max())],
                    }
                )

            # Labelpruefung gegen die Feuermaske um D herum
            fire = raw["MCD14A1"][0]
            label_check = {}
            for k in range(-3, min(lead, 5) + 1):
                i = span + k
                if 0 <= i < len(dates):
                    label_check[f"D{k:+d}"] = int(fire[i])

            results.append(
                {
                    "sample_index": si,
                    "record": rec,
                    "label": label,
                    "meta": [int(x) for x in meta],
                    "channel_matches": matches,
                    "fire_mask_around_D": label_check,
                    "n_channels_identified": sum(1 for m in matches if m["n_hits"] > 0),
                    "n_channels_constant": sum(1 for m in matches if m["constant"]),
                }
            )
            print(
                f"    Sample {si} {rec['date']} ({row},{col}) label={label}: "
                f"{results[-1]['n_channels_identified']}/55 Kanaele zugeordnet, "
                f"Feuermaske {label_check}",
                flush=True,
            )
    return {"h5_attrs": attrs, "n_timesteps": n_t, "samples": results}


def consolidate(samples: list[dict]) -> dict:
    """Kanalzuordnung ueber alle geprueften Samples zusammenfassen.

    Ein Kanal, dessen 61-Schritt-Reihe in einem Sample KONSTANT ist (etwa
    die Feuermaske an einem Negativpixel oder ein statischer Kanal), passt
    dort auf jedes gleich konstante Rohband. Ein Mehrheitsentscheid ueber
    Samples wuerde daraus eine falsche Zuordnung machen. Daher:

    1. Schnittmenge der Kandidaten ueber ALLE Samples - nur was ueberall
       passt, kommt in Frage.
    2. Bleibt mehr als ein Kandidat, entscheidet die Evidenz aus Samples
       mit NICHT konstanter Reihe.
    3. Bleibt es mehrdeutig, wird per Elimination gegen die bereits
       eindeutig vergebenen (Modalitaet, Band) aufgeloest.
    """
    per_channel_sets: dict[int, list[set[str]]] = defaultdict(list)
    informative: dict[int, list[set[str]]] = defaultdict(list)
    per_channel_align: dict[int, list[int]] = defaultdict(list)
    for s in samples:
        for m in s["channel_matches"]:
            names = set(m.get("hit_names") or [])
            if not names:
                continue
            per_channel_sets[m["channel"]].append(names)
            if not m.get("constant"):
                informative[m["channel"]].append(names)
                per_channel_align[m["channel"]].extend(m.get("offsets") or [])

    out: dict[str, dict] = {}
    for ch in range(55):
        sets = per_channel_sets.get(ch, [])
        inter = set.intersection(*sets) if sets else set()
        inf = informative.get(ch, [])
        if len(inter) > 1 and inf:
            narrowed = inter & set.intersection(*inf)
            if narrowed:
                inter = narrowed
        aligns = per_channel_align.get(ch, [])
        out[str(ch)] = {
            "assignment": sorted(inter)[0] if len(inter) == 1 else None,
            "candidates": sorted(inter),
            "ambiguous": len(inter) != 1,
            "n_samples_matched": len(sets),
            "n_samples_informative": len(inf),
            "last_timestep_offset_days": (
                max(set(aligns), key=aligns.count) if aligns else None
            ),
        }

    # Elimination fuer die verbliebenen mehrdeutigen Kanaele.
    expected = {f"{mod}[{b}]" for mod, n in MODALITY_BANDS.items() for b in range(n)}
    for _ in range(len(out)):
        taken = {v["assignment"] for v in out.values() if v["assignment"]}
        free = expected - taken
        progress = False
        for v in out.values():
            if v["assignment"]:
                continue
            cands = [c for c in v["candidates"] if c in free] or sorted(free)
            if len(cands) == 1:
                v["assignment"] = cands[0]
                v["resolved_by"] = "Elimination"
                v["ambiguous"] = False
                progress = True
        if not progress:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--span", type=int, default=70)
    ap.add_argument("--lead", type=int, default=7)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    root = Path(cfg["dataset_root"])
    cdir = root / args.cache
    # je zur Haelfte Positive (Index < 8704) und Negative
    n_pos = 8704
    k = max(1, args.n_samples // 2)
    idx = sorted(
        rng.choice(n_pos, size=k, replace=False).tolist()
        + (n_pos + rng.choice(26112 - n_pos, size=k, replace=False)).tolist()
    )
    print(f"[s5] Cache {args.cache}, Samples {idx}, Fenster D-{args.span}..D+{args.lead}")

    res = identify(cdir, root, idx, args.span, args.lead)
    mapping = consolidate(res["samples"])

    identified = [k_ for k_, v in mapping.items() if v["assignment"]]
    offsets = {v["last_timestep_offset_days"] for v in mapping.values() if v["assignment"]}
    fire_channels = [
        k_ for k_, v in mapping.items()
        if (v.get("assignment") or "").startswith("MCD14A1")
    ]

    print(f"[s5] zugeordnete Kanaele: {len(identified)}/55")
    print(f"[s5] Zeitversatz des letzten Fensterschritts relativ zum Sample-Datum: {offsets}")
    print(f"[s5] Feuermaske-Kanaele im Eingabefenster: {fire_channels}")
    for k_, v in sorted(mapping.items(), key=lambda kv: int(kv[0])):
        print(f"    ch{int(k_):02d} -> {str(v['assignment']):22s} "
              f"offset={v['last_timestep_offset_days']} "
              f"eindeutig={not v['ambiguous']} kandidaten={len(v['candidates'])}")

    leakage = {
        "window_contains_target_day": any(o is not None and o >= 0 for o in offsets),
        "window_contains_future_days": any(o is not None and o > 0 for o in offsets),
        "fire_mask_in_input_channels": bool(fire_channels),
        "fire_mask_channel_indices": fire_channels,
        "last_timestep_offsets_observed": sorted(o for o in offsets if o is not None),
    }
    print(f"[s5] LEAKAGE: {leakage}")

    write_manifest(
        cfg,
        "s5_alignment",
        args.config,
        {
            "cache": args.cache,
            "h5_attrs": res["h5_attrs"],
            "n_timesteps": res["n_timesteps"],
            "channel_map": mapping,
            "leakage": leakage,
            "samples": res["samples"],
        },
    )


if __name__ == "__main__":
    main()

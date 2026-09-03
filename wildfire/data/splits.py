"""Definiert und prueft die leakage-sicheren Splits.

Ein Split wird hier als REGEL definiert (Jahresmengen, Regionenmengen,
Pufferzonen), nicht als materialisierte Sampleliste. Die Regel wird auf
die Zuendungs- und Ereignistabelle angewandt, geprueft und mit Hash
versioniert; der Landmark-Generator konsumiert dieselbe Regel spaeter.

Vier Splits:

  temporal      Train 2000-2017, Val 2018-2020, Test 2021-2025.
                Alle Regionen. Der Standardsplit.
  spatial_east  Raeumlich OOD: der oestliche boreale Wald (Quebec, Ontario,
                Atlantikprovinzen) ausschliesslich im Test, Training im
                Westen. Prueft Generalisierung ueber Feuerregime hinweg
                bei nur ~10 % Datenverlust.
  spatial_bc    Haerterer raeumlicher OOD: British Columbia allein im Test.
                Kordillerenregime, stark reliefgesteuert, 27,6 % der Feuer.
  extreme_2023  Nur die Rekordsaison 2023 im Test, alles davor im Training.

Geprueft wird:
  * keine gemeinsame Ereignis-ID ueber Splits;
  * keine gemeinsamen Kalendertage zwischen zeitlichen Splits, inklusive
    des Prognosehorizonts (ein Landmark darf mit seinem Zielfenster nicht
    in den naechsten Split hineinragen);
  * raeumlicher Puffer zwischen Regionen, mindestens der halben
    Patchkantenlaenge, damit sich keine Eingabepatches ueberlappen;
  * kein Pixel-Zeitpunkt in zwei Splits.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

from wildfire.audit.common import assert_readonly, load_config, write_manifest

WEST = ["Alberta", "British Columbia", "Manitoba", "Northwest Territories",
        "Nunavut", "Saskatchewan", "Yukon"]
EAST = ["Québec", "Ontario", "Newfoundland and Labrador", "New Brunswick",
        "Nova Scotia", "Prince Edward Island"]

SPLIT_DEFS = {
    "temporal": {
        "kind": "temporal",
        "train": {"years": list(range(2000, 2018))},
        "val": {"years": list(range(2018, 2021))},
        "test": {"years": list(range(2021, 2026))},
    },
    "spatial_east": {
        "kind": "spatial",
        "train": {"years": list(range(2000, 2021)), "regions": WEST},
        "val": {"years": list(range(2021, 2026)), "regions": WEST},
        "test": {"years": list(range(2000, 2026)), "regions": EAST},
    },
    "spatial_bc": {
        "kind": "spatial",
        "train": {"years": list(range(2000, 2021)),
                  "regions": [r for r in WEST + EAST if r != "British Columbia"]},
        "val": {"years": list(range(2021, 2026)),
                "regions": [r for r in WEST + EAST if r != "British Columbia"]},
        "test": {"years": list(range(2000, 2026)), "regions": ["British Columbia"]},
    },
    "extreme_2023": {
        "kind": "temporal",
        "train": {"years": list(range(2000, 2021))},
        "val": {"years": [2021, 2022]},
        "test": {"years": [2023]},
    },
}


def assign(rec_year: int, rec_region: str, spec: dict) -> bool:
    if "years" in spec and rec_year not in spec["years"]:
        return False
    if "regions" in spec and rec_region not in spec["regions"]:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--patch-size", type=int, default=33)
    ap.add_argument("--horizon-days", type=int, default=100)
    ap.add_argument("--season-start", default="05-01")
    ap.add_argument("--season-end", default="08-31")
    ap.add_argument("--mask", default="observed_nfdb.npy")
    ap.add_argument("--ignitions", default="derived_data/events/ignitions_nfdb_min1ha.csv",
                    help="Zuendungstabelle - Standard sind die gemeldeten Braende")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    mask_dir = exp / "derived_data" / "masks"
    region_map = np.load(mask_dir / "region.npy")
    valid = np.load(mask_dir / args.mask)
    print(f"[splits] Maske {args.mask}: {int(valid.sum())} Pixel")
    reg_man = json.loads((exp / "data_manifests" / "d2_regions.json").read_text())
    name_of_code = {v: k for k, v in reg_man["code_of"].items()}

    ign_path = exp / args.ignitions
    print(f"[splits] Zuendungen aus {ign_path.name}")
    rows = list(csv.DictReader(ign_path.open()))
    for r in rows:
        r["_region"] = name_of_code.get(int(region_map[int(r["row"]), int(r["col"])]), "?")
        r["_year"] = int(r["year"])
        r["_in_season"] = args.season_start <= r["date"][5:] <= args.season_end
    n_season = sum(r["_in_season"] for r in rows)
    print(f"[splits] {len(rows)} Zuendungen, davon {n_season} in der Saison "
          f"{args.season_start}..{args.season_end} ({100*n_season/len(rows):.1f} %)")

    # ------------------------------------------------ raeumliche Pufferzonen
    half = args.patch_size // 2
    buffers = {}
    for split_name, spec in SPLIT_DEFS.items():
        if spec["kind"] != "spatial":
            continue
        test_codes = [reg_man["code_of"][n] for n in spec["test"]["regions"]]
        test_mask = np.isin(region_map, test_codes)
        dil = ndimage.binary_dilation(test_mask, np.ones((2 * half + 1, 2 * half + 1), bool))
        buffer_band = dil & ~test_mask & valid
        np.save(mask_dir / f"buffer_{split_name}.npy", buffer_band)
        buffers[split_name] = {
            "half_patch": half,
            "n_buffer_pixels": int(buffer_band.sum()),
            "share_of_valid": float(buffer_band.sum() / max(1, valid.sum())),
            "file": str(mask_dir / f"buffer_{split_name}.npy"),
        }
        print(f"[splits] {split_name}: Pufferband {int(buffer_band.sum())} Pixel "
              f"({100*buffer_band.sum()/max(1,valid.sum()):.2f} % der gueltigen Flaeche) "
              f"wird aus dem Training entfernt")

    # ------------------------------------------------ Zuweisung und Pruefung
    out_dir = exp / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for split_name, spec in SPLIT_DEFS.items():
        buf = None
        if spec["kind"] == "spatial":
            buf = np.load(mask_dir / f"buffer_{split_name}.npy")

        assignment: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        dropped = 0
        for r in rows:
            if not r["_in_season"]:
                continue
            placed = None
            for part in ("test", "val", "train"):
                if assign(r["_year"], r["_region"], spec[part]):
                    placed = part
                    break
            if placed is None:
                dropped += 1
                continue
            if placed in ("train", "val") and buf is not None and buf[int(r["row"]), int(r["col"])]:
                dropped += 1     # im Pufferband: verwerfen
                continue
            assignment[placed].append(r)

        # ---- Ereignisgruppierung setzt sich gegen die Regionenregel durch.
        # Ein Brand kann eine Provinzgrenze ueberschreiten; dann liegen
        # Pixel desselben Ereignisses beiderseits. Das Pufferband faengt das
        # NICHT ab, weil ein Ereignis weiter reichen kann als der Puffer.
        # Regel: beruehrt ein Ereignis den Test, gehoert es ganz in den Test.
        test_events = {r["event_id"] for r in assignment["test"] if r["event_id"] != "-1"}
        regrouped = 0
        for part in ("train", "val"):
            keep = []
            for r in assignment[part]:
                if r["event_id"] != "-1" and r["event_id"] in test_events:
                    regrouped += 1
                    continue
                keep.append(r)
            assignment[part] = keep
        # Analog fuer Val gegen Train, damit auch dort kein Ereignis geteilt wird.
        val_events = {r["event_id"] for r in assignment["val"] if r["event_id"] != "-1"}
        keep = []
        for r in assignment["train"]:
            if r["event_id"] != "-1" and r["event_id"] in val_events:
                regrouped += 1
                continue
            keep.append(r)
        assignment["train"] = keep
        dropped += regrouped

        # ---- Pruefung 1: Ereignis-IDs ueber Splits
        ev_of = {p: {r["event_id"] for r in v if r["event_id"] != "-1"}
                 for p, v in assignment.items()}
        ev_overlap = {
            f"{a}|{b}": len(ev_of[a] & ev_of[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        }

        # ---- Pruefung 2: Kalendertage inkl. Horizont
        day_of = {p: {r["date"] for r in v} for p, v in assignment.items()}
        day_overlap = {
            f"{a}|{b}": len(day_of[a] & day_of[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        # Ragt ein Landmark mit Horizont in den naechsten Split?
        horizon_conflicts = 0
        if spec["kind"] == "temporal":
            for a, b in (("train", "val"), ("val", "test")):
                ya, yb = set(spec[a]["years"]), set(spec[b]["years"])
                for r in assignment[a]:
                    end = dt.date.fromisoformat(r["date"]) + dt.timedelta(days=args.horizon_days)
                    if end.year in yb:
                        horizon_conflicts += 1

        # ---- Pruefung 3: identische Pixel-Zeitpunkte
        key_of = {p: {(r["t"], r["row"], r["col"]) for r in v} for p, v in assignment.items()}
        key_overlap = {
            f"{a}|{b}": len(key_of[a] & key_of[b])
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        }

        # ---- Pruefung 4: raeumlicher Mindestabstand bei raeumlichen Splits
        min_dist = None
        if spec["kind"] == "spatial" and assignment["test"] and assignment["train"]:
            test_px = np.array([[int(r["row"]), int(r["col"])] for r in assignment["test"]])
            tm = np.zeros(region_map.shape, bool)
            tm[test_px[:, 0], test_px[:, 1]] = True
            dist = ndimage.distance_transform_edt(~tm)
            tr_px = np.array([[int(r["row"]), int(r["col"])] for r in assignment["train"]])
            min_dist = float(dist[tr_px[:, 0], tr_px[:, 1]].min())

        checks = {
            "shared_event_ids": ev_overlap,
            "shared_dates": day_overlap,
            "shared_pixel_timepoints": key_overlap,
            "horizon_crossings": horizon_conflicts,
            "min_train_test_pixel_distance": min_dist,
            "required_min_distance": half if spec["kind"] == "spatial" else None,
        }
        passed = (
            all(v == 0 for v in ev_overlap.values())
            and all(v == 0 for v in key_overlap.values())
            and horizon_conflicts == 0
            and (min_dist is None or min_dist >= half)
        )

        counts = {p: len(v) for p, v in assignment.items()}
        per_year = {p: dict(sorted(Counter(r["_year"] for r in v).items()))
                    for p, v in assignment.items()}
        per_region = {p: dict(Counter(r["_region"] for r in v).most_common())
                      for p, v in assignment.items()}

        # ---- Split-Datei schreiben
        sp = out_dir / f"{split_name}.csv"
        with sp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["split", "t", "date", "year", "row", "col", "lon", "lat",
                        "event_id", "region"])
            for part in ("train", "val", "test"):
                for r in assignment[part]:
                    w.writerow([part, r["t"], r["date"], r["year"], r["row"], r["col"],
                                r["lon"], r["lat"], r["event_id"], r["_region"]])
        split_hash = hashlib.sha256(sp.read_bytes()).hexdigest()[:16]

        results[split_name] = {
            "definition": spec,
            "counts": counts,
            "dropped": dropped,
            "dropped_by_event_regrouping": regrouped,
            "per_year": per_year,
            "per_region": per_region,
            "checks": checks,
            "passed": passed,
            "file": str(sp),
            "sha256_16": split_hash,
        }
        flag = "OK" if passed else "FEHLGESCHLAGEN"
        print(f"\n[splits] {split_name}: train={counts['train']} val={counts['val']} "
              f"test={counts['test']} verworfen={dropped} "
              f"(davon {regrouped} wegen Ereignisgruppierung)  -> {flag}")
        print(f"    Ereignis-IDs ueber Splits: {ev_overlap}")
        print(f"    Pixel-Zeitpunkte ueber Splits: {key_overlap}")
        print(f"    Horizont-Ueberschreitungen: {horizon_conflicts}")
        if min_dist is not None:
            print(f"    Mindestabstand Train->Test: {min_dist:.1f} Pixel (gefordert {half})")
        print(f"    Hash: {split_hash}")

    all_passed = all(v["passed"] for v in results.values())
    print(f"\n[splits] Gesamtergebnis: {'alle Pruefungen bestanden' if all_passed else 'PRUEFUNG FEHLGESCHLAGEN'}")

    write_manifest(
        cfg,
        "d3_splits",
        args.config,
        {
            "patch_size": args.patch_size,
            "horizon_days": args.horizon_days,
            "season": {"start": args.season_start, "end": args.season_end},
            "n_ignitions_total": len(rows),
            "n_ignitions_in_season": n_season,
            "buffers": buffers,
            "splits": results,
            "all_checks_passed": all_passed,
        },
    )


if __name__ == "__main__":
    main()

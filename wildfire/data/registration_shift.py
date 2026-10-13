"""Was wuerde eine um eine Zelle verschobene Zuordnung kosten?

Der Registrierungstest in :mod:`wildfire.data.nfdb_validation` prueft, OB die
Umrechnung von Grad auf Zeile und Spalte stimmt. Diese Auswertung beantwortet
die andere Haelfte der Frage: Wie schlimm waere es, wenn sie es nicht taete?

Dazu wird jede Zuendung einer Saison zweimal zugeordnet - einmal korrekt, einmal
absichtlich verschoben - und verglichen, was an beiden Stellen im Wuerfel steht.
Gemessen werden drei Dinge:

  * bleibt die Zelle in der Gueltigkeitsmaske,
  * traegt sie dieselbe IGBP-Landbedeckungsklasse,
  * wie weit weichen ausgewaehlte Kanaele ab (in den quantisierten
    Einheiten des Wuerfels, 0-255).

Was hier NICHT gemessen wird: die Auswirkung auf die Modellguete. Dafuer
muesste unter der verschobenen Zuordnung neu trainiert werden. Ebenso sagt der
Landbedeckungsvergleich nur "gleiche Klasse ja/nein", nicht wie aehnlich die
verwechselten Klassen einander sind.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.cube import channel_layout
from wildfire.data.dataset import CubeReader
from wildfire.data.nfdb_validation import load_nfdb

# Kanaele mit sehr unterschiedlichem raeumlichem Verhalten: ERA5 ist aus 0,1°
# hochgesampelt und aendert sich ueber eine Zelle kaum, NDVI und Landbedeckung
# wechseln auf 5 km scharf. Der Kontrast ist der Punkt der Auswertung.
WATCH = ("NDVI", "FWI", "vapor_pressure_deficit", "temperature_2m")
SHIFTS = {"row+1": (1, 0), "col+1": (0, 1), "row+1_col+1": (1, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nfdb", default="/data/Wildfire/artifacts/NFDB_point_txt.zip")
    ap.add_argument("--year", type=int, default=2023)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    tr = man["grid"]["transform"]
    H, W = man["grid"]["height"], man["grid"]["width"]
    valid = np.load(exp / "derived_data" / "masks" / "valid.npy")

    dyn_ch, stat_ch = channel_layout()
    dyn_idx = {c["name"]: i for i, c in enumerate(dyn_ch)}
    stat_idx = {c["name"]: i for i, c in enumerate(stat_ch)}
    watch = {n: dyn_idx[n] for n in WATCH if n in dyn_idx}
    missing = [n for n in WATCH if n not in dyn_idx]
    if missing:
        print(f"[shift] nicht im dynamischen Teil, uebersprungen: {missing}")
    lc = stat_idx.get("landcover_IGBP")

    reader = CubeReader(exp)
    have = reader.available_days()
    t_of = {d: i for i, d in enumerate(reader.dates)}

    recs = []
    for r in load_nfdb(Path(args.nfdb)):
        try:
            y, mo, da = int(r["YEAR"]), int(r["MONTH"]), int(r["DAY"])
            lat, lon = float(r["LATITUDE"]), float(r["LONGITUDE"])
        except (ValueError, TypeError):
            continue
        if y != args.year or not (1 <= mo <= 12 and 1 <= da <= 31):
            continue
        try:
            t = t_of.get(dt.date(y, mo, da).isoformat())
        except ValueError:
            continue
        if t is None or t not in have:
            continue
        rr = math.floor((lat - tr[5]) / tr[4])
        cc = math.floor((lon - tr[2]) / tr[0])
        if 0 <= rr < H and 0 <= cc < W:
            recs.append((t, rr, cc))
    print(f"[shift] {len(recs)} Zuendungen der Saison {args.year} im Wuerfel")

    names = ["unveraendert", *SHIFTS]
    n_mask = dict.fromkeys(names, 0)
    n_same_lc = dict.fromkeys(names, 0)
    n_lc_cmp = dict.fromkeys(names, 0)
    diffs: dict[str, dict[str, list]] = {k: {n: [] for n in watch} for k in SHIFTS}

    for t, rr, cc in recs:
        p = reader.patch(t, rr, cc, 1).astype(np.int16)     # (3,3,C_dyn)
        s = reader.static_patch(t, rr, cc, 1)               # (3,3,C_stat)
        n_mask["unveraendert"] += bool(valid[rr, cc])
        n_same_lc["unveraendert"] += 1
        n_lc_cmp["unveraendert"] += 1
        for key, (dr, dc) in SHIFTS.items():
            r2, c2 = rr + dr, cc + dc
            if not (0 <= r2 < H and 0 <= c2 < W):
                continue
            n_mask[key] += bool(valid[r2, c2])
            if lc is not None:
                n_lc_cmp[key] += 1
                n_same_lc[key] += int(s[1, 1, lc] == s[1 + dr, 1 + dc, lc])
            for nm, ci in watch.items():
                diffs[key][nm].append(abs(int(p[1, 1, ci]) - int(p[1 + dr, 1 + dc, ci])))

    n = max(1, len(recs))
    print(f"\n{'Zuordnung':16s} {'in Maske':>10s} {'gleiche Landbedeckung':>23s}")
    out_rows = {}
    for key in names:
        lc_share = n_same_lc[key] / max(1, n_lc_cmp[key])
        out_rows[key] = {"in_valid_mask": n_mask[key] / n, "same_landcover": lc_share}
        print(f"{key:16s} {100*n_mask[key]/n:9.1f} % {100*lc_share:22.1f} %")

    print(f"\nMittlere absolute Kanalabweichung (uint8, 0-255):")
    print(f"{'Kanal':18s}" + "".join(f"{k:>14s}" for k in SHIFTS))
    chan = {}
    for nm in watch:
        row = {k: float(np.mean(diffs[k][nm])) if diffs[k][nm] else None for k in SHIFTS}
        chan[nm] = row
        print(f"{nm:18s}" + "".join(f"{row[k]:14.2f}" for k in SHIFTS))

    write_manifest(
        cfg, "d10_registration_shift", args.config,
        {
            "purpose": "Kosten einer um eine Zelle verschobenen Zuordnung",
            "year": args.year,
            "n_ignitions": len(recs),
            "shifts": {k: list(v) for k, v in SHIFTS.items()},
            "by_assignment": out_rows,
            "mean_abs_channel_diff_uint8": chan,
            "not_measured": [
                "Auswirkung auf Modellguete - es wurde nicht neu trainiert",
                "nur ein Jahr, nur die gelisteten Kanaele",
                "Landbedeckung nur als Ja/Nein, nicht die Aehnlichkeit der Klassen",
            ],
        },
    )


if __name__ == "__main__":
    main()

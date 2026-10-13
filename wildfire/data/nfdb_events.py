"""Echte Zuendungen aus der kanadischen Branddatenbank als Labelquelle.

Bis hierher war das Label ein PROXY: erstes MODIS-Feuer nach sieben
feuerfreien Tagen in der Umgebung. Die National Forestry Database liefert
stattdessen gemeldete Braende mit Koordinate, Datum, Ursache und Flaeche -
von den Provinzbehoerden erhoben, unabhaengig vom Satelliten.

Was sich dadurch aendert:

  * Aus "Fire Occurrence" wird **Ignition Forecasting**. Der Arbeitsplan
    macht das in Phase 1.3 explizit von der Verfuegbarkeit echter
    Zuendpunkte abhaengig.
  * **Competing Risks** werden moeglich: Blitz gegen Mensch, mit je
    ausreichend Faellen. Das ist Phase 2.3, bisher gestrichen.
  * Die MODIS-Feuermaske ist nicht mehr die Zielgroesse, sondern ein
    EINGABEKANAL. Die Sperre der Kanaele 0 und 38-41 verliert damit ihren
    urspruenglichen Grund - ein abgeschwaechter bleibt, siehe unten.
  * **Ereignis-IDs sind gegeben statt abgeleitet.** Jeder NFDB-Eintrag IST
    ein Brand; die raumzeitliche Zusammenhangsanalyse entfaellt.

Entscheidungen, die hier getroffen werden:

  Mindestflaeche   Standard 1 ha. 73 % der gemeldeten Braende sind kleiner;
                   ein 0,1-ha-Feuer in einer 25-km2-Zelle ist physikalisch
                   etwas anderes als ein Grossbrand, und die Meldepraxis
                   fuer Kleinstbraende unterscheidet sich stark zwischen
                   Provinzen. Der Schwellwert ist ablatierbar.
  Doppelte         Mehrere Meldungen am selben Pixel und Tag werden zu
                   einer zusammengefasst; behalten wird die groesste.
  Risikosperre     Landmarks an Pixeln, die in den letzten C Tagen eine
                   Zuendung hatten, werden ausgeschlossen - "Zeit bis zur
                   naechsten Zuendung" ist dort nicht definiert.

Warum die Kanalsperre trotzdem bleibt: MODIS detektiert ein Feuer am Tag
seiner Entstehung oder kurz danach. Ein Landmark bei t mit Zuendung bei
t+1 haette die Waermesignatur schon im Eingabefenster. Der Grund ist also
nicht mehr "das Label steht im Input", sondern Detektionsvorlauf - und
deshalb genuegt eine kuerzere Sperre.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest
from wildfire.data.landmarks import DAY_BITS
from wildfire.data.nfdb_validation import load_nfdb

CAUSE_CODE = {"N": 1, "H": 2, "U": 0}      # Blitz, Mensch, unbekannt

# Behoerden, die in einem Jahr gar nichts melden, liefern dort keine
# Negative, sondern UNBEOBACHTETE Zeit. Der Unterschied ist entscheidend:
# ein Pixel ohne Meldung ist nicht "nicht gebrannt", sondern "nicht
# beobachtet". Solche Regionen werden aus der Gueltigkeitsmaske entfernt,
# statt ihre Pixel als Negative zu zaehlen.
AGENCY_TO_REGION = {
    "BC": "British Columbia", "AB": "Alberta", "SK": "Saskatchewan",
    "MB": "Manitoba", "ON": "Ontario", "QC": "Québec",
    "NB": "New Brunswick", "NS": "Nova Scotia",
    "PE": "Prince Edward Island", "NL": "Newfoundland and Labrador",
    "YT": "Yukon", "NT": "Northwest Territories", "NU": "Nunavut",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nfdb", default="/data/Wildfire/artifacts/NFDB_point_txt.zip")
    ap.add_argument("--min-size-ha", type=float, default=1.0)
    # Das Label-Fenster eines Landmarks reicht bis t+H: bei Landmarks bis zum
    # 31.08. und H = 100 also bis in den Dezember. Frueher stand hier
    # 05-01..09-30 - Zuendungen ab Oktober fehlten dann im Index, und
    # Landmarks ab Ende Juni wurden als "bis H ueberlebt" gelabelt statt mit
    # ihrer spaeten Zuendung. Der Index deckt deshalb das ganze Jahr ab; die
    # Saison der LANDMARKS filtern landmarks.py, splits.py und climatology.py
    # selbst.
    ap.add_argument("--season-start", default="01-01")
    ap.add_argument("--season-end", default="12-31")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    tr = man["grid"]["transform"]
    H, W = man["grid"]["height"], man["grid"]["width"]
    valid = np.load(exp / "derived_data" / "masks" / "valid.npy")
    region = np.load(exp / "derived_data" / "masks" / "region.npy")

    fi = np.load(exp / "derived_data" / "events" / "fire_index.npz", allow_pickle=False)
    dates = [str(d) for d in fi["dates"]]
    t_of = {d: i for i, d in enumerate(dates)}

    rows = load_nfdb(Path(args.nfdb))
    print(f"[nfdb] {len(rows)} Rohsaetze")

    drop = Counter()
    recs: dict[tuple[int, int], dict] = {}     # (t, flat) -> bester Satz
    merged = 0
    for r in rows:
        try:
            y, mo, da = int(r["YEAR"]), int(r["MONTH"]), int(r["DAY"])
        except (TypeError, ValueError):
            drop["kein Datum"] += 1
            continue
        if not (2000 <= y <= 2025):
            drop["ausserhalb 2000-2025"] += 1
            continue
        if not (1 <= mo <= 12 and 1 <= da <= 31):
            drop["unvollstaendiges Datum"] += 1
            continue
        try:
            d = dt.date(y, mo, da)
        except ValueError:
            drop["ungueltiges Datum"] += 1
            continue
        iso = d.isoformat()
        if not (args.season_start <= iso[5:] <= args.season_end):
            drop["ausserhalb der Saison"] += 1
            continue
        t = t_of.get(iso)
        if t is None:
            drop["Tag nicht im Raster"] += 1
            continue
        try:
            lat, lon = float(r["LATITUDE"]), float(r["LONGITUDE"])
        except (TypeError, ValueError):
            drop["keine Koordinate"] += 1
            continue
        # floor, nicht int(): int() rundet gegen null, ein Punkt knapp
        # westlich/noerdlich des Gitters landete sonst in Spalte/Zeile 0.
        c = math.floor((lon - tr[2]) / tr[0])
        rr = math.floor((lat - tr[5]) / tr[4])
        if not (0 <= rr < H and 0 <= c < W):
            drop["ausserhalb des Gitters"] += 1
            continue
        if not valid[rr, c]:
            drop["ausserhalb der Gueltigkeitsmaske"] += 1
            continue
        try:
            size = float(r.get("SIZE_HA") or "nan")
        except ValueError:
            size = float("nan")
        if not (size >= args.min_size_ha):
            drop[f"kleiner als {args.min_size_ha} ha oder ohne Angabe"] += 1
            continue

        key = (t, rr * W + c)
        cand = {"t": t, "row": rr, "col": c, "date": iso, "year": y,
                "size_ha": size, "cause": (r.get("CAUSE") or "U").strip()[:1].upper(),
                "fire_id": r.get("NFDBFIREID") or r.get("FIRE_ID") or "",
                "agency": (r.get("SRC_AGENCY") or "").strip()}
        if key in recs:
            merged += 1
            if size > recs[key]["size_ha"]:
                recs[key] = cand
        else:
            recs[key] = cand

    ign = sorted(recs.values(), key=lambda x: (x["t"], x["row"], x["col"]))
    print(f"[nfdb] {len(ign)} Zuendungen nach Filterung "
          f"({merged} Doppelmeldungen zusammengefasst)")
    print(f"[nfdb] verworfen: {dict(drop)}")

    cause = Counter(x["cause"] for x in ign)
    print(f"[nfdb] Ursachen: Blitz {cause['N']}, Mensch {cause['H']}, "
          f"unbekannt {cause.get('U', 0) + cause.get('', 0)}")
    sizes = np.array([x["size_ha"] for x in ign])
    print(f"[nfdb] Flaeche: Median {np.median(sizes):.1f} ha, "
          f"p90 {np.percentile(sizes, 90):.0f}, max {sizes.max():.0f}")

    # ---------------------------------------------- Datumsqualitaet
    # Die Zeitaufloesung des Labels entscheidet, wie sinnvoll kurze
    # Horizonte sind. Waeren viele Daten nur jahresgenau oder auf den
    # Monatsersten gesetzt, waere ein 1-Tages-Horizont sinnlos. Das wird
    # hier gemessen und mitgeschrieben, nicht angenommen.
    in_range = [r for r in rows
                if (r.get("YEAR") or "").isdigit() and 2000 <= int(r["YEAR"]) <= 2025]
    def filled(r, k):
        return (r.get(k) or "").strip() not in ("", "0")
    n_md = sum(1 for r in in_range if filled(r, "MONTH") and filled(r, "DAY"))
    n_rep = sum(1 for r in in_range if (r.get("REP_DATE") or "").strip())
    n_atk = sum(1 for r in in_range if (r.get("ATTK_DATE") or "").strip())
    n_out = sum(1 for r in in_range if (r.get("OUT_DATE") or "").strip())

    agree = differ = 0
    for r in in_range:
        rp = (r.get("REP_DATE") or "").strip()[:10]
        if not rp:
            continue
        try:
            if rp == dt.date(int(r["YEAR"]), int(r["MONTH"]), int(r["DAY"])).isoformat():
                agree += 1
            else:
                differ += 1
        except (TypeError, ValueError):
            pass

    dom = Counter(int(r["DAY"]) for r in in_range if filled(r, "DAY"))
    tot_dom = sum(dom.values())
    share_first = dom[1] / max(1, tot_dom)
    expected = 1 / 30.4
    excess = (share_first - expected) * tot_dom

    date_quality = {
        "n_records_2000_2025": len(in_range),
        "with_month_and_day": n_md,
        "with_month_and_day_share": n_md / max(1, len(in_range)),
        "with_rep_date": n_rep,
        "with_attack_date": n_atk,
        "with_out_date": n_out,
        "ymd_equals_rep_date": agree,
        "ymd_differs_from_rep_date": differ,
        "day_of_month_share": {str(d): dom[d] / max(1, tot_dom)
                               for d in sorted(dom)},
        "share_on_first_of_month": share_first,
        "expected_uniform_share": expected,
        "estimated_placeholder_records": int(round(max(0.0, excess))),
        "caveat": (
            "REP_DATE ist das MELDEdatum, nicht der Zuendzeitpunkt. Ein "
            "Blitzbrand im unbewirtschafteten Norden kann Tage vorher "
            "entstanden sein. ATTK_DATE koennte das eingrenzen, ist aber nur "
            f"in {100*n_atk/max(1,len(in_range)):.1f} % der Saetze gefuellt. "
            "Die Zeitangabe ist damit systematisch eher zu spaet als zu "
            "frueh; das unterschaetzt die Vorhersagbarkeit bei kurzen "
            "Horizonten eher, als sie zu ueberschaetzen."
        ),
    }
    print(f"[nfdb] Datumsqualitaet: {100*n_md/max(1,len(in_range)):.1f} % mit Tag, "
          f"{agree} stimmen mit REP_DATE, {differ} weichen ab")
    print(f"[nfdb] Anteil am Monatsersten {100*share_first:.2f} % "
          f"(gleichverteilt waeren {100*expected:.2f} %) -> geschaetzt "
          f"{date_quality['estimated_placeholder_records']} Platzhalter")
    print(f"[nfdb] OUT_DATE gefuellt in {100*n_out/max(1,len(in_range)):.1f} %, "
          f"ATTK_DATE in {100*n_atk/max(1,len(in_range)):.1f} %")

    # ---------------------------------------------- Meldevollstaendigkeit
    reported = defaultdict(set)
    for r in in_range:
        a = (r.get("SRC_AGENCY") or "").strip()
        if a in AGENCY_TO_REGION:
            reported[a].add(int(r["YEAR"]))
    years = set(range(2000, 2026))
    completeness = {}
    never, partial = [], {}
    for a, reg in AGENCY_TO_REGION.items():
        missing = sorted(years - reported.get(a, set()))
        completeness[a] = {"region": reg, "years_reported": len(reported.get(a, set())),
                           "years_missing": missing}
        if len(missing) == len(years):
            never.append(reg)
        elif missing:
            partial[reg] = missing
    print(f"[nfdb] Behoerden ohne JEDE Meldung: {never or 'keine'}")
    for reg, ms in partial.items():
        print(f"[nfdb] {reg}: keine Meldungen in {len(ms)} Jahren ({ms[0]}..{ms[-1]})")

    # Diese Regionen aus der Maske nehmen und die Kosten beziffern
    reg_man = json.loads((exp / "data_manifests" / "d2_regions.json").read_text())
    code_of = reg_man["code_of"]
    unobs_codes = [code_of[r] for r in never if r in code_of]
    observed = valid & ~np.isin(region, unobs_codes) if unobs_codes else valid.copy()
    np.save(exp / "derived_data" / "masks" / "observed_nfdb.npy", observed)
    print(f"[nfdb] Maske ohne unbeobachtete Regionen: {int(observed.sum())} von "
          f"{int(valid.sum())} Pixeln ({100*observed.sum()/valid.sum():.1f} %)")
    n_lost = sum(1 for x in ign if not observed[x["row"], x["col"]])
    print(f"[nfdb] dabei verlorene Zuendungen: {n_lost}")

    ign = [x for x in ign if observed[x["row"], x["col"]]]
    print(f"[nfdb] verbleibende Zuendungen: {len(ign)}")

    # ---------------------------------------------- CSR wie fire_index
    flat = np.array([x["row"] * W + x["col"] for x in ign], np.int64)
    day = np.array([x["t"] for x in ign], np.int64)
    order = np.lexsort((day, flat))
    keys = (flat[order] * DAY_BITS + day[order]).astype(np.int64)
    eid = np.arange(len(ign), dtype=np.int32)             # ein Ereignis je Meldung
    ccode = np.array([CAUSE_CODE.get(ign[i]["cause"], 0) for i in order], np.int8)
    size_o = np.array([ign[i]["size_ha"] for i in order], np.float32)

    out_dir = exp / "derived_data" / "events"
    out = out_dir / "nfdb_index.npz"
    np.savez_compressed(out, grid=np.array([H, W], np.int32),
                        dates=np.array(dates), keys=keys, event_id=eid,
                        cause=ccode, size_ha=size_o,
                        cause_codes=np.array(list(CAUSE_CODE.items()), dtype=object))
    print(f"[nfdb] {out}  ({human_bytes(out.stat().st_size)})")

    csv_path = out_dir / f"ignitions_nfdb_min{args.min_size_ha:g}ha.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "date", "year", "row", "col", "lon", "lat",
                    "event_id", "cause", "size_ha", "agency"])
        # Die Ereignis-ID ist die Position in der sortierten Reihenfolge.
        # Einmal als Umkehrpermutation berechnen statt je Zeile zu suchen.
        for pos, i in enumerate(order.tolist()):
            x = ign[i]
            lon = tr[2] + tr[0] * (x["col"] + 0.5)
            lat = tr[5] + tr[4] * (x["row"] + 0.5)
            w.writerow([x["t"], x["date"], x["year"], x["row"], x["col"],
                        f"{lon:.5f}", f"{lat:.5f}", pos,
                        x["cause"], f"{x['size_ha']:.2f}", x["agency"]])
    print(f"[nfdb] {csv_path}")

    per_year = Counter(x["year"] for x in ign)
    per_region = Counter(int(region[x["row"], x["col"]]) for x in ign)
    write_manifest(
        cfg, "d12_nfdb_events", args.config,
        {
            "source": "Canadian National Fire Database (NRCan/CFS)",
            "file": args.nfdb,
            "min_size_ha": args.min_size_ha,
            "season": {"start": args.season_start, "end": args.season_end},
            "n_raw": len(rows), "n_ignitions": len(ign),
            "n_merged_duplicates": merged,
            "dropped": dict(drop),
            "cause_counts": dict(cause),
            "cause_codes": CAUSE_CODE,
            "size_ha": {"median": float(np.median(sizes)),
                        "p90": float(np.percentile(sizes, 90)),
                        "max": float(sizes.max())},
            "date_quality": date_quality,
            "agency_completeness": completeness,
            "regions_never_reporting": never,
            "regions_partially_reporting": partial,
            "observed_mask": str(exp / "derived_data" / "masks" / "observed_nfdb.npy"),
            "n_pixels_valid": int(valid.sum()),
            "n_pixels_observed": int(observed.sum()),
            "n_ignitions_lost_to_mask": n_lost,
            "unobserved_rationale": (
                "Eine Behoerde, die nie meldet, liefert keine Negative, sondern "
                "unbeobachtete Zeit. Solche Pixel als 'nicht gebrannt' zu zaehlen "
                "wuerde dem Modell eine Sicherheit beibringen, die nur auf einer "
                "Meldeluecke beruht."),
            "per_year": dict(sorted(per_year.items())),
            "per_region_code": dict(sorted(per_region.items())),
            "n_pixels": int(np.unique(flat).size),
            "outputs": {"index": str(out), "csv": str(csv_path)},
            "note": ("Jeder Eintrag ist EIN Brand - Ereignis-IDs sind gegeben, "
                     "nicht aus raumzeitlichen Komponenten abgeleitet."),
        })


if __name__ == "__main__":
    main()

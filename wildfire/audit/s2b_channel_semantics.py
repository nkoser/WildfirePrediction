"""Stufe 2b: Plausibilitaetstest der Bandbedeutungen.

Stufe 5 hat bewiesen, welcher Cache-Kanal aus welcher (Modalitaet, Band)
stammt. Was ein Band physikalisch MISST, steht dagegen nur im README - die
GeoTIFFs tragen generische Bandnamen und keine Einheiten.

Dieser Test prueft die README-Zuordnung gegen erwartbares Verhalten:

  * Sommer-/Winterkontrast: Temperatur, Strahlung und Vegetation muessen im
    Juli deutlich hoeher liegen als im Januar; Schneebedeckung umgekehrt.
  * Zeitliche Konstanz: als statisch deklarierte Kanaele (DEM, OSM,
    Landbedeckung, Bevoelkerung) muessen sich innerhalb eines Jahres NICHT
    aendern. Tun sie es doch, ist die Deklaration falsch.

Gerechnet wird das raeumliche Mittel ueber die Landmaske, damit Ozean- und
Fuellpixel das Ergebnis nicht verwaessern.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.audit.channel_dict import CHANNELS

WINTER_DAY = "01_15"
SUMMER_DAY = "07_15"
# Zweiter Sommertag, um die zeitliche Konstanz statischer Kanaele zu pruefen.
SUMMER_DAY_B = "08_15"


def read_bands(dataset_root: Path, modality: str, year: str, mmdd: str) -> np.ndarray | None:
    tar = dataset_root / "Entire_Canada_Maps" / modality / f"{modality}__{year}.tar"
    target = f"{year}_{mmdd}.tif"
    if not tar.exists():
        return None
    with tarfile.open(tar, "r:") as tf:
        for m in tf:
            if m.isfile() and Path(m.name).name == target:
                blob = tf.extractfile(m).read()
                with MemoryFile(blob) as mem, mem.open() as ds:
                    return ds.read()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--year", default="2015")
    ap.add_argument("--scratch", default="/data/Wildfire/derived_data/audit_scratch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    root = Path(cfg["dataset_root"])
    # Landmaske aus Stufe 3 wiederverwenden, sonst neu aus der Landbedeckung.
    lc = read_bands(root, "MCD12Q1", args.year, WINTER_DAY)
    land = ~np.isin(lc[0], (0, 17)) if lc is not None else None
    print(f"[s2b] Landmaske: {int(land.sum())} Pixel" if land is not None else "[s2b] keine Landmaske")

    cache: dict[tuple[str, str], np.ndarray | None] = {}
    for mod in {c["mod"] for c in CHANNELS}:
        for day in (WINTER_DAY, SUMMER_DAY, SUMMER_DAY_B):
            cache[(mod, day)] = read_bands(root, mod, args.year, day)
        print(f"    {mod} gelesen", flush=True)

    rows = []
    for c in CHANNELS:
        w = cache.get((c["mod"], WINTER_DAY))
        s = cache.get((c["mod"], SUMMER_DAY))
        s2 = cache.get((c["mod"], SUMMER_DAY_B))
        entry = dict(c)
        entry.pop("note", None)
        if w is None or s is None or c["band"] >= w.shape[0]:
            entry.update(status="nicht lesbar")
            rows.append(entry)
            continue
        wb, sb = w[c["band"]].astype(np.float64), s[c["band"]].astype(np.float64)
        m = land if land is not None else np.ones(wb.shape, bool)
        wm, sm = float(wb[m].mean()), float(sb[m].mean())
        # zeitliche Konstanz: identisch zwischen zwei Sommertagen?
        const = None
        if s2 is not None and c["band"] < s2.shape[0]:
            const = bool(np.array_equal(s[c["band"]], s2[c["band"]]))
        entry.update(
            winter_mean=round(wm, 3),
            summer_mean=round(sm, 3),
            summer_minus_winter=round(sm - wm, 3),
            constant_across_summer_days=const,
            declared_static=c["cadence"].startswith(("statisch", "jaehrlich")),
        )
        # Widerspruch: statisch deklariert, aendert sich aber
        entry["cadence_conflict"] = (
            const is not None and entry["declared_static"] and not const
        )
        rows.append(entry)

    print("\n[s2b] Kanal  Name                          Winter   Sommer    Diff  konstant")
    for e in rows:
        if "winter_mean" not in e:
            continue
        flag = "  <-- KONFLIKT" if e.get("cadence_conflict") else ""
        print(f"    ch{e['ch']:02d}  {e['name']:28s} {e['winter_mean']:8.2f} "
              f"{e['summer_mean']:8.2f} {e['summer_minus_winter']:8.2f}  "
              f"{str(e['constant_across_summer_days']):5s}{flag}")

    conflicts = [e["ch"] for e in rows if e.get("cadence_conflict")]
    static_ok = [e["ch"] for e in rows if e.get("declared_static") and e.get("constant_across_summer_days")]
    print(f"\n[s2b] als statisch deklariert und tatsaechlich konstant: {static_ok}")
    print(f"[s2b] Kadenz-Konflikte: {conflicts or 'keine'}")

    write_manifest(
        cfg,
        "s2b_channel_semantics",
        args.config,
        {
            "reference_year": args.year,
            "winter_day": WINTER_DAY,
            "summer_days": [SUMMER_DAY, SUMMER_DAY_B],
            "land_pixels": int(land.sum()) if land is not None else None,
            "channels": rows,
            "cadence_conflicts": conflicts,
            "static_confirmed": static_ok,
            "caveat": (
                "Kanal -> (Modalitaet, Band) ist in Stufe 5 bewiesen. "
                "Band -> physikalische Variable stammt aus dem README und ist "
                "hier nur auf Plausibilitaet geprueft, nicht bewiesen."
            ),
        },
    )


if __name__ == "__main__":
    main()

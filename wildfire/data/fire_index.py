"""Kompakter Index aller Feuertage je Pixel.

Der Survival-Datensatz braucht zwei verschiedene Auskuenfte:

  * "Wann ist an diesem Pixel die NAECHSTE Zuendung?" - daraus entsteht
    die Zeit bis zum Ereignis.
  * "Brennt dieses Pixel gerade oder in den letzten C Tagen?" - solche
    Landmarks muessen ausgeschlossen werden, weil "Zeit bis zur naechsten
    Zuendung" fuer ein brennendes Pixel nicht definiert ist.

Die zweite Frage braucht ALLE Feuertage, nicht nur die Zuendungen. Beides
wird hier einmal aus MCD14A1 extrahiert und duenn abgelegt: innerhalb der
Gueltigkeitsmaske sind es nur rund 290 000 Feuer-Pixeltage ueber 26 Jahre,
also wenige Megabyte statt eines dichten Rasters von 6,6 Milliarden
Eintraegen.

Ablage als CSR-artige Struktur: fuer jedes je betroffene Pixel eine
sortierte Liste seiner Tage, plus ein Offset-Array. Damit ist "naechster
Tag nach t" eine binaere Suche.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import tarfile
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest

DAILY_RE = re.compile(r"(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})\.tif{1,2}$", re.I)


def build_csr(t: np.ndarray, r: np.ndarray, c: np.ndarray, H: int, W: int):
    """(t, row, col) -> (pixel_ids, offsets, days) mit je Pixel sortierten Tagen."""
    flat = r.astype(np.int64) * W + c.astype(np.int64)
    order = np.lexsort((t, flat))
    flat_s, t_s = flat[order], t[order]
    uniq, starts = np.unique(flat_s, return_index=True)
    offsets = np.append(starts, flat_s.size).astype(np.int64)
    return uniq, offsets, t_s.astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    valid = np.load(exp / "derived_data" / "masks" / "valid.npy")
    H, W = valid.shape

    ts, rs, cs = [], [], []
    dates: list[str] = []
    mod_dir = Path(cfg["dataset_root"]) / "Entire_Canada_Maps" / "MCD14A1"
    t = 0
    for tar in sorted(mod_dir.glob("*.tar")):
        with tarfile.open(tar, "r:") as tf:
            members = sorted(
                (m for m in tf if m.isfile() and m.name.lower().endswith((".tif", ".tiff"))),
                key=lambda m: m.name,
            )
            for m in members:
                hit = DAILY_RE.search(Path(m.name).name)
                if not hit:
                    continue
                blob = tf.extractfile(m).read()
                with MemoryFile(blob) as mem, mem.open() as ds:
                    fire = (ds.read(1) == 1) & valid
                rr, cc = np.nonzero(fire)
                if rr.size:
                    ts.append(np.full(rr.size, t, np.int32))
                    rs.append(rr.astype(np.int16))
                    cs.append(cc.astype(np.int16))
                dates.append(f"{hit['y']}-{hit['m']}-{hit['d']}")
                t += 1
        print(f"    {tar.name} -> t={t}", flush=True)

    ft = np.concatenate(ts); fr = np.concatenate(rs); fc = np.concatenate(cs)
    print(f"[fire_index] {ft.size} Feuer-Pixeltage in der Maske, {t} Tage")

    fire_px, fire_off, fire_days = build_csr(ft, fr, fc, H, W)
    print(f"[fire_index] {fire_px.size} je betroffene Pixel")

    # Zuendungen aus der Ereignisableitung
    ign_path = exp / "derived_data" / "events" / "ignitions_c7_r1.csv"
    it, ir, ic, iev = [], [], [], []
    with ign_path.open() as fh:
        for rec in csv.DictReader(fh):
            it.append(int(rec["t"])); ir.append(int(rec["row"]))
            ic.append(int(rec["col"])); iev.append(int(rec["event_id"]))
    it = np.array(it, np.int32); ir = np.array(ir, np.int16)
    ic = np.array(ic, np.int16); iev = np.array(iev, np.int32)
    ign_px, ign_off, ign_days = build_csr(it, ir, ic, H, W)
    # Ereignis-ID in derselben Sortierung wie ign_days
    flat = ir.astype(np.int64) * W + ic.astype(np.int64)
    order = np.lexsort((it, flat))
    ign_events = iev[order].astype(np.int32)
    print(f"[fire_index] {it.size} Zuendungen auf {ign_px.size} Pixeln")

    out = exp / "derived_data" / "events" / "fire_index.npz"
    np.savez_compressed(
        out,
        grid=np.array([H, W], np.int32),
        dates=np.array(dates),
        fire_pixels=fire_px, fire_offsets=fire_off, fire_days=fire_days,
        ign_pixels=ign_px, ign_offsets=ign_off, ign_days=ign_days,
        ign_events=ign_events,
    )
    print(f"[fire_index] {out}  ({human_bytes(out.stat().st_size)})")

    # ---- Kontrolle: Rueckrechnung auf die Zuendungszahl je Jahr
    year_of = np.array([int(d[:4]) for d in dates])
    per_year = {int(y): int((year_of[ign_days] == y).sum()) for y in np.unique(year_of)}

    write_manifest(
        cfg, "d6_fire_index", args.config,
        {
            "n_days": t,
            "date_min": dates[0], "date_max": dates[-1],
            "n_fire_pixel_days": int(ft.size),
            "n_pixels_ever_fire": int(fire_px.size),
            "n_ignitions": int(it.size),
            "n_pixels_ever_ignition": int(ign_px.size),
            "ignitions_per_year": per_year,
            "file": str(out),
            "file_bytes": int(out.stat().st_size),
            "layout": "CSR: pixels (flat r*W+c), offsets, days je Pixel aufsteigend",
        },
    )


if __name__ == "__main__":
    main()

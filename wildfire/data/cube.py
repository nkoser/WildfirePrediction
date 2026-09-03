"""Baut den dichten Datenwuerfel aus den tar-Archiven.

Layout: je Jahr eine memmap der Form ``(T, H, W, C_dyn)`` uint8, dazu ein
kleines statisches Array ``(H, W, C_stat)``.

**Warum kanalletzt?** Der Zugriffsmuster beim Training ist ein Patch
(P x P) ueber L Tage und ALLE Kanaele. Bei ``(T, C, H, W)`` zerfaellt ein
Patch in C x P getrennte Speicherbereiche von je P Byte; bei
``(T, H, W, C)`` sind es nur P zusammenhaengende Bloecke von je P*C Byte.
Das ist der Unterschied zwischen ~1400 und ~33 Seitenzugriffen pro Kanalsatz.
Die Umordnung nach ``(C, T, H, W)`` fuer PyTorch passiert erst im Collate.

**Warum modalitaetsweise statt tageweise?** Die Archive sind unkomprimierte
tars ohne Index. Ein einzelnes Mitglied zu finden kostet einen Scan ueber
alle Header. Tageweise waeren das 13 x 183 Scans je Jahr; modalitaetsweise
wird jedes Archiv genau EINMAL sequentiell gelesen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from wildfire.audit.channel_dict import CHANNELS
from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest

DAILY_RE = re.compile(r"(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})\.tif{1,2}$", re.I)

# Statisch bzw. jaehrlich - in Stufe 2b als zeitlich konstant nachgewiesen.
STATIC_MODALITIES = ("DEMs", "OSMs", "Worldpop", "MCD12Q1")


def channel_layout() -> tuple[list[dict], list[dict]]:
    dyn = [c for c in CHANNELS if c["mod"] not in STATIC_MODALITIES]
    stat = [c for c in CHANNELS if c["mod"] in STATIC_MODALITIES]
    for i, c in enumerate(dyn):
        c["cube_index"] = i
    for i, c in enumerate(stat):
        c["cube_index"] = i
    return dyn, stat


def modality_slices(chans: list[dict]) -> dict[str, tuple[int, int, list[int]]]:
    """Modalitaet -> (Startindex im Wuerfel, Anzahl, Rohbandnummern)."""
    out: dict[str, tuple[int, int, list[int]]] = {}
    for c in chans:
        mod = c["mod"]
        if mod not in out:
            out[mod] = (c["cube_index"], 0, [])
        start, n, bands = out[mod]
        out[mod] = (start, n + 1, bands + [c["band"]])
    return out


def season_days(year: int, start_md: str, end_md: str) -> list[dt.date]:
    a = dt.date.fromisoformat(f"{year}-{start_md}")
    b = dt.date.fromisoformat(f"{year}-{end_md}")
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def fill_modality(job: dict) -> dict:
    """Liest EIN Archiv sequentiell und schreibt seine Kanaele in die memmap."""
    mod = job["modality"]
    year = job["year"]
    path = Path(job["dataset_root"]) / "Entire_Canada_Maps" / mod / f"{mod}__{year}.tar"
    start, n_ch, bands = job["slice"]
    wanted = {d.isoformat(): i for i, d in enumerate(job["days"])}
    written = 0
    if job["static"]:
        arr = np.lib.format.open_memmap(job["target"], mode="r+")
    else:
        arr = np.memmap(job["target"], dtype=np.uint8, mode="r+", shape=tuple(job["shape"]))
    with tarfile.open(path, "r:") as tf:
        for m in tf:
            if not m.isfile():
                continue
            hit = DAILY_RE.search(Path(m.name).name)
            if not hit:
                continue
            key = f"{hit['y']}-{hit['m']}-{hit['d']}"
            if job["static"]:
                if written:            # statisch: ein einziger Tag genuegt
                    break
                if key not in wanted:
                    continue
            elif key not in wanted:
                continue
            blob = tf.extractfile(m).read()
            with MemoryFile(blob) as mem, mem.open() as ds:
                raw = ds.read()
            sub = raw[bands]                      # (n_ch, H, W)
            block = np.ascontiguousarray(sub.transpose(1, 2, 0))
            if job["static"]:
                arr[:, :, start:start + n_ch] = block
            else:
                arr[wanted[key], :, :, start:start + n_ch] = block
            written += 1
    arr.flush() if hasattr(arr, "flush") else None
    del arr
    return {"modality": mod, "year": year, "days_written": written}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--season-start", default="04-01")
    ap.add_argument("--season-end", default="09-30")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    out_dir = Path(args.out) if args.out else exp / "derived_data" / "cube"
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    H, W = man["grid"]["height"], man["grid"]["width"]

    dyn, stat = channel_layout()
    dyn_sl, stat_sl = modality_slices(dyn), modality_slices(stat)
    print(f"[cube] {len(dyn)} dynamische, {len(stat)} statische Kanaele, Gitter {H}x{W}")
    print(f"[cube] dynamisch: {[(m, s[0], s[1]) for m, s in dyn_sl.items()]}")
    print(f"[cube] statisch : {[(m, s[0], s[1]) for m, s in stat_sl.items()]}")

    jobs, per_year = [], {}
    for year in args.years:
        days = season_days(year, args.season_start, args.season_end)
        T = len(days)
        dyn_path = out_dir / f"dyn_{year}.mmap"
        stat_path = out_dir / f"stat_{year}.npy"
        shape = (T, H, W, len(dyn))
        if not dyn_path.exists() or dyn_path.stat().st_size != int(np.prod(shape)):
            np.memmap(dyn_path, dtype=np.uint8, mode="w+", shape=shape).flush()
        if not stat_path.exists():
            np.lib.format.open_memmap(stat_path, mode="w+", dtype=np.uint8,
                                      shape=(H, W, len(stat))).flush()
        per_year[year] = {
            "days": [d.isoformat() for d in days], "T": T,
            "dyn_file": str(dyn_path), "stat_file": str(stat_path),
            "dyn_shape": list(shape), "dyn_bytes": int(np.prod(shape)),
        }
        for mod, sl in dyn_sl.items():
            jobs.append(dict(modality=mod, year=year, dataset_root=cfg["dataset_root"],
                             slice=sl, days=days, target=str(dyn_path),
                             shape=list(shape), static=False))
        for mod, sl in stat_sl.items():
            jobs.append(dict(modality=mod, year=year, dataset_root=cfg["dataset_root"],
                             slice=sl, days=days[:1], target=str(stat_path),
                             shape=[H, W, len(stat)], static=True))

    total = sum(v["dyn_bytes"] for v in per_year.values())
    print(f"[cube] {len(args.years)} Jahre, {len(jobs)} Archive, Zielgroesse {human_bytes(total)}")

    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(fill_modality, jobs):
            done.append(res)
            print(f"    {res['modality']:10s} {res['year']}  {res['days_written']:3d} Tage", flush=True)

    # ---- Manifest ueber ALLE vorhandenen Jahre, nicht nur die eben gebauten.
    # Jeder Lauf baut typischerweise ein Jahr; ein Manifest, das nur dieses
    # Jahr kennt, macht den Wuerfel fuer den Reader unsichtbar. Die Tagesliste
    # ist aus Jahr und Saison eindeutig ableitbar, also wird das Verzeichnis
    # gescannt statt frueherer Manifeste vertraut.
    for f in sorted(out_dir.glob("dyn_*.mmap")):
        y = int(f.stem.split("_")[1])
        if y in per_year:
            continue
        d = season_days(y, args.season_start, args.season_end)
        sh = (len(d), H, W, len(dyn))
        if f.stat().st_size != int(np.prod(sh)):
            print(f"[cube] WARNUNG {f.name}: {f.stat().st_size} Byte, erwartet "
                  f"{int(np.prod(sh))} - unvollstaendig, wird ausgelassen")
            continue
        per_year[y] = {
            "days": [x.isoformat() for x in d], "T": len(d),
            "dyn_file": str(f), "stat_file": str(out_dir / f"stat_{y}.npy"),
            "dyn_shape": list(sh), "dyn_bytes": int(np.prod(sh)),
        }
    per_year = {k: per_year[k] for k in sorted(per_year)}
    total = sum(v["dyn_bytes"] for v in per_year.values())
    print(f"[cube] Manifest umfasst {len(per_year)} Jahre: {sorted(per_year)}")

    # ---- Stichprobenkontrolle: ist der Wuerfel wirklich gefuellt?
    checks = {}
    for year, info in ((y, per_year[y]) for y in args.years if y in per_year):
        arr = np.memmap(info["dyn_file"], dtype=np.uint8, mode="r",
                        shape=tuple(info["dyn_shape"]))
        mid = info["T"] // 2
        sl = np.asarray(arr[mid])
        per_ch_zero = (sl == 0).mean(axis=(0, 1))
        checks[str(year)] = {
            "probe_day": info["days"][mid],
            "all_zero_channels": [dyn[i]["name"] for i in np.nonzero(per_ch_zero == 1.0)[0]],
            "max_zero_fraction": float(per_ch_zero.max()),
            "mean_value": float(sl.mean()),
        }
        print(f"[cube] {year} Kontrolle {info['days'][mid]}: Mittelwert {sl.mean():.1f}, "
              f"komplett leere Kanaele: {checks[str(year)]['all_zero_channels'] or 'keine'}")
        del arr

    write_manifest(
        cfg,
        "d4_cube",
        args.config,
        {
            "out_dir": str(out_dir),
            "years": args.years,
            "season": {"start": args.season_start, "end": args.season_end},
            "grid": {"height": H, "width": W},
            "layout": "(T, H, W, C) uint8, kanalletzt",
            "n_dynamic_channels": len(dyn),
            "n_static_channels": len(stat),
            "dynamic_channels": [{"cube_index": c["cube_index"], "name": c["name"],
                                  "modality": c["mod"], "band": c["band"]} for c in dyn],
            "static_channels": [{"cube_index": c["cube_index"], "name": c["name"],
                                 "modality": c["mod"], "band": c["band"]} for c in stat],
            "per_year": per_year,
            "total_bytes": total,
            "total_human": human_bytes(total),
            "fill_reports": done,
            "sanity_checks": checks,
        },
    )


if __name__ == "__main__":
    main()

"""Stufe 3: vollstaendige Auswertung der Feuer-Labels (MCD14A1).

Die Label-Modalitaet umfasst ueber alle 26 Jahre nur ca. 81 MB, daher
wird sie hier - anders als die Treiber - LUECKENLOS ausgewertet, Tag fuer
Tag. Nichts wird gesampled.

Beantwortet die Kernfragen des Audits:
  * Welche Werte hat die Maske und was bedeuten sie (MODIS-FireMask-Skala)?
  * Wie viele Feuerpixel pro Tag / Jahr / Pixel?
  * Lassen sich ZUENDUNGEN von FORTSETZUNG unterscheiden?
  * Lassen sich Pixel zu gemeinsamen Brandereignissen gruppieren
    (raumzeitliche Zusammenhangskomponenten)?
  * Wie hoch ist die Prävalenz (Klassenverteilung) pro Pixel und Tag?
  * Wie viele Pixel brennen wiederholt?

MODIS-FireMask-Kodierung (MOD14A1/MYD14A1 User Guide):
  0 nicht prozessiert (fehlende Eingangsdaten)   1 nicht prozessiert (sonstiges)
  2 unbenutzt                                     3 Wasser (Nicht-Feuer)
  4 Wolke                                         5 Nicht-Feuer, Land
  6 unbekannt                                     7 Feuer, geringe Konfidenz
  8 Feuer, nominale Konfidenz                     9 Feuer, hohe Konfidenz
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from scipy import ndimage

from wildfire.audit.common import assert_readonly, load_config, write_manifest

DAILY_RE = re.compile(r"(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})\.tif{1,2}$", re.I)

# Die ausgelieferte Maske ist NICHT die rohe MODIS-FireMask-Skala (0..9),
# sondern bereits auf binaer 0/1 reduziert. Die Kodierung wird zur Laufzeit
# aus dem Wertehistogramm bestimmt und nicht angenommen.
MODIS_FIREMASK_MEANING = {
    0: "nicht prozessiert (fehlende Eingangsdaten)",
    1: "nicht prozessiert (sonstiges)",
    2: "unbenutzt",
    3: "Wasser (Nicht-Feuer)",
    4: "Wolke",
    5: "Nicht-Feuer, Land",
    6: "unbekannt",
    7: "Feuer, geringe Konfidenz",
    8: "Feuer, nominale Konfidenz",
    9: "Feuer, hohe Konfidenz",
}
BINARY_MEANING = {0: "kein aktives Feuer detektiert", 1: "aktives Feuer detektiert"}

# IGBP-Landbedeckung (MCD12Q1 LC_Type1): 17 = Wasserflaechen,
# 0 = nicht klassifiziert / Fuellwert. Beides zaehlt nicht als Landflaeche.
LANDCOVER_NON_LAND = (0, 17)

# Zeitfenster, innerhalb dessen ein bereits brennendes Pixel als
# "Fortsetzung" und nicht als neue Zuendung gilt.
IGNITION_COOLDOWNS = (1, 3, 7, 14, 30)
# Raeumliche Nachbarschaft und zeitliche Bruecke fuer die Event-Bildung.
EVENT_TIME_BRIDGE_DAYS = 3


def load_land_mask(cfg: dict, year: str = "2023") -> tuple[np.ndarray, dict]:
    """Landmaske aus der Landbedeckung (MCD12Q1) des Referenzjahres.

    Die Feuermaske selbst ist binaer und enthaelt daher KEINE Information
    darueber, welche Pixel ueberhaupt beobachtbares Land sind. Ohne eine
    externe Maske waere jede Praevalenzangabe durch Ozean- und
    Fuellpixel verwaessert.
    """
    tar = Path(cfg["dataset_root"]) / "Entire_Canada_Maps" / "MCD12Q1" / f"MCD12Q1__{year}.tar"
    with tarfile.open(tar, "r:") as tf:
        member = sorted((m for m in tf.getmembers() if m.isfile()), key=lambda m: m.name)[0]
        blob = tf.extractfile(member).read()
    with MemoryFile(blob) as mem, mem.open() as ds:
        lc = ds.read(1)
    vals, counts = np.unique(lc, return_counts=True)
    land = ~np.isin(lc, LANDCOVER_NON_LAND)
    meta = {
        "source": f"MCD12Q1__{year}.tar/{member.name}",
        "class_histogram": {str(int(v)): int(c) for v, c in zip(vals, counts)},
        "non_land_classes": list(LANDCOVER_NON_LAND),
        "n_land_pixels": int(land.sum()),
        "land_fraction": float(land.mean()),
    }
    return land, meta


def iter_days(tar_path: Path):
    """Liefert (datum, array, profil) fuer jeden Tag eines Jahresarchivs."""
    with tarfile.open(tar_path, "r:") as tf:
        members = sorted(
            (m for m in tf if m.isfile() and m.name.lower().endswith((".tif", ".tiff"))),
            key=lambda m: m.name,
        )
        for m in members:
            hit = DAILY_RE.search(Path(m.name).name)
            if not hit:
                continue
            date = dt.date(int(hit["y"]), int(hit["m"]), int(hit["d"]))
            fh = tf.extractfile(m)
            if fh is None:
                continue
            blob = fh.read()
            with MemoryFile(blob) as mem, mem.open() as ds:
                arr = ds.read(1)
                profile = {
                    "width": ds.width,
                    "height": ds.height,
                    "crs": str(ds.crs) if ds.crs else None,
                    "transform": list(ds.transform)[:6],
                    "dtype": str(arr.dtype),
                    "nodata": None if ds.nodatavals[0] is None else float(ds.nodatavals[0]),
                    "count": ds.count,
                    "compression": str(ds.compression) if ds.compression else None,
                }
            yield date, arr, profile, m.size, len(blob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scratch", default="/data/Wildfire/derived_data/audit_scratch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    mod = cfg["label_modality"]
    tars = sorted((Path(cfg["dataset_root"]) / "Entire_Canada_Maps" / mod).glob("*.tar"))
    print(f"[s3] {len(tars)} Jahresarchive der Label-Modalitaet {mod}")

    land_mask, land_meta = load_land_mask(cfg)
    print(f"[s3] Landmaske: {land_meta['n_land_pixels']} von {land_mask.size} Pixeln "
          f"({100*land_meta['land_fraction']:.1f} %), Klassen {land_meta['class_histogram']}")

    value_hist: Counter = Counter()
    per_day: list[dict] = []
    profiles: set = set()
    profile_example: dict | None = None
    grid_shape: tuple[int, int] | None = None

    # Akkumulatoren ueber das gesamte Gitter
    burn_count: np.ndarray | None = None   # wie oft war ein Pixel Feuer
    ever_valid: np.ndarray | None = None   # jemals gueltige Landbeobachtung
    last_fire_day: np.ndarray | None = None  # Tagesindex des letzten Feuers
    ignitions_by_cooldown = {c: 0 for c in IGNITION_COOLDOWNS}
    ignition_records: list[tuple[int, int, int]] = []  # (t_idx, row, col) fuer cooldown=7

    # Duenne Speicherung: Feuerkoordinaten je Tag
    fire_coords_by_day: dict[int, np.ndarray] = {}
    day_index: dict[dt.date, int] = {}
    all_dates: list[dt.date] = []

    t_idx = 0
    for tar in tars:
        year_fire = 0
        n_days = 0
        for date, arr, profile, msize, bsize in iter_days(tar):
            if grid_shape is None:
                grid_shape = arr.shape
                burn_count = np.zeros(arr.shape, dtype=np.int32)
                ever_valid = np.zeros(arr.shape, dtype=bool)
                last_fire_day = np.full(arr.shape, -10**6, dtype=np.int32)
                profile_example = profile
            if arr.shape != grid_shape:
                raise SystemExit(f"ABBRUCH: abweichendes Gitter {arr.shape} in {tar.name}/{date}")
            profiles.add((profile["width"], profile["height"], profile["crs"], tuple(profile["transform"])))

            vals, counts = np.unique(arr, return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                value_hist[int(v)] += c

            # Kodierung aus den Daten ableiten statt anzunehmen.
            if set(np.unique(arr).tolist()) <= {0, 1}:
                fire = arr == 1
            else:
                fire = np.isin(arr, (7, 8, 9))
            ever_valid |= fire

            rows, cols = np.nonzero(fire)
            n_fire = rows.size
            year_fire += n_fire
            fire_coords_by_day[t_idx] = np.stack([rows, cols], axis=1).astype(np.int32)

            if n_fire:
                gap = t_idx - last_fire_day[rows, cols]
                for c in IGNITION_COOLDOWNS:
                    ignitions_by_cooldown[c] += int((gap > c).sum())
                new7 = gap > 7
                for r, cc in zip(rows[new7].tolist(), cols[new7].tolist()):
                    ignition_records.append((t_idx, r, cc))
                burn_count[rows, cols] += 1
                last_fire_day[rows, cols] = t_idx

            per_day.append(
                {
                    "date": date.isoformat(),
                    "t_idx": t_idx,
                    "n_fire_px": int(n_fire),
                    "n_fire_px_on_land": int((fire & land_mask).sum()),
                    "n_fire_px_off_land": int((fire & ~land_mask).sum()),
                    "member_size_bytes": int(msize),
                }
            )
            day_index[date] = t_idx
            all_dates.append(date)
            t_idx += 1
            n_days += 1
        print(f"    {tar.name:24s} {n_days:3d} Tage  {year_fire:8d} Feuerpixel", flush=True)

    assert burn_count is not None and ever_valid is not None and grid_shape is not None
    n_days_total = t_idx

    # ---------------------------------------------------------------- Ereignisse
    # Raumzeitliche Zusammenhangskomponenten: pro Tag 8er-Nachbarschaft,
    # ueber Tage hinweg verbunden, wenn Pixel innerhalb von
    # EVENT_TIME_BRIDGE_DAYS erneut oder benachbart brennen.
    print("[s3] bilde raumzeitliche Brandereignisse ...", flush=True)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    struct = ndimage.generate_binary_structure(2, 2)
    prev_labels: list[tuple[int, np.ndarray, np.ndarray]] = []  # (t, labelmap_sparse_keys, ids)
    next_id = 0
    day_comp: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for t in range(n_days_total):
        coords = fire_coords_by_day.get(t)
        if coords is None or coords.size == 0:
            continue
        # kleines Bounding-Box-Bild statt Vollraster
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
        lab, n = ndimage.label(sub, structure=struct)
        comp_of_px = lab[coords[:, 0] - r0, coords[:, 1] - c0]
        ids = np.empty(n + 1, dtype=np.int64)
        ids[0] = -1
        for i in range(1, n + 1):
            parent[next_id] = next_id
            ids[i] = next_id
            next_id += 1
        px_ids = ids[comp_of_px]
        day_comp[t] = (coords, px_ids, ids[1:])

        # zeitliche Verbindung zu den letzten Tagen
        for pt, pcoords, ppx_ids in prev_labels:
            if t - pt > EVENT_TIME_BRIDGE_DAYS:
                continue
            key_prev = {(int(r), int(c)): int(i) for r, c, i in zip(pcoords[:, 0], pcoords[:, 1], ppx_ids)}
            for (r, c), i in zip(map(tuple, coords.tolist()), px_ids.tolist()):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        j = key_prev.get((r + dr, c + dc))
                        if j is not None:
                            union(i, j)
        prev_labels.append((t, coords, px_ids))
        prev_labels = [p for p in prev_labels if t - p[0] <= EVENT_TIME_BRIDGE_DAYS]

    event_of: dict[int, int] = {}
    for k in list(parent.keys()):
        event_of[k] = find(k)
    events: dict[int, dict] = defaultdict(
        lambda: {"n_px_days": 0, "t_min": 10**9, "t_max": -1, "rows": [], "cols": []}
    )
    for t, (coords, px_ids, _) in day_comp.items():
        for (r, c), i in zip(map(tuple, coords.tolist()), px_ids.tolist()):
            e = events[event_of[i]]
            e["n_px_days"] += 1
            e["t_min"] = min(e["t_min"], t)
            e["t_max"] = max(e["t_max"], t)
            if len(e["rows"]) < 4:
                e["rows"].append(r)
                e["cols"].append(c)

    ev_sizes = np.array([e["n_px_days"] for e in events.values()], dtype=np.int64)
    ev_dur = np.array([e["t_max"] - e["t_min"] + 1 for e in events.values()], dtype=np.int64)

    # ---------------------------------------------------------------- Statistik
    n_land_px = int(land_mask.sum())
    n_valid_px = n_land_px
    n_grid_px = int(grid_shape[0] * grid_shape[1])
    total_fire_px_days = int(burn_count.sum())
    burned_at_least_once = int((burn_count > 0).sum())
    repeat = {
        f">={k}": int((burn_count >= k).sum()) for k in (1, 2, 3, 5, 10, 20, 50)
    }
    burn_on_land = burn_count * land_mask

    per_year: dict[str, dict] = defaultdict(lambda: {"n_days": 0, "n_fire_px_days": 0})
    for d in per_day:
        y = d["date"][:4]
        per_year[y]["n_days"] += 1
        per_year[y]["n_fire_px_days"] += d["n_fire_px"]

    ign_dates = Counter()
    for t, _, _ in ignition_records:
        ign_dates[all_dates[t].isoformat()[:4]] += 1

    prevalence = {
        "n_pixel_days_total": n_days_total * n_grid_px,
        "n_pixel_days_land": n_days_total * n_valid_px,
        "n_fire_pixel_days": total_fire_px_days,
        "positive_rate_over_grid": total_fire_px_days / max(1, n_days_total * n_grid_px),
        "positive_rate_over_land": total_fire_px_days / max(1, n_days_total * n_valid_px),
        "n_ignition_pixel_days_cooldown7": len(ignition_records),
        "ignition_rate_over_land": len(ignition_records) / max(1, n_days_total * n_valid_px),
    }

    print(f"[s3] Gitter {grid_shape}, {n_days_total} Tage, {n_land_px} Landpixel")
    print(f"[s3] Werteverteilung der Maske: {dict(sorted(value_hist.items()))}")
    print(f"[s3] Feuer-Pixeltage gesamt: {total_fire_px_days}")
    print(f"[s3] Pixel, die je gebrannt haben: {burned_at_least_once} ({100*burned_at_least_once/max(1,n_valid_px):.3f} % der Landpixel)")
    print(f"[s3] Wiederholungen: {repeat}")
    print(f"[s3] Zuendungen je Cooldown: {ignitions_by_cooldown}")
    print(f"[s3] Ereignisse: {len(events)}  Median-Groesse {np.median(ev_sizes) if ev_sizes.size else 0} Pixeltage")
    print(f"[s3] Prävalenz (Land): {prevalence['positive_rate_over_land']:.3e}")

    # Karten fuer spaetere Abbildungen (ausserhalb der Rohdaten)
    np.save(scratch / "burn_count.npy", burn_count)
    np.save(scratch / "ever_valid.npy", ever_valid)

    write_manifest(
        cfg,
        "s3_fire_labels",
        args.config,
        {
            "modality": mod,
            "grid_shape": list(grid_shape),
            "n_distinct_profiles": len(profiles),
            "profile_example": profile_example,
            "n_days": n_days_total,
            "date_min": all_dates[0].isoformat() if all_dates else None,
            "date_max": all_dates[-1].isoformat() if all_dates else None,
            "value_histogram": {str(k): int(v) for k, v in sorted(value_hist.items())},
            "value_meaning_modis": {str(k): v for k, v in MODIS_FIREMASK_MEANING.items()},
            "fire_value_detection": "auto: {0,1} -> fire==1, sonst MODIS 7/8/9",
            "land_mask": land_meta,
            "n_land_pixels": n_land_px,
            "n_pixels_ever_fire": int(ever_valid.sum()),
            "binary_meaning": {str(k): v for k, v in BINARY_MEANING.items()},
            "n_grid_pixels": n_grid_px,
            "total_fire_pixel_days": total_fire_px_days,
            "pixels_burned_at_least_once": burned_at_least_once,
            "repeat_burn_pixels": repeat,
            "burn_count_max": int(burn_count.max()),
            "ignitions_by_cooldown_days": {str(k): v for k, v in ignitions_by_cooldown.items()},
            "ignitions_per_year_cooldown7": dict(sorted(ign_dates.items())),
            "prevalence": prevalence,
            "events": {
                "definition": (
                    "8er-Nachbarschaft je Tag, verbunden ueber bis zu "
                    f"{EVENT_TIME_BRIDGE_DAYS} Tage Zeitbruecke"
                ),
                "n_events": len(events),
                "size_px_days": {
                    "min": int(ev_sizes.min()) if ev_sizes.size else 0,
                    "p50": float(np.median(ev_sizes)) if ev_sizes.size else 0,
                    "p90": float(np.percentile(ev_sizes, 90)) if ev_sizes.size else 0,
                    "p99": float(np.percentile(ev_sizes, 99)) if ev_sizes.size else 0,
                    "max": int(ev_sizes.max()) if ev_sizes.size else 0,
                    "mean": float(ev_sizes.mean()) if ev_sizes.size else 0,
                },
                "duration_days": {
                    "p50": float(np.median(ev_dur)) if ev_dur.size else 0,
                    "p90": float(np.percentile(ev_dur, 90)) if ev_dur.size else 0,
                    "max": int(ev_dur.max()) if ev_dur.size else 0,
                },
                "n_single_pixel_day_events": int((ev_sizes == 1).sum()) if ev_sizes.size else 0,
            },
            "per_year": {k: dict(v) for k, v in sorted(per_year.items())},
            "per_day": per_day,
        },
    )


if __name__ == "__main__":
    main()

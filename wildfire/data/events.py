"""Leitet Proxy-Zuendungen und Brandereignisse aus der Feuermaske ab.

Es gibt keine echten Zuendpunkte. Ableitbar ist nur, wann ein Pixel zum
ersten Mal nach einer feuerfreien Phase brennt. Wie man diese Phase
abgrenzt, ist eine methodische Entscheidung mit Folgen - deshalb werden
hier MEHRERE Definitionen berechnet und gegenuebergestellt, statt eine
stillschweigend zu setzen:

  R=0  reiner Pixel-Cooldown. Ein Pixel zaehlt als Zuendung, wenn es
       selbst C Tage nicht gebrannt hat. Zaehlt AUSBREITUNG mit: schiebt
       sich ein Feuer ins Nachbarpixel, gilt das dort als neue Zuendung.
  R=1  zusaetzlich raeumliche Sperre ueber die 8er-Nachbarschaft (ca. 5 km
       N-S, 2-3 km O-W). Nur wenn auch ringsum nichts brannte, gilt es als
       neu entstanden.
  R=2  wie R=1, aber 5x5-Umgebung - konservativer, verwirft mehr echte
       Zuendungen dicht neben Altbraenden.
  EVENT erste(r) Pixeltag(e) jeder raumzeitlichen Zusammenhangskomponente.
       Genau eine Zuendung je Brandereignis.

Zusaetzlich werden die Ereignisse selbst gebildet (Union-Find ueber
tagesweise 8er-Nachbarschaft mit Zeitbruecke), denn ereignisgruppierte
Splits und die ereigniszentrierte Sentinel-Beschaffung brauchen sie.

Alles wird auf die Gueltigkeitsmaske aus ``wildfire.data.masks``
eingeschraenkt.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile
from scipy import ndimage

from wildfire.audit.common import assert_readonly, load_config, write_manifest

DAILY_RE = re.compile(r"(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})\.tif{1,2}$", re.I)
COOLDOWN_DAYS = 7
RADII = (0, 1, 2)
EVENT_TIME_BRIDGE_DAYS = 3


def iter_fire_days(dataset_root: Path):
    """(Datum, boolsches Feuerraster) fuer alle Tage, chronologisch."""
    mod_dir = dataset_root / "Entire_Canada_Maps" / "MCD14A1"
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
                date = dt.date(int(hit["y"]), int(hit["m"]), int(hit["d"]))
                blob = tf.extractfile(m).read()
                with MemoryFile(blob) as mem, mem.open() as ds:
                    yield date, ds.read(1) == 1


class UnionFind:
    def __init__(self) -> None:
        self.p: dict[int, int] = {}

    def add(self, x: int) -> None:
        self.p.setdefault(x, x)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cooldown", type=int, default=COOLDOWN_DAYS)
    ap.add_argument("--default-radius", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    mask_dir = exp / "derived_data" / "masks"
    valid = np.load(mask_dir / "valid.npy")
    H, W = valid.shape
    C = args.cooldown
    print(f"[events] Gueltige Pixel {int(valid.sum())}, Cooldown {C} Tage, Radien {RADII}")

    structs = {r: np.ones((2 * r + 1, 2 * r + 1), bool) for r in RADII if r > 0}
    ring: list[np.ndarray] = []           # letzte C Tagesmasken
    counts = {r: 0 for r in RADII}
    ign_records: dict[int, list[tuple]] = {r: [] for r in RADII}
    per_day: list[dict] = []

    uf = UnionFind()
    next_comp = 0
    recent_comps: list[tuple[int, np.ndarray, np.ndarray]] = []
    comp_px: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    dates: list[dt.date] = []
    struct8 = ndimage.generate_binary_structure(2, 2)

    for t, (date, fire_raw) in enumerate(iter_fire_days(Path(cfg["dataset_root"]))):
        fire = fire_raw & valid
        dates.append(date)

        # ---------------------------------------------------- Zuendungen
        if ring:
            recent = np.logical_or.reduce(ring)
        else:
            recent = np.zeros_like(fire)
        day_counts = {}
        for r in RADII:
            blocked = recent if r == 0 else ndimage.binary_dilation(recent, structs[r])
            ign = fire & ~blocked
            counts[r] += int(ign.sum())
            day_counts[r] = int(ign.sum())
            if r == args.default_radius:
                rr, cc = np.nonzero(ign)
                for a, b in zip(rr.tolist(), cc.tolist()):
                    ign_records[r].append((t, a, b))
            else:
                rr, cc = np.nonzero(ign)
                for a, b in zip(rr.tolist(), cc.tolist()):
                    ign_records[r].append((t, a, b))

        ring.append(fire)
        if len(ring) > C:
            ring.pop(0)

        # ---------------------------------------------------- Ereignisse
        coords = np.stack(np.nonzero(fire), axis=1).astype(np.int32)
        if coords.size:
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
            sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
            lab, n = ndimage.label(sub, structure=struct8)
            comp_of = lab[coords[:, 0] - r0, coords[:, 1] - c0]
            ids = np.empty(n + 1, np.int64)
            ids[0] = -1
            for i in range(1, n + 1):
                uf.add(next_comp)
                ids[i] = next_comp
                next_comp += 1
            px_ids = ids[comp_of]
            for prev_t, pc, pid in recent_comps:
                if t - prev_t > EVENT_TIME_BRIDGE_DAYS:
                    continue
                lut = {(int(a), int(b)): int(i) for a, b, i in zip(pc[:, 0], pc[:, 1], pid)}
                for (a, b), i in zip(map(tuple, coords.tolist()), px_ids.tolist()):
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            j = lut.get((a + da, b + db))
                            if j is not None:
                                uf.union(i, j)
            recent_comps.append((t, coords, px_ids))
            recent_comps = [x for x in recent_comps if t - x[0] <= EVENT_TIME_BRIDGE_DAYS]
            for (a, b), i in zip(map(tuple, coords.tolist()), px_ids.tolist()):
                comp_px[i].append((t, a, b))

        per_day.append({"date": date.isoformat(), "t": t,
                        "n_fire_px": int(fire.sum()), **{f"ign_r{r}": day_counts[r] for r in RADII}})
        if t % 1000 == 0:
            print(f"    t={t} {date} Feuer={int(fire.sum())} "
                  f"Zuendungen r0/r1/r2={day_counts[0]}/{day_counts[1]}/{day_counts[2]}", flush=True)

    n_days = len(dates)
    print(f"[events] {n_days} Tage verarbeitet")

    # ---------------------------------------------------- Ereignisse verdichten
    events: dict[int, dict] = defaultdict(
        lambda: {"n_px_days": 0, "t_min": 10**9, "t_max": -1,
                 "rows": [], "cols": [], "first_px": []}
    )
    for comp, px in comp_px.items():
        e = events[uf.find(comp)]
        for (t, a, b) in px:
            e["n_px_days"] += 1
            if t < e["t_min"]:
                e["t_min"] = t
                e["first_px"] = [(a, b)]
            elif t == e["t_min"]:
                e["first_px"].append((a, b))
            e["t_max"] = max(e["t_max"], t)
            e["rows"].append(a)
            e["cols"].append(b)

    print(f"[events] {len(events)} Brandereignisse")

    # Pixel -> Ereignis-ID, fuer die ereignisgruppierten Splits
    px_event: dict[tuple[int, int, int], int] = {}
    for eid, comp in enumerate(sorted(events)):
        for c_ in [comp]:
            pass
    eid_of_root = {root: i for i, root in enumerate(sorted(events))}
    for comp, px in comp_px.items():
        eid = eid_of_root[uf.find(comp)]
        for (t, a, b) in px:
            px_event[(t, a, b)] = eid

    # ---------------------------------------------------- Ausgabe
    out_dir = exp / "derived_data" / "events"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = None
    import json as _json
    man = _json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    tr = man["grid"]["transform"]

    def lonlat(r: int, c: int) -> tuple[float, float]:
        return tr[2] + tr[0] * (c + 0.5), tr[5] + tr[4] * (r + 0.5)

    # Zuendungstabelle fuer den Standardradius
    R = args.default_radius
    path = out_dir / f"ignitions_c{C}_r{R}.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "date", "year", "doy", "row", "col", "lon", "lat", "event_id"])
        for (t, a, b) in ign_records[R]:
            d = dates[t]
            lon, lat = lonlat(a, b)
            w.writerow([t, d.isoformat(), d.year, d.timetuple().tm_yday, a, b,
                        f"{lon:.5f}", f"{lat:.5f}", px_event.get((t, a, b), -1)])
    print(f"[events] {path}  ({len(ign_records[R])} Zuendungen)")

    ev_path = out_dir / "events.csv"
    with ev_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "t_start", "t_end", "date_start", "date_end", "year",
                    "duration_days", "n_px_days", "n_first_day_px",
                    "row_min", "row_max", "col_min", "col_max", "lon_first", "lat_first"])
        for root in sorted(events):
            e = events[root]
            eid = eid_of_root[root]
            a, b = e["first_px"][0]
            lon, lat = lonlat(a, b)
            w.writerow([eid, e["t_min"], e["t_max"], dates[e["t_min"]].isoformat(),
                        dates[e["t_max"]].isoformat(), dates[e["t_min"]].year,
                        e["t_max"] - e["t_min"] + 1, e["n_px_days"], len(e["first_px"]),
                        min(e["rows"]), max(e["rows"]), min(e["cols"]), max(e["cols"]),
                        f"{lon:.5f}", f"{lat:.5f}"])
    print(f"[events] {ev_path}")

    np.savez_compressed(
        out_dir / "per_day_counts.npz",
        dates=np.array([d.isoformat() for d in dates]),
        **{f"ign_r{r}": np.array([d[f"ign_r{r}"] for d in per_day], np.int32) for r in RADII},
        n_fire_px=np.array([d["n_fire_px"] for d in per_day], np.int32),
    )

    ev_sizes = np.array([e["n_px_days"] for e in events.values()])
    ev_dur = np.array([e["t_max"] - e["t_min"] + 1 for e in events.values()])
    per_year_ign = defaultdict(int)
    for (t, a, b) in ign_records[R]:
        per_year_ign[dates[t].year] += 1

    print("\n[events] Zuendungsdefinitionen im Vergleich:")
    for r in RADII:
        print(f"    Cooldown {C} d, Radius {r}: {counts[r]:7d} Zuendungen "
              f"({100*counts[r]/max(1,sum(d['n_fire_px'] for d in per_day)):5.1f} % aller Feuer-Pixeltage)")
    print(f"    Ereignisbasiert       : {len(events):7d} Zuendungen (eine je Ereignis)")

    write_manifest(
        cfg,
        "d1_events",
        args.config,
        {
            "cooldown_days": C,
            "radii": list(RADII),
            "default_radius": R,
            "event_time_bridge_days": EVENT_TIME_BRIDGE_DAYS,
            "n_days": n_days,
            "date_min": dates[0].isoformat(),
            "date_max": dates[-1].isoformat(),
            "n_valid_pixels": int(valid.sum()),
            "total_fire_pixel_days_in_mask": int(sum(d["n_fire_px"] for d in per_day)),
            "ignition_counts_by_radius": {str(r): counts[r] for r in RADII},
            "n_events": len(events),
            "event_size_px_days": {
                "p50": float(np.median(ev_sizes)), "p90": float(np.percentile(ev_sizes, 90)),
                "p99": float(np.percentile(ev_sizes, 99)), "max": int(ev_sizes.max()),
            },
            "event_duration_days": {
                "p50": float(np.median(ev_dur)), "p90": float(np.percentile(ev_dur, 90)),
                "max": int(ev_dur.max()),
            },
            "ignitions_per_year_default": dict(sorted(per_year_ign.items())),
            "outputs": {
                "ignitions_csv": str(path),
                "events_csv": str(ev_path),
                "per_day_counts": str(out_dir / "per_day_counts.npz"),
            },
        },
    )


if __name__ == "__main__":
    main()

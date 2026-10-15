"""Survival auf den Paper-Daten, Schritt 1: Landmarks mit Zeit bis zum Ereignis.

Vom Paper uebernommen: Stichproben und Splits (Cache-JSON), 13 x 13-Ausschnitt,
dieselben 55 Kanaele in derselben Reihenfolge und Kodierung, dieselbe Label-
definition (Feuer irgendwo im Ausschnitt). NICHT uebernommen: das Verschiebe-
Protokoll mit festen dt.

Statt dessen je Paper-Stichprobe (Zelle s, Zieltag t, Label y) mehrere
Stichtage L = t - u, u zufaellig aus 1..100, und fuer jeden Stichtag

    T(L)    = erster Tag > L mit MODIS-Feuer irgendwo im 13 x 13-Ausschnitt um s
              (Rohdaten MCD14A1, ohne unsere Gueltigkeitsmaske; ausserdem der
              Zieltag t selbst, wenn das Paper-Label dort 1 ist)
    event   = 1, falls T - L <= 100, sonst 0
    t_event = min(T - L, 100)                   rechtszensiert bei 100

Eingabe: die 30 Tage bis einschliesslich L aus unserem Wuerfel, umsortiert in
die Kanalreihenfolge des Caches. Ein Selbsttest vergleicht fuer Stichproben
mit L = t die Fenster byte-genau mit dem Cache.

Zusaetzlich je Stichtag das Merkmal ``recent_fire``: Feuer im Ausschnitt in
L-6 .. L. Damit laesst sich die Variante "nur neue Braende" bilden.

Hinweis zur Ziehung: Die Stichtage haengen an den Paper-Stichproben, die zu
einem Drittel Feuertage sind. Die Ereignisrate ist deshalb weit hoeher als in
der Wirklichkeit; Rangordnung und Form der Kurven sind auswertbar, die
absoluten Wahrscheinlichkeiten gelten fuer diese Stichprobe.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import tarfile
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest
from wildfire.data.dataset import CubeReader

H_FOLLOW = 100
T_WIN = 30
HALF = 6
DAY_BITS = 1 << 15
DAILY_RE = re.compile(r"(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})\.tif{1,2}$", re.I)

# Cache-Kanal -> (Quelle, Index im Wuerfel). Reihenfolge laut Audit s5.
CACHE_FROM_CUBE = (
    [("dyn", i) for i in range(26)]            # ch0 Feuer, ch1-18 ERA5, ch19-24 FWI, ch25 VPD
    + [("stat", i) for i in range(5)]          # ch26-29 OSM, ch30 Worldpop
    + [("stat", i) for i in range(5, 12)]      # ch31-37 DEM
    + [("dyn", i) for i in range(26, 42)]      # ch38-41 NDVI/EVI/LAI/FPAR, ch42-45 Reflexion,
                                               # ch46-49 LST/Emis, ch50-53 BT
    + [("stat", 12)]                           # ch54 Landbedeckung
)
assert len(CACHE_FROM_CUBE) == 55


def fire_keys(dataset_root: Path, dates_index: dict[str, int], W: int) -> np.ndarray:
    """Alle MODIS-Feuer-Pixeltage, unmaskiert, als sortierte Schluessel pixel*2^15+tag."""
    keys = []
    for tar in sorted((dataset_root / "Entire_Canada_Maps" / "MCD14A1").glob("*.tar")):
        with tarfile.open(tar, "r:") as tf:
            for m in tf:
                hit = DAILY_RE.search(Path(m.name).name) if m.isfile() else None
                if not hit:
                    continue
                t = dates_index.get(f"{hit['y']}-{hit['m']}-{hit['d']}")
                if t is None:
                    continue
                with MemoryFile(tf.extractfile(m).read()) as mem, mem.open() as ds:
                    rr, cc = np.nonzero(ds.read(1) == 1)
                if rr.size:
                    keys.append((rr.astype(np.int64) * W + cc) * DAY_BITS + t)
    k = np.sort(np.concatenate(keys))
    print(f"[surv] {k.size:,} MODIS-Feuer-Pixeltage (unmaskiert)", flush=True)
    return k


def window(cube: CubeReader, L: int, r: int, c: int) -> np.ndarray:
    """(30, 55, 13, 13) uint8 bis einschliesslich L, Cache-Reihenfolge, Rand genullt."""
    year, li = cube._pos[L]
    dyn, stat = cube._arrays(year)
    out = np.zeros((T_WIN, 55, 2 * HALF + 1, 2 * HALF + 1), np.uint8)
    r0, r1 = max(0, r - HALF), min(cube.H, r + HALF + 1)
    c0, c1 = max(0, c - HALF), min(cube.W, c + HALF + 1)
    d = np.asarray(dyn[li - T_WIN + 1:li + 1, r0:r1, c0:c1, :])      # (30, h, w, 42)
    s = np.asarray(stat[r0:r1, c0:c1, :])                           # (h, w, 13)
    rs, cs = slice(r0 - (r - HALF), r1 - (r - HALF)), slice(c0 - (c - HALF), c1 - (c - HALF))
    for ch, (src, i) in enumerate(CACHE_FROM_CUBE):
        if src == "dyn":
            out[:, ch, rs, cs] = d[..., i]
        else:
            out[:, ch, rs, cs] = s[..., i][None]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--per-sample", nargs=3, type=int, default=[8, 2, 2],
                    metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    rng = np.random.default_rng(int(cfg["seed"]))

    cube = CubeReader(exp)
    W = cube.W
    t_of = {d: i for i, d in enumerate(cube.dates)}
    keys = fire_keys(Path(cfg["dataset_root"]), t_of, W)

    pdir = exp / "derived_data" / "paper" / args.cache
    meta, y, split = np.load(pdir / "meta.npy"), np.load(pdir / "y.npy").astype(int), np.load(pdir / "split.npy")
    D = np.array([t_of[dt.date(int(m // 10000), int(m // 100 % 100), int(m % 100)).isoformat()]
                  for m in meta[:, 0]])
    rows, cols = meta[:, 1].astype(int), meta[:, 2].astype(int)

    def landmark_ok(L: int) -> bool:
        return (L in cube._pos and (L - T_WIN + 1) in cube._pos
                and cube._pos[L - T_WIN + 1][0] == cube._pos[L][0]
                and cube._pos[L - T_WIN + 1][1] == cube._pos[L][1] - T_WIN + 1)

    # Nachbarschaft als Versatzliste
    dr, dc = np.meshgrid(np.arange(-HALF, HALF + 1), np.arange(-HALF, HALF + 1), indexing="ij")
    dr, dc = dr.ravel(), dc.ravel()

    def first_fire_after(Ls, rr, cc):
        """Erster Feuertag > L im Ausschnitt (oder sehr gross) und Feuer in L-6..L."""
        R = rr[:, None] + dr[None]; C = cc[:, None] + dc[None]
        ok = (R >= 0) & (R < cube.H) & (C >= 0) & (C < W)
        flat = np.where(ok, R * W + C, 0).astype(np.int64)
        lo = flat * DAY_BITS + (Ls[:, None] + 1)
        pos = np.searchsorted(keys, lo)
        nxt = np.where(pos < keys.size, keys[np.minimum(pos, keys.size - 1)], -1)
        same = ok & (nxt // DAY_BITS == flat)
        day = np.where(same, nxt % DAY_BITS, 10 ** 9).min(1)
        lo7 = flat * DAY_BITS + (Ls[:, None] - 6)
        p7 = np.searchsorted(keys, lo7)
        n7 = np.where(p7 < keys.size, keys[np.minimum(p7, keys.size - 1)], -1)
        recent = (ok & (n7 // DAY_BITS == flat) & (n7 % DAY_BITS <= Ls[:, None])).any(1)
        return day, recent

    out_dir = exp / "derived_data" / "paper_surv" / args.cache
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for part, m in zip(("train", "val", "test"), args.per_sample):
        idx = np.where(split == part)[0]
        sel_i, sel_L, sel_u = [], [], []
        for i in idx:
            us = [u for u in range(1, H_FOLLOW + 1) if landmark_ok(int(D[i] - u))]
            if not us:
                continue
            for u in rng.choice(us, size=min(m, len(us)), replace=False):
                sel_i.append(i); sel_L.append(int(D[i] - u)); sel_u.append(int(u))
        sel_i, sel_L, sel_u = np.array(sel_i), np.array(sel_L), np.array(sel_u)
        day, recent = first_fire_after(sel_L, rows[sel_i], cols[sel_i])
        # Paper-Label am Zieltag einbeziehen: y = 1 heisst Feuer an t
        day = np.where(y[sel_i] == 1, np.minimum(day, D[sel_i]), day)
        tte = day - sel_L
        event = (tte <= H_FOLLOW).astype(np.int8)
        t_event = np.minimum(tte, H_FOLLOW).astype(np.int16)

        X = np.lib.format.open_memmap(out_dir / f"{part}_X.npy", mode="w+", dtype=np.uint8,
                                      shape=(sel_i.size, T_WIN, 55, 2 * HALF + 1, 2 * HALF + 1))
        order = np.argsort(sel_L, kind="mergesort")              # nach Tag lesen: Lokalitaet
        for n, j in enumerate(order):
            X[j] = window(cube, int(sel_L[j]), int(rows[sel_i[j]]), int(cols[sel_i[j]]))
            if n % 5000 == 0:
                print(f"    {part}: {n}/{sel_i.size}", flush=True)
        X.flush(); del X
        np.savez(out_dir / f"{part}_meta.npz", sample=sel_i, L=sel_L, u=sel_u, event=event,
                 t_event=t_event, recent_fire=recent, y_target=y[sel_i],
                 row=rows[sel_i], col=cols[sel_i])
        summary[part] = {
            "n_landmarks": int(sel_i.size), "n_samples": int(np.unique(sel_i).size),
            "event_rate": float(event.mean()),
            "event_rate_without_recent_fire": float(event[~recent].mean()) if (~recent).any() else None,
            "share_recent_fire": float(recent.mean()),
            "t_event_quantiles_events": {q: float(np.percentile(t_event[event == 1], q))
                                         for q in (10, 25, 50, 75, 90)} if event.any() else {},
            "bytes": human_bytes(sel_i.size * T_WIN * 55 * 169),
        }
        print(f"[surv] {part}: {summary[part]}", flush=True)

    # ---- Selbsttest: Fenster bis t gegen den Cache (letzte 30 Tage), byte-genau
    cache_X = np.load(pdir / "X.npy", mmap_mode="r")             # (N, 30, 55, 13, 13), endet an t
    cand = [i for i in rng.permutation(len(D))[:400] if landmark_ok(int(D[i]))][:40]
    mism = {ch: 0 for ch in range(55)}
    for i in cand:
        a = window(cube, int(D[i]), int(rows[i]), int(cols[i]))
        b = np.asarray(cache_X[i])
        for ch in range(55):
            mism[ch] += int((a[:, ch] != b[:, ch]).sum())
    bad = {ch: v for ch, v in mism.items() if v}
    print(f"[surv] Selbsttest an {len(cand)} Stichproben: abweichende Werte je Kanal {bad or 'keine'}")
    if sum(bad.values()) > 0.001 * len(cand) * T_WIN * 55 * 169:
        raise SystemExit("ABBRUCH: Wuerfel-Fenster weichen vom Cache ab - Kanalzuordnung pruefen")

    write_manifest(cfg, f"p7_surv_prepare_{args.cache}", args.config, {
        "cache": args.cache, "follow_up_days": H_FOLLOW, "window_days": T_WIN, "patch": 13,
        "event_definition": "erster Tag > L mit MODIS-Feuer (MCD14A1, unmaskiert) im 13x13-Ausschnitt; "
                            "Zieltag t zusaetzlich, wenn Paper-Label 1",
        "landmarks_per_sample": dict(zip(("train", "val", "test"), args.per_sample)),
        "channel_order": "Cache-Reihenfolge (Audit s5), aus dem Wuerfel umsortiert",
        "self_test": {"n_samples": len(cand), "mismatch_by_channel": bad},
        "parts": summary,
    })


if __name__ == "__main__":
    main()

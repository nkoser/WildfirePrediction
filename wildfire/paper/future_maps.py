"""Feuerkarten der Folgetage je Wuerfel-Stichtag: Ziele fuer den Tages-Transformer (fire_gpt.py).

Je Stichtag von paper_pixel (train_meta.npz: row, col, L) die MODIS-Feuermaske (MCD14A1 == 1, unmaskiert,
Quelle wie Cache-Kanal 0 und ring_targets.py) auf dem 13 x 13-Ausschnitt an den Tagen L+1 .. L+F:
  fut       (n, F, 13, 13) bool, gepackt entlang der Tage (np.packbits, axis 1 -> (n, ceil(F/8), 13, 13) uint8)
  fut_ok    (n, F) bool: Tag L+k liegt im Wuerfel und ist kalendarisch genau k Tage nach L
Pruefungen: erster Feuertag <= F je Pixel == train_t (wo train_event und train_t <= F, und umgekehrt), nur
auf Stichtagen mit allen F Tagen gueltig; Tag-1-Karte gegen train_t == 1.
Ausgabe NEU: derived_data/<cube-dir>_future<F>/ (fut.npy, fut_ok.npy), Manifest p59_future_maps.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.surv_prepare import DAY_BITS, HALF, fire_keys


def future_block(keys, Ls, rr, cc, H, W, F):
    """-> fut (n, F, 13, 13) bool: Feuer an den Tagen L+1 .. L+F."""
    P = 2 * HALF + 1
    d = np.arange(-HALF, HALF + 1)
    dr, dc = np.meshgrid(d, d, indexing="ij")
    R = rr[:, None] + dr.ravel()[None]
    C = cc[:, None] + dc.ravel()[None]
    ok = (R >= 0) & (R < H) & (C >= 0) & (C < W)
    flat = np.where(ok, R * W + C, 0).astype(np.int64)
    L = Ls[:, None]
    fut = np.zeros((rr.size, F, P * P), bool)
    pos = np.searchsorted(keys, flat * DAY_BITS + (L + 1))
    for j in range(F):                                   # hoechstens ein Schluessel je Pixel und Tag
        q = np.minimum(pos + j, keys.size - 1)
        kq = keys[q]
        day = kq % DAY_BITS - (L + 1)
        hit = ok & (pos + j < keys.size) & (kq // DAY_BITS == flat) & (day >= 0) & (day < F)
        n_i, p_i = np.nonzero(hit)
        fut[n_i, day[n_i, p_i], p_i] = True
    return fut.reshape(-1, F, P, P)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--days", type=int, default=31, help="Folgetage F (31 = Paper-Protokoll bis dt 30, Ziel D+1)")
    ap.add_argument("--chunk", type=int, default=10_000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    F = args.days
    src = exp / "derived_data" / args.cube_dir
    out = exp / "derived_data" / f"{args.cube_dir}_future{F}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"ABBRUCH: {out} existiert schon")
    out.mkdir(parents=True, exist_ok=True)
    cube = CubeReader(exp)
    H, W = cube.H, cube.W
    dates = [dt.date.fromisoformat(s) for s in cube.dates]
    keys = fire_keys(Path(cfg["dataset_root"]), {d: i for i, d in enumerate(cube.dates)}, W)
    m = np.load(src / "train_meta.npz")
    rows, cols, Ls = m["row"], m["col"], m["L"]
    n = rows.size
    ok_days = np.zeros((n, F), bool)
    for i, L in enumerate(Ls):
        L = int(L)
        for k in range(1, F + 1):
            ok_days[i, k - 1] = L + k < len(dates) and (dates[L + k] - dates[L]).days == k
    ev0 = np.load(src / "train_event.npy", mmap_mode="r")
    t0 = np.load(src / "train_t.npy", mmap_mode="r")
    fut_o = np.lib.format.open_memmap(out / "fut.npy", mode="w+", dtype=np.uint8, shape=(n, (F + 7) // 8, 13, 13))
    np.save(out / "fut_ok.npy", ok_days)
    rate = np.zeros(F)
    n_cmp = mism = 0
    for a in range(0, n, args.chunk):
        b = min(n, a + args.chunk)
        f = future_block(keys, Ls[a:b], rows[a:b], cols[a:b], H, W, F)
        full = ok_days[a:b].all(1)
        if full.any():                                   # erster Feuertag <= F muss train_t entsprechen
            ff = f[full]
            first = np.where(ff.any(1), ff.argmax(1) + 1, 10 ** 6)
            e, t = np.asarray(ev0[a:b])[full], np.asarray(t0[a:b])[full].astype(int)
            ref = np.where(e & (t <= F), t, 10 ** 6)
            n_cmp += ref.size
            mism += int((first != ref).sum())
        rate += f.mean((0, 2, 3)) * (b - a)
        fut_o[a:b] = np.packbits(f, axis=1)
        print(f"    {b}/{n}", flush=True)
    fut_o.flush()
    frac = mism / max(n_cmp, 1)
    rate /= n
    print(f"[fut] erster Feuertag vs. train_t: {mism} von {n_cmp} Pixeln abweichend ({frac:.2e})", flush=True)
    if frac > 1e-4:
        raise SystemExit("ABBRUCH: Folgetage passen nicht zu train_event/train_t")
    summ = {"n": int(n), "F": F, "all_days_valid": float(ok_days.all(1).mean()), "mismatch_first_day": frac,
            "fire_rate_per_day": [float(v) for v in rate]}
    print(f"[fut] {summ}", flush=True)
    write_manifest(cfg, "p59_future_maps", args.config, {
        "source": str(src), "out": str(out), "summary": summ,
        "def": "Feuerkanal MCD14A1 == 1 unmaskiert; Tage L+1..L+F gepackt (axis 1); fut_ok: Tag im Wuerfel und kalendarisch L+k"})


if __name__ == "__main__":
    main()

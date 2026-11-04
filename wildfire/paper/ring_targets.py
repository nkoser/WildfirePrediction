"""Feuerring um die Trainingsausschnitte: beobachtetes Feuer und Zuendtage auf 23 x 23 statt 13 x 13.

Der Contagion-Kern (Radius R = 5) sieht im 13 x 13-Ausschnitt nur bei den inneren 3 x 3 Pixeln alle Nachbarn;
Feuer, das von aussen hereinlaeuft, muss das Modell sonst der eigenen Zuendung a zuschreiben. Hier wird je
Wuerfel-Stichtag von paper_pixel (train_meta.npz: row, col, L) ein Rand von R Pixeln dazugenommen, nur fuer
den Feuerkanal:
  ring_hist   (n, 30, 23, 23) bool, gepackt entlang der Tage (np.packbits, axis 1 -> (n, 4, 23, 23) uint8):
              MODIS-Feuer (MCD14A1 == 1, unmaskiert, Quelle wie Cache-Kanal 0) an den Tagen L-29 .. L
  ring_event  (n, 23, 23) bool, ring_t (n, 23, 23) int16: erster Feuertag > L, t = Tag - L, zensiert bei 100
              (wie pixel_landmarks.pixel_events, nur groesser)
Pruefungen: innere 13 x 13 von ring_event/ring_t == train_event/train_t (exakt); innere Historie gegen den
Feuerkanal von train_X auf einer Stichprobe (Uebereinstimmung wird ausgegeben, Abbruch unter 0.999).
Ausgabe NEU: derived_data/<cube-dir>_ring<R>/ (ring_hist.npy, ring_event.npy, ring_t.npy), Manifest p57_ring_targets.
pixel_landmarks.py und paper_pixel bleiben unveraendert.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.pixel_landmarks import H_FOLLOW
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.surv_prepare import DAY_BITS, HALF, T_WIN, fire_keys


def ring_block(keys, Ls, rr, cc, H, W, half):
    """-> hist (n, T_WIN, P, P) bool, event (n, P, P) bool, t (n, P, P) int16, P = 2 half + 1."""
    P = 2 * half + 1
    d = np.arange(-half, half + 1)
    dr, dc = np.meshgrid(d, d, indexing="ij")
    R = rr[:, None] + dr.ravel()[None]
    C = cc[:, None] + dc.ravel()[None]
    ok = (R >= 0) & (R < H) & (C >= 0) & (C < W)
    flat = np.where(ok, R * W + C, 0).astype(np.int64)
    L = Ls[:, None]
    # Historie: alle Schluessel dieses Pixels in L-29 .. L (hoechstens T_WIN Stueck, einer je Tag)
    hist = np.zeros((rr.size, T_WIN, P * P), bool)
    pos = np.searchsorted(keys, flat * DAY_BITS + (L - T_WIN + 1))
    for j in range(T_WIN):
        q = np.minimum(pos + j, keys.size - 1)
        kq = keys[q]
        day = kq % DAY_BITS - (L - T_WIN + 1)
        hit = ok & (pos + j < keys.size) & (kq // DAY_BITS == flat) & (day >= 0) & (day < T_WIN)
        n_i, p_i = np.nonzero(hit)
        hist[n_i, day[n_i, p_i], p_i] = True
    # Zukunft: erster Feuertag > L
    pos = np.searchsorted(keys, flat * DAY_BITS + (L + 1))
    nxt = np.where(pos < keys.size, keys[np.minimum(pos, keys.size - 1)], -1)
    same = ok & (nxt // DAY_BITS == flat)
    tte = np.where(same, nxt % DAY_BITS - L, 10 ** 6)
    return (hist.reshape(-1, T_WIN, P, P), (tte <= H_FOLLOW).reshape(-1, P, P),
            np.minimum(tte, H_FOLLOW).astype(np.int16).reshape(-1, P, P))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--ring", type=int, default=5, help="Randbreite in Pixeln (= Kernradius)")
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument("--check", type=int, default=3000, help="Stichtage fuer den Abgleich mit train_X")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    src = exp / "derived_data" / args.cube_dir
    out = exp / "derived_data" / f"{args.cube_dir}_ring{args.ring}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"ABBRUCH: {out} existiert schon")
    out.mkdir(parents=True, exist_ok=True)
    cube = CubeReader(exp)
    H, W = cube.H, cube.W
    keys = fire_keys(Path(cfg["dataset_root"]), {d: i for i, d in enumerate(cube.dates)}, W)
    m = np.load(src / "train_meta.npz")
    rows, cols, Ls = m["row"], m["col"], m["L"]
    n, half = rows.size, HALF + args.ring
    P = 2 * half + 1
    ev0 = np.load(src / "train_event.npy", mmap_mode="r")
    t0 = np.load(src / "train_t.npy", mmap_mode="r")
    Xm = np.load(src / "train_X.npy", mmap_mode="r")
    hist_o = np.lib.format.open_memmap(out / "ring_hist.npy", mode="w+", dtype=np.uint8, shape=(n, (T_WIN + 7) // 8, P, P))
    ev_o = np.lib.format.open_memmap(out / "ring_event.npy", mode="w+", dtype=bool, shape=(n, P, P))
    t_o = np.lib.format.open_memmap(out / "ring_t.npy", mode="w+", dtype=np.int16, shape=(n, P, P))
    inner = (slice(None), slice(args.ring, args.ring + 2 * HALF + 1), slice(args.ring, args.ring + 2 * HALF + 1))
    agree, ring_fire, ring_event = [], 0, 0
    for a in range(0, n, args.chunk):
        b = min(n, a + args.chunk)
        h, e, t = ring_block(keys, Ls[a:b], rows[a:b], cols[a:b], H, W, half)
        if not (np.array_equal(e[inner], ev0[a:b]) and np.array_equal(t[inner], t0[a:b])):
            raise SystemExit(f"ABBRUCH: innere Ziele weichen von train_event/train_t ab (Zeilen {a}..{b})")
        if a < args.check:
            k = min(b, args.check) - a
            fx = np.asarray(Xm[a:a + k, :, FIRE_CH]) > 0.5                   # (k, 30, 13, 13)
            agree.append((fx == h[:k, :, inner[1], inner[2]]).mean())
        ring = np.ones((P, P), bool); ring[inner[1:]] = False
        ring_fire += int(h[:, :, ring].any(1).sum()); ring_event += int(e[:, ring].sum())
        hist_o[a:b] = np.packbits(h, axis=1)
        ev_o[a:b], t_o[a:b] = e, t
        print(f"    {b}/{n}", flush=True)
    hist_o.flush(); ev_o.flush(); t_o.flush()
    agree = float(np.mean(agree))
    print(f"[ring] Historie innen vs. train_X-Feuerkanal: Uebereinstimmung {agree:.5f}", flush=True)
    if agree < 0.999:
        raise SystemExit("ABBRUCH: Feuerhistorie passt nicht zum Feuerkanal von train_X")
    summ = {"n": int(n), "P": P, "ring": args.ring, "hist_agreement_inner": agree,
            "ring_px_with_hist_fire_per_sample": ring_fire / n, "ring_px_with_event_per_sample": ring_event / n,
            "inner_events_per_sample": float(np.asarray(ev0).sum() / n)}
    print(f"[ring] {summ}", flush=True)
    write_manifest(cfg, "p57_ring_targets", args.config, {
        "source": str(src), "out": str(out), "summary": summ,
        "def": "Feuerkanal MCD14A1 == 1 unmaskiert; hist Tage L-29..L gepackt (axis 1); t erster Tag > L, zensiert 100"})


if __name__ == "__main__":
    main()

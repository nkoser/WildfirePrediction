"""Tageslabels fuer den Rate-Kopf: brennt es an Tag L + tau (tau = 1..100) im Ausschnitt bzw. in der Mitte?

Die Survival-Ziele der Wuerfel-Stichtage (pixel_landmarks) kennen nur das erste Feuer je
Pixel. Das Ziel des Paper-Protokolls ist aber "Feuer im Ausschnitt an Tag D + dt", also ein
wiederkehrendes Ereignis. Hier wird es fuer dieselben Stichtage nachgereicht, aus derselben
Quelle und mit demselben Tagesindex wie die Survival-Ziele (MCD14A1 == 1, unmaskiert,
Wuerfel-Tagesindex aus surv_prepare.fire_keys):

  patch   (n, 100) bool: irgendein Pixel des 13 x 13-Ausschnitts brennt an Tag L + tau
  centre  (n, 100) bool: die Mittelzelle brennt an Tag L + tau
  L       Stichtage wie in train_meta.npz (Abgleich beim Laden)

Selbsttest: der erste Tag mit Feuer im Ausschnitt ist das Survival-Ereignis des Ausschnitts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.surv_prepare import DAY_BITS, HALF, fire_keys

H_FOLLOW = 100
P = 2 * HALF + 1


def daily_fire(keys: np.ndarray, Ls: np.ndarray, rr: np.ndarray, cc: np.ndarray, H: int, W: int,
               horizon: int = H_FOLLOW, chunk: int = 20000):
    """Je Stichtag: Feuer an Tag L + 1 .. L + horizon, (patch (n, horizon), centre (n, horizon)) bool."""
    n = Ls.size
    patch = np.zeros((n, horizon), bool)
    centre = np.zeros((n, horizon), bool)
    dr, dc = np.meshgrid(np.arange(-HALF, HALF + 1), np.arange(-HALF, HALF + 1), indexing="ij")
    mid = (P * P) // 2
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        R = rr[sl, None] + dr.ravel()[None]
        C = cc[sl, None] + dc.ravel()[None]
        ok = (R >= 0) & (R < H) & (C >= 0) & (C < W)
        flat = np.where(ok, R * W + C, 0).astype(np.int64)
        L = Ls[sl, None].astype(np.int64)
        lo = np.searchsorted(keys, flat * DAY_BITS + L + 1)
        hi = np.searchsorted(keys, flat * DAY_BITS + L + horizon, side="right")
        cnt = np.where(ok, hi - lo, 0).ravel()
        if not cnt.sum():
            continue
        cell = np.repeat(np.arange(cnt.size), cnt)                     # (Stichtag, Pixel) je Treffer
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        pos = np.repeat(lo.ravel(), cnt) + np.arange(cnt.sum()) - start
        i = cell // (P * P)
        tau = keys[pos] % DAY_BITS - L.ravel()[i]                       # 1 .. horizon
        patch[a + i, tau - 1] = True
        c = cell % (P * P) == mid
        centre[a + i[c], tau[c] - 1] = True
    return patch, centre


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cube-dir", default="paper_pixel")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    cube = CubeReader(exp)
    keys = fire_keys(Path(cfg["dataset_root"]), {d: i for i, d in enumerate(cube.dates)}, cube.W)
    cdir = exp / "derived_data" / args.cube_dir
    m = np.load(cdir / "train_meta.npz")
    patch, centre = daily_fire(keys, m["L"], m["row"], m["col"], cube.H, cube.W)

    ev, t = np.load(cdir / "train_event.npy"), np.load(cdir / "train_t.npy")
    pe, pt = ev.reshape(len(ev), -1).any(1), t.reshape(len(t), -1).min(1)
    first = np.where(patch.any(1), patch.argmax(1) + 1, H_FOLLOW)
    agree = float(((patch.any(1) == pe) & (np.where(pe, first == pt, True))).mean())
    print(f"[rate] Selbsttest erstes Feuer im Ausschnitt = Survival-Ereignis: {agree:.5f}", flush=True)
    if agree < 0.999:
        raise SystemExit("ABBRUCH: Tageslabels passen nicht zu den Survival-Zielen")
    np.savez_compressed(cdir / "train_rate.npz", patch=patch, centre=centre, L=m["L"])
    kind = m["kind"]
    summary = {
        "n": int(patch.shape[0]),
        "daily_rate_patch": {"fire": float(patch[kind == 0].mean()), "random": float(patch[kind == 1].mean())},
        "daily_rate_centre": {"fire": float(centre[kind == 0].mean()), "random": float(centre[kind == 1].mean())},
        "fire_days_per_patch_with_event": float(patch[pe].sum(1).mean()),
        "rate_at_tau_patch": {str(k): float(patch[:, k - 1].mean()) for k in (1, 2, 8, 16, 31, 60, 100)},
        "cif_at_tau_patch": {str(k): float((pe & (pt <= k)).mean()) for k in (1, 2, 8, 16, 31, 60, 100)},
        "self_test_agreement": agree,
    }
    print(f"[rate] {summary}", flush=True)
    write_manifest(cfg, "p23_rate_labels", args.config, {
        "source": "MCD14A1 == 1 unmaskiert, Wuerfel-Tagesindex (wie pixel_landmarks)",
        "labels": "patch/centre (n, 100) bool: Feuer an Tag L + tau, tau = 1..100",
        "summary": summary,
    })


if __name__ == "__main__":
    main()

"""Risikokarte fuer einen Kartenausschnitt zu beliebigen Zeitpunkten.

Beantwortet die Frage, um die es am Ende geht: Wie gross ist das Risiko,
dass in diesem Gebiet innerhalb von 1, 7, 30, 100 ... Tagen ein Feuer
entsteht?

Zwei Eigenschaften machen das moeglich:

  * Das U-Net ist **vollstaendig faltend**. Trainiert wird auf
    33x33-Ausschnitten, angewandt werden kann es auf eine Kachel
    beliebiger Groesse - bis hin zum ganzen Gitter von 1968x928. Die
    Ausgabe ist dann eine Karte statt eines einzelnen Werts.
  * Der Bernstein-Kopf liefert eine **stetige** Survival-Funktion. Die
    Abfragezeitpunkte werden erst hier festgelegt, nicht beim Training.
    Nach 37,5 Tagen zu fragen kostet nichts extra.

Randbehandlung: Faltungen brauchen Kontext. Eine Kachel wird deshalb mit
``--halo`` Pixeln Rand geladen und die Ausgabe anschliessend darauf
zurueckgeschnitten - sonst zeigen die Kachelraender Artefakte, die im
Zusammensetzen als Gitternetz sichtbar waeren.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import load_config
from wildfire.data.dataset import LOCKED_CHANNELS, CubeReader
from wildfire.models.torch_data import slice_days_back


def build_input(cube: CubeReader, t: int, r0: int, r1: int, c0: int, c1: int,
                offsets: list[int], cooldown: int, norm) -> np.ndarray:
    """(C_in, h, w) fuer eine Kachel - dieselbe Zusammensetzung wie im Training.

    ``offsets`` sind Tage vor t, aelteste zuerst (``slice_days_back``).
    """
    dyn_mean, dyn_std, stat_mean, stat_std = norm
    locked = np.array(sorted(LOCKED_CHANNELS))
    slices = []
    for off in offsets:
        d = t - off
        year, li = cube._pos[d]
        arr, _ = cube._arrays(year)
        sl = np.asarray(arr[li, r0:r1, c0:c1, :], np.float32)      # (h, w, C)
        sl = np.moveaxis(sl, 2, 0)                                  # (C, h, w)
        vis = 1.0 if off >= cooldown else 0.0
        if not vis:
            sl[locked] = 0.0
        sl = (sl - dyn_mean[:, None, None]) / dyn_std[:, None, None]
        mask = np.full((1,) + sl.shape[1:], vis, np.float32)
        slices.append(np.concatenate([sl, mask], 0))
    year, _ = cube._pos[t]
    st = np.load(cube.per_year[year]["stat_file"], mmap_mode="r")
    stat = np.moveaxis(np.asarray(st[r0:r1, c0:c1, :], np.float32), 2, 0)
    stat = (stat - stat_mean[:, None, None]) / stat_std[:, None, None]
    return np.concatenate(slices + [stat], 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--date", required=True, help="Stichtag, z. B. 2023-07-15")
    ap.add_argument("--taus", default="1,7,14,30,60,100",
                    help="Abfragehorizonte in Tagen, beliebig")
    ap.add_argument("--bbox", default=None,
                    help="row0,row1,col0,col1 - Standard ist das ganze Gitter")
    ap.add_argument("--tile", type=int, default=256)
    # Rezeptives Feld des U-Nets (2 Pooling-Stufen, je 2 Conv 3x3): Radius 23.
    # Mit weniger Rand sieht das Netz am Kachelrand Nullen statt Nachbarn.
    ap.add_argument("--halo", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mc = ck["config"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cube = CubeReader(exp)
    t = cube.dates.index(args.date)
    L, C = int(mc["lookback"]), int(mc["cooldown"])
    offsets = slice_days_back(L, int(mc["n_slices"]))

    z = np.load(exp / "derived_data" / "norm_stats.npz")
    norm = (z["dyn_mean"], z["dyn_std"], z["stat_mean"], z["stat_std"])

    from wildfire.models.nets import SpatialHazardUNet
    model = SpatialHazardUNet(ck.get("in_channels", 228), horizon=int(mc["horizon"]),
                              width=int(mc["width"]))
    model.load_state_dict(ck["model"])
    model.eval().to(dev)

    H, W = cube.H, cube.W
    if args.bbox:
        r0, r1, c0, c1 = (int(x) for x in args.bbox.split(","))
    else:
        r0, r1, c0, c1 = 0, H, 0, W
    taus = [float(x) for x in args.taus.split(",")]
    out = np.zeros((len(taus), r1 - r0, c1 - c0), np.float32)
    valid = np.load(exp / "derived_data" / "masks" / "observed_nfdb.npy")

    tile, halo = args.tile, args.halo
    n_tiles = ((r1 - r0 + tile - 1) // tile) * ((c1 - c0 + tile - 1) // tile)
    print(f"[map] {args.date} (t={t}), Gebiet {r1-r0}x{c1-c0}, "
          f"{n_tiles} Kacheln à {tile} px + {halo} Rand, Horizonte {taus}")

    done = 0
    with torch.no_grad():
        for rr in range(r0, r1, tile):
            for cc in range(c0, c1, tile):
                re_, ce_ = min(rr + tile, r1), min(cc + tile, c1)
                # mit Rand laden, danach zurueckschneiden
                pr0, pr1 = max(0, rr - halo), min(H, re_ + halo)
                pc0, pc1 = max(0, cc - halo), min(W, ce_ + halo)
                x = build_input(cube, t, pr0, pr1, pc0, pc1, offsets, C, norm)
                xb = torch.from_numpy(x[None]).to(dev)
                logits = model(xb)[0]                      # (H, h, w)
                haz = torch.sigmoid(logits)
                cif = 1.0 - torch.cumprod(1.0 - haz, dim=0)
                for i, tau in enumerate(taus):
                    k = min(int(round(tau)), cif.shape[0]) - 1
                    tile_out = cif[k].cpu().numpy()
                    out[i, rr - r0:re_ - r0, cc - c0:ce_ - c0] = \
                        tile_out[rr - pr0:re_ - pr0, cc - pc0:ce_ - pc0]
                done += 1
                if done % 10 == 0:
                    print(f"    {done}/{n_tiles}", flush=True)

    out[:, ~valid[r0:r1, c0:c1]] = np.nan
    dest = Path(args.out) if args.out else (
        exp / "predictions" / f"riskmap_{args.date}.npz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, risk=out, taus=np.array(taus),
                        bbox=np.array([r0, r1, c0, c1]), date=args.date,
                        checkpoint=args.checkpoint)
    print(f"[map] {dest}")
    for i, tau in enumerate(taus):
        v = out[i][np.isfinite(out[i])]
        print(f"    tau={tau:6.1f} d: Median {np.median(v):.3e}, "
              f"p99 {np.percentile(v, 99):.3e}, max {v.max():.3e}")


if __name__ == "__main__":
    main()

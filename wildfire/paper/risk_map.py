"""Dichte Risikokarte eines STS-Modells fuer ganz Kanada an einem Stichtag, mit MODIS-Feuern als Referenz.

Das Modell sieht 13 x 13-Ausschnitte. Fuer eine Karte wird das Gitter mit
Fenstern im Raster --stride (Standard 3) abgedeckt; von jedem Fenster zaehlt
nur das mittlere crop x crop-Feld (ungerade, Standard 9), ueberlappende Felder
werden mit einem Zeltgewicht gemittelt (sonst sieht man das Fensterraster).
Jeder Pixel hat damit mindestens (13 - crop) / 2 Pixel Kontext in jede Richtung. Ausserhalb des Gitters wird wie im Training mit
Nullen aufgefuellt (surv_prepare.window).

Ausgabe je Stichtag (npz unter figures/risk_maps):
  F_<tau>        Pixel-CIF F(tau) = 1 - S(tau), Ensemble = Mittel der Survival-Kurven
                 ueber die Seeds; NaN ausserhalb der Landmaske Kanadas
  ign_<tau>      Anteil der Zuendung am kumulativen Pixel-Hazard (nur Contagion)
  fire_window    MODIS-Feuer im Eingabefenster L-29 .. L (vom Modell gesehen)
  fire_today     MODIS-Feuer am Stichtag L
  first_fire     erster Feuertag nach L (1 .. tau_max), 0 = keins (Wuerfel, Kanal 0)
Die Zukunft kommt aus dem Wuerfel, der Stichtag muss also L + tau_max <= 30.9. erlauben.
Die Wahrscheinlichkeiten sind roh (nicht kalibriert).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import P, load_sts
from wildfire.paper.surv_prepare import CACHE_FROM_CUBE, HALF, T_WIN

H = 100


def tile_centres(mask: np.ndarray, crop: int, stride: int | None = None):
    """Fenstermitten im Raster stride (Standard crop), deren Mittelfeld [r - crop//2, r + crop//2]
    mindestens einen Maskenpixel trifft. Mit stride <= crop hat jeder Pixel eine Mitte im Abstand <= crop//2."""
    h, s = crop // 2, stride or crop
    Hg, Wg = mask.shape
    rr = np.arange(s // 2, Hg + s // 2, s)
    cc = np.arange(s // 2, Wg + s // 2, s)
    out = [(r, c) for r in rr for c in cc if mask[max(0, r - h):r + h + 1, max(0, c - h):c + h + 1].any()]
    return np.array(out, np.int64).reshape(-1, 2)


def load_slab(cube: CubeReader, L: int, pad: int) -> np.ndarray:
    """(T_WIN, 55, H + 2 pad, W + 2 pad) uint8 bis einschliesslich L, Cache-Reihenfolge, Rand genullt."""
    year, li = cube._pos[L]
    dyn, stat = cube._arrays(year)
    d = np.asarray(dyn[li - T_WIN + 1:li + 1])                          # (T, H, W, 42)
    s = np.asarray(stat)                                                # (H, W, 13)
    out = np.zeros((T_WIN, len(CACHE_FROM_CUBE), cube.H + 2 * pad, cube.W + 2 * pad), np.uint8)
    for ch, (src, i) in enumerate(CACHE_FROM_CUBE):
        out[:, ch, pad:pad + cube.H, pad:pad + cube.W] = d[..., i] if src == "dyn" else s[..., i][None]
    return out


def future_fire(cube: CubeReader, L: int, tau_max: int) -> np.ndarray:
    """Erster MODIS-Feuertag nach L (1 .. tau_max) je Pixel, 0 = keins; aus Kanal 0 des Wuerfels."""
    year, li = cube._pos[L]
    dyn, _ = cube._arrays(year)
    if li + tau_max >= dyn.shape[0]:
        raise SystemExit(f"ABBRUCH: L + {tau_max} Tage liegt hinter dem Saisonende {cube.per_year[year]['days'][-1]}")
    first = np.zeros((cube.H, cube.W), np.int16)
    for k in range(tau_max, 0, -1):                                     # rueckwaerts: der frueheste Tag gewinnt
        first[np.asarray(dyn[li + k, :, :, 0]) > 0] = k
    return first


@torch.no_grad()
def predict(net, slab, centres, crop, pad, doy, m_t, s_t, device, taus, bs=64):
    """-> log S (n, crop, crop, len(taus)) der Mittelfelder; dazu log S nur aus Zuendung (Contagion) oder None."""
    lo, hi = HALF - crop // 2, HALF + crop // 2 + 1
    contagion = getattr(net, "head_kind", "") == "contagion"
    ls_all, li_all = [], []
    for a in range(0, len(centres), bs):
        cb = centres[a:a + bs]
        x = np.stack([slab[:, :, r + pad - HALF:r + pad + HALF + 1, c + pad - HALF:c + pad + HALF + 1] for r, c in cb])
        x = (torch.from_numpy(x).to(device).float() - m_t) / s_t
        d = torch.full((len(cb),), float(doy), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if contagion:
                obs, aa, k, logD = net.params(x, d)
            else:
                raw_px, _ = net(x, d)
        with torch.autocast(device_type=device.type, enabled=False):
            if contagion:
                ls = net.rollout(net.exogenous(obs, aa, k, logD), k, logD).view(len(cb), P, P, -1)
                lig = torch.nn.functional.pad(torch.log1p(-aa.clamp(max=1 - 1e-6)).cumsum(1), (0, 0, 1, 0))  # (b, H+1, N)
                li_all.append(lig.transpose(1, 2).reshape(len(cb), P, P, -1)[:, lo:hi, lo:hi, taus].cpu().numpy())
            else:
                hz = net.px_head.hazards(raw_px.reshape(-1, raw_px.shape[-1]), H).double()
                ls = torch.nn.functional.pad(torch.log1p(-hz.clamp(max=1 - 1e-7)).cumsum(1), (1, 0)).view(len(cb), P, P, -1)
        ls_all.append(ls[:, lo:hi, lo:hi, taus].float().cpu().numpy())
    return np.concatenate(ls_all), (np.concatenate(li_all) if li_all else None)


def paste(vals: np.ndarray, centres: np.ndarray, crop: int, shape) -> np.ndarray:
    """Mittelfelder (n, crop, crop, ...) zurueck auf das Gitter, ueberlappende Fenster mit Zeltgewicht
    (Mitte crop//2 + 1, Rand 1) gemittelt; ohne Ueberlappung genau der Fensterwert. Pixel ohne Fenster: NaN."""
    h = crop // 2
    tent = (h + 1 - np.abs(np.arange(crop) - h)).astype(np.float64)
    w2 = (tent[:, None] * tent[None, :]).reshape((crop, crop) + (1,) * (vals.ndim - 3))
    acc = np.zeros(shape + vals.shape[3:])
    wsum = np.zeros(shape + (1,) * (vals.ndim - 3))
    for v, (r, c) in zip(vals, centres):
        r0, c0 = r - h, c - h
        a0, b0 = max(0, r0), max(0, c0)
        r1, c1 = min(r0 + crop, shape[0]), min(c0 + crop, shape[1])
        if r1 <= a0 or c1 <= b0:
            continue
        ww = w2[a0 - r0:r1 - r0, b0 - c0:c1 - c0]
        acc[a0:r1, b0:c1] += ww * v[a0 - r0:r1 - r0, b0 - c0:c1 - c0]
        wsum[a0:r1, b0:c1] += ww
    with np.errstate(invalid="ignore"):
        return (acc / wsum).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pattern", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_d128_cg2w_ema_s?_s?.pt")
    ap.add_argument("--dates", nargs="+", default=["2023-06-01", "2023-08-10", "2024-07-15", "2025-05-28"])
    ap.add_argument("--taus", default="1,3,7,14,30")
    ap.add_argument("--crop", type=int, default=9)
    ap.add_argument("--stride", type=int, default=3, help="Fensterraster; < crop mittelt ueberlappende Fenster")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--tag", default="cg2w_ema")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    if args.crop % 2 == 0 or not 1 <= args.crop <= P or not 1 <= args.stride <= args.crop:
        raise SystemExit("--crop muss ungerade und <= 13 sein, 1 <= --stride <= --crop")
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    taus = [int(t) for t in args.taus.split(",")]
    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"ABBRUCH: keine Checkpoints fuer {args.pattern}")
    nets = [load_sts(f, device) for f in files]
    print(f"[map] {len(nets)} Checkpoints: " + ", ".join(Path(f).name for f in files), flush=True)

    cube = CubeReader(exp)
    md = exp / "derived_data" / "masks"
    mask = np.load(md / "land.npy") & np.load(md / "canada.npy")
    centres = tile_centres(mask, args.crop, args.stride)
    pad = HALF + args.crop
    print(f"[map] {len(centres):,} Fenster (Mittelfeld {args.crop} x {args.crop}, Raster {args.stride}) fuer {int(mask.sum()):,} Landpixel", flush=True)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    idx = {d: i for i, d in enumerate(cube.dates)}
    out_dir = exp / "figures" / "risk_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for date in args.dates:
        L = idx.get(date)
        if L is None or L not in cube._pos or (L - T_WIN + 1) not in cube._pos:
            raise SystemExit(f"ABBRUCH: {date} hat kein volles 30-Tage-Fenster im Wuerfel")
        t0 = time.time()
        slab = load_slab(cube, L, pad)
        first = future_fire(cube, L, max(taus))
        doy = dt.date.fromisoformat(date).timetuple().tm_yday
        S, ign = 0.0, []
        for f, net in zip(files, nets):
            ls, lig = predict(net, slab, centres, args.crop, pad, doy, m_t, s_t, device, taus, args.bs)
            S = S + np.exp(ls) / len(nets)
            if lig is not None:                                    # Anteil der Zuendung am kumulativen Hazard
                ign.append(lig / np.minimum(ls, -1e-12))
            print(f"    {Path(f).name}: {time.time() - t0:.0f} s", flush=True)
        F = paste(1 - S, centres, args.crop, mask.shape)
        res = {f"F_{t}": np.where(mask, F[..., j], np.nan).astype(np.float32) for j, t in enumerate(taus)}
        if ign:
            G = paste(np.mean(ign, 0), centres, args.crop, mask.shape)
            res.update({f"ign_{t}": np.where(mask, G[..., j], np.nan).astype(np.float32) for j, t in enumerate(taus)})
        fire_win = slab[:, 0, pad:pad + cube.H, pad:pad + cube.W] > 0
        out = out_dir / f"risk_{args.tag}_{date}.npz"
        np.savez_compressed(out, **res, mask=mask, fire_window=fire_win.any(0), fire_today=fire_win[-1],
                            first_fire=first, taus=np.array(taus), date=date, doy=doy,
                            checkpoints=np.array([Path(f).name for f in files]))
        written.append(str(out))
        print(f"[map] {date}: {out.name} in {time.time() - t0:.0f} s; F(7) Median {np.nanmedian(res.get('F_7', F[..., 0])):.2e}, "
              f"Feuer im Fenster {int(fire_win.any(0).sum())} px, neue Feuer bis {max(taus)} d {int((first > 0).sum())} px",
              flush=True)
    write_manifest(cfg, f"p22_risk_map_{args.tag}", args.config, {
        "checkpoints": files, "dates": args.dates, "taus": taus, "crop": args.crop, "stride": args.stride, "windows": int(len(centres)),
        "outputs": written, "note": "rohe Pixel-CIF, Ensemble = Mittel der Survival-Kurven; Zukunft aus Wuerfel-Kanal 0",
    })


if __name__ == "__main__":
    main()

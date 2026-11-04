"""Risikokarte wie risk_map, aber fuer das Wetter-Oracle (sts_dctm --future-days, Ablation v31).

risk_map.py bleibt unveraendert. Zusaetzlich zum 30-Tage-Eingabefenster bis L bekommt jedes Fenster das
beobachtete Wetter der Tage L + 1 .. L + F (Cache-Kanaele 1..25 wie future_weather.WEATHER_CH, gleiche
Normierung wie im Training: Mittel/Streuung des Caches auf diesen Kanaelen), mit k = F bekannten Folgetagen.
Obere Schranke, kein Modell fuer den Einsatz: das Netz kennt das Wetter der ganzen 30 Tage, auch fuer F(1).
Nur STS-Koepfe (das Oracle gibt es nicht fuer Contagion).

Ausgabe wie risk_map (risk_<tag>_<datum>.npz unter figures/risk_maps), also direkt fuer teaser_spread nutzbar.

  python -m wildfire.paper.risk_map_oracle --config ... --pattern ".../sts_cache_G_next_cube_abl_oracle_s?_s?.pt" \
      --dates 2023-05-19 2023-06-02 2025-05-26 --taus 1,...,30 --tag teaser_lho
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
from wildfire.paper.future_weather import WEATHER_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.risk_map import H, future_fire, load_slab, paste, tile_centres
from wildfire.paper.sts_dctm import P, load_sts
from wildfire.paper.surv_prepare import HALF, T_WIN


@torch.no_grad()
def predict_oracle(net, slab, fslab, centres, crop, pad, doy, m_t, s_t, mf_t, sf_t, device, taus, bs=64):
    """Wie risk_map.predict fuer STS-Koepfe, mit xf = Folgewetter (B, F, 25, 13, 13) und kf = F."""
    lo, hi = HALF - crop // 2, HALF + crop // 2 + 1
    fd = net.future.days
    out = []
    for a in range(0, len(centres), bs):
        cb = centres[a:a + bs]
        win = lambda s: np.stack([s[:, :, r + pad - HALF:r + pad + HALF + 1, c + pad - HALF:c + pad + HALF + 1] for r, c in cb])
        x = (torch.from_numpy(win(slab)).to(device).float() - m_t) / s_t
        xf = (torch.from_numpy(win(fslab)[:, :fd]).to(device).float() - mf_t) / sf_t
        kf = torch.full((len(cb),), fd, device=device)
        d = torch.full((len(cb),), float(doy), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_px, _ = net(x, d, xf=xf, kf=kf)
        with torch.autocast(device_type=device.type, enabled=False):
            hz = net.px_head.hazards(raw_px.reshape(-1, raw_px.shape[-1]), H).double()
            ls = torch.nn.functional.pad(torch.log1p(-hz.clamp(max=1 - 1e-7)).cumsum(1), (1, 0)).view(len(cb), P, P, -1)
        out.append(ls[:, lo:hi, lo:hi, taus].float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pattern", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_oracle_s?_s?.pt")
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--taus", default="1,3,7,14,30")
    ap.add_argument("--crop", type=int, default=9)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--tag", default="oracle")
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
    if any(getattr(n, "future", None) is None for n in nets):
        raise SystemExit("ABBRUCH: Checkpoint ohne Wetter-Oracle (future_days)")
    fd = nets[0].future.days
    print(f"[oracle] {len(nets)} Checkpoints, {fd} Folgetage: " + ", ".join(Path(f).name for f in files), flush=True)

    cube = CubeReader(exp)
    md = exp / "derived_data" / "masks"
    mask = np.load(md / "land.npy") & np.load(md / "canada.npy")
    centres = tile_centres(mask, args.crop, args.stride)
    pad = HALF + args.crop
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    mf_t, sf_t = m_t[:, :, WEATHER_CH], s_t[:, :, WEATHER_CH]
    idx = {d: i for i, d in enumerate(cube.dates)}
    out_dir = exp / "figures" / "risk_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    for date in args.dates:
        if (out_dir / f"risk_{args.tag}_{date}.npz").exists():
            raise SystemExit(f"ABBRUCH: risk_{args.tag}_{date}.npz existiert")
    written = []
    for date in args.dates:
        L = idx.get(date)
        if L is None or L not in cube._pos or (L - T_WIN + 1) not in cube._pos:
            raise SystemExit(f"ABBRUCH: {date} hat kein volles 30-Tage-Fenster im Wuerfel")
        t0 = time.time()
        first = future_fire(cube, L, max(max(taus), fd))                 # prueft auch L + fd in der Saison
        slab = load_slab(cube, L, pad)
        fslab = np.ascontiguousarray(load_slab(cube, L + T_WIN, pad)[:fd, WEATHER_CH])   # Tage L + 1 .. L + fd
        doy = dt.date.fromisoformat(date).timetuple().tm_yday
        S = 0.0
        for f, net in zip(files, nets):
            ls = predict_oracle(net, slab, fslab, centres, args.crop, pad, doy, m_t, s_t, mf_t, sf_t, device, taus, args.bs)
            S = S + np.exp(ls) / len(nets)
            print(f"    {Path(f).name}: {time.time() - t0:.0f} s", flush=True)
        F = paste(1 - S, centres, args.crop, mask.shape)
        res = {f"F_{t}": np.where(mask, F[..., j], np.nan).astype(np.float32) for j, t in enumerate(taus)}
        fire_win = slab[:, 0, pad:pad + cube.H, pad:pad + cube.W] > 0
        out = out_dir / f"risk_{args.tag}_{date}.npz"
        np.savez_compressed(out, **res, mask=mask, fire_window=fire_win.any(0), fire_today=fire_win[-1],
                            first_fire=first, taus=np.array(taus), date=date, doy=doy,
                            checkpoints=np.array([Path(f).name for f in files]))
        written.append(str(out))
        print(f"[oracle] {date}: {out.name} in {time.time() - t0:.0f} s, neue Feuer bis {max(taus)} d {int((first > 0).sum())} px",
              flush=True)
    write_manifest(cfg, f"p48_risk_map_oracle_{args.tag}", args.config, {
        "checkpoints": files, "dates": args.dates, "taus": taus, "crop": args.crop, "stride": args.stride,
        "future_days": fd, "outputs": written,
        "note": "Wetter-Oracle: beobachtetes Wetter L+1..L+fd als Eingabe (kf = fd), rohe Pixel-CIF, Ensemble = Mittel der Survival-Kurven",
    })


if __name__ == "__main__":
    main()

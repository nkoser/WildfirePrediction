"""Karte "brennt an Tag t" aus vorhandenen Contagion-Checkpoints, ohne neues Training.

patch_map_eval bewertet unsere Modelle mit F(dt + 1) je Pixel, also "erste Zuendung bis t". Das Paper-Label je
Pixel (Ziel "tag") ist aber "brennt an Tag t": Pixel, die weiterbrennen oder schon im Eingabefenster brannten,
gehoeren dazu. Das Contagion-Modell hat dafuer schon die Brennkurve D (Wahrscheinlichkeit, dass ein Feuer a Tage
nach dem Ausbruch noch brennt). Hier je Pixel am Tag t = dt + 1 nach dem Stichtag:
    b_obs(t)  = 1 - prod_j (1 - obs_j D(u_j + t))          beobachtete Feuer des Fensters (Alter u_j am Stichtag)
    b_pred(t) = sum_s f(s) D(t - s)                         vorhergesagte Erstzuendungen f (Mean-Field-Rollout)
    P(t)      = 1 - (1 - b_obs(t)) (1 - b_pred(t))
Gespeichert wie patch_map_eval (scores_<name>_burn.npz, dazu scores_<name>_burnobs.npz nur mit b_obs als gelernte
Persistenz), damit patch_map_eval report sie mit auswertet. patch_map_eval.py bleibt unveraendert.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.compare import CENTRE, DTS
from wildfire.paper.joint_eval import set_bias
from wildfire.paper.patch_map_eval import ap, load_data
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import P, load_sts
from wildfire.paper.sts_spread import load_spread


def burn_maps(net, x, doy, t: int):
    """-> (P(t), b_obs(t)) je Pixel (B, 169) am Tag t nach dem Stichtag (t >= 1)."""
    pix = net.encode_input(x, doy)
    obs, a, k, logD = net.params(x, doy, pix)
    with torch.autocast(device_type=x.device.type, enabled=False):
        log_exo = net.exogenous(obs, a, k, logD)
        B, D, N = obs.shape
        u = torch.arange(D - 1, -1, -1, device=x.device)                       # Alter am Stichtag je Fenstertag
        pw = logD[:, :, u + t].exp()                                            # (B, N, D)
        b_obs = 1 - torch.prod(1 - obs.transpose(1, 2) * pw, dim=2)
        c = net.K * net.K // 2
        logS = log_exo.new_zeros(B, N)
        S = log_exo.new_ones(B, N)
        b_pred = log_exo.new_zeros(B, N)
        fs = []
        for s in range(t):                                                      # nach Schritt s: Zustand an Tag s + 1
            ks = k[:, s] if k.dim() == 4 else k
            lp = torch.log1p(-(ks * net._unfold(b_pred)).clamp(max=1 - 1e-6))
            l1mh = log_exo[:, s] + lp.sum(1) - lp[:, c]
            fs.append(S * -torch.expm1(l1mh))
            logS = logS + l1mh
            S = torch.exp(logS)
            b_pred = (torch.stack(fs, -1) * logD[..., :s + 1].flip(-1).exp()).sum(-1).clamp(max=1.0)
    return 1 - (1 - b_obs) * (1 - b_pred), b_obs


def scores(paths, X, Xg, rows, L, dt_, doyD, m_t, s_t, device, cube_rates, bs=64):
    """Mittel ueber die Seeds, je (n, 13, 13): P(t) und b_obs(t) mit t = dt + 1, Fenster wie patch_map_eval."""
    LC = X.shape[1]
    e = L - dt_ - 2
    acc_p = acc_o = 0.0
    for p in paths:
        a = torch.load(p, map_location="cpu").get("args", {})
        if "spread" in a:
            net = load_spread(p, device)
        else:
            net = load_sts(p, device)
            net._joint_args = a
            set_bias(net, "trained", cube_rates)
        if getattr(net, "head_kind", "") != "contagion":
            raise SystemExit(f"ABBRUCH: {p} hat keinen Contagion-Kopf (keine Brennkurve)")
        out_p, out_o = [], []
        with torch.no_grad():
            for i in range(0, rows.size, bs):
                r = rows[i:i + bs]
                x = (windows(Xg, r, np.full(r.size, e)).to(device).float() - m_t) / s_t
                doy = torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pb, po = burn_maps(net, x, doy, dt_ + 1)
                out_p.append(pb.view(len(r), P, P).cpu().numpy())
                out_o.append(po.view(len(r), P, P).cpu().numpy())
        acc_p = acc_p + np.concatenate(out_p) / len(paths)
        acc_o = acc_o + np.concatenate(out_o) / len(paths)
    return np.clip(acc_p, 1e-7, 1 - 1e-7), np.clip(acc_o, 1e-7, 1 - 1e-7)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--config", required=True)
    ap_.add_argument("--cache", default="cache_G")
    ap_.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap_.add_argument("--model", required=True, help="name=glob (Contagion-Checkpoints)")
    ap_.add_argument("--cube-dir", default="paper_pixel")
    ap_.add_argument("--tag", default="")
    args = ap_.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"patch_maps{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    name, pattern = args.model.split("=", 1)
    f_b, f_o = out / f"scores_{name}_burn.npz", out / f"scores_{name}_burnobs.npz"
    for f in (f_b, f_o):
        if f.exists():
            raise SystemExit(f"ABBRUCH: {f} existiert")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"ABBRUCH: keine Checkpoints fuer {name}: {pattern}")
    t0 = time.time()
    d, X, fire, idx, doyD = load_data(exp, args.cache, args.target)
    L = X.shape[1] + targets.shift(args.target)
    print(f"[pb] {name}: " + ", ".join(Path(p).name for p in paths), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    ev_tr = np.load(exp / "derived_data" / args.cube_dir / "train_event.npy").astype(bool)
    cube_rates = (float(ev_tr.mean()), float(ev_tr.reshape(len(ev_tr), -1).any(1).mean()))
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    res_b, res_o = {}, {}
    for dt_ in DTS:
        for s in ("val", "test"):
            res_b[f"{s}_{dt_}"], res_o[f"{s}_{dt_}"] = scores(paths, X, Xg, idx[s], L, dt_, doyD, m_t, s_t, device, cube_rates)
        yt = fire[idx["test"], L - 1]
        print(f"[pb] {name} dt {dt_}: Ziel Tag t, test alle Pixel PR-AUC brennt {ap(res_b[f'test_{dt_}'].ravel(), yt.ravel()):.3f}"
              f" / nur Fenster {ap(res_o[f'test_{dt_}'].ravel(), yt.ravel()):.3f}, Mitte "
              f"{ap(res_b[f'test_{dt_}'][:, CENTRE, CENTRE], yt[:, CENTRE, CENTRE]):.3f} ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(f_b, **{k: v.astype(np.float32) for k, v in res_b.items()})
    np.savez_compressed(f_o, **{k: v.astype(np.float32) for k, v in res_o.items()})
    write_manifest(cfg, f"p68_patch_burn_{name}{args.tag}", args.config, {"model": args.model, "cache": args.cache,
                                                                           "target": args.target,
                                                                           "outputs": [str(f_b), str(f_o)]})


if __name__ == "__main__":
    main()

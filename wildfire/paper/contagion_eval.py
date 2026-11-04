"""Zerlegung des Contagion-STS: wie viel Risiko kommt aus Zuendung, wie viel aus Ausbreitung?

Im Modell gilt fuer den Ausschnitt exakt log(1 - h_exo) = log(1 - a) + sum log(1 - k b_obs);
der kumulative Hazard zerfaellt also additiv,

    Lambda(tau) = Lambda_Zuendung(tau) + Lambda_Ausbreitung(tau).

Gemessen auf den zufaelligen Stichtagen der Testjahre (surv_eval):
  - Anteil der Ausbreitung am erwarteten kumulativen Hazard je tau, ueber alle
    Stichtage und getrennt nach "Feuer im 30-Tage-Fenster" ja/nein
  - empirischer Vergleich: Anteil der Ereignisse bis tau, bei denen es im
    Fenster schon im Ausschnitt brannte (Persistenz-Proxy)
  - Pseudo-Ursache je Ereignis: Anteil der Ausbreitung am Hazard des
    Ereignistags, gemittelt ueber Ereignisse mit / ohne Feuer im Fenster
  - gelernte Parameter: Persistenz (Kernmitte), mittlerer Nachbarkern, erwartete Brenndauer (Tage)
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import load_sts

TAUS = (1, 7, 30, 100)


@torch.no_grad()
def components(net, X, doy, m_t, s_t, device, bs=64):
    """-> Lambda_ign, Lambda_spread (n, H+1) des Ausschnitts, Hazard-Anteile je Tag (n, H), Parameter."""
    li, ls, sh, cen, nb, burn = [], [], [], [], [], []
    c = net.K * net.K // 2
    for a in range(0, X.shape[0], bs):
        x = (X[a:a + bs].float() - m_t) / s_t
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            obs, aa, k, logD = net.params(x, torch.from_numpy(doy[a:a + bs]).to(device))
        log_exo = net.exogenous(obs, aa, k, logD)                                  # (b, H, N)
        l_ign = torch.log1p(-aa.clamp(max=1 - 1e-6)).sum(2)                     # (b, H)
        l_spr = log_exo.sum(2) - l_ign
        li.append(F.pad(-l_ign.cumsum(1), (1, 0)).double().cpu().numpy())
        ls.append(F.pad(-l_spr.cumsum(1), (1, 0)).double().cpu().numpy())
        sh.append((l_spr / (l_spr + l_ign).clamp(max=-1e-30)).float().cpu().numpy())
        cen.append(k[:, c].mean(1).cpu().numpy())
        nb.append(((k.sum(1) - k[:, c]) / (k.shape[1] - 1)).mean(1).cpu().numpy())
        burn.append(logD.exp().sum(-1).mean(1).cpu().numpy())                # erwartete Brenndauer in Tagen
    cat = np.concatenate
    return cat(li), cat(ls), cat(sh), {"persistence_k": cat(cen), "neighbour_k": cat(nb), "burn_days": cat(burn)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--pattern", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_d128_contagion_s*_s*.pt")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = exp / "derived_data" / args.surv_dir
    me = dict(np.load(sd / "eval_meta.npz"))
    ev, t, fw = me["ev_patch"].astype(bool), me["t_patch"].astype(int), me["fire_win_patch"].astype(bool)
    X = to_device_once(np.load(sd / "eval_X.npy"), device, "eval_X")
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    emp = {str(k): float(fw[ev & (t <= k)].mean()) for k in TAUS}
    print("[ce] empirisch: Anteil der Ereignisse bis tau mit Feuer im Fenster  "
          + "  ".join(f"tau {k}: {emp[str(k)]:.3f}" for k in TAUS), flush=True)
    res = {"empirical_share_fire_in_window": emp, "models": {}}
    for f in sorted(glob.glob(args.pattern)):
        net = load_sts(f, device)
        Li, Ls, share, par = components(net, X, me["doy"], m_t, s_t, device)
        r = {"spread_share": {}, "spread_share_fire_in_window": {}, "spread_share_no_fire": {}}
        for k in TAUS:
            tot = Li[:, k] + Ls[:, k]
            r["spread_share"][str(k)] = float(Ls[:, k].sum() / tot.sum())
            r["spread_share_fire_in_window"][str(k)] = float(Ls[fw, k].sum() / tot[fw].sum())
            r["spread_share_no_fire"][str(k)] = float(Ls[~fw, k].sum() / tot[~fw].sum())
        idx = np.flatnonzero(ev)
        s_ev = share[idx, t[idx] - 1]
        r["event_day_spread_share"] = {"events_fire_in_window": float(s_ev[fw[idx]].mean()),
                                       "events_no_fire": float(s_ev[~fw[idx]].mean())}
        r["params"] = {k: {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
                       for k, v in par.items()}
        res["models"][Path(f).stem] = r
        print(f"[ce] {Path(f).name}: Ausbreitungsanteil am erwarteten Hazard  "
              + "  ".join(f"tau {k}: {r['spread_share'][str(k)]:.3f}" for k in TAUS), flush=True)
        print(f"      Stichtage mit Feuer im Fenster  " + "  ".join(f"tau {k}: {r['spread_share_fire_in_window'][str(k)]:.3f}" for k in TAUS)
              + "   ohne  " + "  ".join(f"{r['spread_share_no_fire'][str(k)]:.4f}" for k in TAUS), flush=True)
        print(f"      Ereignistag: Ausbreitungsanteil bei Ereignissen mit Feuer im Fenster "
              f"{r['event_day_spread_share']['events_fire_in_window']:.3f}, ohne {r['event_day_spread_share']['events_no_fire']:.3f}; "
              f"Persistenz k {r['params']['persistence_k']['mean']:.3f}, Nachbar k {r['params']['neighbour_k']['mean']:.4f}, "
              f"Brenndauer {r['params']['burn_days']['mean']:.1f} Tage", flush=True)
    write_manifest(cfg, f"p21_contagion_decomposition{args.tag}", args.config, {
        "landmarks": args.surv_dir, "pattern": args.pattern,
        "decomposition": "Lambda = Lambda_Zuendung + Lambda_Ausbreitung (beobachtetes Feuer), Ausschnitt, exakt additiv",
        "results": res,
    })


if __name__ == "__main__":
    main()

"""Regions-Simulation (globales Mean-Field und Monte Carlo) fuer mehrere Contagion-Modelle, auch mit Kern je Tag.

Wie contagion_mc_map.py (bleibt unveraendert), aber
  - mehrere Modelle in einem Lauf (--models name=glob;...), Checkpoints von sts_dctm/sts_extreme (load_sts) oder
    sts_spread (load_spread, gerichteter Kern, optional Wetter-Oracle);
  - Kern je Tag (k (T, KK, N)) im Mean-Field und in den Pfaden;
  - Oracle-Modelle bekommen das beobachtete Wetter der Tage L + 1 .. L + 30 je Pixelfenster (kf = 30).
Die Beispielpixel kommen wie dort aus teaser_oracle/summary.json; die Pfadauswahl ist fuer alle Modelle gleich
(gleiche Zufallszahlen je Modell).

Ausgabe: figures/teaser_mc2<tag>/<datum>_<modell>_<panel>.png (mf, mc, path0..2; observed einmal je Datum),
metrics/teaser_mc2<tag>/summary.md/.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.contagion_mc_map import T, save_panel, unfold
from wildfire.paper.contagion_spread import WX
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.risk_map import load_slab
from wildfire.paper.risk_map_plot import borders, grid_geo
from wildfire.paper.sts_dctm import P, load_sts
from wildfire.paper.sts_spread import load_spread
from wildfire.paper.surv_prepare import HALF
from wildfire.paper.teaser_spread import arrival_pred, box_slice, box_stats, load

C = P * P // 2
DEFAULT_MODELS = ("cg_scratch=/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt;"
                  "cg_ell11=/data/Wildfire/checkpoints/paper/sts_spread_cg_ell11_s?_s?.pt;"
                  "cg_ell11fw=/data/Wildfire/checkpoints/paper/sts_spread_cg_ell11fw_s?_s?.pt")


def load_net(path, device):
    a = torch.load(path, map_location="cpu").get("args", {})
    return load_spread(path, device) if "spread" in a else load_sts(path, device)


@torch.no_grad()
def pixel_params(net, slab, fslab, cells, pad, doy, m_t, s_t, mf_t, sf_t, device, bs=128):
    """-> a (T, n), k (KK, n) oder (T, KK, n), logD (n, A) des Mittelpixels je zentriertem Fenster."""
    fd = getattr(net, "future_days", 0)
    A_, K_, D_ = [], [], []
    for i in range(0, len(cells), bs):
        cb = cells[i:i + bs]
        win = lambda s: np.stack([s[:, :, r + pad - HALF:r + pad + HALF + 1, c + pad - HALF:c + pad + HALF + 1]  # noqa: E731
                                  for r, c in cb])
        x = (torch.from_numpy(win(slab)).to(device).float() - m_t) / s_t
        d = torch.full((len(cb),), float(doy), device=device)
        if fd:
            xf = (torch.from_numpy(win(fslab)[:, :fd]).to(device).float() - mf_t) / sf_t
            net.set_future(xf, torch.full((len(cb),), fd, device=device, dtype=torch.long))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, aa, k, logD = net.params(x, d)
        if hasattr(net, "set_future"):
            net.set_future(None, None)
        A_.append(aa[:, :T, C].float())
        K_.append((k[:, :T, :, C] if k.dim() == 4 else k[:, :, C]).float())
        D_.append(logD[:, C].float())
    k = torch.cat(K_, 0)
    return torch.cat(A_, 0).t(), (k.permute(1, 2, 0) if k.dim() == 3 else k.t()), torch.cat(D_, 0)


@torch.no_grad()
def simulate(K, a, k, logD, obs, shape, M, gen):
    """Wie contagion_mc_map.simulate, k (KK, N) oder je Tag (T, KK, N)."""
    KK = K * K
    c = KK // 2
    kt = (lambda s: k[s]) if k.dim() == 3 else (lambda s: k)               # noqa: E731
    Dd, N = obs.shape
    idx = torch.arange(T, device=a.device)[:, None] + torch.arange(Dd - 1, -1, -1, device=a.device)[None]
    pw = logD[:, idx].exp().permute(1, 2, 0)
    b_obs = 1 - torch.prod(1 - obs[None] * pw, dim=1)
    kk = k if k.dim() == 3 else k[None]
    log_exo = torch.log1p(-a.clamp(max=1 - 1e-6)) + torch.log1p(-(kk * unfold(b_obs, K, shape)).clamp(max=1 - 1e-6)).sum(1)
    z = torch.zeros(N, device=a.device)
    logS, S, b_pred, fs, out = z.clone(), torch.ones(N, device=a.device), z.clone(), [], [z.clone()]
    for s in range(T):
        lp = torch.log1p(-(kt(s) * unfold(b_pred[None], K, shape)[0]).clamp(max=1 - 1e-6))
        l1mh = log_exo[s] + lp.sum(0) - lp[c]
        fs.append(S * -torch.expm1(l1mh))
        logS = logS + l1mh
        S = torch.exp(logS)
        b_pred = (torch.stack(fs, -1) * logD[:, :s + 1].flip(-1).exp()).sum(-1).clamp(max=1.0)
        out.append(logS)
    fired = torch.full((M, N), -1, dtype=torch.long, device=a.device)
    lD = logD[None].expand(M, -1, -1)
    for s in range(T):
        on = fired >= 0
        age = (s - fired - 1).clamp(min=0, max=logD.shape[-1] - 1)
        b = torch.where(on, lD.gather(2, age[..., None])[..., 0].exp(), torch.zeros_like(fired, dtype=torch.float32))
        lp = torch.log1p(-(kt(s)[None] * unfold(b, K, shape)).clamp(max=1 - 1e-6))
        l1mh = log_exo[s][None] + lp.sum(1) - lp[:, c]
        new = (~on) & (torch.rand(l1mh.shape, generator=gen, device=l1mh.device) < -torch.expm1(l1mh))
        fired = torch.where(new, torch.full_like(fired, s), fired)
    return torch.stack(out, -1), fired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default=DEFAULT_MODELS, help="name=glob;name=glob")
    ap.add_argument("--ref", default="teaser_cgs")
    ap.add_argument("--pixels", default="/data/Wildfire/metrics/teaser_oracle/summary.json")
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--paths", type=int, default=200)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out_m, out_f = exp / "metrics" / f"teaser_mc2{args.tag}", exp / "figures" / f"teaser_mc2{args.tag}"
    for o in (out_m, out_f):
        if o.exists():
            raise SystemExit(f"ABBRUCH: {o} existiert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for spec in args.models.split(";"):
        nm, pat = spec.split("=", 1)
        files = sorted(glob.glob(pat))
        if not files:
            raise SystemExit(f"ABBRUCH: keine Checkpoints fuer {nm}: {pat}")
        models[nm] = (files, [load_net(f, device) for f in files])
        print(f"[mc2] {nm}: " + ", ".join(Path(f).name for f in files), flush=True)
    geo = grid_geo(cfg)
    rings = borders(exp / "artifacts" / "ne_50m_admin_1_states_provinces.geojson")
    cube = CubeReader(exp)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    mf_t, sf_t = m_t[:, :, WX], s_t[:, :, WX]
    idx = {d: i for i, d in enumerate(cube.dates)}
    pix = json.loads(Path(args.pixels).read_text())
    out_m.mkdir(parents=True); out_f.mkdir(parents=True)
    res = {}
    md = ["# Regions-Simulation: Contagion-Varianten", "",
          f"Je Modell alle Seeds x {args.paths} MC-Pfade, Rand {args.margin} px, Schwelle p = {args.p}. mf = Mean-Field auf "
          "der Region, mc = Anteil der Pfade. Oracle-Modelle mit dem beobachteten Wetter der 30 Folgetage. Rohe F(tau).", ""]
    for key, ent in pix.items():
        date, box = key.split(":")[0], tuple(json.loads(key.split(":", 1)[1]))
        L = idx[date]
        d = load(exp, args.ref, date)
        sl = box_slice(box, geo, d["mask"].shape)
        r0, r1 = max(0, sl[0].start - args.margin), min(cube.H, sl[0].stop + args.margin)
        c0, c1 = max(0, sl[1].start - args.margin), min(cube.W, sl[1].stop + args.margin)
        shape = (r1 - r0, c1 - c0)
        cells = np.array([(r, c) for r in range(r0, r1) for c in range(c0, c1)], np.int64)
        pad = HALF + 1
        slab = load_slab(cube, L, pad)
        fslab = np.ascontiguousarray(load_slab(cube, L + 30, pad)[:, WX])     # Tage L + 1 .. L + 30
        doy = dt.date.fromisoformat(date).timetuple().tm_yday
        loc = (slice(sl[0].start - r0, sl[0].stop - r0), slice(sl[1].start - c0, sl[1].stop - c0))
        dl = {"mask": d["mask"][sl], "win": d["win"][sl], "first": d["first"][sl]}
        allsl = (slice(None), slice(None))
        b = ent["pixels"]["b"]
        br, bc = b["row"] - r0, b["col"] - c0
        lon0, dlon, lat0, dla = geo
        extent = (lon0 + dlon * sl[1].start, lon0 + dlon * sl[1].stop, lat0 + dla * sl[0].stop, lat0 + dla * sl[0].start)
        save_panel(out_f / f"{date}_observed.png", np.where(d["first"][sl] > 0, d["first"][sl], 31), d, sl, extent, rings,
                   box, (b["row"], b["col"]), geo)
        e = res[key] = {"b": {"observed_day": b["observed_day"], "distance_px": b["distance_px"]}, "models": {}}
        md += [f"## {date}, Ausschnitt {box}", "", f"Pixel b: beobachtet Tag {b['observed_day']}, Abstand {b['distance_px']:.1f} px", "",
               "| Modell | Variante | AUC 7 d | AUC 30 d | Recall 7 d | Praez. 7 d | Recall 30 d | Praez. 30 d | b: F(8) | b: F(30) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        hits = []
        for nm, (files, nets) in models.items():
            gen = torch.Generator(device=device).manual_seed(args.seed)
            S_mf, fired_all = 0.0, []
            for net in nets:
                a, k, logD = pixel_params(net, slab, fslab, cells, pad, doy, m_t, s_t, mf_t, sf_t, device)
                obs = torch.from_numpy(slab[-net.obs_days:, 0, pad + r0:pad + r1, pad + c0:pad + c1] > 0).float().to(device)
                ls_mf, fired = simulate(net.K, a, k, logD, obs.reshape(net.obs_days, -1), shape, args.paths, gen)
                S_mf = S_mf + ls_mf.exp().cpu().numpy() / len(nets)
                fired_all.append(fired.cpu().numpy())
            fired = np.concatenate(fired_all)
            day = np.where(fired >= 0, fired + 1, 31).reshape(-1, *shape)
            F_mf = (1 - S_mf[:, 1:]).reshape(*shape, T)
            F_mc = np.stack([(day <= t).mean(0) for t in range(1, T + 1)], -1)
            em = e["models"][nm] = {}
            for vn, Fv in (("mf", F_mf), ("mc", F_mc)):
                st = box_stats({**dl, "F": Fv[loc]}, allsl, [args.p])
                em[vn] = {**st, "b_F8": float(Fv[br, bc, 7]), "b_F30": float(Fv[br, bc, 29])}
                g = lambda kk: f"{st[kk]:.3f}" if kk in st else "–"                         # noqa: E731
                md.append(f"| {nm} | {vn} | {g('auc_7')} | {g('auc_30')} | " + " | ".join(
                    g(f"p{args.p}_t{t}_{m}") for t in (7, 30) for m in ("recall", "precision"))
                    + f" | {Fv[br, bc, 7]:.3f} | {Fv[br, bc, 29]:.3f} |")
            obs_box = (dl["first"] > 0) & dl["mask"] & ~dl["win"]
            area = (day[:, loc[0], loc[1]] <= 30) & dl["mask"] & ~dl["win"]
            em["paths"] = {"hit_b_by_8": float((day[:, br, bc] <= 8).mean()), "hit_b_by_30": float((day[:, br, bc] <= 30).mean()),
                           "n": int(day.shape[0]), "burnt_px_30_mean": float(area.sum((1, 2)).mean()),
                           "observed_px_30": int(obs_box.sum())}
            hits.append(f"{nm}: b bis Tag 8 {em['paths']['hit_b_by_8']:.1%}, bis 30 {em['paths']['hit_b_by_30']:.1%}; "
                        f"verbrannte Pixel je Pfad bis 30 d {em['paths']['burnt_px_30_mean']:.0f} (beobachtet {int(obs_box.sum())})")
            panels = {"mf": arrival_pred(F_mf[loc], args.p), "mc": arrival_pred(F_mc[loc], args.p)}
            rng = np.random.default_rng(args.seed)
            for j, pth in enumerate(rng.choice(day.shape[0], 3, replace=False)):
                panels[f"path{j}"] = day[pth][loc]
            for pn, arr in panels.items():
                save_panel(out_f / f"{date}_{nm}_{pn}.png", arr, d, sl, extent, rings, box, (b["row"], b["col"]), geo)
        md += [""] + [f"- {h}" for h in hits] + [""]
        print(f"[mc2] {date}: " + " | ".join(hits), flush=True)
    (out_m / "summary.json").write_text(json.dumps(res, indent=1))
    (out_m / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p51_contagion_mc_map2{args.tag}", args.config, {
        "models": {k: v[0] for k, v in models.items()}, "paths_per_seed": args.paths, "margin": args.margin,
        "p": args.p, "seed": args.seed, "outputs": [str(out_m), str(out_f)]})


if __name__ == "__main__":
    main()

"""Contagion-STS als Simulation ueber eine ganze Region statt je 13 x 13-Ausschnitt (fuer die Teaser-Ereignisse).

contagion.py, risk_map.py und region_eval.py bleiben unveraendert. In risk_map rechnet jedes Fenster die Ausbreitung
nur innerhalb seines Ausschnitts (Mean-Field), Feuer weiter als ~6 Pixel kann einen Pixel also nie erreichen.
Hier bekommt jeder Pixel der Region seine Parameter (Zuendung a(t), Kern k, Brenndauer D) aus dem Fenster, das auf
ihn zentriert ist, und die Ausbreitung laeuft danach auf dem ganzen Regionsgitter (Rand +margin Pixel):

  mf_global   Mean-Field-Rollout wie Contagion.rollout, aber auf der Region (Ausbreitung ueber beliebige Distanz)
  mc          M Monte-Carlo-Pfade je Seed: jeder Pixel zuendet am Tag s mit h_i(s) aus Zuendung, beobachtetem Feuer
              und dem simulierten Brennzustand der Nachbarn (wie region_eval.mc_region_logS, ohne Rao-Blackwell);
              F_mc(tau) = Anteil der Pfade mit erstem Feuer bis tau; dazu einzelne Pfade als Ankunftskarte

Ausgabe:
  figures/teaser_mc/<datum>_<panel>.png   einzelne Kartenpanels ohne Beschriftung (beobachtet, mf_patch, mf_global,
                                          mc, path0..2), Farben wie teaser_spread; Kreis = Pixel b aus teaser_oracle
  metrics/teaser_mc<tag>/summary.md/.json box_stats je Variante, F(8)/F(30) an Pixel b und Anteil der Pfade, die b treffen
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from wildfire.audit.common import assert_readonly, load_config, write_manifest  # noqa: E402
from wildfire.data.dataset import CubeReader  # noqa: E402
from wildfire.paper.reconstruct import cache_norm  # noqa: E402
from wildfire.paper.risk_map import load_slab  # noqa: E402
from wildfire.paper.risk_map_plot import borders, grid_geo  # noqa: E402
from wildfire.paper.sts_dctm import P, load_sts  # noqa: E402
from wildfire.paper.surv_prepare import HALF  # noqa: E402
from wildfire.paper.teaser_spread import (BINS, COLORS, arrival_pred, box_slice, box_stats, draw_arrival,  # noqa: E402
                                          load)

T = 30
C = P * P // 2


@torch.no_grad()
def pixel_params(net, slab, cells, pad, doy, m_t, s_t, device, bs=256):
    """Parameter des Mittelpixels je zentriertem Fenster -> a (T, n), k (KK, n), logD (n, A)."""
    A_, K_, D_ = [], [], []
    for i in range(0, len(cells), bs):
        cb = cells[i:i + bs]
        x = np.stack([slab[:, :, r + pad - HALF:r + pad + HALF + 1, c + pad - HALF:c + pad + HALF + 1] for r, c in cb])
        x = (torch.from_numpy(x).to(device).float() - m_t) / s_t
        d = torch.full((len(cb),), float(doy), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, aa, k, logD = net.params(x, d)
        A_.append(aa[:, :T, C].float()); K_.append(k[:, :, C].float()); D_.append(logD[:, C].float())
    return torch.cat(A_, 0).t(), torch.cat(K_, 0).t(), torch.cat(D_, 0)


def unfold(b, K, shape):
    """(M, N) -> (M, KK, N): Brennzustand der Nachbarn je Empfaenger auf dem Regionsgitter (Rand 0)."""
    return F.unfold(b.reshape(-1, 1, *shape), K, padding=K // 2)


@torch.no_grad()
def simulate(net, a, k, logD, obs, shape, M, gen):
    """-> log S_mf (N, T+1) global Mean-Field, erstes Feuer der MC-Pfade (M, N) als Schritt 0..T-1 oder -1."""
    K, KK = net.K, net.K * net.K
    c = KK // 2
    Dd, N = obs.shape
    idx = torch.arange(T, device=a.device)[:, None] + torch.arange(Dd - 1, -1, -1, device=a.device)[None]   # (T, Dd)
    pw = logD[:, idx].exp().permute(1, 2, 0)                                       # (T, Dd, N)
    b_obs = 1 - torch.prod(1 - obs[None] * pw, dim=1)                              # (T, N)
    log_exo = torch.log1p(-a.clamp(max=1 - 1e-6)) + torch.log1p(
        -(k[None] * unfold(b_obs, K, shape)).clamp(max=1 - 1e-6)).sum(1)          # (T, N)
    # Mean-Field wie Contagion.rollout
    logS, S, b_pred, fs, out = torch.zeros(N, device=a.device), torch.ones(N, device=a.device), \
        torch.zeros(N, device=a.device), [], [torch.zeros(N, device=a.device)]
    for s in range(T):
        lp = torch.log1p(-(k * unfold(b_pred[None], K, shape)[0]).clamp(max=1 - 1e-6))
        l1mh = log_exo[s] + lp.sum(0) - lp[c]
        fs.append(S * -torch.expm1(l1mh))
        logS = logS + l1mh
        S = torch.exp(logS)
        Dk = logD[:, :s + 1].flip(-1).exp()
        b_pred = (torch.stack(fs, -1) * Dk).sum(-1).clamp(max=1.0)
        out.append(logS)
    ls_mf = torch.stack(out, -1)
    # Monte Carlo
    fired = torch.full((M, N), -1, dtype=torch.long, device=a.device)
    lD = logD[None].expand(M, -1, -1)
    for s in range(T):
        on = fired >= 0
        age = (s - fired - 1).clamp(min=0, max=logD.shape[-1] - 1)
        b = torch.where(on, lD.gather(2, age[..., None])[..., 0].exp(), torch.zeros_like(fired, dtype=torch.float32))
        lp = torch.log1p(-(k[None] * unfold(b, K, shape)).clamp(max=1 - 1e-6))       # (M, KK, N)
        l1mh = log_exo[s][None] + lp.sum(1) - lp[:, c]
        h = -torch.expm1(l1mh)
        new = (~on) & (torch.rand(h.shape, generator=gen, device=h.device) < h)
        fired = torch.where(new, torch.full_like(fired, s), fired)
    return ls_mf, fired


def save_panel(path, a, d, sl, extent, rings, box, b_rc, geo):
    cmap, norm = ListedColormap(COLORS), BoundaryNorm(BINS, len(COLORS))
    w = extent[1] - extent[0]
    h = (extent[3] - extent[2]) / np.cos(np.deg2rad(0.5 * (box[2] + box[3])))
    fig, ax = plt.subplots(figsize=(3.0, 3.0 * h / w), dpi=200)
    draw_arrival(ax, a, d, sl, extent, rings, cmap, norm, box)
    lon0, dl, lat0, dla = geo
    ax.plot(lon0 + dl * (b_rc[1] + 0.5), lat0 + dla * (b_rc[0] + 0.5), marker="o", ms=5, mfc="none", mec="#1F4E79", mew=1.4,
            zorder=7)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pattern", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt")
    ap.add_argument("--ref", default="teaser_cgs", help="risk_map-npz des Patch-Mean-Fields (Beobachtung, Maske)")
    ap.add_argument("--pixels", default="/data/Wildfire/metrics/teaser_oracle/summary.json")
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--paths", type=int, default=200, help="MC-Pfade je Seed")
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out_m, out_f = exp / "metrics" / f"teaser_mc{args.tag}", exp / "figures" / f"teaser_mc{args.tag}"
    for o in (out_m, out_f):
        if o.exists():
            raise SystemExit(f"ABBRUCH: {o} existiert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(args.pattern))
    nets = [load_sts(f, device) for f in files]
    if not nets or any(getattr(n, "head_kind", "") != "contagion" for n in nets):
        raise SystemExit("ABBRUCH: keine Contagion-Checkpoints")
    print(f"[mc] {len(nets)} Checkpoints: " + ", ".join(Path(f).name for f in files), flush=True)
    geo = grid_geo(cfg)
    rings = borders(exp / "artifacts" / "ne_50m_admin_1_states_provinces.geojson")
    cube = CubeReader(exp)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    idx = {d: i for i, d in enumerate(cube.dates)}
    pix = json.loads(Path(args.pixels).read_text())
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out_m.mkdir(parents=True); out_f.mkdir(parents=True)
    res, md = {}, ["# Contagion-STS: Simulation ueber die ganze Region", "",
                   f"{len(nets)} Seeds x {args.paths} MC-Pfade, Rand {args.margin} px, Schwelle p = {args.p}. "
                   "mf_patch = risk_map (Ausbreitung nur im 13 x 13-Fenster), mf_global = Mean-Field auf der Region, "
                   "mc = Anteil der Pfade. Rohe F(tau).", ""]
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
        doy = dt.date.fromisoformat(date).timetuple().tm_yday
        S_mf, fired_all = 0.0, []
        for net in nets:
            a, k, logD = pixel_params(net, slab, cells, pad, doy, m_t, s_t, device)
            obs = torch.from_numpy(slab[-net.obs_days:, 0, pad + r0:pad + r1, pad + c0:pad + c1] > 0).float().to(device).reshape(net.obs_days, -1)
            ls_mf, fired = simulate(net, a, k, logD, obs, shape, args.paths, gen)
            S_mf = S_mf + ls_mf.exp().cpu().numpy() / len(nets)
            fired_all.append(fired.cpu().numpy())
        fired = np.concatenate(fired_all)                                         # (M_total, N)
        day = np.where(fired >= 0, fired + 1, 31).reshape(-1, *shape)            # erster Feuertag je Pfad
        F_mf = (1 - S_mf[:, 1:]).reshape(*shape, T)
        F_mc = np.stack([(day <= t).mean(0) for t in range(1, T + 1)], -1)
        loc = (slice(sl[0].start - r0, sl[0].stop - r0), slice(sl[1].start - c0, sl[1].stop - c0))
        full = lambda arr: arr[loc]                                               # noqa: E731
        dl = {"mask": d["mask"][sl], "win": d["win"][sl], "first": d["first"][sl]}
        allsl = (slice(None), slice(None))
        variants = {"mf_patch": d["F"][sl], "mf_global": full(F_mf), "mc": full(F_mc)}
        b = ent["pixels"]["b"]
        br, bc = b["row"] - r0, b["col"] - c0
        e = res[key] = {"box": {}, "b": {"observed_day": b["observed_day"], "distance_px": b["distance_px"]}}
        md += [f"## {date}, Ausschnitt {box}", "", f"Pixel b: beobachtet Tag {b['observed_day']}, Abstand {b['distance_px']:.1f} px", "",
               "| Variante | AUC 7 d | AUC 30 d | Recall 7 d | Praez. 7 d | Recall 30 d | Praez. 30 d | b: F(8) | b: F(30) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for name, Fv in variants.items():
            st = box_stats({**dl, "F": Fv}, allsl, [args.p])
            Fb = d["F"][b["row"], b["col"]] if name == "mf_patch" else (F_mf if name == "mf_global" else F_mc)[br, bc]
            e["box"][name] = {**st, "b_F8": float(Fb[7]), "b_F30": float(Fb[29])}
            g = lambda kk: f"{st[kk]:.3f}" if kk in st else "–"                   # noqa: E731
            md.append(f"| {name} | {g('auc_7')} | {g('auc_30')} | " + " | ".join(g(f"p{args.p}_t{t}_{m}") for t in (7, 30)
                      for m in ("recall", "precision")) + f" | {Fb[7]:.3f} | {Fb[29]:.3f} |")
        hit8, hit30 = float((day[:, br, bc] <= 8).mean()), float((day[:, br, bc] <= 30).mean())
        e["b"].update({"paths_hit_by_8": hit8, "paths_hit_by_30": hit30, "n_paths": int(day.shape[0])})
        md += ["", f"Pfade, in denen b brennt: bis Tag 8 {hit8:.1%}, bis Tag 30 {hit30:.1%} (von {day.shape[0]})", ""]
        lon0, dlon, lat0, dla = geo
        extent = (lon0 + dlon * sl[1].start, lon0 + dlon * sl[1].stop, lat0 + dla * sl[0].stop, lat0 + dla * sl[0].start)
        obs_day = np.where(d["first"][sl] > 0, d["first"][sl], 31)
        panels = {"observed": obs_day, **{n: arrival_pred(v, args.p) for n, v in variants.items()}}
        rng = np.random.default_rng(args.seed)
        for j, pth in enumerate(rng.choice(day.shape[0], 3, replace=False)):
            panels[f"path{j}"] = full(day[pth])
        for n, arr in panels.items():
            save_panel(out_f / f"{date}_{n}.png", arr, d, sl, extent, rings, box, (b["row"], b["col"]), geo)
        print(f"[mc] {date}: " + json.dumps(e, default=float)[:600], flush=True)
    (out_m / "summary.json").write_text(json.dumps(res, indent=1))
    (out_m / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p49_contagion_mc_map{args.tag}", args.config, {
        "checkpoints": files, "paths_per_seed": args.paths, "margin": args.margin, "p": args.p, "seed": args.seed,
        "outputs": [str(out_m), str(out_f)]})


if __name__ == "__main__":
    main()

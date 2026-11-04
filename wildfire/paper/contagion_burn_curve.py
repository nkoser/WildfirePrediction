"""Gelernte Brenndauer der Contagion-STS gegen die Feuer-Episoden in den Daten.

Modell (aus den Checkpoints, auf Paper-Test-Fenstern, Eingabe endet bei D-1):
  D_j(u) = exp(-s_j C(u)) je Pixel; Quantile ueber alle Pixel und ueber Pixel mit Feuer in den letzten 30 Tagen,
  erwartete Dauer sum_u D_j(u). Varianten: cg_ref (Alter ab jeder Detektion, slurm 166) und cg_episode (Alter ab
  Episodenbeginn, slurm 215); D bedeutet dort jeweils etwas anderes.
Daten (Feuerkanal des Paper-Caches, 61 Tage je Ausschnitt, alle Pixel):
  redetect(u)  P(Detektion an Tag t+u | Detektion an Tag t): Gegenstueck zu D im Detektionsmodell (cg_ref)
  episode(u)   P(Episodendauer > u), Episoden = Detektionen mit Luecken <= gap Tagen, nur Episoden, die
               mindestens gap+1 Tage vor dem Fensterende enden und nicht am Fensteranfang beginnen
               (Dauer = letzte - erste Detektion + 1); Gegenstueck zu D im Episodenmodell (cg_episode).
               Lange Episoden sind im 61-Tage-Fenster unterrepraesentiert (Zensierung) - untere Schranke.
Dazu Paper-PR-AUC je dt und Survival-Metriken aus p35_sts_extreme_<name>_s<seed>.
Ausgaben NEU: metrics/contagion_burn_curve<tag>/{summary.md, curves.json, burn_curve.png}, Manifest
p48_contagion_burn_curve<tag>.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import sts_extreme
from wildfire.paper.compare import DTS
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import doy_of

U_MAX = 60


def data_curves(F: np.ndarray, gap: int) -> dict:
    """F (n, L, 169) bool -> redetect(u), episode(u) fuer u = 0..U_MAX, Episodendauern."""
    n, L, N = F.shape
    red = []
    for u in range(U_MAX + 1):
        a, b = F[:, :L - u], F[:, u:]
        red.append(float((a & b).sum() / max(a.sum(), 1)))
    durs = []
    G = F.transpose(0, 2, 1).reshape(-1, L)
    G = G[G.any(1)]
    for row in G:
        d = np.flatnonzero(row)
        cut = np.flatnonzero(np.diff(d) > gap + 1)
        starts, ends = d[np.r_[0, cut + 1]], d[np.r_[cut, d.size - 1]]
        ok = (starts > gap) & (ends < L - 1 - gap)
        durs.extend((ends[ok] - starts[ok] + 1).tolist())
    durs = np.asarray(durs)
    surv = [float((durs > u).mean()) for u in range(U_MAX + 1)]
    return {"redetect": red, "episode": surv, "n_episodes": int(durs.size),
            "episode_dur_mean": float(durs.mean()), "episode_dur_q": np.percentile(durs, [50, 90, 99]).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", default=["cg_ref", "cg_episode"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--n-windows", type=int, default=3000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    man_dir = exp / "data_manifests"
    out_dir = exp / "metrics" / f"contagion_burn_curve{args.tag}"
    stage = f"p48_contagion_burn_curve{args.tag}"
    if (man_dir / f"{stage}.json").exists() or out_dir.exists():
        raise SystemExit(f"ABBRUCH: {stage} oder {out_dir} existiert schon - anderen --tag waehlen")
    out_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    cd = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    Xp = np.load(cd / "X.npy", mmap_mode="r")
    LC = Xp.shape[1]
    Fall = np.asarray(Xp[:, :, FIRE_CH]).reshape(Xp.shape[0], LC, -1) > 0.5
    data = data_curves(Fall, args.gap)
    print(f"[cbc] Daten: {data['n_episodes']:,} Episoden, Dauer Mittel {data['episode_dur_mean']:.2f} Tage, "
          f"Median/90/99 % {data['episode_dur_q']}; redetect(1) {data['redetect'][1]:.3f} ({time.time() - t0:.0f} s)",
          flush=True)

    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    split = np.load(cd / "split.npy")
    te = np.where(split == "test")[0]
    te = te[np.linspace(0, te.size - 1, min(args.n_windows, te.size)).astype(int)]
    doyD = doy_of(np.load(cd / "meta.npy")[:, 0])[te]
    e = LC - 2                                                # Eingabe endet bei D-1 (dt 1, Ziel D+1)
    ckpt_dir = exp / "checkpoints" / "paper"

    res = {}
    for name in args.models:
        res[name] = {"seeds": {}}
        for s in args.seeds:
            cp = ckpt_dir / f"sts_extreme_{name}_s{s}_s{s}.pt"
            if not cp.exists():
                print(f"[cbc] fehlt: {cp.name}", flush=True)
                continue
            c = torch.load(cp, map_location="cpu")
            net, _ = sts_extreme.load_any(str(cp), c["init_px"], c["init_pa"], device)
            Dall, Dfire = [], []
            with torch.no_grad():
                for a in range(0, te.size, 64):
                    r = te[a:a + 64]
                    xr = torch.as_tensor(windows(Xp, r, np.full(r.size, e))).to(device).float()
                    x = (xr - m_t) / s_t
                    doy = torch.from_numpy(doyD[a:a + 64] - (LC - 1 - e)).to(device)
                    _, _, _, logD = net.params(x, doy)                                   # (B, 169, A)
                    Dm = logD[..., :U_MAX + 1].exp().cpu().numpy()
                    fire = (xr[:, :, FIRE_CH] > 0.5).flatten(2).any(1).cpu().numpy()      # (B, 169)
                    Dall.append(Dm.reshape(-1, U_MAX + 1)); Dfire.append(Dm[fire])
            Dall, Dfire = np.concatenate(Dall), np.concatenate(Dfire)
            C = torch.nn.functional.pad(torch.nn.functional.softplus(net.burn_shape.detach().float()).cumsum(0),
                                        (1, 0))[:U_MAX + 1].cpu().numpy() if net.burn == "curve" else None
            man = json.load(open(man_dir / f"p35_sts_extreme_{name}_s{s}.json"))["results"][f"s{s}"]
            res[name]["seeds"][f"s{s}"] = {
                "C": None if C is None else C.tolist(),
                "D_all_q": np.percentile(Dall, [10, 50, 90], axis=0).tolist(),
                "D_fire_q": np.percentile(Dfire, [10, 50, 90], axis=0).tolist() if len(Dfire) else None,
                "burn_days_all": float(Dall.sum(1).mean()),
                "burn_days_fire": float(Dfire.sum(1).mean()) if len(Dfire) else None,
                "paper": {dt_: {h: man["paper"][str(dt_)][h]["pr_auc"] for h in ("patch_head", "centre")} for dt_ in DTS},
                "surv": {h: {k: man["surv"][h]["raw"][k] for k in ("nll", "c_td", "ibs", "pr@7", "pr@30")}
                         for h in ("centre", "patch")},
            }
            print(f"[cbc] {name} s{s}: D_fire Median u=1/3/7/14: "
                  + " ".join(f"{res[name]['seeds'][f's{s}']['D_fire_q'][1][u]:.3f}" for u in (1, 3, 7, 14))
                  + f", Dauer {res[name]['seeds'][f's{s}']['burn_days_fire']:.1f} Tage", flush=True)
            del net
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ Bild
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = np.arange(U_MAX + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for i, name in enumerate(args.models):
        for j, (sk, v) in enumerate(res[name]["seeds"].items()):
            q = np.asarray(v["D_fire_q"])
            ax[i % 2].plot(u, q[1], color="#2a6fdb" if i == 0 else "#d9480f", lw=1.5,
                           label=f"{name} Median (Pixel mit Feuer)" if j == 0 else None)
            ax[i % 2].fill_between(u, q[0], q[2], color="#2a6fdb" if i == 0 else "#d9480f", alpha=0.08)
    ax[0].plot(u, data["redetect"], "k--", lw=1.5, label="Daten: P(Detektion t+u | Detektion t)")
    ax[1].plot(u, data["episode"], "k--", lw=1.5, label=f"Daten: P(Episodendauer > u), Luecke <= {args.gap}")
    ax[0].set_title("Alter ab Detektion (cg_ref)"); ax[1].set_title("Alter ab Episodenbeginn (cg_episode)")
    for a_ in ax:
        a_.set_xlabel("u [Tage]"); a_.set_xlim(0, 30); a_.grid(alpha=0.3); a_.legend(fontsize=8)
    ax[0].set_ylabel("D(u)")
    fig.tight_layout()
    fig.savefig(out_dir / "burn_curve.png", dpi=150)

    # ------------------------------------------------------------------ Tabelle
    def m(name, f):
        vals = [f(v) for v in res[name]["seeds"].values()]
        return (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)

    lines = ["# Contagion-STS: gelernte Brenndauer gegen die Daten", "",
             f"Paper-Test {te.size:,} Fenster ({args.cache}), Eingabe endet bei D-1. Episoden: Luecke <= {args.gap} Tage.",
             "", "## Daten", "",
             f"- Episoden: {data['n_episodes']:,}, Dauer Mittel {data['episode_dur_mean']:.2f} Tage, "
             f"Median/90/99 %: {', '.join(f'{q:.0f}' for q in data['episode_dur_q'])} Tage (61-Tage-Fenster, zensiert)",
             "- redetect(u) u=1/3/7/14: " + " / ".join(f"{data['redetect'][k]:.3f}" for k in (1, 3, 7, 14)),
             "- P(Episode > u) u=1/3/7/14: " + " / ".join(f"{data['episode'][k]:.3f}" for k in (1, 3, 7, 14)),
             "", "## Modelle (Mittel ± Std ueber Seeds; D = Median ueber Pixel mit Feuer in 30 Tagen)", "",
             "| Modell | D(1) | D(3) | D(7) | D(14) | Dauer [Tage] | Dauer alle Pixel |", "|---|---|---|---|---|---|---|"]
    for name in args.models:
        cells = [f"{m(name, lambda v, k=k: v['D_fire_q'][1][k])[0]:.3f}" for k in (1, 3, 7, 14)]
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | {m(name, lambda v: v['burn_days_fire'])[0]:.1f} | {m(name, lambda v: v['burn_days_all'])[0]:.1f} |")
    lines += ["", "## Paper-Protokoll PR-AUC (D+1, Ausschnitt / Mitte, dt 1/7/15/30)", "",
              "| Modell | Ausschnitt | Mitte |", "|---|---|---|"]
    for name in args.models:
        pa = "/".join(f"{m(name, lambda v, d=d: v['paper'][d]['patch_head'])[0]:.3f}" for d in DTS)
        ce = "/".join(f"{m(name, lambda v, d=d: v['paper'][d]['centre'])[0]:.3f}" for d in DTS)
        lines.append(f"| {name} | {pa} | {ce} |")
    lines += ["", "## Survival (Wuerfel-Stichtage, raw; Mitte / Ausschnitt)", "",
              "| Modell | NLL | C^td | IBS | PR@7 | PR@30 |", "|---|---|---|---|---|---|"]
    for name in args.models:
        cells = []
        for k in ("nll", "c_td", "ibs", "pr@7", "pr@30"):
            c_, p_ = m(name, lambda v, k=k: v["surv"]["centre"][k]), m(name, lambda v, k=k: v["surv"]["patch"][k])
            cells.append(f"{c_[0]:.4f} / {p_[0]:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump({"data": data, "models": res, "u": u.tolist()}, open(out_dir / "curves.json", "w"))
    write_manifest(cfg, stage, args.config, {"models": args.models, "seeds": args.seeds, "gap": args.gap,
                                             "n_windows": int(te.size), "out_dir": str(out_dir)})
    print("\n".join(lines), flush=True)
    print(f"[cbc] fertig ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

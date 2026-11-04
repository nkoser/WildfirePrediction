"""Eingabe-Ablation (sts_inputs, slurm 153) im Paper-Protokoll mit Ziel D und D+1 (nur Auswertung).

Die sts_inputs-Netze sind auf Wuerfel-Stichtagen trainiert; das Training haengt nicht vom Paper-Ziel ab.
sts_extreme wertet das Paper-Protokoll nur mit Ziel D+1 aus. Dieses Skript laedt dieselben Checkpoints,
setzt dieselbe Kanalmaske davor (sts_inputs.wrap) und rechnet beide Ziele (targets.py):

  next_patch  Zieltag D+1: Eingabe endet bei D - dt (Index LC-1-dt), Score F(dt + 1), Label y_next_patch
              -> muss die Zahlen aus p35_sts_extreme_<name> exakt wiederholen (Selbsttest)
  cache       Zieltag D:   Eingabe endet bei D - dt - 1 (Index LC-2-dt), Score F(dt + 1), Label y (Cache)
Fenster in beiden Faellen wie im Paper: [t - dt - 30, t - dt - 1]. Mitte = Pixelkopf (6, 6), Ausschnitt =
Ausschnittskopf. Metriken wie sts_extreme (paper_metrics auf den rohen F, ohne Platt; PR-AUC und ROC-AUC
sind davon unabhaengig).

Dazu, nur gelesen: die 3D-U-Net-Rekonstruktionen (p2_reconstruct_*) mit beiden Zielen und Kanalsaetzen.
Modelle als name:inputs, z. B. lh_ref:all lh_nofire:nofire; Checkpoints sts_extreme_<name>_s<k>_s<k>.pt.
Ausgaben NEU: Manifest p46_sts_inputs_target<tag>, metrics/sts_inputs_target<tag>/summary.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import sts_extreme, targets
from wildfire.paper.compare import DTS
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import CENTRE, doy_of
from wildfire.paper.sts_inputs import INPUTS, wrap

DEFAULT_MODELS = [f"{h}_{v}:{i}" for h in ("lh", "cg") for v, i in
                  (("ref", "all"), ("fireonly", "fireonly"), ("nofire", "nofire"), ("nothermal", "nothermal"),
                   ("nomodis", "nomodis"))]
UNET = {  # Kanalsatz -> Manifest je Ziel (p2_reconstruct, cache G, Paper-Hyperparameter, 3 Seeds)
    "paper55": {"cache": "p2_reconstruct_cache_G_cache_paperhp", "next_patch": "p2_reconstruct_cache_G_cache_next_paperhp_w0"},
    "no_fire": {"cache": "p2_reconstruct_cache_G_cache_paperhp_no_fire",
                "next_patch": "p2_reconstruct_cache_G_cache_next_paperhp_w0_nofire"},
    "no_modis": {"cache": "p2_reconstruct_cache_G_cache_paperhp_no_modis",
                 "next_patch": "p2_reconstruct_cache_G_cache_next_paperhp_w0_nomodis"},
    "no_thermal": {"cache": "p2_reconstruct_cache_G_cache_paperhp_no_thermal",          # slurm 211
                   "next_patch": "p2_reconstruct_cache_G_cache_next_paperhp_w0_no_thermal"},
    "fire_only": {"cache": "p2_reconstruct_cache_G_cache_paperhp_fire_only",
                  "next_patch": "p2_reconstruct_cache_G_cache_next_paperhp_w0_fire_only"},
}
KEYS = ("pr_auc", "roc_auc", "f1", "ece", "brier")


def unet_rows(man_dir: Path) -> dict:
    """Seed-Mittel der PR-AUC je dt aus den U-Net-Manifesten (Reihenfolge der Laeufe = DTS)."""
    out = {}
    for cs, by_t in UNET.items():
        for tg, stem in by_t.items():
            f = man_dir / f"{stem}.json"
            if not f.exists():
                continue
            runs = json.load(open(f))["runs"]
            runs = runs if isinstance(runs, list) else list(runs.values())
            ends = sorted({r["end"] for r in runs})
            if len(ends) != len(DTS):
                continue
            out[f"{cs}|{tg}"] = {str(dt_): float(np.mean([r["test"]["pr_auc"] for r in runs if r["end"] == e]))
                                 for dt_, e in zip(DTS, ends)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--limit-eval", type=int, default=0, help="nur fuer Rauchtests")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    man_dir = exp / "data_manifests"
    out_dir = exp / "metrics" / f"sts_inputs_target{args.tag}"
    stage = f"p46_sts_inputs_target{args.tag}"
    if (man_dir / f"{stage}.json").exists() or out_dir.exists():
        raise SystemExit(f"ABBRUCH: {stage} oder {out_dir} existiert schon - anderen --tag waehlen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = exp / "checkpoints" / "paper"
    t0 = time.time()

    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    cd = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    Xp = np.load(cd / "X.npy", mmap_mode="r")
    LC = Xp.shape[1]
    split = np.load(cd / "split.npy")
    te = np.where(split == "test")[0]
    if args.limit_eval:
        te = te[np.linspace(0, te.size - 1, args.limit_eval).astype(int)]   # gleichmaessig, Cache ist nach Label sortiert
    Xg = to_device_once(np.asarray(Xp[te]), device, "cache_X")
    loc = np.arange(te.size)
    doyD = doy_of(np.load(cd / "meta.npy")[:, 0])[te]
    y = {tg: {"patch_head": targets.labels(cd, tg)[te], "centre": targets.centre_labels(cd, Xp, tg)[te]}
         for tg in ("cache", "next_patch")}
    print(f"[sit] Paper-Test {te.size:,}; Positivrate Ausschnitt D {y['cache']['patch_head'].mean():.3f}, "
          f"D+1 {y['next_patch']['patch_head'].mean():.3f} ({time.time() - t0:.0f} s)", flush=True)

    @torch.no_grad()
    def evaluate_net(net, tg: str, bs: int = 64) -> dict:
        out = {}
        for dt_ in DTS:
            e = LC - 1 - dt_ - (1 - targets.shift(tg))       # letzter Eingabetag
            tau = torch.tensor([float(dt_ + 1)], device=device)
            fc, fa = [], []
            for a in range(0, te.size, bs):
                r = loc[a:a + bs]
                x = (torch.as_tensor(windows(Xg, r, np.full(r.size, e))).to(device).float() - m_t) / s_t
                doy = torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw_px, raw_pa = net(x, doy)
                fc.append((-torch.expm1(net.px_head.log_survival(raw_px[:, CENTRE, CENTRE], tau))[:, 0]).cpu().numpy())
                fa.append((-torch.expm1(net.pa_head.log_survival(raw_pa, tau))[:, 0]).cpu().numpy())
            cl = lambda v: np.clip(np.concatenate(v).astype(np.float64), 1e-7, 1 - 1e-7)  # noqa: E731
            out[str(dt_)] = {"centre": paper_metrics(cl(fc), y[tg]["centre"]),
                             "patch_head": paper_metrics(cl(fa), y[tg]["patch_head"])}
        return out

    res, selftest = {}, {}
    for spec in args.models:
        name, inp = spec.split(":")
        keep = INPUTS[inp]
        res[name] = {"inputs": inp, "n_channels": int(keep.sum()), "seeds": {}}
        for s in args.seeds:
            cp = ckpt_dir / f"sts_extreme_{name}_s{s}_s{s}.pt"
            if not cp.exists():
                raise SystemExit(f"ABBRUCH: {cp} fehlt")
            c = torch.load(cp, map_location="cpu")
            net, _ = sts_extreme.load_any(str(cp), c["init_px"], c["init_pa"], device)
            net = wrap(net, keep, "zeros")
            res[name]["seeds"][f"s{s}"] = {tg: evaluate_net(net, tg) for tg in ("cache", "next_patch")}
            del net
        res[name]["mean"] = {tg: {dt_: {h: {k: float(np.mean([v[tg][dt_][h][k] for v in res[name]["seeds"].values()]))
                                            for k in KEYS} for h in ("centre", "patch_head")}
                                  for dt_ in map(str, DTS)} for tg in ("cache", "next_patch")}
        mm = res[name]["mean"]
        print(f"[sit] {name:14s} ({inp}, {keep.sum()} Kanaele) Ausschnitt PR-AUC  D: "
              + "/".join(f"{mm['cache'][str(d)]['patch_head']['pr_auc']:.3f}" for d in DTS) + "   D+1: "
              + "/".join(f"{mm['next_patch'][str(d)]['patch_head']['pr_auc']:.3f}" for d in DTS), flush=True)
        # Selbsttest: D+1 muss die Werte aus sts_extreme wiederholen
        ref = man_dir / f"p35_sts_extreme_{name}_s{args.seeds[0]}.json"
        if ref.exists() and not args.limit_eval:
            old = json.load(open(ref))["results"][f"s{args.seeds[0]}"]["paper"]
            new = res[name]["seeds"][f"s{args.seeds[0]}"]["next_patch"]
            dev = max(abs(old[k][h]["pr_auc"] - new[k][h]["pr_auc"]) for k in old for h in ("centre", "patch_head"))
            selftest[name] = dev
            print(f"[sit]   Selbsttest D+1 gegen p35 (Seed {args.seeds[0]}): max |dPR-AUC| = {dev:.2e}", flush=True)

    unet = unet_rows(man_dir)
    # ---- Zusammenfassung
    lines = ["# Eingabe-Ablation im Paper-Protokoll: Ziel D gegen D+1", "",
             f"Cache {args.cache}, Test n = {te.size:,}, Seed-Mittel ({len(args.seeds)} Seeds), PR-AUC je dt "
             + "/".join(map(str, DTS)) + ". Gleiche Checkpoints fuer beide Ziele (Training auf Wuerfel-Stichtagen).",
             f"Positivrate Ausschnitt: D {y['cache']['patch_head'].mean():.3f}, D+1 {y['next_patch']['patch_head'].mean():.3f}.",
             "", "## Ausschnitt", "", "| Modell | Eingabe | Kanaele | D | D+1 |", "|---|---|---|---|---|"]
    fmt = lambda m, tg, h: "/".join(f"{m[tg][str(d)][h]['pr_auc']:.3f}" for d in DTS)  # noqa: E731
    for name, r in res.items():
        lines.append(f"| {name} | {r['inputs']} | {r['n_channels']} | {fmt(r['mean'], 'cache', 'patch_head')} | "
                     f"{fmt(r['mean'], 'next_patch', 'patch_head')} |")
    for cs in UNET:
        a, b = unet.get(f"{cs}|cache"), unet.get(f"{cs}|next_patch")
        f2 = lambda v: "/".join(f"{v[str(d)]:.3f}" for d in DTS) if v else "–"  # noqa: E731
        lines.append(f"| 3D U-Net (Nachbau) | {cs} | – | {f2(a)} | {f2(b)} |")
    lines += ["| 3D U-Net (Paper-Werte) | – | – | – | 0.718/0.598/0.551/0.481 |", "", "## Mitte", "",
              "| Modell | Eingabe | D | D+1 |", "|---|---|---|---|"]
    for name, r in res.items():
        lines.append(f"| {name} | {r['inputs']} | {fmt(r['mean'], 'cache', 'centre')} | {fmt(r['mean'], 'next_patch', 'centre')} |")
    lines += ["", "## Selbsttest D+1 gegen p35_sts_extreme (max |dPR-AUC|, Seed " + str(args.seeds[0]) + ")", ""]
    lines += [f"- {k}: {v:.2e}" for k, v in selftest.items()] or ["- nicht gelaufen"]
    out_dir.mkdir(parents=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    write_manifest(cfg, stage, args.config, {
        "models": args.models, "seeds": args.seeds, "cache": args.cache, "n_test": int(te.size),
        "protocol": {"next_patch": "Eingabe endet D - dt, Score F(dt + 1), Label y_next_patch",
                     "cache": "Eingabe endet D - dt - 1, Score F(dt + 1), Label y (Cache, Feuer an D)"},
        "bias": "trained (init_px/init_pa aus dem Checkpoint), wie sts_extreme",
        "results": res, "unet": unet, "selftest_next_vs_p35": selftest,
    })
    print(f"[sit] fertig in {time.time() - t0:.0f} s -> {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

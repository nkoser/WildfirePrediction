"""Regionskurven: Pfadzahl-Diagnose, gemeinsame Zufallszahlen und Bootstrap (ergaenzt region_eval.py, das
unveraendert bleibt und hier nur importiert wird).

1. Gemeinsame Zufallszahlen (CRN): Alle Regionen eines Stichtags wuerfeln ihre Pfade mit denselben
   Zufallszahlen. Der Rao-Blackwell-Schaetzer aus region_eval bleibt unverzerrt, Unterschiede zwischen
   Regionen rauschen weniger. Monotonie ueber verschachtelte Regionen ist damit NICHT je Pfad garantiert
   (im groesseren R sind mehr Zuendungen unterdrueckt, die Hazards im kleineren sinken), nur im
   Erwartungswert. Gemessen wird der Anteil der Stichtage mit Verletzung, mit und ohne CRN.
2. Pfadzahl-Diagnose (--Ms 16,64,256): Naehert sich die 1x1-Kurve mit mehr Pfaden Mean-Field an
   (dann war der Abstand Rauschen) oder bleibt er (dann liegt es am Training auf Mean-Field-Randkurven)?
3. NLL je Stichtag und gepaarter Bootstrap ueber die Stichtage fuer die Vergleiche MC gegen Mean-Field,
   MC gegen Produkt, MC gegen STS-LH (gleicher Seed) und MC gegen den STS-LH-Ausschnittskopf.

Ausgabe NEU: metrics/paper_region/region_crn<tag>.json und region_crn<tag>_nll.npz (bricht ab, wenn vorhanden).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config
from wildfire.eval.metrics import EPS, evaluate
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.region_eval import (H, compact, hazards_from_logS, lh_curves, mc_region_logS,
                                        nested_violations, region_target, regions)
from wildfire.paper.sts_dctm import load_sts

TAUS = (7, 30, 100)


# ---------------------------------------------------------------------------- Contagion
@torch.no_grad()
def contagion_curves(ck, X, doy, m_t, s_t, masks_np, Ms, M_ind, device, bs, seed):
    """mc_crn_M<M> je M (gemeinsame Zufallszahlen ueber die Regionen), optional mc_ind_M<M_ind>
    (eigene Zufallszahlen je Region wie region_eval), mf und ind."""
    net = load_sts(ck, device)
    net.eval()
    masks = {k: torch.from_numpy(v).to(device) for k, v in masks_np.items()}
    gens = {M: torch.Generator(device=device).manual_seed(seed) for M in Ms}
    gen_ind = torch.Generator(device=device).manual_seed(seed + 1)
    meths = [f"mc_crn_M{M}" for M in Ms] + ([f"mc_ind_M{M_ind}"] if M_ind else []) + ["mf", "ind"]
    res = {m: {r: [] for r in masks_np} for m in meths}
    full = []
    for a in range(0, X.shape[0], bs):
        x = (X[a:a + bs].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[a:a + bs]).to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            obs, aa, k, logD = net.params(x, d)
            log_exo = net.exogenous(obs, aa, k, logD)
            ls_px = net.rollout(log_exo, k, logD)
            full.append(F.pad(log_exo.sum(2).cumsum(1), (1, 0)).double().cpu().numpy())
            for M in Ms:
                st = gens[M].get_state()
                for r, R in masks.items():
                    gens[M].set_state(st)                      # dieselben Zufallszahlen fuer jede Region
                    res[f"mc_crn_M{M}"][r].append(mc_region_logS(net, log_exo, k, logD, {r: R}, M, gens[M])[r])
            if M_ind:
                mc = mc_region_logS(net, log_exo, k, logD, masks, M_ind, gen_ind)
                for r in masks:
                    res[f"mc_ind_M{M_ind}"][r].append(mc[r])
            for r, R in masks.items():
                res["ind"][r].append(ls_px[:, R].sum(1).double().cpu().numpy())
                res["mf"][r].append(net.region_log_survival(x, d, R).double().cpu().numpy())
    out = {m: {r: np.concatenate(v) for r, v in d_.items()} for m, d_ in res.items()}
    exact = np.concatenate(full)
    dev = max(float(np.abs(out[f"mc_crn_M{M}"]["sq13"] - exact).max()) for M in Ms)
    print(f"[crn] Selbsttest {Path(ck).name}: MC (CRN) gegen exakte Ausschnittsformel, max |d log S| = {dev:.2e}",
          flush=True)
    if dev > 1e-4:
        raise SystemExit("ABBRUCH: Monte-Carlo-Schaetzer weicht fuer den ganzen Ausschnitt von der exakten Formel ab")
    return out, dev


# ---------------------------------------------------------------------------- NLL je Stichtag
def nll_per_sample(h: np.ndarray, event: np.ndarray, t_event: np.ndarray) -> np.ndarray:
    """Wie metrics.negative_log_likelihood, aber je Zeile (ungewichtet)."""
    hb = np.clip(np.asarray(h, np.float64), 1e-15, 1 - EPS)
    k = np.asarray(t_event).astype(int)
    e = np.asarray(event).astype(bool)
    csum = np.cumsum(np.log1p(-hb), axis=1)
    idx = np.arange(len(hb))
    prior = np.where(k > 1, csum[idx, np.maximum(k - 2, 0)], 0.0)
    return -np.where(e, np.log(hb[idx, k - 1]) + prior, csum[:, -1])


def boot_diff(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> dict:
    """Gepaarter Bootstrap ueber die Stichtage: mittlere Differenz a - b mit 95-%-Intervall."""
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "share_a_better": float((bm < 0).mean())}


# ---------------------------------------------------------------------------- Auswertung
def run(cfg, exp, args, device):
    out_dir = exp / "metrics" / "paper_region"
    out = out_dir / f"region_crn{args.tag}.json"
    out_npz = out_dir / f"region_crn{args.tag}_nll.npz"
    for p in (out, out_npz):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    Ms = [int(m) for m in args.Ms.split(",")]
    sd = exp / "derived_data" / args.surv_dir
    me = dict(np.load(sd / "eval_meta.npz"))
    pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
    assert (pe["row"] == me["row"]).all() and (pe["L_abs"] == me["L_abs"]).all(), "Stichtage passen nicht"
    rng = np.random.default_rng(args.seed)                       # dieselbe Teilmenge wie region_eval bei gleichem n
    idx = np.sort(rng.choice(len(me["row"]), size=min(args.n, len(me["row"])), replace=False))
    t0 = time.time()
    X = torch.from_numpy(np.ascontiguousarray(np.load(sd / "eval_X.npy", mmap_mode="r")[idx]))
    print(f"[crn] {len(idx):,} Stichtage geladen in {time.time() - t0:.0f} s; Pfade {Ms}, ohne CRN {args.M_ind}",
          flush=True)
    doy = me["doy"][idx].astype(np.float32)
    ev_px, t_px = pe["ev"][idx], pe["t"][idx].astype(int)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    masks = regions()
    targets = {r: region_target(ev_px, t_px, m) for r, m in masks.items()}
    res = {"n": int(len(idx)), "Ms": Ms, "M_ind": args.M_ind, "seed": args.seed,
           "regions": {r: int(m.sum()) for r, m in masks.items()},
           "event_rate": {r: {str(t): float((e & (tt <= t)).mean()) for t in TAUS} for r, (e, tt) in targets.items()},
           "models": {}, "selftest_mc_vs_exact": {}, "nested_violation_share": {}, "mean_F": {},
           "mc_vs_mf_absF100": {}, "M_convergence_absF100": {}, "bootstrap": {}}
    nll_store, curves_all = {}, {}

    def score(method, s, curves):
        r_out = res["models"].setdefault(method, {})
        for r, ls in curves.items():
            e, t = targets[r]
            h = hazards_from_logS(ls)
            m = compact(evaluate(h, e, t))
            ps = nll_per_sample(h, e, t)
            if abs(ps.mean() - m["nll"]) > 1e-6 * max(1.0, abs(m["nll"])):
                raise SystemExit(f"ABBRUCH: NLL je Stichtag ({ps.mean():.6f}) passt nicht zu evaluate ({m['nll']:.6f})")
            r_out.setdefault(r, {})[s] = m
            nll_store[f"{method}|{s}|{r}"] = ps.astype(np.float32)
            res["mean_F"].setdefault(method, {}).setdefault(r, {})[s] = {
                str(tau): float(-np.expm1(ls[:, tau]).mean()) for tau in TAUS}
        res["nested_violation_share"].setdefault(method, {})[s] = nested_violations(curves)
        curves_all[(method, s)] = curves
        print(f"[crn] {method:12s} {s:3s} Verletzung {res['nested_violation_share'][method][s]:.4f}  " + "  ".join(
            f"{r}: NLL {r_out[r][s]['nll']:.4f} F100 {res['mean_F'][method][r][s]['100']:.4f}"
            for r in ("sq1", "sq5", "sq13", "nw6") if r in r_out), flush=True)   # lh_patch hat nur sq13

    seeds = []
    for ck in sorted(glob.glob(args.cg)):
        s = Path(ck).stem.split("_s")[-1]
        seeds.append(s)
        cur, dev = contagion_curves(ck, X, doy, m_t, s_t, masks, Ms, args.M_ind, device, args.bs, args.seed)
        res["selftest_mc_vs_exact"][s] = dev
        for meth, cv in cur.items():
            score(meth, s, cv)
        F100 = lambda ls: -np.expm1(ls[:, -1])
        ref = f"mc_crn_M{max(Ms)}"
        for r in masks:
            res["mc_vs_mf_absF100"].setdefault(r, {})[s] = float(np.abs(F100(cur[ref][r]) - F100(cur["mf"][r])).mean())
            for M in Ms:
                if M != max(Ms):
                    res["M_convergence_absF100"].setdefault(f"M{M}_vs_M{max(Ms)}", {}).setdefault(r, {})[s] = float(
                        np.abs(F100(cur[f"mc_crn_M{M}"][r]) - F100(cur[ref][r])).mean())
    if not seeds:
        raise SystemExit(f"ABBRUCH: keine Contagion-Checkpoints fuer {args.cg}")
    for ck in sorted(glob.glob(args.lh)):
        s = Path(ck).stem.split("_s")[-1]
        cur, v = lh_curves(ck, X, doy, m_t, s_t, masks, device, args.bs)
        res.setdefault("lh_patch_below_pixel_share", {})[s] = v
        for meth, cv in cur.items():
            score(meth, s, cv)

    # Gepaarter Bootstrap je Seed und Region; negatives Mittel = erstes Verfahren besser
    brng = np.random.default_rng(args.seed + 7)
    mc = f"mc_crn_M{max(Ms)}"
    pairs = [(mc, "mf"), (mc, "ind"), ("mf", "ind"), (mc, "lh_ind"), (mc, "lh_patch")]
    for a, b in pairs:
        for s in seeds:
            for r in masks:
                ka, kb = f"{a}|{s}|{r}", f"{b}|{s}|{r}"
                if ka in nll_store and kb in nll_store:
                    res["bootstrap"].setdefault(f"{a}-{b}", {}).setdefault(r, {})[s] = boot_diff(
                        nll_store[ka].astype(np.float64), nll_store[kb].astype(np.float64), args.boot, brng)
    for key, regs in res["bootstrap"].items():
        print(f"[crn] Bootstrap {key}: " + "  ".join(
            f"{r} {np.mean([v['mean'] for v in regs[r].values()]):+.4f}" for r in ("sq1", "sq5", "sq13", "nw6")
            if r in regs), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, idx=idx, **{k.replace("|", "__"): v for k, v in nll_store.items()})
    out.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[crn] {out}\n[crn] {out_npz}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cg", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_cg_s?_s?.pt")
    ap.add_argument("--lh", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_lh_s?_s?.pt")
    ap.add_argument("--n", type=int, default=20_000, help="zufaellige Teilmenge der Test-Stichtage")
    ap.add_argument("--Ms", default="64", help="Pfadzahlen mit gemeinsamen Zufallszahlen, kommagetrennt")
    ap.add_argument("--M-ind", type=int, default=0, help="zusaetzlich MC ohne CRN wie region_eval (0 = aus)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(cfg, exp, args, device)


if __name__ == "__main__":
    main()

"""Kalibrierung der Zahl brennender Pixel: Trifft die Streuung der verbrannten Flaeche, wenn die Pixel gekoppelt sind?

Ein Modell mit unabhaengigen Pixeln unterschaetzt die Streuung der Zahl brennender Pixel, wenn Feuer in Clustern
kommen; seine Intervalle sind dann zu schmal. Gemessen auf den zufaelligen Test-Stichtagen (wie joint_eval
--part scenarios, dieselbe Teilmenge bei gleichem --n und --seed), mit den nachkalibrierten Modellen (joint_calib):

  Ziehungen   cg       Contagion-Prozess              cg_shuf  dieselben Ziehungen, je Pixel permutiert
                                                                (gleiche Randverteilungen, keine Kopplung)
              cg_ind   unabhaengig aus den Contagion-Pixelkurven (Mean Field)
              lh_ind   unabhaengig aus den STS-LH-Pixelkurven
  Zielgroesse N_tau = Zahl der Pixel im 13 x 13-Ausschnitt mit erstem Feuer bis Tag tau (7, 30, 100)
  Metriken    randomisierter PIT (gleichverteilt, wenn kalibriert), Abdeckung der zentralen 50/80/90-%-Intervalle,
              Streuungsverhaeltnis sum (n - mean)^2 / sum Var (1 = passend, > 1 = Intervalle zu schmal),
              mittlere Vorhersage gegen beobachtet; je Schicht (alle, Feuer im Fenster, ohne) und je Testjahr

  --part run      -> metrics/paper_joint/counts<tag>.{json,npz}
  --part summary  Mittel ueber Seeds, PIT-Histogramme, Bootstrap ueber Stichtage gegen --refs; dazu die
                  Szenario-Scores (joint_eval/joint_calib scenarios-npz, --scen-inputs) je Testjahr
                  -> metrics/paper_joint/counts_summary<tag>.json

Alle Eingaben nur gelesen; bestehende Ausgaben werden nicht ueberschrieben (Abbruch).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config
from wildfire.paper import joint_eval
from wildfire.paper.joint_calib import load_calibrated
from wildfire.paper.joint_eval import TAUS, lh_parts, ls_from_h, parse_models, sample_cg, sample_ind, shuffle
from wildfire.paper.region_eval import H, N

QS = (0.5, 0.8, 0.9)
DAY0 = dt.date(2000, 1, 1)


@torch.no_grad()
def count_stats(T, y, gen):
    """T (B, M, N) Erstfeuer-Tage der Ziehungen, y (B, N) beobachtet -> {item@tau: (B,)}."""
    out = {}
    for tau in TAUS:
        Nm = (T <= tau).sum(-1).float()
        n = (y <= tau).sum(-1).float()
        le = (Nm <= n[:, None]).float().mean(1)
        lt = (Nm < n[:, None]).float().mean(1)
        u = torch.rand(n.shape, generator=gen, device=n.device)
        out[f"pit@{tau}"] = lt + u * (le - lt)
        for q in QS:
            lo = torch.quantile(Nm, (1 - q) / 2, dim=1, interpolation="lower")
            hi = torch.quantile(Nm, (1 + q) / 2, dim=1, interpolation="higher")
            out[f"cov{int(q * 100)}@{tau}"] = ((n >= lo) & (n <= hi)).float()
        out[f"mean@{tau}"] = Nm.mean(1)
        out[f"var@{tau}"] = Nm.var(1)
        out[f"obs@{tau}"] = n
    return {k: v.cpu().numpy() for k, v in out.items()}


def run(cfg, exp, args, device):
    d = exp / "metrics" / "paper_joint"
    js, nz = d / f"counts{args.tag}.json", d / f"counts{args.tag}.npz"
    for p in (js, nz):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    args.bias = f"calib:{args.calib_dir}"
    idx, X, doy, ev_px, t_px, m_t, s_t, cube_rates, fw = joint_eval.setup(cfg, exp, args, device)
    y_all = torch.from_numpy(np.where(ev_px, t_px, H + 1).reshape(len(idx), N))
    res = {"n": int(len(idx)), "M": args.M, "calib_dir": args.calib_dir, "calib": {}, "mean": {}}
    store = {}
    for name, kind, cks in parse_models(args.models):
        for ck in cks:
            seed = Path(ck).stem.rsplit("_s", 1)[1]
            net, used = load_calibrated(exp, args.calib_dir, ck, device, cube_rates)
            res["calib"].setdefault(name, {})[seed] = used
            gen = torch.Generator(device=device).manual_seed(args.seed)
            acc = {}
            t0 = time.time()
            for a0 in range(0, X.shape[0], args.bs):
                x = (X[a0:a0 + args.bs].to(device).float() - m_t) / s_t
                dd = torch.from_numpy(doy[a0:a0 + args.bs]).to(device)
                y = y_all[a0:a0 + args.bs].to(device)
                with torch.no_grad():
                    if kind == "cg":
                        with torch.autocast(device_type=device.type, enabled=False):
                            obs, aa, k, logD = net.params(x, dd)
                            log_exo = net.exogenous(obs, aa, k, logD)
                            ls_px = net.rollout(log_exo, k, logD)
                            T = sample_cg(net, log_exo, k, logD, args.M, gen)
                        draws = {"cg": T, "cg_shuf": shuffle(T, gen), "cg_ind": sample_ind(ls_px.double(), args.M, gen)}
                    else:
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            _, raw_px, _ = lh_parts(net, x, dd)
                        b = x.shape[0]
                        lsp = ls_from_h(net.px_head.hazards(raw_px.reshape(b * N, -1), H).double().view(b, N, H))
                        draws = {"ind": sample_ind(lsp, args.M, gen)}
                for meth, T in draws.items():
                    for item, v in count_stats(T, y, gen).items():
                        acc.setdefault(f"{name}_{meth}__{seed}__{item}", []).append(v)
            for key, v in acc.items():
                v = np.concatenate(v).astype(np.float32)
                store[key] = v
                meth, _, item = key.split("__")
                res["mean"].setdefault(meth, {}).setdefault(item, {})[seed] = {
                    "all": float(v.mean()), "fire_in_window": float(v[fw].mean()), "no_fire": float(v[~fw].mean())}
            for meth in sorted({k.split("__")[0] for k in acc}):
                m_ = res["mean"][meth]
                print(f"[counts] {meth:16s} {seed}  " + "  ".join(
                    f"cov80@{t} {m_[f'cov80@{t}'][seed]['fire_in_window']:.3f}" for t in TAUS)
                      + f"  mean@30 {m_['mean@30'][seed]['fire_in_window']:.3f} (beob. "
                      f"{m_['obs@30'][seed]['fire_in_window']:.3f}, Feuer im Fenster)", flush=True)
            print(f"[counts] {name} {seed}: {time.time() - t0:.0f} s", flush=True)
            del net
            torch.cuda.empty_cache()
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, idx=idx, fire_win=fw, **store)
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[counts] {js}\n[counts] {nz}", flush=True)


# ---------------------------------------------------------------------------- Zusammenfassung
def load_seed_mean(d, files, items_ok=None):
    """npz mit Schluesseln meth__seed__item -> (idx, fire_win, {meth: {item: {seed: (n,)}}})."""
    idx = fw = None
    per = {}
    for f in files:
        z = np.load(d / f)
        if idx is None:
            idx, fw = z["idx"], z["fire_win"].astype(bool)
        elif not np.array_equal(idx, z["idx"]):
            raise SystemExit(f"ABBRUCH: {f} hat andere Stichtage")
        for k_ in z.files:
            if k_ in ("idx", "fire_win"):
                continue
            meth, seed, item = k_.split("__")
            per.setdefault(meth, {}).setdefault(item, {})[seed] = z[k_].astype(np.float64)
    return idx, fw, per


def calib_block(per_m, mask):
    """Kennzahlen der Zahl brennender Pixel in einer Schicht, je Seed und gemittelt."""
    out = {}
    seeds = sorted(per_m["obs@30"])
    for tau in TAUS:
        r = {}
        for s in seeds:
            pit = per_m[f"pit@{tau}"][s][mask]
            hist = np.histogram(pit, bins=10, range=(0, 1))[0] / max(pit.size, 1)
            mu, var, n = (per_m[f"{it}@{tau}"][s][mask] for it in ("mean", "var", "obs"))
            rs = {"pit_hist": hist.tolist(), "pit_dev": float(np.abs(hist - 0.1).sum()),
                  "dispersion": float(((n - mu) ** 2).sum() / max(var.sum(), 1e-12)),
                  "mean_pred": float(mu.mean()), "mean_obs": float(n.mean())}
            for q in QS:
                rs[f"cov{int(q * 100)}"] = float(per_m[f"cov{int(q * 100)}@{tau}"][s][mask].mean())
            r[s] = rs
        keys = [k_ for k_ in r[seeds[0]] if k_ != "pit_hist"]
        r["mean"] = {k_: float(np.mean([r[s][k_] for s in seeds])) for k_ in keys}
        r["sd"] = {k_: float(np.std([r[s][k_] for s in seeds])) for k_ in keys}
        out[str(tau)] = r
    return out


def boot_iid(diff, n_boot, rng):
    bm = np.array([diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi)}


def run_summary(cfg, exp, args):
    d = exp / "metrics" / "paper_joint"
    out = d / f"counts_summary{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderes --tag waehlen")
    me = dict(np.load(exp / "derived_data" / args.surv_dir / "eval_meta.npz"))
    year_of = lambda ix: np.array([(DAY0 + dt.timedelta(days=int(v))).year for v in me["L_abs"][ix]])  # noqa: E731
    res = {"inputs": args.inputs, "scen_inputs": args.scen_inputs, "counts": {}, "counts_boot": {}, "scenarios": {},
           "scenarios_boot": {}}
    rng = np.random.default_rng(args.seed + 19)
    if args.inputs:
        idx, fw, per = load_seed_mean(d, args.inputs)
        year = year_of(idx)
        strata = {"all": np.ones(idx.size, bool), "fire_in_window": fw, "no_fire": ~fw,
                  **{str(y): year == y for y in np.unique(year)}}
        for m, pm in per.items():
            res["counts"][m] = {s: calib_block(pm, msk) for s, msk in strata.items()}
            a = res["counts"][m]["fire_in_window"]
            print(f"[counts] {m:16s} Feuer im Fenster: " + "  ".join(
                f"tau {t}: cov80 {a[str(t)]['mean']['cov80']:.3f} Streuung {a[str(t)]['mean']['dispersion']:.2f}"
                for t in TAUS), flush=True)
        mean_of = lambda m, it: np.mean(list(per[m][it].values()), 0)  # noqa: E731
        for ref in args.refs:
            for m in per:
                if m == ref:
                    continue
                for s in ("all", "fire_in_window"):
                    for tau in TAUS:
                        for q in QS:
                            it = f"cov{int(q * 100)}@{tau}"
                            res["counts_boot"].setdefault(f"{ref}-{m}", {}).setdefault(s, {})[it] = boot_iid(
                                mean_of(ref, it)[strata[s]] - mean_of(m, it)[strata[s]], args.boot, rng)
    if args.scen_inputs:
        idx, fw, per = load_seed_mean(d, args.scen_inputs)
        year = year_of(idx)
        strata = {"all": np.ones(idx.size, bool), "fire_in_window": fw,
                  **{str(y): year == y for y in np.unique(year)},
                  **{f"{y}_fire_in_window": (year == y) & fw for y in np.unique(year)}}
        pred = {m: {it: np.mean(list(s.values()), 0) for it, s in v.items()} for m, v in per.items()}
        for m, pm in pred.items():
            res["scenarios"][m] = {s: {it: float(v[msk].mean()) for it, v in pm.items()} for s, msk in strata.items()}
        for ref in args.scen_refs:
            for m in pred:
                if m == ref:
                    continue
                for s, msk in strata.items():
                    for it in ("es@30", "vs@30", "crps@30", "vs@7", "crps@7"):
                        if it in pred[ref] and it in pred[m]:
                            res["scenarios_boot"].setdefault(f"{ref}-{m}", {}).setdefault(s, {})[it] = boot_iid(
                                pred[ref][it][msk] - pred[m][it][msk], args.boot, rng)
                b = res["scenarios_boot"].get(f"{ref}-{m}", {})
                print(f"[counts] Szenarien {ref} - {m:14s} vs@30 je Jahr: " + "  ".join(
                    f"{s} {b[s]['vs@30']['mean']:+.5f} [{b[s]['vs@30']['lo']:+.5f}, {b[s]['vs@30']['hi']:+.5f}]"
                    for s in b if s.isdigit() and "vs@30" in b[s]), flush=True)
    d.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[counts] {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=["run", "summary"], required=True)
    ap.add_argument("--models", nargs="+", default=[], help="name=cg|lh:glob")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--n", type=int, default=20_000, help="Test-Stichtage wie joint_eval")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", nargs="+", default=[], help="summary: counts-npz")
    ap.add_argument("--refs", nargs="+", default=[])
    ap.add_argument("--scen-inputs", nargs="+", default=[], help="summary: scenarios-npz (joint_eval/joint_calib)")
    ap.add_argument("--scen-refs", nargs="+", default=[])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if args.part == "summary":
        run_summary(cfg, exp, args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(cfg, exp, args, device)


if __name__ == "__main__":
    main()

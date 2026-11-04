"""Survival-Kurven fuer Regionen: erstes Feuer IRGENDWO in einer Region R innerhalb von tau Tagen.

Pixelkurven allein legen das Regionsrisiko nicht fest, dafuer braucht es die gemeinsame Verteilung
der Ereigniszeiten. Verglichen wird auf den zufaelligen Test-Stichtagen (surv_eval_test):

  cg_mc     Contagion-STS, Monte Carlo ueber gemeinsame Rollouts (Rao-Blackwell, s. u.); fuer
            R = ganzer Ausschnitt exakt und ohne Zufall (Selbsttest gegen die geschlossene Formel)
  cg_mf     Contagion-STS, Mean-Field (ContagionSTS.region_log_survival, unveraendert genutzt)
  cg_ind    Contagion-STS, Produkt der Pixel-Randkurven, als waeren die Pixel unabhaengig
  lh_ind    STS-LH, Produkt der Pixel-Randkurven (unabhaengig)
  lh_patch  STS-LH, eigener Ausschnittskopf (nur R = ganzer Ausschnitt; getrennt trainierter Kopf)

Rao-Blackwell-Schaetzer fuer R: Bis zum ersten Feuer in R brennt in R nichts. Deshalb gilt exakt

    S_R(tau) = E[ prod_{s <= tau} prod_{i in R} (1 - h_i(s)) ],

wobei der Erwartungswert ueber die Pfade AUSSERHALB von R laeuft, simuliert mit unterdrueckten
Zuendungen in R (Pixel in R brennen in der Simulation nie). Die Zufallsgroesse ist glatt, die Varianz
klein; fuer R = ganzer Ausschnitt gibt es kein Aussen, der Schaetzer ist dann deterministisch.
Zeitkonvention wie ContagionSTS.rollout: ein Feuer am Tag f + 1 wirkt am Tag s + 1 mit D(s - f - 1),
die Kernmitte wirkt nur ueber das beobachtete Feuer.

Schritte (jeweils eigener sbatch-Job, Ausgaben in NEUEN Ordnern, nichts wird ueberschrieben):
  --build-events   Ereigniszeit je Pixel des Ausschnitts fuer die Test-Stichtage
                   -> derived_data/region_eval/pixel_events.npz
  (Standard)       Auswertung -> metrics/paper_region/region_eval<tag>.json
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
from wildfire.eval.metrics import evaluate
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import load_sts

H = 100
P = 13
HALF = P // 2
N = P * P


# ---------------------------------------------------------------------------- Regionen
def regions() -> dict[str, np.ndarray]:
    """Zentrierte Quadrate 1..13 (verschachtelt) und die vier 6x6-Ecken (nicht zentriert)."""
    out = {}
    for s in (1, 3, 5, 7, 9, 11, 13):
        m = np.zeros((P, P), bool)
        lo = HALF - s // 2
        m[lo:lo + s, lo:lo + s] = True
        out[f"sq{s}"] = m.ravel()
    for name, (r0, c0) in {"nw6": (0, 0), "ne6": (0, 7), "sw6": (7, 0), "se6": (7, 7)}.items():
        m = np.zeros((P, P), bool)
        m[r0:r0 + 6, c0:c0 + 6] = True
        out[name] = m.ravel()
    return out


NESTED = ("sq1", "sq3", "sq5", "sq7", "sq9", "sq11", "sq13")


def region_target(ev_px: np.ndarray, t_px: np.ndarray, mask: np.ndarray):
    """Erstes Ereignis in R: (event (n,), t (n,)); t = 100 fuer zensiert."""
    e = ev_px.reshape(len(ev_px), -1)[:, mask]
    t = np.where(e, t_px.reshape(len(t_px), -1)[:, mask], H)
    return e.any(1), np.clip(t.min(1), 1, H)


# ---------------------------------------------------------------------------- Kurven -> Hazards
def hazards_from_logS(logS: np.ndarray) -> np.ndarray:
    """log S (n, H+1) mit S(0) = 1 -> diskrete Hazards (n, H)."""
    return np.clip(-np.expm1(np.diff(logS, axis=1)), 1e-15, 1 - 1e-7)


# ---------------------------------------------------------------------------- Contagion
@torch.no_grad()
def mc_region_logS(net, log_exo, k, logD, masks: dict[str, torch.Tensor], M: int, gen: torch.Generator):
    """Rao-Blackwell-Monte-Carlo: log S_R(0..H) (B, H+1) je Region.

    log_exo (B, H, N): log(1 - h) aus Zuendung und beobachtetem Feuer; k (B, KK, N); logD (B, N, A)."""
    B, Hh, _ = log_exo.shape
    c = net.K * net.K // 2
    le = log_exo.repeat_interleave(M, 0)                                   # (BM, H, N)
    kk = k.repeat_interleave(M, 0)
    lD = logD.repeat_interleave(M, 0)
    ages = torch.arange(lD.shape[-1], device=le.device)
    out = {}
    for name, R in masks.items():
        fired = torch.full((B * M, N), -1, dtype=torch.long, device=le.device)   # Schritt f des ersten Feuers
        cum = torch.zeros(B * M, dtype=torch.float64, device=le.device)
        curve = [cum.clone()]
        for s in range(Hh):
            on = fired >= 0
            age = (s - fired - 1).clamp(min=0, max=lD.shape[-1] - 1)
            b = torch.where(on, lD.gather(2, age[..., None])[..., 0].exp(), torch.zeros_like(le[:, 0]))
            lp = torch.log1p(-(kk * net._unfold(b)).clamp(max=1 - 1e-6))      # (BM, KK, N)
            l1mh = le[:, s] + lp.sum(1) - lp[:, c]                             # log(1 - h_i(s+1))
            cum = cum + l1mh[:, R].sum(1).double()
            curve.append(cum.clone())
            h = -torch.expm1(l1mh)
            new = (~on) & (~R)[None] & (torch.rand(h.shape, generator=gen, device=h.device) < h)
            fired = torch.where(new, torch.full_like(fired, s), fired)
        lc = torch.stack(curve, 1).view(B, M, Hh + 1)                          # log prod (1 - h_R) je Pfad
        out[name] = (torch.logsumexp(lc, 1) - np.log(M)).cpu().numpy()
    return out


@torch.no_grad()
def contagion_curves(ck, X, doy, m_t, s_t, masks_np, M, device, bs, seed):
    net = load_sts(ck, device)
    net.eval()
    masks = {k: torch.from_numpy(v).to(device) for k, v in masks_np.items()}
    gen = torch.Generator(device=device).manual_seed(seed)
    res = {f"cg_{m}": {r: [] for r in masks_np} for m in ("mc", "mf", "ind")}
    full = []
    for a in range(0, X.shape[0], bs):
        x = (X[a:a + bs].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[a:a + bs]).to(device)
        with torch.autocast(device_type=device.type, enabled=False):     # fp32 wie region_log_survival
            obs, aa, k, logD = net.params(x, d)
            log_exo = net.exogenous(obs, aa, k, logD)
            ls_px = net.rollout(log_exo, k, logD)                                # (b, N, H+1) Randkurven
            mc = mc_region_logS(net, log_exo, k, logD, masks, M, gen)
            full.append(F.pad(log_exo.sum(2).cumsum(1), (1, 0)).double().cpu().numpy())
            for r, R in masks.items():
                res["cg_mc"][r].append(mc[r])
                res["cg_ind"][r].append(ls_px[:, R].sum(1).double().cpu().numpy())
                res["cg_mf"][r].append(net.region_log_survival(x, d, R).double().cpu().numpy())
    out = {m: {r: np.concatenate(v) for r, v in d_.items()} for m, d_ in res.items()}
    exact = np.concatenate(full)
    dev = float(np.abs(out["cg_mc"]["sq13"] - exact).max())
    print(f"[region] Selbsttest {Path(ck).name}: MC gegen exakte Ausschnittsformel, max |d log S| = {dev:.2e}", flush=True)
    if dev > 1e-4:
        raise SystemExit("ABBRUCH: Monte-Carlo-Schaetzer weicht fuer den ganzen Ausschnitt von der exakten Formel ab")
    return out, dev


# ---------------------------------------------------------------------------- STS-LH
@torch.no_grad()
def lh_curves(ck, X, doy, m_t, s_t, masks_np, device, bs):
    net = load_sts(ck, device)
    net.eval()
    ind = {r: [] for r in masks_np}
    patch, viol = [], []
    for a in range(0, X.shape[0], bs):
        x = (X[a:a + bs].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[a:a + bs]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_px, raw_pa = net(x, d)
        b = raw_px.shape[0]
        hp = net.px_head.hazards(raw_px.reshape(b * N, -1), H).double().view(b, N, H)
        lsp = F.pad(torch.log1p(-hp.clamp(max=1 - 1e-7)).cumsum(2), (1, 0))       # (b, N, H+1)
        for r, R in masks_np.items():
            ind[r].append(lsp[:, torch.from_numpy(R).to(device)].sum(1).cpu().numpy())
        ha = net.pa_head.hazards(raw_pa, H).double()
        lpa = F.pad(torch.log1p(-ha.clamp(max=1 - 1e-7)).cumsum(1), (1, 0))
        patch.append(lpa.cpu().numpy())
        # Ausschnittskopf gegen Pixelkoepfe: S_Ausschnitt > S_Pixel (also F_Ausschnitt < F_Pixel) ist unmoeglich
        viol.append((lpa[:, None, :] > lsp + 1e-6).any(2).any(1).cpu().numpy())
    return ({"lh_ind": {r: np.concatenate(v) for r, v in ind.items()}, "lh_patch": {"sq13": np.concatenate(patch)}},
            float(np.concatenate(viol).mean()))


# ---------------------------------------------------------------------------- Konsistenz
def nested_violations(curves: dict[str, np.ndarray], tol: float = 1e-6) -> float:
    """Anteil der Stichtage, an denen F_R1(tau) > F_R2(tau) + tol fuer R1 in R2 (verschachtelte Quadrate)."""
    have = [r for r in NESTED if r in curves]
    if len(have) < 2:
        return float("nan")
    bad = np.zeros(len(curves[have[0]]), bool)
    for r1, r2 in zip(have[:-1], have[1:]):
        bad |= (curves[r2] > curves[r1] + tol).any(1)          # log S groesser = F kleiner
    return float(bad.mean())


def compact(m: dict) -> dict:
    out = {"nll": m["nll"], "ibs": m["integrated_brier"], "c_td": m["c_index_td"],
           "dcal_dev": m["d_cal_max_dev"], "dcal_p": m["d_cal_p"], "n": m["n"], "n_events": m["n_events"]}
    for k, v in m["horizons"].items():
        out.update({f"pr@{k}": v["pr_auc"], f"auc@{k}": v["roc_auc"], f"brier@{k}": v["brier"], f"ece@{k}": v["ece"]})
    return out


# ---------------------------------------------------------------------------- Schritte
def build_events(cfg, exp, args):
    from wildfire.data.dataset import CubeReader
    from wildfire.paper.pixel_landmarks import pixel_events
    from wildfire.paper.surv_eval import raw_fire_keys

    out_dir = exp / "derived_data" / "region_eval"
    out = out_dir / "pixel_events.npz"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - wird nicht ueberschrieben")
    me = dict(np.load(exp / "derived_data" / args.surv_dir / "eval_meta.npz"))
    cube = CubeReader(exp)
    keys, _ = raw_fire_keys(Path(cfg["dataset_root"]))
    ev, t = pixel_events(keys, me["L_abs"], me["row"], me["col"], cube.H, cube.W)
    # Selbsttest gegen die gespeicherten Ziele von surv_eval
    pe, pt = ev.reshape(len(ev), -1).any(1), t.reshape(len(t), -1).min(1)
    checks = {"ev_patch": float((pe == me["ev_patch"].astype(bool)).mean()),
              "t_patch": float((pt[pe] == me["t_patch"][pe]).mean()),
              "ev_centre": float((ev[:, HALF, HALF] == me["ev_centre"].astype(bool)).mean())}
    print(f"[region] Selbsttest Ereignisse je Pixel gegen surv_eval: {checks}", flush=True)
    if min(checks.values()) < 0.9999:
        raise SystemExit("ABBRUCH: Ereignisse je Pixel passen nicht zu eval_meta.npz")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ev=ev, t=t, row=me["row"], col=me["col"], L_abs=me["L_abs"])
    print(f"[region] {out}: {ev.shape}, Pixel-Ereignisrate {ev.mean():.4f}", flush=True)


def run_eval(cfg, exp, args, device):
    out_dir = exp / "metrics" / "paper_region"
    out = out_dir / f"region_eval{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderes --tag waehlen")
    sd = exp / "derived_data" / args.surv_dir
    me = dict(np.load(sd / "eval_meta.npz"))
    pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
    assert (pe["row"] == me["row"]).all() and (pe["L_abs"] == me["L_abs"]).all(), "Stichtage passen nicht"
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(me["row"]), size=min(args.n, len(me["row"])), replace=False))
    t0 = time.time()
    X = torch.from_numpy(np.ascontiguousarray(np.load(sd / "eval_X.npy", mmap_mode="r")[idx]))
    print(f"[region] {len(idx):,} Stichtage geladen in {time.time() - t0:.0f} s", flush=True)
    doy = me["doy"][idx].astype(np.float32)
    ev_px, t_px = pe["ev"][idx], pe["t"][idx].astype(int)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    masks = regions()
    targets = {r: region_target(ev_px, t_px, m) for r, m in masks.items()}
    res = {"n": int(len(idx)), "M": args.M, "seed": args.seed, "regions": {r: int(m.sum()) for r, m in masks.items()},
           "event_rate": {r: float(e.mean()) for r, (e, _) in targets.items()}, "models": {}, "selftest_mc_vs_exact": {}}
    print("[region] Ereignisrate bis 100 Tage: " + "  ".join(f"{r} {v:.4f}" for r, v in res["event_rate"].items()), flush=True)

    def score(method, seed_name, curves):
        r_out = res["models"].setdefault(method, {})
        for r, ls in curves.items():
            e, t = targets[r]
            m = compact(evaluate(hazards_from_logS(ls), e, t))
            r_out.setdefault(r, {})[seed_name] = m
        r_out.setdefault("_nested_violation_share", {})[seed_name] = nested_violations(curves)
        print(f"[region] {method:8s} {seed_name:6s} " + "  ".join(
            f"{r}: NLL {r_out[r][seed_name]['nll']:.4f} PR@30 {r_out[r][seed_name]['pr@30']:.3f}"
            for r in ("sq1", "sq5", "sq13", "nw6") if r in curves), flush=True)

    for ck in sorted(glob.glob(args.cg)):
        s = Path(ck).stem.split("_s")[-1]
        cur, dev = contagion_curves(ck, X, doy, m_t, s_t, masks, args.M, device, args.bs, args.seed)
        res["selftest_mc_vs_exact"][s] = dev
        for meth, cv in cur.items():
            score(meth, s, cv)
    for ck in sorted(glob.glob(args.lh)):
        s = Path(ck).stem.split("_s")[-1]
        cur, v = lh_curves(ck, X, doy, m_t, s_t, masks, device, args.bs)
        res.setdefault("lh_patch_below_pixel_share", {})[s] = v
        print(f"[region] STS-LH {s}: Ausschnittskopf unter einem Pixelkopf (F_Ausschnitt < F_Pixel) "
              f"bei {v:.4f} der Stichtage", flush=True)
        for meth, cv in cur.items():
            score(meth, s, cv)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[region] {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--build-events", action="store_true")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cg", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_cg_s?_s?.pt")
    ap.add_argument("--lh", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_lh_s?_s?.pt")
    ap.add_argument("--n", type=int, default=20_000, help="zufaellige Teilmenge der Test-Stichtage")
    ap.add_argument("--M", type=int, default=64, help="Monte-Carlo-Pfade je Stichtag")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if args.build_events:
        build_events(cfg, exp, args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_eval(cfg, exp, args, device)


if __name__ == "__main__":
    main()

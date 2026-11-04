"""Auswertung der gemeinsamen Verteilung auf den zufaelligen Test-Stichtagen (surv_eval_test, 2023-2025).

  --part shapes     Regionskurven "erstes Feuer irgendwo in R" fuer viele Formen: Quadrate und Ecken (Familie
                    rect, wie die Trainingsmasken von lh_region mask) und Formen anderer Art (Ringe, Kreuz,
                    Diagonale, verstreute Pixel). Je Modellfamilie:
                      cg  Contagion: mc (Rao-Blackwell, gemeinsame Zufallszahlen je Region), mf, ind
                      lh  STS-LH: ind (Produkt der Pixel), scaled (Ausschnittskopf, log S mit |R| / 169 skaliert);
                          lh_region fixed: own (eigene Koepfe sq5, sq9, nw6), near (Kopf mit naechster Flaeche,
                          skaliert; Flaeche 1 = Produkt); lh_region mask: mask (Maskenkopf fuer jede Form)
                    Metriken wie region_eval, dazu Verletzungen der Verschachtelung (F_R1 > F_R2 bei R1 in R2):
                    Anteil > 1e-4 und mittlere Hoehe.
  --part scenarios  Gemeinsame Szenarien: je Stichtag M Felder der Erstfeuer-Tage aller 169 Pixel.
                      cg       Simulation des Contagion-Prozesses (wie mc_region_logS ohne Unterdrueckung)
                      cg_shuf  dieselben Ziehungen, je Pixel unabhaengig permutiert: gleiche Randverteilungen,
                               keine Abhaengigkeit (isoliert den Wert der Abhaengigkeit)
                      cg_ind   unabhaengig aus den Mean-Field-Pixelkurven
                      lh_ind   unabhaengig aus den STS-LH-Pixelkurven
                    Scores je tau (7, 30, 100) auf dem Feld Y = 1{T <= tau}: Energy Score, Variogramm-Score
                    (Paare mit Abstand <= 2, p = 0,5), CRPS der Zahl brennender Pixel. Kleiner ist besser.
  --part summary    Mittel ueber die Seeds je Stichtag, gepaarter Bootstrap gegen Referenzmethoden.

--bias: Die Logistic-Hazard-Koepfe und ign_bias von Contagion haben einen festen Bias aus der Ereignisrate der
Trainingsdaten (Pixel 0,086, Ausschnitt 0,534), der nicht im state dict steht; load_sts setzt 0,02 / 0,2.
  default  wie load_sts (alle bisherigen Auswertungen)
  trained  wie im Training (Raten aus args, sonst aus train_event.npy des Wuerfels)
Regionskoepfe von lh_region behalten in beiden Faellen den Trainings-Bias.

Alle Eingaben nur gelesen. Ausgaben NEU unter metrics/paper_joint/ (bricht ab, wenn sie existieren).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config
from wildfire.eval.metrics import evaluate
from wildfire.paper.joint_train import FIXED, STSRegion, load_joint
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.region_eval import (H, HALF, N, P, compact, hazards_from_logS, mc_region_logS, regions,
                                        region_target)
from wildfire.paper.region_eval_crn import boot_diff, nll_per_sample

TAUS = (7, 30, 100)


# ---------------------------------------------------------------------------- Formen
def shapes() -> dict[str, tuple[np.ndarray, str]]:
    """Name -> (Maske (169,), Familie). rect: Rechtecke wie im Maskentraining; other: andere Formen."""
    rg = regions()
    out = {k: (v, "rect") for k, v in rg.items()}
    g = lambda: np.zeros((P, P), bool)          # noqa: E731
    out["ring9_5"] = (rg["sq9"] & ~rg["sq5"], "other")
    out["ring13_9"] = (rg["sq13"] & ~rg["sq9"], "other")
    m = g(); m[HALF, HALF - 3:HALF + 4] = True; m[HALF - 3:HALF + 4, HALF] = True
    out["cross7"] = (m.ravel(), "other")
    m = g(); m[np.arange(P), np.arange(P)] = True
    out["diag13"] = (m.ravel(), "other")
    m = g(); m.ravel()[np.random.default_rng(7).choice(N, 12, replace=False)] = True
    out["scatter12"] = (m.ravel(), "other")
    return out


NEST = [("sq1", "sq3"), ("sq3", "sq5"), ("sq5", "sq7"), ("sq7", "sq9"), ("sq9", "sq11"), ("sq11", "sq13"),
        ("sq1", "cross7"), ("cross7", "sq7"), ("ring9_5", "sq9"), ("ring13_9", "sq13"), ("nw6", "sq13"),
        ("diag13", "sq13"), ("scatter12", "sq13")]


def nest_violations(curves: dict[str, np.ndarray]) -> dict:
    """Je Paar R1 in R2: Anteil der Stichtage mit max_tau (F_R1 - F_R2) > 1e-4 und dessen Mittel."""
    out = {}
    for r1, r2 in NEST:
        if r1 in curves and r2 in curves:
            v = np.clip(np.exp(curves[r2]) - np.exp(curves[r1]), 0, None).max(1)
            out[f"{r1}<{r2}"] = {"share": float((v > 1e-4).mean()), "mean": float(v.mean()), "max": float(v.max())}
    return out


# ---------------------------------------------------------------------------- Bias
def h0_logit(q: float) -> float:
    h0 = -math.expm1(math.log1p(-q) / H)
    return math.log(h0 / (1 - h0))


def set_bias(net, mode: str, cube_rates) -> tuple:
    """Bias der Pixel- und Ausschnittskoepfe bzw. von ign_bias setzen; -> verwendete Raten."""
    a = getattr(net, "_joint_args", {})
    px, pa = (0.02, 0.2) if mode == "default" else (a.get("init_cif_px", cube_rates[0]), a.get("init_cif_patch", cube_rates[1]))
    if getattr(net, "head_kind", "") == "contagion":
        net.ign_bias = h0_logit(0.5 * px)
    else:
        net.px_head.bias = h0_logit(px)
        net.pa_head.bias = h0_logit(pa)
    return px, pa


def load_model(ck, device, mode, cube_rates):
    net = load_joint(ck, device)
    net._joint_args = torch.load(ck, map_location="cpu").get("args", {})
    used = set_bias(net, mode, cube_rates)
    return net, used


# ---------------------------------------------------------------------------- Kurven je Form
@torch.no_grad()
def mf_region(net, log_exo, k, b, R):
    """Wie ContagionSTS.region_log_survival, aber mit dem schon gerollten Brennzustand b (B, H, N)."""
    B, Hh, Nn = b.shape
    b_out = (b * (~R).float()).reshape(B * Hh, Nn)
    lp = torch.log1p(-(k.repeat_interleave(Hh, 0) * net._unfold(b_out)).clamp(max=1 - 1e-6))
    c = net.K * net.K // 2
    l_pred = (lp.sum(1) - lp[:, c]).view(B, Hh, Nn)
    return F.pad((log_exo + l_pred)[..., R].sum(2).cumsum(1), (1, 0)).double().cpu().numpy()


@torch.no_grad()
def cg_shape_curves(net, X, doy, m_t, s_t, sh, M, device, bs, seed):
    masks = {k: torch.from_numpy(v).to(device) for k, (v, _) in sh.items()}
    gen = torch.Generator(device=device).manual_seed(seed)
    res = {m: {r: [] for r in masks} for m in ("mc", "mf", "ind")}
    exact = []
    for a0 in range(0, X.shape[0], bs):
        x = (X[a0:a0 + bs].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[a0:a0 + bs]).to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            obs, aa, k, logD = net.params(x, d)
            log_exo = net.exogenous(obs, aa, k, logD)
            ls_px, b = net.rollout(log_exo, k, logD, keep_b=True)
            st = gen.get_state()
            for r, R in masks.items():
                gen.set_state(st)                                   # gemeinsame Zufallszahlen fuer alle Formen
                res["mc"][r].append(mc_region_logS(net, log_exo, k, logD, {r: R}, M, gen)[r])
                res["mf"][r].append(mf_region(net, log_exo, k, b, R))
                res["ind"][r].append(ls_px[:, R].sum(1).double().cpu().numpy())
            exact.append(F.pad(log_exo.sum(2).cumsum(1), (1, 0)).double().cpu().numpy())
    out = {m: {r: np.concatenate(v) for r, v in d_.items()} for m, d_ in res.items()}
    ex = np.concatenate(exact)
    dev = {m: float(np.abs(out[m]["sq13"] - ex).max()) for m in ("mc", "mf")}
    print(f"[joint] Selbsttest sq13 gegen exakte Ausschnittsformel: {dev}", flush=True)
    if max(dev.values()) > 1e-4:
        raise SystemExit("ABBRUCH: Regionskurve des ganzen Ausschnitts weicht von der exakten Formel ab")
    return out, dev


def lh_parts(net, x, d):
    """Pixel-Token, roh Pixel (B, N, H), roh Ausschnitt (B, H) fuer STSDCTM und STSRegion."""
    B = x.shape[0]
    pix = net.encode(x, d)
    raw_px = net.px_lin(pix).float()
    a = net.pool(net.cls.expand(B, -1, -1), pix, pix, need_weights=False)[0][:, 0]
    return pix, raw_px, net.pa_lin(net.pool_norm(a)).float()


def ls_from_h(h):
    return F.pad(torch.log1p(-h.clamp(max=1 - 1e-7)).cumsum(-1), (1, 0))


@torch.no_grad()
def lh_shape_curves(net, X, doy, m_t, s_t, sh, device, bs):
    masks = {k: torch.from_numpy(v).to(device) for k, (v, _) in sh.items()}
    area = {k: int(v.sum()) for k, (v, _) in sh.items()}
    meth = ["ind", "scaled"]
    if isinstance(net, STSRegion):
        meth += ["own", "near"] if net.mode == "fixed" else ["mask"]
    res = {m: {r: [] for r in masks} for m in meth}
    near_area = {"sq5": 25, "nw6": 36, "sq9": 81}
    for a0 in range(0, X.shape[0], bs):
        x = (X[a0:a0 + bs].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[a0:a0 + bs]).to(device)
        b = x.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pix, raw_px, raw_pa = lh_parts(net, x, d)
            reg = {}
            if "own" in meth:
                reg = {r: net.reg_heads[r](pix, net.fixed_masks[i][None].expand(b, -1)) for i, r in enumerate(FIXED)}
            if "mask" in meth:
                reg = {r: net.region_raw(pix, R) for r, R in masks.items()}
        lsp = ls_from_h(net.px_head.hazards(raw_px.reshape(b * N, -1), H).double().view(b, N, H))
        lpa = ls_from_h(net.pa_head.hazards(raw_pa, H).double())
        own = {r: ls_from_h(net.reg_lh[r].hazards(reg[r], H).double()) for r in FIXED} if "own" in meth else {}
        for r, R in masks.items():
            ind = lsp[:, R].sum(1)
            res["ind"][r].append(ind.cpu().numpy())
            res["scaled"][r].append((lpa * area[r] / N).cpu().numpy())
            if "own" in meth:
                if r in own:
                    res["own"][r].append(own[r].cpu().numpy())
                cands = {1: None, **{v: k for k, v in near_area.items()}, N: "patch"}
                best = min(cands, key=lambda A: abs(math.log(A) - math.log(area[r])))
                src = cands[best]
                near = ind if src is None else (lpa if src == "patch" else own[src]) * area[r] / best
                res["near"][r].append(near.cpu().numpy())
            if "mask" in meth:
                res["mask"][r].append(ls_from_h(net.reg_lh_mask.hazards(reg[r], H).double()).cpu().numpy())
    return {m: {r: np.concatenate(v) for r, v in d_.items() if v} for m, d_ in res.items()}


# ---------------------------------------------------------------------------- Szenarien
@torch.no_grad()
def sample_cg(net, log_exo, k, logD, M, gen):
    """Gemeinsame Ziehungen der Erstfeuer-Tage (B, M, N), 1..H oder H + 1 = kein Feuer bis H."""
    B, Hh, _ = log_exo.shape
    c = net.K * net.K // 2
    le, kk, lD = (v.repeat_interleave(M, 0) for v in (log_exo, k, logD))
    fired = torch.full((B * M, N), -1, dtype=torch.long, device=le.device)
    for s in range(Hh):
        on = fired >= 0
        age = (s - fired - 1).clamp(min=0, max=lD.shape[-1] - 1)
        b = torch.where(on, lD.gather(2, age[..., None])[..., 0].exp(), torch.zeros_like(le[:, 0]))
        lp = torch.log1p(-(kk * net._unfold(b)).clamp(max=1 - 1e-6))
        h = -torch.expm1(le[:, s] + lp.sum(1) - lp[:, c])
        new = (~on) & (torch.rand(h.shape, generator=gen, device=h.device) < h)
        fired = torch.where(new, torch.full_like(fired, s), fired)
    return torch.where(fired >= 0, fired + 1, torch.full_like(fired, Hh + 1)).view(B, M, N)


@torch.no_grad()
def sample_ind(ls, M, gen):
    """Unabhaengige Ziehungen aus Pixelkurven log S (B, N, H+1) -> (B, M, N)."""
    cdf = (-torch.expm1(ls[..., 1:])).contiguous()
    u = torch.rand(cdf.shape[0], cdf.shape[1], M, generator=gen, device=cdf.device, dtype=cdf.dtype)
    return (torch.searchsorted(cdf, u) + 1).permute(0, 2, 1)


def shuffle(T, gen):
    idx = torch.rand(T.shape, generator=gen, device=T.device).argsort(1)
    return T.gather(1, idx)


def pair_index():
    r, c = np.divmod(np.arange(N), P)
    i, j = np.triu_indices(N, 1)
    keep = np.maximum(np.abs(r[i] - r[j]), np.abs(c[i] - c[j])) <= 2
    return i[keep], j[keep]


@torch.no_grad()
def scenario_scores(T, y, pi, pj):
    """T (B, M, N), y (B, N) -> {tau: {es, vs, crps} (B,)}."""
    out = {}
    for tau in TAUS:
        X, Y = (T <= tau).float(), (y <= tau).float()
        es = (X - Y[:, None]).norm(dim=-1).mean(1) - 0.5 * torch.cdist(X, X).mean((1, 2))
        p_neq = (X[..., pi] != X[..., pj]).float().mean(1)
        vs = ((Y[:, pi] != Y[:, pj]).float() - p_neq).pow(2).mean(1)
        Nm, n = X.sum(-1), Y.sum(-1)
        crps = (Nm - n[:, None]).abs().mean(1) - 0.5 * (Nm[:, :, None] - Nm[:, None, :]).abs().mean((1, 2))
        out[tau] = {"es": es.cpu().numpy(), "vs": vs.cpu().numpy(), "crps": crps.cpu().numpy()}
    return out


# ---------------------------------------------------------------------------- Laeufe
def setup(cfg, exp, args, device):
    sd = exp / "derived_data" / args.surv_dir
    me = dict(np.load(sd / "eval_meta.npz"))
    pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
    assert (pe["row"] == me["row"]).all() and (pe["L_abs"] == me["L_abs"]).all(), "Stichtage passen nicht"
    rng = np.random.default_rng(args.seed)                       # dieselbe Teilmenge wie region_eval bei gleichem n
    idx = np.sort(rng.choice(len(me["row"]), size=min(args.n, len(me["row"])), replace=False))
    t0 = time.time()
    X = torch.from_numpy(np.ascontiguousarray(np.load(sd / "eval_X.npy", mmap_mode="r")[idx]))
    print(f"[joint] {len(idx):,} Stichtage geladen in {time.time() - t0:.0f} s", flush=True)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    ev_tr = np.load(exp / "derived_data" / args.cube_dir / "train_event.npy").astype(bool)
    cube_rates = (float(ev_tr.mean()), float(ev_tr.reshape(len(ev_tr), -1).any(1).mean()))
    print(f"[joint] Raten der Trainingsdaten: Pixel {cube_rates[0]:.5f}, Ausschnitt {cube_rates[1]:.4f}; Bias {args.bias}",
          flush=True)
    return (idx, X, me["doy"][idx].astype(np.float32), pe["ev"][idx], pe["t"][idx].astype(int), m_t, s_t, cube_rates,
            me["fire_win_patch"][idx].astype(bool))


def parse_models(specs):
    out = []
    for s in specs:
        name, rest = s.split("=", 1)
        kind, pat = rest.split(":", 1)
        cks = sorted(glob.glob(pat))
        if not cks:
            raise SystemExit(f"ABBRUCH: keine Checkpoints fuer {s}")
        out.append((name, kind, cks))
    return out


def outputs(exp, part, tag):
    d = exp / "metrics" / "paper_joint"
    js, nz = d / f"{part}{tag}.json", d / f"{part}{tag}.npz"
    for p in (js, nz):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    return d, js, nz


def run_shapes(cfg, exp, args, device):
    out_dir, js, nz = outputs(exp, "shapes", args.tag)
    idx, X, doy, ev_px, t_px, m_t, s_t, cube_rates, fw = setup(cfg, exp, args, device)
    sh = shapes()
    tg = {r: region_target(ev_px, t_px, m) for r, (m, _) in sh.items()}
    res = {"n": int(len(idx)), "M": args.M, "bias": args.bias, "shapes": {r: {"area": int(m.sum()), "family": f}
                                                                         for r, (m, f) in sh.items()},
           "event_rate": {r: {str(t): float((e & (tt <= t)).mean()) for t in TAUS} for r, (e, tt) in tg.items()},
           "models": {}, "mean_F": {}, "nest": {}, "bias_rates": {}, "selftest": {}}
    store = {}
    for name, kind, cks in parse_models(args.models):
        for ck in cks:
            seed = Path(ck).stem.rsplit("_s", 1)[1]
            net, used = load_model(ck, device, args.bias, cube_rates)
            res["bias_rates"].setdefault(name, {})[seed] = used
            t0 = time.time()
            if kind == "cg":
                cur, dev = cg_shape_curves(net, X, doy, m_t, s_t, sh, args.M, device, args.bs, args.seed)
                res["selftest"].setdefault(name, {})[seed] = dev
            else:
                cur = lh_shape_curves(net, X, doy, m_t, s_t, sh, device, args.bs)
            for meth, curves in cur.items():
                key = f"{name}_{meth}"
                for r, ls in curves.items():
                    e, t = tg[r]
                    h = hazards_from_logS(ls)
                    res["models"].setdefault(key, {}).setdefault(r, {})[seed] = compact(evaluate(h, e, t))
                    store[f"{key}|{seed}|{r}"] = nll_per_sample(h, e, t).astype(np.float32)
                    res["mean_F"].setdefault(key, {}).setdefault(r, {})[seed] = {
                        str(tau): float(-np.expm1(ls[:, tau]).mean()) for tau in TAUS}
                res["nest"].setdefault(key, {})[seed] = nest_violations(curves)
                print(f"[joint] {key:22s} {seed}  " + "  ".join(
                    f"{r} NLL {res['models'][key][r][seed]['nll']:.4f} F100 {res['mean_F'][key][r][seed]['100']:.4f}"
                    for r in ("sq1", "sq5", "nw6", "ring9_5", "scatter12") if r in curves), flush=True)
            print(f"[joint] {name} {seed}: {time.time() - t0:.0f} s", flush=True)
            del net
            torch.cuda.empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, idx=idx, **{k.replace("|", "__"): v for k, v in store.items()})
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[joint] {js}\n[joint] {nz}", flush=True)


def run_scenarios(cfg, exp, args, device):
    out_dir, js, nz = outputs(exp, "scenarios", args.tag)
    idx, X, doy, ev_px, t_px, m_t, s_t, cube_rates, fw = setup(cfg, exp, args, device)
    y_all = torch.from_numpy(np.where(ev_px, t_px, H + 1).reshape(len(idx), N))
    pi, pj = (torch.from_numpy(v).to(device) for v in pair_index())
    res = {"n": int(len(idx)), "M": args.M, "bias": args.bias, "share_fire_in_window": float(fw.mean()),
           "share_event": float(ev_px.reshape(len(idx), -1).any(1).mean()), "mean": {}, "bias_rates": {}}
    store = {}
    for name, kind, cks in parse_models(args.models):
        for ck in cks:
            seed = Path(ck).stem.rsplit("_s", 1)[1]
            net, used = load_model(ck, device, args.bias, cube_rates)
            res["bias_rates"].setdefault(name, {})[seed] = used
            gen = torch.Generator(device=device).manual_seed(args.seed)
            acc = {}
            t0 = time.time()
            for a0 in range(0, X.shape[0], args.bs):
                x = (X[a0:a0 + args.bs].to(device).float() - m_t) / s_t
                d = torch.from_numpy(doy[a0:a0 + args.bs]).to(device)
                y = y_all[a0:a0 + args.bs].to(device)
                with torch.no_grad():
                    if kind == "cg":
                        with torch.autocast(device_type=device.type, enabled=False):
                            obs, aa, k, logD = net.params(x, d)
                            log_exo = net.exogenous(obs, aa, k, logD)
                            ls_px = net.rollout(log_exo, k, logD)
                            T = sample_cg(net, log_exo, k, logD, args.M, gen)
                        draws = {"cg": T, "cg_shuf": shuffle(T, gen), "cg_ind": sample_ind(ls_px.double(), args.M, gen)}
                    else:
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            _, raw_px, _ = lh_parts(net, x, d)
                        b = x.shape[0]
                        lsp = ls_from_h(net.px_head.hazards(raw_px.reshape(b * N, -1), H).double().view(b, N, H))
                        draws = {"ind": sample_ind(lsp, args.M, gen)}
                for meth, T in draws.items():
                    for tau, sc in scenario_scores(T, y, pi, pj).items():
                        for sname, v in sc.items():
                            acc.setdefault(f"{name}_{meth}|{seed}|{sname}@{tau}", []).append(v)
            for key, v in acc.items():
                v = np.concatenate(v).astype(np.float32)
                store[key] = v
                meth, _, item = key.split("|")
                res["mean"].setdefault(meth, {}).setdefault(item, {})[seed] = {
                    "all": float(v.mean()), "fire_in_window": float(v[fw].mean()), "no_fire": float(v[~fw].mean())}
            for meth in {k.split("|")[0] for k in acc}:
                m_ = res["mean"][meth]
                print(f"[joint] {meth:22s} {seed}  " + "  ".join(
                    f"{it} {m_[it][seed]['all']:.4f}/{m_[it][seed]['fire_in_window']:.4f}"
                    for it in ("es@30", "vs@30", "crps@30", "crps@100")), flush=True)
            print(f"[joint] {name} {seed}: {time.time() - t0:.0f} s", flush=True)
            del net
            torch.cuda.empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, idx=idx, fire_win=fw, **{k.replace("|", "__"): v for k, v in store.items()})
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[joint] {js}\n[joint] {nz}", flush=True)


def run_summary(cfg, exp, args):
    d = exp / "metrics" / "paper_joint"
    out = d / f"summary{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderes --tag waehlen")
    per, idx0, fw = {}, None, None
    for f in args.inputs:
        z = np.load(d / f)
        if idx0 is None:
            idx0 = z["idx"]
        elif not np.array_equal(idx0, z["idx"]):
            raise SystemExit(f"ABBRUCH: {f} hat andere Stichtage")
        if "fire_win" in z.files:
            fw = z["fire_win"]
        for k in z.files:
            if k in ("idx", "fire_win"):
                continue
            meth, seed, item = k.split("__")
            per.setdefault((meth, item), {})[seed] = z[k].astype(np.float64)
    vec = {k: np.mean(list(v.values()), 0) for k, v in per.items()}           # Mittel ueber die Seeds je Stichtag
    rng = np.random.default_rng(args.seed + 11)
    res = {"inputs": args.inputs, "mean": {}, "boot": {}}
    for (meth, item), v in sorted(vec.items()):
        res["mean"].setdefault(item, {})[meth] = {"all": float(v.mean()), "n_seeds": len(per[(meth, item)]),
                                                  **({"fire_in_window": float(v[fw].mean())} if fw is not None else {})}
    for ref in args.refs:
        for (meth, item), v in sorted(vec.items()):
            if meth == ref or (ref, item) not in vec:
                continue
            r = {"all": boot_diff(vec[(ref, item)], v, args.boot, rng)}
            if fw is not None:
                r["fire_in_window"] = boot_diff(vec[(ref, item)][fw], v[fw], args.boot, rng)
            res["boot"].setdefault(item, {})[f"{ref}-{meth}"] = r
            print(f"[joint] {item:14s} {ref} - {meth}: {r['all']['mean']:+.5f} [{r['all']['lo']:+.5f}, {r['all']['hi']:+.5f}]",
                  flush=True)
    d.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[joint] {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=["shapes", "scenarios", "summary"], required=True)
    ap.add_argument("--models", nargs="+", default=[], help="name=cg|lh:glob")
    ap.add_argument("--bias", choices=["default", "trained"], default="trained")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", nargs="+", default=[], help="summary: npz-Dateien unter metrics/paper_joint")
    ap.add_argument("--refs", nargs="+", default=[], help="summary: Referenzmethoden fuer den Bootstrap")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if args.part == "summary":
        run_summary(cfg, exp, args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (run_shapes if args.part == "shapes" else run_scenarios)(cfg, exp, args, device)


if __name__ == "__main__":
    main()

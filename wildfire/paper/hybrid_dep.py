"""Hybride Abhaengigkeitsschicht: STS-LH-Pixelkurven als Raender, Contagion-Kern als Kopplung.

STS-LH liefert gute Randkurven (Val-Pixel-NLL nach joint_calib 0,0282 gegen 0,0308 fuer Contagion-STS), kann aber
nichts Gemeinsames ausdruecken. Contagion-STS koppelt die Pixel, unterschaetzt aber bei der bedingten Vorhersage das
Niveau nach einem Ereignis rund dreifach. Die Hybride nimmt von STS-LH die Raender und von Contagion-STS nur den Kern
kappa_ij und die Brenndauer D_j:

    log(1 - h_i(s | Pfad)) = log(1 - h_i^LH(s)) + lam_i(s) * [ sum_j log(1 - kappa_ij b_j(s-1))
                                                              - sum_j log(1 - kappa_ij bbar_j(s-1)) ]

  b_j      simulierter Brennzustand (Feuer im Pfad, mit D_j abklingend), wie im Contagion-Prozess
  bbar_j   erwarteter Brennzustand aus den STS-LH-Raendern (Mean Field): bbar_j(s) = sum_u f_j^LH(u) D_j(s-u),
           f^LH(u) = S^LH(u-1) h^LH(u)
  lam_i(s) = min(1, log(1 - h_i^LH(s)) / sum_j log(1 - kappa_ij bbar_j(s-1)))   lokale Kappung
Das ist ein Contagion-Prozess mit Zuendung log(1 - a~) = log(1 - h^LH) - lam * sum_j log(1 - kappa bbar) <= 0 und einer
je Pixel und Tag mit lam skalierten Ausbreitung. Wo die erwartete Ausbreitung den Rand-Hazard von STS-LH
ueberstiege, wird die Kopplung dort abgeschwaecht (a~ = 0), statt den Rand zu veraendern; der Mean-Field-Rand ist
damit fuer jedes theta exakt STS-LH (Selbsttest). Die Kopplung wirkt nur ueber die Abweichung des Pfads vom
Erwarteten: brennt ein Nachbar, steigt der Hazard ueber STS-LH; bleibt es ruhig, sinkt er darunter (STS-LH enthaelt
die erwartete Ausbreitung schon). Beobachtetes Feuer vor dem Stichtag steckt in h^LH.

Niveau der Kopplung: theta = (t_1 .. t_R, t_D) auf den Val-Jahren gefittet (Teil fit), R = K // 2 Ringe des Kerns:
    logit kappa'_ij = logit kappa_ij + t_ring(j - i),   D' = D(dur + t_D)
Ziel ist die NLL der Restzeit der bedingten Vorhersage (Zeilen wie joint_cond auf den Val-Stichtagen, wie lh_adj in
joint_refc), Mustersuche mit gemeinsamen Zufallszahlen. Kern und Dauer kommen aus Contagion-STS nach joint_calib
(d_k), STS-LH ebenfalls nach joint_calib (d_px). Kein Netz wird neu trainiert.

Die Monte-Carlo-Bausteine sind Kopien von joint_calib.mc_pixel, region_eval.mc_region_logS und
joint_eval.sample_cg mit dem zusaetzlichen Faktor lam (die Originale bleiben unveraendert).

Teile (je Seed-Paar STS-LH s / Contagion-STS s, Name des Seeds = --seed-name):
  fit        theta auf Val -> metrics/paper_joint/<hyb-dir>/hyb_s<seed>.json
  cond       bedingte Vorhersage wie joint_cond (dieselben Zeilen): hyb_cond (theta), hyb0_cond (theta = 0)
             -> metrics/paper_joint/cond<tag>.{json,npz}
  shapes     Regionskurven wie joint_eval shapes (Rao-Blackwell-MC): hyb_mc -> shapes<tag>.{json,npz}
             (Form sq1 = Mittelpixel: Pruefung des Rands gegen lh_ind)
  scenarios  wie joint_eval scenarios: hyb_sim, hyb_shuf -> scenarios<tag>.npz; dazu wie joint_counts die Zahl
             brennender Pixel (64 Ziehungen) -> counts<tag>.npz
Die Ausgaben haben das Format der Originale, damit deren Zusammenfassungen (joint_eval/joint_refc/joint_counts
--part summary) sie zusammen mit den vorhandenen Laeufen auswerten.

Alle Eingaben nur gelesen; bestehende Ausgaben werden nicht ueberschrieben (Abbruch).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config
from wildfire.eval.metrics import evaluate
from wildfire.paper import joint_cond, joint_eval
from wildfire.paper.joint_calib import cube_rates_of, load_calibrated, nll_curve, norm_tensors, path_mean
from wildfire.paper.joint_counts import count_stats
from wildfire.paper.joint_eval import lh_parts, pair_index, scenario_scores, shapes, shuffle
from wildfire.paper.joint_refc import cond_rows, val_events
from wildfire.paper.region_eval import H, N, compact, hazards_from_logS, region_target
from wildfire.paper.region_eval_crn import nll_per_sample


# ---------------------------------------------------------------------------- Bausteine
@dataclass
class Coupled:
    exo: torch.Tensor     # (B, H, N) log(1 - a~)
    kappa: torch.Tensor   # (B, KK, N)
    logD: torch.Tensor    # (B, N, A)
    lam: torch.Tensor     # (B, H, N) lokale Kappung der Ausbreitung, 1 = ungekappt


def ring_index(K: int, device) -> torch.Tensor:
    q = torch.arange(K * K, device=device)
    dr, dc = q // K - K // 2, q % K - K // 2
    return torch.maximum(dr.abs(), dc.abs())                       # 0 = Mitte, 1 .. K//2


@torch.no_grad()
def raw_parts(lh, cg, x, d):
    """-> log(1 - h^LH) (B, H, N), Kern-Logits (B, KK, N) und Dauer-Rohwert (B, N) von Contagion-STS."""
    b = x.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        _, raw_px, _ = lh_parts(lh, x, d)
    hz = lh.px_head.hazards(raw_px.reshape(b * N, -1), H).double().view(b, N, H)
    l1mh = torch.log1p(-hz.clamp(max=1 - 1e-7)).transpose(1, 2).float().contiguous()
    with torch.autocast(device_type=x.device.type, enabled=False):
        pix = cg.encode_input(x[:, -cg.hist:], d).float()
        kl = cg.ker(pix).transpose(1, 2).contiguous()
        dur = cg.dur(pix)[..., 0]
    return l1mh, kl, dur


@torch.no_grad()
def couple(cg, l1mh, kl, dur, theta: torch.Tensor):
    """theta (R + 1,) -> Coupled und Statistik der lokalen Kappung."""
    ring = ring_index(cg.K, kl.device)
    R = cg.K // 2
    shift = torch.where(ring > 0, theta[(ring - 1).clamp(min=0)], torch.zeros_like(theta[:1]).expand(ring.numel()))
    kappa = torch.sigmoid(kl + shift.float()[None, :, None])
    logD = cg.log_burning(dur + float(theta[R]))
    Hh = l1mh.shape[1]
    c = cg.K * cg.K // 2
    h = -torch.expm1(l1mh)
    S_prev = torch.exp(F.pad(l1mh.cumsum(1)[:, :-1], (0, 0, 1, 0)))
    f = (S_prev * h).transpose(1, 2)                                  # (B, N, H): erstes Feuer an Tag u + 1
    lpred = torch.zeros_like(l1mh)
    for s in range(1, Hh):                                            # Schritt 0: noch nichts vorhergesagt
        Dk = logD[..., :s].flip(-1).exp()                             # D(s - 1 - u) fuer das Feuer von Tag u + 1
        b = (f[..., :s] * Dk).sum(-1).clamp(max=1.0)
        lp = torch.log1p(-(kappa * cg._unfold(b)).clamp(max=1 - 1e-6))
        lpred[:, s] = lp.sum(1) - lp[:, c]
    neg = lpred < 0
    lam = torch.where(neg, (l1mh / torch.where(neg, lpred, -torch.ones_like(lpred))).clamp(max=1.0),
                      torch.ones_like(lpred))
    exo = (l1mh - lam * lpred).clamp(max=0.0)
    capped = lam < 1
    st = {"cap_share": float(capped.float().mean()), "lam_mean_capped": float(lam[capped].mean()) if capped.any()
          else 1.0, "spread_share_of_marginal": float((lam * lpred).sum() / l1mh.sum().clamp(max=-1e-30))}
    return Coupled(exo, kappa, logD, lam), st


@torch.no_grad()
def _spread(cg, kk, b):
    lp = torch.log1p(-(kk * cg._unfold(b)).clamp(max=1 - 1e-6))
    c = cg.K * cg.K // 2
    return lp.sum(1) - lp[:, c]


@torch.no_grad()
def rollout_l(cg, cp: Coupled):
    """Mean-Field-Pixelkurven log S (B, N, H+1) der Hybride (wie ContagionSTS.rollout, mit lam)."""
    B, Hh, Nn = cp.exo.shape
    logS = cp.exo.new_zeros(B, Nn)
    S = cp.exo.new_ones(B, Nn)
    b_pred = cp.exo.new_zeros(B, Nn)
    out, fs = [logS], []
    for s in range(Hh):
        l1mh = cp.exo[:, s] + cp.lam[:, s] * _spread(cg, cp.kappa, b_pred)
        fs.append(S * -torch.expm1(l1mh))
        logS = logS + l1mh
        S = torch.exp(logS)
        Dk = cp.logD[..., :s + 1].flip(-1).exp()
        b_pred = (torch.stack(fs, -1) * Dk).sum(-1).clamp(max=1.0)
        out.append(logS)
    return torch.stack(out, -1)


@torch.no_grad()
def mc_pixel_l(cg, cp: Coupled, land, pix, gen, start=None, trig=None):
    """Wie joint_calib.mc_pixel, mit lam: log prod (1 - h_p) (R, H+1) je Zeile (Pfad), Pixel p unterdrueckt."""
    R = land.numel()
    Hh, A = cp.exo.shape[1], cp.logD.shape[-1]
    dev = cp.exo.device
    rows = torch.arange(R, device=dev)
    kk = cp.kappa[land]
    sup = torch.zeros(R, N, dtype=torch.bool, device=dev)
    sup[rows, pix] = True
    fired = torch.full((R, N), -1, dtype=torch.long, device=dev)
    cum = torch.zeros(R, dtype=torch.float64, device=dev)
    curve = [cum]
    cols = torch.arange(N, device=dev)[None]
    for s in range(Hh):
        on = fired >= 0
        age = (s - fired - 1).clamp(min=0, max=A - 1)
        b = torch.where(on, cp.logD[land[:, None], cols, age].exp(), torch.zeros(R, N, device=dev))
        l1mh = cp.exo[land, s] + cp.lam[land, s] * _spread(cg, kk, b)
        act = torch.ones(R, dtype=torch.bool, device=dev) if start is None else s >= start
        cum = cum + torch.where(act, l1mh[rows, pix].double(), torch.zeros_like(cum))
        curve.append(cum)
        u = torch.rand((R, N), generator=gen, device=dev)
        new = (~on) & (~sup) & act[:, None] & (u < -torch.expm1(l1mh))
        if trig is not None:
            new = new | (trig & (start - 1 == s)[:, None])
        fired = torch.where(new, torch.full_like(fired, s), fired)
    return torch.stack(curve, 1)


@torch.no_grad()
def mc_region_l(cg, cp: Coupled, Rm: torch.Tensor, M: int, gen) -> np.ndarray:
    """Wie region_eval.mc_region_logS fuer eine Region Rm (N,) bool, mit lam -> log S_R (B, H+1)."""
    B, Hh, _ = cp.exo.shape
    le, kk, lD, lam = (v.repeat_interleave(M, 0) for v in (cp.exo, cp.kappa, cp.logD, cp.lam))
    fired = torch.full((B * M, N), -1, dtype=torch.long, device=le.device)
    cum = torch.zeros(B * M, dtype=torch.float64, device=le.device)
    curve = [cum.clone()]
    for s in range(Hh):
        on = fired >= 0
        age = (s - fired - 1).clamp(min=0, max=lD.shape[-1] - 1)
        b = torch.where(on, lD.gather(2, age[..., None])[..., 0].exp(), torch.zeros_like(le[:, 0]))
        l1mh = le[:, s] + lam[:, s] * _spread(cg, kk, b)
        cum = cum + l1mh[:, Rm].sum(1).double()
        curve.append(cum.clone())
        new = (~on) & (~Rm)[None] & (torch.rand(l1mh.shape, generator=gen, device=le.device) < -torch.expm1(l1mh))
        fired = torch.where(new, torch.full_like(fired, s), fired)
    lc = torch.stack(curve, 1).view(B, M, Hh + 1)
    return (torch.logsumexp(lc, 1) - np.log(M)).cpu().numpy()


@torch.no_grad()
def sample_l(cg, cp: Coupled, M: int, gen):
    """Wie joint_eval.sample_cg, mit lam: Erstfeuer-Tage (B, M, N), H + 1 = kein Feuer bis H."""
    B, Hh, _ = cp.exo.shape
    le, kk, lD, lam = (v.repeat_interleave(M, 0) for v in (cp.exo, cp.kappa, cp.logD, cp.lam))
    fired = torch.full((B * M, N), -1, dtype=torch.long, device=le.device)
    for s in range(Hh):
        on = fired >= 0
        age = (s - fired - 1).clamp(min=0, max=lD.shape[-1] - 1)
        b = torch.where(on, lD.gather(2, age[..., None])[..., 0].exp(), torch.zeros_like(le[:, 0]))
        h = -torch.expm1(le[:, s] + lam[:, s] * _spread(cg, kk, b))
        new = (~on) & (torch.rand(h.shape, generator=gen, device=h.device) < h)
        fired = torch.where(new, torch.full_like(fired, s), fired)
    return torch.where(fired >= 0, fired + 1, torch.full_like(fired, Hh + 1)).view(B, M, N)


def load_pair(exp, args, device, rates):
    lh, _ = load_calibrated(exp, args.calib_dir, args.lh_ckpt, device, rates)
    cg, _ = load_calibrated(exp, args.calib_dir, args.cg_ckpt, device, rates)
    if getattr(cg, "head_kind", "") != "contagion":
        raise SystemExit("ABBRUCH: --cg-ckpt ist kein Contagion-Modell")
    return lh.eval(), cg.eval()


def theta_file(exp, args) -> Path:
    return exp / "metrics" / "paper_joint" / args.hyb_dir / f"hyb_s{args.seed_name}.json"


def load_theta(exp, args, device):
    f = theta_file(exp, args)
    if not f.exists():
        raise SystemExit(f"ABBRUCH: keine Hybrid-Kalibrierung {f}")
    return torch.tensor(json.loads(f.read_text())["theta"], dtype=torch.float64, device=device)


def blocks_X(X, idx, bs, m_t, s_t, doy, device):
    for a0 in range(0, len(idx), bs):
        ii = idx[a0:a0 + bs]
        x = (X[torch.from_numpy(ii)].to(device).float() - m_t) / s_t
        yield a0, ii, x, torch.from_numpy(doy[ii]).to(device)


@torch.no_grad()
def selftest(lh, cg, X, doy, m_t, s_t, device):
    """Mean-Field-Rollout der Hybride muss die STS-LH-Kurven reproduzieren, fuer theta = 0 und einen starken Kern."""
    x = (X[:16].to(device).float() - m_t) / s_t
    d = torch.from_numpy(doy[:16]).to(device)
    l1mh, kl, dur = raw_parts(lh, cg, x, d)
    ls_lh = F.pad(l1mh.cumsum(1), (0, 0, 1, 0)).transpose(1, 2)
    out = {}
    for name, th in (("theta0", 0.0), ("theta+2", 2.0)):
        theta = torch.full((cg.K // 2 + 1,), th, dtype=torch.float64, device=device)
        theta[-1] = 0.0
        cp, st = couple(cg, l1mh, kl, dur, theta)
        out[name] = {"max_dev": float((rollout_l(cg, cp) - ls_lh).abs().max()), **st}
    print(f"[hyb] Selbsttest Mean-Field-Rand = STS-LH: {out}", flush=True)
    if max(v["max_dev"] for v in out.values()) > 1e-3:
        raise SystemExit("ABBRUCH: Mean-Field-Rand der Hybride weicht von STS-LH ab")
    return out


# ---------------------------------------------------------------------------- bedingte Vorhersage (Bloecke)
@torch.no_grad()
def cond_block(cg, l1mh, kl, dur, theta, loc, pix, e, t, k, trig_l, M, seed):
    """Zeilen eines Blocks (Stichtage lokal loc) -> log S der Restzeit (R, H+1) je Zeile."""
    cp, st = couple(cg, l1mh, kl, dur, theta)
    lr = loc.repeat_interleave(M)
    gen = torch.Generator(device=l1mh.device).manual_seed(seed)
    ls = path_mean(mc_pixel_l(cg, cp, lr, pix.repeat_interleave(M), gen, start=k.repeat_interleave(M),
                              trig=trig_l[lr]), M)
    return ls, st


# ---------------------------------------------------------------------------- Teil fit
def run_fit(cfg, exp, args, device):
    out = theta_file(exp, args)
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon")
    m_t, s_t = norm_tensors(cfg, args, device)
    rates = cube_rates_of(exp, args)
    lh, cg = load_pair(exp, args, device, rates)
    me_v, evf_v, tf_v = val_events(exp, args)
    cand_v, k_v, trig_v, rows_v = cond_rows(me_v, evf_v, tf_v, argparse.Namespace(kmax=args.kmax, dmax=args.dmax, n=0,
                                                                                   seed=args.seed))
    rng = np.random.default_rng(args.seed)
    pos = np.sort(rng.choice(cand_v.size, size=min(args.fit_landmarks, cand_v.size), replace=False))
    sel = np.isin(rows_v["land"], pos)
    rows = {k_: v[sel] for k_, v in rows_v.items()}
    loc_all = np.searchsorted(pos, rows["land"])
    X = torch.from_numpy(np.ascontiguousarray(
        np.load(exp / "derived_data" / args.surv_dir / "val_X.npy", mmap_mode="r")[cand_v[pos]]))
    doy = me_v["doy"][cand_v[pos]].astype(np.float32)
    print(f"[hyb] Val: {cand_v.size:,} Stichtage mit Ausloeser, davon {pos.size:,} fuer den Fit, {loc_all.size:,} "
          f"Zeilen; Ereignis bis r=7 {(rows['e'] & (rows['t'] - rows['k'] <= 7)).mean():.4f}", flush=True)
    st0 = selftest(lh, cg, X, doy, m_t, s_t, device)
    parts = [raw_parts(lh, cg, x, d) for _, _, x, d in blocks_X(X, np.arange(pos.size), args.bs, m_t, s_t, doy, device)]
    L1, KL, DUR = (torch.cat([p[i] for p in parts]) for i in range(3))
    trig = torch.from_numpy(trig_v[pos]).to(device)
    tg = {k_: torch.from_numpy(np.asarray(rows[k_])).to(device) for k_ in ("pix", "e", "t", "k")}
    loc_t = torch.from_numpy(loc_all).to(device)
    per = max(1, args.rows // args.M)
    R = cg.K // 2
    cache = {}

    def objective(th: tuple) -> float:
        key = tuple(round(v, 4) for v in th)
        if key in cache:
            return cache[key][0]
        t0 = time.time()
        theta = torch.tensor(key, dtype=torch.float64, device=device)
        tot, cs = 0.0, []
        for bi, r0 in enumerate(range(0, loc_all.size, per)):
            r1 = min(loc_all.size, r0 + per)
            ll = torch.unique(loc_t[r0:r1])
            loc = torch.searchsorted(ll, loc_t[r0:r1])
            ls, st = cond_block(cg, L1[ll], KL[ll], DUR[ll], theta, loc, tg["pix"][r0:r1], tg["e"][r0:r1],
                                tg["t"][r0:r1], tg["k"][r0:r1], trig[ll], args.M, args.seed * 100_003 + bi)
            tot += float(nll_curve(ls, tg["e"][r0:r1], tg["t"][r0:r1], k0=tg["k"][r0:r1]).sum())
            cs.append(st["cap_share"])
        v = tot / loc_all.size
        cache[key] = (v, float(np.mean(cs)))
        print(f"[hyb]   theta {list(key)}: Val-NLL {v:.5f}  Kappung {np.mean(cs):.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
        return v

    base = tuple([0.0] * (R + 1))
    v0 = objective(base)
    best = min((tuple([g] * R + [0.0]) for g in args.line), key=objective)       # gemeinsamer Ring-Shift
    step = args.step
    while step >= args.min_step - 1e-9 and len(cache) < args.max_evals:
        moved = False
        for i in range(R + 1):
            for sgn in (1, -1):
                cand = list(best)
                cand[i] += sgn * step
                cand = tuple(cand)
                if objective(cand) < objective(best) - 1e-6:
                    best, moved = cand, True
                    break
            if moved or len(cache) >= args.max_evals:
                break
        if not moved:
            step /= 2
    v1 = objective(best)
    res = {"lh_ckpt": args.lh_ckpt, "cg_ckpt": args.cg_ckpt, "calib_dir": args.calib_dir, "theta": list(best),
           "theta_names": [f"ring{r}" for r in range(1, R + 1)] + ["dur"], "K": cg.K, "val_nll_theta0": v0,
           "val_nll": v1, "n_val_landmarks": int(pos.size), "n_val_rows": int(loc_all.size), "M": args.M,
           "selftest": st0, "trace": [{"theta": list(k_), "nll": v[0], "cap_share": v[1]} for k_, v in cache.items()]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[hyb] theta {list(best)}: Val-NLL {v0:.5f} -> {v1:.5f} ({len(cache)} Auswertungen)\n[hyb] {out}", flush=True)


# ---------------------------------------------------------------------------- Teil cond
def run_cond(cfg, exp, args, device):
    d = exp / "metrics" / "paper_joint"
    js, nz = d / f"cond{args.tag}.json", d / f"cond{args.tag}.npz"
    for p in (js, nz):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    m_t, s_t = norm_tensors(cfg, args, device)
    rates = cube_rates_of(exp, args)
    lh, cg = load_pair(exp, args, device, rates)
    theta = load_theta(exp, args, device)
    zero = torch.zeros_like(theta)
    cand, X, doy, trig, rows = joint_cond.select(exp, args)
    selftest(lh, cg, X, doy, m_t, s_t, device)
    land_all = rows["land"]
    tg = {k_: torch.from_numpy(np.asarray(rows[k_])).to(device) for k_ in ("pix", "e", "t", "k")}
    trig_t = torch.from_numpy(trig)
    per = max(1, args.rows // args.M)
    meths = {"hyb_cond": theta, "hyb0_cond": zero}
    acc = {m: {"nll": [], **{f"F{r}": [] for r in joint_cond.RS}} for m in meths}
    clamp = []
    t0 = time.time()
    for bi, r0 in enumerate(range(0, land_all.size, per)):
        r1 = min(land_all.size, r0 + per)
        ls_land = np.unique(land_all[r0:r1])
        loc = torch.from_numpy(np.searchsorted(ls_land, land_all[r0:r1])).to(device)
        x = (X[torch.from_numpy(ls_land)].to(device).float() - m_t) / s_t
        dd = torch.from_numpy(doy[ls_land]).to(device)
        l1mh, kl, dur = raw_parts(lh, cg, x, dd)
        tr = trig_t[torch.from_numpy(ls_land)].to(device)
        e, t, k = (tg[k_][r0:r1] for k_ in ("e", "t", "k"))
        for m, th in meths.items():
            ls, st = cond_block(cg, l1mh, kl, dur, th, loc, tg["pix"][r0:r1], e, t, k, tr, args.M,
                                args.seed * 100_003 + bi)
            acc[m]["nll"].append(nll_curve(ls, e, t, k0=k).float().cpu().numpy())
            for key, v in joint_cond.residual_F(ls, k).items():
                acc[m][key].append(v.float().cpu().numpy())
            if m == "hyb_cond":
                clamp.append(st["cap_share"])
        if bi % 20 == 0:
            print(f"[hyb]   Block {bi}: {r1:,}/{land_all.size:,} Zeilen ({time.time() - t0:.0f} s)", flush=True)
    res = {"n_landmarks": int(cand.size), "n_rows": int(land_all.size), "kmax": args.kmax, "dmax": args.dmax,
           "M": args.M, "theta": theta.cpu().tolist(), "cap_share": float(np.mean(clamp)), "models": {}}
    store = {}
    seed = args.seed_name
    for m, o in acc.items():
        p = {k_: np.concatenate(v) for k_, v in o.items()}
        res["models"].setdefault(m, {})[seed] = joint_cond.describe(p, rows)
        for item, v in p.items():
            store[f"{m}__{seed}__{item}"] = v.astype(np.float32)
        a = res["models"][m][seed]["all"]
        print(f"[hyb] {m:14s} {seed}  NLL " + "  ".join(f"d{b} {a[b]['nll']:.4f}" for b in joint_cond.BANDS if b in a)
              + f"  AUC@7 {a['all']['auc@7']:.3f}  F@7 {a['all']['F@7']:.4f} (beob. {a['all']['obs@7']:.4f})",
              flush=True)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, cand=cand, **{f"row_{k_}": v for k_, v in rows.items()}, **store)
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[hyb] {js}\n[hyb] {nz}", flush=True)


# ---------------------------------------------------------------------------- Teile shapes / scenarios
def run_shapes(cfg, exp, args, device):
    out_dir, js, nz = joint_eval.outputs(exp, "shapes", args.tag)
    args.bias = f"calib:{args.calib_dir}"
    idx, X, doy, ev_px, t_px, m_t, s_t, rates, fw = joint_eval.setup(cfg, exp, args, device)
    lh, cg = load_pair(exp, args, device, rates)
    theta = load_theta(exp, args, device)
    sh = shapes()
    if args.shapes:
        sh = {r: v for r, v in sh.items() if r in args.shapes}
    tg = {r: region_target(ev_px, t_px, m) for r, (m, _) in sh.items()}
    masks = {k_: torch.from_numpy(v).to(device) for k_, (v, _) in sh.items()}
    st0 = selftest(lh, cg, X, doy, m_t, s_t, device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    cur = {r: [] for r in masks}
    clamp = []
    t0 = time.time()
    for a0, ii, x, d in blocks_X(X, np.arange(len(idx)), args.bs, m_t, s_t, doy, device):
        l1mh, kl, dur = raw_parts(lh, cg, x, d)
        cp, st = couple(cg, l1mh, kl, dur, theta)
        clamp.append(st["cap_share"])
        g_state = gen.get_state()
        for r, R in masks.items():
            gen.set_state(g_state)                                    # gemeinsame Zufallszahlen fuer alle Formen
            cur[r].append(mc_region_l(cg, cp, R, args.M, gen))
        if (a0 // args.bs) % 100 == 0:
            print(f"[hyb]   {a0:,}/{len(idx):,} ({time.time() - t0:.0f} s)", flush=True)
    seed, key = args.seed_name, "hyb_mc"
    res = {"n": int(len(idx)), "M": args.M, "bias": args.bias, "theta": theta.cpu().tolist(),
           "cap_share": float(np.mean(clamp)), "selftest": st0, "models": {}, "mean_F": {}}
    store = {}
    curves = {r: np.concatenate(v) for r, v in cur.items()}
    for r, ls in curves.items():
        e, t = tg[r]
        h = hazards_from_logS(ls)
        res["models"].setdefault(key, {}).setdefault(r, {})[seed] = compact(evaluate(h, e, t))
        store[f"{key}|{seed}|{r}"] = nll_per_sample(h, e, t).astype(np.float32)
        res["mean_F"].setdefault(key, {}).setdefault(r, {})[seed] = {
            str(tau): float(-np.expm1(ls[:, tau]).mean()) for tau in joint_eval.TAUS}
    res["nest"] = {key: {seed: joint_eval.nest_violations(curves)}}
    print(f"[hyb] {key} {seed}  " + "  ".join(
        f"{r} NLL {res['models'][key][r][seed]['nll']:.4f}" for r in ("sq1", "sq5", "nw6", "ring9_5", "scatter12")
        if r in curves) + f"  ({time.time() - t0:.0f} s)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, idx=idx, **{k_.replace("|", "__"): v for k_, v in store.items()})
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[hyb] {js}\n[hyb] {nz}", flush=True)


def run_scenarios(cfg, exp, args, device):
    out_dir, js, nz = joint_eval.outputs(exp, "scenarios", args.tag)
    d_ = exp / "metrics" / "paper_joint"
    cjs, cnz = d_ / f"counts{args.tag}.json", d_ / f"counts{args.tag}.npz"
    for p in (cjs, cnz):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    args.bias = f"calib:{args.calib_dir}"
    idx, X, doy, ev_px, t_px, m_t, s_t, rates, fw = joint_eval.setup(cfg, exp, args, device)
    lh, cg = load_pair(exp, args, device, rates)
    theta = load_theta(exp, args, device)
    y_all = torch.from_numpy(np.where(ev_px, t_px, H + 1).reshape(len(idx), N))
    pi, pj = (torch.from_numpy(v).to(device) for v in pair_index())
    seed = args.seed_name
    gen = torch.Generator(device=device).manual_seed(args.seed)
    gen_c = torch.Generator(device=device).manual_seed(args.seed)
    acc, acc_c = {}, {}
    t0 = time.time()
    for a0, ii, x, d in blocks_X(X, np.arange(len(idx)), args.bs, m_t, s_t, doy, device):
        y = y_all[a0:a0 + len(ii)].to(device)
        l1mh, kl, dur = raw_parts(lh, cg, x, d)
        cp, _ = couple(cg, l1mh, kl, dur, theta)
        T = sample_l(cg, cp, args.M, gen)
        for meth, TT in (("sim", T), ("shuf", shuffle(T, gen))):
            for tau, sc in scenario_scores(TT, y, pi, pj).items():
                for sname, v in sc.items():
                    acc.setdefault(f"hyb_{meth}|{seed}|{sname}@{tau}", []).append(v)
        Tc = sample_l(cg, cp, args.M_counts, gen_c)
        for meth, TT in (("sim", Tc), ("shuf", shuffle(Tc, gen_c))):
            for item, v in count_stats(TT, y, gen_c).items():
                acc_c.setdefault(f"hyb_{meth}__{seed}__{item}", []).append(v)
    res = {"n": int(len(idx)), "M": args.M, "M_counts": args.M_counts, "theta": theta.cpu().tolist(), "mean": {}}
    store = {k_.replace("|", "__"): np.concatenate(v).astype(np.float32) for k_, v in acc.items()}
    for k_, v in store.items():
        meth, _, item = k_.split("__")
        res["mean"].setdefault(meth, {})[item] = {"all": float(v.mean()), "fire_in_window": float(v[fw].mean())}
    print(f"[hyb] Szenarien {seed}: " + "  ".join(f"{m} crps@30 {res['mean'][m]['crps@30']['all']:.4f} es@30 "
                                                 f"{res['mean'][m]['es@30']['all']:.4f}" for m in res["mean"])
          + f"  ({time.time() - t0:.0f} s)", flush=True)
    store_c = {k_: np.concatenate(v).astype(np.float32) for k_, v in acc_c.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, idx=idx, fire_win=fw, **store)
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    np.savez_compressed(cnz, idx=idx, fire_win=fw, **store_c)
    cres = {"n": int(len(idx)), "M": args.M_counts, "mean": {
        k_: {"all": float(v.mean()), "fire_in_window": float(v[fw].mean())} for k_, v in store_c.items()}}
    cjs.write_text(json.dumps({"args": vars(args), **cres}, indent=1))
    print(f"[hyb] {js}\n[hyb] {nz}\n[hyb] {cjs}\n[hyb] {cnz}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=["fit", "cond", "shapes", "scenarios"], required=True)
    ap.add_argument("--lh-ckpt", required=True)
    ap.add_argument("--cg-ckpt", required=True)
    ap.add_argument("--seed-name", required=True, help="Name des Seed-Paars in den Ausgaben (z. B. 0)")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--hyb-dir", default="hybrid_calib")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--kmax", type=int, default=70)
    ap.add_argument("--dmax", type=int, default=4)
    ap.add_argument("--n", type=int, default=None, help="cond: Stichtage (0 = alle); shapes/scenarios: 20.000")
    ap.add_argument("--M", type=int, default=None, help="Pfade (fit 8, cond 16, shapes/scenarios 32)")
    ap.add_argument("--M-counts", type=int, default=64)
    ap.add_argument("--rows", type=int, default=65_536)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--fit-landmarks", type=int, default=400)
    ap.add_argument("--line", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--min-step", type=float, default=0.125)
    ap.add_argument("--max-evals", type=int, default=40)
    ap.add_argument("--shapes", nargs="*", default=[], help="shapes: nur diese Formen (Rauchtest)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.n is None:
        args.n = 0 if args.part == "cond" else 20_000
    if args.M is None:
        args.M = {"fit": 8, "cond": 16}.get(args.part, 32)
    if args.part != "fit" and not args.tag:
        raise SystemExit("ABBRUCH: --tag noetig")
    {"fit": run_fit, "cond": run_cond, "shapes": run_shapes, "scenarios": run_scenarios}[args.part](cfg, exp, args,
                                                                                                    device)


if __name__ == "__main__":
    main()

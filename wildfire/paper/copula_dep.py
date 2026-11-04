"""Gemeinsames unbeobachtetes Risiko auf STS-LH: additive Gamma-Frailty mit exakt erhaltenen Raendern.

Die bedingte Vorhersage zeigt fuer alle Modelle (STS-LH, Contagion-STS, Hawkes) dieselbe Niveau-Luecke: nach einem
beobachteten Feuer im Ausschnitt brennen die Nachbarn rund dreimal haeufiger als vorhergesagt (F@7 0,02 gegen 0,069).
Die Hybride (hybrid_dep.py) zeigt, dass Ausbreitung allein das nicht erklaert. Hier wird stattdessen ein gemeinsamer
latenter Risikofaktor Z_i auf die STS-LH-Pixelkurven gelegt; gegeben Z sind die Pixel unabhaengig:

    S_i(t | Z) = exp(-Z_i x_i(t)),   Z_i = sum_c W_ic Y_c,   Y_c ~ Gamma(alpha_c, Rate alpha_c)  (Mittel 1)

  c = 0         globale Komponente des Ausschnitts, W_i0 = pi
  c = 1 .. N    lokale Komponenten an jedem Pixel, W_ic = (1 - pi) k(i, c) / sum_c' k(i, c'),
                k(i, c) = exp(-|i - c|^2 / (2 ell^2))  (euklidisch in Pixeln)
  alpha_0 = 1 / v_g,  alpha_c = 1 / v_l (c >= 1);  E Z_i = 1, Cov(Z_i, Z_j) = pi^2 v_g + (1 - pi)^2 v_l sum_c w_ic w_jc

Randtreue: S_i(t) = E exp(-Z_i x_i(t)) = exp(-phi_i(x_i(t))), phi_i(x) = sum_c alpha_c log(1 + W_ic x / alpha_c).
x_i(t) = phi_i^{-1}(-log S_i^LH(t)) wird je Stichtag numerisch invertiert (phi_i streng monoton). Damit ist jeder
Pixelrand fuer jedes theta exakt die kalibrierte STS-LH-Kurve (Selbsttest). Mit pi = 1 ist das die Clayton-Copula
(Variante "copg", ein Parameter v_g); die volle Variante "cop" hat 4 Parameter (v_g, v_l, pi, ell).

Warum Frailty und keine Gauss-Copula: Die Laplace-Transformation gibt jede Regionskurve geschlossen,
    log S_R(t) = -sum_c alpha_c log(1 + sum_{i in R} W_ic x_i(t) / alpha_c),
Ziehungen sind trivial (Gamma, dann unabhaengige Pixel), und die bedingte Vorhersage nach einem exakt beobachteten
Muster (Ausloeser an Tag k, alle anderen bis k ruhig) ist eine Posterior-Rechnung ueber Y. Bei einer Gauss-Copula
braeuchte jede Regionskurve 169-dimensionale Normal-Wahrscheinlichkeiten und das Bedingen auf diskrete Erstfeuer-Tage
waere nur per Verwerfen moeglich.

Bedingte Vorhersage (Zeilen wie joint_cond): Posterior von Y gegeben "alle Pixel bis Tag k ruhig, Ausloeser j an
Tag k" ist proportional zu Gamma(alpha_c, alpha_c + sum_i W_ic x_i(k)) * prod_j (exp(Z_j D_j) - 1), D_j = x_j(k) -
x_j(k-1). Importance Sampling: Zuordnung jedes Ausloesers zu einer Komponente nacheinander (Polya-Urne, exakt fuer
den Faktor prod_j Z_j), dann Y | Zuordnung ~ Gamma(alpha_c + n_c, alpha_c + sum_i W_ic x_i(k)); Gewicht = Produkt der
Normierer * prod_j g(Z_j D_j) mit g(u) = (e^u - 1) / u. Restzeit des Pixels p:
    S_p(k + r | Info) = sum_m w_m exp(-Z_p^(m) (x_p(k + r) - x_p(k))) / sum_m w_m.
Fuer pi = 1 ist die Zuordnung deterministisch (nur g weicht ab).

Fit (Teil fit) auf den Val-Jahren wie hybrid_dep / lh_adj: mittlere NLL der bedingten Restzeit (Zeilen wie
joint_refc.cond_rows), gemeinsame Zufallszahlen; copg per Liniensuche + Goldener Schnitt, cop per Mustersuche ab dem
besten copg. Zur Kontrolle die Val-Regions-NLL (geschlossen) einiger Formen bei theta = 0 und den gefitteten theta.

Teile (je STS-LH-Seed, Name = --seed-name):
  fit        -> metrics/paper_joint/<cop-dir>/cop_s<seed>.json
  cond       -> metrics/paper_joint/cond<tag>.{json,npz}        cop_cond, copg_cond
  shapes     -> metrics/paper_joint/shapes<tag>.{json,npz}      cop_cf, copg_cf (geschlossen, kein MC)
  scenarios  -> metrics/paper_joint/scenarios<tag>.{json,npz}   cop_sim/_shuf, copg_sim/_shuf
                metrics/paper_joint/counts<tag>.{json,npz}      wie joint_counts (64 Ziehungen)
Formate wie die Originale, damit joint_eval/joint_refc/joint_counts --part summary sie mit den vorhandenen Laeufen
auswerten. Kein Netz wird neu trainiert; alle Eingaben nur gelesen; bestehende Ausgaben fuehren zum Abbruch.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config
from wildfire.eval.metrics import evaluate
from wildfire.paper import joint_cond, joint_eval
from wildfire.paper.joint_calib import cube_rates_of, load_calibrated, nll_curve, norm_tensors
from wildfire.paper.joint_counts import count_stats
from wildfire.paper.joint_eval import lh_parts, ls_from_h, pair_index, scenario_scores, shapes, shuffle
from wildfire.paper.joint_refc import cond_rows, val_events
from wildfire.paper.region_eval import H, N, P, compact, hazards_from_logS, region_target
from wildfire.paper.region_eval_crn import nll_per_sample

VARIANTS = ("cop", "copg")
VAL_SHAPES = ("sq1", "sq5", "sq9", "sq13", "ring9_5")


# ---------------------------------------------------------------------------- Modell
class Frailty:
    """Gewichte W (N, C), Formen alpha (C,) und die Inversion x = phi^{-1}(Lambda) je Pixel."""

    def __init__(self, v_g: float, v_l: float, pi: float, ell: float, device, G: int = 4096):
        self.dev = device
        self.global_only = pi >= 1.0 - 1e-12 or v_l <= 0
        r, c = np.divmod(np.arange(N), P)
        d2 = (r[:, None] - r[None]) ** 2 + (c[:, None] - c[None]) ** 2
        k = np.exp(-d2 / (2.0 * max(ell, 1e-3) ** 2))
        k /= k.sum(1, keepdims=True)
        if self.global_only:
            W = np.full((N, 1), 1.0)
            alpha = np.array([1.0 / max(v_g, 1e-12)])
        else:
            W = np.concatenate([np.full((N, 1), pi), (1.0 - pi) * k], 1)
            alpha = np.concatenate([[1.0 / max(v_g, 1e-12)], np.full(N, 1.0 / max(v_l, 1e-12))])
        self.W = torch.tensor(W, dtype=torch.float64, device=device)
        self.alpha = torch.tensor(alpha, dtype=torch.float64, device=device)
        self.zero = v_g <= 0 and (self.global_only or v_l <= 0)
        if not self.global_only:                                  # Tabelle phi_i(e^u) fuer den Startwert der Inversion
            self.u = torch.linspace(-40.0, 60.0, G, dtype=torch.float64, device=device)
            xg = self.u.exp()[:, None].expand(-1, N)                              # (G, N)
            self.tab = torch.cat([self._phi(xg[g0:g0 + 256]) for g0 in range(0, G, 256)]).T.contiguous()  # (N, G)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """x (..., N) -> phi_i(x_i) (..., N)."""
        a = self.alpha
        return (a * torch.log1p(x[..., None] * self.W / a)).sum(-1)

    def _dphi_du(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        return (self.W * x[..., None] / (1.0 + x[..., None] * self.W / a)).sum(-1)

    @torch.no_grad()
    def invert(self, lam: torch.Tensor, newton: int = 3, chunk: int = 4) -> torch.Tensor:
        """Lambda (B, N, T) = -log S^LH -> x (B, N, T) mit phi_i(x) = Lambda (float64)."""
        lam = lam.double().clamp(min=0.0)
        if self.zero:
            return lam.clone()
        if self.global_only:
            a = self.alpha[0]
            return a * torch.expm1((lam / a).clamp(max=700.0))                        # x < 1e300
        B, Nn, T = lam.shape
        flat = lam.permute(1, 0, 2).reshape(Nn, B * T).contiguous()          # (N, B*T)
        j = torch.searchsorted(self.tab, flat).clamp(1, self.u.numel() - 1)
        f0, f1 = self.tab.gather(1, j - 1), self.tab.gather(1, j)
        u0, u1 = self.u[j - 1], self.u[j]
        w = ((flat - f0) / (f1 - f0).clamp(min=1e-300)).clamp(0.0, 1.0)
        u = (u0 + w * (u1 - u0)).view(Nn, B, T).permute(1, 0, 2).contiguous()
        out = torch.empty_like(lam)
        for b0 in range(0, B, chunk):
            uu, ll = u[b0:b0 + chunk], lam[b0:b0 + chunk]
            for _ in range(newton):
                x = uu.exp().transpose(1, 2)                                    # (b, T, N)
                f = self._phi(x).transpose(1, 2) - ll
                d = self._dphi_du(x).transpose(1, 2).clamp(min=1e-300)
                uu = (uu - f / d).clamp(max=690.0)
            out[b0:b0 + chunk] = uu.exp()
        small = lam < self.tab[None, :, :1]                                       # unter der Tabelle: phi(x) = x
        return torch.where(small, lam, out)

    def region_logS(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """x (B, N, T), R (N,) bool -> log S_R (B, T) geschlossen."""
        s = torch.einsum("bnt,nc->btc", x[:, R], self.W[R])
        return -(self.alpha * torch.log1p(s / self.alpha)).sum(-1)

    def sample_Y(self, shape_c: torch.Tensor, rate_c: torch.Tensor, gen) -> torch.Tensor:
        return torch._standard_gamma(shape_c.contiguous(), generator=gen) / rate_c

    @torch.no_grad()
    def sample_T(self, x: torch.Tensor, M: int, gen) -> torch.Tensor:
        """Gemeinsame Ziehungen der Erstfeuer-Tage (B, M, N), 1..H oder H + 1."""
        B = x.shape[0]
        a = self.alpha.expand(B, M, -1)
        Y = self.sample_Y(a, a, gen)
        Z = torch.einsum("bmc,nc->bmn", Y, self.W)
        E = -torch.log(torch.rand(Z.shape, generator=gen, device=x.device, dtype=torch.float64).clamp(min=1e-300))
        thr = E / Z.clamp(min=1e-300)                                             # Feuer, sobald x(t) >= E / Z
        xs = x[:, :, 1:].contiguous()                                             # (B, N, H)
        T = torch.searchsorted(xs.view(B * N, H), thr.permute(0, 2, 1).reshape(B * N, M).contiguous()) + 1
        T = T.view(B, N, M).permute(0, 2, 1)
        return T.clamp(max=H + 1)


def log_g(u: torch.Tensor) -> torch.Tensor:
    """log((e^u - 1) / u), stabil."""
    big = u > 30
    tiny = u < 1e-8
    us = torch.where(big | tiny, torch.ones_like(u), u)
    mid = torch.log(torch.expm1(us)) - torch.log(us)
    return torch.where(big, u - torch.log(u.clamp(min=1e-300)), torch.where(tiny, 0.5 * u, mid))


@torch.no_grad()
def cond_posterior(fr: Frailty, x: torch.Tensor, kk: torch.Tensor, trig: torch.Tensor, M: int, gen):
    """Stichtage eines Blocks: x (L, N, H+1), k (L,), trig (L, N) -> log-Gewichte (L, M), Z (L, M, N), ESS (L,)."""
    L = x.shape[0]
    dev = x.device
    li = torch.arange(L, device=dev)
    xk = x[li, :, kk]                                                             # (L, N)
    dk = (xk - x[li, :, kk - 1]).clamp(min=0.0)
    rate = fr.alpha + xk @ fr.W                                                   # (L, C)
    C = fr.W.shape[1]
    n = torch.zeros(L, M, C, dtype=torch.float64, device=dev)
    logw = torch.zeros(L, M, dtype=torch.float64, device=dev)
    ntr = trig.sum(1)
    order = torch.argsort((~trig).to(torch.int8), dim=1, stable=True)            # Ausloeser zuerst
    if C > 1:
        for s in range(int(ntr.max()) if L else 0):
            valid = (s < ntr)[:, None]                                            # (L, 1)
            jw = fr.W[order[:, s]]                                                # (L, C)
            pr = jw[:, None, :] * (fr.alpha + n) / rate[:, None, :]               # (L, M, C)
            tot = pr.sum(-1)
            cum = pr.cumsum(-1)
            u = torch.rand(L, M, 1, generator=gen, device=dev, dtype=torch.float64) * tot[..., None]
            c = torch.searchsorted(cum, u).clamp(max=C - 1)
            n = n + valid[..., None] * torch.zeros_like(n).scatter_(-1, c, 1.0)
            logw = logw + torch.where(valid, torch.log(tot.clamp(min=1e-300)), torch.zeros_like(tot))
    else:
        n = n + ntr[:, None, None].double()
    Y = fr.sample_Y(fr.alpha + n, rate[:, None, :].expand(-1, M, -1), gen)
    Z = torch.einsum("lmc,nc->lmn", Y, fr.W)
    logw = logw + (trig[:, None, :] * log_g(Z * dk[:, None, :])).sum(-1)
    lw = logw - torch.logsumexp(logw, 1, keepdim=True)
    ess = 1.0 / torch.exp(2 * lw).sum(1)
    return lw, Z, ess


@torch.no_grad()
def cond_curves(x, lw, Z, loc, pix, k, chunk: int = 2048):
    """Zeilen (Stichtag loc, Pixel pix, Tag k) -> log S der Restzeit (R, H+1); fuer t <= k gleich 0."""
    T = x.shape[-1]
    tt = torch.arange(T, device=x.device)
    out = []
    for r0 in range(0, loc.numel(), chunk):
        l, p, kr = loc[r0:r0 + chunk], pix[r0:r0 + chunk], k[r0:r0 + chunk]
        xr = x[l, p]                                                              # (R, T)
        idx = torch.maximum(tt[None], kr[:, None])
        dx = xr.gather(1, idx) - xr.gather(1, kr[:, None])                        # (R, T)
        zp = Z[l, :, p]                                                           # (R, M)
        out.append(torch.logsumexp(lw[l][:, :, None] - zp[:, :, None] * dx[:, None, :], 1))
    return torch.cat(out).clamp(max=0.0)


# ---------------------------------------------------------------------------- Eingaben
@torch.no_grad()
def lh_logS(lh, x, d) -> torch.Tensor:
    """Kalibrierte STS-LH-Pixelkurven log S (B, N, H+1) in float64."""
    b = x.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        _, raw_px, _ = lh_parts(lh, x, d)
    return ls_from_h(lh.px_head.hazards(raw_px.reshape(b * N, -1), H).double().view(b, N, H))


def blocks_X(X, idx, bs, m_t, s_t, doy, device):
    for a0 in range(0, len(idx), bs):
        ii = idx[a0:a0 + bs]
        x = (X[torch.from_numpy(ii)].to(device).float() - m_t) / s_t
        yield a0, ii, x, torch.from_numpy(doy[ii]).to(device)


def lh_curves_all(lh, X, doy, m_t, s_t, device, bs) -> torch.Tensor:
    out = [lh_logS(lh, x, d) for _, _, x, d in blocks_X(X, np.arange(X.shape[0]), bs, m_t, s_t, doy, device)]
    return torch.cat(out)


def frailty_of(theta: dict, device) -> Frailty:
    return Frailty(theta["v_g"], theta.get("v_l", 0.0), theta.get("pi", 1.0), theta.get("ell", 1.0), device)


def cop_file(exp, args) -> Path:
    return exp / "metrics" / "paper_joint" / args.cop_dir / f"cop_s{args.seed_name}.json"


def load_thetas(exp, args) -> dict:
    f = cop_file(exp, args)
    if not f.exists():
        raise SystemExit(f"ABBRUCH: keine Frailty-Anpassung {f}")
    js = json.loads(f.read_text())
    return {"cop": js["cop"]["theta"], "copg": js["copg"]["theta"]}


@torch.no_grad()
def selftest(ls: torch.Tensor, device, thetas: dict, gen) -> dict:
    """Rand = STS-LH: Inversion, Form sq1 geschlossen und Ziehungen (Haeufigkeit F(30) gegen die Kurve)."""
    out = {}
    lam = -ls[:8]
    R1 = torch.zeros(N, dtype=torch.bool, device=device)
    R1[N // 2] = True
    for name, th in thetas.items():
        fr = frailty_of(th, device)
        x = fr.invert(lam)
        ok = (lam <= 50.0) & (x < 1e299)                                        # S >= 2e-22 und ungekappt
        dev_inv = float(torch.where(ok, fr._phi(x.transpose(1, 2)).transpose(1, 2) - lam, 0.0).abs().max())
        dev_sq1 = float(torch.where(ok[:, N // 2], fr.region_logS(x, R1) - ls[:8, N // 2], 0.0).abs().max())
        clamped = float((~ok & (lam > 0)).double().mean())
        M = 4000
        T = fr.sample_T(x[:2], M, gen)
        f_emp = (T <= 30).double().mean(1)                                        # (2, N)
        f_lh = -torch.expm1(ls[:2, :, 30])
        z = (f_emp - f_lh) / torch.sqrt(f_lh * (1 - f_lh) / M + 1e-12)
        z = z * ((f_lh >= 0.01) & (f_lh <= 0.99))                                # Normalnaeherung nur dort
        out[name] = {"theta": th, "max_dev_inversion": dev_inv, "max_dev_sq1": dev_sq1, "share_clamped": clamped,
                     "sample_F30_max_abs_z": float(z.abs().max()), "sample_F30_mean_dev": float((f_emp - f_lh).mean())}
    print(f"[cop] Selbsttest Rand = STS-LH: {json.dumps(out)}", flush=True)
    if max(max(v["max_dev_inversion"], v["max_dev_sq1"]) for v in out.values()) > 1e-6:
        raise SystemExit("ABBRUCH: Rand der Frailty weicht von STS-LH ab")
    if max(v["sample_F30_max_abs_z"] for v in out.values()) > 6:
        raise SystemExit("ABBRUCH: Ziehungen treffen den STS-LH-Rand nicht")
    return out


def load_lh(exp, args, device, rates):
    lh, _ = load_calibrated(exp, args.calib_dir, args.lh_ckpt, device, rates)
    return lh.eval()


# ---------------------------------------------------------------------------- Teil fit
def run_fit(cfg, exp, args, device):
    out = cop_file(exp, args)
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon")
    m_t, s_t = norm_tensors(cfg, args, device)
    lh = load_lh(exp, args, device, cube_rates_of(exp, args))
    me_v, evf_v, tf_v = val_events(exp, args)
    cand_v, k_v, trig_v, rows_v = cond_rows(me_v, evf_v, tf_v, argparse.Namespace(kmax=args.kmax, dmax=args.dmax, n=0,
                                                                                   seed=args.seed))
    rng = np.random.default_rng(args.seed)
    pos = np.sort(rng.choice(cand_v.size, size=min(args.fit_landmarks, cand_v.size), replace=False))
    sel = np.isin(rows_v["land"], pos)
    rows = {k_: v[sel] for k_, v in rows_v.items()}
    loc_all = np.searchsorted(pos, rows["land"])
    valX = np.load(exp / "derived_data" / args.surv_dir / "val_X.npy", mmap_mode="r")
    X = torch.from_numpy(np.ascontiguousarray(valX[cand_v[pos]]))
    doy = me_v["doy"][cand_v[pos]].astype(np.float32)
    LS = lh_curves_all(lh, X, doy, m_t, s_t, device, args.bs)                    # (n, N, H+1)
    obs7 = float((rows["e"] & (rows["t"] - rows["k"] <= 7)).mean())
    print(f"[cop] Val: {cand_v.size:,} Stichtage mit Ausloeser, davon {pos.size:,} fuer den Fit, {loc_all.size:,} "
          f"Zeilen; Ereignis bis r=7 {obs7:.4f}", flush=True)
    # Kontrolle: Val-Regionen auf zufaelligen Val-Stichtagen (alle, nicht nur mit Feuer)
    n_all = len(me_v["row"])
    rpos = np.sort(rng.choice(n_all, size=min(args.val_shape_landmarks, n_all), replace=False))
    LSr = lh_curves_all(lh, torch.from_numpy(np.ascontiguousarray(valX[rpos])),
                        me_v["doy"][rpos].astype(np.float32), m_t, s_t, device, args.bs)
    sh = {r: shapes()[r][0] for r in VAL_SHAPES}
    tg_sh = {r: region_target(evf_v[rpos], tf_v[rpos], m) for r, m in sh.items()}
    masks = {r: torch.from_numpy(m).to(device) for r, m in sh.items()}
    trig = torch.from_numpy(trig_v[pos]).to(device)
    kk_l = torch.from_numpy(k_v[pos]).to(device)
    tg = {k_: torch.from_numpy(np.asarray(rows[k_])).to(device) for k_ in ("pix", "e", "t", "k")}
    loc_t = torch.from_numpy(loc_all).to(device)
    st0 = selftest(LS, device, {"cop": {"v_g": 0.5, "v_l": 1.0, "pi": 0.5, "ell": 2.0}, "copg": {"v_g": 0.5}},
                   torch.Generator(device=device).manual_seed(args.seed))
    cache = {}

    def evaluate_theta(th: dict, tag: str) -> float:
        key = (tag,) + tuple(round(float(v), 5) for v in th.values())
        if key in cache:
            return cache[key]["nll"]
        t0 = time.time()
        fr = frailty_of(th, device)
        gen = torch.Generator(device=device).manual_seed(args.seed * 7919 + 1)
        tot, F7, ess = 0.0, 0.0, []
        for l0 in range(0, pos.size, args.lblock):
            l1 = min(pos.size, l0 + args.lblock)
            x = fr.invert(-LS[l0:l1])
            lw, Z, es = cond_posterior(fr, x, kk_l[l0:l1], trig[l0:l1], args.M, gen)
            ess.append(es)
            m = (loc_t >= l0) & (loc_t < l1)
            ls = cond_curves(x, lw, Z, loc_t[m] - l0, tg["pix"][m], tg["k"][m])
            tot += float(nll_curve(ls, tg["e"][m], tg["t"][m], k0=tg["k"][m]).sum())
            F7 += float(joint_cond.residual_F(ls, tg["k"][m])["F7"].sum())
        v = tot / loc_all.size
        es = torch.cat(ess)
        # Val-Regionen (geschlossen)
        reg = {}
        for r, R in masks.items():
            lsr = torch.cat([fr.region_logS(fr.invert(-LSr[a:a + 256]), R) for a in range(0, LSr.shape[0], 256)])
            e, t = tg_sh[r]
            reg[r] = float(nll_per_sample(hazards_from_logS(lsr.cpu().numpy()), e, t).mean())
        cache[key] = {"tag": tag, "theta": dict(th), "nll": v, "F7": F7 / loc_all.size, "obs7": obs7,
                      "ess_mean": float(es.mean()), "ess_p05": float(es.quantile(0.05)), "val_region_nll": reg}
        print(f"[cop]   {tag} {json.dumps({k_: round(float(v_), 4) for k_, v_ in th.items()})}: Val-NLL {v:.5f}  "
              f"F@7 {F7 / loc_all.size:.4f} (beob. {obs7:.4f})  ESS {float(es.mean()):.0f}/{args.M} "
              f"(p05 {float(es.quantile(0.05)):.0f})  Regionen " + " ".join(f"{r} {q:.4f}" for r, q in reg.items())
              + f"  ({time.time() - t0:.0f} s)", flush=True)
        return v

    # copg: Liniensuche in log v_g, dann Goldener Schnitt
    base = evaluate_theta({"v_g": 0.0}, "copg")
    grid = [float(g) for g in args.line]
    vals = [evaluate_theta({"v_g": g}, "copg") for g in grid]
    i = int(np.argmin(vals))
    lo = math.log(grid[max(i - 1, 0)])
    hi = math.log(grid[min(i + 1, len(grid) - 1)])
    gr = (math.sqrt(5) - 1) / 2
    a_, b_ = lo, hi
    c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
    for _ in range(args.golden):
        if evaluate_theta({"v_g": math.exp(c_)}, "copg") < evaluate_theta({"v_g": math.exp(d_)}, "copg"):
            b_ = d_
        else:
            a_ = c_
        c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
    cg_best = min((v for v in cache.values() if v["tag"] == "copg"), key=lambda v: v["nll"])
    # cop: Mustersuche in (log v_g, log v_l, logit pi, log ell) ab dem besten copg
    def unpack(z):
        return {"v_g": math.exp(z[0]), "v_l": math.exp(z[1]), "pi": 1 / (1 + math.exp(-z[2])),
                "ell": min(math.exp(z[3]), 13.0)}

    z = [math.log(max(cg_best["theta"]["v_g"], 1e-3)), math.log(max(cg_best["theta"]["v_g"], 1e-3)), 0.0,
         math.log(2.0)]
    best = evaluate_theta(unpack(z), "cop")
    starts = [[z[0], z[1] + d1, z[2] + d2, z[3] + d3] for d1 in (0.0, 1.0) for d2 in (-1.5, 1.5) for d3 in (0.0, 0.7)]
    for s in starts:
        v = evaluate_theta(unpack(s), "cop")
        if v < best - 1e-6:
            best, z = v, s
    step = args.step
    n_cop = lambda: sum(1 for v in cache.values() if v["tag"] == "cop")      # noqa: E731
    while step >= args.min_step - 1e-9 and n_cop() < args.max_evals:
        moved = False
        for i in range(4):
            for sgn in (1, -1):
                cand = list(z)
                cand[i] += sgn * step
                v = evaluate_theta(unpack(cand), "cop")
                if v < best - 1e-6:
                    best, z, moved = v, cand, True
                    break
            if moved or n_cop() >= args.max_evals:
                break
        if not moved:
            step /= 2
    cop_best = min((v for v in cache.values() if v["tag"] == "cop"), key=lambda v: v["nll"])
    zero = next(v for v in cache.values() if v["tag"] == "copg" and v["theta"]["v_g"] == 0.0)
    res = {"lh_ckpt": args.lh_ckpt, "calib_dir": args.calib_dir, "n_val_landmarks": int(pos.size),
           "n_val_rows": int(loc_all.size), "n_val_shape_landmarks": int(rpos.size), "M": args.M, "selftest": st0,
           "zero": zero, "copg": cg_best, "cop": cop_best, "trace": list(cache.values())}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[cop] theta 0: Val-NLL {base:.5f}; copg {cg_best['theta']}: {cg_best['nll']:.5f}; cop {cop_best['theta']}: "
          f"{cop_best['nll']:.5f} ({len(cache)} Auswertungen)\n[cop] {out}", flush=True)


# ---------------------------------------------------------------------------- Teil cond
def run_cond(cfg, exp, args, device):
    d = exp / "metrics" / "paper_joint"
    js, nz = d / f"cond{args.tag}.json", d / f"cond{args.tag}.npz"
    for p in (js, nz):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    m_t, s_t = norm_tensors(cfg, args, device)
    lh = load_lh(exp, args, device, cube_rates_of(exp, args))
    thetas = load_thetas(exp, args)
    cand, X, doy, trig, rows = joint_cond.select(exp, args)
    land_all = rows["land"]
    tg = {k_: torch.from_numpy(np.asarray(rows[k_])).to(device) for k_ in ("pix", "e", "t", "k")}
    land_t = torch.from_numpy(land_all).to(device)
    trig_t = torch.from_numpy(trig).to(device)
    assert trig.any(1).all(), "Stichtag ohne Ausloeser"
    # k je Stichtag, auch fuer Stichtage ohne Zeilen -> aus den Ereignissen (wie joint_cond.select)
    pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
    evf = pe["ev"].reshape(-1, N)[cand]
    tf = pe["t"].reshape(-1, N)[cand].astype(np.int64)
    k_land = np.where(evf, tf, H + 1).min(1)
    assert (k_land[land_all] == rows["k"]).all()
    k_t = torch.from_numpy(k_land).to(device)
    frs = {m: frailty_of(th, device) for m, th in thetas.items()}
    acc = {m: {"nll": [], **{f"F{r}": [] for r in joint_cond.RS}} for m in frs}
    ess = {m: [] for m in frs}
    gen = torch.Generator(device=device).manual_seed(args.seed * 7919 + 2)
    st0 = None
    t0 = time.time()
    for bi, (a0, ii, x_in, dd) in enumerate(blocks_X(X, np.arange(cand.size), args.lblock, m_t, s_t, doy, device)):
        LSb = torch.cat([lh_logS(lh, x_in[j:j + args.bs], dd[j:j + args.bs]) for j in range(0, len(ii), args.bs)])
        if st0 is None:
            st0 = selftest(LSb, device, thetas, torch.Generator(device=device).manual_seed(args.seed))
        m = (land_t >= a0) & (land_t < a0 + len(ii))
        for name, fr in frs.items():
            x = fr.invert(-LSb)
            lw, Z, es = cond_posterior(fr, x, k_t[a0:a0 + len(ii)], trig_t[a0:a0 + len(ii)], args.M, gen)
            ess[name].append(es.cpu())
            ls = cond_curves(x, lw, Z, land_t[m] - a0, tg["pix"][m], tg["k"][m])
            acc[name]["nll"].append(nll_curve(ls, tg["e"][m], tg["t"][m], k0=tg["k"][m]).float().cpu().numpy())
            for key, v in joint_cond.residual_F(ls, tg["k"][m]).items():
                acc[name][key].append(v.float().cpu().numpy())
        if bi % 20 == 0:
            print(f"[cop]   Block {bi}: {a0 + len(ii):,}/{cand.size:,} Stichtage ({time.time() - t0:.0f} s)", flush=True)
    res = {"n_landmarks": int(cand.size), "n_rows": int(land_all.size), "kmax": args.kmax, "dmax": args.dmax,
           "M": args.M, "thetas": thetas, "selftest": st0,
           "ess": {m: {"mean": float(torch.cat(v).mean()), "p05": float(torch.cat(v).quantile(0.05))}
                   for m, v in ess.items()}, "models": {}}
    store = {}
    seed = args.seed_name
    for name, o in acc.items():
        p = {k_: np.concatenate(v) for k_, v in o.items()}
        key = f"{name}_cond"
        res["models"].setdefault(key, {})[seed] = joint_cond.describe(p, rows)
        for item, v in p.items():
            store[f"{key}__{seed}__{item}"] = v.astype(np.float32)
        a = res["models"][key][seed]["all"]
        print(f"[cop] {key:12s} {seed}  NLL " + "  ".join(f"d{b} {a[b]['nll']:.4f}" for b in joint_cond.BANDS if b in a)
              + f"  AUC@7 {a['all']['auc@7']:.3f}  F@7 {a['all']['F@7']:.4f} (beob. {a['all']['obs@7']:.4f})  "
              f"ESS {res['ess'][name]['mean']:.0f}/{args.M}", flush=True)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, cand=cand, **{f"row_{k_}": v for k_, v in rows.items()}, **store)
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[cop] {js}\n[cop] {nz}", flush=True)


# ---------------------------------------------------------------------------- Teile shapes / scenarios
def run_shapes(cfg, exp, args, device):
    out_dir, js, nz = joint_eval.outputs(exp, "shapes", args.tag)
    args.bias = f"calib:{args.calib_dir}"
    idx, X, doy, ev_px, t_px, m_t, s_t, rates, fw = joint_eval.setup(cfg, exp, args, device)
    lh = load_lh(exp, args, device, rates)
    thetas = load_thetas(exp, args)
    sh = shapes()
    if args.shapes:
        sh = {r: v for r, v in sh.items() if r in args.shapes}
    tg = {r: region_target(ev_px, t_px, m) for r, (m, _) in sh.items()}
    masks = {k_: torch.from_numpy(v).to(device) for k_, (v, _) in sh.items()}
    frs = {m: frailty_of(th, device) for m, th in thetas.items()}
    cur = {m: {r: [] for r in masks} for m in frs}
    st0 = None
    t0 = time.time()
    for a0, ii, x_in, d in blocks_X(X, np.arange(len(idx)), args.bs, m_t, s_t, doy, device):
        LSb = lh_logS(lh, x_in, d)
        if st0 is None:
            st0 = selftest(LSb, device, thetas, torch.Generator(device=device).manual_seed(args.seed))
        for name, fr in frs.items():
            x = fr.invert(-LSb)
            for r, R in masks.items():
                cur[name][r].append(fr.region_logS(x, R).cpu().numpy())
        if (a0 // args.bs) % 100 == 0:
            print(f"[cop]   {a0:,}/{len(idx):,} ({time.time() - t0:.0f} s)", flush=True)
    seed = args.seed_name
    res = {"n": int(len(idx)), "bias": args.bias, "thetas": thetas, "selftest": st0, "models": {}, "mean_F": {},
           "nest": {}}
    store = {}
    for name in frs:
        key = f"{name}_cf"
        curves = {r: np.concatenate(v) for r, v in cur[name].items()}
        for r, ls in curves.items():
            e, t = tg[r]
            h = hazards_from_logS(ls)
            res["models"].setdefault(key, {}).setdefault(r, {})[seed] = compact(evaluate(h, e, t))
            store[f"{key}|{seed}|{r}"] = nll_per_sample(h, e, t).astype(np.float32)
            res["mean_F"].setdefault(key, {}).setdefault(r, {})[seed] = {
                str(tau): float(-np.expm1(ls[:, tau]).mean()) for tau in joint_eval.TAUS}
        res["nest"][key] = {seed: joint_eval.nest_violations(curves)}
        print(f"[cop] {key} {seed}  " + "  ".join(
            f"{r} NLL {res['models'][key][r][seed]['nll']:.4f}" for r in ("sq1", "sq5", "nw6", "ring9_5", "sq13",
                                                                          "scatter12") if r in curves)
              + f"  ({time.time() - t0:.0f} s)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, idx=idx, **{k_.replace("|", "__"): v for k_, v in store.items()})
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[cop] {js}\n[cop] {nz}", flush=True)


def run_scenarios(cfg, exp, args, device):
    out_dir, js, nz = joint_eval.outputs(exp, "scenarios", args.tag)
    d_ = exp / "metrics" / "paper_joint"
    cjs, cnz = d_ / f"counts{args.tag}.json", d_ / f"counts{args.tag}.npz"
    for p in (cjs, cnz):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    args.bias = f"calib:{args.calib_dir}"
    idx, X, doy, ev_px, t_px, m_t, s_t, rates, fw = joint_eval.setup(cfg, exp, args, device)
    lh = load_lh(exp, args, device, rates)
    thetas = load_thetas(exp, args)
    frs = {m: frailty_of(th, device) for m, th in thetas.items()}
    y_all = torch.from_numpy(np.where(ev_px, t_px, H + 1).reshape(len(idx), N))
    pi, pj = (torch.from_numpy(v).to(device) for v in pair_index())
    seed = args.seed_name
    gen = torch.Generator(device=device).manual_seed(args.seed)
    gen_c = torch.Generator(device=device).manual_seed(args.seed)
    acc, acc_c = {}, {}
    st0 = None
    t0 = time.time()
    for a0, ii, x_in, d in blocks_X(X, np.arange(len(idx)), args.bs, m_t, s_t, doy, device):
        y = y_all[a0:a0 + len(ii)].to(device)
        LSb = lh_logS(lh, x_in, d)
        if st0 is None:
            st0 = selftest(LSb, device, thetas, torch.Generator(device=device).manual_seed(args.seed))
        for name, fr in frs.items():
            x = fr.invert(-LSb)
            T = fr.sample_T(x, args.M, gen)
            for meth, TT in (("sim", T), ("shuf", shuffle(T, gen))):
                for tau, sc in scenario_scores(TT, y, pi, pj).items():
                    for sname, v in sc.items():
                        acc.setdefault(f"{name}_{meth}|{seed}|{sname}@{tau}", []).append(v)
            Tc = fr.sample_T(x, args.M_counts, gen_c)
            for meth, TT in (("sim", Tc), ("shuf", shuffle(Tc, gen_c))):
                for item, v in count_stats(TT, y, gen_c).items():
                    acc_c.setdefault(f"{name}_{meth}__{seed}__{item}", []).append(v)
        if (a0 // args.bs) % 100 == 0:
            print(f"[cop]   {a0:,}/{len(idx):,} ({time.time() - t0:.0f} s)", flush=True)
    res = {"n": int(len(idx)), "M": args.M, "M_counts": args.M_counts, "thetas": thetas, "selftest": st0, "mean": {}}
    store = {k_.replace("|", "__"): np.concatenate(v).astype(np.float32) for k_, v in acc.items()}
    for k_, v in store.items():
        meth, _, item = k_.split("__")
        res["mean"].setdefault(meth, {})[item] = {"all": float(v.mean()), "fire_in_window": float(v[fw].mean())}
    print(f"[cop] Szenarien {seed}: " + "  ".join(f"{m} crps@30 {res['mean'][m]['crps@30']['all']:.4f} es@30 "
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
    print(f"[cop] {js}\n[cop] {nz}\n[cop] {cjs}\n[cop] {cnz}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=["fit", "cond", "shapes", "scenarios"], required=True)
    ap.add_argument("--lh-ckpt", required=True)
    ap.add_argument("--seed-name", required=True, help="Name des Seeds in den Ausgaben (z. B. 0)")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--cop-dir", default="copula_calib")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--kmax", type=int, default=70)
    ap.add_argument("--dmax", type=int, default=4)
    ap.add_argument("--n", type=int, default=None, help="cond: Stichtage (0 = alle); shapes/scenarios: 20.000")
    ap.add_argument("--M", type=int, default=None, help="Ziehungen (fit/cond 256 Importance-Ziehungen, scenarios 32)")
    ap.add_argument("--M-counts", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lblock", type=int, default=128, help="fit/cond: Stichtage je Block")
    ap.add_argument("--fit-landmarks", type=int, default=3000)
    ap.add_argument("--val-shape-landmarks", type=int, default=5000)
    ap.add_argument("--line", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    ap.add_argument("--golden", type=int, default=8)
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
        args.M = 256 if args.part in ("fit", "cond") else 32
    if args.part != "fit" and not args.tag:
        raise SystemExit("ABBRUCH: --tag noetig")
    {"fit": run_fit, "cond": run_cond, "shapes": run_shapes, "scenarios": run_scenarios}[args.part](cfg, exp, args,
                                                                                                    device)


if __name__ == "__main__":
    main()

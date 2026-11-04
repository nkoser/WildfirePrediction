"""Survival-Koepfe zum Vergleich mit dem DCTM-Kopf, alle mit derselben Schnittstelle.

Fuer die Frage "was bringt der Bernstein-DCTM-Kopf?" bekommen die Alternativen
dasselbe Backbone, dieselben Daten, dieselbe Auswahl und dieselbe Auswertung.
Jeder Kopf bildet die rohen Netzausgaben (N, n_out) auf eine Survival-Funktion
ab und bietet

  log_survival(raw, tau)   log S(tau), float64; tau (T,) oder (N, T), ganze Tage
  hazards(raw, H)          diskrete Hazards fuer Tag 1..H (Auswertung)
  loss(raw, event, t)      Trainingsverlust; Ereignis an Tag t oder zensiert bei t

Koepfe (make_head):
  dctm          Bernstein-Transformationsmodell, logistischer Link (unser Kopf)
  dctm_cloglog  dasselbe mit cloglog-Link (Minimum-Extremwert)
  lh            Logistic Hazard: ein Hazard je Tag 1..H, diskrete Likelihood
                (Gensheimer & Narasimhan 2019; Kvamme & Borgan 2021). Bei
                tagesgenauen Ereignissen ist auch das intervallzensiert exakt -
                der Unterschied zum DCTM liegt in der Glattheit und Parameterzahl.
  deephit       Verteilung ueber Tag 1..H und "nach H" (Softmax), Verlust
                alpha * NLL + (1 - alpha) * Rangverlust (Lee et al. 2018,
                Gewichtung wie pycox: alpha = 0,2, sigma = 0,1)
  cox           DeepSurv: ein Risikowert eta, partielle Likelihood im Batch
                (Breslow-Bindungen), Basis-Hazard nach dem Training per
                Breslow-Schaetzer auf Trainingsdaten (fit_baseline)
  bce           Ablation ohne Survival-Formulierung: je Horizont tau_k ein
                eigener Sigmoid-Ausgang fuer P(T <= tau_k), BCE je Horizont
                (wie gestapelte Klassifikatoren, nur mit gemeinsamem Rumpf).
                Nichts erzwingt F(tau_1) <= F(tau_2); fuer Kurvenmetriken wird
                monotonisiert (BinaryHorizonHead.log_survival)

Die Intervall-NLL (interval_nll) ist fuer alle Koepfe dieselbe Groesse:
log(S(t-1) - S(t)) fuer ein Ereignis an Tag t, log S(t) fuer Zensierung bei t.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.models.bernstein import BernsteinSurvivalHead

HEADS = ("dctm", "dctm_cloglog", "lh", "deephit", "cox", "bce")
BCE_TAUS = (2, 7, 8, 16, 30, 31, 100)          # Paper-Protokoll dt + 1 = 2, 8, 16, 31; natuerliche Praevalenz 7, 30, 100


def _tau_index(tau: torch.Tensor, n: int, device, horizon: int) -> torch.Tensor:
    tau = tau.to(device=device, dtype=torch.float64)
    if tau.dim() == 1:
        tau = tau.unsqueeze(0).expand(n, -1)
    return tau.floor().clamp(0, horizon).long()


def interval_nll(head: nn.Module, raw: torch.Tensor, event: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Mittlere NLL; Ereignis an Tag t -> log(S(t-1) - S(t)), sonst log S(t)."""
    raw = raw.reshape(-1, raw.shape[-1])
    event, t = event.reshape(-1).bool(), t.reshape(-1).double()
    ls = head.log_survival(raw, torch.stack([t - 1.0, t], 1))
    ll_ev = ls[:, 0] + torch.log((-torch.expm1(ls[:, 1] - ls[:, 0])).clamp_min(1e-300))
    return -torch.where(event, ll_ev, ls[:, 1]).mean()


class _Head(nn.Module):
    horizon: int

    def hazards(self, raw: torch.Tensor, horizon: int) -> torch.Tensor:
        tau = torch.arange(0, horizon + 1, device=raw.device, dtype=torch.float64)
        ls = self.log_survival(raw, tau)
        return (-torch.expm1(ls[:, 1:] - ls[:, :-1])).to(raw.dtype)

    def loss(self, raw, event, t):
        return interval_nll(self, raw, event, t)


class LogisticHazardHead(_Head):
    def __init__(self, horizon: int = 100, init_cif: float = 0.02):
        super().__init__()
        self.horizon = self.n_out = int(horizon)
        h0 = -math.expm1(math.log1p(-init_cif) / horizon)       # Tages-Hazard mit F(H) = init_cif
        self.bias = math.log(h0 / (1 - h0))

    def log_survival(self, raw, tau):
        ls = F.logsigmoid(-(raw.double() + self.bias))         # log(1 - h_k), k = 1..H
        cs = torch.cat([ls.new_zeros(ls.shape[0], 1), ls.cumsum(1)], 1)
        return cs.gather(1, _tau_index(tau, raw.shape[0], raw.device, self.horizon))


class DeepHitHead(_Head):
    def __init__(self, horizon: int = 100, init_cif: float = 0.02, alpha: float = 0.2, sigma: float = 0.1,
                 max_rank: int = 1024):
        super().__init__()
        self.horizon, self.n_out = int(horizon), int(horizon) + 1
        self.alpha, self.sigma, self.max_rank = alpha, sigma, max_rank
        b = torch.zeros(self.n_out, dtype=torch.float64)
        b[-1] = math.log((1 - init_cif) / (init_cif / horizon))    # P(nach H) = 1 - init_cif
        self.register_buffer("bias", b, persistent=False)

    def log_survival(self, raw, tau):
        lp = F.log_softmax(raw.double() + self.bias, -1)        # Index j < H: Tag j + 1; Index H: nach H
        R = torch.logcumsumexp(lp.flip(-1), -1).flip(-1)       # R[k] = log P(T > k), k = 0..H
        return R.gather(1, _tau_index(tau, raw.shape[0], raw.device, self.horizon))

    def loss(self, raw, event, t):
        raw = raw.reshape(-1, raw.shape[-1])
        event, t = event.reshape(-1).bool(), t.reshape(-1).long()
        nll = interval_nll(self, raw, event, t)
        if self.max_rank and raw.shape[0] > self.max_rank:
            sel = torch.randperm(raw.shape[0], device=raw.device)[:self.max_rank]
            raw, event, t = raw[sel], event[sel], t[sel]
        S = self.log_survival(raw, torch.arange(self.horizon + 1, device=raw.device)).exp()   # (n, H+1)
        St = S[:, t.clamp(0, self.horizon)]                  # St[j, i] = S_j(T_i)
        own = St.diagonal()
        # Paar (i, j) vergleichbar: Ereignis bei i und T_i < T_j; F_i(T_i) soll ueber F_j(T_i) liegen
        mask = event[None, :] & (t[None, :] < t[:, None])
        rank = (torch.exp(-(St - own[None, :]) / self.sigma) * mask).mean()
        return self.alpha * nll + (1 - self.alpha) * rank.to(nll.dtype)


class CoxHead(_Head):
    def __init__(self, horizon: int = 100, init_cif: float = 0.02):
        super().__init__()
        self.horizon, self.n_out = int(horizon), 1
        base = -math.log1p(-init_cif) * torch.arange(horizon + 1, dtype=torch.float64) / horizon
        self.register_buffer("base", base)                     # Lambda_0(k), k = 0..H

    def log_survival(self, raw, tau):
        eta = raw.double()[:, :1]
        return -torch.exp(eta) * self.base[_tau_index(tau, raw.shape[0], raw.device, self.horizon)]

    def loss(self, raw, event, t):
        """Negative partielle Log-Likelihood, Breslow; Risikomenge an Tag k: alle mit t >= k."""
        eta = raw.reshape(-1).double()
        event, t = event.reshape(-1).bool(), t.reshape(-1).long().clamp(0, self.horizon + 1)
        m = eta.max().detach()
        s = eta.new_zeros(self.horizon + 2).index_add_(0, t, torch.exp(eta - m))
        log_r = torch.log(s.flip(0).cumsum(0).flip(0).clamp_min(1e-300)) + m
        ll = (eta - log_r[t])[event].sum() / event.sum().clamp_min(1)
        return (-ll).to(raw.dtype)

    @torch.no_grad()
    def fit_baseline(self, eta: torch.Tensor, event: torch.Tensor, t: torch.Tensor) -> None:
        """Breslow: Lambda_0(k) = sum_{j <= k} d_j / sum_{t_i >= j} exp(eta_i)."""
        eta, event, t = eta.reshape(-1).double(), event.reshape(-1).bool(), t.reshape(-1).long()
        m = eta.max()
        r = eta.new_zeros(self.horizon + 2).index_add_(0, t.clamp(0, self.horizon + 1), torch.exp(eta - m))
        r = r.flip(0).cumsum(0).flip(0)
        d = eta.new_zeros(self.horizon + 2).index_add_(0, t[event].clamp(0, self.horizon + 1),
                                                       torch.ones_like(eta[event]))
        inc = torch.where(d > 0, d / r.clamp_min(1e-300), torch.zeros_like(d)) * torch.exp(-m)
        inc[0] = 0.0
        self.base.copy_(inc[: self.horizon + 1].cumsum(0).to(self.base))


class BinaryHorizonHead(_Head):
    """Je Horizont tau_k ein Sigmoid-Ausgang fuer F(tau_k) = P(T <= tau_k), Verlust BCE.

    Label je Horizont: Ereignis bis tau_k; bekannt, wenn ein Ereignis beobachtet ist oder die
    Zensurzeit >= tau_k. Die Ausgaenge sind unabhaengig, F muss also nicht monoton in tau sein.

    log_survival fuer die Kurvenmetriken: kumulierter Hazard -log(1 - F) an den Stuetzstellen,
    mit monotone = True per laufendem Maximum monotonisiert, linear zwischen den Stuetzstellen
    (0 bei tau = 0), nach dem letzten tau_k konstant, plus 1e-9 je Tag (sonst haette ein Tag nach
    dem Monotonisieren Wahrscheinlichkeit 0 und die NLL waere unendlich). Mit monotone = False
    sind die Werte an den Stuetzstellen die rohen Ausgaenge (Paper-Protokoll), dazwischen nicht
    sinnvoll."""

    def __init__(self, horizon: int = 100, init_cif: float = 0.02, taus=BCE_TAUS):
        super().__init__()
        self.horizon = int(horizon)
        self.taus = tuple(sorted(int(t) for t in taus))
        if not self.taus or self.taus[0] < 1 or self.taus[-1] > self.horizon:
            raise ValueError(f"taus {taus!r} nicht in 1..{horizon}")
        self.n_out = len(self.taus)
        self.monotone = True
        tt = torch.tensor(self.taus, dtype=torch.float64)
        f0 = -torch.expm1(math.log1p(-init_cif) * tt / horizon)            # F(tau) bei konstantem Hazard
        self.register_buffer("bias", torch.log(f0 / (1 - f0)), persistent=False)
        self.register_buffer("tau_k", tt, persistent=False)
        knots = [0.0] + list(self.taus)
        W = torch.zeros(self.horizon + 1, len(knots), dtype=torch.float64)   # Lambda(Tag j) = W[j] @ Lambda(Stuetzstellen)
        for j in range(self.horizon + 1):
            if j >= knots[-1]:
                W[j, -1] = 1.0
                continue
            s = max(i for i in range(len(knots)) if knots[i] <= j)
            w = (j - knots[s]) / (knots[s + 1] - knots[s])
            W[j, s], W[j, s + 1] = 1.0 - w, w
        self.register_buffer("interp", W, persistent=False)

    def cif_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """Rohe F(tau_k) (N, K), float64, nicht monotonisiert."""
        return torch.sigmoid(raw.double() + self.bias)

    def violations(self, raw: torch.Tensor) -> torch.Tensor:
        """(N,) bool: irgendein F(tau_{k+1}) < F(tau_k)."""
        F_ = self.cif_raw(raw)
        return (F_[:, 1:] < F_[:, :-1]).any(1)

    def log_survival(self, raw, tau):
        lam = -torch.log1p(-self.cif_raw(raw).clamp(max=1 - 1e-12))
        if self.monotone:
            lam = torch.cummax(lam, 1).values
        full = torch.cat([lam.new_zeros(lam.shape[0], 1), lam], 1) @ self.interp.t()     # (N, H+1)
        if self.monotone:                                   # streng steigend: kein Tag mit Ereigniswahrscheinlichkeit 0
            full = full + 1e-9 * torch.arange(self.horizon + 1, device=full.device, dtype=full.dtype)
        return -full.gather(1, _tau_index(tau, raw.shape[0], raw.device, self.horizon))

    def loss(self, raw, event, t):
        raw = raw.reshape(-1, raw.shape[-1])
        event, t = event.reshape(-1).bool()[:, None], t.reshape(-1).double()[:, None]
        y = event & (t <= self.tau_k)
        known = event | (t >= self.tau_k)
        bce = F.binary_cross_entropy_with_logits(raw.double() + self.bias, y.double(), reduction="none")
        return ((bce * known).sum() / known.sum().clamp_min(1)).to(raw.dtype)


def make_head(kind: str, tau_max: float, degree: int, init_cif: float, taus=None) -> nn.Module:
    if kind in ("dctm", "dctm_cloglog"):
        return BernsteinSurvivalHead(degree, tau_max, init_offset=-8.0, init_cif=init_cif,
                                     link="cloglog" if kind == "dctm_cloglog" else "logistic")
    if kind == "bce":
        return BinaryHorizonHead(int(tau_max), init_cif, taus or BCE_TAUS)
    cls = {"lh": LogisticHazardHead, "deephit": DeepHitHead, "cox": CoxHead}[kind]
    return cls(int(tau_max), init_cif)


def head_loss(head: nn.Module, raw: torch.Tensor, event: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Trainingsverlust eines Kopfs; der Bernstein-Kopf nimmt die Intervall-NLL."""
    raw = raw.reshape(-1, raw.shape[-1])
    if isinstance(head, BernsteinSurvivalHead):
        return interval_nll(head, raw, event, t)
    return head.loss(raw, event, t)

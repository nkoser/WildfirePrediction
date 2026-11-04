"""Self-Forcing fuer den Contagion-Kern: Verlust unter der eigenen 0/1-Simulation statt unter der beobachteten Front.

Teacher Forcing (contagion_ring.tf_nll_ring) schaetzt den Kern aus der beobachteten Front; in der Regionssimulation
breitet sich das Feuer aber aus den SELBST gezogenen Zuendungen aus, Fehler summieren sich ueber die Tage
(cg_tfonly: Flaeche stimmt, Form nicht, Praezision 30 d 0.14-0.36). Hier:
  1. ohne Gradient M Pfade ueber R Tage ziehen wie contagion_mc_map2.simulate (0/1-Zuendungen, Brennkurve D);
     innen gezogen, Ring = beobachtete Zuendtage (Rand aus den Daten, wie im Teacher Forcing);
  2. mit Gradient je Pfad m die bedingte Survival der inneren Pixel aus den gezogenen Nachbarzustaenden;
  3. Randverteilung = Mittel ueber die Pfade, NLL der beobachteten Zuendung (Tag tt <= R) bzw. Zensur bei R:
        P_i(Zuendung an t) ~ 1/M sum_m S_i^m(t-1) h_i^m(t),   S_i(R) ~ 1/M sum_m S_i^m(R).
Der Gradient laeuft nicht durch das Ziehen (wie Scheduled Sampling), die Vorwaertsrechnung hat aber die
Korrelationen der echten Simulation (keine Mean-Field-Verduennung). contagion_ring.py bleibt unveraendert.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from wildfire.paper.contagion_ring import RingContagion, tf_nll_ring
from wildfire.paper.sts_dctm import P


@torch.no_grad()
def sample_paths(net: RingContagion, log_exo, k, lD, ev, tt, M: int, R: int) -> torch.Tensor:
    """-> Zuendtage (M, B, PE*PE) long: innen gezogen (1..R, sonst R + 1), Ring beobachtet (tt bzw. nie)."""
    B, Hh, N = log_exo.shape
    KK, A, c = k.shape[1], lD.shape[-1], k.shape[1] // 2
    never = Hh + 10
    fired = torch.where(ev, tt.long(), torch.full_like(tt.long(), never))[None].repeat(M, 1, 1)
    fired[:, :, net.inner] = never
    lDm = lD[None].expand(M, -1, -1, -1)
    for s in range(R):
        age = s - fired
        b = torch.where(age >= 0, lDm.gather(-1, age.clamp(0, A - 1)[..., None])[..., 0].exp(), torch.zeros((), device=k.device))
        U = net.unfold_ext(b.reshape(M * B, -1)).view(M, B, KK, N)
        lp = torch.log1p(-(k[None] * U).clamp(max=1 - 1e-6))
        l1mh = log_exo[None, :, s] + lp.sum(2) - lp[:, :, c]
        fi = fired[:, :, net.inner]
        new = (fi == never) & (torch.rand(l1mh.shape, device=l1mh.device) < -torch.expm1(l1mh))
        fired[:, :, net.inner] = torch.where(new, torch.full_like(fi, s + 1), fi)
    return fired


def sf_nll_ring(net: RingContagion, log_exo, k, logD, ev, tt, M: int = 8, R: int = 30) -> torch.Tensor:
    """Mittlere Self-Forcing-NLL der inneren Pixel ueber die ersten R Tage. log_exo (B, H, 169), k (B, KK, 169),
    logD (B, 169, A), ev/tt (B, PE*PE)."""
    B, Hh, N = log_exo.shape
    R = min(R, Hh)
    KK, c = k.shape[1], k.shape[1] // 2
    lD = net.ext_logD(logD)
    A = lD.shape[-1]
    fired = sample_paths(net, log_exo.detach(), k.detach(), lD.detach(), ev, tt, M, R)       # (M, B, PE2)
    s = torch.arange(R, device=k.device)
    age = s[None, None, None] - fired[..., None]                                                # (M, B, PE2, R)
    b = torch.where(age >= 0, lD[None].expand(M, -1, -1, -1).gather(-1, age.clamp(0, A - 1)).exp(),
                    torch.zeros((), device=k.device))
    U = net.unfold_ext(b.permute(0, 1, 3, 2).reshape(M * B * R, -1)).view(M, B, R, KK, N)
    lp = torch.log1p(-(k[None, :, None] * U).clamp(max=1 - 1e-6))
    l1mh = (log_exo[None, :, :R] + lp.sum(3) - lp[:, :, :, c]).double()                         # (M, B, R, N)
    logS = F.pad(l1mh.cumsum(2), (0, 0, 1, 0))                                                  # (M, B, R+1, N)
    ev_i, tt_i = ev[:, net.inner], tt[:, net.inner].long()
    e = ev_i & (tt_i <= R)
    t = tt_i.clamp(1, R)[None, :, None].expand(M, -1, -1, -1)                                  # (M, B, 1, N)
    ls_prev = logS.gather(2, t - 1).squeeze(2)
    l_t = l1mh.gather(2, t - 1).squeeze(2)
    ll_ev = ls_prev + torch.log((-torch.expm1(l_t)).clamp_min(1e-300))
    ll_c = logS[:, :, R]
    ll = torch.where(e[None], ll_ev, ll_c)
    return -(torch.logsumexp(ll, 0) - math.log(M)).mean()


def forward_ring_sf(net: RingContagion, x, doy, hist, ev=None, tt=None, sf_paths: int = 8, sf_days: int = 30,
                    tf: bool = True, sf: bool = True):
    """Wie contagion_ring.forward_ring, dazu die Self-Forcing-NLL.
    -> (ls_px (B, 13, 13, H+1), ls_pa (B, H+1), tf_nll oder None, sf_nll oder None)."""
    pix = net.encode_input(x, doy)
    obs, a, k, logD = net.params(x, doy, pix)
    net.set_ring(hist)
    try:
        with torch.autocast(device_type=x.device.type, enabled=False):
            log_exo = net.exogenous(obs, a, k, logD)
            ls_px = net.rollout(log_exo, k, logD)
            ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0))
            tfl = sfl = None
            if ev is not None:
                evb, ttf = ev.reshape(x.shape[0], -1).bool(), tt.reshape(x.shape[0], -1)
                if tf:
                    tfl = tf_nll_ring(net, log_exo, k, logD, evb, ttf)
                if sf:
                    sfl = sf_nll_ring(net, log_exo, k.float(), logD.float(), evb, ttf, sf_paths, sf_days)
    finally:
        net.set_ring(None)
    return ls_px.view(x.shape[0], P, P, -1), ls_pa, tfl, sfl

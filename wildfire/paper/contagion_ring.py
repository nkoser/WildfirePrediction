"""Contagion mit Feuerring: der Kern sieht auch Feuer bis R = K // 2 Pixel ausserhalb des 13 x 13-Ausschnitts.

contagion_spread.SpreadContagion (unveraendert) rechnet die Nachbarn mit Null-Rand: am Ausschnittsrand
fehlen sie, Feuer von aussen muss die eigene Zuendung a erklaeren. RingContagion erbt davon und nimmt, wenn
set_ring gesetzt ist, einen Rand der Breite R dazu (23 x 23 bei K = 11), nur fuer den Feuerkanal:
  b_obs  aus ring_hist (Feuer der Eingabetage, ring_targets.py) statt null; innen weiter aus x
  D      der Ringpixel: Mittel von log D ueber den Ausschnitt (ihre Token kennt das Netz nicht)
  b_pred bleibt innen (die Zukunft des Rings sagt das Modell nicht voraus)
Teacher Forcing (tf_nll_ring) nimmt dazu die beobachteten Zuendtage des Rings. Ohne set_ring identisch zu
SpreadContagion; Checkpoints haben dieselben Parameter (sts_spread.load_spread laedt sie).
Nur mit tageskonstantem Kern (ohne Wetter-Oracle).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wildfire.paper.contagion_spread import SpreadContagion
from wildfire.paper.sts_dctm import P


class RingContagion(SpreadContagion):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.R = self.K // 2
        self.PE = P + 2 * self.R
        m = torch.zeros(self.PE, self.PE, dtype=torch.bool)
        m[self.R:self.R + P, self.R:self.R + P] = True
        self.register_buffer("inner", m.flatten(), persistent=False)
        self._ring = None

    def set_ring(self, hist=None) -> None:
        """hist (B, T, PE, PE) 0/1, Feuer der Eingabetage (letzter = Stichtag); gilt fuer genau den naechsten Aufruf."""
        self._ring = hist

    def ext_logD(self, logD):
        """(B, 169, A) -> (B, PE*PE, A): innen log D, Ring Mittel ueber den Ausschnitt."""
        B, _, A = logD.shape
        out = logD.mean(1, keepdim=True).expand(B, self.PE * self.PE, A).clone()
        out[:, self.inner] = logD
        return out

    def unfold_ext(self, b):
        """(M, PE*PE) -> (M, K*K, 169): Nachbarn je innerem Empfaengerpixel, ohne Null-Rand."""
        return F.unfold(b.view(-1, 1, self.PE, self.PE), self.K)

    def exogenous(self, obs, a, k, logD):
        if self._ring is None:
            return super().exogenous(obs, a, k, logD)
        if k.dim() != 3:
            raise ValueError("Ring nur ohne Wetter-Oracle")
        B, D, N = obs.shape
        KK = self.K * self.K
        ring = self._ring[:, -D:].to(obs.dtype).flatten(2).clone()                    # (B, D, PE*PE)
        ring[:, :, self.inner] = obs                                                    # innen wie bisher aus x
        lD = self.ext_logD(logD)
        idx = (torch.arange(self.H, device=obs.device)[:, None]
               + torch.arange(D - 1, -1, -1, device=obs.device)[None, :])
        pw = lD[:, :, idx].exp().permute(0, 2, 3, 1)                                    # (B, H, D, PE*PE)
        b_obs = 1 - torch.prod(1 - ring[:, None] * pw, dim=2)                          # (B, H, PE*PE)
        U = self.unfold_ext(b_obs.reshape(B * self.H, -1)).view(B, self.H, KK, N)
        lo = torch.log1p(-(k[:, None] * U).clamp(max=1 - 1e-6))
        return torch.log1p(-a.clamp(max=1 - 1e-6)) + lo.sum(2)

    def forward(self, x, doy, rate: bool = False):
        out = super().forward(x, doy, rate)
        self._ring = None
        return out


def tf_nll_ring(net: RingContagion, log_exo, k, logD, ev, tt) -> torch.Tensor:
    """Wie contagion_tf.tf_nll, Brennzustand aus den beobachteten Zuendtagen auf PE x PE (Ring eingeschlossen);
    Verlust nur fuer die inneren Pixel. ev, tt (B, PE*PE)."""
    B, Hh, N = log_exo.shape
    KK = k.shape[1]
    lD = net.ext_logD(logD)
    s = torch.arange(Hh, device=log_exo.device)
    age = s[None, None] - tt[:, :, None]
    on = ev[:, :, None] & (age >= 0)
    b = torch.where(on, lD.gather(-1, age.clamp(0, lD.shape[-1] - 1)).exp(), torch.zeros((), device=k.device))
    U = net.unfold_ext(b.transpose(1, 2).reshape(B * Hh, -1)).view(B, Hh, KK, N)
    lp = torch.log1p(-(k[:, None] * U).clamp(max=1 - 1e-6))
    l1mh = (log_exo + lp.sum(2) - lp[:, :, KK // 2]).double()
    logS = F.pad(l1mh.cumsum(1), (0, 0, 1, 0))
    ev_i, tt_i = ev[:, net.inner], tt[:, net.inner].long().clamp(1, Hh)
    ls_t = logS.gather(1, tt_i[:, None]).squeeze(1)
    ls_prev = logS.gather(1, (tt_i - 1)[:, None]).squeeze(1)
    l_t = l1mh.gather(1, (tt_i - 1)[:, None]).squeeze(1)
    ll_ev = ls_prev + torch.log((-torch.expm1(l_t)).clamp_min(1e-300))
    return -torch.where(ev_i, ll_ev, ls_t).mean()


def forward_ring(net: RingContagion, x, doy, hist, ev=None, tt=None):
    """Ein Durchlauf mit Ring; -> (ls_px (B, 13, 13, H+1), ls_pa (B, H+1), tf_nll oder None).
    hist (B, T, PE, PE), ev/tt (B, PE, PE) fuer Teacher Forcing."""
    pix = net.encode_input(x, doy)
    obs, a, k, logD = net.params(x, doy, pix)
    net.set_ring(hist)
    try:
        with torch.autocast(device_type=x.device.type, enabled=False):
            log_exo = net.exogenous(obs, a, k, logD)
            ls_px = net.rollout(log_exo, k, logD)
            ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0))
            tf = None
            if ev is not None:
                tf = tf_nll_ring(net, log_exo, k, logD, ev.reshape(x.shape[0], -1).bool(), tt.reshape(x.shape[0], -1))
    finally:
        net.set_ring(None)
    return ls_px.view(x.shape[0], P, P, -1), ls_pa, tf

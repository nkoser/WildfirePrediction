"""Self-Forcing + Karten-Verlust: weiches IoU der simulierten 13 x 13-Karte gegen die beobachtete Karte.

contagion_selfforce.sf_nll_ring bewertet jedes Pixel einzeln (NLL der Zuendung unter der Randverteilung der
eigenen 0/1-Pfade). Fuer Feuerkarten zaehlt aber, ob die ganze Karte stimmt. Hier aus denselben gezogenen Pfaden
die Randverteilung P_i(t) = 1 - 1/M sum_m S_i^m(t) je innerem Pixel und Horizont t (--iou-days) und dazu
    weiches IoU_t = (sum_i P_i Y_i + 1) / (sum_i P_i + sum_i Y_i - sum_i P_i Y_i + 1),   Y_i = Zuendung bis t,
Verlust 1 - IoU gemittelt ueber Ausschnitte und Horizonte. Die + 1 haelt Ausschnitte ohne Feuer definiert und
bestraft dort vorhergesagte Flaeche. Gradient wie bei Self-Forcing nur durch die bedingten Hazards (nicht durch
das Ziehen). contagion_selfforce.py bleibt unveraendert.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from wildfire.paper.contagion_ring import RingContagion, tf_nll_ring
from wildfire.paper.contagion_selfforce import sample_paths
from wildfire.paper.sts_dctm import P


def sf_losses(net: RingContagion, log_exo, k, logD, ev, tt, M: int = 8, R: int = 30, iou_days=(1, 7, 15, 30)):
    """-> (Self-Forcing-NLL, 1 - weiches IoU) der inneren Pixel ueber die ersten R Tage."""
    B, Hh, N = log_exo.shape
    R = min(R, Hh)
    KK, c = k.shape[1], k.shape[1] // 2
    lD = net.ext_logD(logD)
    A = lD.shape[-1]
    fired = sample_paths(net, log_exo.detach(), k.detach(), lD.detach(), ev, tt, M, R)
    s = torch.arange(R, device=k.device)
    age = s[None, None, None] - fired[..., None]
    b = torch.where(age >= 0, lD[None].expand(M, -1, -1, -1).gather(-1, age.clamp(0, A - 1)).exp(),
                    torch.zeros((), device=k.device))
    U = net.unfold_ext(b.permute(0, 1, 3, 2).reshape(M * B * R, -1)).view(M, B, R, KK, N)
    lp = torch.log1p(-(k[None, :, None] * U).clamp(max=1 - 1e-6))
    l1mh = (log_exo[None, :, :R] + lp.sum(3) - lp[:, :, :, c]).double()
    logS = F.pad(l1mh.cumsum(2), (0, 0, 1, 0))                                                  # (M, B, R+1, N)
    ev_i, tt_i = ev[:, net.inner], tt[:, net.inner].long()
    e = ev_i & (tt_i <= R)
    t = tt_i.clamp(1, R)[None, :, None].expand(M, -1, -1, -1)
    ll_ev = logS.gather(2, t - 1).squeeze(2) + torch.log((-torch.expm1(l1mh.gather(2, t - 1).squeeze(2))).clamp_min(1e-300))
    ll = torch.where(e[None], ll_ev, logS[:, :, R])
    nll = -(torch.logsumexp(ll, 0) - math.log(M)).mean()
    loss_iou = []
    for d in iou_days:
        d = min(d, R)
        Pm = 1 - logS[:, :, d].exp().mean(0)                                                    # (B, N)
        Y = (ev_i & (tt_i <= d)).double()
        inter = (Pm * Y).sum(1)
        iou = (inter + 1) / (Pm.sum(1) + Y.sum(1) - inter + 1)
        loss_iou.append(1 - iou)
    return nll, torch.stack(loss_iou).mean()


def forward_ring_sfiou(net: RingContagion, x, doy, hist, ev, tt, sf_paths: int = 8, sf_days: int = 30,
                       iou_days=(1, 7, 15, 30), tf: bool = True):
    """Wie contagion_selfforce.forward_ring_sf; -> (ls_px, ls_pa, tf_nll oder None, sf_nll, 1 - IoU)."""
    pix = net.encode_input(x, doy)
    obs, a, k, logD = net.params(x, doy, pix)
    net.set_ring(hist)
    try:
        with torch.autocast(device_type=x.device.type, enabled=False):
            log_exo = net.exogenous(obs, a, k, logD)
            ls_px = net.rollout(log_exo, k, logD)
            ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0))
            evb, ttf = ev.reshape(x.shape[0], -1).bool(), tt.reshape(x.shape[0], -1)
            tfl = tf_nll_ring(net, log_exo, k, logD, evb, ttf) if tf else None
            sfl, iol = sf_losses(net, log_exo, k.float(), logD.float(), evb, ttf, sf_paths, sf_days, iou_days)
    finally:
        net.set_ring(None)
    return ls_px.view(x.shape[0], P, P, -1), ls_pa, tfl, sfl, iol

"""Teacher Forcing fuer den Contagion-Kern: Ausbreitung aus der BEOBACHTETEN Front statt aus dem eigenen Mean-Field.

Im normalen Training (rollout) haengt die Zuendung von Pixel i am Tag s+1 vom vorhergesagten Brennzustand
b_pred der Nachbarn ab. Der Kern bekommt so fuer weite Spruenge kaum Gradienten (der Nebel verduennt sich
ueber mehrere Tage). Hier wird zusaetzlich die Survival-NLL mit dem beobachteten Brennzustand gerechnet:

    b_tf_j(s) = D_j(s - T_j)  wenn Pixel j am Tag T_j <= s zum ersten Mal brennt (Ziel des Stichtags), sonst 0
    log(1 - h_i(s+1)) = log_exo_i(s) + sum_{j != i} log(1 - k_ij b_tf_j(s))

wie b_pred (eine Zuendung je Pixel, Brenndauer D), nur mit der wahren statt der vorhergesagten Zuendung.
Nur Training; in der Vorhersage sind die Folgetage unbekannt (dort weiter rollout bzw. Monte Carlo).
contagion.py, contagion_spread.py und sts_spread.py bleiben unveraendert.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wildfire.paper.sts_dctm import P


def tf_nll(net, log_exo, k, logD, ev, tt) -> torch.Tensor:
    """Mittlere Teacher-Forcing-NLL je Pixel. log_exo (B, H, N), k (B, KK, N), logD (B, N, A),
    ev (B, N) bool, tt (B, N) Tag des ersten Feuers bzw. Zensur (1..H)."""
    if k.dim() != 3:
        raise ValueError("Teacher Forcing nur mit tageskonstantem Kern (ohne Wetter-Oracle)")
    B, Hh, N = log_exo.shape
    KK = k.shape[1]
    s = torch.arange(Hh, device=log_exo.device)
    age = s[None, None] - tt[:, :, None]                                  # (B, N, H): Alter der Zuendung am Tag s
    on = ev[:, :, None] & (age >= 0)
    b = torch.where(on, logD.gather(-1, age.clamp(0, logD.shape[-1] - 1)).exp(), torch.zeros((), device=k.device))
    U = net._unfold(b.transpose(1, 2).reshape(B * Hh, N)).view(B, Hh, KK, N)
    lp = torch.log1p(-(k[:, None] * U).clamp(max=1 - 1e-6))               # (B, H, KK, N)
    l1mh = (log_exo + lp.sum(2) - lp[:, :, KK // 2]).double()            # Mitte nur beobachtet, wie rollout
    logS = F.pad(l1mh.cumsum(1), (0, 0, 1, 0))                            # (B, H+1, N)
    t = tt.long().clamp(1, Hh)
    ls_t = logS.gather(1, t[:, None]).squeeze(1)
    ls_prev = logS.gather(1, (t - 1)[:, None]).squeeze(1)
    l_t = l1mh.gather(1, (t - 1)[:, None]).squeeze(1)
    ll_ev = ls_prev + torch.log((-torch.expm1(l_t)).clamp_min(1e-300))
    return -torch.where(ev, ll_ev, ls_t).mean()


def forward_tf(net, x, doy, ev, tt):
    """Wie ContagionSTS.forward, dazu die Teacher-Forcing-NLL aus demselben Durchlauf.
    -> (ls_px (B, 13, 13, H+1), ls_pa (B, H+1), tf_nll)."""
    pix = net.encode_input(x, doy)
    obs, a, k, logD = net.params(x, doy, pix)
    with torch.autocast(device_type=x.device.type, enabled=False):
        log_exo = net.exogenous(obs, a, k, logD)
        ls_px = net.rollout(log_exo, k, logD)
        ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0))
        tf = tf_nll(net, log_exo, k, logD, ev.reshape(x.shape[0], -1).bool(), tt.reshape(x.shape[0], -1))
    return ls_px.view(x.shape[0], P, P, -1), ls_pa, tf

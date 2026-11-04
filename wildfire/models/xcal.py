"""X-CAL (Goldstein et al., NeurIPS 2020): differenzierbare D-Calibration als Zusatzterm im Training.

D-Calibration fordert, dass S(T | x) ueber die Stichproben gleichverteilt ist. Hier je Minibatch:

  Ereignis an Tag t   Punkt u = (S(t-1) + S(t)) / 2, weich auf die Faecher verteilt
                      (Sigmoid-Kanten der Breite gamma, auf Summe 1 normiert)
  zensiert bei c      gleichverteilt auf [0, S(c)] wie in eval.metrics.d_calibration; der Anteil
                      je Fach ist stueckweise linear in S(c) und damit differenzierbar

Strafterm = n_bins * sum_b (Anteil_b - 1 / n_bins)^2, die Chi-Quadrat-Form aus X-CAL. Bei kleinen
Minibatches ist der Schaetzer verzerrt (die quadrierten Anteile enthalten die Stichprobenvarianz);
der Term zieht trotzdem in die richtige Richtung, die Staerke steuert lambda.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def xcal_penalty(head: nn.Module, raw: torch.Tensor, event: torch.Tensor, t: torch.Tensor,
                 n_bins: int = 10, gamma: float = 0.02) -> torch.Tensor:
    raw = raw.reshape(-1, raw.shape[-1])
    event, t = event.reshape(-1).bool(), t.reshape(-1).double()
    ls = head.log_survival(raw, torch.stack([t - 1.0, t], 1))          # log S(t-1), log S(t)
    s_prev, s_t = ls[:, 0].exp(), ls[:, 1].exp()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=raw.device, dtype=s_t.dtype)
    lo, hi = edges[:-1], edges[1:]
    u = 0.5 * (s_prev + s_t)
    m_ev = torch.sigmoid((u[:, None] - lo) / gamma) * torch.sigmoid((hi - u[:, None]) / gamma)
    m_ev = m_ev / m_ev.sum(1, keepdim=True).clamp_min(1e-12)
    sc = s_t.clamp_min(1e-6)
    m_ce = (torch.minimum(sc[:, None], hi) - lo).clamp_min(0.0) / sc[:, None]
    share = torch.where(event[:, None], m_ev, m_ce).mean(0)
    return n_bins * ((share - 1.0 / n_bins) ** 2).sum()

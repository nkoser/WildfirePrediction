"""Verlustfunktion fuer diskrete Survival-Modelle mit Rechtszensierung.

Das Modell gibt je Pixel H Logits aus, aus denen ueber die Sigmoidfunktion
die diskreten Hazards werden. Die Likelihood eines Beispiels ist

    Ereignis an Tag k:  h_k * prod_{j<k} (1 - h_j)
    zensiert bei H:           prod_{j<=H} (1 - h_j)

Gerechnet wird durchgehend im Log-Raum ueber ``logsigmoid``, nie ueber
``log(sigmoid(x))``. Der Unterschied ist nicht kosmetisch: Bei einer
Praevalenz um 1e-3 werden die meisten Hazards sehr klein, und
``log(sigmoid(-20))`` verliert in float32 jede Genauigkeit, waehrend
``logsigmoid(-20)`` exakt bleibt.

Der Loss ist damit dieselbe Groesse wie die NLL in der Auswertung - Training
und Evaluation optimieren nachweislich dasselbe. Ein Test prueft genau das.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def discrete_survival_nll(logits: torch.Tensor, event: torch.Tensor,
                          t_event: torch.Tensor,
                          weight: torch.Tensor | None = None,
                          reduction: str = "mean") -> torch.Tensor:
    """(N, H) Logits -> skalarer Verlust.

    ``t_event`` ist 1-basiert: 1 bedeutet Ereignis am ersten Horizonttag.
    """
    N, H = logits.shape
    log_h = F.logsigmoid(logits)              # log h_k
    log_1mh = F.logsigmoid(-logits)           # log(1 - h_k), stabil
    csum = torch.cumsum(log_1mh, dim=1)

    k = t_event.long().clamp(1, H) - 1
    idx = torch.arange(N, device=logits.device)
    prior = torch.where(k > 0, csum[idx, (k - 1).clamp(min=0)],
                        torch.zeros_like(csum[:, 0]))
    ll_event = log_h[idx, k] + prior
    ll_cens = csum[:, H - 1]
    ll = torch.where(event.bool(), ll_event, ll_cens)

    if weight is not None:
        loss = -(weight * ll).sum() / weight.sum().clamp(min=1e-12)
    else:
        loss = -ll.mean()
    if reduction == "none":
        return -ll
    return loss


def hazards_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def cif_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Kumulatives Risiko, monoton per Konstruktion."""
    return 1.0 - torch.cumprod(torch.sigmoid(-logits), dim=1)

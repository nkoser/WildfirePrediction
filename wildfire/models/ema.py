"""EMA der Gewichte, gemeinsam fuer unsere Survival-Modelle und den Nachbau des Paper-Modells."""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """Exponentieller gleitender Mittelwert der Gewichte (Polyak-Mittel).

    Nach jedem Optimierer-Schritt: w_ema <- d * w_ema + (1 - d) * w, mit
    d = min(decay, (1 + n) / (10 + n)), damit die ersten Schritte nicht am
    Startwert kleben. Auswahl, Checkpoint und Auswertung laufen dann mit w_ema.
    Ganzzahlige Puffer werden kopiert, nicht gemittelt."""

    def __init__(self, net: nn.Module, decay: float):
        self.decay, self.n = decay, 0
        self.shadow = {k: v.detach().clone().float() if v.is_floating_point() else v.detach().clone()
                       for k, v in net.state_dict().items()}

    @torch.no_grad()
    def update(self, net: nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in net.state_dict().items():
            s = self.shadow[k]
            if v.is_floating_point():
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def state(self, like: dict) -> dict:
        """Gemittelte Gewichte in den dtypes von ``like`` (state_dict des Netzes)."""
        return {k: s.to(like[k].dtype).clone() for k, s in self.shadow.items()}

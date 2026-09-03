"""Stetiger Survival-Kopf: monotone Transformationsfunktion in Bernstein-Basis.

Bisher gab jedes Modell eine TABELLE aus - H Hazards fuer H feste Tage.
Damit ist das Risiko nur an den gelernten Stuetzstellen definiert; nach
37,5 Tagen zu fragen ginge nur ueber Interpolation, und ein anderer
Horizont braeuchte ein neues Modell.

Hier wird stattdessen eine FUNKTION gelernt:

    F(tau | x) = sigmoid( h(tau | x) )

mit einer in tau monoton wachsenden Transformationsfunktion h. Sie wird in
Bernstein-Basis vom Grad M entwickelt:

    h(tau | x) = sum_k theta_k(x) * B_{k,M}(tau / tau_max)

**Warum Bernstein.** Ein Bernstein-Polynom ist genau dann monoton wachsend,
wenn seine Koeffizienten es sind: theta_0 <= theta_1 <= ... <= theta_M.
Das laesst sich durch die Parametrisierung erzwingen statt durch eine
Strafe im Verlust - die Monotonie gilt dann exakt und fuer jede Eingabe,
nicht nur ungefaehr und nur dort, wo trainiert wurde. Das Netz gibt
M+1 rohe Werte aus; daraus werden die theta durch kumulative Addition von
softplus gebildet.

**Wie gross ist M?** Der Grad steuert, wie viele Kruemmungen die Kurve
haben kann. Fuer den kanadischen Jahresgang - Anstieg im Sommer, Null im
Winter, Wiederanstieg - braucht es mindestens zwei Wendepunkte; M = 12 gibt
reichlich Reserve bei nur 13 Ausgabewerten je Pixel. Zum Vergleich: ein
taegliches Gitter ueber zwei Jahre haette 730.

**Likelihood.** Die Ereigniszeiten sind tagesaufgeloest. Ein Ereignis "an
Tag k" heisst also nicht T = k, sondern T in (k-1, k] - das ist
Intervallzensierung, und die exakte Likelihood ist F(k) - F(k-1). Wir
benutzen sie direkt, statt ueber eine Dichte und deren Ableitung zu gehen.
Das ist genauer UND numerisch harmloser.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7


def bernstein_basis(u: torch.Tensor, degree: int) -> torch.Tensor:
    """B_{k,M}(u) fuer k = 0..M. u beliebige Form -> (..., M+1)."""
    u = u.clamp(0.0, 1.0).unsqueeze(-1)
    k = torch.arange(degree + 1, device=u.device, dtype=u.dtype)
    log_coef = (math.lgamma(degree + 1)
                - torch.lgamma(k + 1) - torch.lgamma(degree - k + 1))
    log_b = (log_coef + k * torch.log(u.clamp_min(EPS))
             + (degree - k) * torch.log((1 - u).clamp_min(EPS)))
    return torch.exp(log_b)


class BernsteinSurvivalHead(nn.Module):
    """Wandelt Netzausgaben in eine monotone Survival-Funktion.

    ``tau_max`` legt fest, ueber welchen Zeitraum die Basis gespannt ist.
    Abfragen darueber hinaus werden geklemmt - die Funktion ist auf
    [0, tau_max] definiert, nicht darueber hinaus extrapolierbar.
    """

    def __init__(self, degree: int = 12, tau_max: float = 365.0,
                 init_offset: float = -7.0) -> None:
        super().__init__()
        self.degree = degree
        self.tau_max = float(tau_max)
        self.n_out = degree + 1
        self.init_offset = init_offset

    def thetas(self, raw: torch.Tensor) -> torch.Tensor:
        """(..., M+1) rohe Werte -> monoton wachsende Koeffizienten."""
        first = raw[..., :1] + self.init_offset
        steps = F.softplus(raw[..., 1:])
        return torch.cat([first, first + torch.cumsum(steps, dim=-1)], dim=-1)

    def cif(self, raw: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Kumulatives Risiko F(tau). ``tau`` broadcastet gegen ``raw``.

        raw:  (N, M+1)      tau: (N, T) oder (T,)   ->   (N, T)
        """
        theta = self.thetas(raw)                       # (N, M+1)
        if tau.dim() == 1:
            tau = tau.unsqueeze(0).expand(raw.shape[0], -1)
        u = (tau / self.tau_max).clamp(0.0, 1.0)
        B = bernstein_basis(u, self.degree)            # (N, T, M+1)
        h = (B * theta.unsqueeze(1)).sum(-1)           # (N, T)
        return torch.sigmoid(h)

    def survival(self, raw: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.cif(raw, tau)

    def log_likelihood(self, raw: torch.Tensor, event: torch.Tensor,
                       t_event: torch.Tensor, horizon: float) -> torch.Tensor:
        """Log-Likelihood bei tagesaufgeloester Intervallzensierung.

        Ereignis an Tag k  ->  log( F(k) - F(k-1) )
        zensiert bei H     ->  log( 1 - F(H) )
        """
        t = t_event.to(raw.dtype)
        tau = torch.stack([t - 1.0, t, torch.full_like(t, float(horizon))], dim=1)
        Fv = self.cif(raw, tau)
        f_lo, f_hi, f_H = Fv[:, 0], Fv[:, 1], Fv[:, 2]
        ll_event = torch.log((f_hi - f_lo).clamp_min(EPS))
        ll_cens = torch.log((1.0 - f_H).clamp_min(EPS))
        return torch.where(event.bool(), ll_event, ll_cens)


def bernstein_nll(head: BernsteinSurvivalHead, raw: torch.Tensor,
                  event: torch.Tensor, t_event: torch.Tensor, horizon: float,
                  weight: torch.Tensor | None = None) -> torch.Tensor:
    ll = head.log_likelihood(raw, event, t_event, horizon)
    if weight is not None:
        return -(weight * ll).sum() / weight.sum().clamp_min(EPS)
    return -ll.mean()

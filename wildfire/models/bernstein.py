"""Stetiger Survival-Kopf: monotone Transformationsfunktion in Bernstein-Basis.

Bisher gab jedes Modell eine TABELLE aus - H Hazards fuer H feste Tage.
Damit ist das Risiko nur an den gelernten Stuetzstellen definiert; nach
37,5 Tagen zu fragen ginge nur ueber Interpolation, und ein anderer
Horizont braeuchte ein neues Modell.

Hier wird stattdessen eine FUNKTION gelernt. Eine in tau monoton wachsende
Transformationsfunktion h wird in Bernstein-Basis vom Grad M entwickelt,

    h(tau | x) = sum_k theta_k(x) * B_{k,M}(tau / tau_max),

und daraus die Survival-Funktion

    S(tau | x) = sigmoid(-h(tau)) / sigmoid(-h(0)),      F = 1 - S.

**Warum der Bruch.** sigmoid(h(tau)) allein waere bei tau = 0 nicht null,
sondern sigmoid(theta_0). Die erste Fassung benutzte genau das: F(0) lag bei
Initialisierung bei 9e-4 - rund 70 % des echten 100-Tage-Risikos auf einem
Zeitpunkt, an dem nichts passieren kann. Das Training sah log(1 - F(H)), die
Auswertung ueber S(k)/S(k-1) dagegen die normierte Kurve; Training und
berichtete NLL waren nicht dieselbe Groesse. Mit der Normierung ist
S(0) = 1 per Konstruktion, und beide Wege rechnen dasselbe.

**Warum Bernstein.** Ein Bernstein-Polynom ist genau dann monoton wachsend,
wenn seine Koeffizienten es sind: theta_0 <= theta_1 <= ... <= theta_M.
Das laesst sich durch die Parametrisierung erzwingen statt durch eine
Strafe im Verlust - die Monotonie gilt dann exakt und fuer jede Eingabe,
nicht nur ungefaehr und nur dort, wo trainiert wurde. Das Netz gibt
M+1 rohe Werte aus; daraus werden die theta durch kumulative Addition von
softplus gebildet. Die Normierung durch eine Konstante je Beispiel aendert
an der Monotonie nichts.

**Wie gross ist M?** Der Grad steuert, wie viele Kruemmungen die Kurve
haben kann. M = 12 gibt bei 13 Ausgabewerten je Pixel Reserve fuer Anstieg,
Sommergipfel und Herbstabfall innerhalb von H = 100 Tagen.

**Likelihood.** Die Ereigniszeiten sind tagesaufgeloest. Ein Ereignis "an
Tag k" heisst also nicht T = k, sondern T in (k-1, k] - das ist
Intervallzensierung, und die exakte Likelihood ist S(k-1) - S(k). Gerechnet
wird im Log-Raum und in float64:

    log(S(k-1) - S(k)) = log S(k-1) + log(-expm1(log S(k) - log S(k-1)))

Die erste Fassung bildete F(k) - F(k-1) in float32 und klemmte auf 1e-7.
Bei einem Tages-Hazard um 1e-6 - typisch fuer ruhige Pixel - lag die
Differenz unter der Klemme; der Gradient war genau fuer die
"ueberraschenden" Zuendungen null.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.models.losses import INIT_CIF_H

EPS = 1e-7


def bernstein_basis(u: torch.Tensor, degree: int) -> torch.Tensor:
    """B_{k,M}(u) fuer k = 0..M. u beliebige Form -> (..., M+1).

    Direkt ueber Potenzen statt ueber log(u): bei u = 0 ist B_0 exakt 1 und
    alle uebrigen exakt 0 (torch.pow(0, 0) = 1). Die fruehere Log-Form
    klemmte u auf 1e-7 und lieferte dort B_1(0) = 1,2e-6 statt 0.
    """
    u = u.clamp(0.0, 1.0).unsqueeze(-1)
    k = torch.arange(degree + 1, device=u.device, dtype=u.dtype)
    coef = torch.tensor([math.comb(degree, i) for i in range(degree + 1)],
                        device=u.device, dtype=u.dtype)
    return coef * torch.pow(u, k) * torch.pow(1.0 - u, degree - k)


class BernsteinSurvivalHead(nn.Module):
    """Wandelt Netzausgaben in eine monotone Survival-Funktion mit S(0) = 1.

    ``tau_max`` legt fest, ueber welchen Zeitraum die Basis gespannt ist.
    Abfragen darueber hinaus werden geklemmt - die Funktion ist auf
    [0, tau_max] definiert, nicht darueber hinaus extrapolierbar.

    Initialisierung: Bei rohen Ausgaben von null soll F(tau_max) = ``init_cif``
    gelten. theta_0 = ``init_offset``; die M Schritte bekommen einen festen
    Offset, sodass theta linear bis zum passenden theta_M steigt. Frueher
    lag theta_M bei Null-Ausgabe bei -7 + 12 * softplus(0) = +1,3, also
    F(tau_max) = 79 %.
    """

    def __init__(self, degree: int = 12, tau_max: float = 365.0,
                 init_offset: float = -12.0, init_cif: float = INIT_CIF_H) -> None:
        super().__init__()
        self.degree = degree
        self.tau_max = float(tau_max)
        self.n_out = degree + 1
        self.init_offset = float(init_offset)
        g0 = 1.0 / (1.0 + math.exp(-self.init_offset))
        g_end = g0 + init_cif * (1.0 - g0)
        step = (math.log(g_end / (1.0 - g_end)) - self.init_offset) / degree
        self.step_offset = math.log(math.expm1(step))      # softplus^-1(step)

    def thetas(self, raw: torch.Tensor) -> torch.Tensor:
        """(..., M+1) rohe Werte -> monoton wachsende Koeffizienten."""
        first = raw[..., :1] + self.init_offset
        steps = F.softplus(raw[..., 1:] + self.step_offset)
        return torch.cat([first, first + torch.cumsum(steps, dim=-1)], dim=-1)

    def log_survival(self, raw: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """log S(tau), float64. ``tau`` broadcastet gegen ``raw``.

        raw:  (N, M+1)      tau: (N, T) oder (T,)   ->   (N, T)
        """
        theta = self.thetas(raw.double())                   # (N, M+1)
        tau = tau.to(device=raw.device, dtype=torch.float64)
        if tau.dim() == 1:
            tau = tau.unsqueeze(0).expand(raw.shape[0], -1)
        B = bernstein_basis(tau / self.tau_max, self.degree)    # (N, T, M+1)
        h = (B * theta.unsqueeze(1)).sum(-1)                # (N, T)
        # h(0) = theta_0, weil B_0(0) = 1 und alle anderen 0
        return F.logsigmoid(-h) - F.logsigmoid(-theta[:, :1])

    def cif(self, raw: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Kumulatives Risiko F(tau) mit F(0) = 0, im dtype von ``raw``."""
        return (-torch.expm1(self.log_survival(raw, tau))).to(raw.dtype)

    def survival(self, raw: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_survival(raw, tau)).to(raw.dtype)

    def hazards(self, raw: torch.Tensor, horizon: int) -> torch.Tensor:
        """(N, H) diskrete Hazards h(k) = 1 - S(k)/S(k-1), k = 1..H.

        Ueber die Log-Survival in float64 - als Quotient von 1 - F in float32
        laege die Aufloesung bei 6e-8 und damit im Bereich der Hazards
        ruhiger Pixel.
        """
        tau = torch.arange(0, horizon + 1, device=raw.device, dtype=torch.float64)
        ls = self.log_survival(raw, tau)
        return (-torch.expm1(ls[:, 1:] - ls[:, :-1])).to(raw.dtype)

    def log_likelihood(self, raw: torch.Tensor, event: torch.Tensor,
                       t_event: torch.Tensor, horizon: float) -> torch.Tensor:
        """Log-Likelihood bei tagesaufgeloester Intervallzensierung.

        Ereignis an Tag k  ->  log( S(k-1) - S(k) )
        zensiert bei H     ->  log S(H)
        """
        t = t_event.to(torch.float64)
        tau = torch.stack([t - 1.0, t, torch.full_like(t, float(horizon))], dim=1)
        ls = self.log_survival(raw, tau)
        ls_lo, ls_hi, ls_H = ls[:, 0], ls[:, 1], ls[:, 2]
        # Die Klemme greift nur, wenn S(k) und S(k-1) in float64 gleich sind -
        # dann ist der Gradient ohnehin null. Sie verhindert -inf, das in
        # torch.where auch im NICHT gewaehlten Zweig NaN-Gradienten erzeugt.
        ll_event = ls_lo + torch.log((-torch.expm1(ls_hi - ls_lo)).clamp_min(1e-300))
        return torch.where(event.bool(), ll_event, ls_H).to(raw.dtype)


def bernstein_nll(head: BernsteinSurvivalHead, raw: torch.Tensor,
                  event: torch.Tensor, t_event: torch.Tensor, horizon: float,
                  weight: torch.Tensor | None = None) -> torch.Tensor:
    ll = head.log_likelihood(raw, event, t_event, horizon)
    if weight is not None:
        return -(weight * ll).sum() / weight.sum().clamp_min(EPS)
    return -ll.mean()

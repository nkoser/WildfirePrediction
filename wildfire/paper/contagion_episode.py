"""Contagion-STS mit Brenndauer ab Episodenbeginn statt ab jeder einzelnen MODIS-Detektion.

contagion.py und sts_extreme.py bleiben unveraendert. Im Original ist jede Detektion der letzten obs_days Tage
ein eigenes Ereignis mit Alter u (Tage seit der Detektion) und klingt mit D_j(u) ab; ein Pixel, das 5 Tage am
Stueck detektiert wird, zaehlt als 5 Feuer der Alter 0..4. D ist dort eher eine Nachwirkzeit nach der letzten
Detektion als eine Brenndauer.

Hier wird je Pixel die juengste Feuer-Episode gebildet: aufeinanderfolgende Detektionen mit Luecken von
hoechstens ep_gap Tagen (Wolken, Rauch, Ueberflug) gehoeren zu einer Episode. u_s = Alter des Episodenbeginns,
u_l = Alter der letzten Detektion (beide am Stichtag, u_l <= u_s). Am Tag t nach dem Stichtag (Vortag von Tag
t+1, wie im Original) brennt der Pixel mit

    b_j(t) = D_j(u_s + t) / D_j(u_s - u_l)        (Episode hat bis zur letzten Detektion gebrannt)

D_j(a) ist damit die Ueberlebensfunktion der Episodendauer: P(Feuer brennt nach a Tagen seit Ausbruch noch).
Aeltere, schon beendete Episoden im Fenster gehen nicht ein. Die vorhergesagte Ausbreitung (rollout) zaehlt das
Alter schon im Original ab dem ersten vorhergesagten Feuer und bleibt unveraendert; dasselbe gilt fuer
Zuendung, Kern, Koepfe und alle Auswertungen.

Der Checkpoint traegt ep_gap nicht; log_burning/params sind identisch zum Original, nur exogenous aendert sich.
"""

from __future__ import annotations

import torch

from wildfire.paper import sts_extreme


def episode_ages(obs: torch.Tensor, gap: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """obs (B, D, N) binaer, letzter Index = Stichtag (Alter 0) -> u_s, u_l (B, N) long, any (B, N) bool."""
    o = obs.flip(1) > 0                                       # o[:, u] = Detektion mit Alter u
    any_ = o.any(1)
    u_l = o.float().argmax(1)                                 # erste (juengste) Detektion
    u_s = u_l.clone()
    for u in range(1, o.shape[1]):
        ext = o[:, u] & (u > u_l) & (u - u_s <= gap + 1)
        u_s = torch.where(ext, torch.full_like(u_s, u), u_s)
    return u_s, u_l, any_


class EpisodeMixin:
    ep_gap = 2

    def exogenous(self, obs, a, k, logD):
        """log(1 - h_exo) (B, H, 169): Zuendung und Ausbreitung der juengsten beobachteten Episode, Tag 1..H."""
        B, _, N = obs.shape
        KK = self.K * self.K
        u_s, u_l, any_ = episode_ages(obs, self.ep_gap)
        t = torch.arange(self.H, device=obs.device)
        num = logD.gather(2, u_s[..., None] + t)                                        # (B, N, H)
        den = logD.gather(2, (u_s - u_l)[..., None])                                     # (B, N, 1)
        b_obs = ((num - den).clamp(max=0).exp() * any_[..., None].float()).transpose(1, 2)  # (B, H, N)
        U = self._unfold(b_obs.reshape(B * self.H, N)).view(B, self.H, KK, N)
        lo = torch.log1p(-(k[:, None] * U).clamp(max=1 - 1e-6))
        return torch.log1p(-a.clamp(max=1 - 1e-6)) + lo.sum(2)


def episode_class(gap: int):
    """ContagionExt aus sts_extreme mit Episoden-Brenndauer (Klasse, damit build/load_any sie benutzen)."""
    return type("ContagionEpisodeExt", (EpisodeMixin, sts_extreme.ContagionExt), {"ep_gap": int(gap)})

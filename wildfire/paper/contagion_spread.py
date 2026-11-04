"""Contagion-STS mit gerichtetem, wetterabhaengigem Ausbreitungskern (Antwort auf die runde Ausbreitung im Teaser).

contagion.py und sts_extreme.py bleiben unveraendert; SpreadContagion erbt von sts_extreme.ContagionExt und
ersetzt nur params, exogenous und rollout. Im Original ist der K x K-Kern je Empfaengerpixel frei
(sigmoid(ker(token))) und ueber alle Tage gleich. Hier:

    logit k_ij(t) = base_ij + (r_ij / R) * ( w_i(t) . u_ij + m_i(t) )        (j != i; Mitte wie im Original)

  base_ij   freier Kern wie im Original (ker(token)); Start: -4.5 fuer direkte Nachbarn, je Pixel Abstand
            0.7 weniger (sonst zuendet ein grosser Kern anfangs zu viel), Mitte -1.0 (Persistenz)
  u_ij      Einheitsvektor der Ausbreitungsrichtung von j nach i (Gitter: Zeile, Spalte), r_ij Abstand,
            R = K // 2; der Richtungsterm waechst mit dem Abstand, der Kern wird also zur Ellipse gestreckt
  w_i(t)    Schubvektor (2): dir(token) + A (u10, v10)_i(t); A startet bei 0.3 * (Wind nach Osten -> Spalte +,
            nach Norden -> Zeile -), alles andere bei null
  m_i(t)    Reichweite: mag(token) + lin(25 Wetterkanaele des Pixels am Tag t), Start null

Wetter am Tag t:  ohne Oracle der letzte Eingabetag (Kern ueber alle Tage gleich, k (B, KK, N));
                  mit Oracle (future_days > 0, set_future) das beobachtete Wetter der Tage 1..kf, danach der
                  letzte bekannte Tag (Kern je Tag, k (B, H, KK, N)). Obere Schranke, kein Einsatzmodell.
--spread free     nur der groessere Kern (Richtungs- und Reichweitenterm aus).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wildfire.paper.sts_extreme import ContagionExt

WX = list(range(1, 26))          # Wetterkanaele im Cache (ERA5, FWI-System, VPD), wie future_weather.WEATHER_CH
U10, V10 = 1, 2                  # Position von u10, v10 innerhalb von WX


class SpreadContagion(ContagionExt):
    def __init__(self, k_px: int = 0, k_reg: int = 0, spread: str = "ell", future_days: int = 0, **kw):
        super().__init__(k_px, k_reg, **kw)
        if spread not in ("ell", "free"):
            raise ValueError(f"spread {spread!r}")
        self.spread, self.future_days = spread, int(future_days)
        d, K = kw.get("d", 96), self.K
        R = K // 2
        a = torch.arange(K, dtype=torch.float32) - R
        dy, dx = torch.meshgrid(a, a, indexing="ij")                        # Nachbar j relativ zum Empfaenger i
        r = torch.sqrt(dy ** 2 + dx ** 2).flatten()
        u = torch.stack([-dy, -dx], -1).flatten(0, 1) / r.clamp_min(1e-6)[:, None]   # Richtung j -> i
        self.register_buffer("u_dir", u)                                    # (KK, 2)
        self.register_buffer("r_rel", r / max(R, 1))                        # (KK,)
        with torch.no_grad():
            self.ker.bias.copy_(-4.5 - 0.7 * (r - 1).clamp_min(0))
            self.ker.bias[K * K // 2] = -1.0
        self._future = (None, None)
        if spread == "ell":
            self.dir_tok = nn.Linear(d, 2)
            self.mag_tok = nn.Linear(d, 1)
            self.wind = nn.Linear(2, 2, bias=False)
            self.mag_wx = nn.Linear(len(WX), 1, bias=False)
            for lin in (self.dir_tok, self.mag_tok, self.mag_wx):
                nn.init.zeros_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            with torch.no_grad():                                           # (u, v) -> (Zeile, Spalte)
                self.wind.weight.copy_(torch.tensor([[0.0, -0.3], [0.3, 0.0]]))

    def set_future(self, xf=None, kf=None) -> None:
        """xf (B, F, 25, 13, 13) normiert, kf (B,) Zahl der bekannten Folgetage; gilt fuer genau den naechsten Aufruf."""
        self._future = (xf, kf)

    def weather_days(self, x):
        """-> Wetter je Tag (B, T, 25, N): T = 1 ohne Oracle (letzter Eingabetag), sonst H."""
        last = x[:, -1, WX].flatten(2)                                      # (B, 25, N)
        xf, kf = self._future
        if not self.future_days or xf is None:
            return last[:, None]
        B, Fd = xf.shape[:2]
        xf = xf.flatten(3)                                                  # (B, F, 25, N)
        t = torch.arange(self.H, device=x.device)
        known = t[None] < kf[:, None]                                       # (B, H)
        # Tag t >= kf: letzter bekannter Tag (kf - 1) bzw. letzter Eingabetag, wenn kf = 0
        src = torch.minimum(t[None].expand(B, -1), (kf - 1).clamp_min(0)[:, None]).clamp(max=Fd - 1)
        seq = xf.gather(1, src[:, :, None, None].expand(-1, -1, xf.shape[2], xf.shape[3]))
        has = (kf > 0)[:, None, None, None] | known[:, :, None, None]
        return torch.where(has, seq, last[:, None].expand_as(seq))

    def params(self, x, doy, pix=None):
        x = x[:, -self.hist:]
        if self.spread == "free":
            return super().params(x, doy, pix)
        pix = self.encode_input(x, doy) if pix is None else pix
        obs, a, _, logD = super().params(x, doy, pix)
        with torch.autocast(device_type=x.device.type, enabled=False):
            pix = pix.float()
            wx = self.weather_days(x.float())                               # (B, T, 25, N)
            base = self.ker(pix)                                            # (B, N, KK)
            w = self.dir_tok(pix)[:, None] + self.wind(wx[:, :, [U10, V10]].transpose(2, 3))   # (B, T, N, 2)
            m = self.mag_tok(pix)[:, None] + self.mag_wx(wx.transpose(2, 3))                    # (B, T, N, 1)
            ell = self.r_rel * (w @ self.u_dir.t() + m)                     # (B, T, N, KK); Mitte r = 0
            k = torch.sigmoid(base[:, None] + ell).transpose(2, 3)          # (B, T, KK, N)
            if k.shape[1] == 1:
                k = k[:, 0]
            elif k.shape[1] != self.H:
                raise RuntimeError("Oracle-Kern braucht H Tage")
        return obs, a, k, logD

    def exogenous(self, obs, a, k, logD):
        if k.dim() == 3:
            return super().exogenous(obs, a, k, logD)
        B, D, N = obs.shape
        KK = self.K * self.K
        idx = (torch.arange(self.H, device=obs.device)[:, None]
               + torch.arange(D - 1, -1, -1, device=obs.device)[None, :])
        pw = logD[:, :, idx].exp().permute(0, 2, 3, 1)
        b_obs = 1 - torch.prod(1 - obs[:, None] * pw, dim=2)
        U = self._unfold(b_obs.reshape(B * self.H, N)).view(B, self.H, KK, N)
        lo = torch.log1p(-(k * U).clamp(max=1 - 1e-6))
        return torch.log1p(-a.clamp(max=1 - 1e-6)) + lo.sum(2)

    def rollout(self, log_exo, k, logD, keep_b: bool = False):
        if k.dim() == 3:
            return super().rollout(log_exo, k, logD, keep_b)
        B, _, N = log_exo.shape
        c = self.K * self.K // 2
        logS = log_exo.new_zeros(B, N)
        S = log_exo.new_ones(B, N)
        b_pred = log_exo.new_zeros(B, N)
        out, bs, fs = [logS], [], []
        for s in range(self.H):
            if keep_b:
                bs.append(b_pred)
            lp = torch.log1p(-(k[:, s] * self._unfold(b_pred)).clamp(max=1 - 1e-6))
            l1mh = log_exo[:, s] + lp.sum(1) - lp[:, c]
            f = S * -torch.expm1(l1mh)
            logS = logS + l1mh
            S = torch.exp(logS)
            fs.append(f)
            Dk = logD[..., :s + 1].flip(-1).exp()
            b_pred = (torch.stack(fs, -1) * Dk).sum(-1).clamp(max=1.0)
            out.append(logS)
        return (torch.stack(out, -1), torch.stack(bs, 1)) if keep_b else torch.stack(out, -1)

    def forward(self, x, doy, rate: bool = False):
        out = super().forward(x, doy, rate)
        self._future = (None, None)
        return out


def kernel_summary(k: torch.Tensor, K: int) -> dict:
    """Mittlere Richtungsstaerke eines Kerns (B, [T,] KK, N): Laenge des gewichteten Schwerpunkts der Nicht-Mitte."""
    R = K // 2
    a = torch.arange(K, dtype=torch.float32, device=k.device) - R
    dy, dx = torch.meshgrid(a, a, indexing="ij")
    w = k.clone()
    w[..., K * K // 2, :] = 0
    tot = w.sum(-2).clamp_min(1e-9)
    cy = (w * dy.flatten()[:, None]).sum(-2) / tot
    cx = (w * dx.flatten()[:, None]).sum(-2) / tot
    return {"centroid_len": float(torch.sqrt(cy ** 2 + cx ** 2).mean()), "k_sum": float(tot.mean())}

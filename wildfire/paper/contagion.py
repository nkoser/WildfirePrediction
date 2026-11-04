"""Survival-Feld mit Zuendung und Ausbreitung: der Contagion-STS.

Statt einer freien Risikokurve je Pixel wird der Tages-Hazard aus zwei Quellen
zusammengesetzt, die das Backbone (STS-Rumpf) je Pixel parametrisiert:

    1 - h_i(t) = (1 - a_i(t)) * prod_{j in N_i} (1 - k_ij * b_j(t-1))

  a_i(t)   Zuendung: ein Hazard je Tag 1..H (wie der Logistic-Hazard-Kopf)
  k_ij     Ausbreitung von Nachbar j auf Pixel i, gelernter K x K-Kern je
           Empfaengerpixel (Wind, Hang, Brennstoff stecken im Pixel-Token)
  b_j(t)   Wahrscheinlichkeit, dass j am Tag t brennt:
             beobachtet  MODIS-Feuer der letzten obs_days Fenstertage, klingt
                         mit der Brenndauer D_j(Alter) ab (burn: geometrisch
                         rho_j^Alter oder gelernte Kurve ueber das Alter)
             vorhergesagt  eigenes Ereignis f_j(s) = S_j(s-1) h_j(s), ebenfalls
                         mit D_j abklingend (Mean-Field, ueber tau gerollt)
           Der Mittelpunkt des Kerns (j = i) wirkt nur ueber das beobachtete
           Feuer: Persistenz eines am Stichtag brennenden Pixels.

Pixel:   S_i(t) = prod_{s <= t} (1 - h_i(s)), durch Vorwaertsrollen ueber
         t = 1..H; die vorhergesagte Ausbreitung ist eine Mean-Field-Naeherung.
Ausschnitt: Bis zum ersten Feuer im Ausschnitt brennt dort nach dem Stichtag
         nichts, interne Ausbreitung gibt es also noch nicht. Der Hazard des
         ersten Ereignisses ist exakt das Produkt der Pixel-Hazards aus
         Zuendung und beobachtetem Feuer - keine Naeherung, kein eigener Kopf.
         Teilregionen R (region_log_survival): wie der Ausschnitt, dazu die
         vorhergesagte Ausbreitung aus Pixeln ausserhalb von R (Mean-Field).

Feuer von ausserhalb des 13 x 13-Ausschnitts ist unbeobachtet (Rand mit
Nullen); es landet im Zuendungs-Hazard der Randpixel.

blind (Version 3): Der Rumpf sieht die Feuerkanaele nicht (normiert auf 0
gesetzt, wie masked_std in channels.py), a, k und D haengen also nicht von
der Feuerhistorie ab. Feuer in der Naehe wirkt dann nur ueber b_j und den
Kern; sonst kann die Zuendung die Ausbreitung uebernehmen.
  thermal  Feuermaske und die feuerempfindlichen MODIS-Kanaele: refl_b07,
           LST, Emissivitaet, Helligkeitstemperaturen (45-53)
  modis    zusaetzlich Vegetation und Reflexion (38-53), auch Brandnarben

Die Ausgabe passt in die Schnittstelle der STS-Koepfe: roh = log S(0..H)
(CurveHead), damit laufen Training, Auswahl und alle Auswertungen unveraendert.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.models.surv_heads import _Head, _tau_index
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.sts_dctm import P, STSDCTM

H = 100
BLIND = {"none": (), "thermal": (FIRE_CH, *range(45, 54)), "modis": (FIRE_CH, *range(38, 54))}


class CurveHead(_Head):
    """Die Rohausgabe ist schon die Kurve log S(0..H); abgelesen an ganzen Tagen."""

    def __init__(self, horizon: int = H):
        super().__init__()
        self.horizon, self.n_out = int(horizon), int(horizon) + 1

    def log_survival(self, raw, tau):
        return raw.double().gather(1, _tau_index(tau, raw.shape[0], raw.device, self.horizon))


class ContagionSTS(STSDCTM):
    def __init__(self, c_in: int = 55, d: int = 96, heads: int = 4, depth: int = 4, p: float = 0.1,
                 init_cif_px: float = 0.02, init_cif_patch: float = 0.2, kernel: int = 5, obs_days: int = 7,
                 burn: str = "geometric", horizon: int = H, blind: str = "none", **kw):
        super().__init__(c_in, d, heads, depth, p, init_cif_px, init_cif_patch, head="lh", **kw)
        del self.px_lin, self.pa_lin, self.pool, self.pool_norm, self.cls
        self.head_kind = "contagion"
        self.H, self.K, self.obs_days = int(horizon), int(kernel), int(obs_days)
        self.A = self.H + self.obs_days                       # Feueralter 0 .. A-1 Tage
        self.burn = burn
        self.blind = blind
        self.blind_ch = list(BLIND[blind])
        if burn == "curve":                                   # Brenndauer als Kurve ueber das Alter, Start 0,7 je Tag
            self.burn_shape = nn.Parameter(torch.full((self.A - 1,), math.log(math.expm1(-math.log(0.7)))))
        elif burn != "geometric":
            raise ValueError(f"burn {burn!r}")
        self.px_head, self.pa_head = CurveHead(horizon), CurveHead(horizon)
        q = 0.5 * float(init_cif_px)                          # Zuendung allein: halbes Pixelrisiko bis H
        h0 = -math.expm1(math.log1p(-q) / horizon)
        self.ign_bias = math.log(h0 / (1 - h0))
        self.ign = nn.Linear(d, horizon)
        self.ker = nn.Linear(d, self.K * self.K)
        self.dur = nn.Linear(d, 1)
        for lin in (self.ign, self.ker, self.dur):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        with torch.no_grad():
            self.ker.bias.fill_(-4.5)                         # k ~ 0,011 je Nachbar und Tag
            self.ker.bias[self.K * self.K // 2] = -1.0         # Persistenz ~ 0,27
        self.register_buffer("fire_mu", torch.zeros(()))
        self.register_buffer("fire_sd", torch.ones(()))

    def set_fire_norm(self, mu: float, sd: float) -> None:
        """Normierung des Feuerkanals, damit der Brennzustand aus der Eingabe zurueckgerechnet werden kann."""
        self.fire_mu.fill_(mu); self.fire_sd.fill_(sd)

    def log_burning(self, dur):
        """log D_j(a) (B, 169, A): Wahrscheinlichkeit, dass ein Feuer a Tage nach dem Ausbruch noch brennt, D(0) = 1.

        geometric  D = rho_j^a, ein Wert je Pixel (Version 1)
        curve      D = exp(-s_j C(a)), C monoton aus gelernten Tagesraten (softplus), s_j in (0, 2) je Pixel;
                   grosse Braende brennen Wochen, eine geometrische Rate unterschaetzt das."""
        if self.burn == "curve":
            C = F.pad(F.softplus(self.burn_shape).cumsum(0), (1, 0))
            return -(2 * torch.sigmoid(dur))[..., None] * C
        age = torch.arange(self.A, device=dur.device, dtype=torch.float32)
        return torch.log(torch.sigmoid(dur).clamp_min(1e-6))[..., None] * age

    def encode_input(self, x, doy):
        """Token fuer a, k und D; mit blind ohne die Feuerkanaele (normiert 0 = Mittelwert, keine Information)."""
        if self.blind_ch:
            x = x.clone()
            x[:, :, self.blind_ch] = 0.0
        return self.encode(x, doy)

    def params(self, x, doy, pix=None):
        """-> beobachtetes Feuer (B, D, 169), a (B, H, 169), k (B, K*K, 169), log D (B, 169, A); float32.
        Beobachtet wird hoechstens die Historie des Rumpfs (self.hist Tage); pix: schon kodierte Token.
        Der Brennzustand kommt immer aus dem unveraenderten Feuerkanal."""
        x = x[:, -self.hist:]
        pix = self.encode_input(x, doy) if pix is None else pix
        with torch.autocast(device_type=x.device.type, enabled=False):
            pix = pix.float()
            obs = ((x[:, -self.obs_days:, FIRE_CH].float() * self.fire_sd + self.fire_mu) > 0.5).float().flatten(2)
            a = torch.sigmoid(self.ign(pix) + self.ign_bias).transpose(1, 2)
            k = torch.sigmoid(self.ker(pix)).transpose(1, 2)
            logD = self.log_burning(self.dur(pix)[..., 0])
        return obs, a, k, logD

    def _unfold(self, b):
        """(M, 169) -> (M, K*K, 169): Werte der Nachbarn je Empfaengerpixel."""
        return F.unfold(b.view(-1, 1, P, P), self.K, padding=self.K // 2)

    def exogenous(self, obs, a, k, logD):
        """log(1 - h_exo) (B, H, 169): Zuendung und Ausbreitung beobachteter Feuer, Tag 1..H."""
        B, D, N = obs.shape
        KK = self.K * self.K
        # Brennzustand am Vortag t = 0..H-1 einer Beobachtung mit Alter u am Stichtag (obs[:, -1]: u = 0)
        idx = (torch.arange(self.H, device=obs.device)[:, None]
               + torch.arange(D - 1, -1, -1, device=obs.device)[None, :])               # (H, D)
        pw = logD[:, :, idx].exp().permute(0, 2, 3, 1)                                  # (B, H, D, N)
        b_obs = 1 - torch.prod(1 - obs[:, None] * pw, dim=2)                           # (B, H, N)
        U = self._unfold(b_obs.reshape(B * self.H, N)).view(B, self.H, KK, N)
        lo = torch.log1p(-(k[:, None] * U).clamp(max=1 - 1e-6))                         # (B, H, KK, N)
        return torch.log1p(-a.clamp(max=1 - 1e-6)) + lo.sum(2)

    def rollout(self, log_exo, k, logD, keep_b: bool = False):
        """Pixel: log S(0..H) (B, 169, H+1) mit vorhergesagter Ausbreitung (Mean-Field);
        mit keep_b zusaetzlich der vorhergesagte Brennzustand am Vortag je Schritt (B, H, 169)."""
        B, _, N = log_exo.shape
        c = self.K * self.K // 2
        logS = log_exo.new_zeros(B, N)
        S = log_exo.new_ones(B, N)
        b_pred = log_exo.new_zeros(B, N)
        out, bs, fs = [logS], [], []
        for s in range(self.H):
            if keep_b:
                bs.append(b_pred)
            lp = torch.log1p(-(k * self._unfold(b_pred)).clamp(max=1 - 1e-6))           # (B, KK, N)
            l1mh = log_exo[:, s] + lp.sum(1) - lp[:, c]                                 # Mitte nur beobachtet
            f = S * -torch.expm1(l1mh)
            logS = logS + l1mh
            S = torch.exp(logS)
            fs.append(f)                                       # erstes Feuer an Tag s + 1
            Dk = logD[..., :s + 1].flip(-1).exp()              # D(s - i) fuer das Feuer von Tag i + 1
            b_pred = (torch.stack(fs, -1) * Dk).sum(-1).clamp(max=1.0)
            out.append(logS)
        return (torch.stack(out, -1), torch.stack(bs, 1)) if keep_b else torch.stack(out, -1)

    def forward(self, x, doy, rate: bool = False):
        pix = self.encode_input(x, doy)
        obs, a, k, logD = self.params(x, doy, pix)
        with torch.autocast(device_type=x.device.type, enabled=False):
            log_exo = self.exogenous(obs, a, k, logD)
            ls_px = self.rollout(log_exo, k, logD)
            ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0))
        if rate:
            return ls_px.view(x.shape[0], P, P, -1), ls_pa, self.rate_head(pix)
        return ls_px.view(x.shape[0], P, P, -1), ls_pa

    @torch.no_grad()
    def decompose(self, x, doy):
        """Zerlegung fuer die Auswertung: kumulatives Risiko des Ausschnitts gesamt, nur aus Zuendung
        und nur aus Ausbreitung beobachteter Feuer, je (B, H+1); dazu die erwartete Brenndauer
        (Tage, Summe von D) und der mittlere Kern."""
        obs, a, k, logD = self.params(x, doy)
        log_exo = self.exogenous(obs, a, k, logD)
        log_ign = torch.log1p(-a.clamp(max=1 - 1e-6))
        cif = lambda l: -torch.expm1(F.pad(l.sum(2).cumsum(1), (1, 0)))  # noqa: E731
        return {"total": cif(log_exo), "ignition": cif(log_ign), "spread_obs": cif(log_exo - log_ign),
                "burn_days": logD.exp().sum(-1), "kernel": k.mean(-1)}

    @torch.no_grad()
    def region_log_survival(self, x, doy, mask):
        """log S (B, H+1) einer Region (mask (169,) bool) bis zum ersten Feuer darin.

        Bis dahin brennt in R nichts; Ausbreitung kommt nur aus beobachtetem Feuer und aus
        vorhergesagtem Feuer AUSSERHALB von R. Fuer R = ganzer Ausschnitt ist das exakt."""
        obs, a, k, logD = self.params(x, doy)
        log_exo = self.exogenous(obs, a, k, logD)
        _, b = self.rollout(log_exo, k, logD, keep_b=True)                              # (B, H, N)
        B, Hh, N = b.shape
        b_out = (b * (~mask).float()).reshape(B * Hh, N)
        lp = torch.log1p(-(k.repeat_interleave(Hh, 0) * self._unfold(b_out)).clamp(max=1 - 1e-6))
        c = self.K * self.K // 2
        l_pred = (lp.sum(1) - lp[:, c]).view(B, Hh, N)
        return F.pad((log_exo + l_pred)[..., mask].sum(2).cumsum(1), (1, 0))

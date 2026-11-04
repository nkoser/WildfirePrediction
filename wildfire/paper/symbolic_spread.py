"""Schritt A: symbolische Ausbreitungskerne (zellulaerer Automat, Reaktion-Diffusion, beides) mit wenigen Parametern.

Frage: Trifft ein festes physikalisches bzw. CA-Gesetz, dessen Parameter nur linear von Wetter, Gelaende und
Brennstoff abhaengen, die Ausbreitung grosser Feuer besser als der gelernte Contagion-Kern? Ausgewertet auf
denselben Ereignissen und mit denselben Metriken wie firemap_bench (val 2020-22, test 2023-25); angepasst auf
Ereignissen der Trainingsjahre (gleiche Auswahlregel). Keine Gewichte aus anderen Modellen.

Region je Ereignis: Rahmen aus firemap_bench, Gitter = Wuerfelpixel. Kovariaten z je Pixel (37): Wetter des
letzten Eingabetags (t2m, Niederschlag, Bodenfeuchte, t2m_max, FFMC, DMC, DC, ISI, BUI, FWI, VPD, |Wind|),
Gebaeude, Worldpop, Hang, TPI, TWI, NDVI, LAI, Landbedeckung one-hot. Wind (u10, v10), Hangvektor g = grad DEM.
Brennstoff: Landpixel, die im Fenster noch nicht gebrannt haben.

ca      Probabilistischer CA (Alexandridis-Typ) auf 5 x 5, M_CA Teilschritte je Tag. Von Nachbar j (Versatz o,
        Abstand r, Richtung u von j nach i) auf Empfaenger i:
            log q_io = log q0_i + c_r (r - 1) + (W (u10, v10)_i + b + a_s g_i) . u
            Druck P_i = sum_o q_io I(Alter_j) B_j,   h_i = 1 - exp(-((P_i + e)^gamma_i - e^gamma_i) - a_i / M)
        CA-Term gamma_i (kooperative Zuendung: gamma > 1 = Schwellenregel, mehrere brennende Nachbarn zuenden
        ueberproportional), I(Alter) = Intensitaet einer Zelle nach Alter ihrer Episode (Aufbau / Abklingen),
        Brenndauer: Fortbestehen je Tag sigmoid(rho_i + beta log(1 + Alter)), gegen die beobachtete Fortsetzung.
        Training: Teacher Forcing (beobachteter Zustand am Tag d -> Zuendung am Tag d + 1, innerhalb des Tages
        Mean-Field ueber die Teilschritte) + Fortbestehen. Vorhersage: binaere Monte-Carlo-Pfade.
        Frontgroesse (Rueckkopplung aus spread_mechanisms: P(Zuendung | 1 Nachbar) verdoppelt sich mit der Zahl der
        Feuerpixel im Umkreis 10 px, auch innerhalb einer FWI-Klasse): P_i *= (1 + Masse_i)^delta, Masse_i =
        Summe der Intensitaet im 21 x 21-Fenster um i.
        Varianten: ca_lin (gamma = 1, I = 1), ca_coop (gamma frei), ca_age (I frei), ca_size (delta frei),
        ca_full (alles).
rd      Reaktion-Diffusion (Asensio-Ferragut / Mandel) mit Waerme T und Brennstoff S, M_RD Teilschritte je Tag:
            dT/dt = kappa_i Lap T - v_i . grad T + A_i S sigmoid((T - theta_i) / tau) - c_i T
            dS/dt = -eps A_i S sigmoid((T - theta_i) / tau)
        v_i = vmax tanh(W (u10, v10) + b + a_s g) (Wind und Hang), kappa, v innerhalb der Stabilitaetsgrenzen.
        Positive Rueckkopplung: mehr Waerme -> schnellere Reaktion -> mehr Waerme; Schwelle theta; Brennstoff.
        Start: T0 = T_b sum_a w_a Feuer(L - a), S0 = Brennstoff (brennende Zellen s_b). Ablesen:
            P(Ankunft <= t) = 1 - exp(-eta int_0^t R - a_i t),  P(brennt am Tag t) = 1 - exp(-eta_b int_Tag R)
        Training: Survival-NLL der ersten Ankunft ueber 30 Tage aus dem freien deterministischen Lauf (exakt, kein
        Mean-Field) + Teacher Forcing ueber einen Tag + BCE des Brennzustands. Vorhersage deterministisch.
rdca    rd plus CA-Heizterm (gerichtet, 5 x 5, kooperativ): dT/dt += (sum_o q_io B_j + e)^gamma_i - e^gamma_i
        mit B_j = S_j sigmoid((T_j - theta_j) / tau) (Zelle j brennt).

Schritte: events (Trainings-Ereignisse) | fit --model <name> (anpassen, val/test auswerten) | report.
Ausgaben NEU unter metrics/symbolic_spread<tag>/; Laeufe im Format von firemap_bench/runs.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.firemap_bench import HOR, agg, load_box, map_metrics, select_events, split_years
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.surv_prepare import T_WIN

T = 30
ZCH = [1, 5, 9, 13, 19, 20, 21, 22, 23, 24, 25, 28, 30, 32, 36, 37, 38, 40]
U10, V10, DEM, LC = 2, 3, 31, 54
N_LC = 18
DZ = len(ZCH) + 1 + N_LC
AGE = 10                                       # Intensitaet I(Alter) fuer Alter 0 .. AGE (darueber konstant)
R_CA = 2
A_MAX, C_MAX, T_MAX = 20.0, 5.0, 50.0       # Schranken des RD-Kerns (Stabilitaet)
SIZE_R = 10                                    # Frontgroesse: brennende Intensitaet im Umkreis (21 x 21), vgl. spread_mechanisms
# Name -> (Art, kooperativ / CA-Term, Alter, Frontgroesse)
MODELS = {"ca_lin": ("ca", False, False, False), "ca_coop": ("ca", True, False, False), "ca_age": ("ca", False, True, False),
          "ca_size": ("ca", False, False, True), "ca_full": ("ca", True, True, True),
          "rd": ("rd", False, False, False), "rdca": ("rd", True, False, False),
          "rd_ens": ("rdens", "mix", None, None), "rd_ens_fss": ("rdens", "mix_fss", None, None), "rd_fss": ("rdens", "fss", None, None),
          "ca_dir": ("cadir", None, None, None),
          # ohne ML: dasselbe Gesetz, aber nur globale Konstanten (Gewichte der Kovariaten-Felder auf 0 eingefroren)
          "ca_classic": ("ca", False, False, False), "rd_classic": ("rd", False, False, False),
          "rd_ens_classic": ("rdens", "mix", None, None),
          "ca_wx": ("cawx", None, None, None), "ca_wx3": ("cawx", 3, None, None), "ca_wx7": ("cawx", 7, None, None)}


# ----------------------------------------------------------------------------------------------- Daten
def episode_age(fire: np.ndarray, gap: int = 2) -> np.ndarray:
    """fire (D, h, w) bool -> Alter der laufenden Episode an brennenden Tagen, sonst -1 (Luecke <= gap Tage)."""
    age = np.full(fire.shape, -1, np.int16)
    last = np.full(fire.shape[1:], -100, np.int32)
    start = np.zeros(fire.shape[1:], np.int32)
    for d in range(fire.shape[0]):
        f = fire[d]
        new = f & (last < d - gap - 1)
        start[new] = d
        last[f] = d
        age[d][f] = (d - start)[f]
    return age


def load_event(cube: CubeReader, e: dict, mean: np.ndarray, std: np.ndarray, masks: tuple) -> dict:
    L, (r0, r1, c0, c1) = e["L"], e["box"]
    slab = load_box(cube, L, r0, r1, c0, c1, 0)                                   # (30, 55, h, w) uint8
    year, li = cube._pos[L]
    dyn, _ = cube._arrays(year)
    fut = np.asarray(dyn[li + 1:li + T + 1, r0:r1, c0:c1, 0]) > 0                  # (30, h, w)
    fire = np.concatenate([slab[:, 0] > 0, fut], 0)                               # Tage L-29 .. L+30
    x = (slab[-1].astype(np.float32) - mean[:, None, None]) / std[:, None, None]
    lc = np.clip(slab[-1, LC], 0, N_LC - 1)
    z = np.concatenate([x[ZCH], np.sqrt(x[U10] ** 2 + x[V10] ** 2)[None],
                        np.eye(N_LC, dtype=np.float32)[lc].transpose(2, 0, 1)], 0)
    gr, gc = np.gradient(x[DEM])
    land, canada = masks
    win = fire[:T_WIN].any(0)
    fut_any = fire[T_WIN:]
    first = np.where(fut_any.any(0), fut_any.argmax(0) + 1, 31).astype(np.int16)
    return {"z": np.clip(z, -5, 5).astype(np.float32), "wind": x[[U10, V10]], "grad": np.stack([gr, gc]).astype(np.float32),
            "fire": fire, "age": episode_age(fire), "land": land[r0:r1, c0:c1], "m": (land & canada)[r0:r1, c0:c1] & ~win,
            "first": first, "date": e["date"], "box": e["box"]}


N_WXZ = 11                                     # die ersten 11 Eintraege von ZCH sind Wetterkanaele (1 .. 25)


def load_wxd(cube: CubeReader, e: dict, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Wetter aller 60 Tage (L-29 .. L+30) fuer ca_wx: (60, 14, h, w) = 11 Wetterkanaele aus ZCH, |Wind|, u10, v10."""
    L, (r0, r1, c0, c1) = e["L"], e["box"]
    slab = load_box(cube, L + T, r0, r1, c0, c1, 0, days=T_WIN + T)
    x = (slab.astype(np.float32) - mean[None, :, None, None]) / std[None, :, None, None]
    sp = np.sqrt(x[:, U10] ** 2 + x[:, V10] ** 2)[:, None]
    return np.clip(np.concatenate([x[:, ZCH[:N_WXZ]], sp, x[:, [U10, V10]]], 1), -5, 5).astype(np.float16)


def to_dev(ev: dict, device) -> dict:
    out = {}
    for k, v in ev.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(v)).to(device)
            out[k] = t.float() if v.dtype == np.float32 else t
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------------------------- Modelle
def ca_geometry(device):
    a = torch.arange(-R_CA, R_CA + 1, dtype=torch.float32)
    dy, dx = torch.meshgrid(a, a, indexing="ij")
    o = torch.stack([dy.flatten(), dx.flatten()], -1)                             # Versatz j - i (unfold-Reihenfolge)
    r = o.norm(dim=-1)
    u = -o / r.clamp_min(1e-6)[:, None]                                            # Richtung j -> i
    return o.to(device), r.to(device), u.to(device)


class Fields(nn.Module):
    """Lineare Felder aus z: nf Kanaele je Pixel, Start mit konstanten Werten bias."""

    def __init__(self, bias: list[float]):
        super().__init__()
        self.lin = nn.Linear(DZ, len(bias))
        nn.init.zeros_(self.lin.weight)
        with torch.no_grad():
            self.lin.bias.copy_(torch.tensor(bias))

    def forward(self, z):                                                          # z (DZ, h, w) -> (nf, N)
        return self.lin(z.flatten(1).t()).t()


class Directed(nn.Module):
    """Gerichteter Term (W (u10, v10) + b + a_s g) je Pixel, (2, N) in (Zeile, Spalte)."""

    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.tensor([[0.0, -0.3], [0.3, 0.0]]))           # Ost -> Spalte +, Nord -> Zeile -
        self.b = nn.Parameter(torch.zeros(2))
        self.a_s = nn.Parameter(torch.zeros(()))

    def forward(self, ev):
        return self.W @ ev["wind"].flatten(1) + self.b[:, None] + self.a_s * ev["grad"].flatten(1)


class CAKernel(nn.Module):
    """q_io (25, N) und gamma_i (N,) des CA-Terms."""

    def __init__(self, coop: bool, q0: float = 0.05):
        super().__init__()
        self.coop = coop
        self.f = Fields([math.log(q0), 0.0])
        self.c_r = nn.Parameter(torch.tensor(-0.7))
        self.dir = Directed()

    def forward(self, ev, geo):
        _, r, u = geo
        fl = self.f(ev["z"])
        w = self.dir(ev)                                                           # (2, N)
        lq = (fl[0][None] + self.c_r * (r - 1).clamp_min(0)[:, None] + u @ w).clamp(max=3.0)   # (25, N)
        q = lq.exp() * (r > 0).float()[:, None]
        gam = fl[1].exp() if self.coop else torch.ones_like(fl[1])
        return q, gam.clamp(0.2, 5.0)

    @staticmethod
    def pressure(q, B, shape):
        """B (M, N) Intensitaet der Quellen -> Druck (M, N) = sum_o q_io B_{i+o}."""
        h, w = shape
        U = F.unfold(B.view(-1, 1, h, w), 2 * R_CA + 1, padding=R_CA)              # (M, 25, N)
        return (q[None] * U).sum(1)


def coop_rate(P, gam, eps: float = 1e-3):
    return (P + eps).pow(gam) - eps ** gam


class CA(nn.Module):
    kind = "ca"

    def __init__(self, coop: bool, age: bool, size: bool = False, substeps: int = 3):
        super().__init__()
        self.M, self.use_age, self.use_size = substeps, age, size
        self.delta = nn.Parameter(torch.zeros(()))
        self.ker = CAKernel(coop)
        self.f = Fields([math.log(1e-4), math.log(0.6 / 0.4)])                     # Zuendung a, Fortbestehen rho
        self.beta = nn.Parameter(torch.zeros(()))
        self.logI = nn.Parameter(torch.zeros(AGE + 1))

    def intensity(self, age):
        """age (.., N) long, -1 = brennt nicht -> I (.., N)."""
        I = self.logI.exp() if self.use_age else torch.ones_like(self.logI)
        return torch.where(age >= 0, I[age.clamp(0, AGE)], torch.zeros((), device=age.device))

    def press(self, q, B, shape):
        """Druck des CA-Kerns, mit Frontgroesse verstaerkt."""
        P = CAKernel.pressure(q, B, shape)
        if not self.use_size:
            return P
        h, w = shape
        mass = F.avg_pool2d(B.view(-1, 1, h, w), 2 * SIZE_R + 1, 1, SIZE_R, count_include_pad=True) * (2 * SIZE_R + 1) ** 2
        return P * torch.exp(self.delta * torch.log1p(mass.flatten(1)))

    def params(self, ev, geo):
        q, gam = self.ker(ev, geo)
        fl = self.f(ev["z"])
        return q, gam, fl[0].exp(), fl[1]

    def day_prob(self, q, gam, a, B0, fresh, shape):
        """Mean-Field ueber die Teilschritte eines Tages: B0 (Bt, N) Intensitaet, fresh (Bt, N) -> P(Zuendung)."""
        Fp = torch.zeros_like(B0)
        I0 = self.intensity(torch.zeros(1, dtype=torch.long, device=B0.device))
        for _ in range(self.M):
            P = self.press(q, B0 + Fp * I0, shape)
            h = -torch.expm1(-coop_rate(P, gam) - a / self.M)
            Fp = Fp + (1 - Fp) * h * fresh
        return Fp

    def loss(self, ev, geo):
        """Teacher Forcing ueber alle 30 Folgetage (Quelle Tag L .. L+29) + Fortbestehen brennender Zellen."""
        q, gam, a, rho = self.params(ev, geo)
        shape = ev["land"].shape
        fire, age = ev["fire"].flatten(1), ev["age"].flatten(1).long()
        d = torch.arange(T_WIN - 1, T_WIN - 1 + T, device=fire.device)
        burnt = fire.cumsum(0)[d] > 0                                              # gebrannt bis einschliesslich Tag d
        fresh = (ev["land"].flatten()[None] & ~burnt).float()
        B0 = self.intensity(age[d])
        Fp = self.day_prob(q, gam, a, B0, fresh, shape)
        y = fire[d + 1].float()
        w = fresh
        l_tf = F.binary_cross_entropy(Fp.clamp(1e-6, 1 - 1e-6), y, weight=w, reduction="sum") / w.sum().clamp_min(1)
        on = fire[d]
        pc = torch.sigmoid(rho[None] + self.beta * torch.log1p(age[d].clamp_min(0).float()))
        l_c = F.binary_cross_entropy(pc[on].clamp(1e-6, 1 - 1e-6), y[on]) if on.any() else pc.sum() * 0
        return l_tf + 0.2 * l_c, {"tf": l_tf.item(), "cont": l_c.item()}

    @torch.no_grad()
    def predict(self, ev, geo, paths: int, gen):
        """-> P(Ankunft <= t) (len(HOR), h, w), Ankunftstag je Pfad (paths, h, w) (31 = keiner)."""
        q, gam, a, rho = self.params(ev, geo)
        shape = ev["land"].shape
        fire, agev = ev["fire"].flatten(1), ev["age"].flatten(1).long()
        L = T_WIN - 1
        burning = (fire[L] | fire[L - 1])
        age0 = torch.where(fire[L], agev[L], torch.where(fire[L - 1], agev[L - 1] + 1, -1))
        age = age0[None].repeat(paths, 1)
        burnt = (fire[:T_WIN].any(0) & ~burning)[None].repeat(paths, 1)
        land = ev["land"].flatten()[None]
        arrival = torch.full_like(age, 31)
        for s in range(T):
            for _ in range(self.M):
                P = self.press(q, self.intensity(age), shape)
                h = -torch.expm1(-coop_rate(P, gam) - a / self.M)
                fresh = land & ~burnt & (age < 0)
                new = fresh & (torch.rand(age.shape, generator=gen, device=age.device) < h)
                age = torch.where(new, torch.zeros_like(age), age)
                arrival = torch.where(new & (arrival > 30), torch.full_like(arrival, s + 1), arrival)
            on = age >= 0
            pc = torch.sigmoid(rho[None] + self.beta * torch.log1p(age.clamp_min(0).float()))
            stop = on & (torch.rand(age.shape, generator=gen, device=age.device) >= pc)
            burnt = burnt | stop
            age = torch.where(stop, torch.full_like(age, -1), torch.where(on, age + 1, age))
        arr = arrival.view(paths, *shape)
        return torch.stack([(arr <= t).float().mean(0) for t in HOR]), arr


def front_motion(new, prev, shape, R: int = SIZE_R):
    """Lokale Frontbewegung (Bt, 2, N) in (Zeile, Spalte): Schwerpunkt der neuen Feuerpixel minus Schwerpunkt des
    Feuers am Vortag, je Pixel im (2R+1)^2-Fenster; 0 wo eines davon fehlt; gestaucht auf |m| / (1 + |m|)."""
    h, w = shape
    rr = torch.arange(h, device=new.device, dtype=torch.float32)[:, None].expand(h, w)
    cc = torch.arange(w, device=new.device, dtype=torch.float32)[None].expand(h, w)
    pool = lambda X: F.avg_pool2d(X.view(-1, 1, h, w), 2 * R + 1, 1, R, count_include_pad=True)   # noqa: E731
    def cent(X):
        X = X.view(-1, 1, h, w)
        return pool(X), pool(X * rr), pool(X * cc)
    Sn, rn, cn = cent(new)
    Sp, rp, cp = cent(prev)
    ok = (Sn > 1e-6) & (Sp > 1e-6)
    m = torch.cat([rn / Sn.clamp_min(1e-6) - rp / Sp.clamp_min(1e-6), cn / Sn.clamp_min(1e-6) - cp / Sp.clamp_min(1e-6)], 1)
    m = torch.where(ok, m, torch.zeros_like(m)).flatten(2)                           # (Bt, 2, N)
    return m / (1 + m.norm(dim=1, keepdim=True))


class CADir(CA):
    """Vortest fuer das Tag-fuer-Tag-Modell: CA (wie ca_lin), dessen Kern sich jeden Tag nach der Frontbewegung des
    Vortags richtet: q_io(t) = q_io * exp(b_dir (u_o . m_i(t)) + b_mag |m_i(t)|), m = front_motion(neu an t, Feuer an t-1).
    Training: beobachtete Bewegung (Teacher Forcing). Vorhersage: Bewegung des eigenen gezogenen Pfads."""

    def __init__(self):
        super().__init__(coop=False, age=False, size=False)
        self.b_dir = nn.Parameter(torch.zeros(()))
        self.b_mag = nn.Parameter(torch.zeros(()))
        self._u = None

    def q_day(self, q, m, geo):
        u = geo[2]                                                                 # (25, 2) Richtung j -> i
        ex = (self.b_dir * torch.einsum("ok,bkn->bon", u, m) + self.b_mag * m.norm(dim=1, keepdim=True)).clamp(max=3.0)
        return q[None] * ex.exp()                                                  # (Bt, 25, N)

    def press(self, q, B, shape):
        h, w = shape
        U = F.unfold(B.view(-1, 1, h, w), 2 * R_CA + 1, padding=R_CA)
        return ((q if q.dim() == 3 else q[None]) * U).sum(1)

    def loss(self, ev, geo):
        q, gam, a, rho = self.params(ev, geo)
        shape = ev["land"].shape
        fire = ev["fire"].flatten(1)
        d = torch.arange(T_WIN - 1, T_WIN - 1 + T, device=fire.device)
        cum = fire.cumsum(0)
        burnt = cum[d] > 0
        new = (fire[d] & ~(cum[d - 1] > 0)).float()
        m = front_motion(new, fire[d - 1].float(), shape)
        fresh = (ev["land"].flatten()[None] & ~burnt).float()
        Fp = self.day_prob(self.q_day(q, m, geo), gam, a, fire[d].float(), fresh, shape)
        y = fire[d + 1].float()
        l_tf = F.binary_cross_entropy(Fp.clamp(1e-6, 1 - 1e-6), y, weight=fresh, reduction="sum") / fresh.sum().clamp_min(1)
        on = fire[d]
        age = ev["age"].flatten(1).long()[d]
        pc = torch.sigmoid(rho[None] + self.beta * torch.log1p(age.clamp_min(0).float()))
        l_c = F.binary_cross_entropy(pc[on].clamp(1e-6, 1 - 1e-6), y[on]) if on.any() else pc.sum() * 0
        return l_tf + 0.2 * l_c, {"tf": l_tf.item(), "cont": l_c.item()}

    @torch.no_grad()
    def predict(self, ev, geo, paths: int, gen):
        q, gam, a, rho = self.params(ev, geo)
        shape = ev["land"].shape
        fire, agev = ev["fire"].flatten(1), ev["age"].flatten(1).long()
        L = T_WIN - 1
        burning = (fire[L] | fire[L - 1])
        age0 = torch.where(fire[L], agev[L], torch.where(fire[L - 1], agev[L - 1] + 1, -1))
        age = age0[None].repeat(paths, 1)
        burnt = (fire[:T_WIN].any(0) & ~burning)[None].repeat(paths, 1)
        land = ev["land"].flatten()[None]
        arrival = torch.full_like(age, 31)
        new_last = (fire[L] & ~(fire[:L].any(0)))[None].repeat(paths, 1).float()
        prev = fire[L - 1][None].repeat(paths, 1).float()
        for s in range(T):
            qd = self.q_day(q, front_motion(new_last, prev, shape), geo)
            start = (age >= 0).float()
            new_day = torch.zeros_like(new_last, dtype=torch.bool)
            for _ in range(self.M):
                P = self.press(qd, self.intensity(age), shape)
                h = -torch.expm1(-coop_rate(P, gam) - a / self.M)
                fresh = land & ~burnt & (age < 0)
                new = fresh & (torch.rand(age.shape, generator=gen, device=age.device) < h)
                age = torch.where(new, torch.zeros_like(age), age)
                new_day |= new
                arrival = torch.where(new & (arrival > 30), torch.full_like(arrival, s + 1), arrival)
            new_last, prev = new_day.float(), start
            on = age >= 0
            pc = torch.sigmoid(rho[None] + self.beta * torch.log1p(age.clamp_min(0).float()))
            stop = on & (torch.rand(age.shape, generator=gen, device=age.device) >= pc)
            burnt = burnt | stop
            age = torch.where(stop, torch.full_like(age, -1), torch.where(on, age + 1, age))
        arr = arrival.view(paths, *shape)
        return torch.stack([(arr <= t).float().mean(0) for t in HOR]), arr


class CAWx(CA):
    """ca_lin mit Tageswetter: Grundstaerke q0, Zuendung a, Fortbestehen rho und Wind kommen jeden Tag aus dem Wetter
    des vorhergesagten Tages (Oracle: beobachtetes Wetter der Folgetage, obere Schranke). Statik wie ca_lin.
    known > 0: in der Vorhersage ist das Wetter nur fuer die Tage L+1 .. L+known bekannt (wie eine Vorhersage mit
    begrenzter Reichweite), danach bleibt es auf Tag L+known; das Training nutzt immer das Tageswetter."""

    def __init__(self, known: int = 0):
        super().__init__(coop=False, age=False, size=False)
        self.known = int(known)

    def params_day(self, ev, geo, days):
        """days (Bt,) Index in fire -> q (Bt, 25, N), a, rho (Bt, N)."""
        _, r, u = geo
        wd = ev["wxd"][days].float()                                               # (Bt, 14, h, w)
        Bt = wd.shape[0]
        z = ev["z"][None].repeat(Bt, 1, 1, 1)
        z[:, :N_WXZ] = wd[:, :N_WXZ]
        z[:, len(ZCH)] = wd[:, N_WXZ]
        flat = z.flatten(2).transpose(1, 2)                                        # (Bt, N, DZ)
        fk = self.ker.f.lin(flat).transpose(1, 2)
        fl = self.f.lin(flat).transpose(1, 2)
        d = self.ker.dir
        w = (torch.einsum("ij,bjn->bin", d.W, wd[:, N_WXZ + 1:].flatten(2)) + d.b[None, :, None]
             + d.a_s * ev["grad"].flatten(1)[None])
        lq = (fk[:, 0:1] + self.ker.c_r * (r - 1).clamp_min(0)[None, :, None] + torch.einsum("ok,bkn->bon", u, w)).clamp(max=3.0)
        return lq.exp() * (r > 0).float()[None, :, None], fl[:, 0].exp(), fl[:, 1]

    def press(self, q, B, shape):
        h, w = shape
        U = F.unfold(B.view(-1, 1, h, w), 2 * R_CA + 1, padding=R_CA)
        return ((q if q.dim() == 3 else q[None]) * U).sum(1)

    def loss(self, ev, geo):
        shape = ev["land"].shape
        fire = ev["fire"].flatten(1)
        d = torch.arange(T_WIN - 1, T_WIN - 1 + T, device=fire.device)
        q, a, rho = self.params_day(ev, geo, d + 1)                                # Wetter des Zieltags
        burnt = fire.cumsum(0)[d] > 0
        fresh = (ev["land"].flatten()[None] & ~burnt).float()
        Fp = self.day_prob(q, torch.ones_like(a), a, fire[d].float(), fresh, shape)
        y = fire[d + 1].float()
        l_tf = F.binary_cross_entropy(Fp.clamp(1e-6, 1 - 1e-6), y, weight=fresh, reduction="sum") / fresh.sum().clamp_min(1)
        on = fire[d]
        age = ev["age"].flatten(1).long()[d]
        pc = torch.sigmoid(rho + self.beta * torch.log1p(age.clamp_min(0).float()))
        l_c = F.binary_cross_entropy(pc[on].clamp(1e-6, 1 - 1e-6), y[on]) if on.any() else pc.sum() * 0
        return l_tf + 0.2 * l_c, {"tf": l_tf.item(), "cont": l_c.item()}

    @torch.no_grad()
    def predict(self, ev, geo, paths: int, gen):
        shape = ev["land"].shape
        fire, agev = ev["fire"].flatten(1), ev["age"].flatten(1).long()
        L = T_WIN - 1
        burning = (fire[L] | fire[L - 1])
        age0 = torch.where(fire[L], agev[L], torch.where(fire[L - 1], agev[L - 1] + 1, -1))
        age = age0[None].repeat(paths, 1)
        burnt = (fire[:T_WIN].any(0) & ~burning)[None].repeat(paths, 1)
        land = ev["land"].flatten()[None]
        arrival = torch.full_like(age, 31)
        days = torch.arange(L + 1, L + 1 + T, device=fire.device)
        if self.known:
            days = days.clamp(max=L + self.known)
        q_all, a_all, rho_all = self.params_day(ev, geo, days)
        for s in range(T):
            q, a, rho = q_all[s:s + 1], a_all[s:s + 1], rho_all[s:s + 1]
            for _ in range(self.M):
                P = self.press(q, self.intensity(age), shape)
                h = -torch.expm1(-P - a / self.M)
                fresh = land & ~burnt & (age < 0)
                new = fresh & (torch.rand(age.shape, generator=gen, device=age.device) < h)
                age = torch.where(new, torch.zeros_like(age), age)
                arrival = torch.where(new & (arrival > 30), torch.full_like(arrival, s + 1), arrival)
            on = age >= 0
            pc = torch.sigmoid(rho + self.beta * torch.log1p(age.clamp_min(0).float()))
            stop = on & (torch.rand(age.shape, generator=gen, device=age.device) >= pc)
            burnt = burnt | stop
            age = torch.where(stop, torch.full_like(age, -1), torch.where(on, age + 1, age))
        arr = arrival.view(paths, *shape)
        return torch.stack([(arr <= t).float().mean(0) for t in HOR]), arr


class RD(nn.Module):
    kind = "rd"

    def __init__(self, ca_term: bool, substeps: int = 12):
        super().__init__()
        self.M, self.dt = substeps, 1.0 / substeps
        self.kmax, self.vmax = 0.2 * substeps, 0.5 * substeps                      # Stabilitaet: kappa dt <= 0.2, |v| dt <= 0.5
        # kappa (logit), log A, theta, log c, log a (Hintergrund-Zuendung je Tag)
        self.f = Fields([math.log(0.15 / 0.85), math.log(3.0 / 17.0), 1.0, math.log(0.5 / 4.5), math.log(1e-4)])
        self.dir = Directed()
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.2)))
        self.log_eps = nn.Parameter(torch.tensor(0.0))
        self.log_eta = nn.Parameter(torch.tensor(0.0))
        self.log_eta_b = nn.Parameter(torch.tensor(0.0))
        self.log_Tb = nn.Parameter(torch.tensor(math.log(2.0)))
        self.w_init = nn.Parameter(torch.tensor([3.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]))
        self.s_b = nn.Parameter(torch.zeros(()))
        self.ca = CAKernel(True, q0=0.02) if ca_term else None

    def params(self, ev, geo):
        fl = self.f(ev["z"])
        # A <= A_MAX, c <= C_MAX (c dt < 1): sonst explodiert T bei einzelnen Ereignissen (NaN, Job 6065_5)
        p = {"kap": self.kmax * torch.sigmoid(fl[0]), "A": A_MAX * torch.sigmoid(fl[1]), "theta": fl[2],
             "c": C_MAX * torch.sigmoid(fl[3]),
             "a": fl[4].exp(), "v": self.vmax * torch.tanh(self.dir(ev) / self.vmax)}
        if self.ca is not None:
            p["q"], p["gam"] = self.ca(ev, geo)
        return p

    def init_state(self, ev, d):
        """Zustand am Ende von Tag d (Index in fire, Tensor (Bt,)) -> T0, S0 (Bt, N)."""
        fire = ev["fire"].flatten(1).float()
        wa = torch.sigmoid(self.w_init)
        T0 = sum(wa[k] * fire[(d - k).clamp_min(0)] for k in range(len(wa))) * self.log_Tb.exp()
        burnt = ev["fire"].flatten(1).cumsum(0)[d] > 0
        land = ev["land"].flatten()[None].float()
        S0 = land * torch.where(burnt, torch.sigmoid(self.s_b) * fire[d], torch.ones_like(T0))
        return T0, S0

    def run(self, p, T0, S0, shape, days: int):
        """-> int R dt je Tag (days, Bt, N)."""
        h, w = shape
        Tt, S = T0.view(-1, h, w), S0.view(-1, h, w)
        kap, A, th, c = (p[k].view(-1, h, w) for k in ("kap", "A", "theta", "c"))   # (1 oder E, h, w)
        v = p["v"] if p["v"].dim() == 3 else p["v"][None]                             # (1 oder E, 2, N)
        vr, vc = v[:, 0].view(-1, h, w), v[:, 1].view(-1, h, w)
        tau, eps = self.log_tau.exp(), self.log_eps.exp()
        out = []
        for _ in range(days):
            acc = torch.zeros_like(Tt)
            for _ in range(self.M):
                Tp = F.pad(Tt[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
                up, dn, lf, rt = Tp[:, :-2, 1:-1], Tp[:, 2:, 1:-1], Tp[:, 1:-1, :-2], Tp[:, 1:-1, 2:]
                lap = up + dn + lf + rt - 4 * Tt
                adv = (torch.where(vr > 0, vr * (Tt - up), vr * (dn - Tt))
                       + torch.where(vc > 0, vc * (Tt - lf), vc * (rt - Tt)))
                burn = S * torch.sigmoid((Tt - th) / tau)
                R = A * burn
                dT = kap * lap - adv + R - c * Tt
                if self.ca is not None:
                    P = CAKernel.pressure(p["q"], burn.flatten(1), shape).view_as(Tt)
                    dT = dT + coop_rate(P, p["gam"].view(1, h, w))
                Tt = (Tt + self.dt * dT).clamp(0.0, T_MAX)
                S = (S - self.dt * eps * R).clamp_min(0)
                acc = acc + self.dt * R
            out.append(acc.flatten(1))
        return torch.stack(out)

    def loss(self, ev, geo, w_tf: float = 1.0, w_b: float = 0.5):
        p = self.params(ev, geo)
        shape = ev["land"].shape
        fire = ev["fire"].flatten(1)
        L = T_WIN - 1
        eta, eta_b = self.log_eta.exp(), self.log_eta_b.exp()
        # freier Lauf ab L: Survival-NLL der ersten Ankunft (exakt fuer die deterministische Intensitaet)
        T0, S0 = self.init_state(ev, torch.tensor([L], device=fire.device))
        Rd = self.run(p, T0, S0, shape, T)[:, 0]                                   # (30, N)
        lam = eta * Rd + p["a"][None]                                              # Tages-Hazard-Rate
        Lc = F.pad(lam.cumsum(0), (0, 0, 1, 0))                                    # (31, N)
        m = ev["m"].flatten()
        first = ev["first"].flatten().long()
        ev_ = (first <= T) & m
        t = first.clamp(1, T)
        ll_ev = -Lc.gather(0, (t - 1)[None])[0] + torch.log((-torch.expm1(-lam.gather(0, (t - 1)[None])[0])).clamp_min(1e-12))
        nll = -(torch.where(ev_, ll_ev, -Lc[T]))[m].mean()
        # Brennzustand je Tag gegen MODIS (Landpixel)
        land = ev["land"].flatten()
        pb = -torch.expm1(-eta_b * Rd)
        l_b = F.binary_cross_entropy(pb[:, land].clamp(1e-6, 1 - 1e-6), fire[L + 1:L + 1 + T][:, land].float())
        # Teacher Forcing: beobachteter Zustand Tag d -> Zuendung Tag d + 1
        d = torch.arange(L, L + T, device=fire.device)
        T0, S0 = self.init_state(ev, d)
        Rtf = self.run(p, T0, S0, shape, 1)[0]                                     # (30, N)
        pf = -torch.expm1(-(eta * Rtf + p["a"][None]))
        fresh = (land[None] & ~(fire.cumsum(0)[d] > 0)).float()
        l_tf = F.binary_cross_entropy(pf.clamp(1e-6, 1 - 1e-6), fire[d + 1].float(), weight=fresh, reduction="sum") / fresh.sum().clamp_min(1)
        return nll + w_tf * l_tf + w_b * l_b, {"nll": nll.item(), "tf": l_tf.item(), "burn": l_b.item()}

    @torch.no_grad()
    def predict(self, ev, geo, paths: int, gen):
        p = self.params(ev, geo)
        shape = ev["land"].shape
        T0, S0 = self.init_state(ev, torch.tensor([T_WIN - 1], device=ev["fire"].device))
        Rd = self.run(p, T0, S0, shape, T)[:, 0]
        Lc = (self.log_eta.exp() * Rd + p["a"][None]).cumsum(0)
        return torch.stack([-torch.expm1(-Lc[t - 1]).view(shape) for t in HOR]), None


class RDEns(RD):
    """Stochastisches RD: E Mitglieder mit gestoerter Windrichtung (Drehung um phi), Windstaerke, Reaktionsrate A und
    Schwelle theta, Streuungen gelernt (Reparametrisierung, eps ~ N(0, 1) je Schritt neu).

    loss "mix"      Mischung auf Ereignisebene: -log (1/E sum_e exp(ll_e)), ll_e = Survival-Loglikelihood aller Pixel
                    + Brennzustand (Bernoulli) des ganzen freien Laufs von Mitglied e. Ein Mitglied ist ein Szenario;
                    scharf sein darf es, wenn eines der Szenarien die Front trifft (statt im Mittel nirgends).
    loss "mix_fss"  dazu Fractions Skill Score je Mitglied (gemittelt) auf P(Ankunft <= 7/15/30) in 5 x 5 und 11 x 11.
    loss "fss"      Survival-Likelihood aufgebrochen: ein Mitglied ohne Stoerung, nur FSS + Brennzustand + Teacher Forcing.
    FSS: 1 - FSS = sum (Pf - Of)^2 / (sum Pf^2 + sum Of^2) mit Pf, Of = Mittel im Fenster (Versatz unter der Fenstergroesse
    kostet kaum; die Pixel-NLL bestraft ihn doppelt, double penalty).
    Vorhersage: min(paths, 32) Mitglieder, P = Mittel; Pfad = Karte eines Mitglieds (P_e >= 0,5)."""

    def __init__(self, loss: str = "mix", ens: int = 8, **kw):
        super().__init__(ca_term=False, **kw)
        self.loss_kind, self.E = loss, (1 if loss == "fss" else ens)
        self.log_sig = nn.Parameter(torch.log(torch.tensor([0.5, 0.3, 0.3, 0.3])))   # phi (rad), log |v|, logit A, theta

    def members(self, ev, geo, eps):
        """eps (E, 4) -> Parameter je Mitglied: A, theta (E, N), v (E, 2, N); Rest wie params."""
        p = self.params(ev, geo)
        if eps is None:
            return p
        fl = self.f(ev["z"])
        sig = self.log_sig.exp()
        phi, sv = sig[0] * eps[:, 0], torch.exp(sig[1] * eps[:, 1])
        raw = self.dir(ev)                                                         # (2, N)
        cs, sn = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
        rot = torch.stack([cs * raw[0] - sn * raw[1], sn * raw[0] + cs * raw[1]], 1) * sv[:, None, None]
        p["v"] = self.vmax * torch.tanh(rot / self.vmax)
        p["A"] = A_MAX * torch.sigmoid(fl[1][None] + sig[2] * eps[:, 2:3])
        p["theta"] = fl[2][None] + sig[3] * eps[:, 3:4]
        return p

    @staticmethod
    def fss_loss(P, O, m, shape, wins=(5, 11)):
        """P (E, N) Wahrscheinlichkeit, O (N,) beobachtet, m (N,) Maske -> mittleres 1 - FSS."""
        h, w = shape
        Pm, Om = (P * m).view(-1, 1, h, w), (O.float() * m).view(1, 1, h, w)
        out = []
        for k in wins:
            Pf = F.avg_pool2d(Pm, k, 1, k // 2, count_include_pad=False)
            Of = F.avg_pool2d(Om, k, 1, k // 2, count_include_pad=False)
            out.append(((Pf - Of) ** 2).sum((1, 2, 3)) / ((Pf ** 2).sum((1, 2, 3)) + (Of ** 2).sum((1, 2, 3))).clamp_min(1e-6))
        return torch.stack(out).mean()

    def free_run(self, ev, geo, eps):
        p = self.members(ev, geo, eps)
        shape = ev["land"].shape
        T0, S0 = self.init_state(ev, torch.tensor([T_WIN - 1], device=ev["fire"].device))
        E = 1 if eps is None else eps.shape[0]
        Rd = self.run(p, T0.expand(E, -1), S0.expand(E, -1), shape, T)              # (30, E, N)
        return p, Rd

    def loss(self, ev, geo, w_tf: float = 1.0, w_b: float = 0.5, w_fss: float = 1.0):
        dev = ev["fire"].device
        eps = None if self.E == 1 else torch.randn(self.E, 4, device=dev)
        p, Rd = self.free_run(ev, geo, eps)
        shape = ev["land"].shape
        fire = ev["fire"].flatten(1)
        L = T_WIN - 1
        eta, eta_b = self.log_eta.exp(), self.log_eta_b.exp()
        lam = eta * Rd + p["a"][None, None]                                        # (30, E, N)
        Lc = F.pad(lam.cumsum(0), (0, 0, 0, 0, 1, 0))                              # (31, E, N)
        m = ev["m"].flatten()
        first = ev["first"].flatten().long()
        ev_ = (first <= T) & m
        t = first.clamp(1, T)
        idx = (t - 1)[None, None].expand(1, Lc.shape[1], -1)
        ll_ev = -Lc.gather(0, idx)[0] + torch.log((-torch.expm1(-lam.gather(0, idx)[0])).clamp_min(1e-12))
        ll_px = torch.where(ev_[None], ll_ev, -Lc[T])                             # (E, N)
        land = ev["land"].flatten()
        pb = (-torch.expm1(-eta_b * Rd)).clamp(1e-6, 1 - 1e-6)                    # (30, E, N)
        yb = fire[L + 1:L + 1 + T].float()[:, None]
        llb = (yb * pb.log() + (1 - yb) * (1 - pb).log())[:, :, land].sum((0, 2))   # (E,)
        n_m, n_b = m.sum().clamp_min(1), (land.sum() * T).clamp_min(1)
        parts = {}
        loss = 0.0
        if self.loss_kind in ("mix", "mix_fss"):
            ll_e = ll_px[:, m].sum(1) + w_b * llb * n_m / n_b                     # Brennzustand auf die Pixelzahl von m skaliert
            nll = -(torch.logsumexp(ll_e, 0) - math.log(self.E)) / n_m
            loss = loss + nll
            parts["nll_mix"] = nll.item()
        else:
            l_b = -llb.mean() / n_b
            loss = loss + w_b * l_b
            parts["burn"] = l_b.item()
        if self.loss_kind in ("mix_fss", "fss"):
            mf = m.float()
            fss = sum(self.fss_loss(-torch.expm1(-Lc[tt]), first <= tt, mf, shape) for tt in (7, 15, 30)) / 3
            loss = loss + w_fss * fss
            parts["fss"] = fss.item()
        # Teacher Forcing ueber einen Tag mit den ungestoerten Parametern (wie RD)
        p0 = self.params(ev, geo)
        d = torch.arange(L, L + T, device=dev)
        T0, S0 = self.init_state(ev, d)
        Rtf = self.run(p0, T0, S0, shape, 1)[0]
        pf = -torch.expm1(-(eta * Rtf + p0["a"][None]))
        fresh = (land[None] & ~(fire.cumsum(0)[d] > 0)).float()
        l_tf = F.binary_cross_entropy(pf.clamp(1e-6, 1 - 1e-6), fire[d + 1].float(), weight=fresh, reduction="sum") / fresh.sum().clamp_min(1)
        parts["tf"] = l_tf.item()
        return loss + w_tf * l_tf, parts

    @torch.no_grad()
    def predict(self, ev, geo, paths: int, gen):
        E = 1 if self.E == 1 else min(paths, 32)
        eps = None if E == 1 else torch.randn(E, 4, generator=gen, device=ev["fire"].device)
        p, Rd = self.free_run(ev, geo, eps)
        shape = ev["land"].shape
        Pt = -torch.expm1(-(self.log_eta.exp() * Rd + p["a"][None, None]).cumsum(0))   # (30, E, N)
        P = torch.stack([Pt[t - 1].mean(0).view(shape) for t in HOR])
        hit = Pt >= 0.5
        arr = torch.where(hit.any(0), hit.float().argmax(0) + 1, torch.full_like(hit[0], 31, dtype=torch.long))
        return P, arr.view(E, *shape)


def build(name: str) -> nn.Module:
    kind, a, b, c = MODELS[name]
    if kind == "rdens":
        net = RDEns(loss=a)
    elif kind == "cadir":
        net = CADir()
    elif kind == "cawx":
        net = CAWx(known=a or 0)
    else:
        net = CA(coop=a, age=b, size=c) if kind == "ca" else RD(ca_term=a)
    if name.endswith("_classic"):                 # klassisch kalibriert: keine Felder aus Kovariaten, nur Konstanten
        for m in net.modules():
            if isinstance(m, Fields):
                nn.init.zeros_(m.lin.weight)
                m.lin.weight.requires_grad_(False)
    return net


def summary(net) -> dict:
    """Gelernte globale Groessen fuer den Bericht."""
    out = {}
    ker = getattr(net, "ker", None) or getattr(net, "ca", None)
    if ker is not None:
        out.update({"ca_c_r": float(ker.c_r), "ca_wind_W": ker.dir.W.tolist(), "ca_a_slope": float(ker.dir.a_s),
                    "ca_gamma_bias": float(ker.f.lin.bias[1].exp()) if ker.coop else 1.0})
    if isinstance(net, CADir):
        out.update({"b_dir": float(net.b_dir), "b_mag": float(net.b_mag)})
    if isinstance(net, CA):
        out.update({"I_age": net.logI.exp().tolist() if net.use_age else None, "beta_cont": float(net.beta),
                    "delta_size": float(net.delta) if net.use_size else None})
    if isinstance(net, RDEns):
        out.update({"sig_phi_wind_dir": float(net.log_sig[0].exp()), "sig_log_speed": float(net.log_sig[1].exp()),
                    "sig_logit_A": float(net.log_sig[2].exp()), "sig_theta": float(net.log_sig[3].exp())})
    if isinstance(net, RD):
        out.update({"tau": float(net.log_tau.exp()), "eps": float(net.log_eps.exp()), "eta": float(net.log_eta.exp()),
                    "eta_b": float(net.log_eta_b.exp()), "T_b": float(net.log_Tb.exp()),
                    "w_init": torch.sigmoid(net.w_init).tolist(), "s_b": float(torch.sigmoid(net.s_b)),
                    "rd_wind_W": net.dir.W.tolist(), "rd_a_slope": float(net.dir.a_s),
                    "bias_kap": float(net.kmax * torch.sigmoid(net.f.lin.bias[0])), "bias_A": float(A_MAX * torch.sigmoid(net.f.lin.bias[1])),
                    "bias_theta": float(net.f.lin.bias[2]), "bias_c": float(C_MAX * torch.sigmoid(net.f.lin.bias[3]))})
    return out


# ----------------------------------------------------------------------------------------------- Ablauf
def load_split(exp, cube, events, mean, std, masks, device, wxd: bool = False):
    t0 = time.time()
    evs = [to_dev({**load_event(cube, e, mean, std, masks), **({"wxd": load_wxd(cube, e, mean, std)} if wxd else {})}, device)
           for e in events]
    print(f"[sym] {len(evs)} Ereignisse geladen ({time.time() - t0:.0f} s)", flush=True)
    return evs


def evaluate(net, evs, geo, paths, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    res = []
    for i, ev in enumerate(evs):
        P, arr = net.predict(ev, geo, paths, gen)
        P = P.float().cpu().numpy()
        m, first = ev["m"].cpu().numpy(), ev["first"].cpu().numpy()
        st = {}
        for j, t in enumerate(HOR):
            pt = (arr.cpu().numpy() <= t) if arr is not None else None
            st[str(t)] = map_metrics(P[j], first <= t, m, pt)
        res.append({"i": i, "date": ev["date"], "box": ev["box"], "n_obs": {t: int(((first <= t) & m).sum()) for t in HOR},
                    "scales": {"1.0": st}})
    return res


def fit(exp, cfg, cube, out, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(args.anomaly)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    masks = (np.load(exp / "derived_data" / "masks" / "land.npy"), np.load(exp / "derived_data" / "masks" / "canada.npy"))
    ev = {sp: json.loads((out / "events_train.json").read_text()) if sp == "train" else
          json.loads((exp / "metrics" / f"firemap_bench{args.bench_tag}" / f"events_{sp}.json").read_text())
          for sp in ("train", "val", "test")}
    if args.limit:
        ev = {sp: v[:args.limit] for sp, v in ev.items()}
    data = {sp: load_split(exp, cube, ev[sp], mean, std, masks, device, wxd=MODELS[args.model][0] == "cawx") for sp in ev}
    geo = ca_geometry(device)
    net = build(args.model).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best, best_state, hist, skipped = float("inf"), None, [], 0
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        net.train()
        tl, t0 = [], time.time()
        for i in rng.permutation(len(data["train"])):
            loss, parts = net.loss(data["train"][i], geo)
            opt.zero_grad()
            ok = bool(torch.isfinite(loss))
            if ok:
                loss.backward()
                ok = all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None)
            if not ok:                                   # Schritt verwerfen statt NaN in die Parameter zu schreiben
                skipped += 1
                print(f"[sym] Schritt verworfen (Ereignis {i}, Verlust {float(loss):.4g}, {parts})", flush=True)
                if skipped > 0.05 * (ep + 1) * len(data["train"]):
                    raise SystemExit(f"ABBRUCH: {skipped} verworfene Schritte bis Epoche {ep}")
                continue
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tl.append(float(loss))
        with torch.no_grad():
            vl = float(np.mean([float(net.loss(e, geo)[0]) for e in data["val"]]))
        hist.append({"epoch": ep, "train": float(np.mean(tl)), "val": vl})
        print(f"[sym] {args.model} Epoche {ep}: train {np.mean(tl):.4f} val {vl:.4f} ({time.time() - t0:.0f} s) {parts}", flush=True)
        if vl < best or best_state is None:
            best, best_state = vl, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()
    (out / "ckpt").mkdir(exist_ok=True)
    torch.save({"state": best_state, "model": args.model, "args": vars(args)}, out / "ckpt" / f"{args.model}.pt")
    (out / "runs").mkdir(exist_ok=True)
    for sp in ("val", "test"):
        res = evaluate(net, data[sp], geo, args.paths, args.seed, device)
        (out / "runs" / f"{sp}_{args.model}.json").write_text(json.dumps(res))
        a = agg(res, "1.0", "30")
        print(f"[sym] {args.model} {sp} 30 d: AP {a['ap']:.3f} IoU {a['iou']:.3f} IoU top-n {a['iou_topk']:.3f} "
              f"Praez. {a['prec']:.3f} Recall {a['rec']:.3f} log Flaeche {a['med_log_area']:+.2f}", flush=True)
    (out / "params").mkdir(exist_ok=True)
    (out / "params" / f"{args.model}.json").write_text(json.dumps({"summary": summary(net), "history": hist,
                                                                    "best_val": best}, indent=1))
    print(json.dumps(summary(net), indent=1), flush=True)


BENCH = ("dist", "cg_ell11", "cg_tfonly", "cg_tfonlybig", "cg_sfiou20", "cg_scratch", "lh_indep")


VIEW_BENCH = (("cg_ell11", "1.0"), ("cg_tfonly", "1.0"), ("cg_sfiou20", "1.0"), ("dist", "2.0"))
VIEW_SYM = ("ca_classic", "ca_lin", "ca_coop", "ca_age", "ca_size", "ca_dir", "ca_full", "rd_classic", "rd", "rdca",
            "rd_ens_classic", "rd_ens", "rd_ens_fss", "rd_fss")
VIEW_DAYCA = (("dayca", "fire_dayca"), ("dayca_ema", "fire_dayca_ema"), ("dayca_emass", "fire_dayca_emass"))


def maps_viewer(exp, cfg, cube, out, args):
    """Beobachtete Masken und vorhergesagte Karten P(Ankunft <= 7 / 30) aller Laeufe fuer die groessten test-Ereignisse.
    -> metrics/symbolic_spread<tag>/maps_viewer/data_<report-name>.json (PNG je Karte, base64). Beobachtet, je Pixel:
    0 Land ohne Feuer, 1 kein Land / ausserhalb Kanada, 2 Feuer im Fenster vor L, 3 Feuer der letzten 3 Tage bis L,
    4 neu an Tag 1-7, 5 neu an Tag 8-30. Symbolische Modelle neu vorhergesagt aus den Checkpoints
    (CA aus symbolic_spread_lr, falls vorhanden), Bench-Modelle aus firemap_bench/maps (Skala 1, dist r = 2)."""
    import base64
    import io
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    land, canada = np.load(exp / "derived_data" / "masks" / "land.npy"), np.load(exp / "derived_data" / "masks" / "canada.npy")
    bench = exp / "metrics" / f"firemap_bench{args.bench_tag}"
    events = json.loads((bench / "events_test.json").read_text())
    sel = sorted(sorted(range(len(events)), key=lambda i: -events[i]["n_new_30"])[:args.viewer_events])
    data = {i: to_dev(load_event(cube, events[i], mean, std, (land, canada)), device) for i in sel}
    geo = ca_geometry(device)
    png = lambda a: base64.b64encode((lambda b: (Image.fromarray(a).save(b, "PNG", optimize=True), b.getvalue())[1])(io.BytesIO())).decode()  # noqa: E731
    models, runs = [], {}
    for nm in VIEW_SYM:
        ck = next((d / "ckpt" / f"{nm}.pt" for d in (exp / "metrics" / "symbolic_spread_lr", out)
                   if (d / "ckpt" / f"{nm}.pt").exists()), None)
        if ck is None:
            continue
        net = build(nm).to(device)
        net.load_state_dict(torch.load(ck, map_location=device)["state"])
        net.eval()
        gen = torch.Generator(device=device).manual_seed(args.seed)
        maps = {}
        for i in sel:
            P, _ = net.predict(data[i], geo, args.paths, gen)
            maps[i] = P.float().cpu().numpy()
        models.append((nm, "symbolisch", maps))
        rf = ck.parent.parent / "runs" / f"test_{nm}.json"
        runs[nm] = json.loads(rf.read_text()) if rf.exists() else None
        print(f"[sym] Karten {nm} ({ck})", flush=True)
    for nm, sub in VIEW_DAYCA:
        f = exp / "metrics" / sub / "maps" / "test_dayca.npz"
        if not f.exists():
            continue
        z = np.load(f)
        models.append((nm, "Tag für Tag", {i: np.stack([z[f"{i}_1.0_{t}"].astype(np.float32) for t in HOR]) for i in sel}))
        rf = exp / "metrics" / sub / "runs" / "test_dayca.json"
        runs[nm] = json.loads(rf.read_text()) if rf.exists() else None
    for nm, sc in VIEW_BENCH:
        f = bench / "maps" / f"test_{nm}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        models.append((nm, "Bench", {i: np.stack([z[f"{i}_{sc}_{t}"].astype(np.float32) for t in HOR]) for i in sel}))
        rf = bench / "runs" / f"test_{nm}.json"
        runs[nm] = json.loads(rf.read_text()) if rf.exists() else None
    res = {"horizons": list(HOR), "models": [{"name": nm, "kind": k} for nm, k, _ in models], "events": []}
    for i in sel:
        ev = data[i]
        fire = ev["fire"].cpu().numpy()
        r0, r1, c0, c1 = ev["box"]
        obs = np.zeros(fire.shape[1:], np.uint8)
        obs[~(land & canada)[r0:r1, c0:c1]] = 1
        obs[fire[:T_WIN].any(0)] = 2
        obs[fire[T_WIN - 3:T_WIN].any(0)] = 3
        first, m = ev["first"].cpu().numpy(), ev["m"].cpu().numpy()
        obs[m & (first <= 7)] = 4
        obs[m & (first > 7) & (first <= 30)] = 5
        e = {"i": i, "date": events[i]["date"], "box": events[i]["box"], "h": int(obs.shape[0]), "w": int(obs.shape[1]),
             "n_obs": {str(t): int((m & (first <= t)).sum()) for t in HOR}, "obs": png(obs), "pred": {}, "metrics": {}}
        for nm, _, maps in models:
            e["pred"][nm] = {str(t): png(np.clip(maps[i][j] * 255 + 0.5, 0, 255).astype(np.uint8)) for j, t in enumerate(HOR)}
            r = runs.get(nm)
            if r:
                st = r[i]["scales"]
                sc = "1.0" if "1.0" in st else dict(VIEW_BENCH).get(nm, next(iter(st)))
                e["metrics"][nm] = {str(t): {k: st[sc][str(t)].get(k) for k in ("ap", "iou_topk", "log_area")} for t in HOR}
        res["events"].append(e)
    od = out / "maps_viewer"
    od.mkdir(exist_ok=True)
    f = od / f"data_{args.report_name}.json"
    if f.exists():
        raise SystemExit(f"ABBRUCH: {f} existiert")
    f.write_text(json.dumps(res))
    print(f"[sym] Viewer: {len(sel)} Ereignisse, {len(models)} Modelle -> {f}", flush=True)


def report(exp, out, args):
    bench = exp / "metrics" / f"firemap_bench{args.bench_tag}" / "runs"
    rows = {t: [] for t in HOR}
    names = [(n, bench) for n in BENCH if (bench / f"test_{n}.json").exists()]
    names += [(n, out / "runs") for n in MODELS if (out / "runs" / f"test_{n}.json").exists()]
    for nm, d in names:
        rt = json.loads((d / f"test_{nm}.json").read_text())
        rv = json.loads((d / f"val_{nm}.json").read_text())
        scales = list(rt[0]["scales"])
        for t in HOR:
            best = max(scales, key=lambda s: agg(rv, s, str(t))["iou"])
            for s in dict.fromkeys((["1.0"] if "1.0" in scales else []) + [best]):
                rows[t].append((nm, "Bench" if d == bench else "symbolisch", s, agg(rt, s, str(t))))
    md = ["# Symbolische Ausbreitung (CA / Reaktion-Diffusion) gegen Contagion, firemap_bench-Ereignisse (test)", "",
          "Angepasst auf Ereignissen der Trainingsjahre, Auswahl der Epoche nach val-Verlust. Bench-Modelle: Skala s = 1 und "
          "die auf val gewaehlte Skala (dist: Radius). CA: 200 MC-Pfade; RD: deterministische Karte.", ""]
    f = lambda x: "–" if x != x else f"{x:.3f}"                                         # noqa: E731
    for t in HOR:
        md += [f"## {t} Tage", "", "| Modell | Art | Skala | AP | AUC | IoU | IoU top-n | IoU Pfad | Praez. | Recall | "
               "Median log Flaeche | ASSD | Brier |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for nm, kind, s, a in rows[t]:
            md.append(f"| {nm} | {kind} | {s} | {f(a['ap'])} | {f(a['auc'])} | {f(a['iou'])} | {f(a['iou_topk'])} | "
                      f"{f(a['iou_path'])} | {f(a['prec'])} | {f(a['rec'])} | {a['med_log_area']:+.2f} | {a['assd']:.1f} | {f(a['brier'])} |")
        md.append("")
    md += ["## Gelernte Parameter", ""]
    for n in MODELS:
        pf = out / "params" / f"{n}.json"
        if pf.exists():
            md.append(f"- **{n}**: `{json.dumps(json.loads(pf.read_text())['summary'])}`")
    (out / f"{args.report_name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["events", "fit", "report", "maps"])
    ap.add_argument("--viewer-events", type=int, default=24)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--bench-tag", default="")
    ap.add_argument("--model", default="rd", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--paths", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--step", dest="step_days", type=int, default=3)
    ap.add_argument("--min-act", type=int, default=15)
    ap.add_argument("--min-new", type=int, default=50)
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--max-box", type=int, default=112)
    ap.add_argument("--max-events", type=int, default=120)
    ap.add_argument("--report-name", default="summary")
    ap.add_argument("--anomaly", action="store_true", help="NaN-Suche (torch.autograd.detect_anomaly)")
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten n Ereignisse je Split (Smoke-Test)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"symbolic_spread{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    cube = CubeReader(exp)
    if args.stage == "maps":
        maps_viewer(exp, cfg, cube, out, args)
    elif args.stage == "events":
        f = out / "events_train.json"
        if f.exists():
            raise SystemExit(f"ABBRUCH: {f} existiert")
        ev = select_events(exp, cube, split_years(exp)["train"], args)
        f.write_text(json.dumps(ev, indent=1))
        print(f"[sym] train: {len(ev)} Ereignisse, neue Pixel 30 d Median {np.median([e['n_new_30'] for e in ev]):.0f}", flush=True)
    elif args.stage == "fit":
        if (out / "runs" / f"test_{args.model}.json").exists():
            raise SystemExit(f"ABBRUCH: Lauf {args.model} existiert")
        fit(exp, cfg, cube, out, args)
    else:
        if (out / f"{args.report_name}.md").exists():
            raise SystemExit(f"ABBRUCH: {args.report_name}.md existiert")
        report(exp, out, args)
    write_manifest(cfg, f"p71_symbolic_spread_{args.stage}_{args.model if args.stage == 'fit' else ''}{args.tag}",
                   args.config, {"args": vars(args), "out": str(out)})


if __name__ == "__main__":
    main()

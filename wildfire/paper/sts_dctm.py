"""STS-DCTM: raeumlich-zeitlicher Transformer mit DCTM-Survival-Koepfen.

Ein Modell, das fuer jedes Vorlauf-Intervall tau in [0, 100] Tagen das Risiko
liefert, dass es brennt: fuer jedes Pixel des 13x13-Ausschnitts und fuer den
Ausschnitt als Ganzes.

  Eingabe   30 Tage x 55 Kanaele x 13 x 13 bis zum Stichtag e, dazu der
            Kalendertag von e (Saison, als sin/cos)
  Stem      je Tag zwei 3x3-Faltungen (lokale Nachbarschaft) -> ein Token je
            Pixel und Tag: 30 x 169 Token der Breite d
  Rumpf     Bloecke mit faktorisierter Attention: ueber den Raum (die 169
            Pixel eines Tages sehen sich gegenseitig), dann ueber die Zeit
            (jedes Pixel sieht seine 30 Tage), dann MLP
  Auslese   je Pixel letzter Tag || Mittel ueber die Zeit
  Koepfe    DCTM pro Pixel und DCTM fuer den Ausschnitt (Attention-Pooling
            ueber die Pixel); Transformationsfunktion in Bernstein-Basis vom
            Grad 12 auf [0, 100], monoton per Konstruktion, S(0) = 1
  Verlust   exakte intervallzensierte Likelihood: Ereignis an Tag k oder
            zensiert bei c, je Pixel und fuer den Ausschnitt (erstes Feuer
            irgendwo), Summe mit Gewicht --patch-weight

Abfrage: F(tau) = 1 - S(tau) fuer beliebiges tau; die Mittelzelle ist das
Pixel (6, 6) des Pixelkopfs.

Training wie pixel_stdctm: --train-source cube (Stichtage aus unserem Wuerfel,
pixel_landmarks, 2000-2019, Ereignisse bis 100 Tage) zum Vortraining, dann
--train-source cache mit --init-from (Stichtage im Paper-Cache, Ereignisse bis
zum Zieltag, optional --extra). Auswertung im Protokoll des Papers: Stichtag
e = Zieltag - dt - 1, Score F(dt + 1) - fuer die Mittelzelle, alle Pixel, den
Ausschnittskopf und das Maximum ueber die Pixel; Ziel D oder D+1 (--target).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.models.ema import EMA  # noqa: F401  (auch fuer Tests und Aufrufer hier importierbar)
from wildfire.models.surv_heads import HEADS, BinaryHorizonHead, CoxHead, head_loss, interval_nll, make_head
from wildfire.models.xcal import xcal_penalty
from wildfire.paper import extra, targets
from wildfire.paper.channels import CHANNELS, masked_std
from wildfire.paper.channels import tag as channel_tag
from wildfire.paper.compare import DTS, KEYS
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics
from wildfire.paper.future_weather import WEATHER_CH
from wildfire.paper.pixel_stdctm import pixel_targets, windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm, gather_windows, to_device_once

T_WIN = 30
P = 13
CENTRE = 6
TAU_MAX = 100.0
DEGREE = 12


# ------------------------------------------------------------------ Modell
class Block(nn.Module):
    """Pre-Norm-Block, Attention nach attn, dann MLP.

    factorized  Raum (die 169 Pixel eines Tages), dann Zeit (die 30 Tage eines Pixels)
    joint       eine Attention ueber alle 30 x 169 Token (Raum und Zeit zugleich)
    local       Raum wie factorized; im Zeit-Schritt sieht jedes Pixel die 30 Tage seiner
                3 x 3-Nachbarschaft (270 Schluessel), also das Feuer des Nachbarn von vor 5
                Tagen direkt statt ueber ein Zwischen-Token
    """

    def __init__(self, d: int, heads: int, p: float, attn: str = "factorized"):
        super().__init__()
        if attn not in ("factorized", "joint", "local"):
            raise ValueError(f"attn {attn!r}")
        self.attn = attn
        self.ns, self.nt, self.nf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        if attn != "joint":
            self.ta = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))

    def forward(self, x):                                   # (B, T, N, d)
        B, T, N, d = x.shape
        if self.attn == "joint":
            s = x.reshape(B, T * N, d)
            h = self.ns(s)
            x = (s + self.sa(h, h, h, need_weights=False)[0]).view(B, T, N, d)
            return x + self.ff(self.nf(x))
        s = x.reshape(B * T, N, d)
        h = self.ns(s)
        s = s + self.sa(h, h, h, need_weights=False)[0]
        t = s.reshape(B, T, N, d).transpose(1, 2).reshape(B * N, T, d)
        h = self.nt(t)
        if self.attn == "local":
            kv, pad = self._neighbour_keys(h.view(B, N, T, d), N)
            t = t + self.ta(h, kv, kv, key_padding_mask=pad, need_weights=False)[0]
        else:
            t = t + self.ta(h, h, h, need_weights=False)[0]
        x = t.reshape(B, N, T, d).transpose(1, 2)
        return x + self.ff(self.nf(x))

    @staticmethod
    def _neighbour_keys(h, N):
        """h (B, N, T, d) -> Schluessel (B*N, T*9, d) der 3 x 3-Nachbarn ueber alle Tage, Maske fuer den Rand."""
        B, _, T, d = h.shape
        g = int(round(math.sqrt(N)))
        z = h.permute(0, 2, 3, 1).reshape(B * T, d, g, g)
        u = F.unfold(z, 3, padding=1).view(B, T, d, 9, N)                  # Nachbar j fuer Pixel n
        kv = u.permute(0, 4, 1, 3, 2).reshape(B * N, T * 9, d)
        ones = torch.ones(1, 1, g, g, device=h.device, dtype=h.dtype)
        valid = F.unfold(ones, 3, padding=1).view(9, N).t() > 0.5           # (N, 9)
        pad = (~valid)[:, None, :].expand(N, T, 9).reshape(1, N, T * 9).expand(B, N, T * 9).reshape(B * N, T * 9)
        return kv, pad


class RateHead(nn.Module):
    """Rate wiederkehrender Ereignisse: P(Feuer an Tag tau), tau = 1..H, fuer Ausschnitt und Mitte.

    Die Survival-Kurve beantwortet "wann brennt es zum ersten Mal", das Ziel des Paper-Protokolls
    ist "brennt es an Tag D + dt" - bei dt = 30 nicht dasselbe (ein Feuer an Tag 5, das an Tag 20
    erloschen ist, zaehlt fuer F(31), nicht fuer die Rate an Tag 31). Logit als Bernstein-Polynom
    in tau (glatt, nicht monoton), Koeffizienten aus einem eigenen Attention-Pooling ueber die
    Pixel (Ausschnitt) bzw. aus dem Token der Mittelzelle."""

    def __init__(self, d: int, heads: int, horizon: int = int(TAU_MAX), degree: int = DEGREE,
                 init_patch: float = 0.02, init_centre: float = 0.002):
        super().__init__()
        self.horizon = int(horizon)
        u = torch.arange(1, self.horizon + 1, dtype=torch.float64) / self.horizon
        B = torch.stack([math.comb(degree, k) * u ** k * (1 - u) ** (degree - k) for k in range(degree + 1)], 1)
        self.register_buffer("basis", B.float(), persistent=False)          # (H, M+1), Zeilensumme 1
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.pa, self.px = nn.Linear(d, degree + 1), nn.Linear(d, degree + 1)
        for lin, p0 in ((self.pa, init_patch), (self.px, init_centre)):   # konstante Koeffizienten = konstanter Logit
            nn.init.zeros_(lin.weight); nn.init.constant_(lin.bias, math.log(p0 / (1 - p0)))

    def forward(self, pix):
        """Pixel-Token (B, 169, d) -> Logits (B, 2, H): [Ausschnitt, Mitte]."""
        a = self.pool(self.cls.expand(pix.shape[0], -1, -1), pix, pix, need_weights=False)[0][:, 0]
        coef = torch.stack([self.pa(self.norm(a)), self.px(pix[:, CENTRE * P + CENTRE])], 1).float()
        return coef @ self.basis.t()


class FutureWeather(nn.Module):
    """Wetter-Oracle: beobachtetes Wetter der Tage L + 1 .. L + k (k <= F) -> Zusatz zu den Pixel-Token.

    Je Tag ein 3 x 3-Stem ueber die Wetterkanaele, Tagesposition, dann je Pixel Mittel und Maximum
    ueber die k vorhandenen Tage. Der Ausgang startet bei null: ohne Folgetage (k = 0) und zu
    Beginn des Trainings ist das Netz genau das Modell ohne Oracle."""

    def __init__(self, c_in: int, d: int, days: int):
        super().__init__()
        self.days = int(days)
        self.stem = nn.Sequential(nn.Conv2d(c_in, d, 3, padding=1), nn.GroupNorm(8, d), nn.SiLU())
        self.pos = nn.Parameter(torch.zeros(1, self.days, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU())
        self.out = nn.Linear(2 * d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, xf, kf):
        """xf (B, F, C_w, 13, 13) normiert, kf (B,) Zahl der gueltigen Tage -> (B, 169, d)."""
        B, Fd = xf.shape[:2]
        z = self.stem(xf.flatten(0, 1)).flatten(2).transpose(1, 2).reshape(B, Fd, -1, self.pos.shape[-1])
        z = self.mlp(z + self.pos[:, :Fd])
        m = (torch.arange(Fd, device=xf.device)[None] < kf[:, None])[:, :, None, None]   # (B, F, 1, 1)
        n = m.sum(1).clamp(min=1)
        mean = (z * m).sum(1) / n
        mx = z.masked_fill(~m, -1e4).max(1).values * (n > 0)
        has = (kf > 0)[:, None, None].to(z.dtype)
        return self.out(torch.cat([mean, mx * has], -1)) * has


class STSDCTM(nn.Module):
    def __init__(self, c_in: int = 55, d: int = 96, heads: int = 4, depth: int = 4, p: float = 0.1,
                 init_cif_px: float = 0.02, init_cif_patch: float = 0.2, head: str = "dctm",
                 attn: str = "factorized", stem: str = "2d", hist: int = T_WIN, rate: bool = False,
                 init_rate: tuple = (0.02, 0.002), bce_taus=None, future: int = 0, c_future: int = 25):
        super().__init__()
        if not 1 <= hist <= T_WIN:
            raise ValueError(f"hist {hist} nicht in 1..{T_WIN}")
        self.head_kind = head
        self.stem_kind = stem
        self.hist = int(hist)                                 # nur die letzten hist Fenstertage gehen ein
        conv = {"2d": nn.Conv2d, "3d": nn.Conv3d}[stem]       # 3d: 3 x 3 Pixel x 3 Tage
        self.stem = nn.Sequential(
            conv(c_in, d, 3, padding=1), nn.GroupNorm(8, d), nn.SiLU(),
            conv(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.SiLU())
        self.pos_s = nn.Parameter(torch.zeros(1, 1, P * P, d))
        self.pos_t = nn.Parameter(torch.zeros(1, T_WIN, 1, d))
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        self.season = nn.Linear(2, d)
        self.blocks = nn.ModuleList([Block(d, heads, p, attn) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.read = nn.Linear(2 * d, d)
        self.px_head = make_head(head, TAU_MAX, DEGREE, init_cif_px, bce_taus)
        self.px_lin = nn.Linear(d, self.px_head.n_out)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(d)
        self.pa_head = make_head(head, TAU_MAX, DEGREE, init_cif_patch, bce_taus)
        self.pa_lin = nn.Linear(d, self.pa_head.n_out)
        for lin in (self.px_lin, self.pa_lin):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.rate_head = RateHead(d, heads, init_patch=init_rate[0], init_centre=init_rate[1]) if rate else None
        self.future = FutureWeather(c_future, d, future) if future else None

    def forward(self, x, doy, rate: bool = False, xf=None, kf=None):
        """x (B, 30, C, 13, 13) normiert, doy (B,) Kalendertag des Stichtags
        -> roh je Pixel (B, 13, 13, M+1), roh Ausschnitt (B, M+1); mit rate dazu die
        Rate-Logits (B, 2, H) des Rate-Kopfs. Wetter-Oracle: xf (B, F, 25, 13, 13) normiert,
        kf (B,) Zahl der bekannten Folgetage; ohne xf wie ohne Oracle."""
        B, H, W = x.shape[0], x.shape[3], x.shape[4]
        pix = self.encode(x, doy)
        if self.future is not None and xf is not None:
            pix = pix + self.future(xf, kf)
        raw_px = self.px_lin(pix).view(B, H, W, -1)
        a = self.pool(self.cls.expand(B, -1, -1), pix, pix, need_weights=False)[0][:, 0]
        raw_pa = self.pa_lin(self.pool_norm(a))
        if rate:
            return raw_px.float(), raw_pa.float(), self.rate_head(pix)
        return raw_px.float(), raw_pa.float()

    def encode(self, x, doy):
        """Rumpf: (B, 30, C, 13, 13), (B,) -> Pixel-Token (B, 169, d); nur die letzten self.hist Tage."""
        x = x[:, -self.hist:]
        B, T, C, H, W = x.shape
        if self.stem_kind == "3d":
            z = self.stem(x.transpose(1, 2))                 # (B, d, T, 13, 13)
            z = z.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, -1)
        else:
            z = self.stem(x.flatten(0, 1))                   # (B*T, d, 13, 13)
            z = z.flatten(2).transpose(1, 2).reshape(B, T, H * W, -1)
        ang = 2 * math.pi * doy.float() / 365.25
        z = z + self.pos_s + self.pos_t[:, -T:] + self.season(torch.stack([ang.sin(), ang.cos()], -1))[:, None, None]
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        return self.read(torch.cat([z[:, -1], z.mean(1)], -1))  # (B, 169, d)


# ------------------------------------------------------------------ Hilfen
def build_sts(head: str = "dctm", **kw) -> STSDCTM:
    """STS-Netz mit dem gewaehlten Kopf; "contagion" ist ein eigenes Modell (paper/contagion.py)."""
    if head == "contagion":
        from wildfire.paper.contagion import ContagionSTS
        return ContagionSTS(**kw)
    return STSDCTM(head=head, **kw)


def load_sts(path, device="cpu") -> STSDCTM:
    """Checkpoint -> Netz im eval-Modus; Breite, Tiefe und Kopf aus den gespeicherten Argumenten."""
    c = torch.load(path, map_location="cpu")
    a = c.get("args", {})
    head = a.get("head", "dctm")
    cg = ({"kernel": a.get("cg_kernel", 5), "obs_days": a.get("cg_obs_days", 7), "burn": a.get("cg_burn", "geometric"),
           "blind": a.get("cg_blind", "none")}
          if head == "contagion" else {})
    net = build_sts(head, d=a.get("d", 96), heads=a.get("heads", 4), depth=a.get("depth", 4),
                    attn=a.get("attn", "factorized"), stem=a.get("stem", "2d"), hist=a.get("hist_days", T_WIN),
                    rate=a.get("rate_weight", 0.0) > 0, bce_taus=a.get("bce_taus"), **cg,
                    **({"future": a["future_days"]} if a.get("future_days") else {}))
    net.load_state_dict(c["state"])
    return net.to(device).eval()


def doy_of(dates: np.ndarray) -> np.ndarray:
    """JJJJMMTT -> Kalendertag (float)."""
    return np.array([dt.date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100).timetuple().tm_yday
                     for d in dates], np.float32)


def patch_events(ev: np.ndarray, t: np.ndarray):
    """Pixel-Ereignisse (n, 13, 13) -> erstes Feuer irgendwo im Ausschnitt (n,), Zeit (n,).
    Alle Pixel eines Stichtags haben dieselbe Zensurzeit, das Minimum ist also das
    erste Ereignis bzw. die Zensurzeit."""
    return ev.reshape(len(ev), -1).any(1), t.reshape(len(t), -1).min(1)


def summarize(per_seed: list[dict]) -> dict:
    return {k: {"mean": float(np.mean([m[k] for m in per_seed])), "std": float(np.std([m[k] for m in per_seed]))}
            for k in KEYS}


def spatial_holdout(row: np.ndarray, col: np.ndarray, share: float, block: int = 64, buffer: int = 13,
                    seed: int = 12345):
    """Raeumliche Bloecke (block x block Pixel) zufaellig mit Anteil ``share`` herausgehalten.

    -> hold: Stichtag liegt in einem herausgehaltenen Block; buf: nicht herausgehalten, aber
    ein herausgehaltener Block liegt naeher als ``buffer`` Pixel (Ausschnitte koennten sich
    ueberlappen). Gilt block > buffer, decken die 9 Punkte (row +- buffer, col +- buffer)
    alle Bloecke in Chebyshev-Abstand <= buffer ab."""
    rng = np.random.default_rng(seed)
    nb = int(max(row.max(), col.max()) + buffer) // block + 2
    held = rng.random((nb, nb)) < share
    blk = lambda r, c: held[np.clip(r, 0, None) // block, np.clip(c, 0, None) // block]
    hold = blk(row, col)
    near = np.zeros_like(hold)
    for dr in (-buffer, 0, buffer):
        for dc in (-buffer, 0, buffer):
            near |= blk(row + dr, col + dc)
    return hold, near & ~hold


def lr_factor(step: int, warmup: int, total: int, kind: str = "const", min_frac: float = 0.0) -> float:
    """Faktor auf die Lernrate: linearer Warmup, danach konstant oder Cosine bis min_frac am letzten Schritt."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if kind == "const":
        return 1.0
    p = min(1.0, (step - warmup) / max(total - warmup, 1))
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * p))


# ------------------------------------------------------------------ Hauptteil
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap.add_argument("--extra", choices=extra.MODES, default="none")
    ap.add_argument("--channels", choices=list(CHANNELS), default="paper55")
    ap.add_argument("--train-source", choices=["cube", "cache"], default="cache")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--init-from", default="", help="Checkpoint; {seed} wird ersetzt")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0 = ein Durchlauf ueber die Trainingsdaten")
    ap.add_argument("--landmarks-per-epoch", type=int, default=4, help="cache: Stichtage je Stichprobe und Epoche")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--sched", choices=["const", "cosine"], default="const",
                    help="Lernrate nach dem Warmup: konstant oder Cosine-Abklingen bis --min-lr-frac ueber alle Schritte")
    ap.add_argument("--min-lr-frac", type=float, default=0.0, help="cosine: Endwert als Anteil von --lr")
    ap.add_argument("--xcal", type=float, default=0.0,
                    help="Gewicht des X-CAL-Terms (differenzierbare D-Calibration, models/xcal.py); 0 = aus")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="Abklingfaktor fuer den EMA der Gewichte (z. B. 0.999); 0 = aus. Mit EMA laufen "
                         "Auswahl, Checkpoint und Test mit den gemittelten Gewichten")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patch-weight", type=float, default=1.0)
    ap.add_argument("--head", choices=HEADS + ("contagion",), default="dctm",
                    help="Survival-Kopf fuer Pixel und Ausschnitt (models/surv_heads.py); dctm = Bernstein")
    ap.add_argument("--attn", choices=["factorized", "joint", "local"], default="factorized",
                    help="Attention im Rumpf: Raum dann Zeit, voll ueber alle Token, oder Zeit ueber 3 x 3-Nachbarn")
    ap.add_argument("--stem", choices=["2d", "3d"], default="2d", help="Stem: Faltung je Tag oder 3 x 3 x 3 ueber Raum und Zeit")
    ap.add_argument("--cg-kernel", type=int, default=5, help="contagion: Kantenlaenge des Ausbreitungskerns")
    ap.add_argument("--cg-obs-days", type=int, default=7, help="contagion: beobachtete Fenstertage fuer brennende Pixel")
    ap.add_argument("--cg-burn", choices=["geometric", "curve"], default="geometric",
                    help="contagion: Brenndauer geometrisch (rho^Alter) oder als gelernte Kurve")
    ap.add_argument("--cg-blind", choices=["none", "thermal", "modis"], default="none",
                    help="contagion: Feuerkanaele fuer den Rumpf ausblenden (Feuer wirkt nur ueber b_j und den Kern)")
    ap.add_argument("--init-backbone-only", action="store_true",
                    help="--init-from laedt nur passende Gewichte (z. B. Rumpf eines anderen Kopfs)")
    ap.add_argument("--baseline-n", type=int, default=20000,
                    help="cox: so viele Trainings-Stichtage fuer den Breslow-Basis-Hazard")
    ap.add_argument("--limit-eval", type=int, default=0, help="nur fuer Rauchtests: so viele Val/Test-Stichproben")
    ap.add_argument("--cube-kind", choices=["all", "random", "fire"], default="all",
                    help="Wuerfel-Stichtage: alle, nur zufaellige (echte Haeufigkeit) oder nur an Feuer verankerte")
    ap.add_argument("--select", choices=["prauc", "survnll", "last"], default="prauc",
                    help="Modellauswahl: PR-AUC auf den Paper-Val-Stichproben, Survival-NLL auf "
                         "zufaelligen Val-Stichtagen (surv_eval, 2020-2022) oder keine (feste Epochenzahl, "
                         "letzte Gewichte; die Val-NLL wird nur protokolliert)")
    ap.add_argument("--rate-weight", type=float, default=0.0,
                    help="Gewicht des Rate-Kopfs (P(Feuer an Tag tau), BCE ueber tau = 1..100, Ausschnitt und "
                         "Mitte); 0 = kein Rate-Kopf. Braucht --train-source cube und train_rate.npz (rate_labels)")
    ap.add_argument("--hist-days", type=int, default=T_WIN, help="nur die letzten so viele Fenstertage als Eingabe")
    ap.add_argument("--bce-taus", type=int, nargs="+", default=None,
                    help="--head bce: Horizonte der Sigmoid-Ausgaenge (Standard surv_heads.BCE_TAUS)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="cube: nur dieser Anteil der Trainings-Stichtage (feste Auswahl, Seed 123)")
    ap.add_argument("--future-days", type=int, default=0,
                    help="Wetter-Oracle: beobachtetes Wetter der Folgetage (cube, train_Xf.npy aus future_weather); "
                         "im Training zufaellig auf 0..k Tage gekuerzt, im Paper-Protokoll die dt Tage bis D")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--diag-holdout", type=float, default=0.0,
                    help="Overfitting-Diagnose: Anteil raeumlicher Bloecke des Wuerfels, die aus dem Training "
                         "herausgehalten werden (0 = aus). Je Epoche dann Survival-NLL auf Trainings-Stichtagen, "
                         "auf den herausgehaltenen Orten (gleiche Jahre) und auf den Val-Jahren (--surv-dir)")
    ap.add_argument("--diag-block", type=int, default=64, help="Diagnose: Kantenlaenge der Bloecke in Pixeln")
    ap.add_argument("--diag-n", type=int, default=20000, help="Diagnose: hoechstens so viele Stichtage je Menge")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.diag_holdout > 0 and args.train_source != "cube":
        ap.error("--diag-holdout nur mit --train-source cube")
    if (args.rate_weight > 0 or args.train_frac < 1) and args.train_source != "cube":
        ap.error("--rate-weight und --train-frac nur mit --train-source cube")
    if args.future_days and (args.train_source != "cube" or args.head == "contagion"):
        ap.error("--future-days nur mit --train-source cube und einem STS-Kopf (nicht contagion)")
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = (f"p15_sts_dctm_{args.cache}{targets.tag(args.target)}{extra.tag(args.extra)}"
            f"{channel_tag(args.channels)}_{args.train_source}{args.tag}")

    # ---- Paper-Cache (Validierung, Test, ggf. Training)
    t0 = time.time()
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")                                     # (N, 61, 55, 13, 13)
    y = targets.labels(d, args.target)
    split = np.load(d / "split.npy")
    doyD = doy_of(np.load(d / "meta.npy")[:, 0])
    fire_target = targets.fire_map(d, X, args.target)
    if args.extra != "none":
        ex = extra.load(exp, args.target, args.extra)
        X, y, split = extra.append(X, y, split, ex)
        fire_target = np.concatenate([fire_target, ex["fire_map"]])
        doyD = np.concatenate([doyD, doy_of(ex["date"])])
    LC = X.shape[1]                                              # Cache-Tage, Index LC - 1 = D
    L = LC + targets.shift(args.target)                          # effektive Laenge: Zieltag = Index L - 1
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    ok = targets.ok_mask(d)
    if ok is not None:                                           # flaechentreu: nur Fenster ganz im Wuerfel
        if args.extra != "none":
            ap.error("--extra nicht mit einem flaechentreuen Cache")
        idx = {s: v[ok[v]] for s, v in idx.items()}
        print(f"[sts] {args.cache}: nur Stichproben mit allen 61 Tagen im Wuerfel", flush=True)
    if args.limit_eval:
        idx["val"], idx["test"] = idx["val"][:args.limit_eval], idx["test"][:args.limit_eval]
    fire = X[:, :, FIRE_CH] > 0
    if targets.shift(args.target):
        fire = np.concatenate([fire, fire_target[:, None]], 1)   # Zieltag D+1 als Tag L - 1
    Y = fire[:, -1]
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    std = masked_std(std, args.channels)
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    mf_t, sf_t = m_t[:, :, WEATHER_CH], s_t[:, :, WEATHER_CH]          # Wetter-Oracle
    E_LO, E_HI = T_WIN - 1, L - 2
    print(f"[sts] {args.cache} Ziel {args.target}, Zusatz {args.extra}: X {X.shape}, Train {idx['train'].size}, "
          f"Val {idx['val'].size}, Test {idx['test'].size} ({time.time() - t0:.0f} s)", flush=True)

    def to_dev(xw):
        xw = xw if isinstance(xw, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(xw))
        return (xw.to(device).float() - m_t) / s_t

    def to_dev_f(xf):
        xf = xf if isinstance(xf, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(xf))
        return (xf.to(device).float() - mf_t) / sf_t

    # ---- Wuerfel-Stichtage (Vortraining)
    if args.train_source == "cube":
        cdir = exp / "derived_data" / args.cube_dir
        t1 = time.time()
        kind_c = np.load(cdir / "train_meta.npz")["kind"]           # 0 an Feuer verankert, 1 zufaellig
        keep = (np.ones(kind_c.size, bool) if args.cube_kind == "all"
                else kind_c == (1 if args.cube_kind == "random" else 0))
        rows_c = np.where(keep)[0]
        Xc = np.asarray(np.load(cdir / "train_X.npy", mmap_mode="r")[rows_c])   # (n, 30, 55, 13, 13), in den RAM
        evc, tc = np.load(cdir / "train_event.npy")[rows_c], np.load(cdir / "train_t.npy")[rows_c].astype(np.int64)
        dates = CubeReader(exp).dates
        doyc = np.array([dt.date.fromisoformat(dates[int(Lc)]).timetuple().tm_yday
                         for Lc in np.load(cdir / "train_meta.npz")["L"][rows_c]], np.float32)
        init_px, init_pa = float(evc.mean()), float(evc.reshape(len(evc), -1).any(1).mean())
        print(f"[sts] Wuerfel: {Xc.shape[0]:,} Stichtage in {time.time() - t1:.0f} s; Pixel-Ereignisse "
              f"{init_px:.4f}, Ausschnitt {init_pa:.3f}", flush=True)
        tr_c = np.arange(Xc.shape[0])                                # Trainings-Stichtage (Zeilen von Xc)
        if args.train_frac < 1:
            tr_c = np.sort(np.random.default_rng(123).choice(tr_c, int(round(args.train_frac * tr_c.size)), replace=False))
            print(f"[sts] --train-frac {args.train_frac}: {tr_c.size:,} Stichtage", flush=True)
        if args.rate_weight > 0:
            rz = np.load(cdir / "train_rate.npz")
            if not np.array_equal(rz["L"], np.load(cdir / "train_meta.npz")["L"]):
                raise SystemExit("ABBRUCH: train_rate.npz passt nicht zu train_meta.npz")
            rate_y = np.stack([rz["patch"], rz["centre"]], 1)[rows_c]          # (n, 2, 100) bool
            init_rate = tuple(float(np.clip(v, 1e-4, 0.5)) for v in rate_y.mean((0, 2)))
            print(f"[sts] Rate-Labels: Feuer an einem Tag, Ausschnitt {init_rate[0]:.4f}, Mitte {init_rate[1]:.4f}",
                  flush=True)
        if args.future_days:
            Xfc = np.asarray(np.load(cdir / "train_Xf.npy", mmap_mode="r")[rows_c][:, :args.future_days])
            kfc = np.minimum(np.load(cdir / "train_kf.npy")[rows_c].astype(np.int64), args.future_days)
            print(f"[sts] Wetter-Oracle: {Xfc.shape}, volle {args.future_days} Folgetage bei {(kfc == args.future_days).mean():.3f}",
                  flush=True)
        if args.diag_holdout > 0:
            mc = np.load(cdir / "train_meta.npz")
            hold, buf = spatial_holdout(mc["row"][rows_c], mc["col"][rows_c], args.diag_holdout, args.diag_block)
            tr_c = np.where(~hold & ~buf)[0]
            rnd_c = kind_c[rows_c] == 1                              # zufaellige Stichtage: echte Haeufigkeit
            rs_d = np.random.default_rng(1)
            pick = lambda m: np.sort(rs_d.choice(np.where(m)[0], min(args.diag_n, int(m.sum())), replace=False))
            diag_sets = {"train": pick(rnd_c & ~hold & ~buf), "holdout": pick(rnd_c & hold)}
            print(f"[sts] Diagnose: {hold.sum():,} Stichtage an herausgehaltenen Orten, {buf.sum():,} im Puffer "
                  f"verworfen, {tr_c.size:,} im Training; NLL auf je {len(diag_sets['train']):,} / "
                  f"{len(diag_sets['holdout']):,} zufaelligen Stichtagen", flush=True)
    else:
        ev0, _ = pixel_targets(fire[idx["train"]], np.full(idx["train"].size, E_LO))
        init_px, init_pa = float(ev0.mean()), float(ev0.reshape(len(ev0), -1).any(1).mean())

    # ---- Auswertung im Protokoll des Papers
    @torch.no_grad()
    def paper_scores(net, rows, dt_, bs=64, with_rate=False):
        """Stichtag e = Zieltag - dt - 1, Score F(dt + 1): Karte (n, 13, 13) und Ausschnittskopf (n,);
        mit with_rate dazu die Rate an Tag dt + 1 (n, 2): Ausschnitt, Mitte."""
        net.eval()
        e = L - dt_ - 2
        k = torch.tensor([float(dt_ + 1)], device=device)
        maps, heads, rates = [], [], []
        fd = net.future.days if getattr(net, "future", None) is not None else 0
        kk = min(fd, LC - 1 - e)                                    # Oracle: Wetter der Tage e + 1 .. D (dt Tage)
        for a in range(0, rows.size, bs):
            r = rows[a:a + bs]
            fut = {}
            if kk > 0:
                xf = torch.zeros(r.size, fd, len(WEATHER_CH), P, P, device=device)
                xf[:, :kk] = to_dev_f(gather_windows(Xg, r, np.full(r.size, e + 1), kk)[:, :, WEATHER_CH])
                fut = {"xf": xf, "kf": torch.full((r.size,), kk, device=device)}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outs = net(to_dev(windows(Xg, r, np.full(r.size, e))),
                           torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device), rate=with_rate, **fut)
            raw_px, raw_pa = outs[0], outs[1]
            fp = -torch.expm1(net.px_head.log_survival(raw_px.reshape(-1, raw_px.shape[-1]), k))[:, 0]
            fa = -torch.expm1(net.pa_head.log_survival(raw_pa, k))[:, 0]
            maps.append(fp.view(len(r), P, P).cpu().numpy()); heads.append(fa.cpu().numpy())
            if with_rate:
                rates.append(torch.sigmoid(outs[2][:, :, dt_].float()).cpu().numpy())    # Tag dt + 1 = Index dt
        cl = lambda v: np.clip(np.concatenate(v), 1e-7, 1 - 1e-7)  # noqa: E731
        return (cl(maps), cl(heads), cl(rates)) if with_rate else (cl(maps), cl(heads))

    if args.select in ("survnll", "last") or args.diag_holdout > 0:
        sd = exp / "derived_data" / args.surv_dir
        sv = dict(np.load(sd / "val_meta.npz"))
        Xsv = to_device_once(np.load(sd / "val_X.npy"), device, "surv_val_X")
        print(f"[sts] Auswahl ueber Survival-NLL auf {Xsv.shape[0]:,} zufaelligen Val-Stichtagen", flush=True)

    @torch.no_grad()
    def val_surv(net, bs=128) -> float:
        """Minus (NLL Mitte + NLL Ausschnitt) auf zufaelligen Val-Stichtagen, Horizont 100 Tage."""
        net.eval()
        tot, n = 0.0, Xsv.shape[0]
        for a in range(0, n, bs):
            sl = slice(a, a + bs)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw_px, raw_pa = net(to_dev(Xsv[sl]), torch.from_numpy(sv["doy"][sl]).to(device))
            for head, raw, e, t in ((net.px_head, raw_px[:, CENTRE, CENTRE], sv["ev_centre"][sl], sv["t_centre"][sl]),
                                    (net.pa_head, raw_pa, sv["ev_patch"][sl], sv["t_patch"][sl])):
                tot -= float(interval_nll(head, raw, torch.from_numpy(e).to(device),
                                          torch.from_numpy(t.astype(np.int64)).to(device))) * len(e)
        return tot / n

    @torch.no_grad()
    def cube_ll(net, rows, bs=128) -> float:
        """Wie val_surv (Mitte + Ausschnitt, mittlere Log-Likelihood), auf Stichtagen des Wuerfels."""
        net.eval()
        tot = 0.0
        for a in range(0, rows.size, bs):
            r = rows[a:a + bs]
            pe, pt = patch_events(evc[r], tc[r])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw_px, raw_pa = net(to_dev(Xc[r]), torch.from_numpy(doyc[r]).to(device))
            for head, raw, e, t in ((net.px_head, raw_px[:, CENTRE, CENTRE], evc[r][:, CENTRE, CENTRE], tc[r][:, CENTRE, CENTRE]),
                                    (net.pa_head, raw_pa, pe, pt)):
                tot -= float(interval_nll(head, raw, torch.from_numpy(np.asarray(e)).to(device),
                                          torch.from_numpy(np.asarray(t, np.int64)).to(device))) * len(e)
        return tot / rows.size

    def diagnose(net) -> dict:
        """Overfitting-Diagnose: Minus-NLL auf Training, herausgehaltenen Orten und Val-Jahren."""
        return {"diag_train_ll": cube_ll(net, diag_sets["train"]), "diag_holdout_ll": cube_ll(net, diag_sets["holdout"]),
                "diag_valyears_ll": val_surv(net)}

    def val_rank(net) -> float:
        va, sc = idx["val"], []
        for dt_ in DTS:
            m, h = paper_scores(net, va, dt_)
            sc.append(average_precision_score(Y[va, CENTRE, CENTRE], m[:, CENTRE, CENTRE]))
            sc.append(average_precision_score(y[va], h))
        return float(np.mean(sc))

    val_fn = val_surv if args.select in ("survnll", "last") else val_rank

    @torch.no_grad()
    def fit_cox_baseline(net, bs=64):
        """cox: Breslow-Basis-Hazard je Kopf aus einer festen Stichprobe der Trainings-Stichtage."""
        net.eval()
        rs = np.random.default_rng(0)
        if args.train_source == "cube":
            rows = np.sort(rs.choice(Xc.shape[0], min(args.baseline_n, Xc.shape[0]), replace=False))
        else:
            rows = rs.choice(idx["train"], args.baseline_n)
            e_b = rs.integers(E_LO, E_HI + 1, size=rows.size)
        acc = {k: [] for k in ("eta_px", "ev_px", "t_px", "eta_pa", "ev_pa", "t_pa")}
        for a in range(0, rows.size, bs):
            r = rows[a:a + bs]
            if args.train_source == "cube":
                ev, tt, xw, doy = evc[r], tc[r], Xc[r], doyc[r]
            else:
                e = e_b[a:a + bs]
                ev, tt = pixel_targets(fire[r], e)
                xw, doy = windows(Xg, r, e), doyD[r] - (LC - 1 - e)
            pe, pt = patch_events(ev, tt)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw_px, raw_pa = net(to_dev(xw), torch.from_numpy(np.asarray(doy, np.float32)).to(device))
            for k, v in (("eta_px", raw_px), ("eta_pa", raw_pa)):
                acc[k].append(v.reshape(-1).double())
            for k, v in (("ev_px", ev), ("t_px", tt), ("ev_pa", pe), ("t_pa", pt)):
                acc[k].append(torch.from_numpy(np.asarray(v).reshape(-1)).to(device))
        c = {k: torch.cat(v) for k, v in acc.items()}
        net.px_head.fit_baseline(c["eta_px"], c["ev_px"], c["t_px"])
        net.pa_head.fit_baseline(c["eta_pa"], c["ev_pa"], c["t_pa"])
        print(f"[sts] Breslow aus {rows.size:,} Stichtagen: Lambda_0(100) Pixel {float(net.px_head.base[-1]):.4g}, "
              f"Ausschnitt {float(net.pa_head.base[-1]):.4g}", flush=True)

    # ---- Training je Seed
    te = idx["test"]
    res_seed = []
    hists = []
    ckpt_dir = exp / "checkpoints" / "paper"
    for seed in args.seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        cg = ({"kernel": args.cg_kernel, "obs_days": args.cg_obs_days, "burn": args.cg_burn, "blind": args.cg_blind}
              if args.head == "contagion" else {})
        net = build_sts(args.head, d=args.d, heads=args.heads, depth=args.depth, p=args.dropout,
                        attn=args.attn, stem=args.stem, hist=args.hist_days, bce_taus=args.bce_taus,
                        rate=args.rate_weight > 0, init_rate=init_rate if args.rate_weight > 0 else (0.02, 0.002),
                        init_cif_px=float(np.clip(init_px, 1e-4, 0.5)),
                        init_cif_patch=float(np.clip(init_pa, 1e-3, 0.9)), **cg,
                        **({"future": args.future_days} if args.future_days else {})).to(device)
        if hasattr(net, "set_fire_norm"):
            net.set_fire_norm(float(mean[FIRE_CH]), float(std[FIRE_CH]))
        if args.init_from:
            sd0 = torch.load(args.init_from.format(seed=seed), map_location="cpu")["state"]
            if args.init_backbone_only:                     # nur der Rumpf, Koepfe neu
                own = net.state_dict()
                sd0 = {k: v for k, v in sd0.items()
                       if k in own and own[k].shape == v.shape and not k.startswith(("px_", "pa_"))}
                print(f"[sts] Rumpf aus --init-from: {len(sd0)} von {len(own)} Tensoren", flush=True)
            net.load_state_dict(sd0, strict=not args.init_backbone_only)
            print(f"[sts] Start von {args.init_from.format(seed=seed)}", flush=True)
        n_par = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        n_rows = tr_c.size if args.train_source == "cube" else idx["train"].size * args.landmarks_per_epoch
        total = args.epochs * (args.steps_per_epoch or int(np.ceil(n_rows / args.bs)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.warmup, total, args.sched, args.min_lr_frac))
        ema = EMA(net, args.ema) if args.ema > 0 else None
        best, state, bad, hist = -np.inf, None, 0, []
        if args.init_from:
            best = val_fn(net)
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"      Start: val {args.select} {best:.4f}", flush=True)
        print(f"[sts] Seed {seed}: {n_par / 1e6:.2f} Mio. Parameter", flush=True)
        for ep in range(args.epochs):
            net.train()
            t_ep = time.time()
            if args.train_source == "cube":
                rows = rng.permutation(tr_c)
            else:
                rows = rng.permutation(np.repeat(idx["train"], args.landmarks_per_epoch))
                e_all = rng.integers(E_LO, E_HI + 1, size=rows.size)
            if args.steps_per_epoch:
                rows = rows[:args.steps_per_epoch * args.bs]
            tl = tn = 0.0
            for a in range(0, rows.size, args.bs):
                if args.train_source == "cube":
                    r = np.sort(rows[a:a + args.bs])
                    ev, tt, xw, doy = evc[r], tc[r], Xc[r], doyc[r]
                else:
                    r, e = rows[a:a + args.bs], e_all[a:a + args.bs]
                    ev, tt = pixel_targets(fire[r], e)
                    xw, doy = windows(Xg, r, e), doyD[r] - (LC - 1 - e)
                pe, pt = patch_events(ev, tt)
                fut = {}
                if args.future_days:                        # Oracle: zufaellig 0..k bekannte Folgetage
                    fut = {"xf": to_dev_f(Xfc[r]), "kf": torch.from_numpy(rng.integers(0, kfc[r] + 1)).to(device)}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outs = net(to_dev(xw), torch.from_numpy(np.asarray(doy, np.float32)).to(device),
                               rate=args.rate_weight > 0, **fut)
                raw_px, raw_pa = outs[0], outs[1]
                loss = head_loss(net.px_head, raw_px, torch.from_numpy(ev).to(device), torch.from_numpy(tt).to(device)) \
                    + args.patch_weight * head_loss(net.pa_head, raw_pa, torch.from_numpy(pe).to(device),
                                                    torch.from_numpy(pt).to(device))
                if args.xcal > 0:                           # X-CAL: Mitte und Ausschnitt
                    loss = loss + args.xcal * (
                        xcal_penalty(net.px_head, raw_px[:, CENTRE, CENTRE], torch.from_numpy(np.ascontiguousarray(ev[:, CENTRE, CENTRE])).to(device),
                                     torch.from_numpy(np.ascontiguousarray(tt[:, CENTRE, CENTRE])).to(device))
                        + xcal_penalty(net.pa_head, raw_pa, torch.from_numpy(pe).to(device), torch.from_numpy(pt).to(device)))
                if args.rate_weight > 0:                    # Rate: BCE ueber alle Tage, Ausschnitt und Mitte
                    loss = loss + args.rate_weight * F.binary_cross_entropy_with_logits(
                        outs[2].float(), torch.from_numpy(rate_y[r]).to(device).float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step()
                if ema is not None:
                    ema.update(net)
                tl += loss.item() * len(r); tn += len(r)
            live = None
            if ema is not None:                             # Validierung mit den gemittelten Gewichten
                live = {k: v.detach().clone() for k, v in net.state_dict().items()}
                net.load_state_dict(ema.state(live))
            vr = val_fn(net)
            dg = diagnose(net) if args.diag_holdout > 0 else {}
            better = args.select == "last" or vr > best + 1e-5
            cand = {k: v.detach().clone() for k, v in net.state_dict().items()} if better else None
            if live is not None:
                net.load_state_dict(live)
            hist.append({"epoch": ep, "train_nll": tl / tn, "val_prauc_mean": vr, **dg})
            print(f"      Epoche {ep:2d}: train NLL {tl / tn:.5f}  val {args.select} {vr:.4f}"
                  f"{' (EMA)' if ema is not None else ''}  ({time.time() - t_ep:.0f} s)"
                  + "".join(f"  {k[5:]} {v:.5f}" for k, v in dg.items()), flush=True)
            if cand is not None:
                best, bad, state = max(best, vr), 0, cand
            else:
                bad += 1
                if bad >= args.patience:
                    break
        net.load_state_dict(state)
        if isinstance(net.px_head, CoxHead):
            fit_cox_baseline(net)
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        hists.append(hist)
        if args.save_ckpt:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cp = ckpt_dir / (f"sts_{args.cache}{targets.tag(args.target)}{extra.tag(args.extra)}"   # Kanal-Tag wie im Manifest
                             f"{channel_tag(args.channels)}_{args.train_source}{args.tag}_s{seed}.pt")
            torch.save({"state": state, "args": vars(args)}, cp)
            print(f"[sts] Checkpoint {cp}", flush=True)
        out = {}
        has_rate = getattr(net, "rate_head", None) is not None
        for hd in (net.px_head, net.pa_head):              # bce: rohe Ausgaenge an den Horizonten, wie gestapelte Modelle
            if isinstance(hd, BinaryHorizonHead):
                hd.monotone = False
        for dt_ in DTS:
            sc = paper_scores(net, te, dt_, with_rate=has_rate)
            m, h = sc[0], sc[1]
            out[str(dt_)] = {
                "centre": paper_metrics(m[:, CENTRE, CENTRE], Y[te, CENTRE, CENTRE].astype(int)),
                "pixel": paper_metrics(m.reshape(-1), Y[te].reshape(-1).astype(int)),
                "patch_head": paper_metrics(h, y[te]),
                "patch_max": paper_metrics(m.max(axis=(1, 2)), y[te]),
            }
            if has_rate:                                     # Rate an Tag dt + 1 statt F(dt + 1)
                out[str(dt_)]["patch_rate"] = paper_metrics(sc[2][:, 0], y[te])
                out[str(dt_)]["centre_rate"] = paper_metrics(sc[2][:, 1], Y[te, CENTRE, CENTRE].astype(int))
        res_seed.append(out)

    evs = list(res_seed[0][str(DTS[0])])
    res = {str(dt_): {ev: {"summary": summarize([r[str(dt_)][ev] for r in res_seed]),
                           "per_seed": [r[str(dt_)][ev] for r in res_seed]}
                      for ev in evs} for dt_ in DTS}
    for dt_ in DTS:
        print(f"[sts] TEST dt={dt_:2d}  " + "   ".join(
            f"{ev}: PR-AUC {res[str(dt_)][ev]['summary']['pr_auc']['mean']:.3f}"
            f"±{res[str(dt_)][ev]['summary']['pr_auc']['std']:.3f}" for ev in res[str(dt_)]), flush=True)
    write_manifest(cfg, name, args.config, {
        "cache": args.cache, "target": args.target, "extra": args.extra, "channels": args.channels,
        "train_source": args.train_source, "init_from": args.init_from, "seeds": args.seeds,
        "model": {"d": args.d, "depth": args.depth, "heads": args.heads, "dropout": args.dropout,
                  "head": args.head, "dctm": f"Bernstein Grad {DEGREE} auf [0, {TAU_MAX:.0f}]",
                  "patch_weight": args.patch_weight},
        "training": {"bs": args.bs, "lr": args.lr, "wd": args.wd, "warmup": args.warmup, "sched": args.sched, "min_lr_frac": args.min_lr_frac, "epochs": args.epochs,
                     "steps_per_epoch": args.steps_per_epoch, "patience": args.patience, "selection": args.select, "cube_kind": args.cube_kind,
                     "ema": args.ema, "xcal": args.xcal, "rate_weight": args.rate_weight,
                     "hist_days": args.hist_days, "bce_taus": args.bce_taus, "train_frac": args.train_frac,
                     "future_days": args.future_days},
        "diagnosis": ({"holdout_share": args.diag_holdout, "block_px": args.diag_block, "n_per_set": args.diag_n,
                       "sets": "train = zufaellige Trainings-Stichtage, holdout = zufaellige Stichtage an "
                               "herausgehaltenen Bloecken (gleiche Jahre, 13 px Puffer), valyears = --surv-dir val",
                       "metric": "mittlere Log-Likelihood Mitte + Ausschnitt (hoeher = besser)"}
                      if args.diag_holdout > 0 else None),
        "score": "F(dt + 1) bei Stichtag e = Zieltag - dt - 1; *_rate: Rate-Kopf P(Feuer an Tag dt + 1)",
        "metric_assumptions": ASSUMPTIONS, "histories": hists, "results": res,
    })


if __name__ == "__main__":
    main()

"""sts_extreme mit Trainingsvarianten, die das Netz staerker auf Wetter, Brennstoff und Lage lenken.

Gleiches Protokoll und gleicher Code wie sts_extreme (Ablation: Wuerfel-Stichtage, Ziel D+1, d 64, 4 Bloecke,
EMA 0.999, 10 x 3000 Schritte, Contagion ohne Warmstart); sts_extreme.py bleibt unveraendert. Dieses Skript
ersetzt sts_extreme.build (Huelle um forward) und, fuer ignw, sts_extreme.head_loss. --variants, kommagetrennt:
  fdrop  Feuer-Dropout, nur im Training (net.training): je Stichprobe mit --p-drop der ganze Feuerkanal (alle 30
         Tage, alle Pixel) auf "kein Feuer". Die Maske wirkt vor dem ganzen Netz, bei Contagion sieht also auch der
         Ausbreitungspfad (Brennzustand aus dem Feuerkanal) kein Feuer. Auswertung ohne Dropout.
  fdropall wie fdrop, aber gemeinsam mit den feuerempfindlichen Kanaelen (sts_inputs.FIRE_THERMAL: LST_day/night,
         BT_b20/21/31/32; normiert auf 0 = Kanalmittel), sonst weicht das Netz nur auf die Thermalbaender aus.
  resid  Rest ueber einer festen Feuerhistorie-Basis (nur LH): logistische Regression auf 14 Merkmalen aus dem
         Feuerkanal und dem Kalendertag (features_px; wie fire_baseline, aber "Tage seit dem letzten Feuer" richtig
         gezaehlt, 0 = Feuer am letzten Fenstertag, 30 = keins), angepasst nur auf den Wuerfel-Stichtagen fuer "Feuer
         bis 30 Tage" (Pixel bzw. Ausschnitt). Offset o = log(-log(1 - p_basis)) - Mittel_train(...), d. h. log des
         kumulativen Basis-Hazards bei 30 Tagen, zentriert. o wird auf alle Tages-Logits des LH-Kopfs addiert
         (Pixelkopf: Basis je Pixel, Merkmale um dieses Pixel; Ausschnittskopf: Basis des Ausschnitts). Bei kleinen
         Tages-Hazards ist das ein proportionaler Faktor e^o auf das Hazard: das Netz lernt den Faktor relativ zur
         Basis. Im Training und in jeder Auswertung (Huelle bleibt aktiv).
  ignw   Zuendungsgewicht: Stichproben ohne Feuer im Ausschnitt in den 30 Fenstertagen gehen mit --w-ign in die
         Intervall-NLL ein (gewichtetes Mittel, beide Koepfe), sonst 1.
  --pad reflect  zusaetzlich padding_mode 'reflect' in den Stem-Faltungen (wie sts_inputs --pad), Laeufe mit Suffix _rf.
Weitere Argumente gehen an sts_extreme (--head, --mono, --seeds, --tag, ...). Ausgaben wie sts_extreme (Checkpoints,
calib_cache_ext, p35_sts_extreme<tag>), dazu Manifest p44_sts_focus<tag> und fuer resid derived_data/sts_focus/
coef<tag>.npz (Koeffizienten, noetig zum spaeteren Laden mit build_focus).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.audit.common import load_config, write_manifest
from wildfire.models.bernstein import BernsteinSurvivalHead
from wildfire.paper import sts_extreme
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_inputs import FIRE_THERMAL

VARIANTS = ("fdrop", "fdropall", "resid", "ignw")
_ORIG_BUILD = sts_extreme.build
_ORIG_LOSS = sts_extreme.head_loss
_STATE: dict = {"w": None}
N_FEAT = 14
TAU_BASE = 30


# ------------------------------------------------------------------ Feuerhistorie-Merkmale
def features_px(fire: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
    """fire (B, T, 13, 13) {0, 1}, doy (B,) -> Merkmale je Pixel (B, 13, 13, 14):
    je Region (Pixel, 3 x 3 um das Pixel, ganzer Ausschnitt) log1p(Feuer-Pixeltage der letzten 1 / 7 / 30 Tage)
    und Tage seit dem letzten Feuer (0 = letzter Fenstertag, 30 = keins); dazu sin / cos des Kalendertags."""
    B, T, Hh, W = fire.shape
    f = fire.float()
    n3 = F.conv2d(f.reshape(B * T, 1, Hh, W), f.new_ones(1, 1, 3, 3), padding=1).view(B, T, Hh, W)
    na = f.sum((2, 3), keepdim=True).expand(B, T, Hh, W)
    tix = torch.arange(1, T + 1, device=f.device, dtype=f.dtype)[None, :, None, None]
    out = []
    for r in (f, n3, na):
        for k in (1, 7, 30):
            out.append(torch.log1p(r[:, -k:].sum(1)))
        last = ((r > 0).to(f.dtype) * tix).amax(1) - 1.0                         # Index des letzten Feuertags, -1 = keins
        out.append((T - 1) - last)
    ang = 2 * math.pi * doy.float() / 365.25
    out += [ang.sin()[:, None, None].expand(B, Hh, W), ang.cos()[:, None, None].expand(B, Hh, W)]
    return torch.stack(out, -1)


def fit_base(fire_tr: np.ndarray, doy_tr: np.ndarray, y_c: np.ndarray, y_p: np.ndarray, chunk: int = 8000) -> dict:
    """Logistische Basis (Mitte, Ausschnitt) auf den Mittelpixel-Merkmalen; Koeffizienten im Rohraum + Zentrierung."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([features_px(torch.from_numpy(fire_tr[a:a + chunk]), torch.from_numpy(doy_tr[a:a + chunk]))[:, 6, 6]
                        .numpy() for a in range(0, len(fire_tr), chunk)]).astype(np.float64)
    coef = {}
    for k, y in (("centre", y_c), ("patch", y_p)):
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(X), y.astype(int))
        w = lr.coef_[0] / sc.scale_
        b = float(lr.intercept_[0] - (w * sc.mean_).sum())
        p = np.clip(1 / (1 + np.exp(-(X @ w + b))), 1e-6, 1 - 1e-6)
        o = np.log(-np.log1p(-p))
        coef[f"w_{k}"], coef[f"b_{k}"], coef[f"c_{k}"] = w.astype(np.float32), np.float32(b), np.float32(o.mean())
        print(f"[sfo] Basis {k}: Rate {y.mean():.4f}, mittleres p {p.mean():.4f}, Offset-Zentrum {o.mean():.3f}", flush=True)
    return coef


def offsets(fire: torch.Tensor, doy: torch.Tensor, coef: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """-> Offset je Pixel (B, 13, 13) und je Ausschnitt (B,), float32."""
    with torch.autocast(device_type=fire.device.type, enabled=False):
        Xf = features_px(fire.float(), doy)
        res = []
        for k in ("centre", "patch"):
            w = torch.as_tensor(coef[f"w_{k}"], device=fire.device, dtype=torch.float32)
            p = torch.sigmoid(Xf @ w + float(coef[f"b_{k}"])).clamp(1e-6, 1 - 1e-6)
            res.append(torch.log(-torch.log1p(-p)) - float(coef[f"c_{k}"]))
    return res[0], res[1][:, 6, 6]


# ------------------------------------------------------------------ Huelle und Verlust
def wrap(net: nn.Module, variants: tuple, p_drop: float, w_ign: float, coef: dict | None, mu_f: float, sd_f: float,
         pad: str = "zeros"):
    if pad != "zeros":
        for m in net.stem.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                m.padding_mode = pad
    fwd = net.forward
    z0 = (0.0 - mu_f) / sd_f

    def focus(x, doy, *a, **kw):
        fire = (x[:, :, FIRE_CH].float() * sd_f + mu_f) > 0.5                     # (B, T, 13, 13)
        if net.training:
            if "ignw" in variants:
                _STATE["w"] = torch.where(fire.flatten(1).any(1), 1.0, float(w_ign)).float()
            if "fdrop" in variants or "fdropall" in variants:
                drop = torch.rand(x.shape[0], device=x.device) < p_drop
                x = x.clone()
                x[:, :, FIRE_CH] = torch.where(drop[:, None, None, None], torch.full_like(x[:, :, FIRE_CH], z0),
                                               x[:, :, FIRE_CH])
                if "fdropall" in variants:
                    th = [c for c in FIRE_THERMAL if c != FIRE_CH]
                    x[:, :, th] = torch.where(drop[:, None, None, None, None], torch.zeros_like(x[:, :, th]), x[:, :, th])
                fire = fire & ~drop[:, None, None, None]
        out = fwd(x, doy, *a, **kw)
        if "resid" in variants:
            o_px, o_pa = offsets(fire, doy, coef)
            out = (out[0] + o_px[..., None].to(out[0].dtype), out[1] + o_pa[:, None].to(out[1].dtype), *out[2:])
        return out

    net.forward = focus
    return net


def head_loss_w(head, raw, event, t):
    """Wie head_loss (Intervall-NLL), aber mit den Stichproben-Gewichten des letzten Trainings-Vorwaertslaufs."""
    w = _STATE["w"]
    if w is None or isinstance(head, BernsteinSurvivalHead):
        return _ORIG_LOSS(head, raw, event, t)
    raw2 = raw.reshape(-1, raw.shape[-1])
    ev, tt = event.reshape(-1).bool(), t.reshape(-1).double()
    ls = head.log_survival(raw2, torch.stack([tt - 1.0, tt], 1))
    ll_ev = ls[:, 0] + torch.log((-torch.expm1(ls[:, 1] - ls[:, 0])).clamp_min(1e-300))
    nll = -torch.where(ev, ll_ev, ls[:, 1])
    ww = w.to(nll.device, torch.float64).repeat_interleave(raw2.shape[0] // w.numel())
    return (nll * ww).sum() / ww.sum()


def build_focus(variants: tuple, coef: dict | None, mu_f: float = 0.0, sd_f: float = 1.0,
                p_drop: float = 0.5, w_ign: float = 4.0, pad: str = "zeros"):
    return lambda *a, **kw: wrap(_ORIG_BUILD(*a, **kw), variants, p_drop, w_ign, coef, mu_f, sd_f, pad)


def load_coef(path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ Hauptteil
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", required=True, help="kommagetrennt aus " + ", ".join(VARIANTS))
    ap.add_argument("--p-drop", type=float, default=0.5)
    ap.add_argument("--w-ign", type=float, default=4.0)
    ap.add_argument("--pad", choices=("zeros", "reflect"), default="zeros")
    own, rest = ap.parse_known_args()
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", required=True)
    peek.add_argument("--tag", required=True)
    peek.add_argument("--head", default="lh")
    peek.add_argument("--cache", default="cache_G")
    peek.add_argument("--cube-dir", default="paper_pixel")
    pk, _ = peek.parse_known_args(rest)
    variants = tuple(v for v in own.variants.split(",") if v)
    if any(v not in VARIANTS for v in variants):
        raise SystemExit(f"ABBRUCH: --variants {own.variants}")
    if "resid" in variants and pk.head != "lh":
        raise SystemExit("ABBRUCH: resid nur mit --head lh")
    cfg = load_config(pk.config)
    exp = Path(cfg["experiment_root"])
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / pk.cache / "norm_stats.npz")
    mu_f, sd_f = float(mean[FIRE_CH]), float(std[FIRE_CH])
    coef, coef_path = None, None
    if "resid" in variants:
        import datetime as dt

        from wildfire.data.dataset import CubeReader
        from wildfire.paper.fire_baseline import fire_of
        from wildfire.paper.sts_dctm import CENTRE, patch_events
        pdir = exp / "derived_data" / pk.cube_dir
        Xtr = np.load(pdir / "train_X.npy", mmap_mode="r")
        n = Xtr.shape[0]
        fire = fire_of(Xtr, np.arange(n))
        dates = CubeReader(exp).dates
        doy = np.array([dt.date.fromisoformat(dates[int(L)][:10]).timetuple().tm_yday
                        for L in np.load(pdir / "train_meta.npz")["L"][:n]], np.float32)
        ev = np.asarray(np.load(pdir / "train_event.npy", mmap_mode="r"))
        tt = np.asarray(np.load(pdir / "train_t.npy", mmap_mode="r")).astype(np.int64)
        pe, pt = patch_events(ev, tt)
        coef = fit_base(fire, doy, ev[:, CENTRE, CENTRE] & (tt[:, CENTRE, CENTRE] <= TAU_BASE), pe & (pt <= TAU_BASE))
        del fire
        coef_path = exp / "derived_data" / "sts_focus" / f"coef{pk.tag}.npz"
        if coef_path.exists():
            raise SystemExit(f"ABBRUCH: {coef_path} existiert schon")
        coef_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(coef_path, **coef)
    print(f"[sfo] Varianten {variants}, p_drop {own.p_drop}, w_ign {own.w_ign}, Padding {own.pad}", flush=True)
    sts_extreme.build = build_focus(variants, coef, mu_f, sd_f, own.p_drop, own.w_ign, own.pad)
    if "ignw" in variants:
        sts_extreme.head_loss = head_loss_w
    sys.argv = [sys.argv[0]] + rest
    sts_extreme.main()
    write_manifest(cfg, f"p44_sts_focus{pk.tag}", pk.config, {
        "variants": variants, "pad": own.pad, "p_drop": own.p_drop, "w_ign": own.w_ign, "coef": str(coef_path) if coef_path else None,
        "coef_values": {k: np.asarray(v).tolist() for k, v in (coef or {}).items()},
        "resid_form": "raw_LH + (log(-log(1 - p_basis30)) - Mittel_train)" if coef else None,
        "sts_extreme_args": rest, "sts_extreme_manifest": f"p35_sts_extreme{pk.tag}"})


if __name__ == "__main__":
    main()

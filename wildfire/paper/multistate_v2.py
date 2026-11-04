"""Multi-State-Contagion v2: Zustandsmerkmale ueber das Fenster hinaus, Rueckkehr-Hazard ueber die Zeit seit dem
letzten Feuer, optionale Ursachen-Supervision, Horizont-BCE.

Ausgangspunkt multistate.py (v1):
    1 - h_i(t) = (1 - a_i(t)) (1 - r_i(t) o_i(t)) prod_{j != i} (1 - g_i k_ij b_j(t-1)) (1 - k_ii b_i^obs(t-1))
Schwachstellen von v1 (state_eval, multistate_eval): Rueckkehr und Fortsetzung ranken schlecht (Typ-AUC@30 0,78 /
0,80), v1 sieht erloschene Pixel nur innerhalb der 30 Fenstertage, und r_i(t) haengt nur am Tag nach L, nicht an
der Pause seit dem letzten Feuer.

Bausteine (je per --kind / Schalter einzeln ablierbar):
  1  Zustandsmerkmale (state_features.py, 5 Werte je Pixel: log Tage seit dem letzten Feuer, kein Feuer im Jahr,
     log Feuertage in 183 Tagen, Zustand O, Zustand B) gehen frueh in den Rumpf ein: eine kleine MLP bettet sie je
     Pixel ein und addiert sie zu allen Zeit-Token vor den Attention-Bloecken (letzte Schicht mit Nullen
     initialisiert, Start = Modell ohne Merkmale). Der Erloschen-Indikator ever_i umfasst dann 183 statt 30 Tage.
  2  Rueckkehr-Hazard als glatte Funktion der Pause: logit r_i(t) = r0 + sum_m (c_im + c_m) phi_m(log(1 + ds_i + t)),
     ds_i = Tage seit dem letzten Feuer am Stichtag, phi_m 8 Gauss-Basisfunktionen in log-Tagen (1 .. 466),
     c_im linear aus dem Pixel-Token, c_m global. Fortsetzung bleibt k_ii b_i^obs mit der gelernten Brenndauer-Kurve.
  3  Ursachen-Supervision: Typ des ersten Pixel-Feuers aus state_features (continuation, return, spread im 7 x 7-Kern,
     ignition) -> -log w_c(k), w_c = log(1 - h_c(k)) / log(1 - h(k)) (Anteil am kumulativen Hazard des Ereignistags,
     wie decompose_causes); Gewicht --lam-type (klein). Die Randkurve F bleibt die Summe der Ursachen-CIFs.
  4  Horizont-BCE (horizon_bce.py): + beta * sum_{tau in 1,7,14,30} BCE(F(tau)) je Kopf (Pixel und Ausschnitt).

Arten (--kind):
  lh     STS-LH (Logistic-Hazard-Kopf), dieselbe Schleife                 -> Kontrolle der Schleife (= _abl_lhbce10)
  lhf    STS-LH + Zustandsmerkmale                                       -> trennt Merkmale von Multi-State-Struktur
  ms     Multi-State v1-Struktur (Rueckkehr je Tag, ever 30 Tage)          -> Effekt von beta auf v1
  msf    ms + Zustandsmerkmale (ever 183 Tage), Rueckkehr je Tag           -> Baustein 1
  msv2   msf + Rueckkehr ueber die Pause (Baustein 2)                       -> mit --lam-type > 0 Baustein 3

Schleife wie contagion_cause (Wuerfel-Stichtage paper_pixel, d 64, 4 Bloecke, Batch 32, AdamW 3e-4, wd 0,05,
500 Warm-up, EMA 0,999, letzte EMA-Gewichte), von Grund auf. Checkpoints NEU unter checkpoints/paper_msv2/.
Die Netze brauchen die Merkmale (set_features) vor jedem Vorwaertslauf; die Auswertung (msv2_eval.py) setzt sie.

  python -m wildfire.paper.multistate_v2 train --config ... --kind msv2 --seed 0 --tag _msv2 [--beta 1 --lam-type 0.3]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config
from wildfire.data.dataset import CubeReader
from wildfire.models.ema import EMA
from wildfire.models.surv_heads import head_loss
from wildfire.paper import horizon_bce, targets
from wildfire.paper.channels import masked_std
from wildfire.paper.contagion import ContagionSTS
from wildfire.paper.multistate import MultiStateContagionSTS, self_burning
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.state_features import NF
from wildfire.paper.sts_dctm import STSDCTM, lr_factor, patch_events

P = 13
KINDS = ("lh", "lhf", "ms", "msf", "msv2")
N_BASIS = 8
U_LO, U_HI = math.log1p(1.0), math.log1p(466.0)


class FeatMixin:
    """Zustandsmerkmale (B, 169, NF) frueh in den Rumpf: Einbettung je Pixel, zu allen Zeit-Token addiert."""

    def _init_feat(self, d: int, use: bool):
        self.use_feat = bool(use)
        self._feat = None
        if self.use_feat:
            self.feat_emb = nn.Sequential(nn.Linear(NF, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(self.feat_emb[2].weight); nn.init.zeros_(self.feat_emb[2].bias)

    def set_features(self, f):
        self._feat = f

    def encode(self, x, doy):
        """Wie STSDCTM.encode (stem 2d), dazu die Merkmal-Einbettung vor den Bloecken."""
        x = x[:, -self.hist:]
        B, T, C, H, W = x.shape
        z = self.stem(x.flatten(0, 1))
        z = z.flatten(2).transpose(1, 2).reshape(B, T, H * W, -1)
        ang = 2 * math.pi * doy.float() / 365.25
        z = z + self.pos_s + self.pos_t[:, -T:] + self.season(torch.stack([ang.sin(), ang.cos()], -1))[:, None, None]
        if self.use_feat:
            if self._feat is None:
                raise RuntimeError("Zustandsmerkmale fehlen: set_features vor dem Vorwaertslauf")
            z = z + self.feat_emb(self._feat.to(z.dtype))[:, None]
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        return self.read(torch.cat([z[:, -1], z.mean(1)], -1))


class LHFeat(FeatMixin, STSDCTM):
    def __init__(self, use_feat: bool = True, **kw):
        STSDCTM.__init__(self, head="lh", **kw)
        self._init_feat(kw.get("d", 96), use_feat)


class MSv2(FeatMixin, MultiStateContagionSTS):
    def __init__(self, use_feat: bool = True, ret_mode: str = "spline", **kw):
        MultiStateContagionSTS.__init__(self, **kw)
        d = kw.get("d", 64)
        self._init_feat(d, use_feat)
        if ret_mode not in ("day", "spline"):
            raise ValueError(ret_mode)
        if ret_mode == "spline" and not use_feat:
            raise ValueError("Rueckkehr ueber die Pause braucht die Zustandsmerkmale")
        self.ret_mode = ret_mode
        if ret_mode == "spline":
            del self.ret
            self.ret_px = nn.Linear(d, N_BASIS)
            nn.init.zeros_(self.ret_px.weight); nn.init.zeros_(self.ret_px.bias)
            self.ret_g = nn.Parameter(torch.zeros(N_BASIS))
            self.register_buffer("ret_c", torch.linspace(U_LO, U_HI, N_BASIS))

    def params(self, x, doy, pix=None):
        x = x[:, -self.hist:]
        pix = self.encode_input(x, doy) if pix is None else pix
        obs, a, k, logD = ContagionSTS.params(self, x, doy, pix)
        with torch.autocast(device_type=x.device.type, enabled=False):
            ever = (obs.amax(1) > 0).float()                                            # (B, N) im Fenster
            if self.use_feat:
                f = self._feat.float()
                ever = torch.maximum(ever, (f[..., 3] + f[..., 4]).clamp(max=1.0))      # Feuer in 183 Tagen
            b_self = self_burning(self, obs, logD)                                       # (B, H, N)
            o = ever[:, None] * (1 - b_self)
            if self.ret_mode == "spline":
                f = self._feat.float()
                ds = torch.where(f[..., 1] > 0.5, torch.full_like(f[..., 0], 366.0),
                                 torch.expm1(f[..., 0] * math.log(366.0)))               # (B, N) Tage seit Feuer
                t = torch.arange(1, self.H + 1, device=x.device, dtype=torch.float32)
                u = torch.log1p(ds[:, None, :] + t[None, :, None])                       # (B, H, N)
                w = (self.ret_c[1] - self.ret_c[0])
                phi = torch.exp(-0.5 * ((u[..., None] - self.ret_c) / w) ** 2)          # (B, H, N, M)
                coef = self.ret_px(pix.float()) + self.ret_g                             # (B, N, M)
                r = torch.sigmoid(self.ret_bias + (phi * coef[:, None]).sum(-1))
            else:
                r = torch.sigmoid(self.ret(pix.float()) + self.ret_bias).transpose(1, 2)
            ro = r * o
            a_eff = 1 - (1 - a) * (1 - ro)
            g = torch.sigmoid(self.fuel)
            damp = 1 - (1 - g) * ever
            c = self.K * self.K // 2
            rowmask = torch.ones(self.K * self.K, 1, device=k.device)
            rowmask[c] = 0.0
            k_eff = k * (1 - rowmask * (1 - damp[:, None, :]))
        self._last = {"a": a, "ro": ro, "ever": ever, "b_self": b_self, "k": k_eff}
        return obs, a_eff, k_eff, logD


def build(kind: str, **kw):
    cg = {k: kw.pop(k) for k in ("kernel", "obs_days", "burn", "blind") if k in kw}
    if kind in ("lh", "lhf"):
        return LHFeat(use_feat=kind == "lhf", **kw)
    return MSv2(use_feat=kind in ("msf", "msv2"), ret_mode="spline" if kind == "msv2" else "day", **cg, **kw)


def needs_features(net) -> bool:
    return bool(getattr(net, "use_feat", False))


def load(path, device="cpu"):
    c = torch.load(path, map_location="cpu")
    a = c["args"]
    # init_cif bestimmt Bias-Konstanten (Python-Floats, nicht im state dict) - wie im Training setzen
    kw = {"d": a["d"], "heads": a["heads"], "depth": a["depth"], "init_cif_px": a["init_cif_px"],
          "init_cif_patch": a["init_cif_patch"]}
    if a["kind"] not in ("lh", "lhf"):
        kw.update(kernel=a["cg_kernel"], obs_days=a["cg_obs_days"], burn=a["cg_burn"], blind="none")
    net = build(a["kind"], **kw)
    net.load_state_dict(c["state"])                            # fire_mu / fire_sd sind Puffer im state dict
    return net.to(device).eval()


def is_msv2(path) -> bool:
    return "paper_msv2" in str(path)


# ------------------------------------------------------------------ Training
def forward_ms(net, x, doy, device):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pix = net.encode_input(x, doy)
    obs, a_eff, k, logD = net.params(x, doy, pix)
    with torch.autocast(device_type=device.type, enabled=False):
        log_exo = net.exogenous(obs, a_eff, k, logD)
        ls_px = net.rollout(log_exo, k, logD)
        ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0))
    return ls_px, ls_pa


def type_term(net, ls_px, ev, tt, typ):
    """Summe von -log w_c(k) ueber Pixel-Ereignisse mit Typ (1 cont, 2 ret, 3 spr, 4 ign) und deren Zahl."""
    B, N, _ = ls_px.shape
    ev, tt, typ = ev.reshape(B, N), tt.reshape(B, N), typ.reshape(B, N)
    lab = ev & (typ > 0)
    if not lab.any():
        return ls_px.new_zeros(()), 0
    b, i = torch.nonzero(lab, as_tuple=True)
    k = tt[b, i].long()
    l1mh = (ls_px[b, i, k] - ls_px[b, i, k - 1]).clamp(max=-1e-12)
    L = net._last
    c = net.K * net.K // 2
    l_ign = torch.log1p(-L["a"][b, k - 1, i].clamp(max=1 - 1e-6))
    l_ret = torch.log1p(-L["ro"][b, k - 1, i].clamp(max=1 - 1e-6))
    l_cont = torch.log1p(-(L["k"][b, c, i] * L["b_self"][b, k - 1, i]).clamp(max=1 - 1e-6))
    l_spr = (l1mh - l_ign - l_ret - l_cont).clamp(max=0.0)
    parts = torch.stack([l_cont, l_ret, l_spr, l_ign], 1)                               # Typ 1..4
    w = (parts.gather(1, (typ[b, i].long() - 1)[:, None])[:, 0] / l1mh).clamp(1e-6, 1.0)
    return -torch.log(w).sum(), int(lab.sum())


def train(args) -> None:
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    ck_dir = exp / "checkpoints" / "paper_msv2"
    ck = ck_dir / f"sts_{args.cache}_next_cube{args.tag}_s{args.seed}.pt"
    if ck.exists():
        raise SystemExit(f"ABBRUCH: {ck} existiert schon - wird nicht ueberschrieben")
    if args.lam_type > 0 and args.kind not in ("ms", "msf", "msv2"):
        raise SystemExit("ABBRUCH: --lam-type nur fuer Multi-State-Arten")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    horizon_bce.CFG.update(beta=args.beta, taus=tuple(sorted(args.hz_taus)))

    cdir = exp / "derived_data" / args.cube_dir
    meta = dict(np.load(cdir / "train_meta.npz"))
    n_all = meta["L"].size
    sfd = exp / "derived_data" / "state_features"
    use_feat = args.kind in ("lhf", "msf", "msv2")
    fc = np.load(sfd / f"train{args.feat_tag}.npy", mmap_mode="r") if use_feat or args.lam_type > 0 else None
    n_pool = n_all
    if fc is not None and fc.shape[0] < n_all:
        if not args.limit:                                     # Rauchtest-Merkmale decken nur die ersten Zeilen ab
            raise SystemExit(f"ABBRUCH: Merkmale ({fc.shape[0]}) kuerzer als train_meta ({n_all})")
        n_pool = fc.shape[0]
    rows = np.arange(n_all) if not args.limit else np.sort(rng.choice(n_pool, min(args.limit, n_pool), replace=False))
    t0 = time.time()
    Xc = np.asarray(np.load(cdir / "train_X.npy", mmap_mode="r")[rows])
    evc = np.load(cdir / "train_event.npy")[rows].astype(bool)
    tc = np.load(cdir / "train_t.npy")[rows].astype(np.int64)
    fc = np.asarray(fc[rows]) if use_feat else None
    tyc = np.load(sfd / f"train_types{args.feat_tag}.npy", mmap_mode="r")[rows] if args.lam_type > 0 else None
    if tyc is not None and not ((tyc > 0) == evc.reshape(len(rows), -1)).all():
        raise SystemExit("ABBRUCH: Typen passen nicht zu train_event.npy")
    dates = CubeReader(exp).dates
    doyc = np.array([dt.date.fromisoformat(dates[int(L)]).timetuple().tm_yday for L in meta["L"][rows]], np.float32)
    init_px, init_pa = float(evc.mean()), float(evc.reshape(len(evc), -1).any(1).mean())
    print(f"[msv2] {args.kind}: {len(rows):,} Stichtage in {time.time() - t0:.0f} s; Pixel-Ereignisse {init_px:.4f}; "
          f"Merkmale {'ja' if use_feat else 'nein'}; beta {args.beta}; lam_type {args.lam_type}", flush=True)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    std = masked_std(std, "paper55")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    kw = dict(d=args.d, heads=args.heads, depth=args.depth, p=args.dropout,
              init_cif_px=float(np.clip(init_px, 1e-4, 0.5)), init_cif_patch=float(np.clip(init_pa, 1e-3, 0.9)))
    if args.kind not in ("lh", "lhf"):
        kw.update(kernel=args.cg_kernel, obs_days=args.cg_obs_days, burn=args.cg_burn, blind="none")
    net = build(args.kind, **kw).to(device)
    if hasattr(net, "set_fire_norm"):
        net.set_fire_norm(float(mean[FIRE_CH]), float(std[FIRE_CH]))
    print(f"[msv2] {sum(p.numel() for p in net.parameters()) / 1e6:.3f} Mio. Parameter", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.warmup, total, "const", 0.0))
    ema = EMA(net, args.ema)
    hist = []
    for ep in range(args.epochs):
        net.train()
        t_ep = time.time()
        order = rng.permutation(len(rows))[:args.steps_per_epoch * args.bs]
        sums = {"nll": 0.0, "bce": 0.0, "typ": 0.0, "n": 0, "n_lab": 0}
        for a0 in range(0, order.size, args.bs):
            r = np.sort(order[a0:a0 + args.bs])
            x = (torch.from_numpy(Xc[r]).to(device).float() - m_t) / s_t
            doy = torch.from_numpy(doyc[r]).to(device)
            if use_feat:
                net.set_features(torch.from_numpy(fc[r]).to(device).float())
            ev, tt = torch.from_numpy(evc[r]).to(device), torch.from_numpy(tc[r]).to(device)
            pe, pt = patch_events(evc[r], tc[r])
            pe, pt = torch.from_numpy(pe).to(device), torch.from_numpy(pt).to(device)
            if args.kind in ("lh", "lhf"):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw_px, raw_pa = net(x, doy)
            else:
                raw_px, raw_pa = forward_ms(net, x, doy, device)
            nll = head_loss(net.px_head, raw_px, ev, tt) + args.patch_weight * head_loss(net.pa_head, raw_pa, pe, pt)
            bce = (horizon_bce.horizon_bce(net.px_head, raw_px, ev, tt)
                   + args.patch_weight * horizon_bce.horizon_bce(net.pa_head, raw_pa, pe, pt)) if args.beta > 0 \
                else nll.new_zeros(())
            loss = nll + args.beta * bce
            n_lab = 0
            if args.lam_type > 0:
                tt_, n_lab = type_term(net, raw_px, ev, tt, torch.from_numpy(np.asarray(tyc[r])).to(device))
                loss = loss + args.lam_type * tt_ / (len(r) * P * P)
                sums["typ"] += float(tt_)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            ema.update(net)
            sums["nll"] += nll.item() * len(r); sums["bce"] += float(bce) * len(r)
            sums["n"] += len(r); sums["n_lab"] += n_lab
        h = {"epoch": ep, "train_nll": sums["nll"] / sums["n"], "train_bce": sums["bce"] / sums["n"],
             "type_nll_per_label": sums["typ"] / max(sums["n_lab"], 1), "labels": sums["n_lab"]}
        if hasattr(net, "fuel"):
            h["fuel_gate"] = float(torch.sigmoid(net.fuel).item())
        hist.append(h)
        print(f"      Epoche {ep}: NLL {h['train_nll']:.5f}  BCE {h['train_bce']:.5f}  Typ -log w {h['type_nll_per_label']:.4f} "
              f"({h['labels']:,})  {('g %.3f' % h['fuel_gate']) if 'fuel_gate' in h else ''} ({time.time() - t_ep:.0f} s)",
              flush=True)
        if not all(np.isfinite([h["train_nll"], h["train_bce"]])):
            raise SystemExit("ABBRUCH: Verlust nicht endlich")
    state = ema.state(net.state_dict())
    ck_dir.mkdir(parents=True, exist_ok=True)
    saved = {"kind": args.kind, "d": args.d, "heads": args.heads, "depth": args.depth, "hist_days": 30,
             "cg_kernel": args.cg_kernel, "cg_obs_days": args.cg_obs_days, "cg_burn": args.cg_burn,
             "feat_tag": args.feat_tag, "init_cif_px": kw["init_cif_px"], "init_cif_patch": kw["init_cif_patch"],
             "msv2": vars(args)}
    torch.save({"state": state, "args": saved, "history": hist}, ck)
    (ck_dir / (ck.stem + ".json")).write_text(json.dumps({"args": vars(args), "history": hist}, indent=1))
    print(f"[msv2] Checkpoint {ck}", flush=True)


def main() -> None:
    import sys
    if len(sys.argv) < 2 or sys.argv[1] != "train":
        raise SystemExit(__doc__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--feat-tag", default="")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--hz-taus", type=int, nargs="+", default=[1, 7, 14, 30])
    ap.add_argument("--lam-type", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patch-weight", type=float, default=1.0)
    ap.add_argument("--cg-kernel", type=int, default=7)
    ap.add_argument("--cg-obs-days", type=int, default=30)
    ap.add_argument("--cg-burn", default="curve")
    ap.add_argument("--limit", type=int, default=0, help="nur fuer Rauchtests")
    ap.add_argument("--tag", required=True)
    train(ap.parse_args(sys.argv[2:]))


if __name__ == "__main__":
    main()

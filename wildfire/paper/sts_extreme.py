"""STS mit Zusatzeingaben fuer Extremlagen und Monotonie-Regularisierung (Reviewer-Kritik, Punkte 2, 3, 5).

Gleiches Protokoll wie die Ablation (slurm 74: Wuerfel-Stichtage 2000-2019, Ziel D+1, d 64, 4 Bloecke,
EMA 0.999, Batch 32, lr 3e-4, 10 x 3000 Schritte, letzte EMA-Gewichte), Koepfe LH oder Contagion
(K 7, 30 Beobachtungstage, Brennkurve, ohne Warmstart). sts_dctm.py bleibt unveraendert; die Netze
hier erben von STSDCTM bzw. ContagionSTS und ueberschreiben nur encode.

  --ctx reg    regionale Feueraktivitaet der Saison (ctx_features.REG_NAMES, 8 Skalare je Stichtag)
               -> MLP -> auf alle Token addiert
  --ctx px     Feuerwetter relativ zur eigenen Zelle (Perzentil 2000-2019, ueber Zellmaximum, Kappung
               bei 255, NoData; 14 Kanaele je Pixel am Stichtag) -> linear je Pixel -> auf die Token
               des Pixels (alle Tage) addiert
  --ctx both   beides
               Beide Zweige starten bei null: zu Beginn ist das Netz genau das Modell ohne Zusatz.
  --mono w     Monotonie-Strafterm: hoeheres Feuerwetter (t2m_max, FFMC, DMC, DC, ISI, BUI, FWI, VPD;
               je Stichprobe zufaellige Teilmenge, Verschiebung U(0, --mono-scale) Standardabweichungen
               ueber das ganze Fenster, mit --ctx px auch die Perzentile) darf das kumulative Risiko
               Lambda(7), Lambda(30), Lambda(100) nicht senken: w * mean relu(log Lambda(x) - log Lambda(x')),
               Ausschnitt und alle Pixel, auf der halben Batch, Dropout aus.

Auswertung je Seed und als Ensemble (Mittel der Survival-Kurven):
  surv     zufaellige Stichtage (--surv-dir): Val 2020-2022, Test 2023-2025; Metriken wie surv_eval
           (roh und mit Lambda' = e^a Lambda^b auf Val), Hazards nach derived_data/calib_cache_ext/
           (Format wie calib_cache, fuer extreme_eval)
  paper    Paper-Protokoll (Ziel D+1): Stichtag e = D - dt, Score F(dt + 1), Mitte und Ausschnittskopf
Kopf-Bias: wie im Training (Ereignisraten der Wuerfel-Stichtage), siehe Memory load_sts-Bias. Mit
--eval-ckpt werden vorhandene sts_dctm-Checkpoints (ohne Zusatz) genauso ausgewertet, Bias ebenso aus
den Wuerfel-Raten, damit Referenz und Varianten vergleichbar sind.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.eval.metrics import evaluate
from wildfire.models.ema import EMA
from wildfire.models.surv_heads import head_loss
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.contagion import ContagionSTS
from wildfire.paper.ctx_features import PX_NAMES, REG_NAMES
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import CENTRE, P, STSDCTM, doy_of, lr_factor, patch_events
from wildfire.paper.surv_eval import calib_apply, calib_fit, compact, ensemble

H = 100
MONO_CH = (13, 19, 20, 21, 22, 23, 24, 25)          # t2m_max, FFMC, DMC, DC, ISI, BUI, FWI, VPD
PCT_OF = {19: 0, 20: 1, 21: 2, 22: 3, 23: 4, 24: 5}  # Wetterkanal -> pct_-Kanal in PX_NAMES
MONO_TAUS = (7.0, 30.0, 100.0)


# ------------------------------------------------------------------ Modell
class CtxMixin:
    """Zusatzeingaben fuer encode; set_ctx vor dem Aufruf, Werte gelten fuer genau diesen Batch."""

    def init_ctx(self, d: int, k_px: int, k_reg: int) -> None:
        self.ctx_px = nn.Linear(k_px, d) if k_px else None
        self.ctx_reg = nn.Sequential(nn.Linear(k_reg, d), nn.SiLU(), nn.Linear(d, d)) if k_reg else None
        for lin in (self.ctx_px, self.ctx_reg[-1] if k_reg else None):
            if lin is not None:
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self._ctx = (None, None)

    def set_ctx(self, px=None, reg=None) -> None:
        self._ctx = (px, reg)

    def encode(self, x, doy):
        """Wie STSDCTM.encode, dazu die Zusatzeingaben vor den Bloecken."""
        x = x[:, -self.hist:]
        B, T, C, Hh, W = x.shape
        if self.stem_kind == "3d":
            z = self.stem(x.transpose(1, 2)).permute(0, 2, 3, 4, 1).reshape(B, T, Hh * W, -1)
        else:
            z = self.stem(x.flatten(0, 1)).flatten(2).transpose(1, 2).reshape(B, T, Hh * W, -1)
        ang = 2 * math.pi * doy.float() / 365.25
        z = z + self.pos_s + self.pos_t[:, -T:] + self.season(torch.stack([ang.sin(), ang.cos()], -1))[:, None, None]
        px, reg = self._ctx
        if self.ctx_px is not None and px is not None:
            z = z + self.ctx_px(px.flatten(2).transpose(1, 2))[:, None]          # (B, 1, 169, d)
        if self.ctx_reg is not None and reg is not None:
            z = z + self.ctx_reg(reg)[:, None, None]
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        return self.read(torch.cat([z[:, -1], z.mean(1)], -1))


class STSExt(CtxMixin, STSDCTM):
    def __init__(self, k_px: int = 0, k_reg: int = 0, **kw):
        STSDCTM.__init__(self, **kw)
        self.init_ctx(kw.get("d", 96), k_px, k_reg)


class ContagionExt(CtxMixin, ContagionSTS):
    def __init__(self, k_px: int = 0, k_reg: int = 0, **kw):
        ContagionSTS.__init__(self, **kw)
        self.init_ctx(kw.get("d", 96), k_px, k_reg)


def build(head: str, ctx: str, init_px: float, init_pa: float, d: int, depth: int, heads: int, dropout: float,
          cg: dict) -> nn.Module:
    k_px = len(PX_NAMES) if ctx in ("px", "both") else 0
    k_reg = len(REG_NAMES) if ctx in ("reg", "both") else 0
    kw = dict(d=d, depth=depth, heads=heads, p=dropout, init_cif_px=float(np.clip(init_px, 1e-4, 0.5)),
              init_cif_patch=float(np.clip(init_pa, 1e-3, 0.9)))
    if head == "contagion":
        return ContagionExt(k_px, k_reg, **kw, **cg)
    return STSExt(k_px, k_reg, head=head, **kw)


def load_any(path: str, init_px: float, init_pa: float, device) -> tuple[nn.Module, str]:
    """Checkpoint von sts_extreme oder sts_dctm (dann ohne Zusatz) -> Netz im eval-Modus, Bias aus den
    uebergebenen Wuerfel-Raten."""
    c = torch.load(path, map_location="cpu")
    a = c.get("args", {})
    head = a.get("head", "dctm")
    cg = ({"kernel": a.get("cg_kernel", 5), "obs_days": a.get("cg_obs_days", 7), "burn": a.get("cg_burn", "geometric")}
          if head == "contagion" else {})
    ctx = a.get("ctx", "none")
    net = build(head, ctx, init_px, init_pa, a.get("d", 96), a.get("depth", 4), a.get("heads", 4), 0.1, cg)
    if a.get("hist_days", 30) != 30 or a.get("attn", "factorized") != "factorized" or a.get("stem", "2d") != "2d":
        raise SystemExit(f"ABBRUCH: {path}: nur Standard-Rumpf (hist 30, factorized, 2d) unterstuetzt")
    net.load_state_dict(c["state"])
    return net.to(device).eval(), ctx


# ------------------------------------------------------------------ Hilfen
def cum_hazard_at(net, raw_px, raw_pa, taus) -> torch.Tensor:
    """log Lambda an taus fuer alle Pixel und den Ausschnitt -> (B, 170, len(taus))."""
    B, M = raw_pa.shape[0], raw_pa.shape[-1]
    ls_px = net.px_head.log_survival(raw_px.reshape(-1, raw_px.shape[-1]), taus).view(B, -1, taus.numel())
    ls_pa = net.pa_head.log_survival(raw_pa.reshape(B, M), taus).view(B, 1, taus.numel())
    return torch.log((-torch.cat([ls_px, ls_pa], 1)).clamp_min(1e-10))


def load_ctx(cdir: Path, name: str, n_full: int, use_px: bool, use_reg: bool):
    """<name>_px/_reg (nur die benutzten, sonst None) und die Zeilen dazu (Rauchtest: <name>_rows.npy, sonst alle)."""
    rows = np.load(cdir / f"{name}_rows.npy") if (cdir / f"{name}_rows.npy").exists() else np.arange(n_full)
    return (rows, np.load(cdir / f"{name}_px.npy") if use_px else None,
            np.load(cdir / f"{name}_reg.npy") if use_reg else None)


# ------------------------------------------------------------------ Hauptteil
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--head", choices=["lh", "contagion"], default="lh")
    ap.add_argument("--ctx", choices=["none", "reg", "px", "both"], default="none")
    ap.add_argument("--mono", type=float, default=0.0, help="Gewicht des Monotonie-Strafterms; 0 = aus")
    ap.add_argument("--mono-scale", type=float, default=1.5, help="max. Verschiebung in Standardabweichungen")
    ap.add_argument("--ctx-dir", default="ctx_features")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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
    ap.add_argument("--cg-kernel", type=int, default=7)
    ap.add_argument("--cg-obs-days", type=int, default=30)
    ap.add_argument("--cg-burn", choices=["geometric", "curve"], default="curve")
    ap.add_argument("--eval-ckpt", default="", help="nur auswerten: glob vorhandener Checkpoints (Referenz)")
    ap.add_argument("--limit-train", type=int, default=0, help="nur fuer Rauchtests")
    ap.add_argument("--limit-eval", type=int, default=0, help="nur fuer Rauchtests: so viele Stichtage je Menge")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"sts_extreme{args.tag}"
    ckpt_dir = exp / "checkpoints" / "paper"
    hz_dir = exp / "derived_data" / "calib_cache_ext"
    if not args.eval_ckpt and list(ckpt_dir.glob(f"{name}_s*.pt")):
        raise SystemExit(f"ABBRUCH: Checkpoints {name}_s*.pt existieren schon - anderen --tag waehlen")
    use_px, use_reg = args.ctx in ("px", "both"), args.ctx in ("reg", "both")
    cdir = exp / "derived_data" / args.ctx_dir
    t0 = time.time()

    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    def to_dev(xw):
        xw = xw if isinstance(xw, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(xw))
        return (xw.to(device).float() - m_t) / s_t

    # ---- Wuerfel-Stichtage (Training); Ereignisraten auch fuer den Kopf-Bias der Referenz
    pdir = exp / "derived_data" / args.cube_dir
    evc_all = np.load(pdir / "train_event.npy", mmap_mode="r")
    n_full = evc_all.shape[0]
    rows_tr, px_tr, reg_tr = (load_ctx(cdir, "train", n_full, use_px, use_reg) if args.ctx != "none"
                              else (np.arange(n_full), None, None))
    if args.limit_train:
        rows_tr, px_tr, reg_tr = rows_tr[:args.limit_train], *(None if a is None else a[:args.limit_train]
                                                                for a in (px_tr, reg_tr))
    evc = np.asarray(evc_all[rows_tr])
    tc = np.load(pdir / "train_t.npy", mmap_mode="r")[rows_tr].astype(np.int64)
    init_px, init_pa = float(evc.mean()), float(evc.reshape(len(evc), -1).any(1).mean())
    reg_mu = reg_sd = None
    if use_reg:
        reg_mu, reg_sd = reg_tr.mean(0), reg_tr.std(0) + 1e-6
    norm_reg = lambda r: (r - reg_mu) / reg_sd  # noqa: E731
    print(f"[sx] Wuerfel: {rows_tr.size:,} Stichtage; Pixel-Ereignisse {init_px:.4f}, Ausschnitt {init_pa:.3f}", flush=True)

    # ---- Auswertemengen
    sd_ = exp / "derived_data" / args.surv_dir
    surv = {}
    for s, f in (("val", "val"), ("test", "eval")):
        meta = dict(np.load(sd_ / f"{f}_meta.npz"))
        n_s = meta["row"].size
        rws, cpx, creg = (load_ctx(cdir, f"surv_{f}", n_s, use_px, use_reg) if args.ctx != "none" else (np.arange(n_s), None, None))
        if args.limit_eval:
            rws, cpx, creg = rws[:args.limit_eval], *(None if a is None else a[:args.limit_eval] for a in (cpx, creg))
        surv[s] = {"rows": rws, "X": np.load(sd_ / f"{f}_X.npy", mmap_mode="r"), "doy": meta["doy"][rws],
                   "px": cpx, "reg": None if creg is None else norm_reg(creg),
                   "e": {"centre": meta["ev_centre"][rws].astype(bool), "patch": meta["ev_patch"][rws].astype(bool)},
                   "t": {"centre": meta["t_centre"][rws].astype(int), "patch": meta["t_patch"][rws].astype(int)}}
    cd = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    Xp = np.load(cd / "X.npy", mmap_mode="r")
    LC = Xp.shape[1]
    split = np.load(cd / "split.npy")
    y_pa = targets.labels(cd, "next_patch")
    y_px = targets.fire_map(cd, Xp, "next_patch")[:, CENTRE, CENTRE].astype(int)
    doyD = doy_of(np.load(cd / "meta.npy")[:, 0])
    rows_c, px_c, reg_c = (load_ctx(cdir, "cache", Xp.shape[0], use_px, use_reg) if args.ctx != "none"
                           else (np.arange(Xp.shape[0]), None, None))
    pos_c = {int(r): i for i, r in enumerate(rows_c)}
    te_c = np.array([r for r in np.where(split == "test")[0] if int(r) in pos_c])
    if args.limit_eval:
        te_c = te_c[:args.limit_eval]
    Xg = to_device_once(np.asarray(Xp[np.sort(te_c)]) if args.limit_eval else np.asarray(Xp), device, "cache_X")
    gi = ({int(r): i for i, r in enumerate(np.sort(te_c))} if args.limit_eval else None)
    print(f"[sx] Auswertung: Val {surv['val']['rows'].size:,}, Test {surv['test']['rows'].size:,}, "
          f"Paper-Test {te_c.size:,} ({time.time() - t0:.0f} s)", flush=True)

    def run(net, x, doy, px=None, reg=None):
        if hasattr(net, "set_ctx"):
            net.set_ctx(px if use_px else None, reg if use_reg else None)
        return net(x, doy)

    @torch.no_grad()
    def surv_hazards(net, s, bs=128):
        S = surv[s]
        hc, hp = [], []
        net.eval()
        for a in range(0, S["rows"].size, bs):
            r = S["rows"][a:a + bs]
            x = to_dev(np.asarray(S["X"][r[0]:r[-1] + 1] if r[-1] - r[0] == r.size - 1 else S["X"][r]))
            px = None if S["px"] is None else torch.from_numpy(S["px"][a:a + bs].astype(np.float32)).to(device)
            reg = None if S["reg"] is None else torch.from_numpy(S["reg"][a:a + bs].astype(np.float32)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw_px, raw_pa = run(net, x, torch.from_numpy(S["doy"][a:a + bs]).to(device), px, reg)
            hc.append(net.px_head.hazards(raw_px[:, CENTRE, CENTRE], H).double().cpu().numpy())
            hp.append(net.pa_head.hazards(raw_pa, H).double().cpu().numpy())
        return {"centre": np.concatenate(hc), "patch": np.concatenate(hp)}

    @torch.no_grad()
    def paper_eval(net, bs=64) -> dict:
        net.eval()
        out = {}
        for k, dt_ in enumerate(DTS):
            e = LC - 1 - dt_
            tau = torch.tensor([float(dt_ + 1)], device=device)
            fc, fa = [], []
            for a in range(0, te_c.size, bs):
                r = te_c[a:a + bs]
                rg = np.array([gi[int(v)] for v in r]) if gi is not None else r
                ci = np.array([pos_c[int(v)] for v in r])
                px = None if px_c is None else torch.from_numpy(px_c[ci, k].astype(np.float32)).to(device)
                reg = None if reg_c is None else torch.from_numpy(norm_reg(reg_c[ci, k]).astype(np.float32)).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw_px, raw_pa = run(net, to_dev(windows(Xg, rg, np.full(r.size, e))),
                                         torch.from_numpy(doyD[r] - dt_).to(device), px, reg)
                fc.append((-torch.expm1(net.px_head.log_survival(raw_px[:, CENTRE, CENTRE], tau))[:, 0]).cpu().numpy())
                fa.append((-torch.expm1(net.pa_head.log_survival(raw_pa, tau))[:, 0]).cpu().numpy())
            cl = lambda v: np.clip(np.concatenate(v), 1e-7, 1 - 1e-7)  # noqa: E731
            out[str(dt_)] = {"centre": paper_metrics(cl(fc), y_px[te_c]), "patch_head": paper_metrics(cl(fa), y_pa[te_c])}
        return out

    def surv_metrics(hz: dict) -> dict:
        """Metriken auf Test, roh und kalibriert (Lambda' = e^a Lambda^b auf Val)."""
        res = {}
        for tg in ("centre", "patch"):
            e, t = surv["test"]["e"][tg], surv["test"]["t"][tg]
            a, b = calib_fit(hz["val"][tg], surv["val"]["e"][tg], surv["val"]["t"][tg])
            res[tg] = {"raw": compact(evaluate(hz["test"][tg], e, t)),
                       "curve": compact(evaluate(calib_apply(hz["test"][tg], a, b), e, t)), "curve_ab": [a, b]}
            for v in ("raw", "curve"):
                m = res[tg][v]
                print(f"[sx]   {tg:6s} {v:5s} NLL {m['nll']:.4f}  Ctd {m['c_td']:.3f}  IBS {m['ibs']:.5f}  "
                      f"D-Cal {m['dcal_dev']:.3f}  PR@7 {m['pr@7']:.3f}  PR@30 {m['pr@30']:.3f}", flush=True)
        return res

    # ---- Training bzw. Laden
    nets = []
    hists = []
    if args.eval_ckpt:
        for f in sorted(glob.glob(args.eval_ckpt)):
            net, ctx_ck = load_any(f, init_px, init_pa, device)
            if ctx_ck != args.ctx:
                raise SystemExit(f"ABBRUCH: {f} hat ctx {ctx_ck}, Lauf --ctx {args.ctx}")
            nets.append((Path(f).stem.split("_")[-1], net))
            print(f"[sx] geladen: {f}", flush=True)
        if not nets:
            raise SystemExit(f"ABBRUCH: keine Checkpoints fuer {args.eval_ckpt}")
    else:
        print(f"[sx] lade {rows_tr.size:,} Trainingsfenster in den RAM ...", flush=True)
        t1 = time.time()
        Xm = np.load(pdir / "train_X.npy", mmap_mode="r")
        Xc = np.asarray(Xm[:rows_tr.size]) if np.array_equal(rows_tr, np.arange(rows_tr.size)) else np.asarray(Xm[rows_tr])
        dates = CubeReader(exp).dates
        Lm = np.load(pdir / "train_meta.npz")["L"][rows_tr]
        doyc = np.array([dt.date.fromisoformat(dates[int(L)]).timetuple().tm_yday for L in Lm], np.float32)
        px_g = None if not use_px else torch.from_numpy(px_tr.astype(np.float32)).to(device)
        reg_g = None if not use_reg else torch.from_numpy(norm_reg(reg_tr).astype(np.float32)).to(device)
        print(f"[sx] Trainingsfenster {Xc.shape} in {time.time() - t1:.0f} s", flush=True)
        mono_ch = torch.tensor(MONO_CH, device=device)
        pct_idx = [PCT_OF.get(c, -1) for c in MONO_CH]
        taus_m = torch.tensor(MONO_TAUS, device=device, dtype=torch.float64)
        cg = {"kernel": args.cg_kernel, "obs_days": args.cg_obs_days, "burn": args.cg_burn}
        for seed in args.seeds:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = build(args.head, args.ctx, init_px, init_pa, args.d, args.depth, args.heads, args.dropout,
                        cg if args.head == "contagion" else {}).to(device)
            if hasattr(net, "set_fire_norm"):
                net.set_fire_norm(float(mean[FIRE_CH]), float(std[FIRE_CH]))
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
            total = args.epochs * args.steps_per_epoch
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.warmup, total))
            ema = EMA(net, args.ema) if args.ema > 0 else None
            print(f"[sx] Seed {seed}: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} Mio. Parameter", flush=True)
            hist = []
            for ep in range(args.epochs):
                net.train()
                t_ep = time.time()
                rows = rng.permutation(rows_tr.size)[:args.steps_per_epoch * args.bs]
                tl = tp = tn = 0.0
                for a in range(0, rows.size, args.bs):
                    r = np.sort(rows[a:a + args.bs])
                    ev, tt = evc[r], tc[r]
                    pe, pt = patch_events(ev, tt)
                    x = to_dev(Xc[r])
                    doy = torch.from_numpy(doyc[r]).to(device)
                    rt = torch.from_numpy(r).to(device)
                    px = px_g[rt] if px_g is not None else None
                    reg = reg_g[rt] if reg_g is not None else None
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        raw_px, raw_pa = run(net, x, doy, px, reg)
                    loss = head_loss(net.px_head, raw_px, torch.from_numpy(ev).to(device), torch.from_numpy(tt).to(device)) \
                        + head_loss(net.pa_head, raw_pa, torch.from_numpy(pe).to(device), torch.from_numpy(pt).to(device))
                    pen = torch.zeros((), device=device)
                    if args.mono > 0:                       # Monotonie: halbe Batch, Dropout aus
                        net.eval()
                        hb = max(1, len(r) // 2)
                        u = (torch.rand(hb, len(MONO_CH), device=device) * args.mono_scale
                             * (torch.rand(hb, len(MONO_CH), device=device) < 0.5))
                        x2 = x[:hb].clone()
                        x2[:, :, mono_ch] += u[:, None, :, None, None]
                        px1 = px[:hb] if px is not None else None
                        px2 = None
                        if px is not None:
                            px2 = px1.clone()
                            for j, pi in enumerate(pct_idx):
                                if pi >= 0:
                                    px2[:, pi] += 0.1 * u[:, j, None, None]
                        reg1 = reg[:hb] if reg is not None else None
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            o1 = run(net, x[:hb], doy[:hb], px1, reg1)
                            o2 = run(net, x2, doy[:hb], px2, reg1)
                        l1 = cum_hazard_at(net, o1[0].float(), o1[1].float(), taus_m)
                        l2 = cum_hazard_at(net, o2[0].float(), o2[1].float(), taus_m)
                        pen = F.relu(l1 - l2).mean()
                        net.train()
                    opt.zero_grad(set_to_none=True)
                    (loss + args.mono * pen).backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step(); sched.step()
                    if ema is not None:
                        ema.update(net)
                    tl += loss.item() * len(r); tp += float(pen) * len(r); tn += len(r)
                hist.append({"epoch": ep, "train_nll": tl / tn, "mono_pen": tp / tn})
                print(f"      Epoche {ep:2d}: train NLL {tl / tn:.5f}  Monotonie {tp / tn:.5f}  "
                      f"({time.time() - t_ep:.0f} s)", flush=True)
            if ema is not None:
                net.load_state_dict(ema.state({k: v.detach().clone() for k, v in net.state_dict().items()}))
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cp = ckpt_dir / f"{name}_s{seed}.pt"
            torch.save({"state": state, "args": vars(args), "init_px": init_px, "init_pa": init_pa}, cp)
            print(f"[sx] Checkpoint {cp}", flush=True)
            nets.append((f"s{seed}", net.eval()))
            hists.append(hist)
        del Xc

    # ---- Auswertung
    hz_dir.mkdir(parents=True, exist_ok=True)
    res, acc = {}, {s: {"centre": [], "patch": []} for s in ("val", "test")}
    for sname, net in nets:
        t1 = time.time()
        hz = {s: surv_hazards(net, s) for s in ("val", "test")}
        for s in hz:
            for tg in hz[s]:
                acc[s][tg].append(hz[s][tg])
        print(f"[sx] {sname}: Hazards in {time.time() - t1:.0f} s", flush=True)
        res[sname] = {"surv": surv_metrics(hz), "paper": paper_eval(net)}
        print("[sx]   Paper-Protokoll PR-AUC Mitte/Ausschnitt: " + "  ".join(
            f"dt {k}: {v['centre']['pr_auc']:.3f}/{v['patch_head']['pr_auc']:.3f}" for k, v in res[sname]["paper"].items()),
            flush=True)
        np.savez(hz_dir / f"{name}_{sname}.npz", **{f"{s}_{tg}": acc[s][tg][-1].astype(np.float32)
                                                     for s in acc for tg in acc[s]})
    if len(nets) > 1:
        ens = {s: {tg: ensemble(v) for tg, v in acc[s].items()} for s in acc}
        print("[sx] Ensemble:", flush=True)
        res["ens"] = {"surv": surv_metrics(ens)}
        np.savez(hz_dir / f"{name}.npz", **{f"{s}_{tg}": ens[s][tg].astype(np.float32) for s in ens for tg in ens[s]})
    write_manifest(cfg, f"p35_{name}", args.config, {
        "head": args.head, "ctx": args.ctx, "mono": args.mono, "mono_scale": args.mono_scale,
        "mono_channels": MONO_CH, "mono_taus": MONO_TAUS, "eval_ckpt": args.eval_ckpt, "seeds": args.seeds,
        "ctx_dir": args.ctx_dir, "px_names": PX_NAMES if use_px else None, "reg_names": REG_NAMES if use_reg else None,
        "bias": {"mode": "trained (Ereignisraten der Wuerfel-Stichtage)", "init_px": init_px, "init_pa": init_pa},
        "training": {k: getattr(args, k) for k in ("epochs", "steps_per_epoch", "bs", "lr", "wd", "warmup", "ema",
                                                    "d", "depth", "heads", "dropout", "cg_kernel", "cg_obs_days", "cg_burn")},
        "n": {"train": int(rows_tr.size), "val": int(surv["val"]["rows"].size), "test": int(surv["test"]["rows"].size),
              "paper_test": int(te_c.size)},
        "hazards": str(hz_dir / f"{name}.npz"), "histories": hists, "results": res,
        "paper_protocol": "Ziel D+1, Stichtag e = D - dt, Score F(dt + 1); Mitte = Pixelkopf (6, 6)",
    })


if __name__ == "__main__":
    main()

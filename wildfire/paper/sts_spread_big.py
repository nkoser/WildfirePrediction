"""Contagion-STS mit Ring + Teacher Forcing und Uebergewicht aktiver Braende im Training.

Kopie von sts_spread_ring.py (bleibt unveraendert). Diagnose spread_distance.py (Job 5759): an Tagen grosser
Braende liegen 93 % der neuen Feuerpixel innerhalb Radius 5 vom Vortagesfeuer, der Kern reicht also; die
Regionssimulation brennt trotzdem zu wenig. Die Wuerfel-Stichtage sind aber meist kleine/keine Feuer, die
Ausbreitung grosser Fronten kommt im Verlust kaum vor. Neu:
  --big-thr N     aktiv = mindestens N Feuerpixel (23 x 23 mit Ring) an den letzten --big-days Eingabetagen
  --big-share s   Anteil aktiver Stichtage je Batch (Ziehen mit Zuruecklegen nach Gewicht; 0 = wie bisher)
Ausgaben NEU: checkpoints/paper/sts_spread_big<tag>_s<seed>.pt, derived_data/calib_cache_ext/sts_spread_big<tag>*.npz,
Manifest p62_sts_spread_big<tag>.

--- Beschreibung von sts_spread_ring.py ---
Contagion-STS mit gerichtetem Kern und Feuerring im Training (contagion_ring.py, Daten aus ring_targets.py).

Kopie von sts_spread_tf.py (bleibt unveraendert). Im Training sieht der Kern auch Feuer bis R = K // 2 Pixel
ausserhalb des 13 x 13-Ausschnitts (beobachtete Eingabetage; mit --tf-weight > 0 auch die beobachteten
Zuendtage des Rings im Teacher Forcing). Auswertung (surv, Paper-Protokoll) wie bisher OHNE Ring, also
direkt vergleichbar; die Regionssimulation (contagion_mc_map2) braucht keinen Ring (globales Gitter).
--tf-weight 0 = nur Ring. Ausgaben NEU: checkpoints/paper/sts_spread_ring<tag>_s<seed>.pt,
derived_data/calib_cache_ext/sts_spread_ring<tag>*.npz, Manifest p58_sts_spread_ring<tag>.

--- Beschreibung von sts_spread_tf.py ---
Contagion-STS mit gerichtetem Kern (wie sts_spread.py) und zusaetzlichem Teacher-Forcing-Verlust (contagion_tf.py).

Kopie von sts_spread.py (bleibt unveraendert). Neu: --tf-weight lam; Trainingsverlust
    NLL (Mean-Field-Rollout, wie bisher) + lam * NLL (Ausbreitung aus der beobachteten Front der Folgetage).
Nur Training; Auswertung, Paper-Protokoll und Checkpoint-Format wie sts_spread (load_spread laedt sie).
Ohne Wetter-Oracle (--future-days 0). Ausgaben NEU: checkpoints/paper/sts_spread_tf<tag>_s<seed>.pt,
derived_data/calib_cache_ext/sts_spread_tf<tag>*.npz, Manifest p56_sts_spread_tf<tag>.

--- urspruengliche Beschreibung von sts_spread.py ---
Training und Auswertung des Contagion-STS mit gerichtetem, wetterabhaengigem Kern (contagion_spread.py).

Kopie von sts_extreme.main mit gleichem Protokoll (Wuerfel-Stichtage 2000-2019, Ziel D+1, d 64, 4 Bloecke,
EMA 0.999, Batch 32, lr 3e-4, 10 x 3000 Schritte, letzte EMA-Gewichte, ohne Warmstart, Brennkurve,
30 Beobachtungstage); sts_extreme.py bleibt unveraendert. Neu:
  --spread ell|free   Richtungs-/Reichweitenterm (ell) oder nur freier Kern (free)
  --cg-kernel K       Kerngroesse (Standard 11 statt 7)
  --future-days F     Wetter-Oracle: beobachtetes Wetter der Folgetage aus train_Xf.npy (future_weather, 78);
                      im Training je Stichprobe zufaellig 0..kf bekannte Tage (wie sts_dctm v31), im
                      Paper-Protokoll die dt Tage e+1..D aus dem Cache. Die Survival-Mengen (surv) werden
                      OHNE Folgewetter ausgewertet (kf = 0), dort ist das Oracle also das Modell ohne Zusatz.
Ausgaben NEU: checkpoints/paper/sts_spread<tag>_s<seed>.pt, derived_data/calib_cache_ext/sts_spread<tag>*.npz,
Manifest p50_sts_spread<tag>. Ergebnisse in derselben Form wie sts_extreme (surv + paper je Seed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
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
from wildfire.paper.contagion_spread import WX, SpreadContagion
from wildfire.paper.contagion_ring import RingContagion, forward_ring
from wildfire.paper.ctx_features import PX_NAMES, REG_NAMES
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import CENTRE, doy_of, lr_factor, patch_events
from wildfire.paper.sts_extreme import MONO_CH, MONO_TAUS, PCT_OF, cum_hazard_at, load_ctx
from wildfire.paper.surv_eval import calib_apply, calib_fit, compact, ensemble

H = 100


def build(ctx: str, init_px: float, init_pa: float, d: int, depth: int, heads: int, dropout: float, cg: dict,
          spread: str, future_days: int) -> nn.Module:
    k_px = len(PX_NAMES) if ctx in ("px", "both") else 0
    k_reg = len(REG_NAMES) if ctx in ("reg", "both") else 0
    return RingContagion(k_px, k_reg, spread=spread, future_days=future_days, d=d, depth=depth, heads=heads, p=dropout,
                           init_cif_px=float(np.clip(init_px, 1e-4, 0.5)), init_cif_patch=float(np.clip(init_pa, 1e-3, 0.9)),
                           **cg)


def load_spread(path: str, device) -> SpreadContagion:
    """Checkpoint von sts_spread -> Netz im eval-Modus."""
    c = torch.load(path, map_location="cpu")
    a = c["args"]
    net = build(a.get("ctx", "none"), c["init_px"], c["init_pa"], a["d"], a["depth"], a["heads"], 0.1,
                {"kernel": a["cg_kernel"], "obs_days": a["cg_obs_days"], "burn": a["cg_burn"]},
                a["spread"], a.get("future_days", 0))
    net.load_state_dict(c["state"])
    return net.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--head", choices=["contagion"], default="contagion")
    ap.add_argument("--spread", choices=["ell", "free"], default="ell")
    ap.add_argument("--tf-weight", type=float, default=1.0, help="Gewicht lam des Teacher-Forcing-Verlusts (0 = aus)")
    ap.add_argument("--ring-dir", default="paper_pixel_ring5", help="derived_data/<ring-dir> aus ring_targets.py")
    ap.add_argument("--big-thr", type=int, default=5, help="aktiv: so viele Feuerpixel an den letzten --big-days Tagen")
    ap.add_argument("--big-days", type=int, default=3)
    ap.add_argument("--big-share", type=float, default=0.5, help="Anteil aktiver Stichtage je Batch (0 = gleichverteilt)")
    ap.add_argument("--future-days", type=int, default=0, help="Wetter-Oracle: Folgetage (0 = aus, max 30)")
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
    ap.add_argument("--cg-kernel", type=int, default=11)
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
    name = f"sts_spread_big{args.tag}"
    if args.future_days:
        ap.error("Teacher Forcing nur ohne --future-days")
    if args.future_days and args.spread != "ell":
        ap.error("--future-days nur mit --spread ell")
    if not 0 <= args.future_days <= 30:
        ap.error("--future-days 0..30")
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

    mf_t, sf_t = m_t[:, :, WX], s_t[:, :, WX]                         # Wetter-Oracle

    def to_dev_f(xf):
        xf = xf if isinstance(xf, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(xf))
        return (xf.to(device).float() - mf_t) / sf_t

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
    rdir = exp / "derived_data" / args.ring_dir
    ring_h = np.asarray(np.load(rdir / "ring_hist.npy", mmap_mode="r")[rows_tr])          # gepackt (n, 4, PE, PE)
    ring_e = np.asarray(np.load(rdir / "ring_event.npy", mmap_mode="r")[rows_tr])
    ring_t = np.load(rdir / "ring_t.npy", mmap_mode="r")[rows_tr].astype(np.int64)
    if ring_e.shape[-1] != 13 + 2 * (args.cg_kernel // 2):
        ap.error(f"Ring {ring_e.shape[-1]} passt nicht zu --cg-kernel {args.cg_kernel}")
    inner = (slice(None), slice(args.cg_kernel // 2, args.cg_kernel // 2 + 13), slice(args.cg_kernel // 2, args.cg_kernel // 2 + 13))
    if not (np.array_equal(ring_e[inner], evc) and np.array_equal(ring_t[inner], tc)):
        raise SystemExit("ABBRUCH: Ring-Ziele innen != train_event/train_t")
    print(f"[sx] Feuerring {ring_e.shape[1:]}: Ereignisse je Stichtag innen {evc.reshape(len(evc), -1).sum(1).mean():.2f}, "
          f"gesamt {ring_e.reshape(len(ring_e), -1).sum(1).mean():.2f}", flush=True)
    n_fire = np.concatenate([np.unpackbits(ring_h[a:a + 20000], axis=1)[:, 30 - args.big_days:30].any(1).sum((1, 2))
                             for a in range(0, len(ring_h), 20000)])
    active = n_fire >= args.big_thr
    act_frac = float(active.mean())
    samp_p = None
    if args.big_share > 0 and 0 < act_frac < 1:
        samp_p = np.where(active, args.big_share / active.sum(), (1 - args.big_share) / (~active).sum())
        samp_p = samp_p / samp_p.sum()
    print(f"[sx] aktiv (>= {args.big_thr} Feuerpixel an {args.big_days} Tagen): {act_frac:.3f} der Stichtage, "
          f"Ereignisse innen aktiv {evc[active].reshape(active.sum(), -1).sum(1).mean():.1f} / "
          f"sonst {evc[~active].reshape((~active).sum(), -1).sum(1).mean():.1f}; Batch-Anteil {args.big_share if samp_p is not None else act_frac:.2f}",
          flush=True)
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

    def run(net, x, doy, px=None, reg=None, fut=(None, None)):
        if hasattr(net, "set_ctx"):
            net.set_ctx(px if use_px else None, reg if use_reg else None)
        net.set_future(*fut)
        return net(x, doy)

    def paper_future(rg, dt_):
        """Oracle im Paper-Protokoll: Wetter der dt Tage e+1 .. D aus dem Cache (Tag D+1 bleibt unbekannt)."""
        kk = min(dt_, args.future_days)
        if not args.future_days or kk <= 0:
            return None, None
        w = windows(Xg, rg, np.full(rg.size, LC - 1))
        w = w[:, w.shape[1] - dt_:w.shape[1] - dt_ + kk]                   # Tage e+1 .. e+kk
        xf = torch.zeros(rg.size, args.future_days, len(WX), w.shape[-2], w.shape[-1], device=device)
        xf[:, :kk] = to_dev_f(w[:, :, WX])
        return xf, torch.full((rg.size,), kk, device=device, dtype=torch.long)

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
                                         torch.from_numpy(doyD[r] - dt_).to(device), px, reg, paper_future(rg, dt_))
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
            net = load_spread(f, device)
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
        if args.future_days:                                # Oracle: Folgewetter je Batch aus dem memmap
            Xfm = np.load(pdir / "train_Xf.npy", mmap_mode="r")
            kfc = np.minimum(np.load(pdir / "train_kf.npy")[rows_tr].astype(np.int64), args.future_days)
            print(f"[sx] Wetter-Oracle: {Xfm.shape}, volle {args.future_days} Folgetage bei "
                  f"{(kfc == args.future_days).mean():.3f}", flush=True)
        mono_ch = torch.tensor(MONO_CH, device=device)
        pct_idx = [PCT_OF.get(c, -1) for c in MONO_CH]
        taus_m = torch.tensor(MONO_TAUS, device=device, dtype=torch.float64)
        cg = {"kernel": args.cg_kernel, "obs_days": args.cg_obs_days, "burn": args.cg_burn}
        for seed in args.seeds:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = build(args.ctx, init_px, init_pa, args.d, args.depth, args.heads, args.dropout, cg,
                        args.spread, args.future_days).to(device)
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
                rows = (rng.choice(rows_tr.size, args.steps_per_epoch * args.bs, p=samp_p) if samp_p is not None
                        else rng.permutation(rows_tr.size)[:args.steps_per_epoch * args.bs])
                tl = tp = tn = ttf = 0.0
                for a in range(0, rows.size, args.bs):
                    r = np.sort(rows[a:a + args.bs])
                    ev, tt = evc[r], tc[r]
                    pe, pt = patch_events(ev, tt)
                    x = to_dev(Xc[r])
                    doy = torch.from_numpy(doyc[r]).to(device)
                    rt = torch.from_numpy(r).to(device)
                    px = px_g[rt] if px_g is not None else None
                    reg = reg_g[rt] if reg_g is not None else None
                    fut = (None, None)
                    if args.future_days:                    # Oracle: zufaellig 0..k bekannte Folgetage
                        xf = to_dev_f(np.asarray(Xfm[rows_tr[r]][:, :args.future_days]))
                        fut = (xf, torch.from_numpy(rng.integers(0, kfc[r] + 1)).to(device))
                    net.set_ctx(px if use_px else None, reg if use_reg else None)
                    net.set_future(*fut)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        rh = torch.from_numpy(np.unpackbits(ring_h[r], axis=1)[:, 30 - args.cg_obs_days:30]).to(device)
                        raw_px, raw_pa, tfl = forward_ring(
                            net, x, doy, rh, *((torch.from_numpy(ring_e[r]).to(device), torch.from_numpy(ring_t[r]).to(device))
                                                 if args.tf_weight > 0 else (None, None)))
                        if tfl is None:
                            tfl = torch.zeros((), device=device)
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
                    (loss + args.tf_weight * tfl + args.mono * pen).backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step(); sched.step()
                    if ema is not None:
                        ema.update(net)
                    tl += loss.item() * len(r); tp += float(pen) * len(r); tn += len(r); ttf += tfl.item() * len(r)
                hist.append({"epoch": ep, "train_nll": tl / tn, "mono_pen": tp / tn, "tf_nll": ttf / tn})
                print(f"      Epoche {ep:2d}: train NLL {tl / tn:.5f}  TF-NLL {ttf / tn:.5f}  Monotonie {tp / tn:.5f}  "
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
    write_manifest(cfg, f"p62_{name}", args.config, {
        "big": {"thr": args.big_thr, "days": args.big_days, "share": args.big_share, "active_frac": act_frac},
        "tf_weight": args.tf_weight, "ring_dir": args.ring_dir, "head": args.head, "spread": args.spread, "future_days": args.future_days, "cg_kernel": args.cg_kernel,
        "ctx": args.ctx, "mono": args.mono, "mono_scale": args.mono_scale,
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

"""Contagion-STS mit teilweise beobachteter Ursache: Zuendung und Ausbreitung als konkurrierende Ursachen.

Das Modell ist unveraendert ContagionSTS (paper/contagion.py),

    1 - h_i(t) = (1 - a_i(t)) * prod_j (1 - k_ij b_j(t-1)),   h_ign = a_i(t),
    log(1 - h_spr) = log(1 - h) - log(1 - a),

nur der Verlust bekommt fuer Pixel-Ereignisse mit bekannter Ursache c (cause_labels.py) einen Term:

    log P(erstes Feuer an Tag k, Ursache c trat ein) = [log S(k-1) + log h(k)] + log(h_c(k) / h(k)).

    L = L_bisher + lambda * mean_Pixel( -log(h_c(k) / h(k)) ueber Ereignisse mit Label )

L_bisher ist exakt der Verlust von sts_dctm (Intervall-NLL je Pixel + Gewicht 1 x Ausschnitt), lambda = 0
reproduziert also den Lauf _abl_cg. Einstellungen wie 74_paper_ablation v21 (_abl_cg): Rumpf aus
_abl_lh (Warmstart, nur Rumpf), 3 x 3.000 Schritte, d 64, 4 Bloecke, Batch 32, AdamW 3e-4, wd 0,05,
500 Warm-up, EMA 0,999, letzte EMA-Gewichte, Kern 7, 30 Beobachtungstage, Brenndauer als Kurve.

sts_dctm.py bleibt unveraendert; die Schleife hier ist eine schlanke Kopie fuer --train-source cube.
Checkpoints NEU unter checkpoints/paper_cause/ im Format von sts_dctm (load_sts liest sie), damit
surv_eval, contagion_eval und region_eval sie unveraendert auswerten. Bricht ab, wenn das Ziel existiert.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
from wildfire.paper import targets
from wildfire.paper.cause_labels import SPREAD
from wildfire.paper.channels import masked_std
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import build_sts, lr_factor, patch_events

P = 13


def forward_parts(net, x, doy, device):
    """Wie ContagionSTS.forward, dazu a (B, H, N): -> log S Pixel (B, N, H+1), log S Ausschnitt (B, H+1), a."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pix = net.encode_input(x, doy)
    obs, a, k, logD = net.params(x, doy, pix)
    with torch.autocast(device_type=device.type, enabled=False):
        log_exo = net.exogenous(obs, a, k, logD)
        ls_px = net.rollout(log_exo, k, logD)
        ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0))
    return ls_px, ls_pa, a


def cause_term(ls_px, a, ev, tt, code, use_spread: bool):
    """Summe von -log(h_c(k) / h(k)) ueber Pixel-Ereignisse mit Label, und deren Zahl."""
    B, N, _ = ls_px.shape
    ev, tt, code = ev.reshape(B, N), tt.reshape(B, N), code.reshape(B, N)
    ign = ev & (code >= 1) & (code <= 3)
    spr = ev & (code == SPREAD) if use_spread else torch.zeros_like(ev)
    lab = ign | spr
    if not lab.any():
        return ls_px.new_zeros(()), 0
    b, i = torch.nonzero(lab, as_tuple=True)
    k = tt[b, i].long()                                                   # Ereignistag 1..H
    l1mh = ls_px[b, i, k] - ls_px[b, i, k - 1]                            # log(1 - h(k))
    h = (-torch.expm1(l1mh)).clamp_min(1e-12)
    ak = a[b, k - 1, i].clamp(1e-12, 1 - 1e-6)
    h_spr = (-torch.expm1(l1mh - torch.log1p(-ak))).clamp_min(1e-12)
    log_pi = torch.where(ign[b, i], torch.log(ak) - torch.log(h), torch.log(h_spr) - torch.log(h))
    return -log_pi.clamp(max=0).sum(), int(lab.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--labels", default="train_cause.npz", help="Datei unter derived_data/cause_labels")
    ap.add_argument("--lam", type=float, default=1.0, help="Gewicht des Ursachen-Terms; 0 = wie _abl_cg")
    ap.add_argument("--use-spread", type=int, default=1, help="1 = Ausbreitungs-Labels nutzen, 0 = nur Zuendung")
    ap.add_argument("--init-from", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_lh_s{seed}_s{seed}.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
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
    ap.add_argument("--cg-blind", default="none")
    ap.add_argument("--limit", type=int, default=0, help="nur fuer Rauchtests: so viele Trainings-Stichtage")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    ck_dir = exp / "checkpoints" / "paper_cause"
    ck = ck_dir / f"sts_{args.cache}_next_cube_cause{args.tag}_s{args.seed}.pt"
    if ck.exists():
        raise SystemExit(f"ABBRUCH: {ck} existiert schon - wird nicht ueberschrieben")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- Daten (nur gelesen)
    cdir = exp / "derived_data" / args.cube_dir
    meta = dict(np.load(cdir / "train_meta.npz"))
    lab = dict(np.load(exp / "derived_data" / "cause_labels" / args.labels))
    if not (np.array_equal(lab["row"], meta["row"]) and np.array_equal(lab["col"], meta["col"])):
        raise SystemExit("ABBRUCH: Ursachen-Labels passen nicht zu train_meta.npz")
    n_all = meta["L"].size
    rows = np.arange(n_all) if not args.limit else np.sort(rng.choice(n_all, args.limit, replace=False))
    t0 = time.time()
    Xc = np.asarray(np.load(cdir / "train_X.npy", mmap_mode="r")[rows])
    evc = np.load(cdir / "train_event.npy")[rows].astype(bool)
    tc = np.load(cdir / "train_t.npy")[rows].astype(np.int64)
    cc = lab["cause"][rows]
    if not ((cc > 0) == evc).all():
        raise SystemExit("ABBRUCH: Ursachen-Codes passen nicht zu train_event.npy")
    dates = CubeReader(exp).dates
    doyc = np.array([dt.date.fromisoformat(dates[int(L)]).timetuple().tm_yday for L in meta["L"][rows]], np.float32)
    init_px, init_pa = float(evc.mean()), float(evc.reshape(len(evc), -1).any(1).mean())
    e = cc[evc]
    print(f"[cause] {len(rows):,} Stichtage in {time.time() - t0:.0f} s; Pixel-Ereignisse {int(evc.sum()):,}: "
          f"Zuendung {np.isin(e, (1, 2, 3)).mean():.3f}, Ausbreitung {(e == SPREAD).mean():.3f}, "
          f"ohne Label {(e == 5).mean():.3f}", flush=True)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    std = masked_std(std, "paper55")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    # ---- Modell wie _abl_cg
    net = build_sts("contagion", d=args.d, heads=args.heads, depth=args.depth, p=args.dropout,
                    init_cif_px=float(np.clip(init_px, 1e-4, 0.5)), init_cif_patch=float(np.clip(init_pa, 1e-3, 0.9)),
                    kernel=args.cg_kernel, obs_days=args.cg_obs_days, burn=args.cg_burn, blind=args.cg_blind).to(device)
    net.set_fire_norm(float(mean[FIRE_CH]), float(std[FIRE_CH]))
    if args.init_from:
        src = args.init_from.format(seed=args.seed)
        sd0 = torch.load(src, map_location="cpu")["state"]
        own = net.state_dict()
        sd0 = {k: v for k, v in sd0.items() if k in own and own[k].shape == v.shape and not k.startswith(("px_", "pa_"))}
        net.load_state_dict(sd0, strict=False)
        print(f"[cause] Rumpf aus {src}: {len(sd0)} von {len(own)} Tensoren", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.warmup, total, "const", 0.0))
    ema = EMA(net, args.ema)
    hist = []
    for ep in range(args.epochs):
        net.train()
        t_ep = time.time()
        order = rng.permutation(len(rows))[:args.steps_per_epoch * args.bs]
        sums = {"nll": 0.0, "cause": 0.0, "n": 0, "n_lab": 0}
        for a0 in range(0, order.size, args.bs):
            r = np.sort(order[a0:a0 + args.bs])
            x = (torch.from_numpy(Xc[r]).to(device).float() - m_t) / s_t
            doy = torch.from_numpy(doyc[r]).to(device)
            ev, tt = torch.from_numpy(evc[r]).to(device), torch.from_numpy(tc[r]).to(device)
            pe, pt = patch_events(evc[r], tc[r])
            ls_px, ls_pa, a = forward_parts(net, x, doy, device)
            nll = head_loss(net.px_head, ls_px.reshape(len(r), P, P, -1), ev, tt) \
                + args.patch_weight * head_loss(net.pa_head, ls_pa, torch.from_numpy(pe).to(device),
                                                torch.from_numpy(pt).to(device))
            ct, n_lab = cause_term(ls_px, a, ev, tt, torch.from_numpy(cc[r]).to(device), bool(args.use_spread))
            loss = nll + args.lam * ct / (len(r) * P * P)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            ema.update(net)
            sums["nll"] += nll.item() * len(r); sums["cause"] += float(ct); sums["n"] += len(r); sums["n_lab"] += n_lab
        h = {"epoch": ep, "train_nll": sums["nll"] / sums["n"],
             "cause_nll_per_label": sums["cause"] / max(sums["n_lab"], 1), "labels": sums["n_lab"]}
        hist.append(h)
        print(f"      Epoche {ep}: NLL {h['train_nll']:.5f}  Ursache -log pi je Label {h['cause_nll_per_label']:.4f} "
              f"({h['labels']:,} Labels, {time.time() - t_ep:.0f} s)", flush=True)
    state = ema.state(net.state_dict())
    ck_dir.mkdir(parents=True, exist_ok=True)
    saved = {"head": "contagion", "d": args.d, "heads": args.heads, "depth": args.depth, "attn": "factorized",
             "stem": "2d", "hist_days": 30, "rate_weight": 0.0, "cg_kernel": args.cg_kernel,
             "cg_obs_days": args.cg_obs_days, "cg_burn": args.cg_burn, "cg_blind": args.cg_blind,
             "cause": vars(args)}
    torch.save({"state": state, "args": saved, "history": hist}, ck)
    (ck_dir / (ck.stem + ".json")).write_text(json.dumps({"args": vars(args), "history": hist}, indent=1))
    print(f"[cause] Checkpoint {ck}", flush=True)


if __name__ == "__main__":
    main()

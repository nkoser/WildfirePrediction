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
from wildfire.models.bernstein import BernsteinSurvivalHead
from wildfire.paper import extra, targets
from wildfire.paper.channels import CHANNELS, masked_std
from wildfire.paper.channels import tag as channel_tag
from wildfire.paper.compare import DTS, KEYS
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics
from wildfire.paper.pixel_stdctm import pixel_nll, pixel_targets, windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm, to_device_once

T_WIN = 30
P = 13
CENTRE = 6
TAU_MAX = 100.0
DEGREE = 12


# ------------------------------------------------------------------ Modell
class Block(nn.Module):
    """Pre-Norm: Attention ueber den Raum, Attention ueber die Zeit, MLP."""

    def __init__(self, d: int, heads: int, p: float):
        super().__init__()
        self.ns, self.nt, self.nf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ta = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))

    def forward(self, x):                                   # (B, T, N, d)
        B, T, N, d = x.shape
        s = x.reshape(B * T, N, d)
        h = self.ns(s)
        s = s + self.sa(h, h, h, need_weights=False)[0]
        t = s.reshape(B, T, N, d).transpose(1, 2).reshape(B * N, T, d)
        h = self.nt(t)
        t = t + self.ta(h, h, h, need_weights=False)[0]
        x = t.reshape(B, N, T, d).transpose(1, 2)
        return x + self.ff(self.nf(x))


class STSDCTM(nn.Module):
    def __init__(self, c_in: int = 55, d: int = 96, heads: int = 4, depth: int = 4, p: float = 0.1,
                 init_cif_px: float = 0.02, init_cif_patch: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, d, 3, padding=1), nn.GroupNorm(8, d), nn.SiLU(),
            nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.SiLU())
        self.pos_s = nn.Parameter(torch.zeros(1, 1, P * P, d))
        self.pos_t = nn.Parameter(torch.zeros(1, T_WIN, 1, d))
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        self.season = nn.Linear(2, d)
        self.blocks = nn.ModuleList([Block(d, heads, p) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.read = nn.Linear(2 * d, d)
        self.px_head = BernsteinSurvivalHead(DEGREE, TAU_MAX, init_offset=-8.0, init_cif=init_cif_px)
        self.px_lin = nn.Linear(d, self.px_head.n_out)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(d)
        self.pa_head = BernsteinSurvivalHead(DEGREE, TAU_MAX, init_offset=-8.0, init_cif=init_cif_patch)
        self.pa_lin = nn.Linear(d, self.pa_head.n_out)
        for lin in (self.px_lin, self.pa_lin):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, x, doy):
        """x (B, 30, C, 13, 13) normiert, doy (B,) Kalendertag des Stichtags
        -> roh je Pixel (B, 13, 13, M+1), roh Ausschnitt (B, M+1)."""
        B, T, C, H, W = x.shape
        z = self.stem(x.flatten(0, 1))                       # (B*T, d, 13, 13)
        z = z.flatten(2).transpose(1, 2).reshape(B, T, H * W, -1)
        ang = 2 * math.pi * doy.float() / 365.25
        z = z + self.pos_s + self.pos_t[:, -T:] + self.season(torch.stack([ang.sin(), ang.cos()], -1))[:, None, None]
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        pix = self.read(torch.cat([z[:, -1], z.mean(1)], -1))  # (B, 169, d)
        raw_px = self.px_lin(pix).view(B, H, W, -1)
        a = self.pool(self.cls.expand(B, -1, -1), pix, pix, need_weights=False)[0][:, 0]
        raw_pa = self.pa_lin(self.pool_norm(a))
        return raw_px.float(), raw_pa.float()


# ------------------------------------------------------------------ Hilfen
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
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patch-weight", type=float, default=1.0)
    ap.add_argument("--limit-eval", type=int, default=0, help="nur fuer Rauchtests: so viele Val/Test-Stichproben")
    ap.add_argument("--cube-kind", choices=["all", "random", "fire"], default="all",
                    help="Wuerfel-Stichtage: alle, nur zufaellige (echte Haeufigkeit) oder nur an Feuer verankerte")
    ap.add_argument("--select", choices=["prauc", "survnll"], default="prauc",
                    help="Modellauswahl: PR-AUC auf den Paper-Val-Stichproben oder Survival-NLL auf "
                         "zufaelligen Val-Stichtagen (surv_eval, 2020-2022)")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
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
    if args.limit_eval:
        idx["val"], idx["test"] = idx["val"][:args.limit_eval], idx["test"][:args.limit_eval]
    fire = X[:, :, FIRE_CH] > 0
    if targets.shift(args.target):
        fire = np.concatenate([fire, fire_target[:, None]], 1)   # Zieltag D+1 als Tag L - 1
    Y = fire[:, -1]
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    std = masked_std(std, args.channels)
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    E_LO, E_HI = T_WIN - 1, L - 2
    print(f"[sts] {args.cache} Ziel {args.target}, Zusatz {args.extra}: X {X.shape}, Train {idx['train'].size}, "
          f"Val {idx['val'].size}, Test {idx['test'].size} ({time.time() - t0:.0f} s)", flush=True)

    def to_dev(xw):
        xw = xw if isinstance(xw, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(xw))
        return (xw.to(device).float() - m_t) / s_t

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
    else:
        ev0, _ = pixel_targets(fire[idx["train"]], np.full(idx["train"].size, E_LO))
        init_px, init_pa = float(ev0.mean()), float(ev0.reshape(len(ev0), -1).any(1).mean())

    # ---- Auswertung im Protokoll des Papers
    @torch.no_grad()
    def paper_scores(net, rows, dt_, bs=64):
        """Stichtag e = Zieltag - dt - 1, Score F(dt + 1): Karte (n, 13, 13) und Ausschnittskopf (n,)."""
        net.eval()
        e = L - dt_ - 2
        k = torch.tensor([float(dt_ + 1)], device=device)
        maps, heads = [], []
        for a in range(0, rows.size, bs):
            r = rows[a:a + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw_px, raw_pa = net(to_dev(windows(Xg, r, np.full(r.size, e))),
                                     torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device))
            fp = -torch.expm1(net.px_head.log_survival(raw_px.reshape(-1, raw_px.shape[-1]), k))[:, 0]
            fa = -torch.expm1(net.pa_head.log_survival(raw_pa, k))[:, 0]
            maps.append(fp.view(len(r), P, P).cpu().numpy()); heads.append(fa.cpu().numpy())
        return np.clip(np.concatenate(maps), 1e-7, 1 - 1e-7), np.clip(np.concatenate(heads), 1e-7, 1 - 1e-7)

    if args.select == "survnll":
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
                ll = head.log_likelihood(raw, torch.from_numpy(e).to(device),
                                         torch.from_numpy(t.astype(np.int64)).to(device), TAU_MAX)
                tot += float(ll.sum())
        return tot / n

    def val_rank(net) -> float:
        va, sc = idx["val"], []
        for dt_ in DTS:
            m, h = paper_scores(net, va, dt_)
            sc.append(average_precision_score(Y[va, CENTRE, CENTRE], m[:, CENTRE, CENTRE]))
            sc.append(average_precision_score(y[va], h))
        return float(np.mean(sc))

    val_fn = val_surv if args.select == "survnll" else val_rank

    # ---- Training je Seed
    te = idx["test"]
    res_seed = []
    hists = []
    ckpt_dir = exp / "checkpoints" / "paper"
    for seed in args.seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = STSDCTM(d=args.d, heads=args.heads, depth=args.depth, p=args.dropout,
                      init_cif_px=float(np.clip(init_px, 1e-4, 0.5)),
                      init_cif_patch=float(np.clip(init_pa, 1e-3, 0.9))).to(device)
        if args.init_from:
            net.load_state_dict(torch.load(args.init_from.format(seed=seed), map_location="cpu")["state"])
            print(f"[sts] Start von {args.init_from.format(seed=seed)}", flush=True)
        n_par = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
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
                rows = rng.permutation(Xc.shape[0])
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
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw_px, raw_pa = net(to_dev(xw), torch.from_numpy(np.asarray(doy, np.float32)).to(device))
                loss = pixel_nll(net.px_head, raw_px, torch.from_numpy(ev).to(device), torch.from_numpy(tt).to(device)) \
                    + args.patch_weight * pixel_nll(net.pa_head, raw_pa, torch.from_numpy(pe).to(device),
                                                    torch.from_numpy(pt).to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step()
                tl += loss.item() * len(r); tn += len(r)
            vr = val_fn(net)
            hist.append({"epoch": ep, "train_nll": tl / tn, "val_prauc_mean": vr})
            print(f"      Epoche {ep:2d}: train NLL {tl / tn:.5f}  val {args.select} {vr:.4f}  "
                  f"({time.time() - t_ep:.0f} s)", flush=True)
            if vr > best + 1e-5:
                best, bad, state = vr, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        net.load_state_dict(state)
        hists.append(hist)
        if args.save_ckpt:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cp = ckpt_dir / f"sts_{args.cache}{targets.tag(args.target)}{extra.tag(args.extra)}_{args.train_source}{args.tag}_s{seed}.pt"
            torch.save({"state": state, "args": vars(args)}, cp)
            print(f"[sts] Checkpoint {cp}", flush=True)
        out = {}
        for dt_ in DTS:
            m, h = paper_scores(net, te, dt_)
            out[str(dt_)] = {
                "centre": paper_metrics(m[:, CENTRE, CENTRE], Y[te, CENTRE, CENTRE].astype(int)),
                "pixel": paper_metrics(m.reshape(-1), Y[te].reshape(-1).astype(int)),
                "patch_head": paper_metrics(h, y[te]),
                "patch_max": paper_metrics(m.max(axis=(1, 2)), y[te]),
            }
        res_seed.append(out)

    res = {str(dt_): {ev: {"summary": summarize([r[str(dt_)][ev] for r in res_seed]),
                           "per_seed": [r[str(dt_)][ev] for r in res_seed]}
                      for ev in ("centre", "pixel", "patch_head", "patch_max")} for dt_ in DTS}
    for dt_ in DTS:
        print(f"[sts] TEST dt={dt_:2d}  " + "   ".join(
            f"{ev}: PR-AUC {res[str(dt_)][ev]['summary']['pr_auc']['mean']:.3f}"
            f"±{res[str(dt_)][ev]['summary']['pr_auc']['std']:.3f}" for ev in res[str(dt_)]), flush=True)
    write_manifest(cfg, name, args.config, {
        "cache": args.cache, "target": args.target, "extra": args.extra, "channels": args.channels,
        "train_source": args.train_source, "init_from": args.init_from, "seeds": args.seeds,
        "model": {"d": args.d, "depth": args.depth, "heads": args.heads, "dropout": args.dropout,
                  "dctm": f"Bernstein Grad {DEGREE} auf [0, {TAU_MAX:.0f}]", "patch_weight": args.patch_weight},
        "training": {"bs": args.bs, "lr": args.lr, "wd": args.wd, "warmup": args.warmup, "epochs": args.epochs,
                     "steps_per_epoch": args.steps_per_epoch, "patience": args.patience, "selection": args.select, "cube_kind": args.cube_kind},
        "score": "F(dt + 1) bei Stichtag e = Zieltag - dt - 1",
        "metric_assumptions": ASSUMPTIONS, "histories": hists, "results": res,
    })


if __name__ == "__main__":
    main()

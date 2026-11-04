"""Tages-Transformer (GPT-artig): ein Token je Tag, sagt autoregressiv die Feuerkarte des naechsten Tages voraus.

Kontext: die 30 Eingabetage des Stichtags L, je Tag ein Token aus allen 55 Kanaelen (13 x 13, CNN-Encoder).
Danach je Folgetag ein Token aus der Feuerkarte dieses Tages (Wetter der Folgetage ist unbekannt, nur
Feuer + Tag im Jahr). Kausale Aufmerksamkeit; das Token von Tag L+k-1 sagt die Karte von Tag L+k voraus
(169 Pixel-Logits + 1 Logit "Feuer irgendwo im Ausschnitt"). Ausrollen: die vorhergesagte Wahrscheinlichkeitskarte
wird als naechstes Token zurueckgefuettert (Mean Field).

Training auf den Wuerfel-Stichtagen von paper_pixel (2000-2019, wie sts_spread*), Ziele aus future_maps.py
(Tage L+1 .. L+31). --mode:
  tf    Teacher Forcing: Eingabe sind die beobachteten Karten der Folgetage (ein Durchlauf)
  ss    Scheduled Sampling: je Folgetag mit Wahrscheinlichkeit --ss-prob die eigene Vorhersage
        (aus einem TF-Durchlauf ohne Gradient) statt der beobachteten Karte
  free  freies Ausrollen mit Gradient (wie in der Auswertung; 31 Durchlaeufe je Schritt)
Verlust: BCE je Pixel und Tag + BCE Ausschnitt je Tag, Tage ausserhalb des Wuerfels maskiert.

Auswertung (Test-Cache cache_G_L61, wie alle Paper-Modelle):
  paper  Paper-Protokoll: Eingabe endet bei e = D - dt, Ausrollen dt + 1 Tage, Score fuer Tag D+1:
         Mitte = Pixel (6, 6), Ausschnitt = Ausschnitts-Logit (und 1 - prod(1 - p) als Nebenwert)
  lead   je Vorlauf k = 1..31 ab e = D - 30: Pixel-PR-AUC, Ausschnitts-PR-AUC, vorhergesagte vs. beobachtete
         Feuerpixel je Ausschnitt; Vergleich Persistenz (Karte von Tag e) und Feuer der letzten 7 Tage
Ausgaben NEU: checkpoints/paper/fire_gpt<tag>_s<seed>.pt, predictions/fire_gpt<tag>_s<seed>.npz
(Lead-Wahrscheinlichkeiten Test, float16), Manifest p60_fire_gpt<tag>.
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
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.models.ema import EMA
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import CENTRE, P, doy_of, lr_factor

T_CTX = 30
N_FUT = 31


class DayEnc(nn.Module):
    """(N, c, 13, 13) -> (N, d)."""

    def __init__(self, c: int, d: int, w: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, w, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(w, w, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(),
                                 nn.Linear(w * 7 * 7, d))

    def forward(self, x):
        return self.net(x)


class FireGPT(nn.Module):
    def __init__(self, c_in: int = 55, d: int = 256, depth: int = 6, heads: int = 8, p: float = 0.1,
                 init_px: float = 1e-3, init_pa: float = 0.05):
        super().__init__()
        self.ctx_enc = DayEnc(c_in, d)
        self.fut_enc = DayEnc(1, d)
        self.pos = nn.Parameter(torch.zeros(1, T_CTX + N_FUT - 1, d))
        nn.init.normal_(self.pos, std=0.02)
        self.doy = nn.Linear(4, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, p, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, P * P + 1)
        with torch.no_grad():
            self.head.bias[:P * P] = math.log(init_px / (1 - init_px))
            self.head.bias[P * P] = math.log(init_pa / (1 - init_pa))

    def encode_ctx(self, x):
        """x (B, 30, 55, 13, 13) normiert -> (B, 30, d)."""
        B, T = x.shape[:2]
        return self.ctx_enc(x.flatten(0, 1)).view(B, T, -1)

    def step(self, ctx, doy0, fut):
        """ctx (B, 30, d), doy0 (B,) Tag im Jahr von L, fut (B, k, 13, 13) Karten L+1..L+k (k >= 0)
        -> px (B, k+1, 13, 13), pa (B, k+1): Logits fuer die Tage L+1 .. L+k+1."""
        B, k = ctx.shape[0], fut.shape[1]
        tok = ctx if k == 0 else torch.cat([ctx, self.fut_enc(fut.flatten(0, 1)[:, None].to(ctx.dtype)).view(B, k, -1)], 1)
        n = tok.shape[1]
        day = doy0[:, None].float() + torch.arange(n, device=ctx.device)[None] - (T_CTX - 1)
        ang = 2 * math.pi * day / 365.25
        rel = (torch.arange(n, device=ctx.device) - (T_CTX - 1)).clamp(min=0).float() / N_FUT
        feat = torch.stack([ang.sin(), ang.cos(), rel.expand(B, n), (rel > 0).float().expand(B, n)], -1)
        h = tok + self.pos[:, :n] + self.doy(feat.to(tok.dtype))
        mask = nn.Transformer.generate_square_subsequent_mask(n, device=ctx.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        o = self.head(self.norm(h[:, T_CTX - 1:]))
        return o[..., :P * P].reshape(B, k + 1, P, P), o[..., P * P]

    def forward(self, x, doy0, fut):
        return self.step(self.encode_ctx(x), doy0, fut)

    def rollout(self, x, doy0, steps: int):
        """Mean-Field-Ausrollen ohne beobachtete Folgetage -> Logits px (B, steps, 13, 13), pa (B, steps)."""
        ctx = self.encode_ctx(x)
        fut = ctx.new_zeros(x.shape[0], 0, P, P)
        pxs, pas = [], []
        for _ in range(steps):
            px, pa = self.step(ctx, doy0, fut)
            pxs.append(px[:, -1]); pas.append(pa[:, -1])
            fut = torch.cat([fut, torch.sigmoid(px[:, -1:].float()).to(fut.dtype)], 1)
        return torch.stack(pxs, 1), torch.stack(pas, 1)


def gpt_loss(px, pa, y, ok):
    """px (B, F, 13, 13), pa (B, F) Logits; y (B, F, 13, 13) 0/1; ok (B, F) gueltige Tage."""
    w = ok.float()
    lp = F.binary_cross_entropy_with_logits(px.float(), y, reduction="none").mean((2, 3))
    la = F.binary_cross_entropy_with_logits(pa.float(), y.flatten(2).amax(2), reduction="none")
    return (lp * w).sum() / w.sum().clamp_min(1), (la * w).sum() / w.sum().clamp_min(1)


def load_gpt(path: str, device) -> FireGPT:
    c = torch.load(path, map_location="cpu")
    a = c["args"]
    net = FireGPT(d=a["d"], depth=a["depth"], heads=a["heads"], p=a["dropout"], init_px=c["init_px"], init_pa=c["init_pa"])
    net.load_state_dict(c["state"])
    return net.to(device).eval()


def ap_(y, s) -> float:
    y = np.asarray(y).ravel().astype(int)
    return float(average_precision_score(y, np.asarray(s, np.float64).ravel())) if y.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["tf", "ss", "free"], default="tf")
    ap.add_argument("--ss-prob", type=float, default=0.5, help="nur --mode ss: Anteil eigener Vorhersagen als Eingabe")
    ap.add_argument("--patch-weight", type=float, default=1.0, help="Gewicht des Ausschnitts-BCE")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--fut-dir", default="paper_pixel_future31", help="derived_data/<fut-dir> aus future_maps.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-ckpt", default="", help="nur auswerten: glob vorhandener Checkpoints")
    ap.add_argument("--limit-train", type=int, default=0, help="nur fuer Rauchtests")
    ap.add_argument("--limit-eval", type=int, default=0, help="nur fuer Rauchtests")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"fire_gpt{args.tag}"
    ckpt_dir = exp / "checkpoints" / "paper"
    pred_dir = exp / "predictions"
    if not args.eval_ckpt and list(ckpt_dir.glob(f"{name}_s*.pt")):
        raise SystemExit(f"ABBRUCH: Checkpoints {name}_s*.pt existieren schon - anderen --tag waehlen")
    t0 = time.time()
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")  # noqa: E731

    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    def to_dev(xw):
        xw = xw if isinstance(xw, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(xw))
        return (xw.to(device).float() - m_t) / s_t

    # ---- Auswertung: Test-Cache
    cd = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    Xp = np.load(cd / "X.npy", mmap_mode="r")
    LC = Xp.shape[1]
    split = np.load(cd / "split.npy")
    y_pa = targets.labels(cd, "next_patch")
    fnext = targets.fire_map(cd, Xp, "next_patch")                       # (N, 13, 13) Tag D+1
    y_px = fnext[:, CENTRE, CENTRE].astype(int)
    doyD = doy_of(np.load(cd / "meta.npy")[:, 0])
    te_c = np.where(split == "test")[0]
    if args.limit_eval:
        te_c = te_c[:args.limit_eval]
    Xg = to_device_once(np.asarray(Xp[te_c]) if args.limit_eval else np.asarray(Xp), device, "cache_X")
    gi = {int(r): i for i, r in enumerate(te_c)} if args.limit_eval else None
    print(f"[gpt] Paper-Test {te_c.size:,} ({time.time() - t0:.0f} s)", flush=True)

    @torch.no_grad()
    def paper_eval(net, bs=128) -> dict:
        net.eval()
        out = {}
        for dt_ in DTS:
            e = LC - 1 - dt_
            fc, fa, fu = [], [], []
            for a in range(0, te_c.size, bs):
                r = te_c[a:a + bs]
                rg = np.array([gi[int(v)] for v in r]) if gi is not None else r
                with autocast():
                    px, pa = net.rollout(to_dev(windows(Xg, rg, np.full(r.size, e))),
                                         torch.from_numpy(doyD[r] - dt_).to(device), dt_ + 1)
                p = torch.sigmoid(px[:, -1].float())
                fc.append(p[:, CENTRE, CENTRE].cpu().numpy())
                fa.append(torch.sigmoid(pa[:, -1].float()).cpu().numpy())
                fu.append((1 - torch.prod(1 - p.flatten(1).double(), 1)).cpu().numpy())
            cl = lambda v: np.clip(np.concatenate(v), 1e-7, 1 - 1e-7)  # noqa: E731
            out[str(dt_)] = {"centre": paper_metrics(cl(fc), y_px[te_c]), "patch_head": paper_metrics(cl(fa), y_pa[te_c]),
                             "patch_union": paper_metrics(cl(fu), y_pa[te_c])}
        return out

    @torch.no_grad()
    def lead_eval(net, bs=128):
        """Ab e = D - 30 je Vorlauf 1..31 -> (Metriken, Wahrscheinlichkeiten (n, 31, 13, 13) float16)."""
        net.eval()
        e = LC - 1 - 30
        probs, pas, obs, last, last7 = [], [], [], [], []
        for a in range(0, te_c.size, bs):
            r = te_c[a:a + bs]
            rg = np.array([gi[int(v)] for v in r]) if gi is not None else r
            w = windows(Xg, rg, np.full(r.size, e))
            with autocast():
                px, pa = net.rollout(to_dev(w), torch.from_numpy(doyD[r] - 30).to(device), N_FUT)
            probs.append(torch.sigmoid(px.float()).cpu().numpy().astype(np.float16))
            pas.append(torch.sigmoid(pa.float()).cpu().numpy())
            f = w[:, :, FIRE_CH]
            last.append((f[:, -1] > 0).cpu().numpy() if isinstance(f, torch.Tensor) else f[:, -1] > 0)
            last7.append(((f[:, -7:] > 0).float().mean(1)).cpu().numpy() if isinstance(f, torch.Tensor)
                         else (f[:, -7:] > 0).mean(1))
            fut = Xg[torch.from_numpy(rg).to(Xg.device) if isinstance(Xg, torch.Tensor) else rg, e + 1:, FIRE_CH]
            fut = (fut > 0).cpu().numpy() if isinstance(fut, torch.Tensor) else fut > 0
            obs.append(np.concatenate([fut, fnext[r][:, None]], 1))
        probs, pas, obs = np.concatenate(probs), np.concatenate(pas), np.concatenate(obs)
        last, last7 = np.concatenate(last), np.concatenate(last7)
        res = []
        for k in range(N_FUT):
            pk = probs[:, k].astype(np.float32)
            res.append({"lead": k + 1, "pixel_ap": ap_(obs[:, k], pk), "pixel_ap_persist": ap_(obs[:, k], last),
                        "pixel_ap_last7": ap_(obs[:, k], last7),
                        "patch_ap": ap_(obs[:, k].reshape(len(obs), -1).any(1), pas[:, k]),
                        "pred_px": float(pk.sum((1, 2)).mean()), "obs_px": float(obs[:, k].sum((1, 2)).mean()),
                        "rate_obs": float(obs[:, k].mean())})
        return res, probs, obs

    # ---- Training bzw. Laden
    nets, hists = [], []
    if args.eval_ckpt:
        for f in sorted(glob.glob(args.eval_ckpt)):
            nets.append((Path(f).stem.split("_")[-1], load_gpt(f, device)))
            print(f"[gpt] geladen: {f}", flush=True)
        if not nets:
            raise SystemExit(f"ABBRUCH: keine Checkpoints fuer {args.eval_ckpt}")
    else:
        pdir = exp / "derived_data" / args.cube_dir
        fdir = exp / "derived_data" / args.fut_dir
        n_full = np.load(pdir / "train_event.npy", mmap_mode="r").shape[0]
        n_tr = args.limit_train or n_full
        print(f"[gpt] lade {n_tr:,} Trainingsfenster in den RAM ...", flush=True)
        t1 = time.time()
        Xc = np.asarray(np.load(pdir / "train_X.npy", mmap_mode="r")[:n_tr])
        fut_p = np.asarray(np.load(fdir / "fut.npy", mmap_mode="r")[:n_tr])
        fut_ok = np.load(fdir / "fut_ok.npy")[:n_tr]
        dates = CubeReader(exp).dates
        Lm = np.load(pdir / "train_meta.npz")["L"][:n_tr]
        doyc = np.array([dt.date.fromisoformat(dates[int(L)]).timetuple().tm_yday for L in Lm], np.float32)
        smp = np.unpackbits(fut_p[:20000], axis=1)[:, :N_FUT]
        init_px = float(smp.mean())
        init_pa = float(smp.reshape(len(smp), N_FUT, -1).any(2).mean())
        print(f"[gpt] Trainingsfenster {Xc.shape}, Folgetage {fut_p.shape} in {time.time() - t1:.0f} s; "
              f"Feuerrate Pixel {init_px:.5f}, Ausschnitt {init_pa:.4f}, alle Tage gueltig {fut_ok.all(1).mean():.3f}",
              flush=True)
        for seed in args.seeds:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = FireGPT(d=args.d, depth=args.depth, heads=args.heads, p=args.dropout,
                          init_px=init_px, init_pa=init_pa).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
            total = args.epochs * args.steps_per_epoch
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.warmup, total))
            ema = EMA(net, args.ema) if args.ema > 0 else None
            print(f"[gpt] Seed {seed}: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} Mio. Parameter, "
                  f"Modus {args.mode}", flush=True)
            hist = []
            for ep in range(args.epochs):
                net.train()
                t_ep = time.time()
                rows = rng.permutation(n_tr)
                rows = np.resize(rows, args.steps_per_epoch * args.bs)
                sp = sa = sn = 0.0
                for a in range(0, rows.size, args.bs):
                    r = np.sort(rows[a:a + args.bs])
                    x = to_dev(Xc[r])
                    doy = torch.from_numpy(doyc[r]).to(device)
                    y = torch.from_numpy(np.unpackbits(fut_p[r], axis=1)[:, :N_FUT]).to(device).float()
                    ok = torch.from_numpy(fut_ok[r]).to(device)
                    with autocast():
                        if args.mode == "free":
                            px, pa = net.rollout(x, doy, N_FUT)
                        else:
                            ctx = net.encode_ctx(x)
                            inp = y[:, :N_FUT - 1]
                            if args.mode == "ss":
                                with torch.no_grad():
                                    p0, _ = net.step(ctx.detach(), doy, inp)
                                    own = torch.sigmoid(p0[:, :N_FUT - 1].float())
                                sel = torch.rand(len(r), N_FUT - 1, 1, 1, device=device) < args.ss_prob
                                inp = torch.where(sel, own, inp)
                            px, pa = net.step(ctx, doy, inp)
                    lp, la = gpt_loss(px, pa, y, ok)
                    loss = lp + args.patch_weight * la
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step(); sched.step()
                    if ema is not None:
                        ema.update(net)
                    sp += lp.item() * len(r); sa += la.item() * len(r); sn += len(r)
                hist.append({"epoch": ep, "bce_px": sp / sn, "bce_patch": sa / sn})
                print(f"      Epoche {ep:2d}: BCE Pixel {sp / sn:.5f}  Ausschnitt {sa / sn:.5f}  ({time.time() - t_ep:.0f} s)",
                      flush=True)
            if ema is not None:
                net.load_state_dict(ema.state({k: v.detach().clone() for k, v in net.state_dict().items()}))
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cp = ckpt_dir / f"{name}_s{seed}.pt"
            torch.save({"state": state, "args": vars(args), "init_px": init_px, "init_pa": init_pa}, cp)
            print(f"[gpt] Checkpoint {cp}", flush=True)
            nets.append((f"s{seed}", net.eval()))
            hists.append(hist)
        del Xc

    # ---- Auswertung
    pred_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for sname, net in nets:
        t1 = time.time()
        pap = paper_eval(net)
        print("[gpt]   Paper-Protokoll PR-AUC Mitte/Ausschnitt/Vereinigung: " + "  ".join(
            f"dt {k}: {v['centre']['pr_auc']:.3f}/{v['patch_head']['pr_auc']:.3f}/{v['patch_union']['pr_auc']:.3f}"
            for k, v in pap.items()), flush=True)
        lead, probs, obs = lead_eval(net)
        for m in lead:
            if m["lead"] in (1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 31):
                print(f"[gpt]   Vorlauf {m['lead']:2d}: Pixel-AP {m['pixel_ap']:.3f} (Persistenz {m['pixel_ap_persist']:.3f}, "
                      f"7 Tage {m['pixel_ap_last7']:.3f})  Ausschnitt-AP {m['patch_ap']:.3f}  "
                      f"Feuerpixel vorhergesagt {m['pred_px']:.2f} / beobachtet {m['obs_px']:.2f}", flush=True)
        np.savez_compressed(pred_dir / f"{name}_{sname}.npz", probs=probs, obs=np.packbits(obs, axis=1), rows=te_c)
        res[sname] = {"paper": pap, "lead": lead}
        print(f"[gpt] {sname}: Auswertung in {time.time() - t1:.0f} s", flush=True)
    write_manifest(cfg, f"p60_{name}", args.config, {
        "mode": args.mode, "ss_prob": args.ss_prob, "patch_weight": args.patch_weight, "eval_ckpt": args.eval_ckpt,
        "seeds": args.seeds, "fut_dir": args.fut_dir,
        "training": {k: getattr(args, k) for k in ("epochs", "steps_per_epoch", "bs", "lr", "wd", "warmup", "ema",
                                                    "d", "depth", "heads", "dropout")},
        "n": {"paper_test": int(te_c.size)}, "histories": hists, "results": res,
        "paper_protocol": "Ziel D+1, Eingabe endet e = D - dt, Ausrollen dt + 1 Tage; Mitte = Pixel (6, 6)",
        "lead_protocol": "Eingabe endet e = D - 30, Vorlauf k = 1..31 (k = 31: Tag D+1 aus fire_next)",
    })


if __name__ == "__main__":
    main()

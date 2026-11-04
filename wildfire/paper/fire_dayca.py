"""Tag-fuer-Tag-Ausbreitung: Transformer ueber 30 Kontexttage setzt jeden Tag die Parameter eines CA-Schritts.

Idee (04.11.2026): Die Parameter der Ausbreitung kommen nicht einmal am Stichtag, sondern jeden Tag neu aus dem Kontext
der letzten 30 Tage (Index t-29 .. t); darin steckt auch die schon gelaufene Ausbreitung (bis L beobachtet, danach der
eigene gezogene Zustand). Ein symbolischer CA-Schritt (wie ca_lin in symbolic_spread) rechnet den Tag aus:

  Kontext (Tag k = t-29 .. t): Feuer_k, neu_k, verbrannt_k, Wetter_k (25), Statik (Gelaende, NDVI, Landbedeckung)
     -> Tages-CNN (Stride 4) -> Token je Patch und Tag + Tagesindex -> Transformer ueber die Zeit je Patch
     -> raeumliche Faltungen -> hochgerechnet + Feuer/neu/verbrannt am Tag t
     -> je Pixel: log q0, Richtung w (2), Zuendung a, Fortbestehen rho
  CA-Schritt t -> t+1: q_io = exp(log q0_i + c_r (r - 1) + w_i . u_o) (5 x 5), Druck P_i = sum_o q_io B_{i+o},
     3 Teilschritte, h = 1 - exp(-P - a/3); brennende Zellen brennen mit sigmoid(rho_i) weiter.

Training: Teacher Forcing, beobachteter Kontext bis Tag d, Ziel Zuendung an d + 1 (BCE auf unverbrannten Landpixeln)
  + Fortbestehen (BCE), alle 30 Zieltage eines Ereignisses als Batch; Wetter bis d bekannt.
  --ss: Scheduled Sampling gegen den Exposure Bias (ss_loss: 1..K eigene gezogene Tage im Kontext vor dem Ziel).
Vorhersage: Monte-Carlo-Rollout ab L, je Pfad eigener Kontext; Wetter nach L = Wetter am Tag L (kein Oracle),
  mit --oracle-weather das beobachtete Wetter der Folgetage (obere Schranke; --eval-only nimmt einen fertigen Checkpoint).
Ereignisse und Metriken wie symbolic_spread / firemap_bench (train 120, val 14, test 60). Von Grund auf trainiert;
mit --ema laufen Auswahl (val-Verlust je Epoche) und Auswertung mit den EMA-Gewichten.
Ausgaben NEU: metrics/fire_dayca<tag>/ (runs/<split>_dayca.json, ckpt, maps/<split>_dayca.npz, history.json).
"""

from __future__ import annotations

import argparse
import copy
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
from wildfire.models.ema import EMA
from wildfire.paper.firemap_bench import HOR, agg, load_box, map_metrics
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.surv_prepare import T_WIN
from wildfire.paper.symbolic_spread import R_CA, ca_geometry, load_event, to_dev

T = 30
WX = list(range(1, 26))
STAT = [30, 32, 36, 37, 38, 40]                     # Worldpop, Hang, TPI, TWI, NDVI, LAI
DEM, LC, N_LC = 31, 54, 18
C_DAY = 3 + len(WX)                                  # Feuer, neu, verbrannt, Wetter
C_STAT = 2 + len(STAT) + N_LC                        # grad DEM, Statik, Landbedeckung


def load_seq(cube, e, mean, std, base: dict) -> dict:
    """Ereignis aus symbolic_spread.load_event plus Wetter aller 60 Tage (L-29 .. L+30) und Statik je Pixel."""
    L, (r0, r1, c0, c1) = e["L"], e["box"]
    slab = load_box(cube, L + T, r0, r1, c0, c1, 0, days=T_WIN + T)              # (60, 55, h, w)
    x = (slab.astype(np.float32) - mean[None, :, None, None]) / std[None, :, None, None]
    gr, gc = np.gradient(x[0, DEM])
    lc = np.clip(slab[0, LC], 0, N_LC - 1)
    stat = np.concatenate([np.stack([gr, gc]), x[0, STAT], np.eye(N_LC, dtype=np.float32)[lc].transpose(2, 0, 1)], 0)
    out = dict(base)
    out["wx"] = np.clip(x[:, WX], -5, 5).astype(np.float16)                         # float16 auf der CPU, je Schritt auf die GPU
    out["stat"] = np.clip(stat, -5, 5).astype(np.float32)
    return out


class DayCA(nn.Module):
    def __init__(self, d: int = 96, layers: int = 2, heads: int = 4, ctx: int = 30, substeps: int = 3):
        super().__init__()
        self.ctx, self.M = ctx, substeps
        self.enc = nn.Sequential(nn.Conv2d(C_DAY + C_STAT, 64, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
                                 nn.Conv2d(64, d, 3, stride=2, padding=1))
        self.pos = nn.Parameter(torch.zeros(ctx, d))                                # Tagesindex im Kontext
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True,
                                                                   norm_first=True), layers)
        self.mix = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(d + 3 + C_STAT, 64, 1), nn.GELU(), nn.Conv2d(64, 5, 1))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([math.log(0.05), 0.0, 0.0, math.log(1e-4), math.log(0.6 / 0.4)]))
        self.c_r = nn.Parameter(torch.tensor(-0.7))

    @staticmethod
    def pad4(x):
        h, w = x.shape[-2:]
        return F.pad(x, (0, (-w) % 4, 0, (-h) % 4))

    def encode_days(self, day, stat):
        """day (B, C_DAY, h, w), stat (C_STAT, h, w) -> Token (B, d, h/4, w/4)."""
        x = torch.cat([day, stat[None].expand(day.shape[0], -1, -1, -1)], 1)
        return self.enc(self.pad4(x))

    def params(self, tokens, cur, stat, shape):
        """tokens (B, K, d, H4, W4) Kontext (aeltester zuerst), cur (B, 3, h, w) Feuer/neu/verbrannt am Tag t."""
        B, K, d, H4, W4 = tokens.shape
        seq = tokens.permute(0, 3, 4, 1, 2).reshape(B * H4 * W4, K, d) + self.pos[-K:]
        z = self.tf(seq)[:, -1].view(B, H4, W4, d).permute(0, 3, 1, 2)
        z = self.mix(z)
        h, w = shape
        z = F.interpolate(z, scale_factor=4, mode="bilinear", align_corners=False)[..., :h, :w]
        out = self.head(torch.cat([z, cur, stat[None].expand(B, -1, -1, -1)], 1))   # (B, 5, h, w)
        return out.flatten(2).float()

    def kernel(self, p, geo):
        """p (B, 5, N) -> q (B, 25, N)."""
        _, r, u = geo
        lq = p[:, 0:1] + self.c_r * (r - 1).clamp_min(0)[None, :, None] + torch.einsum("ok,bkn->bon", u, p[:, 1:3])
        return lq.clamp(max=3.0).exp() * (r > 0).float()[None, :, None]

    def step_prob(self, q, a, B0, fresh, shape):
        h, w = shape
        Fp = torch.zeros_like(B0)
        for _ in range(self.M):
            U = F.unfold((B0 + Fp).view(-1, 1, h, w), 2 * R_CA + 1, padding=R_CA)
            hz = -torch.expm1(-(q * U).sum(1) - a / self.M)
            Fp = Fp + (1 - Fp) * hz * fresh
        return Fp


def bce(p, y, w=None):
    """BCE in float32 (F.binary_cross_entropy ist unter autocast nicht erlaubt)."""
    p, y = p.float().clamp(1e-6, 1 - 1e-6), y.float()
    l = -(y * p.log() + (1 - y) * (1 - p).log())
    return (l * w).sum() / w.sum().clamp_min(1) if w is not None else l.mean()


def to_gpu(ev, device):
    return {k: (v.to(device).float() if k == "wx" else v.to(device)) if torch.is_tensor(v) else v for k, v in ev.items()}


def day_features(fire, burnt_prev, wx):
    """fire (B, h, w) bool, burnt_prev (B, h, w) bool (Feuer an einem Tag davor), wx (B, 25, h, w) -> (B, C_DAY, h, w)."""
    new = fire & ~burnt_prev
    burnt = fire | burnt_prev
    return torch.cat([fire[:, None].float(), new[:, None].float(), burnt[:, None].float(), wx], 1)


def observed_days(ev):
    """Tagesmerkmale aller 60 beobachteten Tage (60, C_DAY, h, w) und verbrannt bis einschliesslich Tag (60, h, w)."""
    fire = ev["fire"]
    burnt = fire.cumsum(0) > 0
    prev = torch.cat([torch.zeros_like(burnt[:1]), burnt[:-1]], 0)
    return day_features(fire, prev, ev["wx"]), burnt


def tf_loss(net, ev, geo):
    shape = ev["land"].shape
    feats, burnt = observed_days(ev)
    tok = net.encode_days(feats, ev["stat"])                                         # (60, d, H4, W4)
    d = torch.arange(T_WIN - 1, T_WIN - 1 + T, device=tok.device)                    # Quelltage L .. L+29
    K = net.ctx
    idx = d[:, None] - torch.arange(K - 1, -1, -1, device=tok.device)[None]          # (30, K)
    p = net.params(tok[idx], feats[d, :3], ev["stat"], shape)                        # (30, 5, N)
    with torch.autocast(tok.device.type, enabled=False):
        q = net.kernel(p, geo).float()
    fire = ev["fire"].flatten(1)
    fresh = (ev["land"].flatten()[None] & ~burnt.flatten(1)[d]).float()
    with torch.autocast(tok.device.type, enabled=False):
        Fp = net.step_prob(q.float(), p[:, 3].exp(), fire[d].float(), fresh, shape)
    y = fire[d + 1].float()
    l_tf = bce(Fp, y, fresh)
    on = fire[d]
    pc = torch.sigmoid(p[:, 4])
    l_c = bce(pc[on], y[on]) if on.any() else pc.sum() * 0
    return l_tf + 0.2 * l_c, {"tf": l_tf.item(), "cont": l_c.item()}


def sample_day(net, p, q, burning, burnt, land, shape, gen):
    """Ein gezogener CA-Tag fuer (n, h, w)-Zustaende -> neue Zellen nd, burning, burnt (vorher)."""
    h, w = shape
    n = burning.shape[0]
    a = p[:, 3].exp()
    B = burning.flatten(1).float()
    new_day = torch.zeros_like(B, dtype=torch.bool)
    for _ in range(net.M):
        U = F.unfold(B.view(-1, 1, h, w), 2 * R_CA + 1, padding=R_CA)
        hz = -torch.expm1(-(q * U).sum(1) - a / net.M)
        fresh = (land & ~burnt & ~burning).flatten(1) & ~new_day
        new = fresh & (torch.rand(B.shape, generator=gen, device=B.device) < hz)
        new_day |= new
        B = torch.maximum(B, new.float())
    nd = new_day.view(n, h, w)
    cont = torch.rand(burning.shape, generator=gen, device=B.device) < torch.sigmoid(p[:, 4]).view(n, h, w)
    return nd, (burning & cont) | nd, burnt | nd


def ss_loss(net, ev, geo, gen, n: int = 8, K: int = 7):
    """Scheduled Sampling: ab n zufaelligen Starttagen s (L <= s) k = 1..K Tage mit eigenen gezogenen Zustaenden
    weiterrollen (ohne Gradient; Wetter fest auf dem Starttag wie im Einsatz), dann Zuendung an s + k + 1 gegen die
    Beobachtung (BCE auf den im eigenen Zustand unverbrannten Landpixeln), mit Gradient durch den Kontext."""
    shape = ev["land"].shape
    h, w = shape
    L, C = T_WIN - 1, net.ctx
    dev = ev["fire"].device
    k = int(torch.randint(1, K + 1, (1,), generator=gen, device=dev))
    st = torch.randint(L, L + T - 1 - k, (n,), generator=gen, device=dev)                   # Ziel s + k + 1 <= L + T - 1
    feats, burnt_obs = observed_days(ev)
    tok = net.encode_days(feats, ev["stat"])                                                 # (60, d, H4, W4)
    land = ev["land"][None]
    gen_tok = []
    with torch.no_grad():
        burning, burnt = ev["fire"][st].clone(), burnt_obs[st].clone()
        cur, wx = feats[st, :3], ev["wx"][st]
        for j in range(k):
            ctx = tok[(st + j)[:, None] - torch.arange(C - 1, -1, -1, device=dev)[None]].clone()
            for i, g in enumerate(gen_tok):
                ctx[:, C - 1 - (j - 1 - i)] = g
            p = net.params(ctx, cur, ev["stat"], shape)
            with torch.autocast(dev.type, enabled=False):
                q = net.kernel(p, geo).float()
            prev_burnt = burnt.clone()
            _, burning, burnt = sample_day(net, p, q, burning, burnt, land, shape, gen)
            day = day_features(burning, prev_burnt, wx)
            cur = day[:, :3]
            gen_tok.append(net.encode_days(day, ev["stat"]))
    ctx = tok[(st + k)[:, None] - torch.arange(C - 1, -1, -1, device=dev)[None]]
    ctx = torch.cat([ctx[:, :C - k], torch.stack(gen_tok, 1)], 1)
    p = net.params(ctx, cur, ev["stat"], shape)
    with torch.autocast(dev.type, enabled=False):
        q = net.kernel(p, geo).float()
        fresh = (land & ~burnt & ~burning).flatten(1).float()
        Fp = net.step_prob(q, p[:, 3].exp(), burning.flatten(1).float(), fresh, shape)
    y = ev["fire"][st + k + 1].flatten(1).float()
    return bce(Fp, y, fresh)


@torch.no_grad()
def rollout(net, ev, geo, paths: int, gen, oracle: bool = False):
    """MC-Rollout ab L -> P(Ankunft <= t) (len(HOR), h, w), Ankunftstag je Pfad (paths, h, w)."""
    shape = ev["land"].shape
    h, w = shape
    L, K = T_WIN - 1, net.ctx
    feats, burnt_obs = observed_days(ev)
    tok_obs = net.encode_days(feats[:T_WIN], ev["stat"])                             # (30, d, H4, W4)
    hist = [tok_obs[k][None].expand(paths, -1, -1, -1) for k in range(T_WIN)]
    fire0 = ev["fire"][L]
    burning = fire0[None].repeat(paths, 1, 1)
    burnt = burnt_obs[L][None].repeat(paths, 1, 1)                                   # mit Feuer an L
    cur = feats[L, :3][None].expand(paths, -1, -1, -1)
    wx_L = ev["wx"][L][None].expand(paths, -1, -1, -1)
    land = ev["land"][None]
    arrival = torch.full((paths, h, w), 31, dtype=torch.long, device=burning.device)
    for s in range(T):
        p = net.params(torch.stack(hist[-K:], 1), cur, ev["stat"], shape)
        with torch.autocast(p.device.type, enabled=False):
            q = net.kernel(p, geo).float()
        prev_burnt = burnt.clone()
        nd, burning, burnt = sample_day(net, p, q, burning, burnt, land, shape, gen)
        arrival = torch.where(nd & (arrival > 30), torch.full_like(arrival, s + 1), arrival)
        wx = ev["wx"][L + 1 + s][None].expand(paths, -1, -1, -1) if oracle else wx_L   # Oracle: echtes Wetter des Tages
        day = day_features(burning, prev_burnt, wx)
        cur = day[:, :3]
        hist.append(net.encode_days(day, ev["stat"]))
    return torch.stack([(arrival <= t).float().mean(0) for t in HOR]), arrival


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sym-tag", default="", help="Ereignisse aus metrics/symbolic_spread<sym-tag>/events_train.json")
    ap.add_argument("--bench-tag", default="")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--paths", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ss", type=float, default=0.0, help="Gewicht des Scheduled-Sampling-Verlusts (0 = aus)")
    ap.add_argument("--ss-k", type=int, default=7, help="hoechstens so viele eigene Tage vor dem Ziel")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA-Zerfall der Gewichte (0 = aus); Auswahl und Auswertung mit EMA")
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten n Ereignisse je Split (Smoke-Test)")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--eval-only", default="", help="Checkpoint laden, nicht trainieren (z. B. fuer --oracle-weather)")
    ap.add_argument("--oracle-weather", action="store_true",
                    help="Rollout mit dem beobachteten Wetter der Folgetage statt Wetter von L (obere Schranke)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"fire_dayca{args.tag}"
    if (out / "runs" / "test_dayca.json").exists():
        raise SystemExit(f"ABBRUCH: {out} hat schon einen Lauf")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    cube = CubeReader(exp)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    masks = (np.load(exp / "derived_data" / "masks" / "land.npy"), np.load(exp / "derived_data" / "masks" / "canada.npy"))
    evs = {"train": json.loads((exp / "metrics" / f"symbolic_spread{args.sym_tag}" / "events_train.json").read_text())}
    for sp in ("val", "test"):
        evs[sp] = json.loads((exp / "metrics" / f"firemap_bench{args.bench_tag}" / f"events_{sp}.json").read_text())
    if args.limit:
        evs = {k: v[:args.limit] for k, v in evs.items()}
    t0 = time.time()
    cpu = torch.device("cpu")
    data = {sp: [to_dev(load_seq(cube, e, mean, std, load_event(cube, e, mean, std, masks)), cpu) for e in v]
            for sp, v in evs.items()}
    print(f"[dayca] geladen in {time.time() - t0:.0f} s: " + ", ".join(f"{k} {len(v)}" for k, v in data.items()), flush=True)
    geo = ca_geometry(device)
    net = DayCA(ctx=args.ctx).to(device)
    print(f"[dayca] Parameter: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best, best_state, hist, skipped = float("inf"), None, [], 0
    if args.eval_only:
        best_state = torch.load(args.eval_only, map_location=device)["state"]
        args.epochs = 0
    ema = EMA(net, args.ema) if args.ema > 0 else None
    tgen = torch.Generator(device=device).manual_seed(args.seed + 1)
    for ep in range(args.epochs):
        net.train()
        tl, t0 = [], time.time()
        for i in rng.permutation(len(data["train"])):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                evg = to_gpu(data["train"][i], device)
                loss, parts = tf_loss(net, evg, geo)
                if args.ss > 0:
                    l_ss = ss_loss(net, evg, geo, tgen, K=args.ss_k)
                    loss = loss + args.ss * l_ss
                    parts["ss"] = l_ss.item()
            opt.zero_grad()
            ok = bool(torch.isfinite(loss))
            if ok:
                loss.backward()
                ok = all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None)
            if not ok:
                skipped += 1
                continue
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(net)
            tl.append(float(loss.detach()))
        net.eval()
        raw = copy.deepcopy(net.state_dict()) if ema is not None else None
        if ema is not None:
            net.load_state_dict(ema.state(raw))
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            vl = float(np.mean([float(tf_loss(net, to_gpu(e, device), geo)[0]) for e in data["val"]]))
        hist.append({"epoch": ep, "train": float(np.mean(tl)) if tl else float("nan"), "val": vl, "skipped": skipped})
        print(f"[dayca] Epoche {ep}: train {hist[-1]['train']:.4f} val {vl:.4f} ({time.time() - t0:.0f} s, verworfen {skipped})", flush=True)
        if vl < best or best_state is None:
            best, best_state = vl, copy.deepcopy(net.state_dict())
        if raw is not None:
            net.load_state_dict(raw)
    net.load_state_dict(best_state)
    net.eval()
    for sub in ("ckpt", "runs", "maps"):
        (out / sub).mkdir(exist_ok=True)
    torch.save({"state": best_state, "args": vars(args)}, out / "ckpt" / "dayca.pt")
    for sp in ("val", "test"):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        res, maps = [], {}
        for i, ev in enumerate(data[sp]):
            ev = to_gpu(ev, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                P, arr = rollout(net, ev, geo, args.paths, gen, oracle=args.oracle_weather)
            P, arr = P.float().cpu().numpy(), arr.cpu().numpy()
            m, first = ev["m"].cpu().numpy(), ev["first"].cpu().numpy()
            st = {str(t): map_metrics(P[j], first <= t, m, arr <= t) for j, t in enumerate(HOR)}
            res.append({"i": i, "date": ev["date"], "box": ev["box"], "n_obs": {t: int(((first <= t) & m).sum()) for t in HOR},
                        "scales": {"1.0": st}})
            for j, t in enumerate(HOR):
                maps[f"{i}_1.0_{t}"] = P[j].astype(np.float16)
        (out / "runs" / f"{sp}_dayca.json").write_text(json.dumps(res))
        np.savez_compressed(out / "maps" / f"{sp}_dayca.npz", **maps)
        for t in HOR:
            a = agg(res, "1.0", str(t))
            print(f"[dayca] {sp} {t} d: AP {a['ap']:.3f} IoU {a['iou']:.3f} IoU top-n {a['iou_topk']:.3f} IoU Pfad "
                  f"{a['iou_path']:.3f} Praez. {a['prec']:.3f} Recall {a['rec']:.3f} log Flaeche {a['med_log_area']:+.2f}", flush=True)
    (out / "history.json").write_text(json.dumps({"history": hist, "best_val": best, "args": vars(args)}, indent=1))
    write_manifest(cfg, f"p72_fire_dayca{args.tag}", args.config, {"args": vars(args), "out": str(out)})


if __name__ == "__main__":
    main()

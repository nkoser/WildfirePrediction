"""ST-DCTM pro Pixel auf den Paper-Daten: eine Survival-Kurve fuer jedes der 13 x 13 Pixel.

Ziel je Pixel (r, c) und Stichtag e (letzter Fenstertag):

    T(e, r, c) = erster Tag > e, an dem die Feuermaske (Cache-Kanal 0) an
                 genau diesem Pixel 1 ist
    zensiert am Zieltag t (letzter Cache-Tag), also nach c = t - e Tagen

Nur Cache-Daten: Eingabe und Ereignisse stammen aus denselben 61 Tagen, aus
denen auch das GT-Label des Papers gebildet ist. e laeuft ueber 29 .. t - 1,
der Vorlauf damit ueber 1 .. 31 Tage; im Training wird e je Stichprobe und
Epoche neu gezogen.

Modell (wie das ST-DCTM aus surv_train, aber mit einem Token je Pixel statt
einem je Ausschnitt):

  Tages-CNN     (55, 13, 13) -> (d, 13, 13): je Pixel [eigenes Merkmal (3
                Faltungen, also Nachbarschaft) || Mittel || Maximum ueber den
                Ausschnitt]
  Gedaechtnis   je Pixel 29 Tages-Tokens + Tagesabstand, Self-Attention
  Abfrage       letzter Fenstertag, eigenes CNN, je Pixel
  Zusammenfassen 2 Cross-Attention-Bloecke je Pixel
  DCTM-Kopf     Bernstein Grad 8 auf [0, 31], monoton; Verlust: exakte
                Likelihood mit Ereignis an Tag k oder Zensierung bei c

Auswertung im Protokoll des Papers wie pixel.py: Stichtag e = t - dt - 1,
Risiko je Pixel F(dt + 1) = P(erstes Feuer bis zum Zieltag), bewertet gegen
die Feuermaske je Pixel an t (centre, pixel) und als Maximum gegen das
Paper-Label (patch).

Mit --train-source cube kommen die Trainings-Stichtage statt aus dem Cache aus
pixel_landmarks (ganz Kanada, 2000-2019, Vorlauf bis 100 Tage, Kopf Grad 12 auf
[0, 100]); Validierung und Test bleiben die Stichproben des Papers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.models.bernstein import BernsteinSurvivalHead
from wildfire.paper.compare import DTS
from wildfire.paper.metrics import ASSUMPTIONS
from wildfire.paper.pixel import map_metrics
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.channels import CHANNELS, masked_std
from wildfire.paper.channels import tag as channel_tag
from wildfire.paper import targets
from wildfire.paper.reconstruct import cache_norm, gather_windows, to_device_once
from wildfire.paper.stdctm import CrossBlock

T_WIN = 30
P = 13


class MapDayEncoder(nn.Module):
    """(N, C, 13, 13) -> (N, d, 13, 13): je Pixel eigenes Merkmal || Mittel || Maximum."""

    def __init__(self, c_in: int, w: int = 64, d: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
        )
        self.proj = nn.Conv2d(3 * w, d, 1)

    def forward(self, x):
        f = self.conv(x)
        g = torch.cat([f.mean((2, 3), keepdim=True), f.amax((2, 3), keepdim=True)], 1).expand(-1, -1, *f.shape[2:])
        return self.proj(torch.cat([f, g], 1))


class PixelSTDCTM(nn.Module):
    def __init__(self, c_in: int = 55, d: int = 128, degree: int = 8, tau_max: float = 31.0,
                 init_cif: float = 0.05):
        super().__init__()
        self.enc, self.q_enc = MapDayEncoder(c_in, d=d), MapDayEncoder(c_in, d=d)
        self.time = nn.Embedding(T_WIN, d)
        self.mem_block = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True)
        self.cross = nn.ModuleList([CrossBlock(d) for _ in range(2)])
        self.norm = nn.LayerNorm(d)
        self.dctm = BernsteinSurvivalHead(degree=degree, tau_max=tau_max, init_offset=-8.0, init_cif=init_cif)
        self.lin = nn.Linear(d, self.dctm.n_out)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x):                                   # (B, 30, C, 13, 13) -> (B, 13, 13, M+1)
        B, T, C, H, W = x.shape
        mem = self.enc(x[:, :-1].flatten(0, 1))             # (B*29, d, H, W)
        d = mem.shape[1]
        mem = mem.view(B, T - 1, d, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, T - 1, d)
        lag = torch.arange(T - 1, 0, -1, device=x.device)
        mem = self.mem_block(mem + self.time(lag)[None])
        q = self.q_enc(x[:, -1]).permute(0, 2, 3, 1).reshape(B * H * W, 1, d) + self.time.weight[0]
        for blk in self.cross:
            q = blk(q, mem)
        return self.lin(self.norm(q[:, 0])).view(B, H, W, -1)


def pixel_nll(head: BernsteinSurvivalHead, raw, event, t):
    """Mittlere negative Log-Likelihood; Ereignis an Tag t oder zensiert bei t (je Zeile)."""
    raw = raw.reshape(-1, raw.shape[-1])
    event, t = event.reshape(-1), t.reshape(-1).double()
    ls = head.log_survival(raw, torch.stack([t - 1.0, t], 1))
    ll_ev = ls[:, 0] + torch.log((-torch.expm1(ls[:, 1] - ls[:, 0])).clamp_min(1e-300))
    return -torch.where(event, ll_ev, ls[:, 1]).mean()


def pixel_targets(fire: np.ndarray, e: np.ndarray):
    """fire (n, L, 13, 13) bool, e (n,) letzter Fenstertag -> event, t je Pixel.

    Ereignis: erster Tag j > e mit Feuer, t = j - e. Sonst zensiert bei t = L-1-e.
    """
    n, L = fire.shape[:2]
    days = np.arange(L)[None, :, None, None]
    after = fire & (days > e[:, None, None, None])
    first = np.where(after, days, L).min(1)                  # (n, 13, 13)
    event = first < L
    t = np.where(event, first - e[:, None, None], (L - 1 - e)[:, None, None])
    return event, t.astype(np.int64)


def windows(X, rows, e):
    """(B, 30, C, 13, 13), letzter Tag e je Zeile; X numpy oder GPU-Tensor."""
    return gather_windows(X, rows, np.asarray(e) - T_WIN + 1, T_WIN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--landmarks-per-epoch", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--train-source", choices=["cache", "cube"], default="cache",
                    help="cache: Stichtage im Paper-Cache (Vorlauf bis 31); cube: Stichtage aus pixel_landmarks (bis 100)")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0 = ein voller Durchlauf")
    ap.add_argument("--tag", default="", help="Zusatz am Manifestnamen, z. B. _cube_s0")
    ap.add_argument("--select", choices=["nll", "prauc"], default="nll",
                    help="Modellauswahl: Val-NLL je Pixel oder mittlere Val-PR-AUC (Mittelzelle, Ausschnitt) im dt-Protokoll")
    ap.add_argument("--init-from", default="", help="Checkpoint zum Nachtrainieren; {seed} wird ersetzt")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--target", choices=targets.TARGETS, default="cache",
                    help="cache: Feuer an D; next_patch: Feuer an D+1, Fenster einen Tag spaeter (targets.py)")
    ap.add_argument("--channels", choices=list(CHANNELS), default="paper55",
                    help="Eingabekanaele; gestrichene gehen als 0 ins Netz, Ereignisse bleiben die Feuermaske")
    args = ap.parse_args()
    args.tag = targets.tag(args.target) + channel_tag(args.channels) + args.tag          # Manifest- und Checkpoint-Namen
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_load = time.time()
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")                                 # (N, 61, 55, 13, 13)
    y = targets.labels(d, args.target)
    split = np.load(d / "split.npy")
    L = X.shape[1] + targets.shift(args.target)          # effektive Laenge: Zieltag = Index L - 1
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    std = masked_std(std, args.channels)
    print(f"[pxst] Kanalsatz {args.channels}: {len(CHANNELS[args.channels])} Kanaele", flush=True)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    fire = X[:, :, FIRE_CH] > 0                              # (N, 61, 13, 13)
    if targets.shift(args.target):                           # Folgetag D+1 als Tag L - 1 anhaengen
        fire = np.concatenate([fire, targets.fire_map(d, X, args.target)[:, None]], 1)
    Y = fire[:, -1].astype(np.int8)                          # Label je Pixel am Zieltag
    E_LO, E_HI = T_WIN - 1, L - 2                            # moegliche Stichtage
    tau_max = float(L - 1 - E_LO)                            # 31 Tage
    degree = 8
    if args.train_source == "cube":
        cdir = exp / "derived_data" / args.cube_dir
        Xc = np.load(cdir / "train_X.npy", mmap_mode="r")
        evc, tc = np.load(cdir / "train_event.npy"), np.load(cdir / "train_t.npy").astype(np.int64)
        tau_max, degree = 100.0, 12
        print(f"[pxst] Training aus dem Wuerfel: {Xc.shape[0]:,} Stichtage, Pixel-Ereignisanteil {evc.mean():.4f}",
              flush=True)
    print(f"[pxst] {args.cache}: X {X.shape} in {time.time() - t_load:.0f} s; Stichtage {E_LO}..{E_HI}, "
          f"Vorlauf bis {tau_max:.0f} Tage", flush=True)

    ev0, _ = pixel_targets(fire[idx["train"]], np.full(idx["train"].size, E_LO))
    init_cif = float(np.clip(ev0.mean() if args.train_source == "cache" else evc.mean(), 1e-3, 0.5))
    rng_v = np.random.default_rng(12345)
    val_e = rng_v.integers(E_LO, E_HI + 1, size=(idx["val"].size, 2))
    val_rows = np.repeat(idx["val"], 2)
    val_e = val_e.reshape(-1)
    v_ev, v_t = pixel_targets(fire[val_rows], val_e)
    data_info = {"train_pixel_event_share_from_earliest_landmark": float(ev0.mean()),
                 "val_landmarks": int(val_rows.size), "val_pixel_event_share": float(v_ev.mean())}
    print(f"[pxst] {data_info}", flush=True)

    X = to_device_once(X, device)                            # Ereignisse oben schon aus der Feuermaske im RAM

    def to_dev(xw):
        xw = xw if isinstance(xw, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(xw))
        return (xw.to(device).float() - m_t) / s_t

    @torch.no_grad()
    def val_nll(net):
        net.eval()
        tot = 0.0
        for a in range(0, val_rows.size, 64):
            sl = slice(a, a + 64)
            raw = net(to_dev(windows(X, val_rows[sl], val_e[sl])))
            tot += pixel_nll(net.dctm, raw, torch.from_numpy(v_ev[sl]).to(device),
                             torch.from_numpy(v_t[sl]).to(device)).item() * raw.shape[0]
        return tot / val_rows.size

    @torch.no_grad()
    def paper_maps(net, rows, dt_):
        net.eval()
        e = L - dt_ - 2                                      # letzter Fenstertag = t - dt - 1
        k = float(dt_ + 1)
        out = []
        for a in range(0, rows.size, 64):
            r = rows[a:a + 64]
            raw = net(to_dev(windows(X, r, np.full(len(r), e))))
            flat = raw.reshape(-1, raw.shape[-1])
            Fk = -torch.expm1(net.dctm.log_survival(flat, torch.tensor([k])))[:, 0]
            out.append(Fk.view(len(r), P, P).cpu().numpy())
        return np.clip(np.concatenate(out), 1e-7, 1 - 1e-7)

    va = idx["val"]

    def val_rank(net):
        """Mittlere Val-PR-AUC ueber dt: Mittelzelle gegen ihr Pixel-Label, Maximum gegen das Paper-Label."""
        sc = []
        for dt_ in DTS:
            m = paper_maps(net, va, dt_)
            sc.append(average_precision_score(Y[va, 6, 6], m[:, 6, 6]))
            sc.append(average_precision_score(y[va], m.max(axis=(1, 2))))
        return float(np.mean(sc))

    te = idx["test"]
    maps = {dt_: [] for dt_ in DTS}
    hists = []
    ckpt_dir = exp / "checkpoints" / "paper"
    for seed in args.seeds:
        print(f"\n[pxst] Seed {seed}", flush=True)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        if args.init_from:
            ck = torch.load(args.init_from.format(seed=seed), map_location="cpu")
            degree, tau_max = ck["degree"], ck["tau_max"]
            net = PixelSTDCTM(degree=degree, tau_max=tau_max).to(device)
            net.load_state_dict(ck["state"])
            print(f"[pxst] nachtrainiert von {args.init_from.format(seed=seed)} (Grad {degree}, "
                  f"tau_max {tau_max:.0f})", flush=True)
        else:
            net = PixelSTDCTM(degree=degree, tau_max=tau_max, init_cif=init_cif).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        best, state, bad, hist = np.inf, None, 0, []
        if args.init_from and args.select == "prauc":
            best = -val_rank(net)
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"      Start: val PR-AUC (Mittel) {-best:.4f}", flush=True)
        for ep in range(args.epochs):
            net.train()
            if args.train_source == "cube":
                rows = rng.permutation(Xc.shape[0])
            else:
                rows = rng.permutation(np.repeat(idx["train"], args.landmarks_per_epoch))
                e_all = rng.integers(E_LO, E_HI + 1, size=rows.size)
            if args.steps_per_epoch:
                rows = rows[:args.steps_per_epoch * args.bs]
            tl = tn = 0
            t_ep = time.time()
            for a in range(0, rows.size, args.bs):
                if args.train_source == "cube":
                    r = np.sort(rows[a:a + args.bs])
                    ev, tt, xw = evc[r], tc[r], np.asarray(Xc[r])
                else:
                    r, e = rows[a:a + args.bs], e_all[a:a + args.bs]
                    ev, tt = pixel_targets(fire[r], e)
                    xw = windows(X, r, e)
                raw = net(to_dev(xw))
                loss = pixel_nll(net.dctm, raw, torch.from_numpy(ev).to(device), torch.from_numpy(tt).to(device))
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
                tl += loss.item() * len(r); tn += len(r)
            vl = val_nll(net)
            vr = val_rank(net) if args.select == "prauc" else float("nan")
            crit = -vr if args.select == "prauc" else vl
            hist.append({"epoch": ep, "train_nll": tl / tn, "val_nll": vl, "val_prauc_mean": vr})
            print(f"      Epoche {ep:2d}: train NLL je Pixel {tl / tn:.5f}  val {vl:.5f}  val PR-AUC (Mittel) "
                  f"{vr:.4f}  ({time.time() - t_ep:.0f} s)", flush=True)
            if crit < best - 1e-5:
                best, bad, state = crit, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        net.load_state_dict(state)
        hists.append(hist)
        if args.save_ckpt:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cp = ckpt_dir / f"pxst_{args.cache}{args.tag}_s{seed}.pt"
            torch.save({"state": state, "degree": degree, "tau_max": tau_max}, cp)
            print(f"[pxst] Checkpoint {cp}", flush=True)
        for dt_ in DTS:
            maps[dt_].append(paper_maps(net, te, dt_))

    res = {}
    for dt_ in DTS:
        res[str(dt_)] = map_metrics(maps[dt_], Y[te], y[te])
        print(f"[pxst] st_dctm_px dt={dt_:2d}  " + "   ".join(
            f"{ev}: PR-AUC {res[str(dt_)][ev]['summary']['pr_auc']['mean']:.3f} "
            f"ROC {res[str(dt_)][ev]['summary']['roc_auc']['mean']:.3f}" for ev in res[str(dt_)]), flush=True)

    write_manifest(cfg, f"p10_pixel_stdctm_{args.cache}{args.tag}", args.config, {
        "cache": args.cache, "train_source": args.train_source,
        "target": args.target, "channels": args.channels, "channel_list": CHANNELS[args.channels],
        "target": "je Pixel: erster Tag nach dem Stichtag mit Feuermaske 1 (Cache-Kanal 0 bzw. MCD14A1), zensiert "
                  "am Zieltag (cache) bzw. nach 100 Tagen (cube)",
        "landmarks": (f"Stichtag e in {E_LO}..{E_HI} (Cache-Index), {args.landmarks_per_epoch} je Stichprobe und Epoche"
                      if args.train_source == "cache" else f"pixel_landmarks ({args.cube_dir}), 2000-2019"),
        "head": f"Bernstein Grad {degree} auf [0, {tau_max:.0f}]", "steps_per_epoch": args.steps_per_epoch,
        "selection": args.select, "init_from": args.init_from, "lr": args.lr,
        "score": "F(dt + 1) je Pixel bei e = t - dt - 1",
        "data": data_info, "seeds": args.seeds, "histories": hists,
        "metric_assumptions": ASSUMPTIONS, "results": {"st_dctm_px": res},
    })

    ref_path = exp / "data_manifests" / f"p9_pixel_{args.cache}.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())["results"]
        for ev in ("centre", "pixel", "patch"):
            print(f"\n[pxst] {args.cache} TEST, {ev}, PR-AUC")
            for dt_ in DTS:
                row = [f"dt={dt_:2d}"] + [f"{m} {ref[m][str(dt_)][ev]['summary']['pr_auc']['mean']:.3f}" for m in ref]
                row.append(f"st_dctm_px {res[str(dt_)][ev]['summary']['pr_auc']['mean']:.3f}")
                print("    " + "   ".join(row))


if __name__ == "__main__":
    main()

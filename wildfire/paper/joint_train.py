"""Training fuer die Experimente zur gemeinsamen Verteilung (nur neue Checkpoints, nichts wird ueberschrieben).

  cg_tf      Contagion-STS (paper/contagion.py, unveraendert) mit der exakten gemeinsamen Likelihood des
             Ausbreitungsprozesses statt der Mean-Field-Pixelkurven. Gegeben die echten Erstfeuer-Tage T_j
             aller Pixel des Ausschnitts zerfaellt die Likelihood in Tag- und Pixelfaktoren:

                 log(1 - h_i(t)) = log(1 - h_exo,i(t)) + sum_{j != i} log(1 - k_ij b_j(t-1)),
                 b_j(t-1) = D_j(t - 1 - T_j)  falls T_j <= t - 1, sonst 0,

             also derselbe Hazard wie in der Simulation (region_eval.mc_region_logS), nur mit den echten
             statt gewuerfelten Feuern der Nachbarn. Die Labels sieht nur der Verlust; vorhergesagt wird
             weiter durch Wuerfeln. Damit passen die Parameter zum Prozess, den das Wuerfeln zieht, und nicht
             zur Mean-Field-Naeherung. Ausschnittskurve exakt wie bisher (Gewicht --patch-weight).
  lh_region  STS-LH (Pixel- und Ausschnittskopf wie _abl_lh) plus Regionskoepfe:
               fixed  je ein Kopf fuer sq5, sq9, nw6
               mask   ein Kopf, der die Region als Maske bekommt; trainiert auf zufaelligen Rechtecken,
                      zusammenhaengenden Flecken und Vereinigungen zweier Rechtecke (je Stichtag --masks)
  lh_ign     STS-LH mit Ziel erste NFDB-Zuendung (>= 1 ha) statt MODIS-Feuer, nur Stichtage mit
             NFDB-beobachteter Mitte (wie contagion_cause_eval); fairer Vergleich fuer den Zuendanteil a.

Schleife wie contagion_cause.py (die mit lambda 0 _abl_cg exakt reproduziert): Wuerfel-Stichtage, d 64,
4 Bloecke, Batch 32, AdamW 3e-4, wd 0,05, 500 Warm-up, EMA 0,999, letzte EMA-Gewichte. warm: Rumpf aus
_abl_lh, 3 x 3.000 Schritte; scratch: 10 x 3.000 Schritte.

Der Bias der Logistic-Hazard-Koepfe und ign_bias von Contagion haengen an der Ereignisrate der Trainingsdaten
und stehen nicht im state dict (load_sts setzt die Standardwerte 0,02 / 0,2). Hier werden die Raten in
args gespeichert (init_cif_px, init_cif_patch, init_cif_region), damit joint_eval beide Varianten auswerten kann.

Checkpoints NEU unter checkpoints/paper_joint/; cg_tf und lh_ign im Format von sts_dctm (load_sts liest sie).
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
from wildfire.models.surv_heads import head_loss, make_head
from wildfire.paper import targets
from wildfire.paper.channels import masked_std
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.region_eval import regions
from wildfire.paper.sts_dctm import DEGREE, P, STSDCTM, build_sts, load_sts, lr_factor, patch_events

H = 100
N = P * P
HALF = P // 2
FIXED = ("sq5", "sq9", "nw6")


# ---------------------------------------------------------------------------- Contagion, gemeinsame Likelihood
def tf_log_survival(net, obs, a, k, logD, ev, tt):
    """Pixelkurven (B, N, H+1) mit den echten Erstfeuern der Nachbarn (Teacher Forcing) und log(1 - h_exo).

    Zeitkonvention wie mc_region_logS: ein Feuer an Tag T wirkt an Tag t mit D(t - 1 - T), erstmals an T + 1;
    die Kernmitte wirkt nur ueber das beobachtete Feuer."""
    B = obs.shape[0]
    log_exo = net.exogenous(obs, a, k, logD)                                     # (B, H, N)
    T = torch.where(ev.reshape(B, N), tt.reshape(B, N).long(), torch.full_like(tt.reshape(B, N).long(), 10 ** 6))
    age = torch.arange(net.H, device=obs.device)[None, :, None] - T[:, None, :]  # (B, H, N): t - 1 - T
    on = age >= 0
    lb = logD.gather(2, age.clamp(0, logD.shape[-1] - 1).transpose(1, 2)).transpose(1, 2)
    b = torch.where(on, lb.exp(), torch.zeros_like(lb))
    KK, c = net.K * net.K, net.K * net.K // 2
    U = net._unfold(b.reshape(B * net.H, N)).view(B, net.H, KK, N)
    lp = torch.log1p(-(k[:, None] * U).clamp(max=1 - 1e-6))
    l1mh = log_exo + lp.sum(2) - lp[:, :, c]
    return F.pad(l1mh.cumsum(1).transpose(1, 2), (1, 0)), log_exo                # (B, N, H+1)


# ---------------------------------------------------------------------------- STS-LH mit Regionskoepfen
class RegionHead(nn.Module):
    """Rohwerte eines Logistic-Hazard-Kopfs (B, H) fuer eine Region: Attention-Pooling ueber die Pixel-Token
    in der Maske, Mittel der Token, mit area_feat dazu Flaechenanteil und dessen Logarithmus."""

    def __init__(self, d: int, heads: int, area_feat: bool):
        super().__init__()
        self.area_feat = area_feat
        self.q = nn.Parameter(torch.zeros(1, 1, d))
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        nf = 2 * d + (2 if area_feat else 0)
        self.norm = nn.LayerNorm(nf)
        self.lin = nn.Linear(nf, H)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, pix, mask):
        m = mask.to(pix.dtype)
        q = self.q.expand(pix.shape[0], -1, -1).to(pix.dtype)
        att = self.att(q, pix, pix, key_padding_mask=~mask, need_weights=False)[0][:, 0]
        mean = (pix * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        f = [att, mean]
        if self.area_feat:
            fr = m.sum(1, keepdim=True) / N
            f += [fr, torch.log(fr)]
        return self.lin(self.norm(torch.cat(f, -1))).float()


class STSRegion(STSDCTM):
    def __init__(self, mode: str, init_cif_region, **kw):
        super().__init__(head="lh", **kw)
        self.mode = mode
        d, heads = kw.get("d", 96), kw.get("heads", 4)
        if mode == "fixed":
            self.reg_heads = nn.ModuleDict({r: RegionHead(d, heads, False) for r in FIXED})
            self.reg_lh = nn.ModuleDict({r: make_head("lh", H, DEGREE, init_cif_region[r]) for r in FIXED})
            rg = regions()
            self.register_buffer("fixed_masks", torch.from_numpy(np.stack([rg[r] for r in FIXED])))
        elif mode == "mask":
            self.reg_head = RegionHead(d, heads, True)
            self.reg_lh_mask = make_head("lh", H, DEGREE, float(init_cif_region))
        else:
            raise ValueError(f"mode {mode!r}")

    def forward_all(self, x, doy, masks=None):
        """-> roh Pixel (B, 13, 13, H), roh Ausschnitt (B, H), Regionen: fixed {Name: (B, H)}, mask (B, K, H)."""
        B = x.shape[0]
        pix = self.encode(x, doy)
        raw_px = self.px_lin(pix).view(B, P, P, -1).float()
        a = self.pool(self.cls.expand(B, -1, -1), pix, pix, need_weights=False)[0][:, 0]
        raw_pa = self.pa_lin(self.pool_norm(a)).float()
        if self.mode == "fixed":
            reg = {r: self.reg_heads[r](pix, self.fixed_masks[i][None].expand(B, -1)) for i, r in enumerate(FIXED)}
        else:
            K = masks.shape[1]
            reg = self.reg_head(pix.repeat_interleave(K, 0), masks.reshape(B * K, N)).view(B, K, -1)
        return raw_px, raw_pa, reg

    def region_raw(self, pix, mask):
        """Fuer die Auswertung: roh (B, H) einer beliebigen Maske (169,) mit dem Maskenkopf."""
        return self.reg_head(pix, mask[None].expand(pix.shape[0], -1))


def load_joint(path, device="cpu"):
    """Checkpoint aus joint_train -> Netz im eval-Modus (lh_region eigenes Format, sonst load_sts)."""
    c = torch.load(path, map_location="cpu")
    a = c.get("args", {})
    if a.get("head") != "lh_region":
        return load_sts(path, device)
    net = STSRegion(a["region_mode"], a["init_cif_region"], d=a["d"], heads=a["heads"], depth=a["depth"])
    net.load_state_dict(c["state"])
    return net.to(device).eval()


def mask_bank(n: int, rng) -> np.ndarray:
    """Zufaellige Trainingsregionen (n, 169) bool: Rechtecke 50 %, zusammenhaengende Flecken mit 2 bis 60
    Pixeln 35 %, Vereinigung zweier Rechtecke 15 %."""
    out = np.zeros((n, P, P), bool)

    def rect(m):
        h, w = rng.integers(1, P + 1, 2)
        r0, c0 = rng.integers(0, P - h + 1), rng.integers(0, P - w + 1)
        m[r0:r0 + h, c0:c0 + w] = True

    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for i in range(n):
        u = rng.random()
        if u < 0.5:
            rect(out[i])
        elif u < 0.85:
            size = int(rng.integers(2, 61))
            start = (int(rng.integers(P)), int(rng.integers(P)))
            cells, front = {start}, [start]
            while len(cells) < size:
                r, c = front[rng.integers(len(front))]
                dr, dc = steps[rng.integers(4)]
                q = (r + dr, c + dc)
                if 0 <= q[0] < P and 0 <= q[1] < P and q not in cells:
                    cells.add(q); front.append(q)
            for r, c in cells:
                out[i, r, c] = True
        else:
            rect(out[i]); rect(out[i])
    return out.reshape(n, N)


def region_events(ev, tt, masks):
    """Erstes Feuer in jeder Maske: ev, tt (B, N); masks (B, K, N) -> (B, K) Ereignis, Zeit.
    Alle Pixel eines Stichtags haben dieselbe Zensurzeit (wie patch_events)."""
    big = torch.full_like(tt, 10 ** 6)
    t = torch.where(masks, tt[:, None, :], big[:, None, :]).min(2).values
    e = (ev[:, None, :] & masks).any(2)
    return e, t


# ---------------------------------------------------------------------------- Ziel erste NFDB-Zuendung
def nfdb_targets(exp, n_rows):
    """Erste NFDB-Zuendung je Pixel fuer die Trainings-Stichtage und die Auswahl mit NFDB-beobachteter Mitte."""
    from wildfire.paper.cause_labels import nfdb_keys
    from wildfire.paper.pixel_landmarks import pixel_events

    lab = dict(np.load(exp / "derived_data" / "cause_labels" / "train_cause.npz"))
    if lab["row"].size != n_rows:
        raise SystemExit("ABBRUCH: train_cause.npz passt nicht zu train_meta.npz")
    observed = np.load(exp / "derived_data" / "masks" / "observed_nfdb.npy").astype(bool)
    Hg, Wg = observed.shape
    nk, _ = nfdb_keys(exp / "derived_data" / "events" / "ignitions_nfdb_min1ha.csv", Wg)
    ev, t = pixel_events(nk, lab["L_abs"].astype(np.int64), lab["row"], lab["col"], Hg, Wg)
    keep = observed[lab["row"], lab["col"]]
    return ev, t.astype(np.int64), keep, lab


# ---------------------------------------------------------------------------- Training
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=["cg_tf", "lh_region", "lh_ign"], required=True)
    ap.add_argument("--region-mode", choices=["fixed", "mask"], default="mask")
    ap.add_argument("--region-weight", type=float, default=1.0)
    ap.add_argument("--masks", type=int, default=4, help="lh_region mask: zufaellige Regionen je Stichtag")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--init-from", default="/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_lh_s{seed}_s{seed}.pt",
                    help="Rumpf (warm); leer = von Grund auf")
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
    ap.add_argument("--limit", type=int, default=0, help="nur fuer Rauchtests: so viele Trainings-Stichtage")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    ck_dir = exp / "checkpoints" / "paper_joint"
    kind = {"cg_tf": "cgtf", "lh_region": f"lhreg_{args.region_mode}", "lh_ign": "lhign"}[args.model]
    ck = ck_dir / f"sts_{args.cache}_next_cube_{kind}{args.tag}_s{args.seed}.pt"
    if ck.exists():
        raise SystemExit(f"ABBRUCH: {ck} existiert schon - wird nicht ueberschrieben")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- Daten (nur gelesen)
    cdir = exp / "derived_data" / args.cube_dir
    meta = dict(np.load(cdir / "train_meta.npz"))
    n_all = meta["L"].size
    cand = np.arange(n_all)
    if args.model == "lh_ign":
        ev_ig, t_ig, keep, lab = nfdb_targets(exp, n_all)
        if not (np.array_equal(lab["row"], meta["row"]) and np.array_equal(lab["col"], meta["col"])):
            raise SystemExit("ABBRUCH: train_cause.npz passt nicht zu train_meta.npz")
        cand = cand[keep]
    rows = cand if not args.limit else np.sort(rng.choice(cand, min(args.limit, cand.size), replace=False))
    t0 = time.time()
    Xc = np.asarray(np.load(cdir / "train_X.npy", mmap_mode="r")[rows])
    if args.model == "lh_ign":
        evc, tc = ev_ig[rows], t_ig[rows]
    else:
        evc = np.load(cdir / "train_event.npy")[rows].astype(bool)
        tc = np.load(cdir / "train_t.npy")[rows].astype(np.int64)
    dates = CubeReader(exp).dates
    doyc = np.array([dt.date.fromisoformat(dates[int(L)]).timetuple().tm_yday for L in meta["L"][rows]], np.float32)
    init_px, init_pa = float(evc.mean()), float(evc.reshape(len(evc), -1).any(1).mean())
    print(f"[joint] {args.model}: {len(rows):,} Stichtage in {time.time() - t0:.0f} s; Pixel-Ereignisse {init_px:.4f}, "
          f"Ausschnitt {init_pa:.3f}", flush=True)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    std = masked_std(std, "paper55")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    cif = dict(init_cif_px=float(np.clip(init_px, 1e-4, 0.5)), init_cif_patch=float(np.clip(init_pa, 1e-3, 0.9)))

    # ---- Modell
    init_reg, bank = None, None
    if args.model == "cg_tf":
        net = build_sts("contagion", d=args.d, heads=args.heads, depth=args.depth, p=args.dropout, **cif,
                        kernel=args.cg_kernel, obs_days=args.cg_obs_days, burn=args.cg_burn, blind="none")
        net.set_fire_norm(float(mean[FIRE_CH]), float(std[FIRE_CH]))
    elif args.model == "lh_ign":
        net = build_sts("lh", d=args.d, heads=args.heads, depth=args.depth, p=args.dropout, **cif)
    else:
        rg = regions()
        if args.region_mode == "fixed":
            init_reg = {r: float(np.clip(evc.reshape(len(evc), -1)[:, rg[r]].any(1).mean(), 1e-4, 0.9)) for r in FIXED}
        else:
            bank = mask_bank(20_000, np.random.default_rng(1000 + args.seed))
            sub = rng.choice(len(evc), min(5000, len(evc)), replace=False)
            e = evc.reshape(len(evc), -1)[sub]
            init_reg = float(np.clip(np.mean([e[:, bank[j]].any(1).mean() for j in range(200)]), 1e-4, 0.9))
        net = STSRegion(args.region_mode, init_reg, d=args.d, heads=args.heads, depth=args.depth, p=args.dropout, **cif)
        print(f"[joint] Regionskoepfe {args.region_mode}, Start-Risiko bis {H} Tage: {init_reg}", flush=True)
    net = net.to(device)
    if args.init_from:
        src = args.init_from.format(seed=args.seed)
        sd0 = torch.load(src, map_location="cpu")["state"]
        own = net.state_dict()
        sd0 = {k: v for k, v in sd0.items() if k in own and own[k].shape == v.shape and not k.startswith(("px_", "pa_"))}
        net.load_state_dict(sd0, strict=False)
        print(f"[joint] Rumpf aus {src}: {len(sd0)} von {len(own)} Tensoren", flush=True)
    else:
        print("[joint] von Grund auf", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.warmup, total, "const", 0.0))
    ema = EMA(net, args.ema)
    bank_t = torch.from_numpy(bank).to(device) if bank is not None else None
    hist = []
    for ep in range(args.epochs):
        net.train()
        t_ep = time.time()
        order = rng.permutation(len(rows))[:args.steps_per_epoch * args.bs]
        sums = {"loss": 0.0, "px": 0.0, "pa": 0.0, "reg": 0.0, "n": 0}
        for a0 in range(0, order.size, args.bs):
            r = np.sort(order[a0:a0 + args.bs])
            B = len(r)
            x = (torch.from_numpy(Xc[r]).to(device).float() - m_t) / s_t
            doy = torch.from_numpy(doyc[r]).to(device)
            ev, tt = torch.from_numpy(evc[r]).to(device), torch.from_numpy(tc[r]).to(device)
            pe, pt = patch_events(evc[r], tc[r])
            pe, pt = torch.from_numpy(pe).to(device), torch.from_numpy(pt).to(device)
            reg_loss = torch.zeros((), device=device)
            if args.model == "cg_tf":
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pix = net.encode_input(x, doy)
                obs, a, k, logD = net.params(x, doy, pix)
                with torch.autocast(device_type=device.type, enabled=False):
                    ls_px, log_exo = tf_log_survival(net, obs, a, k, logD, ev, tt)
                    ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0))
                px = head_loss(net.px_head, ls_px.reshape(B, P, P, -1), ev, tt)
                pa = head_loss(net.pa_head, ls_pa, pe, pt)
            else:
                masks = None
                if bank_t is not None:
                    masks = bank_t[torch.from_numpy(rng.integers(0, len(bank), (B, args.masks))).to(device)]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw_px, raw_pa, reg = net.forward_all(x, doy, masks) if args.model == "lh_region" else (*net(x, doy), None)
                px = head_loss(net.px_head, raw_px, ev, tt)
                pa = head_loss(net.pa_head, raw_pa, pe, pt)
                if args.model == "lh_region":
                    evf, ttf = ev.reshape(B, N), tt.reshape(B, N)
                    if args.region_mode == "fixed":
                        ls_ = []
                        for i, name in enumerate(FIXED):
                            e_, t_ = region_events(evf, ttf, net.fixed_masks[i][None, None].expand(B, 1, -1))
                            ls_.append(head_loss(net.reg_lh[name], reg[name], e_[:, 0], t_[:, 0]))
                        reg_loss = torch.stack(ls_).mean()
                    else:
                        e_, t_ = region_events(evf, ttf, masks)
                        reg_loss = head_loss(net.reg_lh_mask, reg.reshape(B * args.masks, -1), e_.reshape(-1), t_.reshape(-1))
            loss = px + args.patch_weight * pa + args.region_weight * reg_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            ema.update(net)
            for key, v in (("loss", loss), ("px", px), ("pa", pa), ("reg", reg_loss)):
                sums[key] += float(v) * B
            sums["n"] += B
        h = {"epoch": ep, **{k: sums[k] / sums["n"] for k in ("loss", "px", "pa", "reg")}}
        hist.append(h)
        print(f"      Epoche {ep}: Verlust {h['loss']:.5f}  Pixel {h['px']:.5f}  Ausschnitt {h['pa']:.5f}  "
              f"Regionen {h['reg']:.5f} ({time.time() - t_ep:.0f} s)", flush=True)
    state = ema.state(net.state_dict())
    ck_dir.mkdir(parents=True, exist_ok=True)
    saved = {"head": {"cg_tf": "contagion", "lh_region": "lh_region", "lh_ign": "lh"}[args.model],
             "d": args.d, "heads": args.heads, "depth": args.depth, "attn": "factorized", "stem": "2d",
             "hist_days": 30, "rate_weight": 0.0, **cif, "joint": vars(args)}
    if args.model == "cg_tf":
        saved.update(cg_kernel=args.cg_kernel, cg_obs_days=args.cg_obs_days, cg_burn=args.cg_burn, cg_blind="none")
    if args.model == "lh_region":
        saved.update(region_mode=args.region_mode, init_cif_region=init_reg)
    torch.save({"state": state, "args": saved, "history": hist}, ck)
    (ck_dir / (ck.stem + ".json")).write_text(json.dumps({"args": saved, "history": hist}, indent=1))
    print(f"[joint] Checkpoint {ck}", flush=True)


if __name__ == "__main__":
    main()

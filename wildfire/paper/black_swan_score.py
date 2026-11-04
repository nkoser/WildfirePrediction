"""Black Swans, GPU: Scores der sts_extreme-Modelle auf der dichten Stichtagsmenge und dem Paper-Test.

Je Modell (--model, z. B. lh_ref) und Seed der Checkpoint checkpoints/paper/sts_extreme_<model>_s<seed>_s<seed>.pt,
geladen wie in sts_extreme (load_any, Kopf-Bias wie im Training aus dem Checkpoint), nur Modelle ohne
Zusatzeingaben (--ctx none). Normierung wie sts_extreme (cache_G/norm_stats.npz), bf16-Autocast wie surv_hazards.
  dense  Fenster aus dem Wuerfel fuer derived_data/black_swan<tag>/set.npz (je Ereignis ein Kasten im RAM,
         Kanaele wie surv_prepare.window) -> Lambda(tau) fuer Mitte (Pixelkopf 6, 6) und Ausschnitt an
         black_swan_dense.TAUS -> score_<model>.npz: lam_centre, lam_patch (seeds, n, taus) float32
  paper  derived_data/paper/cache_G_L61, Test-Zeilen: F(dt + 1) mit Stichtag e = D - dt wie sts_extreme.paper_eval
         -> paper_<model>.npz: rows, F_centre, F_patch (seeds, len(DTS), n)
Ausgaben NEU (bricht ab, wenn vorhanden).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.black_swan_dense import TAUS
from wildfire.paper.compare import DTS
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.sts_dctm import doy_of
from wildfire.paper.sts_extreme import load_any
from wildfire.paper.surv_prepare import CACHE_FROM_CUBE, HALF, T_WIN

CENTRE = HALF


def event_box(cube: CubeReader, L: np.ndarray, rr: np.ndarray, cc: np.ndarray):
    """Kasten (T, 55, h, w) uint8 fuer alle Stichtage eines Ereignisses, Rand genullt, und Versatz."""
    year, li1 = cube._pos[int(L.max())]
    y0, li0 = cube._pos[int(L.min()) - T_WIN + 1]
    assert y0 == year, (y0, year)
    dyn, stat = cube._arrays(year)
    R0, R1 = int(rr.min()) - HALF, int(rr.max()) + HALF + 1
    C0, C1 = int(cc.min()) - HALF, int(cc.max()) + HALF + 1
    r0, r1, c0, c1 = max(0, R0), min(cube.H, R1), max(0, C0), min(cube.W, C1)
    d = np.asarray(dyn[li0:li1 + 1, r0:r1, c0:c1, :])
    s = np.asarray(stat[r0:r1, c0:c1, :])
    box = np.zeros((li1 - li0 + 1, 55, R1 - R0, C1 - C0), np.uint8)
    rs, cs = slice(r0 - R0, r1 - R0), slice(c0 - C0, c1 - C0)
    for ch, (src, i) in enumerate(CACHE_FROM_CUBE):
        if src == "dyn":
            box[:, ch, rs, cs] = d[..., i]
        else:
            box[:, ch, rs, cs] = s[..., i][None]
    # Tag-Index im Kasten fuer Stichtag L: li(L) - li0
    return box, li0, R0, C0, year


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--part", choices=("dense", "paper", "both"), default="both")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--limit-paper", type=int, default=0, help="Rauchtest: so viele Paper-Test-Zeilen")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "derived_data" / f"black_swan{args.tag}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    norm = lambda x: (torch.from_numpy(np.ascontiguousarray(x)).to(device).float() - m_t) / s_t  # noqa: E731

    nets = []
    for s in args.seeds:
        p = exp / "checkpoints" / "paper" / f"sts_extreme_{args.model}_s{s}_s{s}.pt"
        c = torch.load(p, map_location="cpu")
        if c.get("args", {}).get("ctx", "none") != "none":
            raise SystemExit(f"ABBRUCH: {p.name} hat Zusatzeingaben (ctx) - hier nicht unterstuetzt")
        net, _ = load_any(str(p), c["init_px"], c["init_pa"], device)
        nets.append(net)
        print(f"[bss] {p.name}: init_px {c['init_px']:.4f} init_pa {c['init_pa']:.3f}", flush=True)
    t0 = time.time()

    @torch.no_grad()
    def fwd(net, x, doy, tau):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_px, raw_pa = net(norm(x), torch.from_numpy(doy).to(device))
        lc = -net.px_head.log_survival(raw_px[:, CENTRE, CENTRE], tau)
        lp = -net.pa_head.log_survival(raw_pa, tau)
        return lc.float().cpu().numpy(), lp.float().cpu().numpy()

    meta = {"model": args.model, "seeds": args.seeds}
    if args.part in ("dense", "both"):
        fp = out / f"score_{args.model}.npz"
        if fp.exists():
            raise SystemExit(f"ABBRUCH: {fp} existiert schon")
        S = np.load(out / "set.npz")
        L, rr, cc, doy, evid = S["L"], S["row"], S["col"], S["doy"], S["event"]
        n = L.size
        lam_c = np.zeros((len(nets), n, len(TAUS)), np.float32)
        lam_p = np.zeros_like(lam_c)
        tau = torch.tensor([float(t) for t in TAUS], device=device)
        cube = CubeReader(exp)
        for e in np.unique(evid):
            idx = np.where(evid == e)[0]
            box, li0, R0, C0, year = event_box(cube, L[idx], rr[idx], cc[idx])
            li = np.array([cube._pos[int(v)][1] for v in L[idx]]) - li0
            t1 = time.time()
            for a in range(0, idx.size, args.bs):
                j = slice(a, a + args.bs)
                x = np.stack([box[t - T_WIN + 1:t + 1, :, r - R0 - HALF:r - R0 + HALF + 1, c - C0 - HALF:c - C0 + HALF + 1]
                              for t, r, c in zip(li[j], rr[idx][j], cc[idx][j])])
                for k, net in enumerate(nets):
                    lc, lp = fwd(net, x, doy[idx][j], tau)
                    lam_c[k, idx[j]], lam_p[k, idx[j]] = lc, lp
            print(f"[bss] Ereignis {e}: {idx.size:,} Stichtage, Kasten {box.shape} ({time.time() - t1:.0f} s)", flush=True)
            del box
        np.savez(fp, lam_centre=lam_c, lam_patch=lam_p, taus=np.array(TAUS))
        meta["dense"] = {"file": str(fp), "n": int(n)}
        print(f"[bss] {fp} ({time.time() - t0:.0f} s)", flush=True)

    if args.part in ("paper", "both"):
        fp = out / f"paper_{args.model}.npz"
        if fp.exists():
            raise SystemExit(f"ABBRUCH: {fp} existiert schon")
        cd = exp / "derived_data" / "paper" / f"{args.cache}_L61"
        Xp = np.load(cd / "X.npy", mmap_mode="r")
        LC = Xp.shape[1]
        te = np.where(np.load(cd / "split.npy") == "test")[0]
        if args.limit_paper:
            te = te[:args.limit_paper]
        doyD = doy_of(np.load(cd / "meta.npy")[:, 0])
        Xt = np.asarray(Xp[te])                                       # (n, 61, 55, 13, 13)
        Fc = np.zeros((len(nets), len(DTS), te.size), np.float32)
        Fp = np.zeros_like(Fc)
        loc = np.arange(te.size)
        for k, dt_ in enumerate(DTS):
            e = LC - 1 - dt_
            tau = torch.tensor([float(dt_ + 1)], device=device)
            for a in range(0, te.size, args.bs):
                j = slice(a, a + args.bs)
                x = windows(Xt, loc[j], np.full(loc[j].size, e))
                for s, net in enumerate(nets):
                    lc, lp = fwd(net, x, (doyD[te[j]] - dt_).astype(np.float32), tau)
                    Fc[s, k, j], Fp[s, k, j] = -np.expm1(-lc[:, 0]), -np.expm1(-lp[:, 0])
        np.savez(fp, rows=te, F_centre=np.clip(Fc, 1e-7, 1 - 1e-7), F_patch=np.clip(Fp, 1e-7, 1 - 1e-7), dts=np.array(DTS))
        meta["paper"] = {"file": str(fp), "n": int(te.size)}
        print(f"[bss] {fp} ({time.time() - t0:.0f} s)", flush=True)
    write_manifest(cfg, f"p40_black_swan_score_{args.model}{args.tag}",
                   args.config, meta)


if __name__ == "__main__":
    main()

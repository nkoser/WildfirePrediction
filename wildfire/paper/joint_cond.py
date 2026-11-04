"""Bedingte Vorhersage: Was sagt ein Modell ueber die Nachbarn, wenn an Tag k im Ausschnitt ein Feuer entdeckt wird?

Test-Stichtage (surv_eval_test, alle mit Pixel-Ereignissen aus region_eval), an denen das erste Feuer im
13 x 13-Ausschnitt an Tag k (1 <= k <= kmax) nach dem Stichtag ausbricht. Bekannt zur Zeit k: bis Tag k-1 brannte
im Ausschnitt nichts, an Tag k brennen die Ausloeser-Pixel (alle mit erstem Feuer an Tag k). Vorhergesagt wird fuer
jedes andere Pixel mit Chebyshev-Abstand 1..dmax zum naechsten Ausloeser die Restzeit bis zu seinem ersten Feuer
(Tag k+1..H, zensiert bei H). Die Modelle werden NICHT neu gestartet: keine neuen Eingaben, nur das Ereignis.

  lh     STS-LH  S_i(k + r) / S_i(k) aus den Kurven des Stichtags; kann auf das Ereignis nicht reagieren
  cond   Contagion  Prozess ab Tag k mit den Ausloesern als Feuer (Rao-Blackwell-MC je Pixel, joint_calib.mc_pixel)
  quiet  Contagion  wie cond, aber ohne Ausloeser: nur "bis Tag k ruhig" (gleiche Zufallszahlen wie cond);
                    cond - quiet isoliert die Wirkung des Ereignisses im Modell

Alle Modelle mit derselben Nachkalibrierung (joint_calib, --calib-dir). Metriken je Abstandsband und getrennt nach
Feuer im Fenster vor dem Stichtag: NLL der Restzeit, AUC bei r = 7 und 30 Tagen, mittleres vorhergesagtes F(r)
gegen beobachtet. Zusammenfassung: Mittel ueber Seeds je Zeile, Cluster-Bootstrap ueber die Stichtage.

  --part run      -> metrics/paper_joint/cond<tag>.{json,npz}
  --part summary  -> metrics/paper_joint/cond_summary<tag>.json

Alle Eingaben nur gelesen; bestehende Ausgaben werden nicht ueberschrieben (Abbruch).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from wildfire.audit.common import assert_readonly, load_config
from wildfire.paper.joint_calib import (cube_rates_of, load_calibrated, mc_pixel, nll_curve, norm_tensors,
                                        path_mean)
from wildfire.paper.joint_eval import lh_parts, ls_from_h, parse_models
from wildfire.paper.region_eval import H, N, P

RS = (7, 30)
BANDS = ("1", "2", "3", "4+", "all")


# ---------------------------------------------------------------------------- Auswahl
def chebyshev() -> np.ndarray:
    r, c = np.divmod(np.arange(N), P)
    return np.maximum(np.abs(r[:, None] - r[None]), np.abs(c[:, None] - c[None])).astype(np.int16)


def select(exp, args):
    sd = exp / "derived_data" / args.surv_dir
    me = dict(np.load(sd / "eval_meta.npz"))
    pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
    assert (pe["row"] == me["row"]).all() and (pe["L_abs"] == me["L_abs"]).all(), "Stichtage passen nicht"
    n_all = len(me["row"])
    evf = pe["ev"].reshape(n_all, N)
    tf = pe["t"].reshape(n_all, N).astype(np.int64)
    first = np.where(evf, tf, H + 1).min(1)
    cand = np.flatnonzero((first >= 1) & (first <= args.kmax))
    if args.n and cand.size > args.n:
        cand = np.sort(np.random.default_rng(args.seed).choice(cand, size=args.n, replace=False))
    k = first[cand]
    ev, t = evf[cand], tf[cand]
    trig = ev & (t == k[:, None])
    D = chebyshev()
    dist = np.empty((cand.size, N), np.int16)
    for a0 in range(0, cand.size, 1000):
        tr = trig[a0:a0 + 1000]
        dist[a0:a0 + 1000] = np.where(tr[:, None, :], D[None], 99).min(-1)
    keep = (~trig) & (dist >= 1) & (dist <= args.dmax)
    land, pix = np.nonzero(keep)                                  # nach Stichtag sortiert
    rows = {"land": land, "pix": pix, "dist": dist[land, pix].astype(np.int64), "e": ev[land, pix],
            "t": t[land, pix], "k": k[land], "fw": me["fire_win_patch"][cand][land].astype(bool)}
    assert (~rows["e"] | (rows["t"] > rows["k"])).all(), "Ereignis vor oder an Tag k bei einem Nicht-Ausloeser"
    t0 = time.time()
    X = torch.from_numpy(np.ascontiguousarray(np.load(sd / "eval_X.npy", mmap_mode="r")[cand]))
    print(f"[cond] {cand.size:,} Stichtage mit erstem Feuer an Tag 1..{args.kmax} (von {n_all:,}), geladen in "
          f"{time.time() - t0:.0f} s; {land.size:,} Zeilen (Pixel bis Abstand {args.dmax}); Ausloeser je Stichtag "
          f"{trig.sum(1).mean():.2f}; k Median {np.median(k):.0f}; Feuer im Fenster {rows['fw'][np.unique(land, return_index=True)[1]].mean():.3f}",
          flush=True)
    for d in range(1, args.dmax + 1):
        m = rows["dist"] == d
        print(f"[cond]   Abstand {d}: {m.sum():,} Pixel, Rest-Ereignis {rows['e'][m].mean():.4f}, "
              f"bis r=7 {(rows['e'][m] & (rows['t'][m] - rows['k'][m] <= 7)).mean():.4f}", flush=True)
    return cand, X, me["doy"][cand].astype(np.float32), trig, rows


# ---------------------------------------------------------------------------- Vorhersagen
def residual_F(ls: torch.Tensor, k: torch.Tensor) -> dict:
    base = ls.gather(1, k[:, None])[:, 0]
    return {f"F{r}": -torch.expm1(ls.gather(1, (k + r).clamp(max=H)[:, None])[:, 0] - base) for r in RS}


@torch.no_grad()
def run_model(net, kind, X, doy, trig, rows, m_t, s_t, device, args):
    """-> {meth: {nll, F7, F30}} je Zeile (numpy)."""
    out = {}
    land_all = rows["land"]
    tg = {k_: torch.from_numpy(np.asarray(rows[k_])).to(device) for k_ in ("pix", "e", "t", "k")}
    trig_t = torch.from_numpy(trig)
    M = args.M if kind == "cg" else 1
    per = max(1, args.rows // args.M)                             # Zeilen (Stichtag x Pixel) je Block, auch fuer lh
    t0 = time.time()
    for bi, r0 in enumerate(range(0, land_all.size, per)):
        r1 = min(land_all.size, r0 + per)
        ls_land = np.unique(land_all[r0:r1])
        loc = torch.from_numpy(np.searchsorted(ls_land, land_all[r0:r1])).to(device)
        x = (X[torch.from_numpy(ls_land)].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[ls_land]).to(device)
        pix, e, t, k = (tg[k_][r0:r1] for k_ in ("pix", "e", "t", "k"))
        if kind == "cg":
            with torch.autocast(device_type=device.type, enabled=False):
                obs, aa, kk, logD = net.params(x, d)
                log_exo = net.exogenous(obs, aa, kk, logD)
                lr = loc.repeat_interleave(M)
                pr, st = pix.repeat_interleave(M), k.repeat_interleave(M)
                tr = trig_t[torch.from_numpy(ls_land)].to(device)[lr]
                curves = {}
                for meth, trg in (("cond", tr), ("quiet", None)):
                    gen = torch.Generator(device=device).manual_seed(args.seed * 100_003 + bi)   # wie cond
                    curves[meth] = path_mean(mc_pixel(net, log_exo, kk, logD, lr, pr, gen, start=st, trig=trg), M)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, raw_px, _ = lh_parts(net, x, d)
            b = x.shape[0]
            lsp = ls_from_h(net.px_head.hazards(raw_px.reshape(b * N, -1), H).double().view(b, N, H))
            curves = {"lh": lsp[loc, pix]}
        for meth, ls in curves.items():
            o = out.setdefault(meth, {"nll": [], **{f"F{r}": [] for r in RS}})
            o["nll"].append(nll_curve(ls, e, t, k0=k).float().cpu().numpy())
            for key, v in residual_F(ls, k).items():
                o[key].append(v.float().cpu().numpy())
        if bi % 20 == 0:
            print(f"[cond]   Block {bi}: {r1:,}/{land_all.size:,} Zeilen ({time.time() - t0:.0f} s)", flush=True)
    return {m: {k_: np.concatenate(v) for k_, v in o.items()} for m, o in out.items()}


def band_masks(rows) -> dict:
    d = rows["dist"]
    return {"1": d == 1, "2": d == 2, "3": d == 3, "4+": d >= 4, "all": np.ones(d.size, bool)}


def strata(rows) -> dict:
    return {"all": np.ones(rows["fw"].size, bool), "fire_in_window": rows["fw"], "no_fire": ~rows["fw"]}


def auc(score: np.ndarray, y: np.ndarray) -> float:
    n1 = int(y.sum())
    if n1 == 0 or n1 == y.size:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (y.size - n1)))


def describe(pred: dict, rows) -> dict:
    """pred {nll, F7, F30} -> je Schicht und Band: NLL, AUC und F gegen beobachtet."""
    res = {}
    for sname, sm in strata(rows).items():
        for bname, bm in band_masks(rows).items():
            m = sm & bm
            if not m.any():
                continue
            r = {"n": int(m.sum()), "nll": float(pred["nll"][m].mean())}
            for rr in RS:
                y = rows["e"][m] & (rows["t"][m] - rows["k"][m] <= rr)
                r[f"auc@{rr}"] = auc(pred[f"F{rr}"][m], y)
                r[f"F@{rr}"] = float(pred[f"F{rr}"][m].mean())
                r[f"obs@{rr}"] = float(y.mean())
            res.setdefault(sname, {})[bname] = r
    return res


# ---------------------------------------------------------------------------- Laeufe
def run(cfg, exp, args, device):
    d = exp / "metrics" / "paper_joint"
    js, nz = d / f"cond{args.tag}.json", d / f"cond{args.tag}.npz"
    for p in (js, nz):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    m_t, s_t = norm_tensors(cfg, args, device)
    rates = cube_rates_of(exp, args)
    cand, X, doy, trig, rows = select(exp, args)
    res = {"n_landmarks": int(cand.size), "n_rows": int(rows["land"].size), "kmax": args.kmax, "dmax": args.dmax,
           "M": args.M, "calib_dir": args.calib_dir, "calib": {}, "models": {}}
    store = {}
    for name, kind, cks in parse_models(args.models):
        for ck in cks:
            seed = Path(ck).stem.rsplit("_s", 1)[1]
            net, used = load_calibrated(exp, args.calib_dir, ck, device, rates)
            res["calib"].setdefault(name, {})[seed] = used
            t0 = time.time()
            pred = run_model(net, kind, X, doy, trig, rows, m_t, s_t, device, args)
            for meth, p in pred.items():
                key = f"{name}_{meth}"
                res["models"].setdefault(key, {})[seed] = describe(p, rows)
                for item, v in p.items():
                    store[f"{key}__{seed}__{item}"] = v.astype(np.float32)
                a = res["models"][key][seed]["all"]
                print(f"[cond] {key:14s} {seed}  NLL " + "  ".join(f"d{b} {a[b]['nll']:.4f}" for b in BANDS if b in a)
                      + f"  AUC@7 {a['all']['auc@7']:.3f}  F@7 {a['all']['F@7']:.4f} (beob. {a['all']['obs@7']:.4f})",
                      flush=True)
            print(f"[cond] {name} {seed}: {time.time() - t0:.0f} s", flush=True)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, cand=cand, **{f"row_{k_}": v for k_, v in rows.items()}, **store)
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[cond] {js}\n[cond] {nz}", flush=True)


def boot_cluster(diff: np.ndarray, land: np.ndarray, n_boot: int, rng) -> dict:
    """Mittel von diff ueber Zeilen, Bootstrap ueber die Stichtage (Cluster); negativ = erste Methode besser."""
    u, inv = np.unique(land, return_inverse=True)
    S = np.bincount(inv, diff, minlength=u.size)
    C = np.bincount(inv, minlength=u.size).astype(np.float64)
    bm = []
    for _ in range(0, n_boot, 200):
        w = rng.multinomial(u.size, np.full(u.size, 1.0 / u.size), size=200)
        bm.append((w @ S) / (w @ C))
    bm = np.concatenate(bm)[:n_boot]
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "share_first_better": float((bm < 0).mean())}


def run_summary(cfg, exp, args):
    d = exp / "metrics" / "paper_joint"
    out = d / f"cond_summary{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderes --tag waehlen")
    rows, per = None, {}
    for f in args.inputs:
        z = np.load(d / f)
        r = {k_[4:]: z[k_] for k_ in z.files if k_.startswith("row_")}
        if rows is None:
            rows = r
        elif any(not np.array_equal(rows[k_], r[k_]) for k_ in rows):
            raise SystemExit(f"ABBRUCH: {f} hat andere Zeilen")
        for k_ in z.files:
            if k_.startswith("row_") or k_ == "cand":
                continue
            meth, seed, item = k_.split("__")
            per.setdefault(meth, {}).setdefault(item, {})[seed] = z[k_].astype(np.float64)
    pred = {m: {it: np.mean(list(s.values()), 0) for it, s in v.items()} for m, v in per.items()}
    res = {"inputs": args.inputs, "n_seeds": {m: len(v["nll"]) for m, v in per.items()},
           "metrics": {m: describe(p, rows) for m, p in pred.items()}, "boot": {}}
    rng = np.random.default_rng(args.seed + 13)
    bands, st = band_masks(rows), strata(rows)
    for ref in args.refs:
        for m in pred:
            if m == ref:
                continue
            for sname, sm in st.items():
                for bname, bm in bands.items():
                    msk = sm & bm
                    if not msk.any():
                        continue
                    b = boot_cluster(pred[ref]["nll"][msk] - pred[m]["nll"][msk], rows["land"][msk], args.boot, rng)
                    res["boot"].setdefault(f"{ref}-{m}", {}).setdefault(sname, {})[bname] = b
                    if sname == "all":
                        print(f"[cond] {ref} - {m:14s} Abstand {bname:4s}: {b['mean']:+.5f} [{b['lo']:+.5f}, "
                              f"{b['hi']:+.5f}]", flush=True)
    for m, mt in res["metrics"].items():
        a = mt["all"]
        print(f"[cond] {m:14s} NLL " + "  ".join(f"d{b} {a[b]['nll']:.4f}" for b in BANDS if b in a)
              + f"  AUC@7 {a['all']['auc@7']:.3f}  AUC@30 {a['all']['auc@30']:.3f}", flush=True)
    d.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[cond] {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=["run", "summary"], required=True)
    ap.add_argument("--models", nargs="+", default=[], help="name=cg|lh:glob")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--n", type=int, default=0, help="hoechstens so viele Stichtage (0 = alle)")
    ap.add_argument("--kmax", type=int, default=70)
    ap.add_argument("--dmax", type=int, default=4)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--rows", type=int, default=65_536, help="Zeilen (Stichtag x Pixel x Pfad) je Block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", nargs="+", default=[])
    ap.add_argument("--refs", nargs="+", default=[])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if args.part == "summary":
        run_summary(cfg, exp, args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(cfg, exp, args, device)


if __name__ == "__main__":
    main()

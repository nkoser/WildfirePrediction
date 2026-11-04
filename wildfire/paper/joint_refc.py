"""Staerkere Vergleiche fuer die bedingte Vorhersage (joint_cond): Neustart an Tag k und angepasste Basismodelle.

joint_cond vergleicht Contagion (Prozess ab Tag k mit den Ausloesern) mit STS-LH aus den Kurven des Stichtags, das
auf das Ereignis nicht reagieren kann. Hier kommen die Vergleiche dazu, die ein Gutachter verlangen wuerde:

  lh_refc  STS-LH neu gestartet an Tag k: neuer Stichtag L + k mit den Eingaben bis L + k (das Feuer der Ausloeser
           steht dann im Feuerkanal); Restzeit r = t - k direkt aus der neuen Kurve
  cg_refc  Contagion ebenso neu gestartet (Rollout ab L + k, Rao-Blackwell-MC je Pixel wie joint_cond)
  lh_adj   STS-LH-Kurven des Stichtags, nach dem Ereignis je Abstandsband korrigiert:
           logit h' = logit h + b_band + c_band * log r, b und c (8 Zahlen) auf den Val-Jahren gefittet
  emp      Faustregel ohne Eingaben: Hazard der Restzeit je Abstandsband und r aus den Val-Jahren

Zeilen wie joint_cond (gleiche --kmax, --dmax, --n, --seed), eingeschraenkt auf Stichtage, deren Tag L + k samt
30-Tage-Fenster im Wuerfel liegt (Saisonende!). Alle Modelle mit derselben Nachkalibrierung (joint_calib).

  --part build    Fenster an L + k -> derived_data/joint_refc/<name>_{X.npy,meta.npz} (CPU, Wuerfel)
  --part run      -> metrics/paper_joint/refc<tag>.{json,npz}
  --part summary  joint_cond-Ausgaben (cond_*.npz) und refc-Ausgaben auf denselben Zeilen zusammen; Metriken wie
                  joint_cond, dazu je Testjahr; Cluster-Bootstrap gegen --refs
                  -> metrics/paper_joint/refc_summary<tag>.json

Alle Eingaben nur gelesen; bestehende Ausgaben werden nicht ueberschrieben (Abbruch).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config
from wildfire.paper import joint_cond
from wildfire.paper.joint_calib import (EV_DIR as CALIB_EV_DIR, cube_rates_of, load_calibrated, mc_pixel,
                                        nll_curve, norm_tensors, path_mean)
from wildfire.paper.joint_cond import BANDS, band_masks, boot_cluster, chebyshev, describe, residual_F
from wildfire.paper.joint_eval import lh_parts, ls_from_h, parse_models
from wildfire.paper.region_eval import H, N, P

OUT_DIR = "joint_refc"
DAY0 = dt.date(2000, 1, 1)
NB = 4                                                              # Abstandsbaender 1, 2, 3, 4+


# ---------------------------------------------------------------------------- Auswahl (wie joint_cond.select)
def cond_rows(me, evf, tf, args):
    """Kandidaten und Zeilen ohne Eingaben; dieselbe Logik wie joint_cond.select."""
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
        dist[a0:a0 + 1000] = np.where(trig[a0:a0 + 1000, :, None], D[None], 99).min(1)
    keep = (~trig) & (dist >= 1) & (dist <= args.dmax)
    land, pix = np.nonzero(keep)
    rows = {"land": land, "pix": pix, "dist": dist[land, pix].astype(np.int64), "e": ev[land, pix],
            "t": t[land, pix], "k": k[land], "fw": me["fire_win_patch"][cand][land].astype(bool)}
    return cand, k, trig, rows


def test_events(exp, args):
    me = dict(np.load(exp / "derived_data" / args.surv_dir / "eval_meta.npz"))
    pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
    assert (pe["row"] == me["row"]).all() and (pe["L_abs"] == me["L_abs"]).all(), "Stichtage passen nicht"
    n = len(me["row"])
    return me, pe["ev"].reshape(n, N), pe["t"].reshape(n, N).astype(np.int64)


def val_events(exp, args):
    me = dict(np.load(exp / "derived_data" / args.surv_dir / "val_meta.npz"))
    pe = dict(np.load(exp / "derived_data" / CALIB_EV_DIR / "val_pixel_events.npz"))
    assert (pe["row"] == me["row"]).all() and (pe["L_abs"] == me["L_abs"]).all(), "Val-Stichtage passen nicht"
    n = len(me["row"])
    return me, pe["ev"].reshape(n, N), pe["t"].reshape(n, N).astype(np.int64)


def build_paths(exp, name):
    d = exp / "derived_data" / OUT_DIR
    return d / f"{name}_X.npy", d / f"{name}_meta.npz"


# ---------------------------------------------------------------------------- Fenster an L + k
def build(cfg, exp, args):
    from numpy.lib.format import open_memmap

    from wildfire.data.dataset import CubeReader
    from wildfire.paper.surv_prepare import T_WIN, window

    fx, fm = build_paths(exp, args.build_name)
    for p in (fx, fm):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --build-name waehlen")
    me, evf, tf = test_events(exp, args)
    cand, k, trig, _ = cond_rows(me, evf, tf, args)
    cube = CubeReader(exp)
    pos_of = {d: i for i, d in enumerate(cube.dates)}
    Lp = np.full(cand.size, -1, np.int64)
    for j, (la, kk) in enumerate(zip(me["L_abs"][cand], k)):
        g = pos_of.get((DAY0 + dt.timedelta(days=int(la) + int(kk))).isoformat())
        if g is None or g not in cube._pos or (g - T_WIN + 1) not in cube._pos:
            continue
        a, b = cube._pos[g - T_WIN + 1], cube._pos[g]
        if a[0] == b[0] and b[1] - a[1] == T_WIN - 1:
            Lp[j] = g
    valid = Lp >= 0
    doy = np.array([dt.date.fromisoformat(cube.dates[int(g)]).timetuple().tm_yday if g >= 0 else 0 for g in Lp],
                   np.float32)
    print(f"[refc] {cand.size:,} Stichtage mit erstem Feuer an Tag 1..{args.kmax}; davon {valid.sum():,} "
          f"({valid.mean():.3f}) mit Fenster an L + k im Wuerfel", flush=True)
    fx.parent.mkdir(parents=True, exist_ok=True)
    X = open_memmap(fx, mode="w+", dtype=np.uint8, shape=(cand.size, T_WIN, 55, P, P))
    rr, cc = me["row"][cand], me["col"][cand]
    order = np.flatnonzero(valid)[np.argsort(Lp[valid], kind="stable")]
    for m, j in enumerate(order):
        X[j] = window(cube, int(Lp[j]), int(rr[j]), int(cc[j]))
        if m % 2000 == 0:
            print(f"    {m}/{order.size}", flush=True)
    X.flush()
    # Selbsttest: die Ausloeser brennen am letzten Tag des neuen Fensters (Feuerkanal 0); vorher im Ausschnitt nichts
    # seit L (erstes Feuer erst an Tag k)
    Xm = np.load(fx, mmap_mode="r")
    chk = np.flatnonzero(valid)[:: max(1, int(valid.sum()) // 2000)]
    last = np.asarray(Xm[chk, -1, 0]).reshape(chk.size, N) > 0
    hit = (last & trig[chk]).any(1).mean()
    pre = np.array([np.asarray(Xm[j, max(0, T_WIN - int(k[j])):T_WIN - 1, 0]).any() for j in chk])
    print(f"[refc] Selbsttest: Ausloeser im Feuerkanal am Tag L + k {hit:.4f}; Feuer im Ausschnitt an L+1..L+k-1 "
          f"{pre.mean():.4f}", flush=True)
    if hit < 0.95:
        raise SystemExit("ABBRUCH: Ausloeser fehlen im Feuerkanal des neuen Fensters")
    np.savez(fm, cand=cand, k=k, L_cube_new=Lp, doy=doy, valid=valid, selftest_hit=hit,
             selftest_pre=float(pre.mean()))
    print(f"[refc] {fx}\n[refc] {fm}", flush=True)


# ---------------------------------------------------------------------------- Bausteine
def shift_curve(ls_new: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """log S (R, H+1) ab dem neuen Stichtag L + k -> auf der alten Zeitachse, bis Tag k konstant 0."""
    j = (torch.arange(H + 1, device=ls_new.device)[None] - k[:, None]).clamp(min=0)
    return ls_new.gather(1, j)


def residual_curve(logit_h: torch.Tensor, k: torch.Tensor, dl: torch.Tensor | None) -> torch.Tensor:
    """Tages-Hazard-Logits (R, H) an Tag 1..H, ab Tag k + 1 um dl (R, H) verschoben -> log S (R, H+1), 0 bis Tag k."""
    s = torch.arange(1, H + 1, device=logit_h.device)[None]
    after = s > k[:, None]
    z = logit_h + (dl if dl is not None else 0.0)
    return F.pad(torch.where(after, F.logsigmoid(-z), torch.zeros_like(z)).cumsum(1), (1, 0))


def band_of(dist: torch.Tensor) -> torch.Tensor:
    return dist.clamp(1, NB) - 1


def adj_shift(theta: torch.Tensor, band: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r = (torch.arange(1, H + 1, device=k.device)[None] - k[:, None]).clamp(min=1).double()
    return theta[band, 0][:, None] + theta[band, 1][:, None] * torch.log(r)


def logit_from_ls(ls: torch.Tensor) -> torch.Tensor:
    """log S (R, H+1) -> Logits der Tages-Hazards (R, H)."""
    l1mh = (ls[:, 1:] - ls[:, :-1]).clamp(max=-1e-12)
    return torch.log(-torch.expm1(l1mh)) - l1mh


@torch.no_grad()
def lh_curves(net, X, doy, land_idx, m_t, s_t, device, bs):
    """log S (len(land_idx), N, H+1) der Pixelkoepfe, in Bloecken (CPU, float64)."""
    out = []
    for a0 in range(0, land_idx.size, bs):
        li = torch.from_numpy(land_idx[a0:a0 + bs])
        x = (X[li].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[land_idx[a0:a0 + bs]]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, raw_px, _ = lh_parts(net, x, d)
        b = x.shape[0]
        out.append(ls_from_h(net.px_head.hazards(raw_px.reshape(b * N, -1), H).double().view(b, N, H)).cpu())
    return torch.cat(out)


def fit_adj(ls_rows, rows, device, iters=200):
    """theta (NB, 2): logit h' = logit h + theta[band, 0] + theta[band, 1] * log r, NLL der Restzeit minimal."""
    z = logit_from_ls(ls_rows.to(device))
    band = band_of(torch.from_numpy(rows["dist"]).to(device))
    k = torch.from_numpy(rows["k"]).to(device)
    e = torch.from_numpy(rows["e"]).to(device)
    t = torch.from_numpy(rows["t"]).to(device)
    theta = torch.zeros(NB, 2, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll_curve(residual_curve(z, k, adj_shift(theta, band, k)), e, t, k0=k).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        v0 = float(nll_curve(residual_curve(z, k, None), e, t, k0=k).mean())
        v1 = float(nll_curve(residual_curve(z, k, adj_shift(theta, band, k)), e, t, k0=k).mean())
    return theta.detach(), v0, v1


def fit_emp(rows, min_risk=200):
    """Hazard der Restzeit je Band und r (NB, H) aus den Val-Zeilen; r mit wenig Risikomenge gepoolt ab dort."""
    band = np.clip(rows["dist"], 1, NB) - 1
    r_ev = np.where(rows["e"], rows["t"] - rows["k"], 0)
    r_end = np.where(rows["e"], rows["t"] - rows["k"], H - rows["k"])          # letzter beobachteter Resttag
    haz = np.zeros((NB, H))
    for b in range(NB):
        m = band == b
        ev = np.bincount(r_ev[m][rows["e"][m]], minlength=H + 1)[1:H + 1].astype(float)
        risk = np.bincount(r_end[m], minlength=H + 2)[::-1].cumsum()[::-1][1:H + 1].astype(float)
        cut = np.flatnonzero(risk < min_risk)
        c0 = int(cut[0]) if cut.size else H
        haz[b, :c0] = ev[:c0] / np.maximum(risk[:c0], 1)
        if c0 < H:
            haz[b, c0:] = ev[c0:].sum() / max(risk[c0:].sum(), 1)
    return np.clip(haz, 1e-6, 1 - 1e-6)


def emp_curve(haz, rows, device):
    band = band_of(torch.from_numpy(rows["dist"]).to(device))
    k = torch.from_numpy(rows["k"]).to(device)
    s = torch.arange(1, H + 1, device=device)[None]
    r = (s - k[:, None]).clamp(min=1, max=H) - 1
    hz = torch.from_numpy(haz).to(device)[band[:, None], r]
    return residual_curve(torch.log(hz) - torch.log1p(-hz), k, None)


def collect(store_o, ls, rows_t):
    e, t, k = rows_t
    store_o.setdefault("nll", []).append(nll_curve(ls, e, t, k0=k).float().cpu().numpy())
    for key, v in residual_F(ls, k).items():
        store_o.setdefault(key, []).append(v.float().cpu().numpy())


# ---------------------------------------------------------------------------- Lauf
def run(cfg, exp, args, device):
    d = exp / "metrics" / "paper_joint"
    js, nz = d / f"refc{args.tag}.json", d / f"refc{args.tag}.npz"
    for p in (js, nz):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    m_t, s_t = norm_tensors(cfg, args, device)
    rates = cube_rates_of(exp, args)
    cand, X0, doy0, trig, rows_all = joint_cond.select(exp, args)
    fx, fm = build_paths(exp, args.build_name)
    meta = dict(np.load(fm))
    if not np.array_equal(meta["cand"], cand):
        raise SystemExit("ABBRUCH: Stichtage des Builds passen nicht zu --kmax/--n/--seed")
    valid = meta["valid"].astype(bool)
    rowsel = np.flatnonzero(valid[rows_all["land"]])
    rows = {k_: v[rowsel] for k_, v in rows_all.items()}
    Xr = torch.from_numpy(np.load(fx))
    doyr = meta["doy"].astype(np.float32)
    print(f"[refc] {rowsel.size:,} von {rows_all['land'].size:,} Zeilen mit Neustart-Fenster "
          f"({np.unique(rows['land']).size:,} Stichtage)", flush=True)
    tgt = {k_: torch.from_numpy(np.asarray(rows[k_])).to(device) for k_ in ("pix", "e", "t", "k", "dist")}
    res = {"n_rows": int(rowsel.size), "n_landmarks": int(np.unique(rows["land"]).size), "calib_dir": args.calib_dir,
           "calib": {}, "adj": {}, "models": {}}
    store = {}

    def finish(key, seed, o):
        p = {k_: np.concatenate(v) for k_, v in o.items()}
        res["models"].setdefault(key, {})[seed] = describe(p, rows)
        for item, v in p.items():
            store[f"{key}__{seed}__{item}"] = v.astype(np.float32)
        a = res["models"][key][seed]["all"]
        print(f"[refc] {key:14s} {seed}  NLL " + "  ".join(f"d{b} {a[b]['nll']:.4f}" for b in BANDS if b in a)
              + f"  AUC@7 {a['all']['auc@7']:.3f}  F@7 {a['all']['F@7']:.4f} (beob. {a['all']['obs@7']:.4f})",
              flush=True)

    val = None
    if args.with_baselines:                                       # Val-Zeilen fuer lh_adj und emp
        me_v, evf_v, tf_v = val_events(exp, args)
        cand_v, _, _, rows_v = cond_rows(me_v, evf_v, tf_v, argparse.Namespace(kmax=args.kmax, dmax=args.dmax,
                                                                                 n=0, seed=args.seed))
        Xv = torch.from_numpy(np.ascontiguousarray(
            np.load(exp / "derived_data" / args.surv_dir / "val_X.npy", mmap_mode="r")[cand_v]))
        val = (cand_v, rows_v, Xv, me_v["doy"][cand_v].astype(np.float32))
        haz = fit_emp(rows_v)
        res["emp_hazard_r1"] = {b: float(haz[i, 0]) for i, b in enumerate(("1", "2", "3", "4+"))}
        print(f"[refc] Val: {cand_v.size:,} Stichtage, {rows_v['land'].size:,} Zeilen; emp Hazard r=1 je Band "
              f"{np.round(haz[:, 0], 4).tolist()}", flush=True)
        o = {}
        for r0 in range(0, rowsel.size, args.rows):
            sl = slice(r0, r0 + args.rows)
            rr = {k_: v[sl] for k_, v in rows.items()}
            collect(o, emp_curve(haz, rr, device), (tgt["e"][sl], tgt["t"][sl], tgt["k"][sl]))
        finish("emp", "all", o)

    for name, kind, cks in parse_models(args.models):
        for ck in cks:
            seed = Path(ck).stem.rsplit("_s", 1)[1]
            net, used = load_calibrated(exp, args.calib_dir, ck, device, rates)
            res["calib"].setdefault(name, {})[seed] = used
            t0 = time.time()
            if kind == "lh":
                theta = None
                if val is not None:
                    cand_v, rows_v, Xv, doy_v = val
                    lsv = lh_curves(net, Xv, doy_v, np.arange(cand_v.size), m_t, s_t, device, args.bs)
                    theta, v0, v1 = fit_adj(lsv[rows_v["land"], rows_v["pix"]], rows_v, device)
                    res["adj"].setdefault(name, {})[seed] = {"theta": theta.cpu().tolist(), "val_before": v0,
                                                             "val_after": v1}
                    print(f"[refc] {name} {seed} lh_adj theta {np.round(theta.cpu().numpy(), 3).tolist()}; "
                          f"Val-NLL {v0:.4f} -> {v1:.4f}", flush=True)
                    del lsv
                o_ref, o_adj = {}, {}
                lands = np.unique(rows["land"])
                for a0 in range(0, lands.size, args.bs * 8):
                    ll = lands[a0:a0 + args.bs * 8]
                    r0, r1 = np.searchsorted(rows["land"], ll[0]), np.searchsorted(rows["land"], ll[-1], "right")
                    loc = torch.from_numpy(np.searchsorted(ll, rows["land"][r0:r1]))
                    pix = torch.from_numpy(rows["pix"][r0:r1])
                    rt = tuple(tgt[k_][r0:r1] for k_ in ("e", "t", "k"))
                    ls_new = lh_curves(net, Xr, doyr, ll, m_t, s_t, device, args.bs)[loc, pix].to(device)
                    collect(o_ref, shift_curve(ls_new, rt[2]), rt)
                    if theta is not None:
                        ls_old = lh_curves(net, X0, doy0, ll, m_t, s_t, device, args.bs)[loc, pix].to(device)
                        dl = adj_shift(theta, band_of(tgt["dist"][r0:r1]), rt[2])
                        collect(o_adj, residual_curve(logit_from_ls(ls_old), rt[2], dl), rt)
                finish(f"{name}_refc", seed, o_ref)
                if theta is not None:
                    finish(f"{name}_adj", seed, o_adj)
            else:
                o = {}
                M = args.M
                per = max(1, args.mc_rows // M)
                for bi, r0 in enumerate(range(0, rowsel.size, per)):
                    r1 = min(rowsel.size, r0 + per)
                    ll = np.unique(rows["land"][r0:r1])
                    loc = torch.from_numpy(np.searchsorted(ll, rows["land"][r0:r1])).to(device)
                    x = (Xr[torch.from_numpy(ll)].to(device).float() - m_t) / s_t
                    dd = torch.from_numpy(doyr[ll]).to(device)
                    rt = tuple(tgt[k_][r0:r1] for k_ in ("e", "t", "k"))
                    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                        obs, aa, kk, logD = net.params(x, dd)
                        log_exo = net.exogenous(obs, aa, kk, logD)
                        gen = torch.Generator(device=device).manual_seed(args.seed * 100_003 + bi)
                        ls_new = path_mean(mc_pixel(net, log_exo, kk, logD, loc.repeat_interleave(M),
                                                    tgt["pix"][r0:r1].repeat_interleave(M), gen), M)
                    collect(o, shift_curve(ls_new, rt[2]), rt)
                    if bi % 20 == 0:
                        print(f"[refc]   Block {bi}: {r1:,}/{rowsel.size:,} Zeilen ({time.time() - t0:.0f} s)",
                              flush=True)
                finish(f"{name}_refc", seed, o)
            print(f"[refc] {name} {seed}: {time.time() - t0:.0f} s", flush=True)
            del net
            torch.cuda.empty_cache()
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(nz, cand=cand, rowsel=rowsel, **{f"row_{k_}": v for k_, v in rows.items()}, **store)
    js.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[refc] {js}\n[refc] {nz}", flush=True)


# ---------------------------------------------------------------------------- Zusammenfassung
def load_preds(d, files, rowsel=None):
    rows, per, cand = None, {}, None
    for f in files:
        z = np.load(d / f)
        sel = rowsel if (rowsel is not None and "rowsel" not in z.files) else slice(None)
        r = {k_[4:]: z[k_][sel] for k_ in z.files if k_.startswith("row_")}
        if rows is None:
            rows, cand = r, z["cand"]
        elif any(not np.array_equal(rows[k_], r[k_]) for k_ in rows) or not np.array_equal(cand, z["cand"]):
            raise SystemExit(f"ABBRUCH: {f} hat andere Zeilen")
        for k_ in z.files:
            if k_.startswith("row_") or k_ in ("cand", "rowsel"):
                continue
            meth, seed, item = k_.split("__")
            per.setdefault(meth, {}).setdefault(item, {})[seed] = z[k_][sel].astype(np.float64)
    return rows, per, cand


def run_summary(cfg, exp, args):
    d = exp / "metrics" / "paper_joint"
    out = d / f"refc_summary{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderes --tag waehlen")
    rowsel = np.load(d / args.refc_inputs[0])["rowsel"]
    rows, per = None, {}
    for files in (args.refc_inputs, args.inputs):
        r, p, cand = load_preds(d, files, rowsel)
        if rows is not None and any(not np.array_equal(rows[k_], r[k_]) for k_ in rows):
            raise SystemExit("ABBRUCH: joint_cond- und refc-Zeilen passen nicht zusammen")
        rows = r
        for m, v in p.items():
            if m in per:
                raise SystemExit(f"ABBRUCH: Methode {m} doppelt")
            per[m] = v
    me = dict(np.load(exp / "derived_data" / args.surv_dir / "eval_meta.npz"))
    year = np.array([(DAY0 + dt.timedelta(days=int(v))).year for v in me["L_abs"][cand]])[rows["land"]]
    pred = {m: {it: np.mean(list(s.values()), 0) for it, s in v.items()} for m, v in per.items()}
    res = {"inputs": args.inputs + args.refc_inputs, "n_rows": int(rows["land"].size),
           "n_landmarks": int(np.unique(rows["land"]).size), "n_seeds": {m: len(v["nll"]) for m, v in per.items()},
           "metrics": {m: describe(p, rows) for m, p in pred.items()}, "by_year": {}, "boot": {}, "boot_by_year": {}}
    for y in np.unique(year):
        m_ = year == y
        ry = {k_: v[m_] for k_, v in rows.items()}
        res["by_year"][str(y)] = {m: describe({it: v[m_] for it, v in p.items()}, ry) for m, p in pred.items()}
    rng = np.random.default_rng(args.seed + 17)
    bands = band_masks(rows)
    for ref in args.refs:
        for m in pred:
            if m == ref:
                continue
            for bname, bm in bands.items():
                b = boot_cluster(pred[ref]["nll"][bm] - pred[m]["nll"][bm], rows["land"][bm], args.boot, rng)
                res["boot"].setdefault(f"{ref}-{m}", {})[bname] = b
                print(f"[refc] {ref} - {m:14s} Abstand {bname:4s}: {b['mean']:+.5f} [{b['lo']:+.5f}, {b['hi']:+.5f}]",
                      flush=True)
            for y in np.unique(year):
                m_ = year == y
                res["boot_by_year"].setdefault(f"{ref}-{m}", {})[str(y)] = boot_cluster(
                    pred[ref]["nll"][m_] - pred[m]["nll"][m_], rows["land"][m_], args.boot, rng)
    for m, mt in res["metrics"].items():
        a = mt["all"]
        print(f"[refc] {m:14s} NLL " + "  ".join(f"d{b} {a[b]['nll']:.4f}" for b in BANDS if b in a)
              + f"  AUC@7 {a['all']['auc@7']:.3f}  AUC@30 {a['all']['auc@30']:.3f}  F@7 {a['all']['F@7']:.4f} "
              f"(beob. {a['all']['obs@7']:.4f})", flush=True)
    d.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[refc] {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=["build", "run", "summary"], required=True)
    ap.add_argument("--models", nargs="+", default=[], help="name=cg|lh:glob")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--build-name", default="test")
    ap.add_argument("--n", type=int, default=0, help="hoechstens so viele Stichtage (0 = alle), wie joint_cond")
    ap.add_argument("--kmax", type=int, default=70)
    ap.add_argument("--dmax", type=int, default=4)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--mc-rows", type=int, default=65_536, help="cg: Zeilen (Stichtag x Pixel x Pfad) je Block")
    ap.add_argument("--rows", type=int, default=200_000, help="emp: Zeilen je Block")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--with-baselines", action="store_true", help="run: auch lh_adj (je lh-Modell) und emp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", nargs="+", default=[], help="summary: joint_cond-Ausgaben (cond_*.npz)")
    ap.add_argument("--refc-inputs", nargs="+", default=[], help="summary: refc-Ausgaben (refc_*.npz)")
    ap.add_argument("--refs", nargs="+", default=[])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if args.part == "build":
        build(cfg, exp, args)
        return
    if args.part == "summary":
        run_summary(cfg, exp, args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(cfg, exp, args, device)


if __name__ == "__main__":
    main()

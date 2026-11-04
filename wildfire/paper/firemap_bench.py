"""Benchmark der Feuerkarten (Regions-Simulation) ueber alle groesseren Feuer eines Splits statt drei Beispielen.

Drei Schritte (je ein Aufruf, Ausgaben unter metrics/firemap_bench<tag>/):
  events  Ereignisse aus fire_index.npz: Stichtag L (alle --step Tage) und Feuerkomplex = Zusammenhangskomponente
          der MODIS-Feuer der letzten 3 Eingabetage (Dilatation 3 px), >= --min-act Pixel; Region = Rahmen der
          Komponente + --margin, hoechstens --max-box; >= --min-new neue Feuerpixel in L + 1 .. L + 30. Gierig
          nach Zuwachs, ohne Ueberlappung im selben Jahr innerhalb von 25 Tagen, hoechstens --max-events je Split.
          -> events_<split>.json
  run     ein Modell (name=glob ueber die Seeds, Netze wie contagion_mc_map2) oder die Abstands-Baseline (dist) auf
          allen Ereignissen eines Splits; Kern-Skalierung k' = 1 - (1 - k)^s fuer jedes s aus --scales (Seeds je
          --paths MC-Pfade, Mean-Field mit). Fuer dist: Radien statt Skalen. -> runs/<split>_<name>.json,
          maps/<split>_<name>.npz (P(Ankunft <= 7 / 30) je Ereignis und Skala, float16)
  report  Skala bzw. Radius je Modell auf val gewaehlt (bester mittlerer IoU 30 d der MC-Karte), dazu immer s = 1;
          Tabellen fuer test. -> summary.md / summary.json

Metriken je Ereignis und Horizont t in {7, 30} auf den Landpixeln der Region ohne Feuer im Eingabefenster:
  ap / auc       P(Ankunft <= t) gegen beobachtet (erster MODIS-Feuertag nach L, Kanal 0)
  iou / prec / rec  Karte P >= 0.5 (Median-Karte der Pfade)
  iou_topk       die n_obs wahrscheinlichsten Pixel (Form ohne Flaechenfehler)
  iou_path       mittlerer IoU einzelner Pfade (wie realistisch eine Simulation ist)
  log_area       log(mittlere verbrannte Pixel je Pfad / n_obs)
  assd           symmetrischer mittlerer Abstand (px) zwischen Karte P >= 0.5 und beobachteten Pixeln
  brier          Brier-Score von P(Ankunft <= t)
Baseline dist: Score = -Abstand zum Feuer der letzten 3 Tage; Karte = Abstand <= r.
Pixelweise ohne Kopplung (--model name=indep:glob): F(t) des Mittelpixels je zentriertem Fenster (risk_map.predict,
Contagion mit Rollout nur im eigenen 13 x 13-Fenster), keine Regionssimulation.
report: zusaetzlich Schwelle q der Karte P >= q je Modell und Horizont auf val gewaehlt (bester mittlerer IoU).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.contagion_mc_map import T
from wildfire.paper.contagion_mc_map2 import load_net, pixel_params, simulate
from wildfire.paper.contagion_spread import WX
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.risk_map import predict
from wildfire.paper.surv_prepare import CACHE_FROM_CUBE, HALF, T_WIN

HOR = (7, 30)
ACT_DAYS = 3


def split_years(exp: Path) -> dict[str, list[int]]:
    cd = exp / "derived_data" / "paper" / "cache_G_L61"
    meta, split = np.load(cd / "meta.npy"), np.load(cd / "split.npy")
    return {sp: sorted({int(m // 10000) for m in meta[split == sp, 0]}) for sp in ("train", "val", "test")}


def load_box(cube: CubeReader, L: int, r0: int, r1: int, c0: int, c1: int, pad: int, days: int = T_WIN) -> np.ndarray:
    """Wie risk_map.load_slab, aber nur Region [r0, r1) x [c0, c1) + pad; ausserhalb des Gitters 0."""
    year, li = cube._pos[L]
    dyn, stat = cube._arrays(year)
    R0, R1, C0, C1 = max(0, r0 - pad), min(cube.H, r1 + pad), max(0, c0 - pad), min(cube.W, c1 + pad)
    d = np.asarray(dyn[li - days + 1:li + 1, R0:R1, C0:C1])
    s = np.asarray(stat[R0:R1, C0:C1])
    out = np.zeros((days, len(CACHE_FROM_CUBE), r1 - r0 + 2 * pad, c1 - c0 + 2 * pad), np.uint8)
    rr, cc = slice(R0 - (r0 - pad), R1 - (r0 - pad)), slice(C0 - (c0 - pad), C1 - (c0 - pad))
    for ch, (src, i) in enumerate(CACHE_FROM_CUBE):
        out[:, ch, rr, cc] = d[..., i] if src == "dyn" else s[..., i][None]
    return out


def first_fire_box(cube: CubeReader, L: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Erster Feuertag 1 .. T nach L in der Region, 31 = keins (Kanal 0 wie risk_map.future_fire)."""
    year, li = cube._pos[L]
    dyn, _ = cube._arrays(year)
    f = np.asarray(dyn[li + 1:li + T + 1, r0:r1, c0:c1, 0]) > 0
    return np.where(f.any(0), f.argmax(0) + 1, 31).astype(np.int16)


# ----------------------------------------------------------------------------------------------- events
def select_events(exp: Path, cube: CubeReader, years: list[int], args) -> list[dict]:
    z = np.load(exp / "derived_data" / "events" / "fire_index.npz")
    H, W = (int(v) for v in z["grid"])
    pix = np.repeat(z["fire_pixels"], np.diff(z["fire_offsets"]))
    days = z["fire_days"]
    mask = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    cand = []
    for L in sorted(cube._pos):
        year, li = cube._pos[L]
        if int(year) not in years or (L - T_WIN + 1) not in cube._pos or (L + T) not in cube._pos:
            continue
        if cube._pos[L + T][0] != year or cube._pos[L - T_WIN + 1][0] != year:
            continue
        if li % args.step_days:
            continue
        act = np.zeros(H * W, bool)
        act[pix[(days > L - ACT_DAYS) & (days <= L)]] = True
        act = act.reshape(H, W)
        if act.sum() < args.min_act:
            continue
        win = np.zeros(H * W, bool)
        win[pix[(days > L - T_WIN) & (days <= L)]] = True
        new = np.zeros(H * W, bool)
        new[pix[(days > L) & (days <= L + T)]] = True
        new = new.reshape(H, W) & ~win.reshape(H, W) & mask
        lab, n = ndimage.label(ndimage.binary_dilation(act, iterations=3))
        for j, sl in enumerate(ndimage.find_objects(lab), 1):
            comp = (lab[sl] == j) & act[sl]
            if comp.sum() < args.min_act:
                continue
            rc, cc = (int(round(v)) for v in ndimage.center_of_mass(comp))
            rc, cc = rc + sl[0].start, cc + sl[1].start
            r0, r1 = sl[0].start - args.margin, sl[0].stop + args.margin
            c0, c1 = sl[1].start - args.margin, sl[1].stop + args.margin
            if r1 - r0 > args.max_box:
                r0, r1 = rc - args.max_box // 2, rc - args.max_box // 2 + args.max_box
            if c1 - c0 > args.max_box:
                c0, c1 = cc - args.max_box // 2, cc - args.max_box // 2 + args.max_box
            r0, r1, c0, c1 = max(0, r0), min(H, r1), max(0, c0), min(W, c1)
            n_new = int(new[r0:r1, c0:c1].sum())
            if n_new >= args.min_new:
                cand.append({"L": int(L), "date": cube.dates[L], "year": int(year), "box": [r0, r1, c0, c1],
                             "n_act": int(comp.sum()), "n_new_30": n_new})
    cand.sort(key=lambda e: -e["n_new_30"])
    out = []
    for e in cand:
        r0, r1, c0, c1 = e["box"]
        if any(o["year"] == e["year"] and abs(o["L"] - e["L"]) < 25 and r0 < o["box"][1] and o["box"][0] < r1
               and c0 < o["box"][3] and o["box"][2] < c1 for o in out):
            continue
        out.append(e)
        if len(out) >= args.max_events:
            break
    return sorted(out, key=lambda e: (e["L"], e["box"]))


# ----------------------------------------------------------------------------------------------- metrics
def assd(pred: np.ndarray, obs: np.ndarray) -> float:
    if not pred.any() or not obs.any():
        return float("nan")
    d_obs, d_pred = ndimage.distance_transform_edt(~obs), ndimage.distance_transform_edt(~pred)
    return float((d_obs[pred].sum() + d_pred[obs].sum()) / (pred.sum() + obs.sum()))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def map_metrics(P: np.ndarray, y: np.ndarray, m: np.ndarray, paths: np.ndarray | None) -> dict:
    """P, y (h, w) auf der Region, m Auswertemaske; paths (n, h, w) bool oder None (deterministische Karte)."""
    p, t = P[m], y[m]
    n_obs = int(t.sum())
    r = {"n_obs": n_obs, "brier": float(np.mean((p - t) ** 2))}
    if 0 < n_obs < t.size:
        r["ap"], r["auc"] = float(average_precision_score(t, p)), float(roc_auc_score(t, p))
    pred = (P >= 0.5) & m
    yy = y & m
    r["iou"], r["assd"] = iou(pred, yy), assd(pred, yy)
    r["prec"] = float((pred & yy).sum() / pred.sum()) if pred.any() else float("nan")
    r["rec"] = float((pred & yy).sum() / max(n_obs, 1))
    if n_obs:
        thr = np.sort(p)[::-1][n_obs - 1]
        r["iou_topk"] = iou((P >= thr) & m, yy)
    if paths is not None:
        pm = paths & m[None]
        r["iou_path"] = float(np.mean([iou(q, yy) for q in pm]))
        r["log_area"] = float(np.log((pm.sum((1, 2)).mean() + 1) / (n_obs + 1)))
    else:
        r["log_area"] = float(np.log((pred.sum() + 1) / (n_obs + 1)))
    return r


# ----------------------------------------------------------------------------------------------- run
def run(exp: Path, cfg, cube: CubeReader, events: list[dict], name: str, pattern: str, args) -> tuple[list, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_all = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    nets = []
    indep = pattern.startswith("indep:")
    pattern = pattern.removeprefix("indep:")
    if name != "dist":
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"ABBRUCH: keine Checkpoints fuer {name}: {pattern}")
        nets = [load_net(f, device) for f in files]
        print(f"[bench] {name}: " + ", ".join(Path(f).name for f in files), flush=True)
        mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
        m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
        s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
        mf_t, sf_t = m_t[:, :, WX], s_t[:, :, WX]
    scales = [float(s) for s in (args.radii if name == "dist" else "1.0" if indep else args.scales).split(",")]
    res, maps = [], {}
    pad = HALF + 1
    for ei, e in enumerate(events):
        L, (r0, r1, c0, c1) = e["L"], e["box"]
        shape = (r1 - r0, c1 - c0)
        slab = load_box(cube, L, r0, r1, c0, c1, pad)
        first = first_fire_box(cube, L, r0, r1, c0, c1)
        win = (slab[:, 0, pad:pad + shape[0], pad:pad + shape[1]] > 0).any(0)
        act = (slab[-ACT_DAYS:, 0, pad:pad + shape[0], pad:pad + shape[1]] > 0).any(0)
        m = mask_all[r0:r1, c0:c1] & ~win
        ys = {t: first <= t for t in HOR}
        ent = {"i": ei, "date": e["date"], "box": e["box"], "n_obs": {t: int((ys[t] & m).sum()) for t in HOR}, "scales": {}}
        if name == "dist":
            dist = ndimage.distance_transform_edt(~act) if act.any() else np.full(shape, 1e3)
            score = 1.0 / (1.0 + dist)                       # nur Rangfolge zaehlt fuer ap / auc
            for s in scales:
                P = np.where(dist <= s, 0.5 + 0.5 * score, 0.5 * score)   # monoton im Abstand, P >= 0.5 genau bis r
                ent["scales"][str(s)] = {str(t): map_metrics(P, ys[t], m, None) for t in HOR}
                maps[f"{ei}_{s}_30"] = maps[f"{ei}_{s}_7"] = (dist <= s).astype(np.float16)
        elif indep:
            doy = dt.date.fromisoformat(e["date"]).timetuple().tm_yday
            cells = np.array([(r, c) for r in range(shape[0]) for c in range(shape[1])], np.int64)
            S = 0.0
            for net in nets:
                ls, _ = predict(net, slab, cells, 1, pad, doy, m_t, s_t, device, list(HOR))
                S = S + np.exp(ls[:, 0, 0]) / len(nets)                                   # (N, len(HOR))
            st = {}
            for j, t in enumerate(HOR):
                P = (1 - S[:, j]).reshape(shape)
                st[str(t)] = map_metrics(P, ys[t], m, None)
                maps[f"{ei}_1.0_{t}"] = P.astype(np.float16)
            ent["scales"]["1.0"] = st
        else:
            fslab = np.ascontiguousarray(load_box(cube, L + T, r0, r1, c0, c1, pad)[:, WX])
            doy = dt.date.fromisoformat(e["date"]).timetuple().tm_yday
            cells = np.array([(r, c) for r in range(shape[0]) for c in range(shape[1])], np.int64)
            par = []
            for net in nets:
                a, k, logD = pixel_params(net, slab, fslab, cells, pad, doy, m_t, s_t, mf_t, sf_t, device)
                obs = torch.from_numpy(slab[-net.obs_days:, 0, pad:pad + shape[0], pad:pad + shape[1]] > 0).float().to(device)
                par.append((net.K, a, k, logD, obs.reshape(net.obs_days, -1)))
            for s in scales:
                gen = torch.Generator(device=device).manual_seed(args.seed)
                S_mf, fired = 0.0, []
                for K, a, k, logD, obs in par:
                    ks = 1 - (1 - k).clamp(min=0).pow(s)
                    ls_mf, f = simulate(K, a, ks, logD, obs, shape, args.paths, gen)
                    S_mf = S_mf + ls_mf.exp().cpu().numpy() / len(par)
                    fired.append(f.cpu().numpy())
                day = np.where(np.concatenate(fired) >= 0, np.concatenate(fired) + 1, 31).reshape(-1, *shape)
                st = {}
                for t in HOR:
                    paths_t = day <= t
                    P_mc = paths_t.mean(0)
                    P_mf = (1 - S_mf[:, t]).reshape(shape)
                    st[str(t)] = map_metrics(P_mc, ys[t], m, paths_t)
                    st[f"{t}_mf"] = map_metrics(P_mf, ys[t], m, None)
                    maps[f"{ei}_{s}_{t}"] = P_mc.astype(np.float16)
                ent["scales"][str(s)] = st
        res.append(ent)
        s1 = ent["scales"].get("1.0") or next(iter(ent["scales"].values()))
        print(f"[bench] {name} {ei + 1}/{len(events)} {e['date']} {shape}: n_obs30 {ent['n_obs'][30]}, "
              f"iou30 {s1['30']['iou']:.3f}, ap30 {s1['30'].get('ap', float('nan')):.3f}, log_area {s1['30']['log_area']:+.2f}",
              flush=True)
    return res, maps


# ----------------------------------------------------------------------------------------------- report
KEYS = ("ap", "auc", "iou", "iou_topk", "iou_path", "prec", "rec", "log_area", "assd", "brier")


def agg(runs: list[dict], s: str, var: str) -> dict:
    out = {}
    for k in KEYS:
        v = [r["scales"][s][var][k] for r in runs if k in r["scales"][s][var]]
        v = np.array(v, float)
        if k == "log_area":
            out["med_log_area"] = float(np.nanmedian(v)) if v.size else float("nan")
            out["mae_log_area"] = float(np.nanmean(np.abs(v))) if v.size else float("nan")
        else:
            out[k] = float(np.nanmean(v)) if v.size else float("nan")
    return out


QS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def obs_masks(exp: Path, cube: CubeReader, events: list[dict]) -> list[tuple]:
    """Je Ereignis (Auswertemaske m, {t: beobachtet bis t}) wie in run."""
    mask_all = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    out = []
    for e in events:
        L, (r0, r1, c0, c1) = e["L"], e["box"]
        year, li = cube._pos[L]
        dyn, _ = cube._arrays(year)
        win = (np.asarray(dyn[li - T_WIN + 1:li + 1, r0:r1, c0:c1, 0]) > 0).any(0)
        first = first_fire_box(cube, L, r0, r1, c0, c1)
        out.append((mask_all[r0:r1, c0:c1] & ~win, {t: first <= t for t in HOR}))
    return out


def iou_q(maps, obs, s: str, t: int, q: float) -> float:
    v = [iou((maps[f"{i}_{s}_{t}"].astype(np.float32) >= q) & m, y[t] & m) for i, (m, y) in enumerate(obs)]
    return float(np.nanmean(v))


def report(out: Path, args) -> None:
    names = sorted({p.stem.split("_", 1)[1] for p in (out / "runs").glob("test_*.json")})
    js, md = {}, ["# Feuerkarten-Benchmark (Regions-Simulation)", ""]
    ev = {sp: json.loads((out / f"events_{sp}.json").read_text()) for sp in ("val", "test")}
    md += [f"Ereignisse: val {len(ev['val'])} ({sorted({e['year'] for e in ev['val']})}), test {len(ev['test'])} "
           f"({sorted({e['year'] for e in ev['test']})}). Region = Feuerkomplex der letzten 3 Tage + Rand; Auswertung auf "
           "Landpixeln ohne Feuer im 30-Tage-Eingabefenster; beobachtet = erster MODIS-Feuertag nach L. "
           "MC = Anteil der Pfade, Karte = P >= 0.5. Skala s: k' = 1 - (1 - k)^s, auf val nach mittlerem IoU 30 d gewaehlt; "
           "dist: Radius r in px auf val gewaehlt.", ""]
    rows, qrows = {}, {}
    exp = out.parent.parent
    cube = CubeReader(exp)
    obs = {sp: obs_masks(exp, cube, ev[sp]) for sp in ("val", "test")}
    for nm in names:
        rt = json.loads((out / "runs" / f"test_{nm}.json").read_text())
        rv_p = out / "runs" / f"val_{nm}.json"
        scales = list(rt[0]["scales"])
        if rv_p.exists():
            rv = json.loads(rv_p.read_text())
            best = {str(t): max(scales, key=lambda s: agg(rv, s, str(t))["iou"]) for t in HOR}
        else:
            best = {str(t): scales[0] for t in HOR}
        if nm != "dist" and (out / "maps" / f"val_{nm}.npz").exists():
            mv, mt = np.load(out / "maps" / f"val_{nm}.npz"), np.load(out / "maps" / f"test_{nm}.npz")
            for t in HOR:
                for s in dict.fromkeys(["1.0", best[str(t)]]):
                    if s not in scales:
                        continue
                    qb = max(QS, key=lambda q: iou_q(mv, obs["val"], s, t, q))
                    qrows.setdefault(t, []).append((nm, s, qb, iou_q(mt, obs["test"], s, t, qb), iou_q(mt, obs["test"], s, t, 0.5)))
        js[nm] = {"best_scale_val": best, "test": {s: {v: agg(rt, s, v) for v in rt[0]["scales"][s]} for s in scales}}
        for t in HOR:
            cand = [("1.0", "s=1")] if "1.0" in scales and nm != "dist" else []
            cand.append((best[str(t)], f"val: {'r' if nm == 'dist' else 's'}={best[str(t)]}"))
            for s, lab in dict(cand).items():
                rows.setdefault(t, []).append((nm, lab, agg(rt, s, str(t)), agg(rt, s, f"{t}_mf") if f"{t}_mf" in rt[0]["scales"][s] else None))
    for t in HOR:
        md += [f"## Horizont {t} Tage (test)", "",
               "| Modell | Wahl | AP | AUC | IoU | IoU top-n | IoU Pfad | Praez. | Recall | Median log Flaeche | MAE log Flaeche | ASSD px | Brier | AP MF | IoU MF |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for nm, lab, a, mf in rows[t]:
            f = lambda x: "–" if x != x else f"{x:.3f}"                                    # noqa: E731
            md.append(f"| {nm} | {lab} | {f(a['ap'])} | {f(a['auc'])} | {f(a['iou'])} | {f(a['iou_topk'])} | {f(a['iou_path'])} | "
                      f"{f(a['prec'])} | {f(a['rec'])} | {a['med_log_area']:+.2f} | {f(a['mae_log_area'])} | {a['assd']:.1f} | "
                      f"{f(a['brier'])} | {f(mf['ap']) if mf else '–'} | {f(mf['iou']) if mf else '–'} |")
        md.append("")
        md += [f"### Karte mit Schwelle aus val, {t} Tage (test)", "", "| Modell | Skala | q (val) | IoU P >= q | IoU P >= 0.5 |",
               "|---|---|---|---|---|"]
        md += [f"| {nm} | {s} | {q} | {a:.3f} | {b:.3f} |" for nm, s, q, a, b in qrows.get(t, [])]
        md.append("")
        js.setdefault("_threshold", {})[str(t)] = qrows.get(t, [])
    (out / f"{args.report_name}.json").write_text(json.dumps(js, indent=1))
    (out / f"{args.report_name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["events", "run", "report"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--model", default="", help="name=glob (run); name dist = Abstands-Baseline")
    ap.add_argument("--scales", default="0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--radii", default="1,2,4,6,8,12,16")
    ap.add_argument("--paths", type=int, default=100, help="MC-Pfade je Seed")
    ap.add_argument("--step", dest="step_days", type=int, default=3)
    ap.add_argument("--min-act", type=int, default=15)
    ap.add_argument("--min-new", type=int, default=50)
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--max-box", type=int, default=112)
    ap.add_argument("--max-events", type=int, default=60)
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-name", default="summary", help="report: Dateiname (Zwischenstand ohne die finale summary)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"firemap_bench{args.tag}"
    cube = CubeReader(exp)
    step = args.stage
    if step == "events":
        yrs = split_years(exp)
        out.mkdir(parents=True, exist_ok=True)
        for sp in ("val", "test"):
            f = out / f"events_{sp}.json"
            if f.exists():
                raise SystemExit(f"ABBRUCH: {f} existiert")
            ev = select_events(exp, cube, yrs[sp], args)
            f.write_text(json.dumps(ev, indent=1))
            print(f"[bench] {sp} {yrs[sp]}: {len(ev)} Ereignisse, neue Pixel 30 d Median "
                  f"{np.median([e['n_new_30'] for e in ev]) if ev else 0:.0f}", flush=True)
            for e in ev:
                print(f"    {e['date']} box {e['box']} aktiv {e['n_act']} neu {e['n_new_30']}", flush=True)
        write_manifest(cfg, f"p64_firemap_bench_events{args.tag}", args.config, {"years": yrs, "args": vars(args),
                                                                                 "outputs": [str(out)]})
    elif step == "run":
        name, pattern = (args.model.split("=", 1) + [""])[:2]
        f = out / "runs" / f"{args.split}_{name}.json"
        if f.exists():
            raise SystemExit(f"ABBRUCH: {f} existiert")
        events = json.loads((out / f"events_{args.split}.json").read_text())
        res, maps = run(exp, cfg, cube, events, name, pattern, args)
        (out / "runs").mkdir(exist_ok=True)
        (out / "maps").mkdir(exist_ok=True)
        f.write_text(json.dumps(res))
        np.savez_compressed(out / "maps" / f"{args.split}_{name}.npz", **maps)
        write_manifest(cfg, f"p64_firemap_bench_{args.split}_{name}{args.tag}", args.config, {
            "model": args.model, "scales": args.scales, "radii": args.radii, "paths": args.paths, "outputs": [str(f)]})
    else:
        if (out / f"{args.report_name}.md").exists():
            raise SystemExit(f"ABBRUCH: {out / args.report_name}.md existiert")
        report(out, args)


if __name__ == "__main__":
    main()

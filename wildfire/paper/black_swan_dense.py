"""Black Swans, Folgeschritt: dichte Stichtagsmenge je Ereignis und Auswertung. Nur CPU.

  --part set   Stichtage aus den Ereignissen von black_swan (Manifest p39_black_swan): die 15 groessten
               Test-Ereignisse 2023-2025 und Kontrollen aus Trainingsjahren (--controls, Ereignis-ids).
                 dicht (kind 0): jedes --k-te Pixel (Zeile und Spalte) in den Bloecken des Ereignisses, jeder
                   Tag von Beginn - 30 d bis Ende
                 Karte (kind 1): jedes --kmap-te Pixel im Rechteck der Bloecke plus ein Block Rand, an Beginn +
                   --map-offsets Tagen (nicht nach dem Ende)
               Nur Land in Kanada, volles 30-Tage-Fenster im Wuerfel, L + 100 d in den Rohdaten. Labels wie
               surv_eval_test (erstes Feuer bis 100 d, Mitte und Ausschnitt) und fuer das Paper-Protokoll Feuer
               genau am Tag L + dt + 1 (Mitte / Ausschnitt), dt in compare.DTS.
               -> derived_data/black_swan<tag>/set.npz (NEU)
  --part eval  Scores aus black_swan_score (score_<modell>.npz: Lambda je Seed an TAUS, paper_<modell>.npz):
               Ensemble (Mittel der Survival-Kurven), Kalibrierung Lambda' = e^a Lambda^b mit a, b aus der Val-
               Anpassung des Ensembles calib_cache_ext/ens_<modell>.npz (wie sts_extreme_table). Je Ereignis und
               Fenster (vorher / waehrend): O/E@30 roh und kal., AUC@30, PR@30, Paper-PR-AUC und F1max je dt;
               Vorwarnung = mittleres F30 je Tag relativ zum Beginn. Paper-Protokoll (cache_G_L61, Test):
               PR-AUC / F1max je dt in den Ereignissen, davor und im Rest, dazu der U-Net-Nachbau aus
               predictions/paper/rec_cache_G_cache_next_paperhp_w0.npz.
               -> metrics/black_swan_dense<tag>/ (NEU), Manifest p40_black_swan_dense<tag>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from wildfire.audit.common import assert_readonly, load_config, read_manifest, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.metrics import operating_points
from wildfire.paper.pixel_landmarks import pixel_events
from wildfire.paper.surv_eval import DAY_BITS, abs_day, calib_apply, calib_fit, raw_fire_keys

TAUS = (2, 7, 8, 16, 30, 31)          # F(dt + 1) fuer dt 1/7/15/30, Karten F7/F30, O/E@30
TAU = 30
PRE = 30
HALF = 6
H = 100


def fire_on_day(keys, Ls, rr, cc, Hg, Wg, chunk=50_000):
    """Feuer genau am Tag Ls (absolut) in der Mitte bzw. irgendwo im 13x13-Ausschnitt."""
    dr, dc = np.meshgrid(np.arange(-HALF, HALF + 1), np.arange(-HALF, HALF + 1), indexing="ij")
    pa, ce = np.zeros(Ls.size, bool), np.zeros(Ls.size, bool)
    for a in range(0, Ls.size, chunk):
        s = slice(a, a + chunk)
        R, C = rr[s, None] + dr.ravel()[None], cc[s, None] + dc.ravel()[None]
        ok = (R >= 0) & (R < Hg) & (C >= 0) & (C < Wg)
        key = np.where(ok, R * Wg + C, 0).astype(np.int64) * DAY_BITS + Ls[s, None]
        pos = np.minimum(np.searchsorted(keys, key), keys.size - 1)
        hit = ok & (keys[pos] == key)
        pa[s], ce[s] = hit.any(1), hit[:, (2 * HALF + 1) ** 2 // 2]
    return pa, ce


def pick_events(cfg, args) -> list[dict]:
    ev = read_manifest(cfg, f"p39_black_swan{args.bs_tag}")["events"]
    test = [e for e in ev if e["split"] == "test"][:args.top_test]
    ctrl = [e for e in ev if e["id"] in set(args.controls)]
    if args.event_ids:
        test = [e for e in ev if e["id"] in set(args.event_ids)]
        ctrl = []
    for e in ctrl:
        e["control"] = True
    return test + ctrl


def build_set(args, cfg, exp, out):
    cube = CubeReader(exp)
    Hg, Wg = cube.H, cube.W
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    keys, last_raw = raw_fire_keys(Path(cfg["dataset_root"]))
    dates = cube.dates
    absd = np.array([abs_day(dt.date.fromisoformat(d[:10])) for d in dates])

    def ok_L(L):
        if L not in cube._pos or (L - 29) not in cube._pos:
            return False
        a, b = cube._pos[L - 29], cube._pos[L]
        return a[0] == b[0] and b[1] - a[1] == 29 and absd[L] + H <= last_raw

    evs = pick_events(cfg, args)
    cols = {k: [] for k in ("event", "kind", "row", "col", "L")}
    info = []
    for e in evs:
        blocks = [tuple(b) for b in e["blocks"]]
        days = [L for L in range(e["start_day"] - PRE, e["end_day"] + 1) if ok_L(L)]
        if args.limit_days:
            days = days[:args.limit_days]
        # dicht: jedes k-te Pixel in den Ereignisbloecken
        pr, pc = [], []
        for br, bc in blocks:
            r = np.arange(br * args.blk, min((br + 1) * args.blk, Hg))
            c = np.arange(bc * args.blk, min((bc + 1) * args.blk, Wg))
            R, C = np.meshgrid(r[r % args.k == args.k // 2], c[c % args.k == args.k // 2], indexing="ij")
            m = land[R, C]
            pr.append(R[m]); pc.append(C[m])
        pr, pc = np.concatenate(pr), np.concatenate(pc)
        for L in days:
            n = pr.size
            cols["event"].append(np.full(n, e["id"])); cols["kind"].append(np.zeros(n, np.int8))
            cols["row"].append(pr); cols["col"].append(pc); cols["L"].append(np.full(n, L))
        # Karte: Rechteck der Bloecke plus ein Block Rand, jedes kmap-te Pixel
        bb = np.array(blocks)
        r0, r1 = max(0, (bb[:, 0].min() - 1) * args.blk), min(Hg, (bb[:, 0].max() + 2) * args.blk)
        c0, c1 = max(0, (bb[:, 1].min() - 1) * args.blk), min(Wg, (bb[:, 1].max() + 2) * args.blk)
        r, c = np.arange(r0, r1), np.arange(c0, c1)
        R, C = np.meshgrid(r[r % args.kmap == 0], c[c % args.kmap == 0], indexing="ij")
        m = land[R, C]
        mr, mc = R[m], C[m]
        mdays = [e["start_day"] + o for o in args.map_offsets
                 if e["start_day"] + o <= e["end_day"] + args.map_after_end and ok_L(e["start_day"] + o)]
        if args.limit_days:
            mdays = mdays[:1]
        for L in mdays:
            cols["event"].append(np.full(mr.size, e["id"])); cols["kind"].append(np.ones(mr.size, np.int8))
            cols["row"].append(mr); cols["col"].append(mc); cols["L"].append(np.full(mr.size, L))
        info.append({"id": e["id"], "year": e["year"], "split": e["split"], "control": bool(e.get("control", False)),
                     "start": e["start"], "end": e["end"], "start_day": e["start_day"], "end_day": e["end_day"],
                     "lat": e["lat"], "lon": e["lon"], "n_days": len(days), "n_px": int(pr.size),
                     "map_box": [int(r0), int(r1), int(c0), int(c1)], "map_days": mdays, "n_map_px": int(mr.size),
                     "blocks": [list(b) for b in blocks]})
        print(f"[bsd] Ereignis {e['id']} {e['start']} - {e['end']}{' (Kontrolle)' if e.get('control') else ''}: "
              f"{pr.size} Pixel x {len(days)} Tage dicht, {mr.size} Pixel x {len(mdays)} Kartentage", flush=True)
    z = {k: np.concatenate(v) for k, v in cols.items()}
    z["row"], z["col"], z["L"] = z["row"].astype(np.int32), z["col"].astype(np.int32), z["L"].astype(np.int32)
    La = absd[z["L"]]
    n = La.size
    evc, tc, evp, tp = (np.zeros(n, bool), np.zeros(n, np.int16), np.zeros(n, bool), np.zeros(n, np.int16))
    for a in range(0, n, 50_000):
        s = slice(a, a + 50_000)
        ev, t = pixel_events(keys, La[s], z["row"][s], z["col"][s], Hg, Wg)
        evc[s], tc[s] = ev[:, HALF, HALF], t[:, HALF, HALF]
        evp[s], tp[s] = ev.reshape(ev.shape[0], -1).any(1), t.reshape(t.shape[0], -1).min(1)
    z.update(L_abs=La, ev_centre=evc, t_centre=tc, ev_patch=evp, t_patch=tp,
             doy=np.array([dt.date.fromisoformat(dates[int(L)][:10]).timetuple().tm_yday for L in z["L"]], np.float32))
    for d in DTS:
        z[f"y_patch_dt{d}"], z[f"y_centre_dt{d}"] = fire_on_day(keys, La + d + 1, z["row"], z["col"], Hg, Wg)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "set.npz", **z)
    print(f"[bsd] {n:,} Stichtage ({int((z['kind'] == 0).sum()):,} dicht, {int((z['kind'] == 1).sum()):,} Karte); "
          f"Ereignis bis 30 d: Mitte {np.mean(evc & (tc <= TAU)):.3f}, Ausschnitt {np.mean(evp & (tp <= TAU)):.3f}",
          flush=True)
    return info


# ------------------------------------------------------------------ Auswertung
def ens_F(Lam: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Lambda (seeds, n, taus) -> F roh und kalibriert (Ensemble = Mittel der Survival-Kurven)."""
    le = -np.log(np.clip(np.mean(np.exp(-Lam.astype(np.float64)), 0), 1e-300, 1))
    return -np.expm1(-le), -np.expm1(-np.exp(a) * np.power(np.maximum(le, 1e-300), b))


def safe(fn, y, p):
    return float(fn(y, p)) if 0 < y.sum() < y.size else float("nan")


def surv_block(Fr, Fc, e30, m):
    n = int(m.sum())
    o = int(e30[m].sum())
    r = {"n": n, "obs30": o}
    for v, F in (("raw", Fr), ("cal", Fc)):
        E = float(F[m].sum()) if n else float("nan")
        r[f"oe30_{v}"] = o / E if n and E > 0 else float("nan")
    r["auc30"], r["pr30"] = safe(roc_auc_score, e30[m], Fc[m]), safe(average_precision_score, e30[m], Fc[m])
    r["meanF30_cal"] = float(Fc[m].mean()) if n else float("nan")
    r["rate30"] = o / n if n else float("nan")
    return r


def paper_block(p, y):
    if not (0 < y.sum() < y.size):
        return {"pr_auc": float("nan"), "f1_max": float("nan"), "n": int(y.size), "n_pos": int(y.sum())}
    return {"pr_auc": float(average_precision_score(y, p)), "f1_max": operating_points(p, y)["f1_max"],
            "n": int(y.size), "n_pos": int(y.sum())}


def evaluate_all(args, cfg, exp, out_d, mout):
    S = dict(np.load(out_d / "set.npz"))
    man = read_manifest(cfg, f"p40_black_swan_set{args.tag}")
    info = {e["id"]: e for e in man["events"]}
    sd = exp / "derived_data" / "surv_eval_test"
    vm = dict(np.load(sd / "val_meta.npz"))
    dense = S["kind"] == 0
    rel = S["L"] - np.array([info[int(i)]["start_day"] for i in S["event"]])
    rows, curves, prow = [], [], []
    test_ids = [i for i, e in info.items() if not e["control"]]
    ctrl_ids = [i for i, e in info.items() if e["control"]]
    groups = [("alle_test", "pre", np.isin(S["event"], test_ids) & (rel < 0)),
              ("alle_test", "during", np.isin(S["event"], test_ids) & (rel >= 0)),
              ("kontrollen", "pre", np.isin(S["event"], ctrl_ids) & (rel < 0)),
              ("kontrollen", "during", np.isin(S["event"], ctrl_ids) & (rel >= 0))]
    for i in info:
        groups += [(str(i), "pre", (S["event"] == i) & (rel < 0)), (str(i), "during", (S["event"] == i) & (rel >= 0))]
    calib = {}
    for name in args.models:
        f = out_d / f"score_{name}.npz"
        if not f.exists():
            print(f"[bsd] fehlt: {f.name}", flush=True)
            continue
        Z = np.load(f)
        ens = np.load(exp / "derived_data" / "calib_cache_ext" / f"ens_{name}.npz")
        for tg in ("patch", "centre"):
            a, b = calib_fit(ens[f"val_{tg}"].astype(np.float64), vm[f"ev_{tg}"].astype(bool), vm[f"t_{tg}"].astype(int))
            calib[f"{name}_{tg}"] = [a, b]
            Fr, Fc = ens_F(Z[f"lam_{tg}"], a, b)                       # (n, len(TAUS))
            np.save(mout / f"F_cal_{name}_{tg}.npy", Fc.astype(np.float32))
            j30 = TAUS.index(TAU)
            e30 = S[f"ev_{tg}"] & (S[f"t_{tg}"] <= TAU)
            for gname, w, m in groups:
                m = m & dense
                r = {"model": name, "target": tg, "event": gname, "window": w, **surv_block(Fr[:, j30], Fc[:, j30], e30, m)}
                for d in DTS:
                    pb = paper_block(Fc[m, TAUS.index(d + 1)], S[f"y_{tg}_dt{d}"][m])
                    r[f"paper_prauc_dt{d}"], r[f"paper_f1max_dt{d}"] = pb["pr_auc"], pb["f1_max"]
                    r[f"paper_pos_dt{d}"] = pb["n_pos"]
                rows.append(r)
            for i in info:
                for k in range(-PRE, 1):
                    m = dense & (S["event"] == i) & (rel == k)
                    if m.any():
                        curves.append({"model": name, "target": tg, "event": i, "rel_day": k, "n": int(m.sum()),
                                       "meanF30_cal": float(Fc[m, j30].mean()), "rate30": float(e30[m].mean())})
        print(f"[bsd] {name}: Kalibrierung {calib[f'{name}_patch']} / {calib[f'{name}_centre']}", flush=True)

    # ---- Paper-Protokoll auf cache_G_L61 (Test)
    cd = exp / "derived_data" / "paper" / "cache_G_L61"
    split = np.load(cd / "split.npy")
    te = np.where(split == "test")[0]
    meta = np.load(cd / "meta.npy")[te]
    y = {"patch": targets.labels(cd, "next_patch")[te],
         "centre": targets.fire_map(cd, None, "next_patch")[te][:, HALF, HALF].astype(int)}
    dates = CubeReader(exp).dates
    didx = {d[:10]: i for i, d in enumerate(dates)}
    Dg = np.array([didx.get(f"{str(v)[:4]}-{str(v)[4:6]}-{str(v)[6:8]}", -10 ** 6) for v in meta[:, 0]])
    blk = list(zip((meta[:, 1] // args.blk).tolist(), (meta[:, 2] // args.blk).tolist()))
    pin, ppre = np.zeros(te.size, bool), np.zeros(te.size, bool)
    for e in info.values():
        if e["control"]:
            continue
        bs = set(map(tuple, e["blocks"]))
        ib = np.array([b in bs for b in blk])
        pin |= ib & (Dg >= e["start_day"]) & (Dg <= e["end_day"])
        ppre |= ib & (Dg >= e["start_day"] - PRE) & (Dg < e["start_day"])
    yrs = np.array([int(str(v)[:4]) for v in meta[:, 0]])
    print(f"[bsd] Paper-Test {te.size:,} (Jahre {sorted(set(yrs.tolist()))}): in Ereignissen {int(pin.sum())}, "
          f"davor {int(ppre.sum())}", flush=True)
    psub = {"gesamt": np.ones(te.size, bool), "in_ereignis": pin, "vor_ereignis": ppre, "rest": ~(pin | ppre)}

    def paper_rows(model, tg, scores_by_dt, only=None):
        for sn, m in psub.items():
            if only is not None:
                m = m & np.isin(np.arange(te.size), only)
            r = {"model": model, "target": tg, "subset": sn, "n": int(m.sum())}
            for d in DTS:
                per = [paper_block(s[m], y[tg][m]) for s in scores_by_dt[d]]
                r[f"prauc_dt{d}"] = float(np.nanmean([p["pr_auc"] for p in per]))
                r[f"f1max_dt{d}"] = float(np.nanmean([p["f1_max"] for p in per]))
                r[f"pos_dt{d}"] = per[0]["n_pos"]
            prow.append(r)

    for name in args.models:
        f = out_d / f"paper_{name}.npz"
        if not f.exists():
            continue
        Z = np.load(f)
        pos_te = {int(r): i for i, r in enumerate(te)}
        if not all(int(r) in pos_te for r in Z["rows"]):
            raise SystemExit(f"ABBRUCH: {f.name} passt nicht zu den Test-Zeilen")
        sub = np.array([pos_te[int(r)] for r in Z["rows"]])
        full = sub.size == te.size
        for tg in ("patch", "centre"):
            F = Z[f"F_{tg}"]                                           # (seeds, len(DTS), n)
            if not full:                                              # Rauchtest: nur diese Zeilen
                G = np.full((F.shape[0], F.shape[1], te.size), np.nan, np.float32)
                G[:, :, sub] = F
                F = G
            paper_rows(name, tg, {d: list(F[:, k]) for k, d in enumerate(DTS)}, None if full else sub)
            paper_rows(name + "_ens", tg, {d: [F[:, k].mean(0)] for k, d in enumerate(DTS)}, None if full else sub)
    rp = exp / "predictions" / "paper" / "rec_cache_G_cache_next_paperhp_w0.npz"
    if rp.exists():
        z = np.load(rp)
        pos = {int(i): n for n, i in enumerate(z["idx_test"])}
        miss = int(sum(int(i) not in pos for i in te))
        if miss:
            print(f"[bsd] U-Net: {miss} Test-Zeilen fehlen in {rp.name} - uebersprungen", flush=True)
            return rows, curves, prow, calib, {k: int(v.sum()) for k, v in psub.items()}
        sel = np.array([pos[int(i)] for i in te])
        sc = {d: [z[k][sel] for k in sorted(z.files) if k.startswith(f"paper55_T30_end{d}_s") and k.endswith("_test")]
              for d in DTS}
        if all(sc.values()):
            paper_rows("unet_rec", "patch", sc)
            paper_rows("unet_rec_ens", "patch", {d: [np.mean(v, 0)] for d, v in sc.items()})
    return rows, curves, prow, calib, {k: int(v.sum()) for k, v in psub.items()}


def write_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=("set", "eval"), required=True)
    ap.add_argument("--bs-tag", default="", help="Tag des black_swan-Laufs (p39_black_swan<tag>)")
    ap.add_argument("--top-test", type=int, default=15)
    ap.add_argument("--controls", type=int, nargs="*", default=[30, 3], help="Ereignis-ids aus Trainingsjahren")
    ap.add_argument("--event-ids", type=int, nargs="*", default=None, help="nur diese Ereignisse (Rauchtest)")
    ap.add_argument("--limit-days", type=int, default=0, help="Rauchtest: so viele Tage je Ereignis")
    ap.add_argument("--blk", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--kmap", type=int, default=2)
    ap.add_argument("--map-offsets", type=int, nargs="+", default=[-14, 0, 14, 28])
    ap.add_argument("--map-after-end", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=["lh_ref", "lh_mono", "cg_ref", "cg_mono"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out_d = exp / "derived_data" / f"black_swan{args.tag}"
    if args.part == "set":
        if (out_d / "set.npz").exists():
            raise SystemExit(f"ABBRUCH: {out_d / 'set.npz'} existiert schon - anderen --tag waehlen")
        info = build_set(args, cfg, exp, out_d)
        write_manifest(cfg, f"p40_black_swan_set{args.tag}", args.config, {
            "set": str(out_d / "set.npz"), "k": args.k, "kmap": args.kmap, "blk": args.blk,
            "map_offsets": args.map_offsets, "pre_days": PRE, "events": info, "taus": TAUS, "dts": list(DTS)})
        return
    mout = exp / "metrics" / f"black_swan_dense{args.tag}"
    if mout.exists():
        raise SystemExit(f"ABBRUCH: {mout} existiert schon - anderen --tag waehlen")
    mout.mkdir(parents=True)
    rows, curves, prow, calib, psizes = evaluate_all(args, cfg, exp, out_d, mout)
    write_csv(mout / "by_event.csv", rows)
    write_csv(mout / "warning_curve.csv", curves)
    write_csv(mout / "paper_protocol.csv", prow)
    f = lambda x, p=3: "–" if x is None or x != x else f"{x:.{p}f}"  # noqa: E731
    md = ["# Black Swans, dichte Stichtage und Paper-Protokoll", "",
          "Kalibriert = Lambda' = e^a Lambda^b aus der Val-Anpassung des Ensembles (calib_cache_ext). O/E > 1 = "
          "Unterschaetzung. Paper-Metriken auf der dichten Menge: Score F(dt + 1), Ziel Feuer genau am Tag L + dt + 1.", ""]
    for tg in ("patch", "centre"):
        md += [f"## Dichte Stichtage, {'Ausschnitt' if tg == 'patch' else 'Mittelpixel'}", "",
               "| Ereignis | Fenster | Modell | n | Ereignisse@30 | O/E roh | O/E kal. | AUC@30 | PR@30 | "
               + " | ".join(f"PR-AUC dt{d}" for d in DTS) + " |", "|---" * (9 + len(DTS)) + "|"]
        for r in rows:
            if r["target"] == tg:
                md.append(f"| {r['event']} | {r['window']} | {r['model']} | {r['n']} | {r['obs30']} | "
                          f"{f(r['oe30_raw'], 2)} | {f(r['oe30_cal'], 2)} | {f(r['auc30'])} | {f(r['pr30'])} | "
                          + " | ".join(f(r[f'paper_prauc_dt{d}']) for d in DTS) + " |")
        md.append("")
    md += ["## Paper-Protokoll (cache_G_L61, Test), Seed-Mittel bzw. Ensemble", "",
           f"Teilmengen: {psizes}. U-Net-Nachbau gesamt laut Referenz .691/.607/.558/.515.", "",
           "| Modell | Ziel | Teilmenge | n | " + " | ".join(f"PR-AUC dt{d}" for d in DTS) + " | "
           + " | ".join(f"F1max dt{d}" for d in DTS) + " | " + " | ".join(f"pos dt{d}" for d in DTS) + " |",
           "|---" * (4 + 3 * len(DTS)) + "|"]
    for r in prow:
        md.append(f"| {r['model']} | {r['target']} | {r['subset']} | {r['n']} | "
                  + " | ".join(f(r[f'prauc_dt{d}']) for d in DTS) + " | "
                  + " | ".join(f(r[f'f1max_dt{d}']) for d in DTS) + " | "
                  + " | ".join(str(r[f'pos_dt{d}']) for d in DTS) + " |")
    (mout / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p40_black_swan_dense{args.tag}", args.config, {
        "models": args.models, "calib_ab": calib, "paper_subsets": psizes, "taus": TAUS, "out": str(mout),
        "unet_rec": "predictions/paper/rec_cache_G_cache_next_paperhp_w0.npz (paper55_T30_end<dt>)"})


if __name__ == "__main__":
    main()

"""Black Swans: die extremsten Feuerereignisse 2000-2025 und wie gut die Modelle dort sind. Nur CPU.

A  Ereignisse. Alle Feuer-Pixeltage aus derived_data/events/fire_index.npz (MCD14A1 in der Gueltigkeits-
   maske, duenn) -> Zaehlung je Block (--blk x --blk Pixel, Standard 16 x 16 = ca. 80 km N-S) und Fenster
   (--win Tage ab 1. Januar, Standard 14). Lokaler Rekord = groesstes Fenster desselben Blocks in den
   Trainingsjahren 2000-2019; fuer ein Trainingsjahr ohne dieses Jahr (leave-one-year-out), sonst waere es
   nie ueber dem Rekord. Extremzelle = Zaehlung >= --nmin und >= --ratio x Rekord. Benachbarte Extremzellen
   (Raum und Zeit, 26er-Nachbarschaft, je Jahr) bilden ein Ereignis. Je Ereignis: Zeitraum, Schwerpunkt
   (lat/lon aus d0_masks), Bloecke, Feuer-Pixeltage, Summe/Rekord und max. Verhaeltnis.
B  Modelle. Zufaellige Stichtage aus surv_eval_test (Test 2023-2025), deren Mittelpixel in einem Block
   eines Test-Ereignisses liegt: "vorher" = Stichtag in [Beginn - 30 d, Beginn), "waehrend" = [Beginn,
   Ende]. Hazards: Ensembles derived_data/calib_cache_ext/ens_<name>.npz, kalibriert auf Val 2020-2022
   (Lambda' = e^a Lambda^b, wie sts_extreme_table). Je Ereignis und Modell: n, Ereignisse bis 30 d,
   O/E@30 roh und kalibriert, AUC@30, PR@30 (nur bei beiden Klassen), dazu dieselben Kennzahlen fuer alle
   Test-Ereignisse zusammen und fuer den Rest des Tests. "Vorwarnung": mittleres kalibriertes F30 und
   beobachtete Rate der vorher-Stichtage relativ zu Stichtagen derselben Bloecke im selben Saisonfenster
   anderer Jahre (Val+Test), also vorhergesagter gegen beobachteten Anstieg.
Ausgabe NEU: metrics/black_swan<tag>/ events.csv, model_by_event.csv, summary.md; Manifest p39_black_swan<tag>.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.surv_eval import _cum_hazard, calib_apply, calib_fit

TRAIN = (2000, 2019)
SPLIT = {"train": (2000, 2019), "val": (2020, 2022), "test": (2023, 2025)}
TAU = 30
PRE = 30


def split_of(y: int) -> str:
    return next(s for s, (a, b) in SPLIT.items() if a <= y <= b)


def load_fire(exp: Path):
    z = np.load(exp / "derived_data" / "events" / "fire_index.npz", allow_pickle=False)
    H, W = (int(v) for v in z["grid"])
    dates = [str(d) for d in z["dates"]]
    px, off, days = z["fire_pixels"].astype(np.int64), z["fire_offsets"], z["fire_days"].astype(np.int64)
    flat = np.repeat(px, np.diff(off))
    return H, W, dates, flat // W, flat % W, days


def geo(exp: Path):
    tr = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())["grid"]["transform"]
    lon = lambda c: tr[2] + tr[0] * (np.asarray(c, float) + 0.5)  # noqa: E731
    lat = lambda r: tr[5] + tr[4] * (np.asarray(r, float) + 0.5)  # noqa: E731
    return lat, lon


def find_events(args, exp):
    H, W, dates, r, c, d = load_fire(exp)
    D = np.array([dt.date.fromisoformat(x[:10]) for x in dates])
    yr = np.array([x.year for x in D])
    doy = np.array([x.timetuple().tm_yday - 1 for x in D])
    years = sorted(set(yr[d].tolist()))
    nb = 366 // args.win + 1
    nbr, nbc = -(-H // args.blk), -(-W // args.blk)
    Y = {y: i for i, y in enumerate(years)}
    C = np.zeros((len(years), nb, nbr, nbc), np.int32)
    np.add.at(C, (np.array([Y[v] for v in yr[d]]), doy[d] // args.win, r // args.blk, c // args.blk), 1)
    tr_i = [Y[y] for y in years if TRAIN[0] <= y <= TRAIN[1]]
    M = C[tr_i].max(1)                                              # (n_train, nbr, nbc) Jahresmax je Block
    rec_all = M.max(0)
    srt = np.sort(M, 0)
    rec_2nd = srt[-2] if len(tr_i) > 1 else np.zeros_like(rec_all)
    lat, lon = geo(exp)
    annual = {int(y): int(C[Y[y]].sum()) for y in years}
    print(f"[bs] {r.size:,} Feuer-Pixeltage, {len(years)} Jahre, Gitter {nbr} x {nbc} Bloecke, {nb} Fenster je Jahr",
          flush=True)

    events = []
    for y in years:
        i = Y[y]
        if TRAIN[0] <= y <= TRAIN[1]:
            j = tr_i.index(i)
            rec = np.where(M[j] == rec_all, rec_2nd, rec_all)       # Rekord ohne dieses Jahr
        else:
            rec = rec_all
        ratio = C[i] / np.maximum(rec, 1)[None]
        ext = (C[i] >= args.nmin) & (ratio >= args.ratio)
        lab, n = ndimage.label(ext, structure=np.ones((3, 3, 3), bool))
        for k in range(1, n + 1):
            bi, rr, cc = np.nonzero(lab == k)
            cnt = C[i][bi, rr, cc].astype(float)
            # Pixel-genau: Feuer-Pixeltage dieses Jahres in den Bloecken und Fenstern des Ereignisses
            sel = (yr[d] == y)
            cell = set(zip(bi.tolist(), rr.tolist(), cc.tolist()))
            dd, r_, c_ = d[sel], r[sel], c[sel]
            keep = np.array([(int(doy[t]) // args.win, int(a) // args.blk, int(b) // args.blk) in cell
                             for t, a, b in zip(dd, r_, c_)], bool)
            dd, r_, c_ = dd[keep], r_[keep], c_[keep]
            blocks = sorted(set(zip(rr.tolist(), cc.tolist())))
            events.append({
                "year": int(y), "split": split_of(int(y)),
                "start": str(D[dd].min()), "end": str(D[dd].max()),
                "start_day": int(dd.min()), "end_day": int(dd.max()),
                "lat": float(lat(r_).mean()), "lon": float(lon(c_).mean()),
                "lat_min": float(lat(r_).max()), "lat_max": float(lat(r_).min()),
                "lon_min": float(lon(c_).min()), "lon_max": float(lon(c_).max()),
                "n_blocks": len(blocks), "n_cells": int(bi.size),
                "fire_pixel_days": int(cnt.sum()),
                "ratio_sum": float(cnt.sum() / max(float(rec[rr, cc].sum()), 1.0)),
                "ratio_max": float(ratio[bi, rr, cc].max()),
                "blocks": blocks,
                "share_of_year": float(cnt.sum() / max(annual[int(y)], 1)),
            })
    events.sort(key=lambda e: -e["fire_pixel_days"])
    for n, e in enumerate(events):
        e["id"] = n
    return events, annual, dates


# ------------------------------------------------------------------ Teil B
def metrics(F30, e30, mask) -> dict:
    n = int(mask.sum())
    o = int(e30[mask].sum())
    out = {"n": n, "obs30": o, "exp30": float(F30[mask].sum()) if n else float("nan")}
    out["oe30"] = o / out["exp30"] if n and out["exp30"] > 0 else float("nan")
    both = n > 0 and 0 < o < n
    out["auc30"] = float(roc_auc_score(e30[mask], F30[mask])) if both else float("nan")
    out["pr30"] = float(average_precision_score(e30[mask], F30[mask])) if both else float("nan")
    out["meanF30"] = float(F30[mask].mean()) if n else float("nan")
    out["rate30"] = o / n if n else float("nan")
    return out


def model_eval(args, exp, events, dates):
    sd = exp / "derived_data" / args.surv_dir
    meta = {s: dict(np.load(sd / f"{f}_meta.npz")) for s, f in (("val", "val"), ("test", "eval"))}
    D = [dt.date.fromisoformat(x[:10]) for x in dates]
    info = {}
    for s, m in meta.items():
        L = m["L_cube"].astype(int)
        info[s] = {"L": L, "yr": np.array([D[x].year for x in L]), "doy": np.array([D[x].timetuple().tm_yday for x in L]),
                   "blk": list(zip((m["row"] // args.blk).tolist(), (m["col"] // args.blk).tolist()))}
    test_ev = [e for e in events if e["split"] == "test"][:args.top_test]
    te = info["test"]
    groups = {}
    for e in test_ev:
        bs = set(map(tuple, e["blocks"]))
        inb = np.array([b in bs for b in te["blk"]])
        yr_ok = te["yr"] == e["year"]
        groups[e["id"]] = {"pre": inb & yr_ok & (te["L"] >= e["start_day"] - PRE) & (te["L"] < e["start_day"]),
                           "during": inb & yr_ok & (te["L"] >= e["start_day"]) & (te["L"] <= e["end_day"])}
        # Bezug fuer die Vorwarnung: dieselben Bloecke, selbes Saisonfenster (Kalendertag), andere Jahre
        d0 = D[e["start_day"]].timetuple().tm_yday
        ref = {}
        for s in ("val", "test"):
            I = info[s]
            ib = np.array([b in bs for b in I["blk"]])
            ref[s] = ib & (I["yr"] != e["year"]) & (I["doy"] >= d0 - PRE) & (I["doy"] < d0)
        groups[e["id"]]["ref"] = ref
    any_ev = np.zeros(te["L"].size, bool)
    for g in groups.values():
        any_ev |= g["pre"] | g["during"]

    rows, pooled = [], {}
    for name in args.models:
        z = np.load(exp / "derived_data" / "calib_cache_ext" / f"ens_{name}.npz")
        for tg in ("patch", "centre"):
            ev = {s: meta[s][f"ev_{tg}"].astype(bool) for s in meta}
            tt = {s: meta[s][f"t_{tg}"].astype(int) for s in meta}
            e30 = {s: ev[s] & (tt[s] <= TAU) for s in meta}
            hv, ht = z[f"val_{tg}"].astype(np.float64), z[f"test_{tg}"].astype(np.float64)
            a, b = calib_fit(hv, ev["val"], tt["val"])
            F = {"raw": {"test": -np.expm1(-_cum_hazard(ht)[:, TAU - 1]), "val": -np.expm1(-_cum_hazard(hv)[:, TAU - 1])},
                 "cal": {"test": -np.expm1(-_cum_hazard(calib_apply(ht, a, b))[:, TAU - 1]),
                         "val": -np.expm1(-_cum_hazard(calib_apply(hv, a, b))[:, TAU - 1])}}
            for gname, mask in (("alle_test", np.ones(te["L"].size, bool)), ("alle_ereignisse", any_ev),
                                ("rest_test", ~any_ev)):
                r = {"model": name, "target": tg, "event": gname, "window": "-"}
                for v in ("raw", "cal"):
                    m = metrics(F[v]["test"], e30["test"], mask)
                    r.update({f"{k}_{v}": m[k] for k in m})
                rows.append(r)
                pooled[(name, tg, gname)] = r
            for e in test_ev:
                g = groups[e["id"]]
                for w in ("pre", "during"):
                    r = {"model": name, "target": tg, "event": e["id"], "window": w}
                    for v in ("raw", "cal"):
                        m = metrics(F[v]["test"], e30["test"], g[w])
                        r.update({f"{k}_{v}": m[k] for k in m})
                    if w == "pre":
                        fr = np.concatenate([F["cal"][s][g["ref"][s]] for s in ("val", "test")])
                        orf = np.concatenate([e30[s][g["ref"][s]] for s in ("val", "test")])
                        r["ref_n"] = int(fr.size)
                        r["lift_pred"] = (r["meanF30_cal"] / fr.mean()) if fr.size and fr.mean() > 0 else float("nan")
                        r["lift_obs"] = (r["rate30_cal"] / orf.mean()) if fr.size and orf.mean() > 0 else float("nan")
                    rows.append(r)
            print(f"[bs] {name} {tg}: Kalibrierung a {a:.3f} b {b:.3f}; alle Ereignisse O/E@30 kal. "
                  f"{pooled[(name, tg, 'alle_ereignisse')]['oe30_cal']:.2f}, Rest "
                  f"{pooled[(name, tg, 'rest_test')]['oe30_cal']:.2f}", flush=True)
    return rows, test_ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--blk", type=int, default=16)
    ap.add_argument("--win", type=int, default=14)
    ap.add_argument("--nmin", type=int, default=30, help="min. Feuer-Pixeltage je Block und Fenster")
    ap.add_argument("--ratio", type=float, default=1.0, help="min. Vielfaches des lokalen Rekords")
    ap.add_argument("--top", type=int, default=15, help="Ereignisse je Zeitraum in summary.md")
    ap.add_argument("--top-test", type=int, default=15, help="so viele groesste Test-Ereignisse fuer Teil B")
    ap.add_argument("--models", nargs="*", default=["lh_ref", "lh_mono", "lh_reg", "cg_ref", "cg_mono"])
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"black_swan{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --tag waehlen")

    events, annual, dates = find_events(args, exp)
    rows, test_ev = model_eval(args, exp, events, dates) if args.models else ([], [])

    out.mkdir(parents=True)
    ecols = [k for k in events[0] if k != "blocks"] if events else []
    with open(out / "events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ecols, extrasaction="ignore")
        w.writeheader(); w.writerows(events)
    if rows:
        cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "target", "event", "window"), k))
        with open(out / "model_by_event.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(rows)

    f = lambda x, p=2: "–" if x is None or x != x else f"{x:.{p}f}"  # noqa: E731
    md = ["# Black Swans: extremste Feuerereignisse und Modellguete", "",
          f"Block {args.blk} x {args.blk} px, Fenster {args.win} d, Extremzelle >= {args.nmin} Feuer-Pixeltage und "
          f">= {args.ratio} x lokaler Rekord 2000-2019 (Trainingsjahre leave-one-year-out). "
          f"{len(events)} Ereignisse.", "",
          "## Feuer-Pixeltage je Jahr (Gueltigkeitsmaske)", "",
          "| " + " | ".join(str(y) for y in annual) + " |", "|---" * len(annual) + "|",
          "| " + " | ".join(f"{v:,}" for v in annual.values()) + " |", ""]
    for s in SPLIT:
        ev = [e for e in events if e["split"] == s][:args.top]
        md += [f"## Groesste Ereignisse {s} {SPLIT[s][0]}-{SPLIT[s][1]}", "",
               "| id | Zeitraum | lat | lon | Bloecke | Feuer-Pixeltage | Summe/Rekord | max. x Rekord | Anteil Jahr |",
               "|---|---|---|---|---|---|---|---|---|"]
        for e in ev:
            md.append(f"| {e['id']} | {e['start']} – {e['end']} | {e['lat']:.2f} | {e['lon']:.2f} | {e['n_blocks']} | "
                      f"{e['fire_pixel_days']:,} | {e['ratio_sum']:.2f} | {e['ratio_max']:.2f} | {100 * e['share_of_year']:.0f} % |")
        md.append("")
    if rows:
        for tg in ("patch", "centre"):
            md += [f"## Modelle in den Test-Ereignissen ({'Ausschnitt' if tg == 'patch' else 'Mittelpixel'}), kalibriert", "",
                   "| Ereignis | Fenster | Modell | n | Ereignisse@30 | O/E@30 roh | O/E@30 kal. | AUC@30 | PR@30 | Anstieg vorh. | Anstieg beob. |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                if r["target"] != tg:
                    continue
                md.append(f"| {r['event']} | {r['window']} | {r['model']} | {r['n_cal']} | {r['obs30_cal']} | "
                          f"{f(r['oe30_raw'])} | {f(r['oe30_cal'])} | {f(r['auc30_cal'], 3)} | {f(r['pr30_cal'], 3)} | "
                          f"{f(r.get('lift_pred'))} | {f(r.get('lift_obs'))} |")
            md.append("")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p39_black_swan{args.tag}", args.config, {
        "blk": args.blk, "win": args.win, "nmin": args.nmin, "ratio": args.ratio, "models": args.models,
        "surv_dir": args.surv_dir, "n_events": len(events), "annual_fire_pixel_days": annual,
        "events": [{k: v for k, v in e.items() if k != "blocks"} | {"blocks": [list(b) for b in e["blocks"]]}
                   for e in events[:200]],
        "test_events_in_B": [e["id"] for e in test_ev], "out": str(out)})


if __name__ == "__main__":
    main()

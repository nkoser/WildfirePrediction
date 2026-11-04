"""Ursachen-Labels fuer die Ereignisse der Survival-Ziele: Zuendung, Ausbreitung oder unbekannt.

Ereignis = erstes MODIS-Feuer eines Pixels nach dem Stichtag (Tag k). Regeln (Parameter ablatierbar):

  Zuendung     NFDB-Zuendung (>= 1 ha, derived_data/events/ignitions_nfdb_min1ha.csv) mit Datum in
               [k - dd, k + dd] und hoechstens dp Pixel Abstand (Registrierungsversatz); die NFDB-Ursache
               (Blitz / Mensch / unbekannt) wird mitgefuehrt
  Ausbreitung  keine Zuendung; ein ANDERES Pixel im (2 r + 1)^2-Kern brannte (MODIS) an einem Tag in
               [k - w, k - 1]; das Pixel liegt in der NFDB-beobachteten Maske (observed_nfdb.npy)
  unbekannt    alles andere, insbesondere ausserhalb der beobachteten Maske

Codes je Pixel (int8): 0 kein Ereignis, 1 Zuendung Blitz, 2 Zuendung Mensch, 3 Zuendung Ursache
unbekannt, 4 Ausbreitung, 5 Ereignis ohne Label.

Teile:
  train  Wuerfel-Stichtage (derived_data/paper_pixel: train_meta.npz, train_event.npy, train_t.npy)
  eval   Test-Stichtage (surv_eval_test/eval_meta.npz + region_eval/pixel_events.npz, Job 85)
Ausgabe NEU: derived_data/cause_labels/<teil>_cause.npz; bricht ab, wenn die Datei existiert.
Nur gelesen: alle Eingaben.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config
from wildfire.paper.surv_prepare import DAY_BITS

P = 13
HALF = P // 2
DAY0 = dt.date(2000, 1, 1)
CODE = {"N": 1, "H": 2, "U": 3}
SPREAD, UNLABELLED = 4, 5


def abs_day(s: str) -> int:
    return (dt.date.fromisoformat(s) - DAY0).days


def nfdb_keys(path: Path, W: int):
    """NFDB-Zuendungen als sortierte Schluessel Pixel * DAY_BITS + Tag, dazu die Ursache je Schluessel."""
    flat, day, cause = [], [], []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            flat.append(int(r["row"]) * W + int(r["col"]))
            day.append(abs_day(r["date"]))
            cause.append(CODE.get((r["cause"] or "U")[:1].upper(), 3))
    k = np.array(flat, np.int64) * DAY_BITS + np.array(day, np.int64)
    o = np.argsort(k, kind="stable")
    return k[o], np.array(cause, np.int8)[o]


def lookup(keys: np.ndarray, q: np.ndarray):
    """Index des Treffers je Anfrage oder -1."""
    pos = np.searchsorted(keys, q)
    pos_c = np.minimum(pos, keys.size - 1)
    return np.where(keys[pos_c] == q, pos_c, -1)


def first_fire_in(keys: np.ndarray, flat: np.ndarray, d_lo: np.ndarray, d_hi: np.ndarray) -> np.ndarray:
    """MODIS-Feuer an Pixel flat an irgendeinem Tag in [d_lo, d_hi]?"""
    pos = np.searchsorted(keys, flat * DAY_BITS + d_lo)
    nxt = keys[np.minimum(pos, keys.size - 1)]
    return (pos < keys.size) & (nxt // DAY_BITS == flat) & (nxt % DAY_BITS <= d_hi)


def label(ev, t, L_abs, row, col, Hg, Wg, nk, nc, fk, observed, args, chunk=2_000_000):
    """-> Codes (n, 13, 13) int8."""
    out = np.zeros(ev.shape, np.int8)
    n_i, i_i, j_i = np.nonzero(ev)
    for a in range(0, n_i.size, chunk):
        n, i, j = n_i[a:a + chunk], i_i[a:a + chunk], j_i[a:a + chunk]
        k = L_abs[n].astype(np.int64) + t[n, i, j].astype(np.int64)
        R, C = row[n] + i - HALF, col[n] + j - HALF
        code = np.full(n.size, UNLABELLED, np.int8)
        # Zuendung: naechster Treffer in Raum und Zeit, Ursache aus der NFDB
        best = np.full(n.size, 99, np.int64)
        for dr in range(-args.dp, args.dp + 1):
            for dc in range(-args.dp, args.dp + 1):
                rr, cc = R + dr, C + dc
                ok = (rr >= 0) & (rr < Hg) & (cc >= 0) & (cc < Wg)
                for dd in range(-args.dd, args.dd + 1):
                    hit = lookup(nk, np.where(ok, rr * Wg + cc, 0).astype(np.int64) * DAY_BITS + k + dd)
                    dist = abs(dr) + abs(dc) + abs(dd)
                    take = ok & (hit >= 0) & (dist < best)
                    code[take] = nc[hit[take]]
                    best[take] = dist
        ign = best < 99
        # Ausbreitung: ein anderes Pixel im Kern brannte in [k - w, k - 1]
        nb = np.zeros(n.size, bool)
        for dr in range(-args.r, args.r + 1):
            for dc in range(-args.r, args.r + 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = R + dr, C + dc
                ok = (rr >= 0) & (rr < Hg) & (cc >= 0) & (cc < Wg)
                fl = np.where(ok, rr * Wg + cc, 0).astype(np.int64)
                nb |= ok & first_fire_in(fk, fl, k - args.w, k - 1)
        obs = observed[np.clip(R, 0, Hg - 1), np.clip(C, 0, Wg - 1)]
        code[~ign & nb & obs] = SPREAD
        out[n, i, j] = code
    return out


def summary(codes, ev, extra: dict[str, np.ndarray]) -> dict:
    e = codes[ev]
    tot = max(int(ev.sum()), 1)
    s = {"events": int(ev.sum()),
         "share": {"ign_N": float((e == 1).sum() / tot), "ign_H": float((e == 2).sum() / tot),
                   "ign_U": float((e == 3).sum() / tot), "spread": float((e == SPREAD).sum() / tot),
                   "unlabelled": float((e == UNLABELLED).sum() / tot)}}
    for name, grp in extra.items():
        for g in np.unique(grp):
            m = ev & (grp == g)[:, None, None]
            eg = codes[m]
            tg = max(int(m.sum()), 1)
            s.setdefault(name, {})[str(g)] = {"events": int(m.sum()), "ign": float(np.isin(eg, (1, 2, 3)).sum() / tg),
                                             "spread": float((eg == SPREAD).sum() / tg)}
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", choices=["train", "eval"], required=True)
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--dp", type=int, default=1, help="Zuendung: Abstand in Pixeln")
    ap.add_argument("--dd", type=int, default=1, help="Zuendung: Abstand in Tagen")
    ap.add_argument("--r", type=int, default=3, help="Ausbreitung: Kernradius (3 = 7x7 wie der Contagion-Kern)")
    ap.add_argument("--w", type=int, default=3, help="Ausbreitung: Nachbar brannte an Tag k-w .. k-1")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out_dir = exp / "derived_data" / "cause_labels"
    out = out_dir / f"{args.part}_cause{args.tag}.npz"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - wird nicht ueberschrieben")

    from wildfire.data.dataset import CubeReader
    from wildfire.paper.surv_eval import raw_fire_keys
    cube = CubeReader(exp)
    Hg, Wg = cube.H, cube.W
    if args.part == "train":
        # Die Trainingsziele zaehlen Tage als Indexschritte in cube.dates (pixel_landmarks); das ist nur dann
        # gleich der Kalenderzaehlung, wenn cube.dates lueckenlos taeglich ist.
        d0 = abs_day(cube.dates[0])
        gaps = sum(abs_day(s) != d0 + i for i, s in enumerate(cube.dates))
        print(f"[cause] cube.dates {cube.dates[0]} .. {cube.dates[-1]}, {len(cube.dates)} Tage, Luecken {gaps}", flush=True)
        if gaps:
            raise SystemExit("ABBRUCH: cube.dates nicht lueckenlos - Tageszaehlung von train_t und NFDB passt nicht")
        d = exp / "derived_data" / args.cube_dir
        me = dict(np.load(d / "train_meta.npz"))
        ev = np.load(d / "train_event.npy").astype(bool)
        t = np.load(d / "train_t.npy")
        L_abs = np.array([abs_day(cube.dates[int(L)]) for L in me["L"]], np.int64)
        extra = {"kind": me["kind"], "year": np.array([cube.dates[int(L)][:4] for L in me["L"]])}
    else:
        me = dict(np.load(exp / "derived_data" / args.surv_dir / "eval_meta.npz"))
        pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
        assert (pe["row"] == me["row"]).all() and (pe["L_abs"] == me["L_abs"]).all(), "Stichtage passen nicht"
        ev, t, L_abs = pe["ev"].astype(bool), pe["t"], me["L_abs"].astype(np.int64)
        extra = {"year": np.array([(DAY0 + dt.timedelta(days=int(x))).year for x in L_abs]),
                 "fire_in_window": me["fire_win_patch"].astype(int)}
    row, col = me["row"].astype(np.int64), me["col"].astype(np.int64)
    print(f"[cause] {args.part}: {len(row):,} Stichtage, {int(ev.sum()):,} Pixel-Ereignisse", flush=True)

    nk, nc = nfdb_keys(exp / "derived_data" / "events" / "ignitions_nfdb_min1ha.csv", Wg)
    fk, _ = raw_fire_keys(Path(cfg["dataset_root"]))
    observed = np.load(exp / "derived_data" / "masks" / "observed_nfdb.npy").astype(bool)
    print(f"[cause] {nk.size:,} NFDB-Zuendungen, {fk.size:,} MODIS-Feuer-Pixeltage", flush=True)

    codes = label(ev, t, L_abs, row, col, Hg, Wg, nk, nc, fk, observed, args)
    # Selbsttests
    assert ((codes > 0) == ev).all(), "Label ohne Ereignis oder Ereignis ohne Code"
    s = summary(codes, ev, extra)
    print(f"[cause] Anteile an den Ereignissen: {json.dumps(s['share'])}", flush=True)
    for name in extra:
        for g, v in sorted(s[name].items()):
            print(f"    {name} {g}: {v['events']:8d} Ereignisse, Zuendung {v['ign']:.3f}, Ausbreitung {v['spread']:.3f}",
                  flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cause=codes, row=row, col=col, L_abs=L_abs)
    (out_dir / f"{args.part}_cause{args.tag}.json").write_text(json.dumps({"args": vars(args), **s}, indent=1))
    print(f"[cause] {out}", flush=True)


if __name__ == "__main__":
    main()

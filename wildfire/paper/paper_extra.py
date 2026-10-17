"""Mehr Trainingsstichproben im Stil des Papers, aus unserem Wuerfel (nur Trainingsjahre).

Das Paper trainiert auf 4 347 Stichproben (1 449 Positive). Hier werden nach
denselben Regeln weitere Stichproben aus 2000-2019 gezogen; Validierung und
Test bleiben die Stichproben des Papers.

  Positiv   MODIS-Feuer in der Mittelzelle am Tag D (wie alle Train-Positiven
            im Cache, p9: train_samples_with_centre_fire = 1449)
  Negativ   kein Feuer im 13x13-Ausschnitt an D und kein Feuer innerhalb von
            --buf-px Zellen und --buf-days Tagen (HDF5: spatial_buf = 30,
            temporal_buf = 3); zwei Arten:
              random  gleichverteilt aus diesen Kandidaten
              hnm     FWI-HNM mit unserem nachgebauten Score (p3_hnm_score):
                      je zur Haelfte ueber und unter dem Pool-Median

Je Stichprobe wie im Cache: 61 Tage D-60 .. D, 55 Kanaele in Cache-Reihenfolge,
13x13 um (r, c). Dazu die Ziele fuer D und D+1. Der Wuerfel deckt nur den
1.4.-30.9. ab; D liegt deshalb zwischen dem 31.5. und dem 29.9.

Ausgabe unter derived_data/paper_extra/:
  X.npy (n, 61, 55, 13, 13) uint8, y.npy (Feuer im Ausschnitt an D),
  y_next_patch.npy, y_next_centre.npy, fire_next.npy (13x13 an D+1),
  meta.npz (row, col, D als Wuerfel-Tagesindex, date, kind 0 pos / 1 random / 2 hnm, s)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.surv_prepare import DAY_BITS, HALF, fire_keys, window

L_CACHE = 61
P = 2 * HALF + 1
FWI_DYN = list(range(19, 25))                    # Wuerfel-dyn-Indizes von FFMC .. FWI (= Cache-Kanaele 19-24)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-pos", type=int, default=8000)
    ap.add_argument("--n-neg", type=int, default=16000, help="je Art (random, hnm)")
    ap.add_argument("--n-cand", type=int, default=400_000, help="zufaellige Negativ-Kandidaten vor dem Puffer")
    ap.add_argument("--buf-px", type=int, default=30)
    ap.add_argument("--buf-days", type=int, default=3)
    ap.add_argument("--last-train-year", type=int, default=2019)
    ap.add_argument("--out-name", default="paper_extra")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    rng = np.random.default_rng(int(cfg["seed"]) + 7)

    cube = CubeReader(exp)
    H, W = cube.H, cube.W
    t_of = {d: i for i, d in enumerate(cube.dates)}
    year_of = np.array([int(d[:4]) for d in cube.dates])
    keys = fire_keys(Path(cfg["dataset_root"]), t_of, W)
    k_pix, k_day = keys // DAY_BITS, keys % DAY_BITS
    fire_set = set(keys.tolist())

    def day_ok(D: int) -> bool:
        """D-60 .. D+1 liegen zusammenhaengend im selben Jahr des Wuerfels."""
        a, b = cube._pos.get(D - L_CACHE + 1), cube._pos.get(D + 1)
        return a is not None and b is not None and a[0] == b[0] and b[1] - a[1] == L_CACHE

    days = np.array(sorted(D for D in cube._pos if year_of[D] <= args.last_train_year and day_ok(D)))
    day_set = set(days.tolist())
    print(f"[extra] {days.size} moegliche Tage D, {cube.dates[days[0]]} .. {cube.dates[days[-1]]}", flush=True)

    # Feuer je Tag (Pixel-Indizes), fuer Puffer und Ausschnitt-Labels
    order = np.argsort(k_day, kind="stable")
    kd, kp = k_day[order], k_pix[order]
    bounds = np.searchsorted(kd, np.arange(len(cube.dates) + 1))

    def fire_map(t: int) -> np.ndarray:
        m = np.zeros(H * W, bool)
        if 0 <= t < len(cube.dates):
            m[kp[bounds[t]:bounds[t + 1]]] = True
        return m.reshape(H, W)

    def patch_any(m: np.ndarray, r: int, c: int) -> np.ndarray:
        out = np.zeros((P, P), bool)
        r0, r1, c0, c1 = max(0, r - HALF), min(H, r + HALF + 1), max(0, c - HALF), min(W, c + HALF + 1)
        out[r0 - (r - HALF):r1 - (r - HALF), c0 - (c - HALF):c1 - (c - HALF)] = m[r0:r1, c0:c1]
        return out

    # ---- Positive: Feuer in der Mittelzelle an D
    pos_idx = np.where(np.isin(k_day, days))[0]
    pick = rng.choice(pos_idx, size=min(args.n_pos, pos_idx.size), replace=False)
    pr, pc, pD = k_pix[pick] // W, k_pix[pick] % W, k_day[pick]
    print(f"[extra] {pos_idx.size:,} Feuer-Pixeltage als moegliche Positive, gezogen {pick.size:,}", flush=True)

    # ---- Negativ-Kandidaten: gueltige Landzelle, Tag aus days, Puffer, kein Feuer im Ausschnitt
    valid = np.load(exp / "derived_data" / "masks" / "valid.npy")
    vr, vc = np.nonzero(valid)
    ci = rng.integers(0, vr.size, size=args.n_cand)
    cr, cc, cD = vr[ci], vc[ci], days[rng.integers(0, days.size, size=args.n_cand)]
    keep = np.zeros(args.n_cand, bool)
    size = 2 * args.buf_px + 1
    for D in np.unique(cD):
        sel = np.where(cD == D)[0]
        u = np.zeros((H, W), bool)
        for k in range(-args.buf_days, args.buf_days + 1):
            u |= fire_map(int(D) + k)
        buf = maximum_filter(u, size=size)                     # 13x13-Ausschnitt an D liegt im Puffer
        keep[sel] = ~buf[cr[sel], cc[sel]]
    cr, cc, cD = cr[keep], cc[keep], cD[keep]
    print(f"[extra] Negativ-Kandidaten nach Puffer ({args.buf_px} Zellen, +-{args.buf_days} Tage): "
          f"{keep.sum():,} von {args.n_cand:,}", flush=True)

    # FWI-HNM-Score s am Zentrumspixel an D, Gewichte und P99 aus p3_hnm_score
    p3 = json.loads((exp / "data_manifests" / "p3_hnm_score.json").read_text())
    w, p99, med = np.array(p3["w"]), np.array(p3["p99"]), float(p3["pool_median_nofire"])
    s = np.zeros(cr.size)
    for D in np.unique(cD):
        sel = np.where(cD == D)[0]
        year, li = cube._pos[int(D)]
        dyn, _ = cube._arrays(year)
        x = np.asarray(dyn[li, cr[sel], cc[sel], :])[:, FWI_DYN].astype(np.float64)   # nur die Kandidaten lesen
        s[sel] = (np.minimum(x / p99, 1.0) * w).sum(1)
    hi, lo = np.where(s > med)[0], np.where(s <= med)[0]
    n_h = args.n_neg // 2
    if hi.size < n_h or lo.size < args.n_neg - n_h or cr.size < 2 * args.n_neg:
        raise SystemExit(f"ABBRUCH: zu wenige Kandidaten (ueber Median {hi.size}, darunter {lo.size}, gesamt {cr.size})")
    hnm = np.concatenate([rng.choice(hi, n_h, replace=False), rng.choice(lo, args.n_neg - n_h, replace=False)])
    rest = np.setdiff1d(np.arange(cr.size), hnm)
    rnd = rng.choice(rest, args.n_neg, replace=False)
    print(f"[extra] Score: Median der Kandidaten {np.median(s):.3f} (Pool-Median {med:.3f}); "
          f"random ueber Median {(s[rnd] > med).mean():.2f}, hnm {(s[hnm] > med).mean():.2f}", flush=True)

    rows = np.concatenate([pr, cr[rnd], cr[hnm]]).astype(np.int64)
    cols = np.concatenate([pc, cc[rnd], cc[hnm]]).astype(np.int64)
    Ds = np.concatenate([pD, cD[rnd], cD[hnm]]).astype(np.int64)
    kind = np.concatenate([np.zeros(pr.size), np.ones(rnd.size), np.full(hnm.size, 2)]).astype(np.int8)
    sc = np.concatenate([np.full(pr.size, np.nan), s[rnd], s[hnm]])
    n = rows.size

    # ---- Ziele an D und D+1
    y = np.zeros(n, np.int8)
    fnext = np.zeros((n, P, P), np.uint8)
    cache_maps: dict[int, np.ndarray] = {}

    def fm(t: int) -> np.ndarray:
        if t not in cache_maps:
            if len(cache_maps) > 8:
                cache_maps.clear()
            cache_maps[t] = fire_map(t)
        return cache_maps[t]

    srt = np.argsort(Ds, kind="stable")
    for i in srt:
        D, r, c = int(Ds[i]), int(rows[i]), int(cols[i])
        y[i] = patch_any(fm(D), r, c).any()
        fnext[i] = patch_any(fm(D + 1), r, c)
    y_np = fnext.reshape(n, -1).max(1).astype(np.int8)
    y_nc = fnext[:, HALF, HALF].astype(np.int8)
    if not (y[kind == 0] == 1).all() or y[kind > 0].any():
        raise SystemExit("ABBRUCH: Label an D passt nicht zur Ziehung (Positiv = Feuer, Negativ = kein Feuer im Ausschnitt)")
    centre_D = np.array([((r * W + c) * DAY_BITS + D) in fire_set for r, c, D in zip(rows, cols, Ds)])
    assert centre_D[kind == 0].all()

    # ---- Fenster D-60 .. D
    out = exp / "derived_data" / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.uint8, shape=(n, L_CACHE, 55, P, P))
    for m, i in enumerate(srt):
        X[i] = window(cube, int(Ds[i]), int(rows[i]), int(cols[i]), T=L_CACHE)
        if m % 5000 == 0:
            print(f"    {m}/{n}", flush=True)
    bad = int(sum((X[i, -1, 0] > 0).any() != bool(y[i]) for i in range(0, n, max(1, n // 2000))))
    X.flush(); del X
    if bad:
        raise SystemExit(f"ABBRUCH: Feuerkanal im Fenster passt bei {bad} Stichproben nicht zum Label an D")
    np.save(out / "y.npy", y)
    np.save(out / "y_next_patch.npy", y_np)
    np.save(out / "y_next_centre.npy", y_nc)
    np.save(out / "fire_next.npy", fnext)
    dates = np.array([int(cube.dates[d].replace("-", "")) for d in Ds])
    np.savez(out / "meta.npz", row=rows, col=cols, D=Ds, date=dates, kind=kind, s=sc)

    months = {k: {int(mo): int(((dates // 100 % 100) == mo)[kind == k].sum()) for mo in range(5, 11)} for k in (0, 1, 2)}
    summary = {
        "n": int(n), "n_pos": int(pr.size), "n_neg_random": int(rnd.size), "n_neg_hnm": int(hnm.size),
        "bytes": human_bytes(n * L_CACHE * 55 * P * P),
        "years": f"{cube.dates[int(Ds.min())][:4]}..{cube.dates[int(Ds.max())][:4]}",
        "base_rate_next_patch": {k: float(y_np[kind == i].mean()) for i, k in enumerate(("pos", "random", "hnm"))},
        "pos_still_burning_next_day": float(y_np[kind == 0].mean()),
        "share_above_pool_median": {"random": float((s[rnd] > med).mean()), "hnm": float((s[hnm] > med).mean())},
        "months_by_kind": months,
    }
    print(f"[extra] {summary}", flush=True)
    write_manifest(cfg, "p14_paper_extra", args.config, {
        "rules": __doc__.split("Je Stichprobe")[0].strip(), "buf_px": args.buf_px, "buf_days": args.buf_days,
        "hnm_weights": w.tolist(), "hnm_p99": p99.tolist(), "pool_median": med,
        "train_years": f"<= {args.last_train_year}", "summary": summary,
    })


if __name__ == "__main__":
    main()

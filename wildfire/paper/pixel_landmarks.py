"""Mehr Trainingsdaten fuer das ST-DCTM pro Pixel: Stichtage aus dem ganzen Wuerfel.

Das Paper teilt nach Jahren (Training 2000-2019, Validierung 2020-2022, Test
2023-2025). Unser Wuerfel enthaelt dieselben 55 Kanaele (byte-genau gegen den
Cache geprueft, surv_prepare) fuer ganz Kanada an jedem Saisontag. Hier werden
deshalb beliebig viele Trainings-Stichtage aus 2000-2019 gezogen; Validierung
und Test bleiben die Stichproben des Papers.

Je Stichtag (Zelle s, letzter Fenstertag L):
  Eingabe   30 Tage bis einschliesslich L, 55 Kanaele, 13 x 13 um s,
            Cache-Reihenfolge (surv_prepare.window)
  Ziel      je Pixel des Ausschnitts: erster Tag > L mit MODIS-Feuer
            (MCD14A1 == 1, unmaskiert, dieselbe Quelle wie Cache-Kanal 0),
            Zeit t = Tag - L, zensiert bei 100 Tagen

Zwei Stroeme:
  fire    ein zufaelliger MODIS-Feuer-Pixeltag (r, c, d) der Trainingsjahre,
          Stichtag L = d - u mit u gleichverteilt in 1..100, Ausschnitt um
          (r, c) zufaellig um bis zu 6 Zellen verschoben - das Feuer liegt
          also irgendwo im Ausschnitt, nicht nur in der Mitte
  random  zufaellige Landzelle in Kanada, zufaelliger gueltiger Stichtag

Die Ereignisrate ist damit weit hoeher als in Wirklichkeit; ausgewertet wird
wie bisher die Rangordnung auf den Test-Stichproben des Papers.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.surv_prepare import DAY_BITS, HALF, T_WIN, fire_keys, window

H_FOLLOW = 100
P = 2 * HALF + 1


def pixel_events(keys: np.ndarray, Ls: np.ndarray, rr: np.ndarray, cc: np.ndarray, H: int, W: int):
    """Je Stichtag und Pixel des Ausschnitts: (event (n,13,13) bool, t (n,13,13) int16)."""
    dr, dc = np.meshgrid(np.arange(-HALF, HALF + 1), np.arange(-HALF, HALF + 1), indexing="ij")
    R = rr[:, None] + dr.ravel()[None]
    C = cc[:, None] + dc.ravel()[None]
    ok = (R >= 0) & (R < H) & (C >= 0) & (C < W)
    flat = np.where(ok, R * W + C, 0).astype(np.int64)
    pos = np.searchsorted(keys, flat * DAY_BITS + (Ls[:, None] + 1))
    nxt = np.where(pos < keys.size, keys[np.minimum(pos, keys.size - 1)], -1)
    same = ok & (nxt // DAY_BITS == flat)
    tte = np.where(same, nxt % DAY_BITS - Ls[:, None], 10 ** 6)
    event = tte <= H_FOLLOW
    t = np.minimum(tte, H_FOLLOW).astype(np.int16)
    return event.reshape(-1, P, P), t.reshape(-1, P, P)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-fire", type=int, default=120_000)
    ap.add_argument("--n-random", type=int, default=120_000)
    ap.add_argument("--last-train-year", type=int, default=2019)
    ap.add_argument("--out-name", default="paper_pixel")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    rng = np.random.default_rng(int(cfg["seed"]))

    cube = CubeReader(exp)
    H, W = cube.H, cube.W
    t_of = {d: i for i, d in enumerate(cube.dates)}
    year_of = np.array([int(d[:4]) for d in cube.dates])
    keys = fire_keys(Path(cfg["dataset_root"]), t_of, W)

    def landmark_ok(L: int) -> bool:
        return (L in cube._pos and (L - T_WIN + 1) in cube._pos
                and cube._pos[L - T_WIN + 1][0] == cube._pos[L][0]
                and cube._pos[L - T_WIN + 1][1] == cube._pos[L][1] - T_WIN + 1)

    valid_L = np.array(sorted(L for L in cube._pos if year_of[L] <= args.last_train_year and landmark_ok(L)))
    valid_set = set(valid_L.tolist())
    print(f"[pxlm] {valid_L.size} gueltige Stichtage {cube.dates[valid_L[0]]} .. {cube.dates[valid_L[-1]]}", flush=True)

    # ---- Strom fire: Anker auf MODIS-Feuer-Pixeltagen der Trainingsjahre
    k_day = keys % DAY_BITS
    k_pix = keys // DAY_BITS
    train_fire = np.where(year_of[np.minimum(k_day, len(year_of) - 1)] <= args.last_train_year)[0]
    print(f"[pxlm] {train_fire.size:,} Feuer-Pixeltage in den Trainingsjahren", flush=True)
    fr, fc, fL, fu, far, fac = [], [], [], [], [], []
    while len(fr) < args.n_fire:
        j = rng.choice(train_fire, size=args.n_fire)
        u = rng.integers(1, H_FOLLOW + 1, size=j.size)
        L = k_day[j] - u
        keep = np.array([int(x) in valid_set for x in L])
        a_r, a_c = k_pix[j] // W, k_pix[j] % W
        r = a_r + rng.integers(-HALF, HALF + 1, size=j.size)
        c = a_c + rng.integers(-HALF, HALF + 1, size=j.size)
        keep &= (r >= 0) & (r < H) & (c >= 0) & (c < W)
        fr += r[keep].tolist(); fc += c[keep].tolist(); fL += L[keep].tolist(); fu += u[keep].tolist()
        far += a_r[keep].tolist(); fac += a_c[keep].tolist()
    fr, fc, fL, fu, far, fac = (np.array(a[:args.n_fire]) for a in (fr, fc, fL, fu, far, fac))

    # ---- Strom random: Landzelle in Kanada, gueltiger Stichtag
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    lr_, lc_ = np.nonzero(land)
    pick = rng.integers(0, lr_.size, size=args.n_random)
    rr_, rc_ = lr_[pick], lc_[pick]
    rL = valid_L[rng.integers(0, valid_L.size, size=args.n_random)]

    rows = np.concatenate([fr, rr_]).astype(np.int64)
    cols = np.concatenate([fc, rc_]).astype(np.int64)
    Ls = np.concatenate([fL, rL]).astype(np.int64)
    kind = np.concatenate([np.zeros(fr.size, np.int8), np.ones(rr_.size, np.int8)])   # 0 fire, 1 random
    anchor_u = np.concatenate([fu, np.zeros(rr_.size, np.int64)])

    event, t = pixel_events(keys, Ls, rows, cols, H, W)

    # Plausibilitaet: Im Strom fire brennt das Ankerpixel spaetestens u Tage nach L.
    pr, pc = far - fr + HALF, fac - fc + HALF
    i_f = np.arange(fr.size)
    anchor_ok = event[i_f, pr, pc] & (t[i_f, pr, pc] <= fu)
    print(f"[pxlm] Ankerpixel brennt spaetestens nach u Tagen: {anchor_ok.mean():.4f}", flush=True)
    if anchor_ok.mean() < 0.999:
        raise SystemExit("ABBRUCH: Ereignisse passen nicht zu den Ankern - Zeitindex pruefen")
    n = rows.size
    out_dir = exp / "derived_data" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out_dir / "train_X.npy", mode="w+", dtype=np.uint8,
                                  shape=(n, T_WIN, 55, P, P))
    order = np.argsort(Ls, kind="mergesort")
    for m, j in enumerate(order):
        X[j] = window(cube, int(Ls[j]), int(rows[j]), int(cols[j]))
        if m % 20000 == 0:
            print(f"    {m}/{n}", flush=True)
    X.flush(); del X
    np.save(out_dir / "train_event.npy", event)
    np.save(out_dir / "train_t.npy", t)
    np.savez(out_dir / "train_meta.npz", row=rows, col=cols, L=Ls, kind=kind, anchor_u=anchor_u)

    summary = {
        "n": int(n), "n_fire_stream": int(fr.size), "n_random_stream": int(rr_.size),
        "bytes": human_bytes(n * T_WIN * 55 * P * P),
        "years": f"{cube.dates[int(Ls.min())][:4]}..{cube.dates[int(Ls.max())][:4]}",
        "pixel_event_share": {"fire": float(event[kind == 0].mean()), "random": float(event[kind == 1].mean())},
        "landmarks_with_any_event": {"fire": float(event[kind == 0].any((1, 2)).mean()),
                                     "random": float(event[kind == 1].any((1, 2)).mean())},
        "centre_event_share": {"fire": float(event[kind == 0, HALF, HALF].mean()),
                               "random": float(event[kind == 1, HALF, HALF].mean())},
        "t_event_quantiles_pixel_events": {q: float(np.percentile(t[event], q)) for q in (10, 25, 50, 75, 90)},
        "distinct_anchor_fire_pixel_days": int(np.unique((far * W + fac) * DAY_BITS + fL + fu).size),
        "paper_train_reference": {"samples": 4347, "fire_pixels_at_t": 17818},
    }
    print(f"[pxlm] {summary}", flush=True)
    write_manifest(cfg, "p11_pixel_landmarks", args.config, {
        "split": f"Training nur aus Jahren <= {args.last_train_year} (Paper: Validierung 2020-2022, Test 2023-2025)",
        "target": "je Pixel erster Tag > L mit MCD14A1 == 1 (unmaskiert), zensiert bei 100 Tagen",
        "streams": {"fire": "Anker auf Feuer-Pixeltag, u in 1..100, Ausschnitt um bis zu 6 Zellen verschoben",
                    "random": "Landzelle in Kanada, gueltiger Saison-Stichtag"},
        "summary": summary,
    })


if __name__ == "__main__":
    main()

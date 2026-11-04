"""Flaechentreue Datensaetze (equal_area.py) fuer dieselben Stichtage wie bisher.

Die Stichtage (Zelle, Tag) werden aus den bestehenden Datensaetzen uebernommen, nur Eingabe
und Ziel werden flaechentreu neu geschnitten. So bleiben die Vergleiche mit den Originalpixeln
Stichtag fuer Stichtag gepaart.

  train   derived_data/<src> (pixel_landmarks)  -> <src>_ea: train_X, train_event, train_t, train_meta
  surv    derived_data/<src> (surv_eval)        -> <src>_ea: eval/val/km _meta (+ _X fuer eval, val)
  cache   derived_data/paper/<cache>_L61        -> <cache>_ea_L61: X (61 Tage), Labels an D+1,
          ok.npy = alle 61 Fenstertage liegen im Wuerfel (1.4.-30.9.); ausserhalb ist X = 0
  rate    Tageslabels fuer den Rate-Kopf (wie rate_labels) fuer <dir>
  future  Wetter-Oracle (wie future_weather) fuer <dir>

Jeder Schritt prueft sich selbst: Ereignisse im flaechentreuen Ausschnitt muessen die
Ereignisse des Originalausschnitts enthalten (er liegt ganz darin), und der Feuerkanal der
Fenster muss zu den Rohdaten passen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.equal_area import (HALF, P, EqualAreaGrid, ea_daily_fire, ea_fire_between, ea_pixel_events,
                                       summary, window_ea)
from wildfire.paper.surv_prepare import T_WIN, fire_keys

H_FOLLOW = 100
L61 = 61


# ------------------------------------------------------------------ Fenster parallel schreiben
def _fill(job: dict) -> int:
    exp = Path(job["exp"])
    cube = CubeReader(exp)
    grid = EqualAreaGrid.from_experiment(exp)
    out = np.load(job["path"], mmap_mode="r+")
    sel = job.get("channels")
    for j, L, r, c, T in zip(job["idx"], job["L"], job["r"], job["c"], job["T"]):
        x = window_ea(cube, grid, int(L), int(r), int(c), int(T))
        if sel is not None:
            x = x[:, sel]
        out[int(j), :x.shape[0]] = x
    out.flush()
    return len(job["idx"])


def fill_windows(exp: Path, path: Path, shape: tuple, idx, Ls, rr, cc, T, workers: int, channels=None):
    """Schreibt window_ea je Stichtag nach path (uint8-memmap der Form shape), sortiert nach Tag."""
    X = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    del X
    idx, Ls, rr, cc = (np.asarray(a) for a in (idx, Ls, rr, cc))
    T = np.broadcast_to(np.asarray(T), idx.shape)
    order = np.argsort(Ls, kind="mergesort")
    chunks = np.array_split(order, max(1, workers * 8))
    jobs = [{"exp": str(exp), "path": str(path), "idx": idx[ch], "L": Ls[ch], "r": rr[ch], "c": cc[ch],
             "T": T[ch], "channels": channels} for ch in chunks if ch.size]
    done, t0 = 0, time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k in ex.map(_fill, jobs):
            done += k
            print(f"    {done:,}/{idx.size:,} ({time.time() - t0:.0f} s)", flush=True)


def superset_check(name, ev_o, t_o, ev_e, t_e, tol=0.999):
    """Originalausschnitt liegt im flaechentreuen: jedes Ereignis bleibt, und es kommt nicht spaeter."""
    ok = float((~ev_o | (ev_e & (t_e <= t_o))).mean())
    print(f"[ea] Selbsttest {name}: Original-Ereignis im flaechentreuen Ausschnitt enthalten {ok:.5f}", flush=True)
    if ok < tol:
        raise SystemExit(f"ABBRUCH: {name} - flaechentreue Ereignisse passen nicht zu den Originalen")
    return ok


# ------------------------------------------------------------------ Schritte
def build_train(args, cfg, exp, cube, grid):
    src, dst = exp / "derived_data" / args.src, exp / "derived_data" / (args.dst or args.src + "_ea")
    dst.mkdir(parents=True, exist_ok=True)
    m = dict(np.load(src / "train_meta.npz"))
    Ls, rr, cc = m["L"], m["row"], m["col"]
    keys = fire_keys(Path(cfg["dataset_root"]), {d: i for i, d in enumerate(cube.dates)}, cube.W)
    t0 = time.time()
    ev, t = ea_pixel_events(keys, Ls, rr, cc, grid)
    print(f"[ea] Ereignisse {Ls.size:,} Stichtage in {time.time() - t0:.0f} s", flush=True)
    ev_o, t_o = np.load(src / "train_event.npy"), np.load(src / "train_t.npy")
    n = Ls.size
    chk = {"centre": superset_check("Mitte", ev_o[:, HALF, HALF], t_o[:, HALF, HALF], ev[:, HALF, HALF], t[:, HALF, HALF]),
           "patch": superset_check("Ausschnitt", ev_o.reshape(n, -1).any(1), t_o.reshape(n, -1).min(1),
                                   ev.reshape(n, -1).any(1), t.reshape(n, -1).min(1))}
    np.save(dst / "train_event.npy", ev)
    np.save(dst / "train_t.npy", t)
    np.savez(dst / "train_meta.npz", **m)
    fill_windows(exp, dst / "train_X.npy", (n, T_WIN, 55, P, P), np.arange(n), Ls, rr, cc, T_WIN, args.workers)
    X = np.load(dst / "train_X.npy", mmap_mode="r")
    j = np.sort(np.random.default_rng(0).choice(n, min(3000, n), replace=False))
    fire_x = np.asarray(X[j, -1, 0]).reshape(j.size, -1).max(1) > 0
    fire_k = ea_fire_between(keys, Ls[j], Ls[j], rr[j], cc[j], grid).reshape(j.size, -1).any(1)
    agree = float((fire_x == fire_k).mean())
    print(f"[ea] Selbsttest Feuer am Stichtag, Fenster gegen Rohdaten: {agree:.4f}", flush=True)
    if agree < 0.98:
        raise SystemExit("ABBRUCH: Fenster und Rohdaten passen nicht zusammen")
    kind = m["kind"]
    return {"dir": str(dst), "n": int(n), "bytes": human_bytes(n * T_WIN * 55 * P * P), "self_test": chk,
            "fire_channel_agreement": agree,
            "pixel_event_share": {"fire": float(ev[kind == 0].mean()), "random": float(ev[kind == 1].mean())},
            "patch_event_share": {"fire": float(ev[kind == 0].any((1, 2)).mean()),
                                  "random": float(ev[kind == 1].any((1, 2)).mean())},
            "patch_event_share_original": {"fire": float(ev_o[kind == 0].any((1, 2)).mean()),
                                           "random": float(ev_o[kind == 1].any((1, 2)).mean())}}


def build_surv(args, cfg, exp, cube, grid):
    from wildfire.paper.surv_eval import raw_fire_keys
    src, dst = exp / "derived_data" / args.src, exp / "derived_data" / (args.dst or args.src + "_ea")
    dst.mkdir(parents=True, exist_ok=True)
    keys, _ = raw_fire_keys(Path(cfg["dataset_root"]))
    out = {"dir": str(dst)}
    for part in ("eval", "val", "km"):
        if not (src / f"{part}_meta.npz").exists():
            continue
        m = dict(np.load(src / f"{part}_meta.npz"))
        rr, cc, La, Lc = m["row"], m["col"], m["L_abs"], m["L_cube"]
        n = rr.size
        ev, t = ea_pixel_events(keys, La, rr, cc, grid)
        pe, pt = ev.reshape(n, -1).any(1), t.reshape(n, -1).min(1)
        fw = ea_fire_between(keys, La - T_WIN + 1, La, rr, cc, grid)
        new = dict(m, ev_centre=ev[:, HALF, HALF], t_centre=t[:, HALF, HALF], ev_patch=pe, t_patch=pt,
                   fire_win_patch=fw.reshape(n, -1).any(1), fire_win_centre=fw[:, HALF, HALF])
        chk = {"centre": superset_check(f"{part} Mitte", m["ev_centre"], m["t_centre"], new["ev_centre"], new["t_centre"]),
               "patch": superset_check(f"{part} Ausschnitt", m["ev_patch"], m["t_patch"], pe, pt)}
        np.savez(dst / f"{part}_meta.npz", **new)
        info = {"n": int(n), "self_test": chk,
                "event_rate_100d": {"centre": float(new["ev_centre"].mean()), "patch": float(pe.mean())},
                "event_rate_100d_original": {"centre": float(m["ev_centre"].mean()), "patch": float(m["ev_patch"].mean())},
                "fire_in_window": {"centre": float(new["fire_win_centre"].mean()),
                                   "patch": float(new["fire_win_patch"].mean())}}
        print(f"[ea] {part}: {info}", flush=True)
        if (src / f"{part}_X.npy").exists():
            fill_windows(exp, dst / f"{part}_X.npy", (n, T_WIN, 55, P, P), np.arange(n), Lc, rr, cc, T_WIN, args.workers)
            X = np.load(dst / f"{part}_X.npy", mmap_mode="r")
            j = np.arange(0, n, max(1, n // 3000))
            fire_x = np.asarray(X[j, :, 0]).reshape(j.size, -1).max(1) > 0
            info["fire_channel_agreement"] = float((fire_x == new["fire_win_patch"][j]).mean())
            print(f"[ea] Selbsttest {part} Feuer im Fenster, Fenster gegen Rohdaten: "
                  f"{info['fire_channel_agreement']:.4f}", flush=True)
            if info["fire_channel_agreement"] < 0.98:
                raise SystemExit("ABBRUCH: Fenster und Rohdaten passen nicht zusammen")
        out[part] = info
    return out


def build_cache(args, cfg, exp, cube, grid):
    from wildfire.paper.surv_eval import abs_day, raw_fire_keys
    src = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    dst = exp / "derived_data" / "paper" / f"{args.cache}_ea_L61"
    dst.mkdir(parents=True, exist_ok=True)
    meta, split = np.load(src / "meta.npy"), np.load(src / "split.npy")
    n = meta.shape[0]
    D = [dt.date(int(v // 10000), int(v // 100 % 100), int(v % 100)) for v in meta[:, 0]]
    rr, cc = meta[:, 1].astype(np.int64), meta[:, 2].astype(np.int64)
    t_of = {d: i for i, d in enumerate(cube.dates)}

    def full(d):                                         # alle 61 Tage D-60 .. D lueckenlos im Wuerfel
        L = t_of.get(d.isoformat())
        if L is None or L not in cube._pos or (L - L61 + 1) not in cube._pos:
            return -1
        a, b = cube._pos[L - L61 + 1], cube._pos[L]
        return L if (a[0] == b[0] and b[1] - a[1] == L61 - 1) else -1

    Lc = np.array([full(d) for d in D])
    ok = Lc >= 0
    for f in ("meta.npy", "split.npy", "y.npy"):
        shutil.copy(src / f, dst / f)
    shutil.copy(src / "y_next_patch.npy", dst / "y_next_patch_orig.npy")
    shutil.copy(src / "y_next_centre.npy", dst / "y_next_centre_orig.npy")
    np.save(dst / "ok.npy", ok)

    keys, _ = raw_fire_keys(Path(cfg["dataset_root"]))
    a1 = np.array([abs_day(d + dt.timedelta(days=1)) for d in D])
    fmap = ea_fire_between(keys, a1, a1, rr, cc, grid)                 # D+1, aus den Rohdaten, auch im Oktober
    yp, yc = fmap.reshape(n, -1).any(1).astype(np.int64), fmap[:, HALF, HALF].astype(np.int64)
    np.save(dst / "fire_next.npy", fmap)
    np.save(dst / "y_next_patch.npy", yp)
    np.save(dst / "y_next_centre.npy", yc)
    yo = np.load(src / "y_next_patch.npy").astype(int)
    contain = float((yo <= yp).mean())
    print(f"[ea] Selbsttest D+1: Original-Label im flaechentreuen Ausschnitt enthalten {contain:.5f}", flush=True)
    if contain < 0.999:
        raise SystemExit("ABBRUCH: D+1-Labels passen nicht zu den Originalen")

    j = np.where(ok)[0]
    fill_windows(exp, dst / "X.npy", (n, L61, 55, P, P), j, Lc[j], rr[j], cc[j], L61, args.workers)
    X = np.load(dst / "X.npy", mmap_mode="r")
    chk = j[np.arange(0, j.size, max(1, j.size // 2000))]
    aD = np.array([abs_day(D[i]) for i in chk])
    fire_x = np.asarray(X[chk, -1, 0]).reshape(chk.size, -1).max(1) > 0
    fire_k = ea_fire_between(keys, aD, aD, rr[chk], cc[chk], grid).reshape(chk.size, -1).any(1)
    agree = float((fire_x == fire_k).mean())
    print(f"[ea] Selbsttest Feuer an D, Fenster gegen Rohdaten: {agree:.4f}", flush=True)
    if agree < 0.98:
        raise SystemExit("ABBRUCH: Fenster und Rohdaten passen nicht zusammen")
    by = {s: {"n": int((split == s).sum()), "ok": int((ok & (split == s)).sum()),
              "pos_rate_ea": float(yp[ok & (split == s)].mean()), "pos_rate_orig": float(yo[ok & (split == s)].mean())}
          for s in ("train", "val", "test")}
    months = {str(mo): {"n": int(sum(d.month == mo for d in D)), "ok": int(sum(d.month == mo and o for d, o in zip(D, ok)))}
              for mo in range(4, 11)}
    print(f"[ea] Cache {args.cache}: {by}", flush=True)
    return {"dir": str(dst), "ok_rule": "alle 61 Tage D-60 .. D im Wuerfel (Saison 1.4.-30.9.)", "splits": by,
            "months": months, "d1_contained": contain, "fire_channel_agreement": agree,
            "labels": "y_next_patch/centre/fire_next flaechentreu an D+1 (Rohdaten); *_orig = Labels des Papers"}


def build_rate(args, cfg, exp, cube, grid):
    from wildfire.paper.rate_labels import H_FOLLOW as HF
    cdir = exp / "derived_data" / args.src
    m = np.load(cdir / "train_meta.npz")
    keys = fire_keys(Path(cfg["dataset_root"]), {d: i for i, d in enumerate(cube.dates)}, cube.W)
    patch, centre = ea_daily_fire(keys, m["L"], m["row"], m["col"], grid, HF)
    ev, t = np.load(cdir / "train_event.npy"), np.load(cdir / "train_t.npy")
    pe, pt = ev.reshape(len(ev), -1).any(1), t.reshape(len(t), -1).min(1)
    first = np.where(patch.any(1), patch.argmax(1) + 1, HF)
    agree = float(((patch.any(1) == pe) & (np.where(pe, first == pt, True))).mean())
    print(f"[ea] Selbsttest erstes Feuer im Ausschnitt = Survival-Ereignis: {agree:.5f}", flush=True)
    if agree < 0.999:
        raise SystemExit("ABBRUCH: Tageslabels passen nicht zu den Survival-Zielen")
    np.savez_compressed(cdir / "train_rate.npz", patch=patch, centre=centre, L=m["L"])
    return {"dir": str(cdir), "n": int(patch.shape[0]), "self_test_agreement": agree,
            "daily_rate_patch": float(patch.mean()), "daily_rate_centre": float(centre.mean())}


def build_future(args, cfg, exp, cube, grid):
    from wildfire.paper.future_weather import F_DAYS, WEATHER_CH, available_future
    cdir = exp / "derived_data" / args.src
    m = np.load(cdir / "train_meta.npz")
    Ls, rr, cc = m["L"], m["row"], m["col"]
    n = Ls.size
    kf = np.array([available_future(cube, int(L)) for L in Ls], np.int8)
    has = np.where(kf > 0)[0]
    fill_windows(exp, cdir / "train_Xf.npy", (n, F_DAYS, len(WEATHER_CH), P, P), has, Ls[has] + kf[has],
                 rr[has], cc[has], kf[has], args.workers, channels=WEATHER_CH)
    X, Xf = np.load(cdir / "train_X.npy", mmap_mode="r"), np.load(cdir / "train_Xf.npy", mmap_mode="r")
    j = np.random.default_rng(0).choice(np.where(kf == F_DAYS)[0], 50, replace=False)
    ok_f = all(np.array_equal(Xf[i], window_ea(cube, grid, int(Ls[i]) + F_DAYS, int(rr[i]), int(cc[i]), F_DAYS)[:, WEATHER_CH])
               for i in j)
    ok_x = all(np.array_equal(X[i], window_ea(cube, grid, int(Ls[i]), int(rr[i]), int(cc[i]), T_WIN)) for i in j[:20])
    print(f"[ea] Selbsttest Folgefenster {ok_f}, Bezug train_X {ok_x}", flush=True)
    if not (ok_f and ok_x):
        raise SystemExit("ABBRUCH: Folgefenster passen nicht zum Wuerfel")
    np.save(cdir / "train_kf.npy", kf)
    return {"dir": str(cdir), "n": int(n), "share_full_30_days": float((kf == F_DAYS).mean())}


STEPS = {"train": build_train, "surv": build_surv, "cache": build_cache, "rate": build_rate, "future": build_future}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--step", choices=list(STEPS), required=True)
    ap.add_argument("--src", default="", help="train/surv: Quellordner unter derived_data; rate/future: Zielordner")
    ap.add_argument("--dst", default="", help="Standard: <src>_ea")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not args.src:
        args.src = {"train": "paper_pixel", "surv": "surv_eval_test", "rate": "paper_pixel_ea",
                    "future": "paper_pixel_ea", "cache": ""}[args.step]
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    cube = CubeReader(exp)
    grid = EqualAreaGrid.from_experiment(exp)
    print(f"[ea] Schritt {args.step}: Zellen {summary(grid)}", flush=True)
    t0 = time.time()
    info = STEPS[args.step](args, cfg, exp, cube, grid)
    info["seconds"] = round(time.time() - t0)
    print(f"[ea] fertig in {info['seconds']} s", flush=True)
    write_manifest(cfg, f"p30_ea_{args.step}" + (f"_{args.cache}" if args.step == "cache" else f"_{args.src}"),
                   args.config, {"grid": summary(grid), "method": "equal_area.py: Zellen ns x ns, Spalten je Zeile "
                                 "flaechengewichtet, Feuer/Ereignisse ueber die Pixelmitte, Landbedeckung Mehrheitsflaeche",
                                 "result": info})


if __name__ == "__main__":
    main()

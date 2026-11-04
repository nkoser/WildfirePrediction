"""Zustandsmerkmale je Pixel am Stichtag (Multi-State v2): Feuerhistorie ueber das 30-Tage-Fenster hinaus.

Aus dem Fire-Index (state_labels.index_keys, dieselbe Quelle wie der Feuerkanal der Eingabe) je Pixel des
13 x 13-Ausschnitts am Stichtag L (Tage absolut), nur Feuer <= L (kausal):
  ds_log    log(1 + min(days_since, 365)) / log(366); days_since = L - letzter Feuertag (0 = brennt an L)
  none365   1, wenn im letzten Jahr kein Feuer (days_since = 366), sonst 0
  n183_log  log(1 + Feuertage in [L-182, L]) / log(184)
  is_O      Zustand erloschen (Feuer in [L-183, L-7], keins in [L-6, L]; state_labels.pixel_state)
  is_B      Zustand brennend (Feuer in [L-6, L])
Pixel ausserhalb des Gitters: ds_log 1, none365 1, Rest 0.

Teile (je derived_data/state_features/<teil><tag>.npy, float16, + .json):
  train   Wuerfel-Stichtage (paper_pixel/train_meta.npz)                       (n, 169, 5)
          dazu train_types<tag>.npy (n, 169) int8: Typ des ersten Pixel-Feuers (train_event / train_t) wie
          state_labels: 0 kein Ereignis, 1 continuation (B), 2 return (O), 3 spread (U, ein ANDERES Pixel im
          7 x 7-Kern brannte in [k-3, k-1]), 4 ignition (U, sonst)
  eval    Test-Stichtage 2023-2025 (surv_eval_test/eval_meta.npz)              (n, 169, 5)
  val     Val-Stichtage 2020-2022 (surv_eval_test/val_meta.npz)                (n, 169, 5)
  paper   Stichproben des Paper-Caches (paper/<cache>_L61), je dt in DTS am Fensterende
          e = L - dt - 2 wie paper_metrics_ckpt                                 (n, len(DTS), 169, 5)

  python -m wildfire.paper.state_features --config ... --part train|eval|val|paper [--max-n N --tag _smoke]
Bricht ab, wenn die Ausgabe existiert.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config
from wildfire.data.dataset import CubeReader
from wildfire.paper.state_labels import count_in, index_keys, neighbour_fire, patch_flat, pixel_state

FEATS = ("ds_log", "none365", "n183_log", "is_O", "is_B")
NF = len(FEATS)
DAY0 = dt.date(2000, 1, 1)
TYPE_NAMES = ("none", "continuation", "return", "spread", "ignition")


def features(keys, row, col, L, H, W, chunk: int = 4000) -> np.ndarray:
    """row, col, L (n,) -> (n, 169, 5) float16."""
    n = row.size
    out = np.zeros((n, 169, NF), np.float16)
    for a in range(0, n, chunk):
        s = slice(a, min(a + chunk, n))
        flat, ok, _, _ = patch_flat(row[s].astype(np.int64), col[s].astype(np.int64), H, W)
        Ls = np.asarray(L[s], np.int64)[:, None]
        st, ds, _ = pixel_state(keys, flat, Ls)
        n183 = count_in(keys, flat, Ls - 182, Ls)
        f = np.stack([np.log1p(np.minimum(ds, 365)) / np.log(366), (ds >= 366).astype(float),
                      np.log1p(n183) / np.log(184), (st == 1).astype(float), (st == 2).astype(float)], -1)
        f[~ok] = (1.0, 1.0, 0.0, 0.0, 0.0)
        out[s] = f.astype(np.float16)
    return out


def train_types(keys, row, col, L, ev, t, feats, H, W) -> np.ndarray:
    """Typ des ersten Feuers je Pixel (n, 169) int8, Kodierung TYPE_NAMES."""
    n = row.size
    ev = ev.reshape(n, 169).astype(bool)
    t = t.reshape(n, 169).astype(np.int64)
    typ = np.zeros((n, 169), np.int8)
    isB, isO = feats[..., 4] > 0.5, feats[..., 3] > 0.5
    typ[ev & isB] = 1
    typ[ev & isO & ~isB] = 2
    u = ev & ~isB & ~isO
    b, p = np.nonzero(u)
    R = row[b].astype(np.int64) + p // 13 - 6
    C = col[b].astype(np.int64) + p % 13 - 6
    k = np.asarray(L, np.int64)[b] + t[b, p]
    nf = neighbour_fire(keys, R, C, k, H, W, 3)
    typ[b, p] = np.where(nf, 3, 4)
    return typ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--part", required=True, choices=("train", "eval", "val", "paper"))
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", default="next_patch")
    ap.add_argument("--max-n", type=int, default=0, help="Rauchtest: nur die ersten N Stichtage")
    ap.add_argument("--keys", default="index", choices=("index", "raw", "union"),
                    help="Feuerquelle: fire_index (Gueltigkeitsmaske), Roh-MCD14A1 oder Vereinigung")
    ap.add_argument("--no-abort", action="store_true", help="Selbsttest nur berichten (Diagnose)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    od = exp / "derived_data" / "state_features"
    out = od / f"{args.part}{args.tag}.npy"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert")
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cube = CubeReader(exp)
    if args.keys == "index":
        keys = index_keys(exp)
    else:
        from wildfire.paper.state_labels import raw_keys
        keys = raw_keys(cfg) if args.keys == "raw" else np.union1d(index_keys(exp), raw_keys(cfg))
    absd = lambda s: (dt.date.fromisoformat(s) - DAY0).days  # noqa: E731
    info: dict = {"feats": FEATS, "part": args.part, "keys": args.keys}
    if args.part == "train":
        cd = exp / "derived_data" / args.cube_dir
        m = np.load(cd / "train_meta.npz")
        n = args.max_n or m["L"].size
        row, col = m["row"][:n], m["col"][:n]
        L = np.array([absd(cube.dates[int(x)]) for x in m["L"][:n]], np.int64)
        f = features(keys, row, col, L, cube.H, cube.W)
        ev = np.load(cd / "train_event.npy", mmap_mode="r")[:n]
        tt = np.load(cd / "train_t.npy", mmap_mode="r")[:n]
        typ = train_types(keys, row, col, L, np.asarray(ev), np.asarray(tt), f, cube.H, cube.W)
        np.save(od / f"train_types{args.tag}.npy", typ)
        e = typ[typ > 0]
        info["type_share_events"] = {TYPE_NAMES[i]: float((e == i).mean()) for i in range(1, 5)}
        info["n_events"] = int(e.size)
    elif args.part in ("eval", "val"):
        m = np.load(exp / "derived_data" / args.surv_dir / f"{args.part}_meta.npz")
        n = args.max_n or m["row"].size
        L = m["L_abs"][:n].astype(np.int64)
        f = features(keys, m["row"][:n], m["col"][:n], L, cube.H, cube.W)
    else:
        from wildfire.paper import targets
        from wildfire.paper.compare import DTS
        d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
        meta = np.load(d / "meta.npy")
        n = args.max_n or meta.shape[0]
        LC = np.load(d / "X.npy", mmap_mode="r").shape[1]
        Lt = LC + targets.shift(args.target)
        D = np.array([(dt.date(int(v) // 10000, int(v) // 100 % 100, int(v) % 100) - DAY0).days
                      for v in meta[:n, 0]], np.int64)                 # Tag des letzten Fensterindex LC - 1
        f = np.zeros((n, len(DTS), 169, NF), np.float16)
        for j, d_ in enumerate(DTS):
            e = Lt - d_ - 2                                            # Fensterende wie paper_metrics_ckpt.scores
            f[:, j] = features(keys, meta[:n, 1], meta[:n, 2], D - (LC - 1 - e), cube.H, cube.W)
        info.update({"dts": list(DTS), "LC": int(LC), "shift": int(targets.shift(args.target))})
    if args.part != "paper":                                   # Selbsttest: is_B gegen Feuerkanal der letzten 7 Fenstertage
        from wildfire.paper.prepare import FIRE_CH
        xp = (exp / "derived_data" / args.cube_dir / "train_X.npy" if args.part == "train"
              else exp / "derived_data" / args.surv_dir / f"{args.part}_X.npy")
        chk = np.arange(0, n, max(1, n // 2000))
        win = (np.asarray(np.load(xp, mmap_mode="r")[chk, -7:, FIRE_CH]) > 0).any(1).reshape(chk.size, -1)
        isb = f[chk, :, 4] > 0.5
        rec = float((isb & win).sum() / max(win.sum(), 1))
        prec = float((isb & win).sum() / max(isb.sum(), 1))
        info["selftest_isB_vs_window"] = {"recall": rec, "precision": prec, "n_window_fire_px": int(win.sum())}
        print(f"[sfeat] Selbsttest is_B gegen Feuerkanal (7 Tage): Recall {rec:.4f}, Precision {prec:.4f} "
              f"({int(win.sum())} Pixel)", flush=True)
        yr = np.array([(DAY0 + dt.timedelta(days=int(x))).year for x in L[chk]])
        by = {int(y): round(float((isb[yr == y] & win[yr == y]).sum() / max(win[yr == y].sum(), 1)), 4) for y in np.unique(yr)}
        info["selftest_isB_vs_window"]["recall_by_year"] = by
        print(f"[sfeat] Recall je Jahr: {by}", flush=True)
        if win.sum() >= 50 and rec < 0.9 and not args.no_abort:
            raise SystemExit("ABBRUCH: Zustandsmerkmale passen nicht zum Feuerkanal der Eingabe")
    np.save(out, f)
    flat = f.reshape(-1, NF).astype(np.float64)                 # float32-Summe saettigt bei 2^24
    info.update({"n": int(n), "shape": list(f.shape), "mean": dict(zip(FEATS, flat.mean(0).round(5).tolist())),
                 "seconds": round(time.time() - t0)})
    (od / f"{args.part}{args.tag}.json").write_text(json.dumps(info, indent=1))
    print(f"[sfeat] {out} {f.shape} in {time.time() - t0:.0f} s; " + json.dumps(info.get("type_share_events", {}))
          + " Mittel " + json.dumps(info["mean"]), flush=True)


if __name__ == "__main__":
    main()

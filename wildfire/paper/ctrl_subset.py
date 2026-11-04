"""Status-Versuch nur mit Wuerfel-Stichtagen ab 2011 (erst ab da gibt es Lageberichte als Zeitreihe).

Aus einem fertigen Status-Kontext (ctrl_ctx, z. B. derived_data/ctx_ctrl) zwei NEUE Kontext-Ordner fuer
sts_extreme --ctx px:
  <out>       train_rows.npy (Wuerfel-Stichtage mit Jahr >= --first-year) + train_px.npy dieser Zeilen; surv_val_px,
              surv_eval_px, cache_px, channels.json als Symlink auf den Quellordner (unveraendert).
  <zero-out>  gleiche train_rows.npy, aber alle Kontextdateien null: Referenz ohne Status auf derselben Teilmenge
              (der Zweig ctx_px startet bei null und sieht nur Nullen, lernt also hoechstens einen konstanten Bias).
sts_extreme liest <name>_rows.npy und erwartet dazu <name>_px.npy mit genau diesen Zeilen (load_ctx).

Ausgabe NEU: derived_data/<out>/, derived_data/<zero-out>/, Manifest p55_ctrl_subset_<out>.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader

LINKED = ("surv_val_px.npy", "surv_eval_px.npy", "cache_px.npy", "channels.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", default="ctx_ctrl")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--out", default="ctx_ctrl11")
    ap.add_argument("--zero-out", default="ctx_zero11")
    ap.add_argument("--first-year", type=int, default=2011)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    dd = exp / "derived_data"
    src, out, zout = dd / args.src, dd / args.out, dd / args.zero_out
    for d in (out, zout):
        if d.exists():
            raise SystemExit(f"ABBRUCH: {d} existiert schon - anderen Namen waehlen")
    t0 = time.time()
    dates = CubeReader(exp).dates
    L = np.load(dd / args.cube_dir / "train_meta.npz")["L"]
    yr = np.array([dt.date.fromisoformat(dates[int(v)]).year for v in L])
    rows = np.where(yr >= args.first_year)[0]
    px = np.load(src / "train_px.npy", mmap_mode="r")
    assert px.shape[0] == L.size, (px.shape, L.size)
    sub = np.asarray(px[rows])
    rep = (sub[:, :4].reshape(len(sub), -1).max(1) > 0).mean()
    print(f"[sub] {rows.size:,} von {L.size:,} Stichtagen ab {args.first_year}; Anteil mit Bericht "
          f"{rep:.4f} (vorher {(np.asarray(px[:, :4]).reshape(len(px), -1).max(1) > 0).mean():.4f})", flush=True)

    out.mkdir(parents=True)
    zout.mkdir(parents=True)
    for d in (out, zout):
        np.save(d / "train_rows.npy", rows.astype(np.int64))
    np.save(out / "train_px.npy", sub)
    np.save(zout / "train_px.npy", np.zeros_like(sub))
    for f in LINKED:
        os.symlink(os.path.relpath(src / f, out), out / f)
        if f.endswith(".npy"):
            a = np.load(src / f, mmap_mode="r")
            np.save(zout / f, np.zeros(a.shape, a.dtype))
    (zout / "channels.json").write_text(json.dumps({**json.loads((src / "channels.json").read_text()),
                                                    "zero": True}, indent=1))
    info = {"args": vars(args), "n_rows": int(rows.size), "n_all": int(L.size), "share_with_report": float(rep),
            "by_year": {int(k): int(v) for k, v in sorted(Counter(yr.tolist()).items())}}
    write_manifest(cfg, f"p55_ctrl_subset_{args.out}", args.config, info)
    print(f"[sub] Jahre {info['by_year']}", flush=True)
    print(f"[sub] {out}, {zout} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

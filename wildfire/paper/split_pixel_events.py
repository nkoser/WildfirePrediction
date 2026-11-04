"""Ereignisse je Pixel (13 x 13) fuer die Test-Stichtage eines Kandidaten-Splits, wie region_eval.build_events.

region_eval.build_events schreibt fest nach derived_data/region_eval/pixel_events.npz (zeitlicher Split, existiert).
Hier dieselbe Rechnung und derselbe Selbsttest fuer derived_data/<surv-dir>/eval_meta.npz nach
derived_data/region_eval_<sp>/pixel_events.npz. Leser wie cause_labels (Teil eval), multistate_eval und msv2_eval
laufen ueber split_run --redirect region_eval/pixel_events.npz=region_eval_<sp>/pixel_events.npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.pixel_landmarks import pixel_events
from wildfire.paper.surv_eval import raw_fire_keys
from wildfire.paper.surv_prepare import HALF


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--surv-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="z. B. region_eval_ys")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out_dir = exp / "derived_data" / args.out_dir
    out = out_dir / "pixel_events.npz"
    if args.out_dir == "region_eval" or out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon oder ist der zeitliche Pfad")
    me = dict(np.load(exp / "derived_data" / args.surv_dir / "eval_meta.npz"))
    cube = CubeReader(exp)
    keys, _ = raw_fire_keys(Path(cfg["dataset_root"]))
    ev, t = pixel_events(keys, me["L_abs"], me["row"], me["col"], cube.H, cube.W)
    pe, pt = ev.reshape(len(ev), -1).any(1), t.reshape(len(t), -1).min(1)
    checks = {"ev_patch": float((pe == me["ev_patch"].astype(bool)).mean()),
              "t_patch": float((pt[pe] == me["t_patch"][pe]).mean()),
              "ev_centre": float((ev[:, HALF, HALF] == me["ev_centre"].astype(bool)).mean())}
    print(f"[spe] Selbsttest Ereignisse je Pixel gegen eval_meta: {checks}", flush=True)
    if min(checks.values()) < 0.9999:
        raise SystemExit("ABBRUCH: Ereignisse je Pixel passen nicht zu eval_meta.npz")
    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out, ev=ev, t=t, row=me["row"], col=me["col"], L_abs=me["L_abs"])
    write_manifest(cfg, f"p50_split_pixel_events_{args.out_dir}", args.config,
                   {"args": vars(args), "checks": checks, "output": str(out)})


if __name__ == "__main__":
    main()

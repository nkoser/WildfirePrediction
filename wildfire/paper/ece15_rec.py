"""ECE/MCE mit 10 und 15 Bins fuer die gespeicherten Vorhersagen des 3D-U-Net-Nachbaus (roh).

Der Nachbau (reconstruct, reconstruct_f1) speichert seine Val/Test-Vorhersagen je Konfiguration und
Seed in predictions/paper/rec_<cache>_cache<run>.npz. Hier werden fuer die genannten Laeufe die
rohen Test-Vorhersagen mit beiden Bin-Zahlen bewertet, Mittel ueber die Seeds je Delta t, daneben
die Werte des Papers (Tabelle 10, UNet3D). Schreibt metrics/ece15/summary.md und ein Manifest.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.metrics import calibration_errors

PAPER_UNET = {1: (0.158, 0.705), 7: (0.047, 0.164), 15: (0.103, 0.314), 30: (0.055, 0.185)}   # ECE, MCE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap.add_argument("--runs", nargs="+",
                    default=["_next_paperhp_f1w", "_next_paperhp_f1w_nofire", "_next_paperhp_w0",
                             "_next_paperhp_ema95"])
    ap.add_argument("--out-name", default="ece15")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    y = targets.labels(d, args.target)
    shift = targets.shift(args.target)

    rows, res = [], {}
    for run in args.runs:
        f = exp / "predictions" / "paper" / f"rec_{args.cache}_cache{run}.npz"
        if not f.exists():
            print(f"[ece15] {f} fehlt", flush=True)
            continue
        z = np.load(f)
        yt = y[z["idx_test"]]
        acc = defaultdict(list)
        for k in z.files:
            m = re.fullmatch(r"(.+)_T30_end(\d+)_s(\d+)_test", k)
            if not m:
                continue
            dt_ = int(m.group(2)) - 1 + shift
            e10, m10 = calibration_errors(z[k], yt, 10)
            e15, m15 = calibration_errors(z[k], yt, 15)
            acc[(m.group(1), dt_)].append((e10, e15, m10, m15))
        for (cset, dt_), v in sorted(acc.items(), key=lambda t: (t[0][0], t[0][1])):
            e10, e15, m10, m15 = np.mean(v, 0)
            res[f"{run}|{cset}|{dt_}"] = {"n_seeds": len(v), "ece10": e10, "ece15": e15, "mce10": m10, "mce15": m15}
            pe, pm = PAPER_UNET.get(dt_, (np.nan, np.nan))
            rows.append(f"| {run} | {cset} | {dt_} | {len(v)} | {e10:.3f} | {e15:.3f} | {m10:.3f} | {m15:.3f} "
                        f"| {pe:.3f} | {pm:.3f} |")
            print(rows[-1], flush=True)

    out = exp / "metrics" / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(
        f"# ECE/MCE mit 10 vs 15 Bins, 3D-U-Net-Nachbau roh ({args.cache}, Ziel {args.target}, Test)\n\n"
        "Paper Anhang D: M = 15. Paper-Werte: Tabelle 10, UNet3D (mit FWI-HNM).\n\n"
        "| Lauf | Kanaele | dt | Seeds | ECE10 | ECE15 | MCE10 | MCE15 | ECE Paper | MCE Paper |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    write_manifest(cfg, f"p17_ece15_rec_{args.cache}{targets.tag(args.target)}", args.config,
                   {"runs": args.runs, "results": res, "paper_unet_ece_mce": PAPER_UNET})


if __name__ == "__main__":
    main()

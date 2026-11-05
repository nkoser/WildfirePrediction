"""Does the model notice the rebuilt inputs? Compare two risk-map files of the same date and model, one from
MPCFire's released inputs and one from rebuilt inputs, over the forecast area.

    python -m demo.live.compare_maps <released npz> <rebuilt npz> [--taus 1 7 30]

Per lead time: Spearman correlation of log10 F, median and 95th percentile of |delta log10 F|, and the AUC of F
against the fires observed after the date (pixels without a fire in the input window) for both files.

Acceptance, fixed on 2026-11-05 before any comparison was run: at every lead time, Spearman >= 0.95, median
|delta log10 F| <= 0.10 (a factor of 1.26, 2.5 % of the map's color range) and |delta AUC| <= 0.01.
"""

from __future__ import annotations

import argparse

import numpy as np

SPEARMAN_MIN, MEDIAN_DLOG_MAX, DAUC_MAX = 0.95, 0.10, 0.01


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties counted half)."""
    from scipy.stats import rankdata
    pos = label.sum()
    neg = label.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> None:
    from scipy.stats import spearmanr
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("released")
    ap.add_argument("rebuilt")
    ap.add_argument("--taus", type=int, nargs="+", default=[1, 7, 30])
    args = ap.parse_args()
    a, b = np.load(args.released), np.load(args.rebuilt)
    ok_all = True
    print(f"{args.released}\n  vs {args.rebuilt}")
    for t in args.taus:
        fa, fb = a[f"F_{t}"], b[f"F_{t}"]
        m = np.isfinite(fa) & np.isfinite(fb)
        la, lb = np.log10(np.clip(fa[m], 1e-7, 1)), np.log10(np.clip(fb[m], 1e-7, 1))
        rho = float(spearmanr(la, lb).statistic)
        d = np.abs(la - lb)
        keep = m & ~a["fire_window"]
        lab = (a["first_fire"] > 0) & (a["first_fire"] <= t)
        auc_a, auc_b = auc(fa[keep], lab[keep]), auc(fb[keep], lab[keep])
        ok = rho >= SPEARMAN_MIN and np.median(d) <= MEDIAN_DLOG_MAX and abs(auc_a - auc_b) <= DAUC_MAX
        ok_all &= ok
        print(f"  tau {t:2d}: Spearman {rho:.4f}, |dlog10 F| median {np.median(d):.3f} p95 {np.percentile(d, 95):.3f}, "
              f"AUC released {auc_a:.4f} rebuilt {auc_b:.4f} (d {auc_b - auc_a:+.4f}), "
              f"{int(lab[keep].sum())} fires; {'ok' if ok else 'FAIL'}")
    print("  acceptance:", "passed" if ok_all else "failed",
          f"(Spearman >= {SPEARMAN_MIN}, median |dlog10 F| <= {MEDIAN_DLOG_MAX}, |dAUC| <= {DAUC_MAX})")


if __name__ == "__main__":
    main()

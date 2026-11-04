"""Metriken des Paper-Protokolls: Precision, Recall, F1, PR-AUC, ECE, MCE, Brier.

Die Formeln stehen im Anhang D des Papers (Gl. 6-14), der uns nicht vorliegt.
Benutzt werden die ueblichen Definitionen; jede Abweichung waere ein Grund,
hier nachzuziehen:

  Precision/Recall/F1  Schwelle 0,5 auf p
  PR-AUC               Average Precision (Stufenintegral, wie sklearn)
  ECE                  10 gleich breite Bins auf [0, 1], gewichtet mit dem
                       Anteil der Stichproben im Bin: sum_b n_b/N |acc_b - conf_b|
  MCE                  max_b |acc_b - conf_b| ueber nicht leere Bins
  Brier                mean (p - y)^2

Zusaetzlich ROC-AUC, weil sie als einzige dieser Groessen nicht vom
Klassenverhaeltnis abhaengt und damit mit unseren Ergebnissen vergleichbar ist.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ASSUMPTIONS = {
    "threshold": 0.5,
    "pr_auc": "average precision (sklearn)",
    "ece_mce_bins": "10 gleich breite Bins auf [0,1]",
    "note": "Paper Anhang D liegt nicht vor; Standarddefinitionen angenommen.",
}


def calibration_errors(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece, mce = 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(y[m].mean() - p[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def paper_metrics(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    p = np.asarray(p, float)
    y = np.asarray(y).astype(int)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ece, mce = calibration_errors(p, y)
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "ece": ece, "mce": mce,
        "brier": float(np.mean((p - y) ** 2)),
        "n": int(y.size), "n_pos": int(y.sum()), "tp": tp, "fp": fp, "fn": fn,
        **operating_points(p, y),
    }


RECALLS = (0.7, 0.8, 0.81, 0.85, 0.9, 0.91)


def operating_points(p: np.ndarray, y: np.ndarray, recalls=RECALLS) -> dict:
    """Schwellenfreie Arbeitspunkte: groesste Precision bei Recall >= r, bester F1 ueber alle Schwellen.

    Precision und Recall bei 0,5 haengen an der Kalibrierung; ein Modell, das zu hoch schaetzt,
    hat dort hohen Recall und niedrige Precision. Verglichen bei gleichem Recall bleibt nur die
    Rangfolge."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    j = int(np.argmax(f1[:-1]))
    out = {"f1_max": float(f1[j]), "thr_f1_max": float(thr[j]),
           "prec_at_f1_max": float(prec[j]), "rec_at_f1_max": float(rec[j])}
    for r in recalls:
        ok = rec >= r
        out[f"prec_at_rec_{r}"] = float(prec[ok].max()) if ok.any() else 0.0
    return out

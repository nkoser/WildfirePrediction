"""Startet ein paper-Modul mit ECE/MCE ueber 15 Bins statt 10, sonst unveraendert.

metrics.py rechnet ECE und MCE mit 10 gleich breiten Bins, weil Anhang D des Papers bei der
Implementierung fehlte. Laut Anhang D (Gl. 11-13) nimmt das Paper M = 15. Hier wird
metrics.calibration_errors auf 15 Bins umgestellt (paper_metrics liest die Funktion zur
Laufzeit aus den Modul-Globals) und dann split_run gestartet; das Manifest bekommt damit
den Suffix aus --manifest-suffix (z. B. _m15), die alten Manifeste bleiben unberuehrt.

  python -m wildfire.paper.m15_run --manifest-suffix _m15 \
      wildfire.paper.paper_metrics_ckpt --config C --cache cache_G --models ... --tag _abl_g0
"""

from __future__ import annotations

import functools

import wildfire.paper.metrics as metrics
from wildfire.paper import split_run

N_BINS = 15


def main() -> None:
    metrics.calibration_errors = functools.partial(metrics.calibration_errors, n_bins=N_BINS)
    metrics.ASSUMPTIONS["ece_mce_bins"] = f"{N_BINS} gleich breite Bins auf [0,1] (Paper Anhang D, M = 15)"
    metrics.ASSUMPTIONS["note"] = "Anhang D liegt vor; ECE/MCE mit M = 15 wie im Paper."
    print(f"[m15] ECE/MCE mit {N_BINS} Bins", flush=True)
    split_run.main()


if __name__ == "__main__":
    main()

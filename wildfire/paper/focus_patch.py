"""Startet stress_test bzw. importance_v2 unveraendert, legt aber um jedes geladene Netz die Huelle aus
sts_focus (resid: Offset der Feuerhistorie-Basis) bzw. sts_inputs (Kanalmaske). Ohne diese Huelle waeren
lh_resid oder lh_fireonly dort falsch ausgewertet; fdrop und ignw brauchen keine (wirken nur im Training).

  python -m wildfire.paper.focus_patch --target stress --wrap focus=resid -- <Argumente von stress_test>
  python -m wildfire.paper.focus_patch --target importance --wrap inputs=fireonly -- <Argumente von importance_v2>

Die Koeffizienten fuer resid kommen aus derived_data/sts_focus/coef_<name>_s<seed>.npz, abgeleitet aus dem
Checkpoint-Namen sts_extreme_<name>_s<seed>_s<seed>.pt.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from wildfire.audit.common import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("stress", "importance"), required=True)
    ap.add_argument("--wrap", required=True, help="focus=<varianten> oder inputs=<satz>")
    ap.add_argument("--pad", choices=("zeros", "reflect"), default="zeros", help="Stem-Padding (Laeufe _rf)")
    own, rest = ap.parse_known_args()
    rest = [a for a in rest if a != "--"]
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", required=True)
    pk, _ = peek.parse_known_args(rest)
    exp = Path(load_config(pk.config)["experiment_root"])
    kind, val = own.wrap.split("=", 1)

    def wrapper(net, path):
        if kind == "inputs":
            from wildfire.paper.sts_inputs import INPUTS, wrap
            return wrap(net, INPUTS[val], own.pad)
        from wildfire.paper.sts_focus import load_coef, wrap
        v = tuple(val.split(","))
        coef = None
        if "resid" in v:
            stem = re.sub(r"_s\d+$", "", Path(path).stem).removeprefix("sts_extreme_")
            coef = load_coef(exp / "derived_data" / "sts_focus" / f"coef_{stem}.npz")
        return wrap(net, v, 0.5, 4.0, coef, 0.0, 1.0, own.pad)

    def patch(orig):
        def load(path, device="cpu"):
            net = wrapper(orig(path, device), path)
            print(f"[fpatch] {Path(path).name}: Huelle {own.wrap}", flush=True)
            return net
        return load

    if own.target == "stress":
        from wildfire.paper import stress_test as mod
        mod.load_sts = patch(mod.load_sts)
    else:
        from wildfire.paper import importance_v2 as mod
        mod.load_net = patch(mod.load_net)
    sys.argv = [sys.argv[0]] + rest
    mod.main()


if __name__ == "__main__":
    main()

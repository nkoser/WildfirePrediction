"""Regional-Merkmale ohne die kanadaweiten Spalten (ytd_ca, anytd_ca) fuer sts_extreme --ctx reg.

Anlass: reg_shift (p37) zeigt, dass ytd_ca/anytd_ca je Jahr fast konstant sind (nur ~20 Werte im Training),
2023 zu 55 bzw. 90 % ausserhalb des Trainingsbereichs liegen und im Training nicht monoton mit dem Feuer
zusammenhaengen. Statt sts_extreme zu aendern, werden die beiden Spalten auf 0 gesetzt: nach der Normierung
(x - Mittel) / (Std + 1e-6) sind sie konstant 0 und tragen nichts bei.
Liest derived_data/<src>/{train,surv_val,surv_eval,cache}_reg.npy, schreibt NEU nach derived_data/<dst>/
(bricht ab, wenn <dst> existiert). Pixel-Merkmale werden nicht kopiert (nur --ctx reg).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.ctx_features import REG_NAMES

DROP = ("ytd_ca", "anytd_ca")
SETS = ("train", "surv_val", "surv_eval", "cache")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", default="ctx_features")
    ap.add_argument("--dst", default="ctx_reg_local")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    dd = Path(cfg["experiment_root"]) / "derived_data"
    src, dst = dd / args.src, dd / args.dst
    if dst.exists():
        raise SystemExit(f"ABBRUCH: {dst} existiert schon")
    cols = [REG_NAMES.index(k) for k in DROP]
    dst.mkdir(parents=True)
    shapes = {}
    for s in SETS:
        r = np.load(src / f"{s}_reg.npy")
        r = r.copy()
        r[..., cols] = 0
        np.save(dst / f"{s}_reg.npy", r)
        shapes[s] = list(r.shape)
        print(f"[ctxl] {s}_reg {r.shape}: Spalten {DROP} auf 0", flush=True)
    write_manifest(cfg, f"p38_{args.dst}", args.config, {"src": str(src), "dst": str(dst), "dropped": DROP,
                                                          "cols": cols, "shapes": shapes})


if __name__ == "__main__":
    main()

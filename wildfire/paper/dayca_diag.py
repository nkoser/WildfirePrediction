"""Diagnose fuer fire_dayca: liegt die vorhergesagte Ankunft an der Front oder verstreut im Ausschnitt?

Fuer die Ereignisse des Karten-Viewers (symbolic_spread/maps_viewer/data_<v>.json): Abstand jedes Pixels zum Feuer der
letzten 3 Tage bis L (Kategorie 3); Anteil der Ankunftsmasse sum P(Ankunft <= t) in den Abstandsringen 0-2, 3-5,
6-10, 11-20, > 20 px, je Modell, dazu die beobachteten neuen Pixel. dayca aus fire_dayca/maps/test_dayca.npz.
Nur Lesen; Ausgabe auf stdout.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from wildfire.audit.common import load_config

RINGS = ((0, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 6))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--viewer", default="v1")
    ap.add_argument("--dayca-tag", default="")
    args = ap.parse_args()
    exp = Path(load_config(args.config)["experiment_root"])
    d = json.loads((exp / "metrics" / "symbolic_spread" / "maps_viewer" / f"data_{args.viewer}.json").read_text())
    z = np.load(exp / "metrics" / f"fire_dayca{args.dayca_tag}" / "maps" / "test_dayca.npz")
    dec = lambda b: np.asarray(Image.open(io.BytesIO(base64.b64decode(b))))   # noqa: E731
    names = [m["name"] for m in d["models"]] + ["dayca"]
    for t in ("7", "30"):
        tot = {n: np.zeros(len(RINGS)) for n in names + ["beobachtet"]}
        for e in d["events"]:
            obs = dec(e["obs"])
            dist = ndimage.distance_transform_edt(obs != 3)
            new = (obs == 4) | ((obs == 5) if t == "30" else False)
            valid = (obs == 0) | (obs >= 4)
            maps = {n: dec(e["pred"][n][t]).astype(np.float64) / 255 for n in e["pred"]}
            maps["dayca"] = z[f"{e['i']}_1.0_{t}"].astype(np.float64)
            maps["beobachtet"] = new.astype(np.float64)
            for n, P in maps.items():
                for k, (a, b) in enumerate(RINGS):
                    tot[n][k] += P[valid & (dist >= a) & (dist <= b)].sum()
        print(f"\nAnteil der Ankunftsmasse bis Tag {t} je Abstand zum Feuer bei L (px), {len(d['events'])} Ereignisse; "
              f"in Klammern: Masse relativ zu beobachtet")
        print("Modell        " + "  ".join(f"{a:>2}-{b if b < 1e5 else '':<3}" for a, b in RINGS))
        ob = tot["beobachtet"].sum()
        for n, v in tot.items():
            print(f"{n:<13} " + "  ".join(f"{x / max(v.sum(), 1e-9):6.2f}" for x in v) + f"   ({v.sum() / ob:.2f}x)")


if __name__ == "__main__":
    main()

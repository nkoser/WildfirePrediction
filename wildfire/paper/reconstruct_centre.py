"""3D-U-Net des Papers auf dem Mittelzellen-Label an D (Feuer in der Mittelzelle am Stichtag).

reconstruct.py kennt als Ziele cache (Ausschnitt an D), next_patch (Ausschnitt an D+1)
und next_centre (Mittelzelle an D+1). Hier fehlt nur die Mittelzelle an D: alles bleibt
wie in reconstruct.py (Netz, Fenster, Normierung, Training), nur das geladene y.npy wird
durch targets.centre_labels(d, X, "cache") ersetzt, also MODIS-Feuer in der Mittelzelle
am letzten Cache-Tag. Mit --target cache und --end dt + 1 liegt das Fenster wie beim
Cache-Label; der Zieltag ist nie Teil der Eingabe. Im Manifest steht target "centre".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wildfire.paper import reconstruct as rec
from wildfire.paper import targets

LABEL = "Feuer in der Mittelzelle an D (Kanal 0 am letzten Cache-Tag, targets.centre_labels)"


class _NumpyCentre:
    """numpy, nur dass y.npy als Mittelzellen-Label an D geladen wird."""

    def __init__(self):
        self.X = None

    def __getattr__(self, k):
        return getattr(np, k)

    def load(self, p, *a, **kw):
        if Path(p).name == "y.npy":
            if self.X is None:
                raise SystemExit("ABBRUCH: X.npy muss vor y.npy geladen sein")
            y = targets.centre_labels(Path(p).parent, self.X, "cache")
            print(f"[rec-centre] Label: {LABEL}, Basisrate {y.mean():.3f}", flush=True)
            return y
        z = np.load(p, *a, **kw)
        if Path(p).name == "X.npy":
            self.X = z
        return z


def write_manifest_centre(cfg, stage, config_path, payload):
    if payload.get("target") != "cache":
        raise SystemExit("ABBRUCH: reconstruct_centre nur mit --target cache")
    payload["target"] = "centre"
    payload["label"] = LABEL
    return _write_manifest(cfg, stage, config_path, payload)


_write_manifest = rec.write_manifest

if __name__ == "__main__":
    rec.np = _NumpyCentre()
    rec.write_manifest = write_manifest_centre
    rec.main()

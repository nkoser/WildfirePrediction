"""sts_extreme mit maskierten Eingabekanaelen und wahlweise Reflect-Padding im Stem (Treiber- und Randdiagnose).

Gleiches Protokoll, gleicher Code wie sts_extreme (Ablation: Wuerfel-Stichtage, Ziel D+1, d 64, 4 Bloecke,
EMA 0.999, 10 x 3000 Schritte); sts_extreme.py bleibt unveraendert. Dieses Skript ersetzt nur sts_extreme.build
durch eine Huelle:
  --inputs all       alle 55 Kanaele (Referenz)
           nofire    Kanalsatz no_fire aus channels.py (Feuermaske raus, wie slurm 85/86)
           fireonly  nur die Feuermaske (FIRE_CH); Saison ueber doy bleibt wie im Modell
           nomodis   Kanalsatz no_modis aus channels.py (Feuermaske und alle MODIS-Kanaele 38-53 raus)
           nothermal Feuermaske und die feuerempfindlichen Kanaele raus: LST_day, LST_night, BT_b20/21/31/32
                     (FIRE_THERMAL, Indizes aus prepare.CHANNEL_NAMES); Vegetation, Reflexion, Emissivitaet bleiben
  Maskierte Kanaele werden nach der Normierung auf 0 gesetzt (wie channels.masked_std), im Training und in
  jeder Auswertung. Contagion liest den Brennzustand direkt aus dem Feuerkanal der Eingabe; die Maske wirkt vor
  dem ganzen Netz, mit nofire ist deshalb auch der Ausbreitungspfad leer (0 * sd + mu = mu < 0.5 -> nicht brennend).
  --pad reflect      padding_mode 'reflect' in beiden Stem-Faltungen statt Zero-Padding (Randartefakte)
Alle uebrigen Argumente gehen an sts_extreme (--head, --seeds, --tag, ...). Checkpoints, Hazards und Manifest
p35_sts_extreme<tag> wie dort; zusaetzlich Manifest p41_sts_inputs<tag> mit Maske und Padding. Die Checkpoints
tragen die Maske nicht: Auswertungen spaeter immer mit build_masked(...) bzw. diesem Skript.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

from wildfire.audit.common import load_config, write_manifest
from wildfire.paper import sts_extreme
from wildfire.paper.channels import N_CH, keep_mask
from wildfire.paper.prepare import CHANNEL_NAMES, FIRE_CH

FIRE_THERMAL = tuple(CHANNEL_NAMES.index(c) for c in ("MCD14A1_fire", "LST_day", "LST_night", "BT_b20", "BT_b21",
                                                     "BT_b31", "BT_b32"))
INPUTS = {"all": np.ones(N_CH, bool), "nofire": keep_mask("no_fire"), "fireonly": np.arange(N_CH) == FIRE_CH,
          "nomodis": keep_mask("no_modis"), "nothermal": ~np.isin(np.arange(N_CH), FIRE_THERMAL)}
PADS = ("zeros", "reflect")
_ORIG_BUILD = sts_extreme.build


def wrap(net: nn.Module, keep: np.ndarray, pad: str) -> nn.Module:
    """Maske vor dem Netz (normierte Eingabe), Padding-Modus im Stem."""
    if pad != "zeros":
        for m in net.stem.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                m.padding_mode = pad
    if keep.all():
        return net
    k = torch.from_numpy(keep.astype(np.float32))[None, None, :, None, None]
    fwd = net.forward

    def masked(x, doy, *a, **kw):
        return fwd(x * k.to(x.device, x.dtype), doy, *a, **kw)

    net.forward = masked
    return net


def build_masked(inputs: str, pad: str):
    keep = INPUTS[inputs]
    return lambda *a, **kw: wrap(_ORIG_BUILD(*a, **kw), keep, pad)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", choices=tuple(INPUTS), default="all")
    ap.add_argument("--pad", choices=PADS, default="zeros")
    own, rest = ap.parse_known_args()
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", required=True)
    peek.add_argument("--tag", required=True)
    pk, _ = peek.parse_known_args(rest)
    keep = INPUTS[own.inputs]
    print(f"[sin] Eingabe {own.inputs}: {int(keep.sum())} von {N_CH} Kanaelen, Padding {own.pad}", flush=True)
    sts_extreme.build = build_masked(own.inputs, own.pad)
    sys.argv = [sys.argv[0]] + rest
    sts_extreme.main()
    cfg = load_config(pk.config)
    write_manifest(cfg, f"p41_sts_inputs{pk.tag}", pk.config, {
        "inputs": own.inputs, "pad": own.pad, "keep_channels": np.where(keep)[0].tolist(),
        "sts_extreme_args": rest, "sts_extreme_manifest": f"p35_sts_extreme{pk.tag}"})


if __name__ == "__main__":
    main()

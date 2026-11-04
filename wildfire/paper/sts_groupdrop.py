"""sts_extreme mit zufaelligem Gruppen-Dropout im Training, Grundlage fuer Shapley-Werte je Variablengruppe.

Permutation und Hintergrund-SHAP setzen fremde Werte ein (Naturpark mit Stadtstrassen) und messen damit auch, wie
das Netz auf Eingaben ausserhalb der Trainingsverteilung reagiert. Hier lernt das Netz stattdessen, mit fehlenden
Gruppen umzugehen: je Trainingsstichprobe wird eine Teilmenge der GROUPS ausgeblendet (normiert auf 0 = Kanalmittel,
alle 30 Tage, alle Pixel; wie sts_inputs/channels.masked_std, also wie in den Retrain-Ablationen). Die Zahl der
behaltenen Gruppen ist gleichverteilt auf 0..10, die Gruppen darin zufaellig - so kommt jede Teilmengengroesse
gleich oft vor, wie es die Shapley-Gewichte brauchen. Der Kalendertag bleibt immer. Bei Contagion sieht der
Ausbreitungspfad mit ausgeblendetem Feuer kein Feuer (0 * sd + mu < 0.5).
Auswertung (shapley_groups.py) mit fester Maske je Teilmenge, ohne Zufall; im eval-Modus ist die Huelle inaktiv.

sts_extreme.py bleibt unveraendert; ersetzt wird nur sts_extreme.build. Uebrige Argumente an sts_extreme
(--head, --seeds, --tag, ...). Ausgaben wie dort, dazu Manifest p49_sts_groupdrop<tag>.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

from wildfire.audit.common import load_config, write_manifest
from wildfire.paper import sts_extreme
from wildfire.paper.channels import N_CH

# Gruppen wie importance.GROUPS ohne die Sammelgruppen; zusammen genau alle 55 Kanaele
SHAP_GROUPS = {
    "fire": [0],
    "era5_weather": list(range(1, 19)),
    "fire_danger": list(range(19, 26)),
    "infrastructure": list(range(26, 31)),
    "terrain": list(range(31, 38)),
    "vegetation": list(range(38, 42)),
    "reflectance": list(range(42, 46)),
    "lst_emissivity": list(range(46, 50)),
    "brightness_temp": list(range(50, 54)),
    "landcover": [54],
}
assert sorted(c for v in SHAP_GROUPS.values() for c in v) == list(range(N_CH))
G_OF_CH = np.zeros(N_CH, np.int64)
for _g, _cs in enumerate(SHAP_GROUPS.values()):
    G_OF_CH[_cs] = _g
_ORIG_BUILD = sts_extreme.build


def random_keep(B: int, device) -> torch.Tensor:
    """(B, G) bool: je Zeile k ~ U{0..G} behaltene Gruppen, zufaellig gewaehlt."""
    G = len(SHAP_GROUPS)
    k = torch.randint(0, G + 1, (B, 1), device=device)
    rank = torch.rand(B, G, device=device).argsort(1).argsort(1)
    return rank < k


def wrap(net: nn.Module) -> nn.Module:
    fwd = net.forward
    g_of = torch.from_numpy(G_OF_CH)

    def gdrop(x, doy, *a, **kw):
        if net.training:
            keep_g = random_keep(x.shape[0], x.device)                                 # (B, G)
            keep = keep_g[:, g_of.to(x.device)].to(x.dtype)                            # (B, C)
            x = x * keep[:, None, :, None, None]
        return fwd(x, doy, *a, **kw)

    net.forward = gdrop
    return net


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _, rest = ap.parse_known_args()
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", required=True)
    peek.add_argument("--tag", required=True)
    pk, _ = peek.parse_known_args(rest)
    print(f"[sgd] Gruppen-Dropout ueber {len(SHAP_GROUPS)} Gruppen: {', '.join(SHAP_GROUPS)}", flush=True)
    sts_extreme.build = lambda *a, **kw: wrap(_ORIG_BUILD(*a, **kw))
    sys.argv = [sys.argv[0]] + rest
    sts_extreme.main()
    cfg = load_config(pk.config)
    write_manifest(cfg, f"p49_sts_groupdrop{pk.tag}", pk.config, {
        "groups": SHAP_GROUPS, "mask_value": "0 (normiert)", "subset_sampling": "k ~ U{0..G}, Gruppen zufaellig",
        "sts_extreme_args": rest, "sts_extreme_manifest": f"p35_sts_extreme{pk.tag}"})


if __name__ == "__main__":
    main()

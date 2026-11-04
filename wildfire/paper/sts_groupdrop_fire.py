"""sts_groupdrop mit staerkerem Feuer-Dropout: Grundlage fuer Contagion-Shapley ohne Feuer.

Im Gruppen-Dropout-Netz (sts_groupdrop, slurm 217) ist die Feuergruppe in etwa der Haelfte der Trainingsstichproben
sichtbar. Contagion stuetzt sich dann auf den Ausbreitungspfad und lernt den Ersatz ueber Helligkeitstemperatur und
LST schlecht: mit ausgeblendetem Feuer v(S) 0.669 bei dt 1 gegen 0.782 im echten Retrain ohne Feuer (Gegenprobe
metrics/shapley_groups). Hier wird nach random_keep die Feuergruppe zusaetzlich mit --p-fire ausgeblendet, alle
anderen Gruppen bleiben wie in sts_groupdrop (k ~ U{0..G}, zufaellige Gruppen). Feuer sichtbar also mit
Wahrscheinlichkeit 0.5 * (1 - p_fire). Auswertung wie bisher (shapley_groups.py, feste Maske, Huelle inaktiv).

sts_groupdrop.py und sts_extreme.py bleiben unveraendert; ersetzt werden nur sts_groupdrop.random_keep und
sts_extreme.build. Uebrige Argumente an sts_extreme. Ausgaben wie dort, dazu Manifest p53_sts_groupdrop_fire<tag>.
"""

from __future__ import annotations

import argparse
import sys

import torch

from wildfire.audit.common import load_config, write_manifest
from wildfire.paper import sts_extreme, sts_groupdrop

FIRE_G = list(sts_groupdrop.SHAP_GROUPS).index("fire")
_ORIG_KEEP = sts_groupdrop.random_keep


def keep_with_fire_drop(p_fire: float):
    def keep(B: int, device) -> torch.Tensor:
        k = _ORIG_KEEP(B, device)
        k[:, FIRE_G] &= torch.rand(B, device=device) >= p_fire
        return k
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-fire", type=float, required=True, help="zusaetzliche Ausblend-Wahrscheinlichkeit der Feuergruppe")
    args, rest = ap.parse_known_args()
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", required=True)
    peek.add_argument("--tag", required=True)
    pk, _ = peek.parse_known_args(rest)
    sts_groupdrop.random_keep = keep_with_fire_drop(args.p_fire)
    probe = sts_groupdrop.random_keep(200_000, "cpu")[:, FIRE_G].float().mean().item()
    print(f"[sgf] Feuer-Dropout p_fire {args.p_fire}: Feuergruppe sichtbar in {probe:.3f} der Stichproben "
          f"(erwartet {0.5 * (1 - args.p_fire):.3f})", flush=True)
    sts_extreme.build = lambda *a, **kw: sts_groupdrop.wrap(sts_groupdrop._ORIG_BUILD(*a, **kw))
    sys.argv = [sys.argv[0]] + rest
    sts_extreme.main()
    cfg = load_config(pk.config)
    write_manifest(cfg, f"p53_sts_groupdrop_fire{pk.tag}", pk.config, {
        "groups": sts_groupdrop.SHAP_GROUPS, "mask_value": "0 (normiert)",
        "subset_sampling": "k ~ U{0..G}, Gruppen zufaellig, danach Feuer zusaetzlich mit p_fire aus",
        "p_fire": args.p_fire, "fire_visible_share": 0.5 * (1 - args.p_fire),
        "sts_extreme_args": rest, "sts_extreme_manifest": f"p35_sts_extreme{pk.tag}"})


if __name__ == "__main__":
    main()

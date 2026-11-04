"""sts_extreme mit Contagion-Kopf und Episoden-Brenndauer (contagion_episode.py); sts_extreme.py bleibt unveraendert.

Ersetzt nur sts_extreme.ContagionExt (build und load_any lesen die Klasse zur Laufzeit) und ruft dann
sts_extreme.main mit allen uebrigen Argumenten auf (--head contagion --cg-burn curve ..., --seeds, --tag).
  --ep-gap G   Luecke in Tagen, die eine Episode noch zusammenhaelt (Standard 2)
Checkpoints, Hazards und Manifest p35_sts_extreme<tag> wie dort; zusaetzlich Manifest p47_sts_episode<tag>.
Spaetere Auswertungen der Checkpoints immer ueber diese Huelle bzw. contagion_episode.episode_class.
"""

from __future__ import annotations

import argparse
import sys

from wildfire.audit.common import load_config, write_manifest
from wildfire.paper import sts_extreme
from wildfire.paper.contagion_episode import episode_class


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep-gap", type=int, default=2)
    own, rest = ap.parse_known_args()
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", required=True)
    peek.add_argument("--tag", required=True)
    peek.add_argument("--head", default="lh")
    pk, _ = peek.parse_known_args(rest)
    if pk.head != "contagion":
        raise SystemExit("ABBRUCH: sts_episode nur mit --head contagion")
    print(f"[sep] Brenndauer ab Episodenbeginn, Luecke <= {own.ep_gap} Tage", flush=True)
    sts_extreme.ContagionExt = episode_class(own.ep_gap)
    sys.argv = [sys.argv[0]] + rest
    sts_extreme.main()
    cfg = load_config(pk.config)
    write_manifest(cfg, f"p47_sts_episode{pk.tag}", pk.config, {
        "ep_gap": own.ep_gap, "sts_extreme_args": rest, "sts_extreme_manifest": f"p35_sts_extreme{pk.tag}"})


if __name__ == "__main__":
    main()

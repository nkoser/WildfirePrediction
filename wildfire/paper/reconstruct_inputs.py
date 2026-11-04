"""3D-U-Net-Rekonstruktion (reconstruct.py) mit zwei weiteren Kanalsaetzen fuer die Eingabe-Ablation.

reconstruct.py und channels.py bleiben unveraendert. Diese Huelle ergaenzt nur das Woerterbuch der
Kanalsaetze, das reconstruct liest, und ruft dann reconstruct.main mit denselben Argumenten auf:
  no_thermal  ohne Feuermaske, LST_day/night, BT_b20/21/31/32 (wie sts_inputs --inputs nothermal)
  fire_only   nur die Feuermaske (wie sts_inputs --inputs fireonly)
Alle uebrigen Saetze (paper55, no_fire, no_modis, weather) wie in channels.py.
"""

from __future__ import annotations

from wildfire.paper import reconstruct
from wildfire.paper.channels import CHANNELS, N_CH
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.sts_inputs import FIRE_THERMAL

EXTRA_SETS = {
    "no_thermal": [c for c in range(N_CH) if c not in FIRE_THERMAL],
    "fire_only": [FIRE_CH],
}

if __name__ == "__main__":
    reconstruct.CHANNELS = {**CHANNELS, **EXTRA_SETS}
    print("[rin] Kanalsaetze: " + ", ".join(f"{k} {len(v)}" for k, v in reconstruct.CHANNELS.items()), flush=True)
    reconstruct.main()

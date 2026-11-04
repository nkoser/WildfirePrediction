"""sts_inputs_target (Ziel D gegen D+1) mit allen bisherigen Eingabesaetzen plus weather.

sts_inputs_target.py bleibt unveraendert. Diese Huelle ergaenzt INPUTS um weather (sts_inputs_weather), die
Standardmodelle um lh_weather / cg_weather (slurm 212) und die U-Net-Tabelle um den Kanalsatz weather
(slurm 213), dann sts_inputs_target.main mit denselben Argumenten (eigener --tag, z. B. _weather).
"""

from __future__ import annotations

from wildfire.paper import sts_inputs, sts_inputs_target
from wildfire.paper.sts_inputs_weather import EXTRA_INPUTS

if __name__ == "__main__":
    sts_inputs.INPUTS.update(EXTRA_INPUTS)
    sts_inputs_target.DEFAULT_MODELS += ["lh_weather:weather", "cg_weather:weather"]
    sts_inputs_target.UNET["weather"] = {"cache": "p2_reconstruct_cache_G_cache_paperhp_weather",
                                         "next_patch": "p2_reconstruct_cache_G_cache_next_paperhp_w0_weather"}
    sts_inputs_target.main()

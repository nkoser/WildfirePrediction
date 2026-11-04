"""sts_inputs (STS-LH / Contagion mit maskierten Eingaben) mit dem zusaetzlichen Eingabesatz weather.

sts_inputs.py bleibt unveraendert. Diese Huelle ergaenzt nur das Woerterbuch INPUTS um
  weather   Kanalsatz weather aus channels.py (nur Wetter, Feuerwetter und VPD, 25 Kanaele; keine Feuermaske,
            kein MODIS, keine Topographie/Landbedeckung). Contagion: Feuerkanal maskiert -> Ausbreitungspfad leer.
und ruft sts_inputs.main mit denselben Argumenten auf (--inputs weather).
"""

from __future__ import annotations

from wildfire.paper import sts_inputs
from wildfire.paper.channels import keep_mask

EXTRA_INPUTS = {"weather": keep_mask("weather")}

if __name__ == "__main__":
    sts_inputs.INPUTS.update(EXTRA_INPUTS)
    sts_inputs.main()

"""Das Label-Fenster muss vollstaendig im Zuendungsindex liegen.

Landmarks liegen in der Saison (bis 31.08.), ihr Ereignisfenster reicht aber
H Tage weiter. Fehlen Zuendungen aus diesem Fenster im Index, werden sie als
"ueberlebt" gelabelt - kein Rauschen, sondern ein systematisch zu niedriger
Hazard am Fensterende. Genau das passierte, solange nfdb_events nur Meldungen
bis zum 30.09. einlas.
"""

from __future__ import annotations

import datetime as dt

from wildfire.data.nfdb_events import build_parser

LANDMARK_SEASON_END = "08-31"
HORIZON = 100


def test_nfdb_index_covers_every_landmark_window():
    args = build_parser().parse_args(["--config", "x"])
    last_landmark = dt.date(2001, *map(int, LANDMARK_SEASON_END.split("-")))
    window_end = last_landmark + dt.timedelta(days=HORIZON)
    assert window_end.year == last_landmark.year, "Fenster ueber den Jahreswechsel"
    assert args.season_end >= window_end.isoformat()[5:], (
        f"NFDB-Index endet {args.season_end}, Label-Fenster reicht bis "
        f"{window_end.isoformat()[5:]}")

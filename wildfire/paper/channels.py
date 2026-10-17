"""Kanalsaetze des Paper-Protokolls, fuer Laeufe mit und ohne Feuer-Historie.

  paper55   alle 55 Cache-Kanaele (Kanal 0 = MODIS-Feuermaske)
  no_fire   54 Kanaele, Feuermaske entfernt (README: "not included as a model
            input channel")
  no_modis  zusaetzlich ohne jede MODIS-Beobachtung (Vegetation, Reflexion,
            LST, Helligkeitstemperatur): Wetter, Feuerwetter, VPD,
            Infrastruktur, Gelaende, Landbedeckung. Die Feuerinformation
            steckt sonst weiter in den Baendern 20/21 und in LST.
  weather   nur Wetter, Feuerwetter und VPD

Labels und Ereignisse kommen immer aus der Feuermaske; der Kanalsatz betrifft
nur die Eingabe.

Zwei Wege, einen Kanal aus der Eingabe zu nehmen:
  - Merkmale: die Spalten des Kanals fallen weg (summary_columns).
  - Netze mit fester Eingabebreite: masked_std setzt die Standardabweichung
    des Kanals auf unendlich. Nach (x - mean) / std ist er dann ueberall exakt
    0 und traegt keine Information mehr; Architektur und Batch-Code bleiben
    unveraendert.
"""

from __future__ import annotations

import numpy as np

from wildfire.paper.prepare import FIRE_CH

N_CH = 55
CHANNELS = {
    "paper55": list(range(N_CH)),
    "no_fire": [c for c in range(N_CH) if c != FIRE_CH],
    "no_modis": [c for c in range(N_CH) if c != FIRE_CH and not 38 <= c <= 53],
    "weather": list(range(1, 26)),
}


def keep_mask(cset: str) -> np.ndarray:
    m = np.zeros(N_CH, bool)
    m[CHANNELS[cset]] = True
    return m


def masked_std(std: np.ndarray, cset: str) -> np.ndarray:
    """std mit np.inf fuer jeden Kanal ausserhalb des Satzes."""
    return np.where(keep_mask(cset), std, np.inf).astype(std.dtype)


def summary_columns(cset: str, n_stats: int = 5) -> np.ndarray:
    """Spalten der Zusammenfassungen (n_stats Bloecke zu je 55 Kanaelen), die zum Satz gehoeren."""
    return np.concatenate([k * N_CH + np.asarray(CHANNELS[cset]) for k in range(n_stats)])


def has_fire(cset: str) -> bool:
    return FIRE_CH in CHANNELS[cset]


def tag(cset: str) -> str:
    """Suffix fuer Manifeste und Dateien; der volle Satz behaelt die alten Namen."""
    return "" if cset == "paper55" else f"_{cset}"

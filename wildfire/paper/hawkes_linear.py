"""Zeitdiskreter Hawkes-Prozess nach Xu et al. (arXiv:2207.13250, LinearSTHawkes), an unser Raster angepasst.

Xu et al.: Intensitaet lambda(t, k) = (mu_k + sum_{t_j < t} alpha_{u_j,k} beta exp(-beta (t - t_j))) * gamma^T m,
also Grundrate aus linearen Kovariaten, Anregung durch fruehere Feuer mit einem Koeffizienten je Zellpaar und EINER
globalen exponentiellen Abklingrate beta; die Anregung haengt nicht von den Kovariaten ab.

Hier als eingeschraenkte lineare Contagion (contagion_linear): derselbe lineare Encoder fuer die Zuendung (Grundrate,
linear in den Merkmalen auf der Logit-Skala), aber
  Kern        stationaer: k_ij haengt nur vom Versatz j - i ab (ker.weight = 0 und eingefroren, nur ker.bias lernt);
              Xu et al. haben einen Koeffizienten je Zellpaar fuer 36 feste Zellen - fuer zufaellige Ausschnitte in
              ganz Kanada ist die translationsinvariante Form die Entsprechung
  Abklingen   geometrisch rho^Alter = exp(-beta Alter) mit einer globalen Rate (dur.weight = 0 und eingefroren)
  Verknuepfung  1 - h = (1 - a) prod (1 - k b) wie Contagion statt einer additiven Intensitaet; fuer kleine Raten
              gleich (1 - h ~ exp(-mu - sum alpha g)), und so laufen Rollout, Monte Carlo und Kalibrierung unveraendert
Damit misst der Vergleich mit der linearen Contagion, was kovariatenabhaengige Ausbreitung und Brenndauer bringen.

Die eingefrorenen Gewichte sind Null; der Checkpoint laedt deshalb ueber contagion_linear wie eine lineare Contagion
(identische Vorhersage). Auswertungen: python -m wildfire.paper.contagion_linear <modul> ...

  python -m wildfire.paper.hawkes_linear train <Argumente von sts_dctm, mit --head contagion --cg-burn geometric>
"""

from __future__ import annotations

import sys

from wildfire.paper import contagion_linear, sts_dctm


def build_hawkes(head: str = "dctm", **kw):
    if kw.get("burn", "geometric") != "geometric":
        raise SystemExit("ABBRUCH: Hawkes nur mit --cg-burn geometric (exponentielles Abklingen)")
    net = contagion_linear.build_linear(head, **kw)
    for lin in (net.ker, net.dur):
        lin.weight.data.zero_()
        lin.weight.requires_grad_(False)
    return net


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "train":
        raise SystemExit(__doc__)
    sts_dctm.build_sts = build_hawkes
    sys.argv = ["sts_dctm"] + sys.argv[2:]
    sts_dctm.main()


if __name__ == "__main__":
    main()

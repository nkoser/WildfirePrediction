"""STS-LH mit Zusatzverlust fuer das Ranking an festen Horizonten: Intervall-NLL + beta * sum_tau BCE(F(tau), 1{T <= tau}).

Gegenversuch zur Luecke bei kurzen Horizonten (horizon_landmarks: der Stapel getrennter Einzelklassifikatoren rankt
bei natuerlicher Praevalenz bis tau ~ 14 besser als F(tau) von STS-LH). Der Zusatzterm wirkt auf dieselbe Kurve
F(tau) = 1 - S(tau) des LH-Kopfs, die Monotonie bleibt also per Konstruktion erhalten.

  Verlust je Kopf (Pixel und Ausschnitt, wie in sts_dctm mit --patch-weight gewichtet):
      head_loss (Intervall-NLL, unveraendert) + beta * sum_{tau in --hz-taus} mean_i BCE_i(tau)
  BCE_i(tau) = -[y log F(tau) + (1 - y) log S(tau)], y = 1{Ereignis und T <= tau}; Stichtage, die vor tau zensiert
  sind (kein Ereignis, T < tau), zaehlen an diesem tau nicht (Maske; bei administrativer Zensur bei 100 Tagen tritt
  das fuer tau <= 30 nicht auf). Mittel je tau ueber ALLE Zeilen des Batches (maskierte zaehlen als 0), damit der
  Term dieselbe Skala wie die mittlere NLL hat.

Bestehende Module bleiben unveraendert; der Wrapper ersetzt sts_dctm.head_loss (nur die Trainingsschleife ruft es
ueber diesen Namen auf) und startet dann sts_dctm.main(). Die Checkpoints sind gewoehnliche LH-Checkpoints und laden
ohne Wrapper.
  python -m wildfire.paper.horizon_bce train --hz-beta 0.3 [--hz-taus 1 7 14 30] <Argumente von sts_dctm>
"""

from __future__ import annotations

import argparse
import sys

import torch

from wildfire.paper import sts_dctm

_ORIG_LOSS = sts_dctm.head_loss
CFG = {"beta": 0.0, "taus": (1, 7, 14, 30)}


def horizon_bce(head, raw: torch.Tensor, event: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    raw = raw.reshape(-1, raw.shape[-1])
    e, tt = event.reshape(-1).bool(), t.reshape(-1).double()
    taus = torch.tensor(CFG["taus"], device=raw.device, dtype=torch.float64)
    ls = head.log_survival(raw, taus)                                        # (n, K) log S(tau)
    log_f = torch.log((-torch.expm1(ls)).clamp_min(1e-300))                  # log F(tau)
    y = e[:, None] & (tt[:, None] <= taus[None])
    known = e[:, None] | (tt[:, None] >= taus[None])
    bce = -torch.where(y, log_f, ls)
    return (bce * known).mean(0).sum()


def head_loss(head, raw, event, t):
    base = _ORIG_LOSS(head, raw, event, t)
    if CFG["beta"] <= 0:
        return base
    return base + CFG["beta"] * horizon_bce(head, raw, event, t)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "train":
        raise SystemExit(__doc__)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--hz-beta", type=float, required=True)
    ap.add_argument("--hz-taus", type=int, nargs="+", default=list(CFG["taus"]))
    own, rest = ap.parse_known_args(sys.argv[2:])
    if "--head" in rest and rest[rest.index("--head") + 1] != "lh":
        raise SystemExit("ABBRUCH: horizon_bce nur mit --head lh")
    CFG.update(beta=own.hz_beta, taus=tuple(sorted(own.hz_taus)))
    print(f"[hzbce] Zusatzverlust beta {CFG['beta']} an tau {CFG['taus']} (Pixel und Ausschnitt)", flush=True)
    sts_dctm.head_loss = head_loss
    sys.argv = ["sts_dctm"] + rest
    sts_dctm.main()


if __name__ == "__main__":
    main()

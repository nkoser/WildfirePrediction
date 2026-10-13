"""PyTorch-Anbindung des Landmark-Datensatzes.

Wandelt die Ausgabe von ``LandmarkDataset`` in das um, was ein Faltungsnetz
erwartet, und wendet dabei die trainingsseitige Normalisierung an.

**Zeitscheiben statt voller Historie.** B2 ist die RAEUMLICHE Baseline. Sie
bekommt nicht alle 30 Tage, sondern K aequidistante Scheiben, als Kanaele
gestapelt: bei K=5 die Tage t-28, t-21, t-14, t-7 und t. Das ist die
"kurze gestapelte Historie" des Arbeitsplans. Die volle Zeitachse bleibt
B3 vorbehalten - nur so misst der Vergleich zwischen beiden den Beitrag der
zeitlichen Modellierung und nicht einen Nebeneffekt.

**Lesereihenfolge.** Ein Patch kostet 5 Zeitscheiben aus einem 340-GB-Speicher.
Gemessen auf dem Wuerfel, ein Prozess:

    zufaellig               42,5 Beispiele/s
    nach Tag sortiert      159,3 Beispiele/s     3,7x
    chunked shuffle        220,8 Beispiele/s     5,2x

Der Grund ist nicht die Bandbreite, sondern das Suchen: Bei ``shuffle=True``
liegt jeder Zugriff woanders im Wuerfel, und der Seitencache des Systems wird
nie ein zweites Mal getroffen. Es fehlt also kein Cache - es fehlt Lokalitaet.
``ChunkedShuffleSampler`` stellt sie her, ohne die Durchmischung praktisch
aufzugeben.

**Beobachtungsmaske als Kanal.** Fuer die gesperrten Kanaele ist der Wert
an den letzten C Tagen null UND unbeobachtet. Eine Null allein waere ein
gueltiger Messwert. Deshalb wird je Zeitscheibe ein Maskenkanal angehaengt,
der sagt, ob die gesperrten Kanaele dort ueberhaupt etwas zeigen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from wildfire.data.dataset import LOCKED_CHANNELS, LandmarkDataset


def slice_days_back(lookback: int, n_slices: int) -> list[int]:
    """Rueckblick der Zeitscheiben in Tagen vor t, AELTESTE ZUERST.

    Bei L=30, K=5: [28, 21, 14, 7, 0]. Training (PatchDataset) und Inferenz
    (predict_map) muessen dieselbe Reihenfolge benutzen, weil die Scheiben als
    Kanaele gestapelt werden - eine verdrehte Reihenfolge ist fuer das Netz
    ein anderer Input, ohne dass irgendein Shape-Check anschlaegt. Frueher
    rechnete predict_map seine Scheiben selbst und stapelte sie neueste zuerst.
    """
    step = max(1, (lookback - 1) // max(1, n_slices - 1))
    return sorted({k * step for k in range(n_slices)}, reverse=True)


class ChunkedShuffleSampler(Sampler):
    """Mischt in Bloecken statt global - fuer Lokalitaet beim Lesen.

    Vollstaendiges Mischen zerstreut jeden Zugriff ueber den ganzen Wuerfel.
    Vollstaendige Sortierung waere schnell, wuerde aber aufeinanderfolgende
    Batches aus demselben Tag ziehen und damit die Gradienten korrelieren.

    Der Kompromiss: nach Tag sortieren, in Bloecke von ``chunk`` Landmarks
    schneiden, die BLOCKREIHENFOLGE mischen und INNERHALB jedes Blocks noch
    einmal mischen. Ein Block von 20 000 Landmarks umspannt rund 30 Kalendertage,
    ein Batch daraus ist also weiterhin zeitlich gestreut - aber alle Lesezugriffe
    eines Blocks fallen in einen engen Bereich des Wuerfels und treffen den
    Seitencache.

    Jede Epoche mischt anders (``set_epoch`` bzw. automatisches Hochzaehlen).
    """

    def __init__(self, order, chunk: int = 20000, seed: int = 0) -> None:
        self.order = np.asarray(order, dtype=np.int64)
        self.chunk = int(chunk)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.order.size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        starts = np.arange(0, self.order.size, self.chunk)
        rng.shuffle(starts)
        for s0 in starts:
            block = self.order[s0:s0 + self.chunk].copy()
            rng.shuffle(block)
            yield from block.tolist()
        self.epoch += 1


class PatchDataset(Dataset):
    def __init__(self, exp: Path, split: str, part: str, *, patch: int = 33,
                 lookback: int = 30, cooldown: int = 7, n_slices: int = 5,
                 limit: int | None = None, seed: int = 0) -> None:
        self.base = LandmarkDataset(exp, split, part, patch=patch,
                                    lookback=lookback, cooldown=cooldown)
        self.n_slices = n_slices
        # aequidistante Scheiben, aelteste zuerst, letzte ist immer t;
        # offsets sind Indizes in die L-Achse von LandmarkDataset (L-1 == t)
        self.offsets = [lookback - 1 - d for d in slice_days_back(lookback, n_slices)]
        z = np.load(Path(exp) / "derived_data" / "norm_stats.npz")
        self.dyn_mean = z["dyn_mean"][None, :, None, None]
        self.dyn_std = z["dyn_std"][None, :, None, None]
        self.stat_mean = z["stat_mean"][:, None, None]
        self.stat_std = z["stat_std"][:, None, None]
        self.locked = np.array(sorted(LOCKED_CHANNELS))

        n = len(self.base)
        if limit is not None and limit < n:
            rng = np.random.default_rng(seed)
            self.subset = np.sort(rng.choice(n, limit, replace=False))
        else:
            self.subset = np.arange(n)

    def day_order(self) -> np.ndarray:
        """Eigene Indizes, nach Kalendertag des Landmarks sortiert.

        Stabil sortiert, damit bei gleichem Tag die Reihenfolge im Wuerfel
        erhalten bleibt - auch das ist Lokalitaet.
        """
        days = self.base.t[self.base.usable[self.subset]] \
            if self.base.usable is not None else self.base.t[self.subset]
        return np.argsort(days, kind="mergesort").astype(np.int64)

    @property
    def in_channels(self) -> int:
        return self.n_slices * (self.base.cube.n_dyn + 1) + self.base.cube.n_stat

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, i: int):
        s = self.base[int(self.subset[i])]
        x = s["x_dyn"].astype(np.float32)                 # (C, L, P, P)
        sl = x[:, self.offsets]                           # (C, K, P, P)
        sl = np.moveaxis(sl, 1, 0)                        # (K, C, P, P)
        sl = (sl - self.dyn_mean) / self.dyn_std
        obs = s["obs"][:, self.offsets]                   # (C, K)
        # ein Maskenkanal je Scheibe: sind die gesperrten Kanaele sichtbar?
        vis = obs[self.locked].min(axis=0).astype(np.float32)   # (K,)
        mask = np.broadcast_to(vis[:, None, None, None],
                               (len(self.offsets), 1) + sl.shape[-2:])
        sl = np.concatenate([sl, mask], axis=1)
        flat = sl.reshape(-1, *sl.shape[-2:])             # (K*(C+1), P, P)

        st = (s["x_stat"].astype(np.float32) - self.stat_mean) / self.stat_std
        return {
            "x": torch.from_numpy(np.ascontiguousarray(
                np.concatenate([flat, st], axis=0))),
            "event": torch.tensor(s["event"], dtype=torch.float32),
            "t_event": torch.tensor(s["t_event"], dtype=torch.long),
            "weight": torch.tensor(s["weight"], dtype=torch.float32),
            "valid": torch.from_numpy(s["valid"].astype(np.float32)),
        }

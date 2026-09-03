"""PyTorch-Anbindung des Landmark-Datensatzes.

Wandelt die Ausgabe von ``LandmarkDataset`` in das um, was ein Faltungsnetz
erwartet, und wendet dabei die trainingsseitige Normalisierung an.

**Zeitscheiben statt voller Historie.** B2 ist die RAEUMLICHE Baseline. Sie
bekommt nicht alle 30 Tage, sondern K aequidistante Scheiben, als Kanaele
gestapelt: bei K=5 die Tage t-28, t-21, t-14, t-7 und t. Das ist die
"kurze gestapelte Historie" des Arbeitsplans. Die volle Zeitachse bleibt
B3 vorbehalten - nur so misst der Vergleich zwischen beiden den Beitrag der
zeitlichen Modellierung und nicht einen Nebeneffekt.

**Beobachtungsmaske als Kanal.** Fuer die gesperrten Kanaele ist der Wert
an den letzten C Tagen null UND unbeobachtet. Eine Null allein waere ein
gueltiger Messwert. Deshalb wird je Zeitscheibe ein Maskenkanal angehaengt,
der sagt, ob die gesperrten Kanaele dort ueberhaupt etwas zeigen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from wildfire.data.dataset import LOCKED_CHANNELS, LandmarkDataset


class PatchDataset(Dataset):
    def __init__(self, exp: Path, split: str, part: str, *, patch: int = 33,
                 lookback: int = 30, cooldown: int = 7, n_slices: int = 5,
                 limit: int | None = None, seed: int = 0) -> None:
        self.base = LandmarkDataset(exp, split, part, patch=patch,
                                    lookback=lookback, cooldown=cooldown)
        self.n_slices = n_slices
        # aequidistante Scheiben, letzte ist immer t
        step = max(1, (lookback - 1) // max(1, n_slices - 1))
        self.offsets = sorted({lookback - 1 - k * step for k in range(n_slices)})
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

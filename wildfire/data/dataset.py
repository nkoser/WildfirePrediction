"""Zusammenbau eines Trainingssamples aus Wuerfel und Landmark-Tabelle.

Ein Sample zu einem Landmark (t, r, c):

    x_dyn   (C_dyn, L, P, P)   Historie t-L+1 .. t
    x_stat  (C_stat, P, P)     statische Kanaele
    obs     (C_dyn, L)         1 = Kanal an diesem Zeitschritt sichtbar
    event   0/1                Zuendung im Fenster t+1 .. t+H beobachtet
    t_event 1..H               Tage bis zum Ereignis bzw. bis zur Zensierung
    weight  float              Inverses-Wahrscheinlichkeits-Gewicht

**Kanalsperre.** Die Feuermaske und die vier MODIS-Helligkeitstemperaturen
sind Rohmaterial des Labels (Audit-Stufen 6 und 2c). Sie werden nur bis
``t - C`` gezeigt; die letzten C Zeitschritte werden auf null gesetzt UND
in ``obs`` als unbeobachtet markiert. Beides ist noetig: die Null allein
waere ein gueltiger Messwert und damit nicht von echter Abwesenheit zu
unterscheiden.

**Raender.** Patches am Kartenrand werden mit null aufgefuellt; eine
zusaetzliche Gueltigkeitsmaske haelt fest, welche Pixel echt sind.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

LOCKED_CHANNELS = (0, 38, 39, 40, 41)


class CubeReader:
    """Lazy-Zugriff auf die Jahres-memmaps des Wuerfels."""

    def __init__(self, exp: Path) -> None:
        man = json.loads((exp / "data_manifests" / "d4_cube.json").read_text())
        self.per_year = man["per_year"]
        self.n_dyn = man["n_dynamic_channels"]
        self.n_stat = man["n_static_channels"]
        self.H = man["grid"]["height"]
        self.W = man["grid"]["width"]
        # globaler Tagesindex -> (Jahr, lokaler Index)
        idx = np.load(exp / "derived_data" / "events" / "fire_index.npz", allow_pickle=False)
        self.dates = [str(d) for d in idx["dates"]]
        self._pos: dict[int, tuple[str, int]] = {}
        for year, info in self.per_year.items():
            local = {d: i for i, d in enumerate(info["days"])}
            for gt, d in enumerate(self.dates):
                if d in local and d[:4] == str(year):
                    self._pos[gt] = (str(year), local[d])
        self._open: dict[str, tuple[np.memmap, np.ndarray]] = {}

    def available_days(self) -> set[int]:
        return set(self._pos)

    def _arrays(self, year: str):
        if year not in self._open:
            info = self.per_year[year]
            dyn = np.memmap(info["dyn_file"], dtype=np.uint8, mode="r",
                            shape=tuple(info["dyn_shape"]))
            stat = np.load(info["stat_file"], mmap_mode="r")
            self._open[year] = (dyn, stat)
        return self._open[year]

    def patch(self, t: int, r: int, c: int, half: int):
        """(H_patch, W_patch, C_dyn) fuer einen Tag, randsicher genullt."""
        year, li = self._pos[t]
        dyn, _ = self._arrays(year)
        P = 2 * half + 1
        out = np.zeros((P, P, dyn.shape[3]), np.uint8)
        r0, r1 = max(0, r - half), min(self.H, r + half + 1)
        c0, c1 = max(0, c - half), min(self.W, c + half + 1)
        if r1 > r0 and c1 > c0:
            out[r0 - (r - half):r1 - (r - half), c0 - (c - half):c1 - (c - half)] = \
                dyn[li, r0:r1, c0:c1, :]
        return out

    def static_patch(self, t: int, r: int, c: int, half: int):
        year, _ = self._pos[t]
        _, stat = self._arrays(year)
        P = 2 * half + 1
        out = np.zeros((P, P, stat.shape[2]), np.uint8)
        r0, r1 = max(0, r - half), min(self.H, r + half + 1)
        c0, c1 = max(0, c - half), min(self.W, c + half + 1)
        if r1 > r0 and c1 > c0:
            out[r0 - (r - half):r1 - (r - half), c0 - (c - half):c1 - (c - half)] = \
                stat[r0:r1, c0:c1, :]
        return out

    def valid_patch(self, r: int, c: int, half: int, valid: np.ndarray):
        P = 2 * half + 1
        out = np.zeros((P, P), bool)
        r0, r1 = max(0, r - half), min(self.H, r + half + 1)
        c0, c1 = max(0, c - half), min(self.W, c + half + 1)
        if r1 > r0 and c1 > c0:
            out[r0 - (r - half):r1 - (r - half), c0 - (c - half):c1 - (c - half)] = \
                valid[r0:r1, c0:c1]
        return out


class LandmarkDataset:
    def __init__(self, exp: Path, split: str, part: str, patch: int = 33,
                 lookback: int = 30, cooldown: int = 7,
                 locked: tuple[int, ...] = LOCKED_CHANNELS) -> None:
        self.exp = Path(exp)
        self.half = patch // 2
        self.L = lookback
        self.C = cooldown
        self.locked = tuple(locked)
        z = np.load(self.exp / "derived_data" / "landmarks" / f"{split}_{part}.npz")
        self.t = z["t"]; self.row = z["row"]; self.col = z["col"]
        self.event = z["event"]; self.t_event = z["t_event"]
        self.event_id = z["event_id"]; self.weight = z["weight"]
        self.year = z["year"]
        self.cube = CubeReader(self.exp)
        self.valid = np.load(self.exp / "derived_data" / "masks" / "valid.npy")
        # Nur Landmarks, deren gesamtes Historienfenster im Wuerfel liegt.
        # Vektorisiert ueber eine Verfuegbarkeitsmaske statt einer Schleife
        # mit Mengenabfragen - bei Millionen Zeilen ist das der Unterschied
        # zwischen Sekundenbruchteilen und Minuten.
        n_days = len(self.cube.dates)
        avail = np.zeros(n_days, bool)
        avail[np.fromiter(self.cube.available_days(), np.int64)] = True
        ok = np.ones(len(self.t), bool)
        for k in range(self.L):
            d = self.t.astype(np.int64) - k
            ok &= (d >= 0) & avail[np.clip(d, 0, n_days - 1)]
        self.usable = np.nonzero(ok)[0]

    def __len__(self) -> int:
        return len(self.t) if self.usable is None else len(self.usable)

    def _map(self, i: int) -> int:
        return int(i if self.usable is None else self.usable[i])

    def __getitem__(self, i: int) -> dict:
        j = self._map(i)
        t = int(self.t[j]); r = int(self.row[j]); c = int(self.col[j])
        days = [t - k for k in range(self.L - 1, -1, -1)]     # t-L+1 .. t
        frames = [self.cube.patch(d, r, c, self.half) for d in days]
        x = np.stack(frames, axis=0)                          # (L, P, P, C)
        obs = np.ones((x.shape[3], self.L), np.uint8)

        # Kanalsperre: nur bis t - C sichtbar
        cutoff = t - self.C
        for k, d in enumerate(days):
            if d > cutoff:
                for ch in self.locked:
                    x[k, :, :, ch] = 0
                    obs[ch, k] = 0

        stat = self.cube.static_patch(t, r, c, self.half)
        return {
            "x_dyn": np.ascontiguousarray(x.transpose(3, 0, 1, 2)),   # (C, L, P, P)
            "x_stat": np.ascontiguousarray(stat.transpose(2, 0, 1)),  # (C_stat, P, P)
            "obs": obs,
            "valid": self.cube.valid_patch(r, c, self.half, self.valid),
            "event": int(self.event[j]),
            "t_event": int(self.t_event[j]),
            "weight": float(self.weight[j]),
            "event_id": int(self.event_id[j]),
            "t": t, "row": r, "col": c, "year": int(self.year[j]),
            "date": self.cube.dates[t],
        }

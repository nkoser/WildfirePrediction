"""Prueft den Sample-Zusammenbau am kuenstlichen Miniaturwuerfel.

Getestet wird der ECHTE Code-Pfad aus wildfire.data.dataset, nicht eine
Nachbildung - sonst wuerde der Test nur sich selbst bestaetigen. Dafuer
wird ein vollstaendiges, winziges Experimentverzeichnis erzeugt, in dem
jeder Wert bekannt ist: der Wuerfel enthaelt an Position (Tag, Zeile,
Spalte, Kanal) den Wert ``Tag`` selbst. Damit ist jede Verschiebung in der
Zeitachse sofort sichtbar.

Der Kern ist die Kanalsperre: Feuermaske und die vier MODIS-Helligkeits-
temperaturen sind Rohmaterial des Labels und duerfen nur bis ``t - C``
sichtbar sein.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wildfire.data.dataset import LOCKED_CHANNELS, LandmarkDataset

H, W, C_DYN, C_STAT = 20, 30, 42, 13
YEAR, N_DAYS = 2023, 60
PATCH, LOOKBACK, COOLDOWN = 5, 10, 3


@pytest.fixture(scope="module")
def exp(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("exp")
    (root / "data_manifests").mkdir()
    (root / "derived_data" / "masks").mkdir(parents=True)
    (root / "derived_data" / "events").mkdir(parents=True)
    (root / "derived_data" / "landmarks").mkdir(parents=True)
    cube = root / "derived_data" / "cube"
    cube.mkdir()

    days = [f"{YEAR}-{1 + d // 31:02d}-{1 + d % 31:02d}" for d in range(N_DAYS)]
    dyn_path = cube / f"dyn_{YEAR}.mmap"
    shape = (N_DAYS, H, W, C_DYN)
    arr = np.memmap(dyn_path, np.uint8, "w+", shape=shape)
    for t in range(N_DAYS):
        arr[t] = np.uint8(t % 251 + 1)      # Wert == Tag, nie 0
    arr.flush(); del arr
    np.lib.format.open_memmap(cube / f"stat_{YEAR}.npy", mode="w+",
                              dtype=np.uint8, shape=(H, W, C_STAT))[:] = 7

    (root / "data_manifests" / "d4_cube.json").write_text(json.dumps({
        "per_year": {str(YEAR): {"days": days, "T": N_DAYS,
                                 "dyn_file": str(dyn_path),
                                 "stat_file": str(cube / f"stat_{YEAR}.npy"),
                                 "dyn_shape": list(shape)}},
        "n_dynamic_channels": C_DYN, "n_static_channels": C_STAT,
        "grid": {"height": H, "width": W},
    }))
    np.savez(root / "derived_data" / "events" / "fire_index.npz",
             grid=np.array([H, W], np.int32), dates=np.array(days),
             fire_pixels=np.zeros(0, np.int64), fire_offsets=np.zeros(1, np.int64),
             fire_days=np.zeros(0, np.int32), ign_pixels=np.zeros(0, np.int64),
             ign_offsets=np.zeros(1, np.int64), ign_days=np.zeros(0, np.int32),
             ign_events=np.zeros(0, np.int32))
    np.save(root / "derived_data" / "masks" / "valid.npy", np.ones((H, W), bool))
    # Kanalweise verschiedene Statistik, damit eine Kanal- oder Scheiben-
    # vertauschung nach der Normalisierung nicht zufaellig gleich aussieht.
    np.savez(root / "derived_data" / "norm_stats.npz",
             dyn_mean=np.arange(C_DYN, dtype=np.float32) * 0.5,
             dyn_std=1.0 + np.arange(C_DYN, dtype=np.float32) * 0.1,
             stat_mean=np.full(C_STAT, 2.0, np.float32),
             stat_std=np.full(C_STAT, 3.0, np.float32))

    t0 = 40
    np.savez(root / "derived_data" / "landmarks" / "temporal_train.npz",
             t=np.array([t0, t0], np.int32), row=np.array([10, 0], np.int16),
             col=np.array([15, 0], np.int16), event=np.array([1, 0], np.int8),
             t_event=np.array([5, 30], np.int16),
             event_id=np.array([42, -1], np.int32),
             weight=np.array([1.0, 178.0], np.float32),
             year=np.array([YEAR, YEAR], np.int16))
    return root


@pytest.fixture(scope="module")
def ds(exp):
    return LandmarkDataset(exp, "temporal", "train", patch=PATCH,
                           lookback=LOOKBACK, cooldown=COOLDOWN)


def test_shapes(ds):
    s = ds[0]
    assert s["x_dyn"].shape == (C_DYN, LOOKBACK, PATCH, PATCH)
    assert s["x_stat"].shape == (C_STAT, PATCH, PATCH)
    assert s["obs"].shape == (C_DYN, LOOKBACK)


def test_time_axis_is_ordered_and_ends_at_t(ds):
    """Letzter Zeitschritt muss der Landmarktag sein, nicht t+1 oder t-1."""
    s = ds[0]
    open_ch = next(c for c in range(C_DYN) if c not in LOCKED_CHANNELS)
    series = s["x_dyn"][open_ch, :, PATCH // 2, PATCH // 2]
    t = s["t"]
    expected = [(t - k) % 251 + 1 for k in range(LOOKBACK - 1, -1, -1)]
    assert list(series) == expected, "Zeitachse verschoben oder verdreht"


def test_locked_channels_are_blanked_in_cooldown(ds):
    s = ds[0]
    for ch in LOCKED_CHANNELS:
        vals = s["x_dyn"][ch, :, PATCH // 2, PATCH // 2]
        obs = s["obs"][ch]
        # letzte COOLDOWN Schritte: gesperrt
        assert np.all(vals[-COOLDOWN:] == 0), f"Kanal {ch} nicht genullt"
        assert np.all(obs[-COOLDOWN:] == 0), f"Kanal {ch} nicht als unbeobachtet markiert"
        # davor: unveraendert sichtbar
        assert np.all(vals[:-COOLDOWN] > 0), f"Kanal {ch} zu frueh gesperrt"
        assert np.all(obs[:-COOLDOWN] == 1)


def test_open_channels_are_untouched(ds):
    s = ds[0]
    for ch in range(C_DYN):
        if ch in LOCKED_CHANNELS:
            continue
        assert np.all(s["obs"][ch] == 1), f"Kanal {ch} faelschlich gesperrt"
        assert np.all(s["x_dyn"][ch] > 0), f"Kanal {ch} faelschlich genullt"


def test_locked_set_matches_audit_findings():
    """ch0 ist das Label, ch38-41 sind die MODIS-Feuerbaender."""
    assert set(LOCKED_CHANNELS) == {0, 38, 39, 40, 41}


def test_border_patch_is_zero_padded_and_marked(ds):
    """Landmark bei (0,0): drei Viertel des Patches liegen ausserhalb."""
    s = ds[1]
    assert s["row"] == 0 and s["col"] == 0
    half = PATCH // 2
    assert not s["valid"][:half, :].any(), "Rand nicht als ungueltig markiert"
    assert s["valid"][half:, half:].all(), "Innenbereich faelschlich ungueltig"
    open_ch = next(c for c in range(C_DYN) if c not in LOCKED_CHANNELS)
    assert np.all(s["x_dyn"][open_ch, :, :half, :] == 0), "Rand nicht genullt"


def test_labels_and_weights_pass_through(ds):
    a, b = ds[0], ds[1]
    assert (a["event"], a["t_event"], a["event_id"]) == (1, 5, 42)
    assert (b["event"], b["t_event"], b["event_id"]) == (0, 30, -1)
    assert a["weight"] == 1.0 and b["weight"] == 178.0


def test_map_inference_builds_the_training_input(exp):
    """predict_map.build_input muss am Landmark exakt den Trainingstensor liefern.

    Faengt den Fehler, bei dem die Karte die Zeitscheiben neueste-zuerst
    stapelte, waehrend das Training aelteste-zuerst lernte: Shapes stimmen
    in beiden Faellen, nur die Werte nicht.
    """
    from wildfire.models.torch_data import PatchDataset, slice_days_back
    from wildfire.predict_map import build_input
    from wildfire.data.dataset import CubeReader

    n_slices = 4
    pd = PatchDataset(exp, "temporal", "train", patch=PATCH, lookback=LOOKBACK,
                      cooldown=COOLDOWN, n_slices=n_slices)
    x_train = pd[0]["x"].numpy()
    s = pd.base[0]
    half = PATCH // 2
    r0, c0 = int(s["row"]) - half, int(s["col"]) - half
    z = np.load(exp / "derived_data" / "norm_stats.npz")
    norm = (z["dyn_mean"], z["dyn_std"], z["stat_mean"], z["stat_std"])
    x_map = build_input(CubeReader(exp), int(s["t"]), r0, r0 + PATCH, c0, c0 + PATCH,
                        slice_days_back(LOOKBACK, n_slices), COOLDOWN, norm)
    assert x_map.shape == x_train.shape
    np.testing.assert_allclose(x_map, x_train, rtol=1e-6, atol=1e-6)


def test_slices_are_oldest_first_and_end_at_t():
    from wildfire.models.torch_data import slice_days_back
    assert slice_days_back(30, 5) == [28, 21, 14, 7, 0]

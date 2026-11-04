import numpy as np
import torch

from wildfire.paper.contagion import ContagionSTS
from wildfire.paper.risk_map import paste, predict, tile_centres
from wildfire.paper.risk_map_plot import capture


def test_tiles_cover_mask_once():
    rng = np.random.default_rng(0)
    mask = rng.random((40, 61)) < 0.3
    for crop in (1, 5, 7, 13):
        cen = tile_centres(mask, crop)
        cnt = np.zeros(mask.shape, int)
        h = crop // 2
        for r, c in cen:
            cnt[max(0, r - h):r + h + 1, max(0, c - h):c + h + 1] += 1
        assert cnt.max() == 1 and np.all(cnt[mask] == 1)
        v = paste(np.ones((len(cen), crop, crop, 2), np.float32), cen, crop, mask.shape)
        assert v.shape == mask.shape + (2,) and np.all(v[mask] == 1)


def test_overlapping_tiles_blend():
    rng = np.random.default_rng(2)
    mask = rng.random((40, 61)) < 0.3
    cen = tile_centres(mask, 9, 3)
    near = np.zeros(mask.shape, bool)
    for r, c in cen:
        near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True          # jede Maske hat eine Mitte im Abstand <= 1
    assert np.all(near[mask])
    v = paste(np.full((len(cen), 9, 9, 1), 0.25, np.float32), cen, 9, mask.shape)
    assert np.allclose(v[mask], 0.25)                                    # Mittelung erhaelt konstante Felder
    lin = np.stack([np.add.outer(np.arange(r - 4, r + 5), np.zeros(9)) for r, c in cen])[..., None]
    v = paste(lin.astype(np.float32), cen, 9, mask.shape)
    assert np.allclose(v[mask, 0], np.nonzero(mask)[0], atol=1e-4)       # und lineare (Zeile als Wert)


def test_predict_matches_forward():
    torch.manual_seed(0)
    net = ContagionSTS(d=32, heads=4, depth=1, kernel=7, obs_days=30, burn="curve").eval()
    pad, crop = 6 + 7, 7
    slab = np.random.default_rng(1).integers(0, 255, (30, 55, 20 + 2 * pad, 25 + 2 * pad)).astype(np.uint8)
    slab[:, 0] = 0
    slab[-1, 0, pad + 9, pad + 10] = 1                                # Feuer am Stichtag
    cen = tile_centres(np.ones((20, 25), bool), crop)
    m_t, s_t = torch.zeros(1, 1, 55, 1, 1), torch.full((1, 1, 55, 1, 1), 100.0)
    taus = [1, 7, 30]
    ls, lig = predict(net, slab, cen, crop, pad, 180, m_t, s_t, torch.device("cpu"), taus, bs=4)
    r, c = cen[3]
    x = (torch.from_numpy(slab[None, :, :, r + pad - 6:r + pad + 7, c + pad - 6:c + pad + 7]).float() - m_t) / s_t
    with torch.no_grad():
        ref, _ = net(x, torch.tensor([180.0]))
    assert np.allclose(ls[3], ref[0, 3:10, 3:10][..., taus].numpy(), atol=1e-5)
    assert np.all(lig <= 1e-7) and np.all(lig >= ls - 1e-5)          # Zuendung allein: hoechstens das Gesamtrisiko


def test_capture_perfect_ranking():
    mask = np.ones((10, 10), bool)
    first = np.zeros((10, 10), np.int16); first[0, 0] = 1; first[5, 5] = 3
    F = np.zeros((10, 10)); F[0, 0], F[5, 5] = 0.9, 0.8
    near = np.zeros((10, 10), bool); near[0, 0] = True
    r = capture(F, mask, first, 7, near)
    assert r["n_fire_px"] == 2 and r["n_new"] == 1 and r["auc"] == 1.0 and r["top5.0%_all"] == 1.0

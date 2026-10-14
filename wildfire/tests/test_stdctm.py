"""Formprueft das ST-DCTM-Modell des Paper-Protokolls."""

from __future__ import annotations

import torch

from wildfire.paper.stdctm import STDCTM, T


def test_output_shape_and_start_value():
    net = STDCTM().eval()
    x = torch.randn(3, T, 55, 13, 13)
    out = net(x, torch.tensor([2, 8, 31]))
    assert out.shape == (3,)
    # Kopf startet bei logit(1/3) fuer jeden Vorlauf (Basis summiert sich zu 1)
    assert torch.allclose(torch.sigmoid(out), torch.full((3,), 1 / 3), atol=1e-4)


def test_horizon_changes_output_once_head_is_trained():
    torch.manual_seed(0)
    net = STDCTM().eval()
    torch.nn.init.normal_(net.head.weight, std=0.5)
    x = torch.randn(1, T, 55, 13, 13).repeat(2, 1, 1, 1, 1)
    out = net(x, torch.tensor([2, 31]))
    assert not torch.allclose(out[0], out[1])


def test_any_horizon_between_paper_values_is_queryable():
    net = STDCTM().eval()
    x = torch.randn(4, T, 55, 13, 13)
    out = net(x, torch.tensor([4, 11, 21, 17]))
    assert torch.isfinite(out).all()

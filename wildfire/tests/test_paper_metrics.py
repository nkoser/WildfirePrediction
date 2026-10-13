"""Metriken des Paper-Protokolls an Faellen mit bekannter Antwort."""

from __future__ import annotations

import numpy as np
import pytest

from wildfire.paper.metrics import calibration_errors, paper_metrics


def test_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = paper_metrics(np.array([0.0, 0.1, 0.9, 1.0]), y)
    assert m["precision"] == m["recall"] == m["f1"] == 1.0
    assert m["pr_auc"] == m["roc_auc"] == 1.0


def test_f1_by_hand():
    # Schwelle 0,5: tp=1 (0.8), fp=1 (0.6), fn=1 (0.2)
    y = np.array([1, 0, 1, 0])
    m = paper_metrics(np.array([0.8, 0.6, 0.2, 0.1]), y)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_calibration_errors_by_hand():
    # Bin [0.2,0.3): p=0.25 zweimal, eine Eins -> Luecke 0,25
    # Bin [0.9,1.0]: p=0.95 zweimal, zwei Einsen -> Luecke 0,05
    p = np.array([0.25, 0.25, 0.95, 0.95])
    y = np.array([1, 0, 1, 1])
    ece, mce = calibration_errors(p, y)
    assert ece == pytest.approx(0.5 * 0.25 + 0.5 * 0.05)
    assert mce == pytest.approx(0.25)


def test_p_equal_one_lands_in_last_bin():
    ece, mce = calibration_errors(np.array([1.0, 1.0]), np.array([1, 1]))
    assert ece == 0.0 and mce == 0.0


def test_brier_of_constant_prior():
    y = np.array([1, 0, 0])
    m = paper_metrics(np.full(3, 1 / 3), y)
    assert m["brier"] == pytest.approx(2 / 9)
    assert m["roc_auc"] == pytest.approx(0.5)

import numpy as np

from wildfire.paper.paper_metrics_ckpt import calibrate, choose_calibration, curve_fit, curve_map, isotonic, platt


def test_isotonic_fixes_a_shape_platt_cannot():
    # wahre Wahrscheinlichkeit als Treppe in s: Platt (Sigmoid in logit s) passt die Form nicht, isoton schon
    rng = np.random.default_rng(0)
    s = rng.uniform(0.01, 0.99, 40000)
    p = np.where(s < 0.3, 0.05, np.where(s < 0.35, 0.6, 0.65))
    y = (rng.uniform(size=s.size) < p).astype(int)
    sv, yv, st, yt = s[:20000], y[:20000], s[20000:], y[20000:]
    pi, pp = isotonic(sv, yv, st), platt(sv, yv, st)
    assert np.all(np.diff(isotonic(sv, yv, np.sort(st))) >= 0)                  # monoton
    assert 1e-6 <= pi.min() and pi.max() <= 1 - 1e-6
    pt = p[20000:]
    assert np.mean((pi - pt) ** 2) < 0.5 * np.mean((pp - pt) ** 2)


def _sample(truth, n=8000, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(size=n) < truth(s)).astype(int)
    return s, y


def test_choice_picks_platt_when_truth_is_a_sigmoid_and_iso_for_a_step():
    sig = lambda s: 1 / (1 + np.exp(-(2.0 * np.log(s / (1 - s)) + 1.0)))
    s, y = _sample(sig)
    assert choose_calibration(s, y)[0] == "platt"
    step = lambda s: np.where(s < 0.3, 0.05, np.where(s < 0.35, 0.6, 0.65))
    s, y = _sample(step)
    assert choose_calibration(s, y)[0] == "iso"


def test_calibrate_returns_probabilities_and_the_choice():
    s, y = _sample(lambda s: s)
    p, name = calibrate(s, y, np.linspace(0.01, 0.99, 50))
    assert name in ("platt", "iso") and np.all((p >= 0) & (p <= 1)) and np.all(np.diff(p) >= -1e-12)


def test_curve_map_is_monotone_and_curve_fit_recovers_parameters():
    F = np.linspace(0.001, 0.9, 200)
    q = curve_map(F, 1.2, 0.8)
    assert np.all(np.diff(q) > 0)                                   # Reihenfolge und Monotonie in tau bleiben
    assert np.allclose(curve_map(F, 0.0, 1.0), F, atol=1e-9)        # a = 0, b = 1: unveraendert
    rng = np.random.default_rng(0)
    vals, ys = [], []
    for scale in (1.0, 2.0, 4.0):                                   # drei "Horizonte" mit verschieden grossem F
        f = np.clip(rng.uniform(0.001, 0.2, 20000) * scale, 1e-4, 0.95)
        vals.append(f); ys.append((rng.uniform(size=f.size) < curve_map(f, 1.2, 0.8)).astype(int))
    a, b = curve_fit(vals, ys)
    assert abs(a - 1.2) < 0.08 and abs(b - 0.8) < 0.05

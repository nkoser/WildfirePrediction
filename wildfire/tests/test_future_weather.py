import numpy as np
import torch

from wildfire.paper import future_weather as fw
from wildfire.paper.sts_dctm import STSDCTM, load_sts


def _net(**kw):
    torch.manual_seed(0)
    return STSDCTM(d=32, heads=4, depth=1, head="lh", future=10, **kw).eval()


def test_oracle_starts_as_the_model_without_future():
    net = _net()
    x, doy = torch.randn(2, 30, 55, 13, 13), torch.tensor([150.0, 200.0])
    xf, kf = torch.randn(2, 10, 25, 13, 13), torch.tensor([10, 3])
    torch.nn.init.normal_(net.pa_lin.weight)
    with torch.no_grad():
        a, b = net(x, doy)[1], net(x, doy, xf=xf, kf=kf)[1]
    assert torch.allclose(a, b)


def test_oracle_sees_only_the_first_k_days():
    net = _net()
    torch.nn.init.normal_(net.pa_lin.weight)
    torch.nn.init.normal_(net.future.out.weight, std=0.1)
    x, doy = torch.randn(1, 30, 55, 13, 13), torch.tensor([150.0])
    xf = torch.randn(1, 10, 25, 13, 13)
    later, inside = xf.clone(), xf.clone()
    later[:, 4:] = torch.randn(1, 6, 25, 13, 13)
    inside[:, 2] += 1.0
    k = torch.tensor([4])
    with torch.no_grad():
        base = net(x, doy)[1]
        a, b, c = (net(x, doy, xf=v, kf=k)[1] for v in (xf, later, inside))
        z = net(x, doy, xf=xf, kf=torch.tensor([0]))[1]
    assert torch.allclose(a, b) and not torch.allclose(a, c)
    assert torch.allclose(z, base) and not torch.allclose(a, base)


def test_checkpoint_roundtrip_keeps_the_oracle(tmp_path):
    net = _net()
    args = {"d": 32, "heads": 4, "depth": 1, "head": "lh", "future_days": 10}
    torch.save({"state": net.state_dict(), "args": args}, tmp_path / "c.pt")
    net2 = load_sts(tmp_path / "c.pt")
    assert net2.future is not None and net2.future.days == 10


class _Cube:
    """Zwei Wuerfeljahre mit je 20 Tagen; globaler Index 0..39."""
    _pos = {i: ("2001", i) for i in range(20)} | {20 + i: ("2002", i) for i in range(20)}


def test_future_window_stops_at_the_end_of_the_cube_year(monkeypatch):
    calls = []

    def fake_window(cube, L, r, c, T):
        calls.append((L, T))
        out = np.zeros((T, 55, 13, 13), np.uint8)
        out[:, 1] = np.arange(L - T + 1, L + 1)[:, None, None]      # Wetterkanal 1 = Tagesindex
        return out

    monkeypatch.setattr(fw, "window", fake_window)
    xf, k = fw.future_window(_Cube(), 15, 6, 6, F=10)
    assert k == 4 and calls == [(19, 4)]                             # Tage 16..19, dann Jahreswechsel
    assert xf[:4, 0, 0, 0].tolist() == [16, 17, 18, 19] and not xf[4:].any()
    xf, k = fw.future_window(_Cube(), 22, 6, 6, F=10)
    assert k == 10 and xf[:, 0, 0, 0].tolist() == list(range(23, 33))

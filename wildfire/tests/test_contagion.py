import numpy as np
import torch

from wildfire.models.surv_heads import head_loss
from wildfire.paper.contagion import ContagionSTS
from wildfire.paper.sts_dctm import build_sts


def _net(**kw):
    torch.manual_seed(0)
    return ContagionSTS(d=32, heads=4, depth=1, **kw).eval()


def test_shapes_monotone_and_trainable():
    net = _net()
    x = torch.randn(2, 30, 55, 13, 13)
    x[:, :, 0] = 0.0
    x[0, -1, 0, 6, 6] = 1.0                                   # Feuer am Stichtag in der Mitte
    ls_px, ls_pa = net(x, torch.tensor([150.0, 200.0]))
    assert ls_px.shape == (2, 13, 13, 101) and ls_pa.shape == (2, 101)
    assert torch.all(ls_px[..., 0] == 0) and torch.all(ls_pa[:, 0] == 0)
    assert torch.all(ls_px[..., 1:] <= ls_px[..., :-1] + 1e-7) and torch.all(ls_pa[:, 1:] <= ls_pa[:, :-1] + 1e-7)
    # Persistenz und Ausbreitung: brennende Mitte erhoeht das Risiko dort und bei den Nachbarn am Tag 1
    F1 = -torch.expm1(ls_px[..., 1])
    assert F1[0, 6, 6] > 5 * F1[1, 6, 6] and F1[0, 6, 7] > F1[1, 6, 7] and torch.isclose(F1[0, 0, 0], F1[0, 0, 0])
    net.train()
    ls_px, ls_pa = net(x, torch.tensor([150.0, 200.0]))
    ev = torch.zeros(2, 13, 13, dtype=torch.bool); ev[0, 6, 6] = True
    t = torch.full((2, 13, 13), 100); t[0, 6, 6] = 1
    loss = head_loss(net.px_head, ls_px, ev, t) + head_loss(net.pa_head, ls_pa, ev.flatten(1).any(1), t.flatten(1).min(1).values)
    loss.backward()
    assert torch.isfinite(loss) and net.ker.weight.grad.abs().sum() > 0 and net.ign.weight.grad.abs().sum() > 0


def test_patch_is_region_of_all_pixels_and_heads_read_curves():
    net = _net()
    x = torch.randn(1, 30, 55, 13, 13); x[:, :, 0] = 0.0; x[0, -3, 0, 2, 3] = 1.0
    ls_px, ls_pa = net(x, torch.tensor([180.0]))
    full = net.region_log_survival(x, torch.tensor([180.0]), torch.ones(169, dtype=torch.bool))
    assert torch.allclose(full, ls_pa, atol=1e-5)
    m = torch.zeros(169, dtype=torch.bool); m[6 * 13 + 6] = True
    one = net.region_log_survival(x, torch.tensor([180.0]), m)
    assert torch.allclose(one, ls_px[0, 6, 6][None], atol=1e-5)
    k = torch.tensor([7.0])
    assert torch.allclose(net.pa_head.log_survival(ls_pa, k)[:, 0].float(), ls_pa[:, 7])
    assert isinstance(build_sts("contagion", d=32, heads=4, depth=1), ContagionSTS)


def test_fire_norm_recovers_the_burning_state():
    net = _net()
    net.set_fire_norm(0.01, 0.1)
    x = torch.zeros(1, 30, 55, 13, 13)
    x[:, :, 0] = (0 - 0.01) / 0.1
    x[0, -1, 0, 4, 4] = (1 - 0.01) / 0.1
    obs, *_ = net.params(x, torch.tensor([100.0]))
    assert obs.sum() == 1 and obs[0, -1, 4 * 13 + 4] == 1


def test_v2_burn_curve_wide_kernel_full_window():
    torch.manual_seed(0)
    net = ContagionSTS(d=32, heads=4, depth=1, kernel=7, obs_days=30, burn="curve").eval()
    x = torch.randn(1, 30, 55, 13, 13); x[:, :, 0] = 0.0; x[0, 2, 0, 6, 6] = 1.0       # Feuer vor 27 Tagen
    ls_px, ls_pa = net(x, torch.tensor([200.0]))
    assert torch.all(ls_px[..., 1:] <= ls_px[..., :-1] + 1e-7)
    logD = net.log_burning(torch.zeros(1, 169))
    assert logD.shape == (1, 169, 130) and torch.all(logD[..., 0] == 0) and torch.all(logD[..., 1:] <= logD[..., :-1])
    full = net.region_log_survival(x, torch.tensor([200.0]), torch.ones(169, dtype=torch.bool))
    assert torch.allclose(full, ls_pa, atol=1e-5)
    # ein 27 Tage altes Feuer wirkt bei 7 beobachteten Tagen gar nicht, bei 30 schon
    short = ContagionSTS(d=32, heads=4, depth=1, kernel=7, obs_days=7, burn="curve").eval()
    obs7, *_ = short.params(x, torch.tensor([200.0]))
    obs30, *_ = net.params(x, torch.tensor([200.0]))
    assert obs7.sum() == 0 and obs30.sum() == 1


def test_geometric_rollout_matches_the_recursion():
    torch.manual_seed(1)
    net = ContagionSTS(d=32, heads=4, depth=1).eval()
    rho = torch.rand(2, 169) * 0.9
    logD = torch.log(rho)[..., None] * torch.arange(net.A, dtype=torch.float32)
    log_exo = -torch.rand(2, 100, 169) * 1e-2
    k = torch.rand(2, 25, 169) * 0.05
    ls, b = net.rollout(log_exo, k, logD, keep_b=True)
    # Rekursion der Version 1: b(s+1) = rho b(s) + f(s+1)
    S, bp = torch.ones(2, 169), torch.zeros(2, 169)
    for s in range(100):
        assert torch.allclose(b[:, s], bp, atol=1e-5)
        lp = torch.log1p(-(k * net._unfold(bp)).clamp(max=1 - 1e-6))
        l1mh = log_exo[:, s] + lp.sum(1) - lp[:, 12]
        f = S * -torch.expm1(l1mh)
        S = S * torch.exp(l1mh)
        bp = (bp * rho + f).clamp(max=1.0)


def test_blind_backbone_ignores_fire_channels_but_spread_still_uses_them():
    from wildfire.paper.contagion import BLIND
    net = _net(blind="thermal", obs_days=30)
    with torch.no_grad():                                     # Koepfe nicht trivial (Start ist Null)
        for lin in (net.ign, net.ker, net.dur):
            lin.weight.normal_(0, 0.1)
    doy = torch.tensor([200.0])
    x = torch.randn(1, 30, 55, 13, 13); x[:, :, 0] = 0.0
    y = x.clone()
    y[0, -1, 0, 6, 7] = 1.0                                   # Feuer beim Nachbarn am Stichtag
    y[:, :, list(BLIND["thermal"])[1:]] += 3.0                 # thermische Kanaele veraendert
    assert torch.equal(net.encode_input(x, doy), net.encode_input(y, doy))
    _, a0, k0, d0 = net.params(x, doy)
    obs1, a1, k1, d1 = net.params(y, doy)
    assert torch.equal(a0, a1) and torch.equal(k0, k1) and torch.equal(d0, d1)
    assert obs1.sum() == 1                                    # Brennzustand weiter aus dem Feuerkanal
    F0 = -torch.expm1(net(x, doy)[0][0, 6, 6, 1]); F1 = -torch.expm1(net(y, doy)[0][0, 6, 6, 1])
    assert F1 > F0                                            # Risiko steigt nur ueber b_j * k
    x2 = x.clone(); x2[:, :, 30] += 3.0                       # nicht ausgeblendeter Kanal (Gelaende) wirkt weiter
    assert not torch.allclose(net.params(x2, doy)[1], a0)
    full = _net(obs_days=30)                                  # ohne blind sieht der Rumpf den Feuerkanal
    assert not torch.equal(full.encode_input(x, doy), full.encode_input(y, doy))


def test_blind_roundtrip_through_checkpoint(tmp_path):
    from wildfire.paper.sts_dctm import load_sts
    net = _net(blind="modis")
    path = tmp_path / "cg3.pt"
    torch.save({"state": net.state_dict(), "args": {"head": "contagion", "d": 32, "heads": 4, "depth": 1,
                                                    "cg_kernel": 5, "cg_obs_days": 7, "cg_burn": "geometric",
                                                    "cg_blind": "modis"}}, path)
    net2 = load_sts(path)
    assert net2.blind == "modis" and net2.blind_ch == net.blind_ch

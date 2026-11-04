import torch

from wildfire.eval.metrics import d_calibration
from wildfire.models.surv_heads import make_head
from wildfire.models.xcal import xcal_penalty


def _batch(factor, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    lam = torch.exp(torch.empty(n).uniform_(-6.2, -2.8, generator=g))
    T = torch.ceil(-torch.log1p(-torch.rand(n, generator=g)) / lam).long()
    e, t = T <= 100, T.clamp(max=100)
    head = make_head("lh", 100.0, 12, 0.05)
    # Logistic-Hazard-Logits so, dass h = 1 - exp(-lam * factor) (konstant ueber die Tage)
    h = 1 - torch.exp(-lam * factor)
    raw = (torch.log(h) - torch.log1p(-h))[:, None].repeat(1, head.n_out) - head.bias   # h = sigma(raw + bias)
    return head, raw.double(), e, t


def test_xcal_penalty_small_when_calibrated_and_matches_dcal_direction():
    head, raw, e, t = _batch(1.0)
    p_ok = float(xcal_penalty(head, raw, e, t))
    head, raw2, e2, t2 = _batch(3.0)
    p_bad = float(xcal_penalty(head, raw2, e2, t2))
    assert p_ok < 0.05 and p_bad > 5 * p_ok
    F = 1 - torch.exp(head.log_survival(raw2, torch.arange(1, 101, dtype=torch.float64)))
    assert d_calibration(F.numpy(), e2.numpy(), t2.numpy())["d_cal_max_dev"] > 0.05


def test_xcal_penalty_has_gradients():
    head, raw, e, t = _batch(3.0, n=256)
    raw = raw.clone().requires_grad_(True)
    xcal_penalty(head, raw, e, t).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0

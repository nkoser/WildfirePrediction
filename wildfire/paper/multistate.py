"""Multi-State-Contagion-STS: Zustaende unverbrannt (U), brennend (B), erloschen (O) je Pixel, mit Rueckkehr.

ContagionSTS (paper/contagion.py) modelliert U -> B durch Zuendung und Ausbreitung und B -> O ueber die Brenndauer.
Ein Pixel, das im Fenster brannte und erloschen ist, hatte dort bisher nur Zuendung und Ausbreitung wie jedes
andere. Hier kommt der dritte Uebergang O -> B dazu (Wiederaufflammen bzw. Rueckkehr des Feuers, 12,7 % der
Feuertage nach einer feuerfreien Woche, reburn_stats.py):

    1 - h_i(t) = (1 - a_i(t)) (1 - r_i(t) o_i(t)) prod_{j != i} (1 - g_i k_ij b_j(t-1)) (1 - k_ii b_i^obs(t-1))

  a_i(t)   Zuendung (wie ContagionSTS)
  r_i(t)   Rueckkehr-Hazard je Tag 1..H, linear aus dem Pixel-Token (neu, wie a)
  o_i(t)   Wahrscheinlichkeit, dass i am Vortag erloschen ist: i brannte im Fenster (ever_i) und brennt laut
           Brenndauer nicht mehr, o_i = ever_i (1 - b_i^obs(t-1)); fuer das ERSTE Ereignis nach L haengt o_i nur
           von der beobachteten Historie ab (eigene vorhergesagte Feuer beenden die Kurve), also exakt
  g_i      Brennstoff: Ausbreitung in ein schon verbranntes Pixel wird mit g = sigmoid(gamma) gedaempft
           (g_i = g fuer ever_i = 1, sonst 1; gamma gelernt, Start g ~ 0,95)
  k_ii b_i^obs  Fortsetzung eines am Stichtag brennenden Pixels (Mitte des Kerns, wie ContagionSTS)

Konkurrierende Ursachen des ersten Ereignisses: Zuendung, Ausbreitung (beobachtet + vorhergesagt), Fortsetzung,
Rueckkehr. Ursachen-CIFs F_c(tau) = sum_{t <= tau} S(t-1) h(t) w_c(t) mit w_c = -log(1 - h_c) / -log(1 - h)
(Aufteilung des kumulativen Hazards je Tag, decompose_causes).

Umsetzung ohne Aenderung an bestehendem Code: params() faltet Rueckkehr und Brennstoff in a und k
(a_eff = 1 - (1 - a)(1 - r o)), daher laufen exogenous, rollout, region_log_survival, joint_*, surv_eval,
paper_metrics_ckpt unveraendert. Der reine Zuendungs-Hazard bleibt fuer den Ursachen-Term in self._last.
Die Eingabe ist dieselbe wie bei allen Modellen (30 Fenstertage, 55 Kanaele): Zustaende kommen aus dem
Feuerkanal des Fensters; Rueckkehr nach mehr als 30 Tagen Pause sieht das Modell nicht (Abweichung vom
Entwurf mit Zustandskanaelen ueber 365 Tage, siehe paper_notes/multistate_2026-10-26.md).

  python -m wildfire.paper.multistate train <Argumente von contagion_cause>   Training (Schleife von contagion_cause)
  python -m wildfire.paper.multistate <modul> <Argumente>                    Auswertung, Laden erkennt das Modell
"""

from __future__ import annotations

import importlib
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.paper import sts_dctm
from wildfire.paper.contagion import ContagionSTS

_ORIG_LOAD = sts_dctm.load_sts
RET_INIT = 0.002                 # Rueckkehr-Hazard je Tag zu Beginn (~ 18 % in 100 Tagen fuer erloschene Pixel)
FUEL_INIT = 3.0                  # gamma, g = sigmoid(3) ~ 0,95


class MultiStateContagionSTS(ContagionSTS):
    def __init__(self, d: int = 64, horizon: int = 100, **kw):
        super().__init__(d=d, horizon=horizon, **kw)
        self.head_kind = "contagion"
        self.multistate = True
        self.ret = nn.Linear(d, horizon)
        nn.init.zeros_(self.ret.weight); nn.init.zeros_(self.ret.bias)
        self.register_buffer("ret_bias", torch.tensor(math.log(RET_INIT / (1 - RET_INIT))))
        self.fuel = nn.Parameter(torch.tensor(FUEL_INIT))
        self._last = {}

    def params(self, x, doy, pix=None):
        x = x[:, -self.hist:]
        pix = self.encode_input(x, doy) if pix is None else pix
        obs, a, k, logD = super().params(x, doy, pix)
        with torch.autocast(device_type=x.device.type, enabled=False):
            ever = (obs.amax(1) > 0).float()                                            # (B, N)
            o = ever[:, None] * (1 - self_burning(self, obs, logD))                      # (B, H, N)
            r = torch.sigmoid(self.ret(pix.float()) + self.ret_bias).transpose(1, 2)    # (B, H, N)
            ro = r * o
            a_eff = 1 - (1 - a) * (1 - ro)
            g = torch.sigmoid(self.fuel)
            damp = 1 - (1 - g) * ever                                                    # (B, N) je Empfaenger
            c = self.K * self.K // 2
            rowmask = torch.ones(self.K * self.K, 1, device=k.device)
            rowmask[c] = 0.0                                                             # Fortsetzung ungedaempft
            k_eff = k * (1 - rowmask * (1 - damp[:, None, :]))
        self._last = {"a": a, "ro": ro, "ever": ever}
        return obs, a_eff, k_eff, logD

    def decompose_causes(self, x, doy):
        return decompose_causes(self, x, doy)


def self_burning(net, obs, logD):
    """b_j^obs(t-1) (B, H, N): Brennzustand am Vortag aus beobachtetem Feuer (wie ContagionSTS.exogenous)."""
    D = obs.shape[1]
    idx = (torch.arange(net.H, device=obs.device)[:, None]
           + torch.arange(D - 1, -1, -1, device=obs.device)[None, :])                   # (H, D)
    pw = logD[:, :, idx].exp().permute(0, 2, 3, 1)                                      # (B, H, D, N)
    return 1 - torch.prod(1 - obs[:, None] * pw, dim=2)


@torch.no_grad()
def decompose_causes(net, x, doy):
    """Ursachen-CIFs des Pixels (B, N, H+1) je Ursache: ign, ret, cont, spr; dazu log S (B, N, H+1).
    Rollout wie ContagionSTS.rollout, mit getrennt gefuehrten log(1 - h_c) je Tag. Funktioniert auch fuer
    ContagionSTS ohne Rueckkehr (dann ret = 0)."""
    obs, a_eff, k, logD = net.params(x, doy)
    if getattr(net, "multistate", False):
        a, ro = net._last["a"], net._last["ro"]
    else:
        a, ro = a_eff, torch.zeros_like(a_eff)
    B, D, N = obs.shape
    KK, c = net.K * net.K, net.K * net.K // 2
    b_obs = self_burning(net, obs, logD)                                                 # (B, H, N) alle Pixel
    U = net._unfold(b_obs.reshape(B * net.H, N)).view(B, net.H, KK, N)
    lo = torch.log1p(-(k[:, None] * U).clamp(max=1 - 1e-6))                              # (B, H, KK, N)
    l_cont = lo[:, :, c]
    l_spr_obs = lo.sum(2) - l_cont
    l_ign = torch.log1p(-a.clamp(max=1 - 1e-6))
    l_ret = torch.log1p(-ro.clamp(max=1 - 1e-6))
    logS = obs.new_zeros(B, N)
    S = obs.new_ones(B, N)
    b_pred = obs.new_zeros(B, N)
    fs = []
    out = {kk: [obs.new_zeros(B, N)] for kk in ("ign", "ret", "cont", "spr")}
    ls = [logS]
    for s in range(net.H):
        lp = torch.log1p(-(k * net._unfold(b_pred)).clamp(max=1 - 1e-6))
        l_spr = l_spr_obs[:, s] + lp.sum(1) - lp[:, c]                                  # Mitte nur beobachtet
        parts = {"ign": l_ign[:, s], "ret": l_ret[:, s], "cont": l_cont[:, s], "spr": l_spr}
        l1mh = sum(parts.values())
        f = S * -torch.expm1(l1mh)                                                       # P(T = s + 1)
        tot = l1mh.clamp(max=-1e-12)
        for kk, v in parts.items():
            out[kk].append(out[kk][-1] + f * (v / tot))
        logS = logS + l1mh
        S = torch.exp(logS)
        fs.append(f)
        Dk = logD[..., :s + 1].flip(-1).exp()
        b_pred = (torch.stack(fs, -1) * Dk).sum(-1).clamp(max=1.0)
        ls.append(logS)
    res = {kk: torch.stack(v, -1) for kk, v in out.items()}
    res["logS"] = torch.stack(ls, -1)
    return res


def build_ms(head: str = "contagion", **kw):
    if head != "contagion":
        raise SystemExit("ABBRUCH: multistate nur als Contagion-Kopf")
    for key in ("attn", "stem"):
        if kw.get(key, {"attn": "factorized", "stem": "2d"}[key]) != {"attn": "factorized", "stem": "2d"}[key]:
            raise SystemExit(f"ABBRUCH: multistate nur mit {key} wie ContagionSTS")
    return MultiStateContagionSTS(**kw)


def is_multistate(state: dict) -> bool:
    return "ret.weight" in state and "fuel" in state


def load_any(path, device="cpu"):
    """Wie sts_dctm.load_sts; erkennt zusaetzlich die Multi-State-Contagion am state dict."""
    c = torch.load(path, map_location="cpu")
    if not is_multistate(c["state"]):
        return _ORIG_LOAD(path, device)
    a = c.get("args", {})
    net = MultiStateContagionSTS(d=a.get("d", 64), heads=a.get("heads", 4), depth=a.get("depth", 4),
                                 hist=a.get("hist_days", sts_dctm.T_WIN), rate=a.get("rate_weight", 0.0) > 0,
                                 kernel=a.get("cg_kernel", 7), obs_days=a.get("cg_obs_days", 30),
                                 burn=a.get("cg_burn", "curve"), blind=a.get("cg_blind", "none"))
    net.load_state_dict(c["state"])
    return net.to(device).eval()


def patch_loading() -> None:
    sts_dctm.load_sts = load_any
    for mod in list(sys.modules.values()):                    # schon importierte Bindungen "from ... import load_sts"
        if mod is not None and getattr(mod, "load_sts", None) is _ORIG_LOAD:
            mod.load_sts = load_any


def forward_parts(net, x, doy, device):
    """Wie contagion_cause.forward_parts, gibt aber den REINEN Zuendungs-Hazard fuer den Ursachen-Term zurueck."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pix = net.encode_input(x, doy)
    obs, a_eff, k, logD = net.params(x, doy, pix)
    with torch.autocast(device_type=device.type, enabled=False):
        log_exo = net.exogenous(obs, a_eff, k, logD)
        ls_px = net.rollout(log_exo, k, logD)
        ls_pa = F.pad(log_exo.sum(2).cumsum(1), (1, 0))
    return ls_px, ls_pa, net._last["a"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    target, rest = sys.argv[1], sys.argv[2:]
    if target == "train":
        from wildfire.paper import contagion_cause
        contagion_cause.build_sts = build_ms
        contagion_cause.forward_parts = forward_parts
        sys.argv = ["contagion_cause"] + rest
        contagion_cause.main()
        return
    patch_loading()
    sys.argv = [target] + rest
    importlib.import_module(target).main()


if __name__ == "__main__":
    main()

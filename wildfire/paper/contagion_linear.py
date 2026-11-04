"""Lineare Contagion: derselbe Kopf wie ContagionSTS, aber ein linearer Encoder statt des STS-Rumpfs.

Das ist die klassische endemisch-epidemische Struktur (Hoehle 2009 twinSIR; Deardon et al. 2010 ILM) mit
log-linearen Kovariaten: Zuendung a_i(t), Kern k_ij und Brenndauer D_i haengen linear (auf der Logit-Skala) von den
Merkmalen des Empfaengerpixels i ab. Der Vergleich mit ContagionSTS zeigt, was der tiefe Encoder beitraegt.

  Merkmale je Pixel  alle 55 Kanaele am Stichtag, ihr Mittel ueber die letzten 7 Fenstertage,
                     log(1 + Feuertage im Fenster), Kalendertag (sin, cos)  -> 113
  Encoder            ein nn.Linear(113, d); da ign, ker und dur selbst linear sind, ist das ganze Modell linear in
                     den Merkmalen (d >= 113 -> kein Rang-Engpass)

Bestehende Module bleiben unveraendert. Dieses Modul haengt sich von aussen ein:
  - train: ersetzt sts_dctm.build_sts fuer --head contagion und ruft sts_dctm.main() (alle Argumente wie dort)
  - jedes andere Modul mit main(): ersetzt sts_dctm.load_sts (und gleichnamige, schon importierte Bindungen),
    sodass Checkpoints der linearen Contagion erkannt werden (state dict mit lin.weight und ohne stem.*);
    alle anderen Checkpoints laden wie bisher.

  python -m wildfire.paper.contagion_linear train <Argumente von sts_dctm>
  python -m wildfire.paper.contagion_linear <modul, z. B. wildfire.paper.joint_calib> <Argumente des Moduls>
"""

from __future__ import annotations

import importlib
import math
import sys

import torch
import torch.nn as nn

from wildfire.paper import sts_dctm
from wildfire.paper.contagion import ContagionSTS
from wildfire.paper.prepare import FIRE_CH

_ORIG_BUILD = sts_dctm.build_sts
_ORIG_LOAD = sts_dctm.load_sts


class LinearContagionSTS(ContagionSTS):
    def __init__(self, c_in: int = 55, d: int = 128, **kw):
        super().__init__(c_in=c_in, d=d, **kw)
        for name in ("stem", "blocks", "norm", "read", "season", "pos_s", "pos_t"):
            delattr(self, name)
        self.n_feat = 2 * c_in + 3
        self.lin = nn.Linear(self.n_feat, d)

    def encode(self, x, doy):
        """(B, T, C, 13, 13) normiert, (B,) -> Token (B, 169, d), linear in den Merkmalen."""
        x = x[:, -self.hist:]
        last = x[:, -1].flatten(2).transpose(1, 2)                           # (B, 169, C)
        m7 = x[:, -7:].mean(1).flatten(2).transpose(1, 2)
        fire = ((x[:, :, FIRE_CH] * self.fire_sd + self.fire_mu) > 0.5).float().sum(1).flatten(1)
        ang = 2 * math.pi * doy.float() / 365.25
        sea = torch.stack([ang.sin(), ang.cos()], -1)[:, None].expand(-1, last.shape[1], -1)
        f = torch.cat([last, m7, torch.log1p(fire)[..., None], sea.to(last.dtype)], -1)
        return self.lin(f)


def build_linear(head: str = "dctm", **kw):
    if head != "contagion":
        raise SystemExit("ABBRUCH: contagion_linear train nur mit --head contagion")
    kw.pop("attn", None)
    kw.pop("stem", None)
    return LinearContagionSTS(**kw)


def is_linear(state: dict) -> bool:
    return "lin.weight" in state and not any(k.startswith("stem.") for k in state)


def load_any(path, device="cpu"):
    """Wie sts_dctm.load_sts; erkennt zusaetzlich die lineare Contagion am state dict."""
    c = torch.load(path, map_location="cpu")
    if not is_linear(c["state"]):
        return _ORIG_LOAD(path, device)
    a = c.get("args", {})
    net = LinearContagionSTS(d=a.get("d", 128), heads=a.get("heads", 4), depth=a.get("depth", 4),
                             hist=a.get("hist_days", sts_dctm.T_WIN), rate=a.get("rate_weight", 0.0) > 0,
                             kernel=a.get("cg_kernel", 5), obs_days=a.get("cg_obs_days", 7),
                             burn=a.get("cg_burn", "geometric"), blind=a.get("cg_blind", "none"))
    net.load_state_dict(c["state"])
    return net.to(device).eval()


def patch_loading() -> None:
    sts_dctm.load_sts = load_any
    for mod in list(sys.modules.values()):                    # schon importierte Bindungen "from ... import load_sts"
        if mod is not None and getattr(mod, "load_sts", None) is _ORIG_LOAD:
            mod.load_sts = load_any


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    target, rest = sys.argv[1], sys.argv[2:]
    if target == "train":
        sts_dctm.build_sts = build_linear
        sys.argv = ["sts_dctm"] + rest
        sts_dctm.main()
        return
    patch_loading()
    sys.argv = [target] + rest
    importlib.import_module(target).main()


if __name__ == "__main__":
    main()

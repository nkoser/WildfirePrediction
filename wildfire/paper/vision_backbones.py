"""Vision-Baselines U-TAE und ConvLSTM als Rumpf des STS-Modells, mit denselben Koepfen, Verlusten und Auswertungen.

Statt des STS-Transformers liefert ein anderer Rumpf die Pixel-Token (B, 169, d); alles danach bleibt wie in
sts_dctm.STSDCTM (Pixelkopf px_lin/px_head, Ausschnittskopf ueber Attention-Pooling pa_lin/pa_head). Damit misst der
Vergleich mit STS-LH (gleicher Kopf --head lh, gleiche Daten, gleiche Schritte) nur den Rumpf.

  utae      U-TAE nach Garnot & Chanussot (ICCV 2021): je Tag ein geteilter Faltungs-Encoder (U-Net-Abstieg
            13 -> 7 -> 4), L-TAE (lightweight temporal attention, eine gelernte Query je Kopf) auf der groebsten
            Stufe, deren Attention-Masken (bilinear hochskaliert) die Tage an jeder Skip-Stufe gruppenweise mitteln
            ("att_group"), dann U-Net-Aufstieg
  convlstm  ConvLSTM nach Shi et al. (NeurIPS 2015): zwei gestapelte ConvLSTM-Schichten (3 x 3, mit Peepholes) ueber
            die 30 Fenstertage auf voller 13 x 13-Aufloesung; Token = letzter verborgener Zustand der oberen Schicht

Abweichungen von den Originalen (wegen 13 x 13-Ausschnitten und ~0,4 M Parametern wie STS-LH d 64):
  U-TAE     3 statt 4 Stufen, Breiten 32/48/64 statt 64/64/64/128, 8 statt 16 Koepfe, d_model 128 statt 256,
            Stride-Faltung 3 x 3 statt 4 x 4 (13 -> 7 -> 4 geht auf), GroupNorm auch im Decoder statt BatchNorm (EMA),
            Positionskodierung = Fenstertag 0..29 statt Aufnahmedatum
  ConvLSTM  nur der Encoder (kein Forecaster-Netz: die Survival-Koepfe ersetzen ihn), verborgene Breite 48
  beide     Kalendertag des Stichtags (sin, cos) als 2 konstante Zusatzkanaele je Tag statt des additiven
            Saison-Embeddings des STS (bei einem linearen Kopf haette ein additives Embedding kaum Wirkung)

Bestehende Module bleiben unveraendert; dieses haengt sich von aussen ein (wie contagion_linear):
  python -m wildfire.paper.vision_backbones train <utae|convlstm> <Argumente von sts_dctm>
  python -m wildfire.paper.vision_backbones <modul> <Argumente des Moduls>
     ersetzt sts_dctm.load_sts, sodass Checkpoints mit utae.* / clstm.* erkannt werden; alle anderen laden wie bisher
"""

from __future__ import annotations

import importlib
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.paper import sts_dctm

_ORIG_BUILD = sts_dctm.build_sts
_ORIG_LOAD = sts_dctm.load_sts
BACKBONES = ("utae", "convlstm")


# ------------------------------------------------------------------ U-TAE
def conv_layer(c_in, c_out, k=3, s=1, groups=4, act=True):
    mods = [nn.Conv2d(c_in, c_out, k, s, padding=k // 2, padding_mode="reflect"), nn.GroupNorm(groups, c_out)]
    return nn.Sequential(*mods, nn.ReLU()) if act else nn.Sequential(*mods)


class DownBlock(nn.Module):
    """Stride-2-Faltung, dann zwei Faltungen mit Residuum (UTAE DownConvBlock)."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.down = conv_layer(c_in, c_out, 3, 2)
        self.conv1 = conv_layer(c_out, c_out)
        self.conv2 = conv_layer(c_out, c_out)

    def forward(self, x):
        x = self.conv1(self.down(x))
        return x + self.conv2(x)


class UpBlock(nn.Module):
    """Transponierte Faltung, Skip (1 x 1) anhaengen, zwei Faltungen mit Residuum (UTAE UpConvBlock)."""

    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 3, 2, padding=1), nn.GroupNorm(4, c_out), nn.ReLU())
        self.skip = nn.Sequential(nn.Conv2d(c_skip, c_skip, 1), nn.GroupNorm(4, c_skip), nn.ReLU())
        self.conv1 = conv_layer(c_out + c_skip, c_out)
        self.conv2 = conv_layer(c_out, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        x = x[..., :skip.shape[-2], :skip.shape[-1]]
        x = self.conv1(torch.cat([x, self.skip(skip)], 1))
        return x + self.conv2(x)


class LTAE(nn.Module):
    """Lightweight temporal attention (Garnot & Chanussot 2020/2021) je Pixel:
    (B, T, C, h, w) -> (B, C_out, h, w) und Masken (n_head, B, T, h, w)."""

    def __init__(self, c_in, d_model=128, n_head=8, d_k=4, c_out=64, p=0.1, t_pos=1000.0):
        super().__init__()
        self.n_head, self.d_k, self.d_model = n_head, d_k, d_model
        self.in_norm = nn.GroupNorm(n_head, c_in)
        self.inconv = nn.Conv1d(c_in, d_model, 1)
        dh = d_model // n_head
        inv = t_pos ** (-2 * (torch.arange(dh) // 2).float() / dh)
        self.register_buffer("inv", inv, persistent=False)
        self.Q = nn.Parameter(torch.randn(n_head, d_k) * math.sqrt(2.0 / d_k))
        self.fc_k = nn.Linear(d_model, n_head * d_k)
        self.drop = nn.Dropout(p)
        self.mlp = nn.Sequential(nn.Linear(d_model, c_out), nn.GroupNorm(n_head, c_out), nn.ReLU())
        self.out_norm = nn.GroupNorm(n_head, c_out)

    def pos(self, T, device):
        a = torch.arange(T, device=device).float()[:, None] * self.inv[None]   # (T, dh)
        pe = torch.where(torch.arange(a.shape[1], device=device) % 2 == 0, a.sin(), a.cos())
        return pe.repeat(1, self.n_head)                                          # (T, d_model)

    def forward(self, x):
        B, T, C, h, w = x.shape
        z = x.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, C)                    # (N, T, C)
        z = self.in_norm(z.transpose(1, 2))                                      # (N, C, T)
        z = self.inconv(z).transpose(1, 2)                                       # (N, T, d_model)
        z = z + self.pos(T, z.device)[None].to(z.dtype)
        k = self.fc_k(z).view(-1, T, self.n_head, self.d_k)                      # (N, T, nh, dk)
        att = torch.einsum("ntgk,gk->gnt", k, self.Q.to(k.dtype)) / math.sqrt(self.d_k)
        att = self.drop(att.float().softmax(-1)).to(z.dtype)                     # (nh, N, T)
        v = z.view(-1, T, self.n_head, self.d_model // self.n_head)              # Werte = Eingang, gruppenweise
        out = torch.einsum("gnt,ntgc->ngc", att, v).reshape(-1, self.d_model)
        out = self.out_norm(self.drop(self.mlp(out)))                            # (N, c_out)
        out = out.view(B, h, w, -1).permute(0, 3, 1, 2)
        return out, att.view(self.n_head, B, h, w, T).permute(0, 1, 4, 2, 3)


def group_aggregate(x, att):
    """att_group: (B, T, C, H, W) mit Masken (nh, B, T, h, w) -> (B, C, H, W); Kanalgruppe g nimmt Maske g."""
    nh, B, T = att.shape[:3]
    H, W = x.shape[-2:]
    a = F.interpolate(att.reshape(nh * B, T, *att.shape[-2:]).float(), size=(H, W), mode="bilinear",
                      align_corners=False).view(nh, B, T, 1, H, W).to(x.dtype)
    xg = x.view(B, T, nh, x.shape[2] // nh, H, W).permute(2, 0, 1, 3, 4, 5)       # (nh, B, T, C/nh, H, W)
    return (a * xg).sum(2).permute(1, 0, 2, 3, 4).reshape(B, -1, H, W)


class UTAE(nn.Module):
    def __init__(self, c_in, d, widths=(32, 48, 64), n_head=8, d_model=128, d_k=4, p=0.1):
        super().__init__()
        self.inc = nn.Sequential(conv_layer(c_in, widths[0]), conv_layer(widths[0], widths[0]))
        self.down = nn.ModuleList([DownBlock(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.tae = LTAE(widths[-1], d_model, n_head, d_k, widths[-1], p)
        self.up = nn.ModuleList([UpBlock(widths[i + 1], widths[i], widths[i]) for i in reversed(range(len(widths) - 1))])
        self.out = nn.Sequential(conv_layer(widths[0], widths[0]), nn.Conv2d(widths[0], d, 1))

    def forward(self, x):
        """(B, T, C, 13, 13) -> (B, d, 13, 13)."""
        B, T = x.shape[:2]
        f = self.inc(x.flatten(0, 1))
        feats = [f]
        for blk in self.down:
            f = blk(f)
            feats.append(f)
        unflat = [g.view(B, T, *g.shape[1:]) for g in feats]
        out, att = self.tae(unflat[-1])
        for i, blk in enumerate(self.up):
            skip = unflat[-2 - i]
            out = blk(out, group_aggregate(skip, att))
        return self.out(out)


# ------------------------------------------------------------------ ConvLSTM
class ConvLSTMCell(nn.Module):
    """Shi et al. 2015, Gl. (3): Gates aus Faltungen von X_t und H_{t-1}, Peepholes W_c* o C (Hadamard)."""

    def __init__(self, c_in, c_hid, k=3, size=sts_dctm.P):
        super().__init__()
        self.c_hid = c_hid
        self.conv = nn.Conv2d(c_in + c_hid, 4 * c_hid, k, padding=k // 2)
        self.w_ci = nn.Parameter(torch.zeros(1, c_hid, size, size))
        self.w_cf = nn.Parameter(torch.zeros(1, c_hid, size, size))
        self.w_co = nn.Parameter(torch.zeros(1, c_hid, size, size))
        with torch.no_grad():                                    # Vergessens-Gate zu Beginn offen
            self.conv.bias[c_hid:2 * c_hid].fill_(1.0)

    def forward(self, x, h, c):
        i, f, g, o = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        i = torch.sigmoid(i + self.w_ci * c)
        f = torch.sigmoid(f + self.w_cf * c)
        c = f * c + i * torch.tanh(g)
        o = torch.sigmoid(o + self.w_co * c)
        return o * torch.tanh(c), c


class ConvLSTM(nn.Module):
    def __init__(self, c_in, d, hid=(48, 48), k=3):
        super().__init__()
        dims = (c_in,) + tuple(hid)
        self.cells = nn.ModuleList([ConvLSTMCell(dims[i], dims[i + 1], k) for i in range(len(hid))])
        self.out = nn.Conv2d(hid[-1], d, 1)

    def forward(self, x):
        """(B, T, C, 13, 13) -> (B, d, 13, 13): letzter verborgener Zustand der oberen Schicht."""
        B, T, _, H, W = x.shape
        seq = x
        for cell in self.cells:
            h = seq.new_zeros(B, cell.c_hid, H, W)
            c = h.float()
            outs = []
            for t in range(T):
                h, c = cell(seq[:, t], h, c)
                outs.append(h)
            seq = torch.stack(outs, 1)
        return self.out(seq[:, -1])


# ------------------------------------------------------------------ STS-Netz mit anderem Rumpf
class VisionSTS(sts_dctm.STSDCTM):
    def __init__(self, backbone: str = "utae", c_in: int = 55, d: int = 64, p: float = 0.1, **kw):
        if backbone not in BACKBONES:
            raise ValueError(f"backbone {backbone} nicht in {BACKBONES}")
        kw.pop("attn", None); kw.pop("stem", None); kw.pop("depth", None)
        super().__init__(c_in=c_in, d=d, p=p, **kw)
        for name in ("stem", "blocks", "norm", "read", "season", "pos_s", "pos_t"):
            delattr(self, name)
        self.backbone = backbone
        if backbone == "utae":
            self.utae = UTAE(c_in + 2, d, p=p)
        else:
            self.clstm = ConvLSTM(c_in + 2, d)

    def encode(self, x, doy):
        """(B, 30, C, 13, 13) normiert, (B,) -> Pixel-Token (B, 169, d); nur die letzten self.hist Tage."""
        x = x[:, -self.hist:]
        B, T, _, H, W = x.shape
        ang = 2 * math.pi * doy.float() / 365.25
        sea = torch.stack([ang.sin(), ang.cos()], -1).to(x.dtype)[:, None, :, None, None].expand(B, T, 2, H, W)
        x = torch.cat([x, sea], 2)
        z = self.utae(x) if self.backbone == "utae" else self.clstm(x)
        return z.flatten(2).transpose(1, 2)


def builder(backbone: str):
    def build(head: str = "dctm", **kw):
        if head == "contagion":
            raise SystemExit("ABBRUCH: vision_backbones nur mit den STS-Koepfen (lh, dctm, ...), nicht contagion")
        if kw.get("future"):
            raise SystemExit("ABBRUCH: vision_backbones ohne --future-days")
        return VisionSTS(backbone, head=head, **kw)
    return build


def backbone_of(state: dict) -> str | None:
    if any(k.startswith("utae.") for k in state):
        return "utae"
    if any(k.startswith("clstm.") for k in state):
        return "convlstm"
    return None


def load_any(path, device="cpu"):
    """Wie sts_dctm.load_sts; erkennt zusaetzlich U-TAE- und ConvLSTM-Checkpoints am state dict."""
    c = torch.load(path, map_location="cpu")
    bb = backbone_of(c["state"])
    if bb is None:
        return _ORIG_LOAD(path, device)
    a = c.get("args", {})
    net = VisionSTS(bb, d=a.get("d", 64), heads=a.get("heads", 4), head=a.get("head", "lh"),
                    hist=a.get("hist_days", sts_dctm.T_WIN), rate=a.get("rate_weight", 0.0) > 0,
                    bce_taus=a.get("bce_taus"))
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
        if not rest or rest[0] not in BACKBONES:
            raise SystemExit(f"ABBRUCH: train <{'|'.join(BACKBONES)}> <Argumente von sts_dctm>")
        sts_dctm.build_sts = builder(rest[0])
        sys.argv = ["sts_dctm"] + rest[1:]
        sts_dctm.main()
        return
    patch_loading()
    sys.argv = [target] + rest
    importlib.import_module(target).main()


if __name__ == "__main__":
    main()

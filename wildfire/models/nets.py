"""Netze fuer B2 und B3.

``SpatialHazardUNet`` (B2): ein kleines U-Net ueber den Patch. Eingabe sind
mehrere Zeitscheiben als Kanaele gestapelt - "kurze gestapelte Historie"
im Sinne des Arbeitsplans. Die Ausgabe ist eine DICHTE Karte: fuer jedes
Pixel des Patches H Logits.

Dass die Ausgabe dicht ist, obwohl Verlust und Auswertung zunaechst nur am
Zentrumspixel gerechnet werden, ist Absicht: Es haelt B2 direkt mit B0/B1
vergleichbar und baut zugleich die Maschinerie, die das Hauptmodell
braucht. Der dichte Verlust ist der Beitrag von Phase 6, nicht der einer
Baseline.

Die Patchgroesse 33 ist ungerade, damit es ein echtes Zentrumspixel gibt.
Zwei Halbierungen fuehren auf 33 -> 17 -> 9; die Rueckwege nutzen
``interpolate`` auf die gespeicherte Groesse statt fester Faktoren, weil
ungerade Kantenlaengen sonst um ein Pixel danebenliegen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


class SpatialHazardUNet(nn.Module):
    def __init__(self, in_ch: int, horizon: int = 30, width: int = 48,
                 init_bias: float = -7.0) -> None:
        super().__init__()
        w = width
        self.enc1 = block(in_ch, w)
        self.enc2 = block(w, 2 * w)
        self.enc3 = block(2 * w, 4 * w)
        self.dec2 = block(4 * w + 2 * w, 2 * w)
        self.dec1 = block(2 * w + w, w)
        self.head = nn.Conv2d(w, horizon, 1)
        # Startbias auf die Groessenordnung der Praevalenz: sigmoid(-7) ~ 9e-4.
        # Ohne das beginnt das Netz mit Hazard 0,5 und verbringt die ersten
        # Epochen damit, sich auf ein seltenes Ereignis herunterzuarbeiten.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        u2 = F.interpolate(s3, size=s2.shape[-2:], mode="nearest")
        d2 = self.dec2(torch.cat([u2, s2], 1))
        u1 = F.interpolate(d2, size=s1.shape[-2:], mode="nearest")
        d1 = self.dec1(torch.cat([u1, s1], 1))
        return self.head(d1)                      # (N, H, P, P)

    def centre_logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        c = out.shape[-1] // 2
        return out[:, :, c, c]

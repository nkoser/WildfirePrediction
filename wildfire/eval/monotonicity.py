"""Prueft an den trainierten Koepfen, was monoton ist und was nicht.

Die Bedingung im Bernstein-Kopf erzwingt Monotonie der KUMULIERTEN
Inzidenzfunktion F, nicht des Hazards. Das wird leicht verwechselt, deshalb
hier als Messung statt als Behauptung:

  * Hazard h(k) ist ein bedingtes Tagesrisiko und darf frei steigen und fallen.
    Im kanadischen Sommer-Setup MUSS er fallen - Landmarks liegen im Mai bis
    August, Tag 100 ist bereits Herbst.
  * F(k) darf niemals fallen. Das ist keine Modellannahme, sondern Logik:
    mehr Tage koennen nur Gelegenheiten hinzufuegen.

Ein Unit-Test deckt die Konstruktion ab (test_bernstein). Diese Auswertung
prueft zusaetzlich das TRAINIERTE Modell auf echten Landmarks - inklusive des
diskreten Kopfes, bei dem die Monotonie aus der Produktform folgt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.models.mlp_survival import MLPSurvival

EPS = 1e-12


def curves(model: MLPSurvival, X: torch.Tensor, horizon: int, head: str,
           chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """(N, H) Hazards und CIF fuer beide Koepfe, in float64."""
    with torch.no_grad():
        raw = torch.cat([model(X[i:i + chunk]) for i in range(0, len(X), chunk)])
        if head == "discrete":
            h = torch.sigmoid(raw).double()
            F = 1 - torch.cumprod(1 - h, dim=1)
        else:
            tau = torch.arange(1, horizon + 1, dtype=torch.float32)
            F = torch.cat([model.bern.cif(raw[i:i + chunk], tau)
                           for i in range(0, len(raw), chunk)]).double()
            prev = torch.cat([torch.zeros(len(F), 1, dtype=torch.float64),
                              F[:, :-1]], dim=1)
            h = (F - prev) / (1 - prev).clamp(min=EPS)
    return h.numpy(), F.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="temporal")
    ap.add_argument("--part", default="val")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    z = np.load(exp / "derived_data" / "features" / f"{args.split}_{args.part}.npz",
                allow_pickle=False)
    X_all = z["X"]

    out = {}
    for head, deg in (("discrete", None), ("bernstein", 12)):
        ckpt = exp / "checkpoints" / f"b4_{head}_best.pt"
        if not ckpt.exists():
            print(f"[mono] {head}: kein Checkpoint, uebersprungen")
            continue
        ck = torch.load(ckpt, weights_only=False)
        cols, mu, sd = ck["cols"], ck["mu"], ck["sd"]
        H = 100
        model = MLPSurvival(len(cols), H, 256, head=head, degree=deg or 12)
        model.load_state_dict(ck["model"])
        model.eval()
        X = torch.tensor((np.nan_to_num(X_all[:, cols]) - mu) / sd, dtype=torch.float32)
        h, F = curves(model, X, H, head)

        dh, dF = np.diff(h, axis=1), np.diff(F, axis=1)
        rec = {
            "n_landmarks": int(len(h)),
            "share_hazard_ever_falling": float(np.mean((dh < -EPS).any(1))),
            "share_cif_ever_falling": float(np.mean((dF < -EPS).any(1))),
            "mean_hazard_day1": float(h[:, 0].mean()),
            "mean_hazard_day50": float(h[:, 49].mean()),
            "mean_hazard_day100": float(h[:, 99].mean()),
            "mean_cif_day100": float(F[:, 99].mean()),
        }
        rec["mean_hazard_decline_factor"] = rec["mean_hazard_day1"] / max(
            rec["mean_hazard_day100"], EPS)
        out[head] = rec
        print(f"[mono] {head:9s} Hazard faellt irgendwo bei "
              f"{100*rec['share_hazard_ever_falling']:6.2f} % der Landmarks, "
              f"CIF faellt bei {100*rec['share_cif_ever_falling']:5.2f} % "
              f"(muss 0 sein)")
        print(f"[mono]           mittlerer Hazard Tag 1 {rec['mean_hazard_day1']:.3e} "
              f"-> Tag 100 {rec['mean_hazard_day100']:.3e} "
              f"(Faktor {rec['mean_hazard_decline_factor']:.1f})")

    bad = [k for k, v in out.items() if v["share_cif_ever_falling"] > 0]
    if bad:
        raise SystemExit(f"FEHLER: fallende CIF bei {bad}")

    write_manifest(
        cfg, "m5_monotonicity", args.config,
        {
            "split": args.split, "part": args.part,
            "claim": "F muss monoton steigen, h darf beliebig steigen und fallen",
            "heads": out,
        },
    )


if __name__ == "__main__":
    main()

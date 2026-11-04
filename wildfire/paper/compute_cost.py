"""Rechenaufwand der Hauptmodelle: Parameter, FLOPs und Inferenzzeit je 13 x 13-Ausschnitt.

  sts       STS-LH, Contagion-STS (scratch), lineare Contagion: je ein Checkpoint (Seed 0) ueber load_sts
            (ueber den Wrapper contagion_linear, damit die lineare Contagion laedt); Zeit = Vorwaertslauf plus
            diskrete Hazards von Pixel- und Ausschnittskopf (1 .. 100 Tage), wie in surv_eval
  rec       3D-U-Net des Paper-Nachbaus (reconstruct.UNet3D, 55 Kanaele, T 30); EIN Modell je Vorlaufzeit, fuer
            vier Vorlaufzeiten also das Vierfache
Eingaben sind Zufallszahlen in der Form der echten Fenster (die Laufzeit haengt nicht vom Inhalt ab).
Zeit: Mittel ueber --iters Batches nach --warmup Aufwaermlaeufen, bfloat16-Autocast wie in der Auswertung.
FLOPs per torch.utils.flop_counter (zaehlt Matmul/Conv; Attention-Softmax und Normierungen nicht).

  python -m wildfire.paper.contagion_linear wildfire.paper.compute_cost --config ...
Ausgabe: metrics/paper_horizon/compute_cost<tag>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import sts_dctm
from wildfire.paper.reconstruct import UNet3D

H = 100
CK = "/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_"
STS = {"lh": CK + "lh_s0_s0.pt", "cg_scratch": CK + "cg_scratch_s0_s0.pt", "lincg": CK + "lincg_s0_s0.pt"}


def n_params(net) -> dict:
    return {"total": int(sum(p.numel() for p in net.parameters())),
            "trainable": int(sum(p.numel() for p in net.parameters() if p.requires_grad))}


def timed(fn, warmup, iters, device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def flops(fn) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            fn()
        return float(fc.get_total_flops())
    except Exception as e:                                   # noqa: BLE001
        print(f"[cost] FLOP-Zaehlung fehlgeschlagen: {e}", flush=True)
        return None


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / "paper_horizon" / f"compute_cost{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    res: dict = {"device": gpu, "bs": args.bs}
    x = torch.randn(args.bs, 30, 55, 13, 13, device=device)
    doy = torch.full((args.bs,), 180.0, device=device)

    for name, path in STS.items():
        files = glob.glob(path)
        if not files:
            print(f"[cost] {name}: {path} fehlt", flush=True)
            continue
        net = sts_dctm.load_sts(files[0], device)

        def run(net=net):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw_px, raw_pa = net(x, doy)[:2]
            net.px_head.hazards(raw_px.reshape(-1, raw_px.shape[-1]), H)
            net.pa_head.hazards(raw_pa, H)

        sec = timed(run, args.warmup, args.iters, device)
        fl = flops(run)
        res[name] = {"checkpoint": files[0], "params": n_params(net), "flops_per_patch": fl / args.bs if fl else None,
                     "ms_per_batch": sec * 1e3, "ms_per_patch": sec * 1e3 / args.bs,
                     "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}
        print(f"[cost] {name:10s} Parameter {res[name]['params']['total']:,}  FLOPs/Ausschnitt "
              f"{(fl or 0) / args.bs / 1e9:.3f} G  {sec * 1e3 / args.bs:.3f} ms/Ausschnitt", flush=True)
        del net
        torch.cuda.empty_cache() if device.type == "cuda" else None

    unet = UNet3D(55).to(device).eval()
    xu = torch.randn(args.bs, 55, 30, 13, 13, device=device)

    def run_u():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            unet(xu)

    sec = timed(run_u, args.warmup, args.iters, device)
    fl = flops(run_u)
    res["rec_unet3d"] = {"params": n_params(unet), "flops_per_patch": fl / args.bs if fl else None,
                         "ms_per_patch": sec * 1e3 / args.bs, "note": "ein Modell je Vorlaufzeit; 4 Vorlaufzeiten = 4x"}
    print(f"[cost] rec U-Net  Parameter {res['rec_unet3d']['params']['total']:,}  FLOPs/Ausschnitt "
          f"{(fl or 0) / args.bs / 1e9:.3f} G  {sec * 1e3 / args.bs:.3f} ms/Ausschnitt (je Vorlaufzeit)", flush=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    write_manifest(cfg, f"p30_compute_cost{args.tag}", args.config, {"output": str(out), **res})


if __name__ == "__main__":
    main()

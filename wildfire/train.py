"""Konfigurationsgetriebenes Training fuer B2 und Nachfolger.

Der Arbeitsplan verlangt fuer jeden Lauf: Konfiguration, Git-Commit, Seed,
Datenversion, Splitversion, Umgebung, Kommando, Log und Metriken. Das
passiert hier - die Registry wird beim START geschrieben (Status
``running``) und am Ende aktualisiert, damit auch abgebrochene Laeufe
sichtbar bleiben statt spurlos zu verschwinden.

Checkpoints enthalten Modell, Optimizer, Scheduler, Epoche, Konfiguration
und die Hashes von Datenmanifest und Split - ebenfalls Vorgabe des Plans.

Early Stopping laeuft auf einer VORAB in der Konfiguration festgelegten
Metrik. Sie im Nachhinein zu waehlen waere eine stille Form der
Testkontamination.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wildfire.audit.common import load_config, write_manifest
from wildfire.eval.metrics import evaluate
from wildfire.experiment import register
from wildfire.models.losses import discrete_survival_nll
from wildfire.models.nets import SpatialHazardUNet
from wildfire.models.torch_data import ChunkedShuffleSampler, PatchDataset


def collate_eval(model, loader, device, horizon):
    model.eval()
    H, E, T, W = [], [], [], []
    with torch.no_grad():
        for b in loader:
            lo = model.centre_logits(b["x"].to(device, non_blocking=True))
            H.append(torch.sigmoid(lo).float().cpu().numpy())
            E.append(b["event"].numpy()); T.append(b["t_event"].numpy())
            W.append(b["weight"].numpy())
    return (np.concatenate(H), np.concatenate(E).astype(bool),
            np.concatenate(T), np.concatenate(W))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Audit-Config (Pfade)")
    ap.add_argument("--model-config", required=True, help="Modell-/Trainingsconfig")
    ap.add_argument("--smoke", action="store_true",
                    help="winziger Durchlauf zur Pipelinepruefung")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mc = load_config(args.model_config)
    exp = Path(cfg["experiment_root"])

    if args.smoke:
        mc.update(limit_train=256, limit_val=256, epochs=1, batch_size=16,
                  width=8, workers=0, run_id=str(mc["run_id"]) + "_smoke")

    torch.manual_seed(int(mc["seed"]))
    np.random.seed(int(mc["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] {mc['run_id']} auf {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

    common = dict(patch=int(mc["patch"]), lookback=int(mc["lookback"]),
                  cooldown=int(mc["cooldown"]), n_slices=int(mc["n_slices"]),
                  seed=int(mc["seed"]))
    ds_tr = PatchDataset(exp, mc["split"], "train", limit=mc.get("limit_train"), **common)
    ds_va = PatchDataset(exp, mc["split"], "val", limit=mc.get("limit_val"), **common)
    print(f"[train] train={len(ds_tr)} val={len(ds_va)} Eingabekanaele={ds_tr.in_channels}")

    dl = dict(batch_size=int(mc["batch_size"]), num_workers=int(mc["workers"]),
              pin_memory=device.type == "cuda", persistent_workers=int(mc["workers"]) > 0)
    # Gemessen auf dem Wuerfel: zufaelliges Mischen kostet Faktor 5 gegenueber
    # blockweisem (42,5 gegen 220,8 Beispiele/s je Prozess). shuffle_chunk=0
    # schaltet auf das alte Verhalten zurueck.
    chunk = int(mc.get("shuffle_chunk", 0))
    if chunk > 0:
        sampler = ChunkedShuffleSampler(ds_tr.day_order(), chunk, int(mc["seed"]))
        tr = DataLoader(ds_tr, sampler=sampler, drop_last=True, **dl)
        print(f"[train] chunked shuffle, Bloecke à {chunk:,}", flush=True)
    else:
        sampler = None
        tr = DataLoader(ds_tr, shuffle=True, drop_last=True, **dl)
    va = DataLoader(ds_va, shuffle=False, **dl)

    model = SpatialHazardUNet(ds_tr.in_channels, horizon=int(mc["horizon"]),
                              width=int(mc["width"]),
                              init_bias=float(mc["init_bias"])).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[train] {n_par/1e6:.2f} Mio. Parameter")

    opt = torch.optim.AdamW(model.parameters(), lr=float(mc["lr"]),
                            weight_decay=float(mc["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(mc["epochs"])))
    use_amp = bool(mc.get("amp")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = exp / "checkpoints" / str(mc["run_id"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reg = register(exp, str(mc["run_id"]), "model", args.model_config,
                   split=mc["split"], seed=int(mc["seed"]), status="running",
                   checkpoint_path=str(ckpt_dir),
                   notes=f"{n_par} Parameter, {ds_tr.in_channels} Eingabekanaele")

    best, bad, history = float("inf"), 0, []

    def save_manifest(finished: bool) -> None:
        """Zwischenstand sichern - NICHT erst am Ende.

        Bei 2,6 Stunden je Epoche und 12 Stunden Zeitlimit wird ein voller Lauf
        vom Scheduler abgeschnitten. Die Checkpoints ueberleben das (sie werden
        je Epoche geschrieben), das Manifest bislang nicht - damit waeren die
        Metriken nur noch im Log gewesen und haetten von Hand uebertragen
        werden muessen. Genau das soll nicht passieren.
        """
        write_manifest(cfg, f"m2_{mc['run_id']}", args.model_config,
                       {"model_config": mc, "n_parameters": n_par,
                        "in_channels": ds_tr.in_channels, "device": str(device),
                        "n_train": len(ds_tr), "n_val": len(ds_va),
                        "history": history, "best": best,
                        "completed": finished,
                        "epochs_planned": int(mc["epochs"]),
                        "checkpoint": str(ckpt_dir / "best.pt")})

    for ep in range(int(mc["epochs"])):
        if sampler is not None:
            sampler.set_epoch(ep)          # jede Epoche anders mischen
        model.train()
        t0, run, seen = time.perf_counter(), 0.0, 0
        for i, b in enumerate(tr):
            x = b["x"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # Das Ziehungsgewicht MUSS mit. Ohne es trainiert B2 auf der
                # angereicherten Stichprobe (9,15 % Ereignisrate), als waere das
                # die Wirklichkeit (1,27 Promille). Der Logit liegt dann um
                # +4,374 zu hoch, die Hazards um Faktor 79 - die Rangordnung
                # bleibt davon unberuehrt (AUC sah normal aus), die
                # Kalibrierung nicht: val NLL 0,0697 statt der 0,0125 von B4.
                # B4 hat das von Anfang an richtig gemacht, B2 nicht; damit war
                # die Ablation zwischen beiden wertlos.
                loss = discrete_survival_nll(
                    model.centre_logits(x), b["event"].to(device),
                    b["t_event"].to(device), b["weight"].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            run += loss.item() * x.shape[0]; seen += x.shape[0]
            if i % 50 == 0:
                print(f"    Epoche {ep} Schritt {i}/{len(tr)} "
                      f"Verlust {loss.item():.5f} |g|={gn:.2f}", flush=True)
        sched.step()

        h, e, t_, w = collate_eval(model, va, device, int(mc["horizon"]))
        m = evaluate(h, e, t_, w)
        hz = m["horizons"]["30"]
        rec = {"epoch": ep, "train_loss": run / max(1, seen),
               "val_nll": m["nll"], "val_ibs": m["integrated_brier"],
               "val_auc30": hz["roc_auc"], "val_pr30": hz["pr_auc"],
               "val_ece30": hz["ece"], "seconds": time.perf_counter() - t0,
               "lr": sched.get_last_lr()[0]}
        history.append(rec)
        print(f"[train] Epoche {ep}: Verlust {rec['train_loss']:.5f} | "
              f"val NLL {m['nll']:.5f} AUC@30 {hz['roc_auc']:.4f} "
              f"PR@30 {hz['pr_auc']:.4f} | {rec['seconds']:.0f}s", flush=True)

        score = m[mc["early_stopping"]["metric"]]
        state = {"epoch": ep, "model": model.state_dict(),
                 "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                 "config": mc, "history": history,
                 "dataset_manifest_hash": reg["dataset_manifest_hash"],
                 "split_hash": reg["split_hash"], "git_commit": reg["git_commit"]}
        torch.save(state, ckpt_dir / "last.pt")
        if score < best:
            best, bad = score, 0
            torch.save(state, ckpt_dir / "best.pt")
            print(f"    neuer bester Wert ({mc['early_stopping']['metric']}={score:.5f})")
        else:
            bad += 1
            if bad >= int(mc["early_stopping"]["patience"]):
                save_manifest(finished=True)
                print(f"[train] Early Stopping nach Epoche {ep}")
                break
        save_manifest(finished=False)

    register(exp, str(mc["run_id"]), "model", args.model_config,
             split=mc["split"], seed=int(mc["seed"]), status="completed",
             checkpoint_path=str(ckpt_dir / "best.pt"),
             metrics_path=str(exp / "data_manifests" / f"m2_{mc['run_id']}.json"),
             notes=f"beste {mc['early_stopping']['metric']}={best:.6f} nach {len(history)} Epochen",
             extra={"best_val": best, "epochs_run": len(history)})
    save_manifest(finished=True)


if __name__ == "__main__":
    main()

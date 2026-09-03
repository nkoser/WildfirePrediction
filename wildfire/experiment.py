"""Registrierung jedes Laufs, wie der Arbeitsplan es unter
"Rueckverfolgbarkeit jedes Experiments" verlangt.

Eine Zeile je Lauf in ``experiment_registry.jsonl``. JSONL statt CSV, weil
Felder dazukommen werden und ein Anhaengen nie eine bestehende Zeile
umschreibt - die Registry ist damit auch bei parallelen Laeufen sicher.

Die Registry wird beim Lauf geschrieben, nicht hinterher rekonstruiert.
Alles andere waere eine Erzaehlung ueber Experimente statt ihrer
Aufzeichnung.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def _sha16(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _gpu() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def register(exp_root: str | Path, run_id: str, kind: str, config_path: str | Path,
             *, split: str | None = None, seed: int | None = None,
             status: str = "completed", notes: str = "",
             metrics_path: str | Path | None = None,
             checkpoint_path: str | Path | None = None,
             predictions_path: str | Path | None = None,
             extra: dict | None = None) -> dict:
    exp = Path(exp_root)
    cfg = Path(config_path)
    row = {
        "run_id": run_id,
        "kind": kind,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_state": bool(_git("status", "--porcelain")),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "config_path": str(cfg),
        "config_hash": _sha16(cfg),
        "dataset_manifest_hash": _sha16(exp / "data_manifests" / "dataset_manifest.json"),
        "split": split,
        "split_hash": _sha16(exp / "splits" / f"{split}.csv") if split else None,
        "seed": seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu": _gpu(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "predictions_path": str(predictions_path) if predictions_path else None,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "status": status,
        "notes": notes,
    }
    if extra:
        row["extra"] = extra
    path = exp / "experiment_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    print(f"[registry] {run_id} ({kind}) -> {path}")
    return row


def show(exp_root: str | Path) -> None:
    path = Path(exp_root) / "experiment_registry.jsonl"
    if not path.exists():
        print("[registry] noch leer")
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"[registry] {len(rows)} Laeufe\n")
    print(f"{'run_id':32s} {'kind':12s} {'status':10s} {'commit':10s} {'dirty':6s} {'split':14s}")
    for r in rows:
        print(f"{r['run_id']:32s} {r['kind']:12s} {r['status']:10s} "
              f"{(r['git_commit'] or '')[:8]:10s} {str(r['git_dirty_state']):6s} "
              f"{str(r.get('split')):14s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default="/data/Wildfire")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        show(args.exp_root)


if __name__ == "__main__":
    main()

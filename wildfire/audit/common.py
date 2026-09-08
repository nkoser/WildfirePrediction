"""Gemeinsame Hilfsfunktionen fuer alle Audit-Stufen.

Jede Stufe schreibt genau eine JSON-Datei nach
``<experiment_root>/data_manifests/`` und stellt ihr einen
Provenance-Block voran (Git-Commit, Config-Hash, Seed, Slurm-Job-ID,
Host, Zeitpunkt). Ergebnisse werden nie von Hand aus Logs uebertragen.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - Git kann fehlen
        return ""


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Laedt die YAML-Config.

    Faellt auf einen winzigen Parser zurueck, falls PyYAML fehlt - das
    Audit soll nicht an einer Abhaengigkeit scheitern.
    """
    text = Path(path).read_text()
    try:
        import yaml

        cfg = yaml.safe_load(text)
    except ImportError:
        cfg = _mini_yaml(text)

    # Der Rohdatenpfad haengt davon ab, WO gerechnet wird: In dieser Session
    # liegt er unter /firedata, in einem pyxis-Container muss er erst gemountet
    # werden - und zwar an eine Stelle, die es im Image sicher gibt. Statt zwei
    # Configs zu pflegen, ueberschreibt eine Umgebungsvariable den Wert. Sie
    # geht ueber write_manifest in die Provenienz ein, der Lauf bleibt also
    # nachvollziehbar.
    override = os.environ.get("WF_DATASET_ROOT")
    if override:
        cfg["dataset_root"] = override
    return cfg


def _mini_yaml(text: str) -> dict[str, Any]:
    """Minimalparser fuer die flache Struktur von audit.yaml."""

    def scalar(tok: str) -> Any:
        tok = tok.strip()
        if tok in ("true", "false"):
            return tok == "true"
        if tok.startswith("[") and tok.endswith("]"):
            inner = tok[1:-1].strip()
            return [scalar(p) for p in inner.split(",")] if inner else []
        try:
            return int(tok)
        except ValueError:
            pass
        try:
            return float(tok)
        except ValueError:
            pass
        return tok

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if body.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Listeneintrag ohne Liste: {raw!r}")
            parent.append(scalar(body[2:]))
            continue
        key, _, value = body.partition(":")
        key = key.strip()
        if value.strip():
            parent[key] = scalar(value)
        else:
            # Naechste Zeile entscheidet, ob Mapping oder Liste.
            child: Any = {}
            parent[key] = child
            stack.append((indent, child))
    return _fix_empty_lists(root, text)


def _fix_empty_lists(node: Any, text: str) -> Any:
    """Wandelt Mappings in Listen, wenn die Quelle '- ' Eintraege hatte."""
    # Der Miniparser legt fuer Listen zunaechst ein dict an; sammle die
    # Listen anhand der Quelle nach.
    lines = [l.split("#", 1)[0].rstrip() for l in text.splitlines()]
    list_keys = set()
    for i, line in enumerate(lines):
        if not line.strip() or line.strip().startswith("- "):
            continue
        key = line.strip().split(":", 1)[0]
        for nxt in lines[i + 1 :]:
            if not nxt.strip():
                continue
            if nxt.strip().startswith("- "):
                list_keys.add(key)
            break

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: ([] if (k in list_keys and obj[k] == {}) else walk(v))
                for k, v in obj.items()
            }
        return obj

    return walk(node)


def config_hash(path: str | os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def provenance(config_path: str | os.PathLike, stage: str,
               cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "config_path": str(config_path),
        "config_hash": config_hash(config_path),
        # Der Rohdatenpfad kann per WF_DATASET_ROOT umgesetzt werden, je
        # nachdem ob interaktiv oder im pyxis-Container gerechnet wird. Der
        # tatsaechlich benutzte Wert gehoert deshalb in die Provenienz -
        # sonst laesst sich ein Lauf nicht rekonstruieren.
        "dataset_root_used": (cfg or {}).get("dataset_root"),
        "dataset_root_override": os.environ.get("WF_DATASET_ROOT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive-cpu"),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "argv": sys.argv,
    }


def write_manifest(
    cfg: dict[str, Any], stage: str, config_path: str | os.PathLike, payload: dict
) -> Path:
    out_dir = Path(cfg["experiment_root"]) / "data_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stage}.json"
    doc = {"provenance": provenance(config_path, stage, cfg), **payload}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False, default=str))
    tmp.replace(out)
    print(f"[manifest] {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return out


def read_manifest(cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    path = Path(cfg["experiment_root"]) / "data_manifests" / f"{stage}.json"
    return json.loads(path.read_text())


def landmark_horizon(cfg: dict, default: int = 100) -> int:
    """Prognosehorizont aus dem Landmark-Manifest.

    EINE Quelle statt sechs Vorgabewerte. Der Horizont wird beim Bauen der
    Landmarks festgelegt; jedes Modell, das auf ihnen rechnet, muss
    denselben verwenden. Als der Horizont von 30 auf 100 wechselte, hatte
    ich zwei von sechs hartkodierten Stellen uebersehen - B1 stuerzte mit
    einem Indexfehler ab, weil t_event bis 100 lief, die Hazardmatrix aber
    nur 30 Spalten hatte. Diese Fehlerklasse ist damit ausgeschlossen.
    """
    path = Path(cfg["experiment_root"]) / "data_manifests" / "d7_landmarks.json"
    if not path.exists():
        return default
    try:
        return int(json.loads(path.read_text())["horizon"])
    except (KeyError, ValueError):
        return default


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} EB"


def assert_readonly(cfg: dict[str, Any]) -> None:
    """Harte Zusicherung: kein Audit-Schritt darf in die Rohdaten schreiben.

    Wir pruefen nur, dass das Ausgabeziel ausserhalb des Rohdatenbaums
    liegt - schreibende Aufrufe gibt es im Audit-Code bewusst nicht.
    """
    raw = Path(cfg["raw_data"]).resolve()
    exp = Path(cfg["experiment_root"]).resolve()
    if raw == exp or raw in exp.parents:
        raise SystemExit(
            f"ABBRUCH: experiment_root {exp} liegt im Rohdatenbaum {raw}."
        )

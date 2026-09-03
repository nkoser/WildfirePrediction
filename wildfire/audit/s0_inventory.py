"""Stufe 0: abhaengigkeitsfreies Dateiinventar des Rohdatenpfades.

Laeuft mit reiner Standardbibliothek, damit der erste Blick in die Daten
nicht an fehlenden Geo-Paketen scheitern kann. Es wird ausschliesslich
gelesen: os.scandir + stat + die ersten 16 Bytes jeder Datei.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

from wildfire.audit.common import (
    assert_readonly,
    human_bytes,
    load_config,
    write_manifest,
)

MAGIC = {
    b"\x89HDF\r\n\x1a\n": "HDF5",
    b"II*\x00": "TIFF (little endian)",
    b"MM\x00*": "TIFF (big endian)",
    b"PK\x03\x04": "ZIP/NPZ",
    b"\x93NUMPY": "NPY",
    b"\x1f\x8b": "GZIP",
    b"\x89PNG": "PNG",
}


def sniff(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            head = fh.read(16)
    except OSError as exc:
        return f"unlesbar: {exc.__class__.__name__}"
    for magic, name in MAGIC.items():
        if head.startswith(magic):
            return name
    # tar hat kein Magic am Dateianfang, sondern "ustar" bei Offset 257.
    try:
        with path.open("rb") as fh:
            fh.seek(257)
            if fh.read(5) == b"ustar":
                return "TAR (ustar)"
    except OSError:
        pass
    if head[:1] in (b"{", b"["):
        return "JSON/Text"
    return "unbekannt"


def walk(root: Path) -> list[dict]:
    entries: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append(
                {
                    "path": str(p.relative_to(root)),
                    "size_bytes": st.st_size,
                    "mtime_utc": st.st_mtime,
                    "suffix": p.suffix.lower(),
                    "format": sniff(p),
                }
            )
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    root = Path(cfg["raw_data"])
    print(f"[s0] scanne {root} (read-only)")
    files = walk(root)

    by_dir: dict[str, dict] = defaultdict(
        lambda: {"n_files": 0, "size_bytes": 0, "formats": defaultdict(int)}
    )
    by_format: dict[str, dict] = defaultdict(lambda: {"n_files": 0, "size_bytes": 0})
    for f in files:
        top = f["path"].split(os.sep)
        key = os.sep.join(top[:2]) if len(top) > 2 else (top[0] if len(top) > 1 else ".")
        d = by_dir[key]
        d["n_files"] += 1
        d["size_bytes"] += f["size_bytes"]
        d["formats"][f["format"]] += 1
        b = by_format[f["format"]]
        b["n_files"] += 1
        b["size_bytes"] += f["size_bytes"]

    total = sum(f["size_bytes"] for f in files)
    print(f"[s0] {len(files)} Dateien, {human_bytes(total)}")
    for key in sorted(by_dir):
        d = by_dir[key]
        print(f"    {key:45s} {d['n_files']:5d} Dateien  {human_bytes(d['size_bytes']):>10s}")

    write_manifest(
        cfg,
        "s0_inventory",
        args.config,
        {
            "root": str(root),
            "n_files": len(files),
            "total_size_bytes": total,
            "total_size_human": human_bytes(total),
            "by_directory": {
                k: {**v, "formats": dict(v["formats"]), "size_human": human_bytes(v["size_bytes"])}
                for k, v in sorted(by_dir.items())
            },
            "by_format": {
                k: {**v, "size_human": human_bytes(v["size_bytes"])}
                for k, v in sorted(by_format.items())
            },
            "files": files,
        },
    )


if __name__ == "__main__":
    main()

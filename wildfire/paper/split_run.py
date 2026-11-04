"""Startet ein vorhandenes paper-Modul unveraendert, aber mit Split-Suffix an jedem Manifest-Namen.

Viele Module schreiben ihr Manifest unter festem Namen (rate_labels p23_rate_labels, future_weather
p26_future_weather, ring_targets p57_ring_targets, future_maps p59_future_maps, ...) oder lesen feste Pfade des
zeitlichen Splits (region_eval/pixel_events.npz, cause_labels/eval_cause.npz, ...). Auf ys/bs wuerden sie damit
das Manifest des zeitlichen Splits ersetzen bzw. still dessen Daten lesen. Hier wird das Modul importiert und
  - sein write_manifest durch eine Fassung mit angehaengtem --manifest-suffix ersetzt (bricht ab, wenn das
    Ziel-Manifest schon existiert),
  - mit --redirect ALT=NEU jeder Pfad, der auf ALT endet, in np.load / np.save / np.savez / np.savez_compressed
    des Moduls und in Path.write_text auf NEU umgelenkt (NEU muss beim Lesen existieren; jede Umlenkung muss mindestens einmal greifen,
    sonst Abbruch am Ende),
  - mit --set NAME=LITERAL eine Modulkonstante ersetzt (Python-Literal, z. B. TRAIN_YEARS=(2000,2019)),
dann main() mit den restlichen Argumenten aufgerufen.

  python -m wildfire.paper.split_run --manifest-suffix _ys \
      --redirect region_eval/pixel_events.npz=region_eval_ys/pixel_events.npz \
      wildfire.paper.cause_labels --config C --part eval --surv-dir surv_eval_ys --tag _ys
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
import types
from pathlib import Path

import numpy

import wildfire.audit.common as common


class _NpProxy(types.ModuleType):
    """numpy mit umgelenkten Pfaden fuer load/save/savez; alles andere unveraendert."""

    def __init__(self, redirects: dict[str, str], hits: dict[str, int]):
        super().__init__("numpy")
        self._red, self._hits = redirects, hits

    def __getattr__(self, name):
        return getattr(numpy, name)

    def _map(self, p, reading: bool):
        if not isinstance(p, (str, Path)):
            return p
        s = str(p)
        for old, new in self._red.items():
            if s.endswith(old):
                q = Path(s[: -len(old)] + new)
                if reading and not q.exists():
                    raise SystemExit(f"ABBRUCH: umgelenkter Pfad {q} fehlt")
                self._hits[old] += 1
                print(f"[split_run] {s} -> {q}", flush=True)
                return q
        return p

    def load(self, p, *a, **k):
        return numpy.load(self._map(p, True), *a, **k)

    def save(self, p, *a, **k):
        return numpy.save(self._map(p, False), *a, **k)

    def savez(self, p, *a, **k):
        return numpy.savez(self._map(p, False), *a, **k)

    def savez_compressed(self, p, *a, **k):
        return numpy.savez_compressed(self._map(p, False), *a, **k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-suffix", required=True)
    ap.add_argument("--redirect", action="append", default=[], metavar="ALT=NEU")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=LITERAL")
    ap.add_argument("module")
    args, rest = ap.parse_known_args()
    if not args.manifest_suffix.startswith("_"):
        raise SystemExit("ABBRUCH: --manifest-suffix muss mit _ beginnen")
    mod = importlib.import_module(args.module)
    orig = common.write_manifest

    def write_manifest(cfg, stage, config_path, payload):
        stage = f"{stage}{args.manifest_suffix}"
        if (Path(cfg["experiment_root"]) / "data_manifests" / f"{stage}.json").exists():
            raise SystemExit(f"ABBRUCH: Manifest {stage} existiert schon")
        return orig(cfg, stage, config_path, payload)

    if hasattr(mod, "write_manifest"):
        mod.write_manifest = write_manifest
    red = dict(r.split("=", 1) for r in args.redirect)
    hits = {k: 0 for k in red}
    if red:
        proxy = _NpProxy(red, hits)
        if hasattr(mod, "np"):
            mod.np = proxy
        orig_write_text = Path.write_text

        def write_text(self, *a, **k):                              # z. B. risk_map_plot: capture_stats.json
            return orig_write_text(Path(proxy._map(self, False)), *a, **k)

        Path.write_text = write_text
    for s in args.set:
        name, val = s.split("=", 1)
        if not hasattr(mod, name):
            raise SystemExit(f"ABBRUCH: {args.module} hat keine Konstante {name}")
        setattr(mod, name, ast.literal_eval(val))
        print(f"[split_run] {args.module}.{name} = {getattr(mod, name)!r}", flush=True)
    sys.argv = [args.module] + rest
    mod.main()
    miss = [k for k, v in hits.items() if v == 0]
    if miss:
        raise SystemExit(f"ABBRUCH: Umlenkung nie benutzt: {miss}")


if __name__ == "__main__":
    main()

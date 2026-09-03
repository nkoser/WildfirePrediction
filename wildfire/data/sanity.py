"""Sanity Checks aus Phase 4.2 des Arbeitsplans.

Die Unit-Tests arbeiten mit kuenstlichen Daten, in die ich die erwartete
Antwort selbst hineingeschrieben habe. Sie koennen deshalb prinzipiell
nicht ausschliessen, dass die Pipeline an ECHTEN Daten etwas anderes tut.
Hier wird an zufaellig gezogenen, echten Landmarks geprueft:

  1. **Labelkette unabhaengig nachgerechnet.** Fuer jedes Sample wird die
     Zuendungsdefinition direkt aus den tar-Archiven rekonstruiert -
     Feuer am Tag T, davor sieben feuerfreie Tage in der 3x3-Umgebung -
     und mit ``event``/``t_event`` aus der Landmark-Tabelle verglichen.
  2. **Zeitausrichtung.** Der letzte Zeitschritt des Eingabefensters muss
     byte-genau dem Rohraster des Landmarktages entsprechen. Ein Versatz
     um einen Tag bliebe in jeder Metrik unsichtbar.
  3. **Kanalsperre am echten Wuerfel**, nicht am Testfixture.
  4. **Verteilungen ueber die Splits**: Ereignis- und Zensierungsanteile,
     t_event, Jahresabdeckung.
  5. **NaN, konstante Kanaele, Randbehandlung.**

Zwei Punkte aus Phase 4.2 fehlen hier, weil sie ein Modell voraussetzen:
das gezielte Overfitten eines Kleinstdatensatzes und die Monotonie der
Survival-Kurven. Beides kommt mit der ersten Baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile
from rasterio.windows import Window

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import LOCKED_CHANNELS, LandmarkDataset

COOLDOWN = 7
RADIUS = 1


def read_fire_window(root: Path, date: dt.date, r: int, c: int, rad: int, grid):
    """(2rad+1)^2 Feuerausschnitt eines Tages, direkt aus dem tar."""
    tar = root / "Entire_Canada_Maps" / "MCD14A1" / f"MCD14A1__{date.year}.tar"
    target = f"{date.year}_{date.month:02d}_{date.day:02d}.tif"
    if not tar.exists():
        return None
    with tarfile.open(tar, "r:") as tf:
        for m in tf:
            if m.isfile() and Path(m.name).name == target:
                blob = tf.extractfile(m).read()
                break
        else:
            return None
    P = 2 * rad + 1
    out = np.zeros((P, P), np.uint8)
    with MemoryFile(blob) as mem, mem.open() as ds:
        r0, c0 = max(0, r - rad), max(0, c - rad)
        r1, c1 = min(grid[0], r + rad + 1), min(grid[1], c + rad + 1)
        if r1 > r0 and c1 > c0:
            sub = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            out[r0 - (r - rad):r1 - (r - rad), c0 - (c - rad):c1 - (c - rad)] = sub
    return out


def read_raw_day_full(root: Path, modality: str, date: dt.date):
    tar = root / "Entire_Canada_Maps" / modality / f"{modality}__{date.year}.tar"
    target = f"{date.year}_{date.month:02d}_{date.day:02d}.tif"
    with tarfile.open(tar, "r:") as tf:
        for m in tf:
            if m.isfile() and Path(m.name).name == target:
                blob = tf.extractfile(m).read()
                with MemoryFile(blob) as mem, mem.open() as ds:
                    return ds.read()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="temporal")
    ap.add_argument("--n-label-checks", type=int, default=40)
    ap.add_argument("--n-align-checks", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    exp = Path(cfg["experiment_root"])
    root = Path(cfg["dataset_root"])
    H_ = args.horizon
    results = {"label_mismatches": [], "align_mismatches": [], "lock_violations": []}

    # ------------------------------------------------ Verteilungen je Split
    dist = {}
    for part in ("train", "val", "test"):
        f = exp / "derived_data" / "landmarks" / f"{args.split}_{part}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        ev, tev, w = z["event"], z["t_event"], z["weight"]
        pos = ev == 1
        dist[part] = {
            "n": int(ev.size), "n_event": int(pos.sum()),
            "censoring_rate": float(1 - pos.mean()),
            "prevalence_weighted": float((w * ev).sum() / w.sum()),
            "t_event_p50": float(np.median(tev[pos])) if pos.any() else None,
            "t_event_p90": float(np.percentile(tev[pos], 90)) if pos.any() else None,
            "all_censored_at_H": bool(np.all(tev[~pos] == H_)),
            "t_event_in_range": bool(tev.min() >= 1 and tev.max() <= H_),
            "years": {str(int(y)): int(n) for y, n in
                      zip(*np.unique(z["year"], return_counts=True))},
        }
        print(f"[sanity] {part:5s} n={ev.size:9d} Ereignisse={int(pos.sum()):7d} "
              f"Zensierung={100*(1-pos.mean()):5.1f} % "
              f"t_event Median={dist[part]['t_event_p50']} "
              f"in [1,H]={dist[part]['t_event_in_range']}")

    # ------------------------------------------------ Dataset auf echten Daten
    ds = LandmarkDataset(exp, args.split, "test")
    print(f"[sanity] {len(ds)} nutzbare Landmarks (Wuerfeljahre vorhanden)")
    if len(ds) == 0:
        raise SystemExit("keine nutzbaren Landmarks - ist der Wuerfel gebaut?")
    grid = (ds.cube.H, ds.cube.W)

    # ------------------------------------------------ 1. Labelkette
    idx = rng.choice(len(ds), size=min(args.n_label_checks, len(ds)), replace=False)
    n_ok = 0
    for n, i in enumerate(idx.tolist()):
        s = ds[i]
        D = dt.date.fromisoformat(s["date"])
        # unabhaengige Rekonstruktion: erste Zuendung in (t, t+H]
        recon_t = None
        for k in range(1, H_ + 1):
            day = D + dt.timedelta(days=k)
            here = read_fire_window(root, day, s["row"], s["col"], 0, grid)
            if here is None or here[0, 0] != 1:
                continue
            # Cooldown: 3x3-Umgebung in den sieben Tagen davor feuerfrei?
            blocked = False
            for back in range(1, COOLDOWN + 1):
                nb = read_fire_window(root, day - dt.timedelta(days=back),
                                      s["row"], s["col"], RADIUS, grid)
                if nb is not None and nb.any():
                    blocked = True
                    break
            if not blocked:
                recon_t = k
                break
        exp_event = 1 if recon_t is not None else 0
        exp_tev = recon_t if recon_t is not None else H_
        if (s["event"], s["t_event"]) == (exp_event, exp_tev):
            n_ok += 1
        else:
            results["label_mismatches"].append({
                "date": s["date"], "row": s["row"], "col": s["col"],
                "pipeline": [s["event"], s["t_event"]],
                "recomputed": [exp_event, exp_tev]})
        if n % 10 == 0:
            print(f"    Label {n+1}/{len(idx)} ...", flush=True)
    print(f"[sanity] Labelkette: {n_ok}/{len(idx)} stimmen mit der Rohdaten-"
          f"Rekonstruktion ueberein")

    # ------------------------------------------------ 2./3. Ausrichtung + Sperre
    dyn_ch = json.loads((exp / "data_manifests" / "d4_cube.json").read_text())["dynamic_channels"]
    mod_of = {c["cube_index"]: (c["modality"], c["band"]) for c in dyn_ch}
    aidx = rng.choice(len(ds), size=min(args.n_align_checks, len(ds)), replace=False)
    n_align = 0
    for i in aidx.tolist():
        s = ds[i]
        D = dt.date.fromisoformat(s["date"])
        half = ds.half
        open_ch = [c for c in range(s["x_dyn"].shape[0]) if c not in LOCKED_CHANNELS]
        ch = int(rng.choice(open_ch))
        mod, band = mod_of[ch]
        raw = read_raw_day_full(root, mod, D)
        r0, c0 = s["row"] - half, s["col"] - half
        r1, c1 = s["row"] + half + 1, s["col"] + half + 1
        if r0 < 0 or c0 < 0 or r1 > grid[0] or c1 > grid[1]:
            continue
        got = s["x_dyn"][ch, -1]                 # letzter Zeitschritt
        want = raw[band, r0:r1, c0:c1]
        if np.array_equal(got, want):
            n_align += 1
        else:
            results["align_mismatches"].append({
                "date": s["date"], "channel": ch, "modality": mod, "band": band,
                "n_differing": int((got != want).sum())})
        for lc in LOCKED_CHANNELS:
            v = s["x_dyn"][lc, -ds.C:]
            o = s["obs"][lc, -ds.C:]
            if v.any() or o.any():
                results["lock_violations"].append(
                    {"date": s["date"], "channel": lc,
                     "nonzero_values": int((v != 0).sum()), "obs_set": int(o.sum())})
    print(f"[sanity] Zeitausrichtung: {n_align}/{len(aidx)} letzte Fensterschritte "
          f"byte-genau gleich dem Rohraster")
    print(f"[sanity] Kanalsperre: {len(results['lock_violations'])} Verletzungen")

    # ------------------------------------------------ 5. NaN / konstante Kanaele
    probe = rng.choice(len(ds), size=min(64, len(ds)), replace=False)
    const_ch: Counter = Counter()
    n_nan = 0
    for i in probe.tolist():
        x = ds[i]["x_dyn"]
        n_nan += int(np.isnan(x.astype(np.float32)).sum())
        for c in range(x.shape[0]):
            if x[c].min() == x[c].max():
                const_ch[c] += 1
    always_const = [c for c, n in const_ch.items() if n == len(probe)]
    print(f"[sanity] NaN in {len(probe)} Patches: {n_nan}")
    print(f"[sanity] in ALLEN Patches konstante Kanaele: "
          f"{[(c, mod_of[c][0]) for c in sorted(always_const)] or 'keine'}")

    passed = (not results["label_mismatches"] and not results["align_mismatches"]
              and not results["lock_violations"] and n_nan == 0)
    print(f"\n[sanity] Gesamt: {'BESTANDEN' if passed else 'FEHLGESCHLAGEN'}")
    for key in ("label_mismatches", "align_mismatches", "lock_violations"):
        for m in results[key][:5]:
            print(f"    {key}: {m}")

    write_manifest(
        cfg, "d10_sanity", args.config,
        {
            "split": args.split, "horizon": H_,
            "cooldown": COOLDOWN, "radius": RADIUS,
            "n_usable_landmarks": len(ds),
            "n_label_checks": len(idx), "n_label_ok": n_ok,
            "n_align_checks": len(aidx), "n_align_ok": n_align,
            "n_nan_in_patches": n_nan,
            "always_constant_channels": [{"channel": c, "modality": mod_of[c][0]}
                                          for c in sorted(always_const)],
            "split_distributions": dist,
            "passed": passed,
            **results,
        },
    )
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()

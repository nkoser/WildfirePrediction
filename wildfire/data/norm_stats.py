"""Normalisierungsstatistik, ausschliesslich aus Trainingsjahren.

Der Arbeitsplan nennt das unter den Leakage-Pruefungen: "Normalisierungs-
statistiken ausschliesslich aus dem Training". Wuerde man Mittelwert und
Streuung ueber alle Jahre bilden, floesse Information aus Validierung und
Test in jedes Modell ein - unauffaellig, aber wirksam.

Zwei Kanaele werden AUSGENOMMEN und roh durchgereicht:

  * die Feuermaske (Kanal 0) - sie ist binaer, ein z-Wert waere sinnlos;
  * die Landbedeckung (statischer Kanal 12) - sie ist KATEGORIAL. Klasse 8
    liegt nicht "zwischen" 7 und 9; ein z-Wert wuerde eine Ordnung
    erfinden, die es nicht gibt. Sie gehoert spaeter als Embedding oder
    One-Hot ins Modell.

Geschaetzt wird auf einer Stichprobe von Tagen ueber die Trainingsjahre,
nur auf gueltigen Pixeln. Bei 26 Jahren x 183 Tagen x 1,8 Mio. Pixeln waere
eine Vollauswertung unnoetig - der Standardfehler des Mittelwerts ist schon
bei wenigen Millionen Werten vernachlaessigbar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.data.splits import SPLIT_DEFS

EXCLUDE_DYNAMIC = (0,)        # Feuermaske: binaer
EXCLUDE_STATIC = (12,)        # Landbedeckung: kategorial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="temporal")
    ap.add_argument("--days-per-year", type=int, default=6)
    ap.add_argument("--pixels-per-day", type=int, default=40000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    exp = Path(cfg["experiment_root"])
    cube = CubeReader(exp)
    valid = np.load(exp / "derived_data" / "masks" / "valid.npy")
    vr, vc = np.nonzero(valid)
    train_years = set(SPLIT_DEFS[args.split]["train"]["years"])

    years = sorted(y for y in (int(k) for k in cube.per_year) if y in train_years)
    print(f"[norm] Trainingsjahre im Wuerfel: {years[0]}..{years[-1]} ({len(years)})")

    n_dyn, n_stat = cube.n_dyn, cube.n_stat
    cnt = np.zeros(n_dyn, np.float64)
    s1 = np.zeros(n_dyn, np.float64)
    s2 = np.zeros(n_dyn, np.float64)
    mins = np.full(n_dyn, 255.0)
    maxs = np.zeros(n_dyn)

    for y in years:
        info = cube.per_year[str(y)]
        dyn = np.memmap(info["dyn_file"], np.uint8, "r", shape=tuple(info["dyn_shape"]))
        days = rng.choice(info["T"], min(args.days_per_year, info["T"]), replace=False)
        for d in sorted(days.tolist()):
            k = rng.choice(vr.size, min(args.pixels_per_day, vr.size), replace=False)
            v = dyn[d, vr[k], vc[k], :].astype(np.float64)
            cnt += v.shape[0]
            s1 += v.sum(0)
            s2 += (v * v).sum(0)
            mins = np.minimum(mins, v.min(0))
            maxs = np.maximum(maxs, v.max(0))
        del dyn
        print(f"    {y}: {len(days)} Tage", flush=True)

    mean = s1 / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-12))

    # statische Kanaele: ein Jahr genuegt, sie sind zeitlich konstant
    stat = np.load(cube.per_year[str(years[0])]["stat_file"], mmap_mode="r")
    sv = np.asarray(stat[vr[::37], vc[::37], :], np.float64)
    smean, sstd = sv.mean(0), np.maximum(sv.std(0), 1e-12)

    # Ausnahmen neutralisieren: Mittelwert 0, Streuung 1 -> Wert bleibt roh
    for c in EXCLUDE_DYNAMIC:
        mean[c], std[c] = 0.0, 1.0
    for c in EXCLUDE_STATIC:
        smean[c], sstd[c] = 0.0, 1.0

    out = exp / "derived_data" / "norm_stats.npz"
    np.savez(out, dyn_mean=mean.astype(np.float32), dyn_std=std.astype(np.float32),
             stat_mean=smean.astype(np.float32), stat_std=sstd.astype(np.float32),
             dyn_min=mins.astype(np.float32), dyn_max=maxs.astype(np.float32),
             excluded_dynamic=np.array(EXCLUDE_DYNAMIC),
             excluded_static=np.array(EXCLUDE_STATIC),
             train_years=np.array(years))

    ch = json.loads((exp / "data_manifests" / "d4_cube.json").read_text())
    names = {c["cube_index"]: c["name"] for c in ch["dynamic_channels"]}
    print(f"\n[norm] {int(cnt[0]):,} Werte je Kanal ausgewertet".replace(",", " "))
    print(f"{'Kanal':26s} {'Mittel':>8s} {'Streuung':>9s} {'min':>5s} {'max':>5s}")
    for c in range(n_dyn):
        tag = "  (roh)" if c in EXCLUDE_DYNAMIC else ""
        print(f"{names[c][:24]:26s} {mean[c]:8.2f} {std[c]:9.2f} "
              f"{mins[c]:5.0f} {maxs[c]:5.0f}{tag}")

    degenerate = [names[c] for c in range(n_dyn)
                  if c not in EXCLUDE_DYNAMIC and std[c] < 1e-3]
    print(f"\n[norm] Kanaele ohne Streuung: {degenerate or 'keine'}")

    write_manifest(cfg, "d11_norm_stats", args.config,
                   {"split": args.split, "train_years": years,
                    "n_values_per_channel": int(cnt[0]),
                    "days_per_year": args.days_per_year,
                    "pixels_per_day": args.pixels_per_day,
                    "excluded_dynamic": list(EXCLUDE_DYNAMIC),
                    "excluded_static": list(EXCLUDE_STATIC),
                    "exclusion_reason": {
                        "0": "Feuermaske ist binaer",
                        "static_12": "Landbedeckung ist kategorial, nicht ordinal"},
                    "degenerate_channels": degenerate,
                    "dynamic": {names[c]: {"mean": float(mean[c]), "std": float(std[c]),
                                           "min": float(mins[c]), "max": float(maxs[c])}
                                for c in range(n_dyn)},
                    "file": str(out)})


if __name__ == "__main__":
    main()

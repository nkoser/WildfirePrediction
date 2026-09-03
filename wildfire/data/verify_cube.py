"""Verifiziert den Datenwuerfel gegen die Rohdaten.

Der Wuerfel ist eine Kopie mit umsortiertem Layout. Jeder Fehler darin -
vertauschte Kanaele, verschobene Tage, falsche Transposition - wuerde
still durch die gesamte Pipeline propagieren und erst an unerklaerlichen
Modellergebnissen auffallen. Deshalb wird hier stichprobenartig, aber
exakt geprueft:

  1. Vollstaendige Tagesscheiben: fuer zufaellige (Modalitaet, Tag) wird
     das Rohraster einmal gelesen und JEDER Pixel JEDES Bandes gegen den
     Wuerfel verglichen. Das ist strenger als eine Punktstichprobe und
     faengt auch Transpositions- und Kanalvertauschungsfehler.
  2. Statische Kanaele ebenso, vollstaendig.
  3. Lesegeschwindigkeit fuer Patches, wie sie das Training braucht.

Warum keine Punktstichproben? Die Archive haben keinen Index; ein
einzelnes Mitglied zu finden kostet einen Scan ueber alle Header. Pro
Rohtag EINMAL lesen und dafuer alles vergleichen ist um Groessenordnungen
schneller und zugleich strenger.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, load_config, write_manifest


def read_raw_day(dataset_root: Path, modality: str, date: str) -> np.ndarray:
    y = date[:4]
    tar = dataset_root / "Entire_Canada_Maps" / modality / f"{modality}__{y}.tar"
    target = f"{date.replace('-', '_')}.tif"
    with tarfile.open(tar, "r:") as tf:
        for m in tf:
            if m.isfile() and Path(m.name).name == target:
                blob = tf.extractfile(m).read()
                with MemoryFile(blob) as mem, mem.open() as ds:
                    return ds.read()
    raise SystemExit(f"nicht gefunden: {modality}/{target}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-slice-checks", type=int, default=12)
    ap.add_argument("--patch", type=int, default=33)
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--n-patch-reads", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    exp = Path(cfg["experiment_root"])
    cube_man = json.loads((exp / "data_manifests" / "d4_cube.json").read_text())
    root = Path(cfg["dataset_root"])
    dyn_ch = cube_man["dynamic_channels"]
    stat_ch = cube_man["static_channels"]
    by_mod: dict[str, list[dict]] = {}
    for c in dyn_ch:
        by_mod.setdefault(c["modality"], []).append(c)
    stat_by_mod: dict[str, list[dict]] = {}
    for c in stat_ch:
        stat_by_mod.setdefault(c["modality"], []).append(c)

    results = {"years": {}, "slice_mismatches": []}
    n_slice_ok = n_slice_bad = 0
    n_pixels_compared = 0

    for year, info in cube_man["per_year"].items():
        arr = np.memmap(info["dyn_file"], dtype=np.uint8, mode="r",
                        shape=tuple(info["dyn_shape"]))
        stat = np.load(info["stat_file"], mmap_mode="r")
        days = info["days"]
        T, H, W, C = arr.shape
        print(f"[verify] {year}: {arr.shape} dynamisch, {stat.shape} statisch")

        # ---------------------------------------------- 1. statische Kanaele
        for mod, chans in stat_by_mod.items():
            raw = read_raw_day(root, mod, days[0])
            for ch in chans:
                a = np.asarray(stat[:, :, ch["cube_index"]])
                b = raw[ch["band"]]
                n_pixels_compared += a.size
                if np.array_equal(a, b):
                    n_slice_ok += 1
                else:
                    n_slice_bad += 1
                    results["slice_mismatches"].append(
                        {"year": year, "static": True, "channel": ch["name"],
                         "n_differing": int((a != b).sum())}
                    )

        # ---------------------------------------------- 2. ganze Tagesscheiben
        for _ in range(args.n_slice_checks):
            ti = int(rng.integers(T))
            mod = list(by_mod)[int(rng.integers(len(by_mod)))]
            raw = read_raw_day(root, mod, days[ti])
            for ch in by_mod[mod]:
                a = np.asarray(arr[ti, :, :, ch["cube_index"]])
                b = raw[ch["band"]]
                n_pixels_compared += a.size
                if np.array_equal(a, b):
                    n_slice_ok += 1
                else:
                    n_slice_bad += 1
                    results["slice_mismatches"].append(
                        {"year": year, "day": days[ti], "channel": ch["name"],
                         "cube_index": ch["cube_index"],
                         "n_differing": int((a != b).sum()),
                         "max_abs_diff": int(np.abs(a.astype(int) - b.astype(int)).max())}
                    )

        # ---------------------------------------------- 3. Lesegeschwindigkeit
        half = args.patch // 2
        L = args.lookback
        t0 = time.perf_counter()
        acc = 0
        for _ in range(args.n_patch_reads):
            ti = int(rng.integers(L, T))
            r = int(rng.integers(half, H - half))
            c = int(rng.integers(half, W - half))
            blk = np.asarray(arr[ti - L:ti, r - half:r + half + 1, c - half:c + half + 1, :])
            acc += int(blk[0, 0, 0, 0])
        dt_ = time.perf_counter() - t0
        per = dt_ / args.n_patch_reads
        bytes_per = L * args.patch * args.patch * C
        results["years"][year] = {
            "shape": list(arr.shape),
            "patch_read": {
                "n": args.n_patch_reads,
                "seconds_total": dt_,
                "ms_per_sample": 1000 * per,
                "bytes_per_sample": bytes_per,
                "throughput_MB_s": bytes_per * args.n_patch_reads / dt_ / 1e6,
                "samples_per_second": 1 / per,
            },
        }
        print(f"[verify] {year}: Patch ({L}x{args.patch}x{args.patch}x{C}) "
              f"{1000*per:.2f} ms/Sample = {1/per:.0f} Samples/s, "
              f"{bytes_per/1e6:.2f} MB je Sample")
        del arr, stat

    ok = n_slice_bad == 0
    print(f"\n[verify] Scheibenvergleiche: {n_slice_ok} korrekt, {n_slice_bad} abweichend "
          f"({n_pixels_compared/1e6:.1f} Mio. Pixel byte-genau verglichen)")
    print(f"[verify] Gesamt: {'BESTANDEN' if ok else 'FEHLGESCHLAGEN'}")
    if not ok:
        for m in results["slice_mismatches"][:8]:
            print(f"    {m}")

    write_manifest(
        cfg, "d5_verify_cube", args.config,
        {
            "n_slice_checks_ok": n_slice_ok, "n_slice_checks_failed": n_slice_bad,
            "n_pixels_compared": n_pixels_compared,
            "passed": ok,
            "patch_size": args.patch, "lookback": args.lookback,
            **results,
        },
    )
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()

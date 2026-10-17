"""Paper-Protokoll: gespeicherte Vorhersagen des Paper-Modells gegen andere Labels bewerten.

Das Paper berichtet fuer das 3D-U-Net PR-AUC 0,718 / 0,598 / 0,551 / 0,481
(dt = 1 / 7 / 15 / 30); unser Nachbau auf denselben Caches liegt bei
0,84 -> 0,68. Hier wird geprueft, ob ein anderes Ziel bei der Auswertung die
Luecke erklaert. Die Modelle bleiben, wie sie sind (trainiert auf das
Cache-Label); nur das Test-Label wechselt:

  paper          /labels des Caches (Feuer im 13x13-Ausschnitt an D)
  patch_D        Feuermaske des Caches im Ausschnitt an D (Kanal 0, letzter Tag)
  centre_D       Feuermaske nur in der Mittelzelle an D
  patch_D+k      Feuer im Ausschnitt an D+k (Rohraster MCD14A1), k = 1, 2
  centre_D+k     Feuer in der Mittelzelle an D+k
  pixel_D        jedes der 169 Pixel einzeln gegen seine Feuermaske an D,
                 Score = Score der Stichprobe (so saehe eine pixelweise
                 Auswertung eines Modells mit einer Ausgabe je Ausschnitt aus)
"""

from __future__ import annotations

import argparse
import datetime as dt
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.prepare import FIRE_CH

DTS = (1, 7, 15, 30)
HALF = 6
PAPER = {1: 0.7182, 7: 0.5984, 15: 0.5507, 30: 0.4805}


def raw_fire(root: Path, dates: set[dt.date]) -> dict[dt.date, np.ndarray]:
    """Feuermaske (H, W) bool je Tag aus Entire_Canada_Maps/MCD14A1."""
    out = {}
    by_year = defaultdict(set)
    for d in dates:
        by_year[d.year].add(f"{d.year}_{d.month:02d}_{d.day:02d}.tif")
    for year, names in sorted(by_year.items()):
        tar = root / "MCD14A1" / f"MCD14A1__{year}.tar"
        with tarfile.open(tar, "r:") as tf:
            for m in tf:
                base = Path(m.name).name
                if m.isfile() and base in names:
                    with MemoryFile(tf.extractfile(m).read()) as mem, mem.open() as src:
                        out[dt.date(*map(int, base[:10].split("_")))] = src.read(1) > 0
        print(f"[lbl] {year}: {sum(1 for d in out if d.year == year)} Tage gelesen", flush=True)
    return out


def patch(a: np.ndarray, r: int, c: int) -> np.ndarray:
    H, W = a.shape
    return a[max(0, r - HALF):min(H, r + HALF + 1), max(0, c - HALF):min(W, c + HALF + 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--caches", nargs="+", default=["cache_A", "cache_G", "cache_H", "cache_Y"])
    ap.add_argument("--runs", nargs="+", default=["_paperhp", "_paperdt", "_paperhp_last50", "_paperhp_last100"],
                    help="Laeufe von 42_paper_reconstruct (fehlende werden uebersprungen)")
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp, raw = Path(cfg["experiment_root"]), Path(cfg["dataset_root"]) / "Entire_Canada_Maps"

    res: dict = {}
    for cache in args.caches:
        d = exp / "derived_data" / "paper" / f"{cache}_L61"
        X = np.load(d / "X.npy", mmap_mode="r")
        y = np.load(d / "y.npy").astype(int)
        meta = np.load(d / "meta.npy")
        te = np.where(np.load(d / "split.npy") == "test")[0]
        fire_D = np.asarray(X[te, -1, FIRE_CH]) > 0                       # (n, 13, 13)
        dates = [dt.date(int(m // 10000), int(m // 100 % 100), int(m % 100)) for m in meta[te, 0]]
        need = {D + dt.timedelta(days=k) for D in dates for k in args.leads}
        fire = raw_fire(raw, need)

        labels = {"paper": y[te], "patch_D": fire_D.any((1, 2)).astype(int),
                  "centre_D": fire_D[:, HALF, HALF].astype(int)}
        for k in args.leads:
            pk, ck = [], []
            for D, (_, r, c) in zip(dates, meta[te]):
                f = fire.get(D + dt.timedelta(days=k))
                if f is None:
                    pk.append(-1); ck.append(-1)
                    continue
                pk.append(int(patch(f, r, c).any())); ck.append(int(f[r, c]))
            labels[f"patch_D+{k}"], labels[f"centre_D+{k}"] = np.array(pk), np.array(ck)
        base = {k: float(v[v >= 0].mean()) for k, v in labels.items()}
        base["pixel_D"] = float(fire_D.mean())
        print(f"\n[lbl] {cache}: Basisraten " + "  ".join(f"{k} {v:.3f}" for k, v in base.items()), flush=True)
        res[cache] = {"base_rate": base, "agreement_with_paper_label":
                      {k: float((v[v >= 0] == labels["paper"][v >= 0]).mean()) for k, v in labels.items()},
                      "runs": {}}

        for run in args.runs:
            p = exp / "predictions" / "paper" / f"rec_{cache}_cache{run}.npz"
            if not p.exists():
                print(f"[lbl] {p.name} fehlt - uebersprungen", flush=True)
                continue
            z = np.load(p)
            pos = {int(i): n for n, i in enumerate(z["idx_test"])}
            sel = np.array([pos[int(i)] for i in te])
            out = {}
            for dt_ in DTS:
                ks = sorted(k for k in z.files if k.startswith(f"paper55_T30_end{dt_ + 1}_s") and k.endswith("_test"))
                if not ks:
                    continue
                row = {}
                for name, lab in labels.items():
                    ok = lab >= 0
                    row[name] = float(np.mean([average_precision_score(lab[ok], z[k][sel][ok]) for k in ks]))
                row["pixel_D"] = float(np.mean([average_precision_score(
                    fire_D.reshape(-1), np.repeat(z[k][sel], 169)) for k in ks]))
                out[str(dt_)] = row
            res[cache]["runs"][run] = out
            print(f"[lbl] {cache} {run}: PR-AUC je Label (Paper: " + " / ".join(f"{v:.3f}" for v in PAPER.values()) + ")",
                  flush=True)
            for name in list(labels) + ["pixel_D"]:
                print(f"    {name:11s} " + "  ".join(f"dt={dt_:2d} {out[str(dt_)][name]:.3f}" for dt_ in DTS
                                                  if str(dt_) in out), flush=True)

    write_manifest(cfg, "p12_label_variants", args.config, {
        "paper_reported_pr_auc": PAPER, "labels": __doc__.split("Hier wird")[1].strip(),
        "results": res,
    })


if __name__ == "__main__":
    main()

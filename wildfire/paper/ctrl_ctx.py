"""Bekaempfungsstatus der Feuerbehoerden (CWFIS Reported Fires) als Pixel-Zusatzeingabe fuer sts_extreme --ctx px.

Gleiches Dateiformat wie ctx_features (derived_data/<out-name>/{train,surv_val,surv_eval}_px.npy (n, 14, 13, 13),
cache_px.npy (N, 4, 14, 13, 13), float16), damit sts_extreme ohne Aenderung mit --ctx px --ctx-dir <out-name>
laeuft. Der Zweig ctx_px startet bei null: zu Beginn ist das Netz genau die Referenz ohne Zusatz (*_ref).

Je Stichtag t (Wuerfel-Stichtag L, surv-Stichtag L_cube, Cache e = D - dt) zaehlen nur Lageberichte, deren
Intervall [record_start, record_end) den Mittag von t enthaelt - also nur, was bis t gemeldet war. Jeder Bericht
wird als Ellipse der gemeldeten Flaeche (Radius sqrt(ha / pi), Zelle 5 km x 5 km * cos(Breite), mindestens die
Zelle des Meldepunkts) in den 13 x 13-Ausschnitt gezeichnet; je Zelle gilt der dominante Bericht (hoechste Stufe,
dann groesste Flaeche). Keine Rx-Braende.

Kanaele (0..1):
  0-3   Stufe OC / BH / UC / EX (EX nur, wenn in [t - 30, t] geloescht)
  4-6   response_type FUL / MOD / MON
  7     log1p(Flaeche ha) / log1p(1e6)
  8     Tage seit dem letzten Statuswechsel (status_date) / 30, gekappt
  9     Tage seit der ersten Meldung des Brandes / 30, gekappt
  10    Lageberichte verfuegbar (Jahr von t >= --first-year; davor gibt es keine Zeitreihe), ganzer Ausschnitt
  11    percent_contained / 100 (fehlend = 0)
  12-13 leer (0), nur fuer die feste Breite von 14 Kanaelen

Ausgabe NEU: derived_data/<out-name>/{train,surv_val,surv_eval}_px.npy, cache_px.npy, channels.json,
Manifest p55_ctrl_ctx_<out-name>.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.compare import DTS
from wildfire.paper.suppression_daily import t_of

NAMES = ["OC", "BH", "UC", "EX30", "FUL", "MOD", "MON", "log_size", "status_age", "fire_age", "avail",
         "pct_contained", "pad0", "pad1"]
STAGE = {"OC": 0, "BH": 1, "UC": 2, "EX": 3}          # Rang: kleiner = dominanter
RESP = {"FUL": 4, "MOD": 5, "MON": 6}
HALF, BLK, CELL_KM = 6, 16, 4.99


def load(src: Path, tr) -> tuple[dict, dict]:
    """Berichte im Gitter -> {(jahr, bi, bj): Feldarrays der 3 x 3 Nachbarbloecke}, Zaehlungen."""
    recs, seen, disc = [], set(), {}
    for f in sorted(src.glob("status_*.csv.gz")):
        with gzip.open(f, "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                fid, sr = r["national_fire_id"], t_of(r["situation_report_date"])
                if math.isfinite(sr):
                    disc[fid] = min(disc.get(fid, sr), sr)
                if r.get("fire_was_prescribed") == "1" or r["stage_of_control_status"] not in STAGE:
                    continue
                try:
                    lat, lon = float(r["latitude"]), float(r["longitude"])
                    size = max(float(r["fire_size"] or 0), 0.0)
                    pc = float(r["percent_contained"] or -1)
                except ValueError:
                    continue
                t0, t1 = t_of(r["record_start"]), t_of(r["record_end"])
                if not (math.isfinite(t0) and math.isfinite(t1)):
                    continue
                recs.append((math.floor((lat - tr[5]) / tr[4]), math.floor((lon - tr[2]) / tr[0]), lat,
                             STAGE[r["stage_of_control_status"]], RESP.get(r["response_type"], -1), size,
                             t0, t1, t_of(r["status_date"]), fid, pc))
    cols = list(zip(*recs))
    A = {k: np.array(v) for k, v in zip(("row", "col", "lat", "stage", "resp", "size", "t0", "t1", "ts"), cols[:9])}
    A["disc"] = np.array([disc.get(f, math.nan) for f in cols[9]])
    A["pc"] = np.array(cols[10])
    A["year"] = np.array([dt.date.fromordinal(int(v)).year for v in A["t0"]])
    by = defaultdict(list)
    for i, (y, r, c) in enumerate(zip(A["year"], A["row"] // BLK, A["col"] // BLK)):
        by[(int(y), int(r), int(c))].append(i)
    nb = {}
    for (y, bi, bj) in {(y, bi + a, bj + b) for (y, bi, bj) in by for a in (-1, 0, 1) for b in (-1, 0, 1)}:
        idx = [i for a in (-1, 0, 1) for b in (-1, 0, 1) for i in by.get((y, bi + a, bj + b), ())]
        if idx:
            nb[(y, bi, bj)] = np.array(idx)
    cnt = {"records": len(recs), "by_stage": dict(Counter(A["stage"].tolist())),
           "by_year": {int(k): int(v) for k, v in sorted(Counter(A["year"].tolist()).items())}}
    return {"A": A, "nb": nb}, cnt


def raster(R: dict, t: float, r0: int, c0: int, first_year: int) -> np.ndarray:
    """Status am Mittag des Tages t (Ordinal) im Ausschnitt um (r0, c0) -> (14, 13, 13)."""
    out = np.zeros((len(NAMES), 2 * HALF + 1, 2 * HALF + 1), np.float32)
    y = dt.date.fromordinal(int(t)).year
    if y >= first_year:
        out[10] = 1
    idx = R["nb"].get((y, r0 // BLK, c0 // BLK))
    if idx is None:
        return out
    A, tm = R["A"], t + 0.5
    rr, cc = A["row"][idx] - r0, A["col"][idx] - c0
    act = (A["t0"][idx] <= tm) & (tm < A["t1"][idx])
    act &= (A["stage"][idx] != 3) | ((A["ts"][idx] >= tm - 30) & (A["ts"][idx] <= tm))
    rad = np.sqrt(A["size"][idx] / 100 / np.pi) / CELL_KM                    # Radius in Zeilen
    act &= (np.abs(rr) <= HALF + rad + 1) & (np.abs(cc) <= HALF + rad / np.cos(np.radians(A["lat"][idx])) + 1)
    sel = idx[act]
    if sel.size == 0:
        return out
    order = sel[np.lexsort((A["size"][sel], -A["stage"][sel]))]               # dominanter Bericht zuletzt
    I, J = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
    for i in order:
        dr, dc = A["row"][i] - r0, A["col"][i] - c0
        ry = max(math.sqrt(A["size"][i] / 100 / math.pi) / CELL_KM, 0.5)
        rx = max(ry / math.cos(math.radians(A["lat"][i])), 0.5)
        m = ((I - dr) / ry) ** 2 + ((J - dc) / rx) ** 2 <= 1
        if abs(dr) <= HALF and abs(dc) <= HALF:
            m[dr + HALF, dc + HALF] = True
        if not m.any():
            continue
        out[:10, m] = 0
        out[10, m] = 1 if y >= first_year else 0
        out[A["stage"][i], m] = 1
        if A["resp"][i] >= 0:
            out[A["resp"][i], m] = 1
        out[7, m] = math.log1p(A["size"][i]) / math.log1p(1e6)
        out[8, m] = min(max((tm - A["ts"][i]) / 30, 0), 1) if math.isfinite(A["ts"][i]) else 0
        out[9, m] = min(max((tm - A["disc"][i]) / 30, 0), 1) if math.isfinite(A["disc"][i]) else 0
        out[11, m] = A["pc"][i] / 100 if A["pc"][i] >= 0 else 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reports", default="/data/Wildfire/artifacts/cwfis_reportedfires")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--out-name", default="ctx_ctrl")
    ap.add_argument("--first-year", type=int, default=2011, help="erstes Jahr mit Statuszeitreihe (2010: ein Bericht je Brand)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "derived_data" / args.out_name
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --out-name waehlen")
    man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    t0 = time.time()
    R, cnt = load(Path(args.reports), man["grid"]["transform"])
    print(f"[ctrl] {cnt['records']:,} Statusintervalle ({time.time() - t0:.0f} s)", flush=True)
    cube = CubeReader(exp)
    out.mkdir(parents=True)
    sets = {}

    def build(name: str, days: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        px = np.stack([raster(R, float(d), int(r), int(c), args.first_year) for d, r, c in zip(days, rows, cols)])
        nz = (px[:, :4].sum(1) > 0).reshape(len(px), -1).any(1)
        sets[name] = {"n": int(len(px)), "share_with_report": float(nz.mean()),
                      "mean": dict(zip(NAMES, px.mean((0, 2, 3)).round(5).tolist()))}
        print(f"[ctrl] {name}: {len(px):,} Stichtage, Anteil mit Bericht im Ausschnitt {nz.mean():.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        return px.astype(np.float16)

    ordc = lambda L: dt.date.fromisoformat(cube.dates[int(L)]).toordinal()  # noqa: E731
    for name, mpath, lkey in (("train", exp / "derived_data" / args.cube_dir / "train_meta.npz", "L"),
                              ("surv_val", exp / "derived_data" / args.surv_dir / "val_meta.npz", "L_cube"),
                              ("surv_eval", exp / "derived_data" / args.surv_dir / "eval_meta.npz", "L_cube")):
        m = np.load(mpath)
        np.save(out / f"{name}_px.npy", build(name, np.array([ordc(L) for L in m[lkey]]), m["row"], m["col"]))
    meta = np.load(exp / "derived_data" / "paper" / f"{args.cache}_L61" / "meta.npy")
    dD = np.array([dt.date(int(v) // 10000, int(v) // 100 % 100, int(v) % 100).toordinal() for v in meta[:, 0]])
    px_c = np.stack([build(f"cache_dt{d_}", dD - d_, meta[:, 1], meta[:, 2]) for d_ in DTS], 1)
    np.save(out / "cache_px.npy", px_c)
    (out / "channels.json").write_text(json.dumps({"names": NAMES, "first_year": args.first_year}, indent=1))
    write_manifest(cfg, f"p55_ctrl_ctx_{args.out_name}", args.config,
                   {"args": vars(args), "reports": cnt, "sets": sets, "names": NAMES, "out": str(out)})
    print(f"[ctrl] {out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

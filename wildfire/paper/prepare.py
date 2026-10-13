"""Paper-Protokoll, Schritt 1: Fenster-Cache einer Variante in ein lokales Array.

Was das Paper benutzt und woher wir es nehmen:

  Stichproben, Split   samples_variant_<X>.json - exakt die Liste des Papers
  Labels               /labels im HDF5 - exakt die Labels des Papers. Aus
                       unseren Rohdaten laesst sich die Regel nicht exakt
                       rekonstruieren (Audit s6: beste Regel "Feuer im
                       13x13-Ausschnitt an Tag D" trifft 95,5 %; die JSON
                       nennt ein Feld firms_value - vermutlich FIRMS-Daten,
                       die nicht im Datensatz liegen).
  Kanaele              alle 55 aus /windows, Reihenfolge laut Audit s5:
                       ch0 MCD14A1-Feuermaske, ch1-18 ERA5, ch19-24 FWI,
                       ch25 VPD, ch26-29 OSM, ch30 Worldpop, ch31-37 DEM,
                       ch38-39 NDVI/EVI, ch40-41 LAI/FPAR, ch42-45 Reflexion,
                       ch46-49 LST/Emissivitaet, ch50-53 Helligkeitstemp.,
                       ch54 Landbedeckung
  Zeitfenster          Das Paper nennt T = 30 Tage; der Cache haelt 61 Tage
                       (D-60 .. D, letzter Schritt = Sample-Datum D, Audit s5).
                       Wir nehmen die letzten T.

Ausgabe je Variante unter $EXP/derived_data/paper/<cache>/:
  X.npy        (N, T, 55, 13, 13) uint8, zeitzuerst wie (T, C, H, W) im Paper
  y.npy        (N,) int8          Labels aus /labels
  split.npy    (N,) <U5           train / val / test aus der JSON
  meta.npy     (N, 3) int64       Datum JJJJMMTT, Zeile, Spalte

Jede Zeile wird gegen die JSON geprueft (Datum, Zeile, Spalte, Label). Bei der
kleinsten Abweichung bricht der Schritt ab - ein still verschobener Index
waere hier der teuerste moegliche Fehler.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest

CHANNEL_NAMES = (
    ["MCD14A1_fire"]
    + ["t2m", "u10", "v10", "snow_cover", "precip", "latent_heat", "dewpoint", "pressure",
       "swvl1", "swvl2", "swvl3", "swvl4", "t2m_max", "skin_t_max", "pot_evap",
       "tot_evap", "skin_reservoir", "net_solar"]
    + ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "VPD"]
    + ["OSM_road", "OSM_powerline", "OSM_building", "OSM_water", "Worldpop"]
    + ["DEM", "slope", "aspect_sin", "aspect_cos", "hillshade", "TPI", "TWI"]
    + ["NDVI", "EVI", "LAI", "FPAR"]
    + ["refl_b01", "refl_b02", "refl_b03", "refl_b07"]
    + ["LST_day", "LST_night", "Emis_31", "Emis_32"]
    + ["BT_b20", "BT_b21", "BT_b31", "BT_b32"]
    + ["landcover"]
)
assert len(CHANNEL_NAMES) == 55
FIRE_CH = 0
BT_CH = (50, 51, 52, 53)
LANDCOVER_CH = 54


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True, choices=("cache_A", "cache_G", "cache_H", "cache_Y"))
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--tag", default="",
                    help="Suffix des Zielordners, z. B. _L61 fuer das volle Fenster")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    root = Path(cfg["dataset_root"]) / args.cache
    exp = Path(cfg["experiment_root"])
    out = exp / "derived_data" / "paper" / f"{args.cache}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    doc = json.loads(next(root.glob("samples_variant_*.json")).read_text())
    records = doc["positive_samples"] + doc["negative_samples"]
    h5_path = next(root.glob("*.h5"))

    with h5py.File(h5_path, "r") as f:
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        W = f["windows"]
        N, C, P, P2, L = W.shape
        print(f"[paper] {args.cache}: /windows {W.shape} {W.dtype}, attrs lookback="
              f"{attrs.get('lookback')} forecast={attrs.get('forecast')} "
              f"pn_ratio={attrs.get('pn_ratio')}")
        assert C == 55 and P == P2 == 13, W.shape
        assert len(records) == N, (len(records), N)
        T = args.lookback
        assert T <= L, (T, L)

        labels = f["labels"][...].astype(np.int8)
        meta = f["meta"][...].astype(np.int64)

        # ---- Zeile fuer Zeile gegen die JSON pruefen
        bad = Counter()
        for i, r in enumerate(records):
            d = int(r["date"].replace("-", ""))
            if (d, int(r["pixel_row"]), int(r["pixel_col"])) != tuple(meta[i]):
                bad["meta"] += 1
            if labels[i] != (1 if r["sample_type"] == "positive" else 0):
                bad["label"] += 1
        if bad:
            raise SystemExit(f"ABBRUCH: HDF5 und JSON passen nicht zusammen: {dict(bad)}")
        print(f"[paper] {N} Zeilen gegen die JSON geprueft: Datum, Zeile, Spalte, Label stimmen")

        split = np.array([r["split"] for r in records])
        X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.uint8,
                                      shape=(N, T, C, P, P))
        for a in range(0, N, args.chunk):
            b = min(a + args.chunk, N)
            blk = W[a:b, :, :, :, L - T:]                     # (n, C, P, P, T)
            X[a:b] = np.moveaxis(blk, -1, 1)                  # (n, T, C, P, P)
            if (a // args.chunk) % 10 == 0:
                print(f"    {b}/{N}", flush=True)
        X.flush()
        del X

    np.save(out / "y.npy", labels)
    np.save(out / "split.npy", split)
    np.save(out / "meta.npy", meta)

    tab = {s: {"positive": int(((split == s) & (labels == 1)).sum()),
               "negative": int(((split == s) & (labels == 0)).sum())}
           for s in ("train", "val", "test")}
    years = {s: sorted({int(m // 10000) for m in meta[split == s, 0]}) for s in tab}
    months = Counter(int(m // 100 % 100) for m in meta[:, 0])
    print(f"[paper] Splits: {tab}")
    print(f"[paper] Jahre je Split: { {s: (v[0], v[-1]) for s, v in years.items()} }")

    write_manifest(cfg, f"p0_paper_{args.cache}{args.tag}", args.config, {
        "cache": args.cache, "h5": str(h5_path), "h5_attrs": attrs,
        "lookback_used": T, "lookback_in_cache": L,
        "window_days": f"D-{T - 1} .. D (letzter Schritt = Sample-Datum, Audit s5)",
        "channel_names": CHANNEL_NAMES,
        "fire_channel": FIRE_CH, "brightness_temperature_channels": list(BT_CH),
        "split_table": tab, "years": years,
        "months": dict(sorted(months.items())),
        "outputs": {k: str(out / f"{k}.npy") for k in ("X", "y", "split", "meta")},
        "x_bytes": human_bytes(N * T * C * P * P),
    })


if __name__ == "__main__":
    main()

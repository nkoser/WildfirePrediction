"""Stufe 8: Verfuegbarkeitspruefung fuer Sentinel-Bilddaten.

Der vorhandene MPCFire-Wuerfel ist mit 5 km und uint8 zu grob fuer eine
Vision-Argumentation. Diese Stufe klaert faktisch, ob echte
Earth-Observation-Sequenzen ergaenzbar waeren:

  * Sind die offenen STAC-Kataloge ohne Anmeldung nutzbar?
  * Wie viele Sentinel-2-Szenen liegen ueber realen kanadischen
    Brandgebieten in der Feuersaison, und wie wolkig sind sie?
  * Welche effektive Wiederholrate bleibt nach Wolkenfilterung uebrig -
    das ist die eigentliche Frage, denn boreale Sommer sind bewoelkt.
  * Laesst sich ein Pixelfenster tatsaechlich per HTTP-Range aus einem
    COG lesen, ohne den ganzen Kachelinhalt zu laden?
  * Wie gross wuerde ein ereigniszentrierter Chip-Datensatz?

Sentinel-1 (Radar, wolkenunabhaengig) wird zum Vergleich mitgeprueft.

Es werden KEINE Daten heruntergeladen ausser wenigen Kilobyte Metadaten
und einem einzelnen Testfenster.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest

STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTIONS = {
    "sentinel-2-l2a": "Sentinel-2 L2A (optisch, 10-20 m)",
    "sentinel-1-grd": "Sentinel-1 GRD (Radar, wolkenunabhaengig)",
}
CLOUD_THRESHOLDS = (10, 30, 60)


def stac_search(collection: str, lon: float, lat: float, start: str, end: str, limit: int = 500) -> list[dict]:
    body = {
        "collections": [collection],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": limit,
    }
    req = urllib.request.Request(
        STAC,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as fh:
        doc = json.loads(fh.read())
    return doc.get("features", [])


def find_fire_location(cfg: dict, date: dt.date) -> tuple[float, float, int, int]:
    """Ein realer Feuerpixel des angegebenen Tages, als lon/lat."""
    root = Path(cfg["dataset_root"])
    tar = root / "Entire_Canada_Maps" / "MCD14A1" / f"MCD14A1__{date.year}.tar"
    target = f"{date.year}_{date.month:02d}_{date.day:02d}.tif"
    with tarfile.open(tar, "r:") as tf:
        for m in tf:
            if m.isfile() and Path(m.name).name == target:
                blob = tf.extractfile(m).read()
                break
        else:
            raise SystemExit(f"kein Raster fuer {date}")
    with MemoryFile(blob) as mem, mem.open() as ds:
        arr = ds.read(1)
        T = ds.transform
    rr, cc = np.nonzero(arr == 1)
    if rr.size == 0:
        raise SystemExit(f"kein Feuer am {date}")
    # das noerdlichste Feuer, damit wir wirklich in Kanada landen
    i = int(np.argmin(rr))
    r, c = int(rr[i]), int(cc[i])
    lon = T.c + T.a * (c + 0.5)
    lat = T.f + T.e * (r + 0.5)
    return lon, lat, r, c


def summarise(feats: list[dict], collection: str) -> dict:
    dates, clouds = [], []
    for f in feats:
        p = f.get("properties", {})
        d = (p.get("datetime") or "")[:10]
        if d:
            dates.append(d)
        cc = p.get("eo:cloud_cover")
        if cc is not None:
            clouds.append(float(cc))
    uniq = sorted(set(dates))
    gaps = []
    if len(uniq) > 1:
        ds = [dt.date.fromisoformat(x) for x in uniq]
        gaps = [(b - a).days for a, b in zip(ds, ds[1:])]

    out = {
        "collection": collection,
        "n_items": len(feats),
        "n_distinct_dates": len(uniq),
        "date_min": uniq[0] if uniq else None,
        "date_max": uniq[-1] if uniq else None,
        "median_gap_days": float(np.median(gaps)) if gaps else None,
        "max_gap_days": int(max(gaps)) if gaps else None,
        "cloud_cover_available": bool(clouds),
    }
    if clouds:
        arr = np.asarray(clouds)
        out["cloud_cover"] = {
            "mean": float(arr.mean()),
            "p50": float(np.median(arr)),
            "n_scenes": int(arr.size),
        }
        # effektive Wiederholrate nach Wolkenfilter
        for thr in CLOUD_THRESHOLDS:
            keep = sorted({d for d, c in zip(dates, clouds) if c <= thr})
            if len(keep) > 1:
                kd = [dt.date.fromisoformat(x) for x in keep]
                g = [(b - a).days for a, b in zip(kd, kd[1:])]
                out[f"usable_lt_{thr}pct"] = {
                    "n_dates": len(keep),
                    "median_gap_days": float(np.median(g)),
                    "max_gap_days": int(max(g)),
                }
            else:
                out[f"usable_lt_{thr}pct"] = {"n_dates": len(keep), "median_gap_days": None, "max_gap_days": None}
    return out


def test_pixel_read(feats: list[dict]) -> dict:
    """Kann ein kleines Fenster per HTTP-Range aus einem COG gelesen werden?"""
    import rasterio
    from rasterio.windows import Window

    for f in feats:
        assets = f.get("assets", {})
        for key in ("red", "B04", "nir", "B08", "vv", "VV"):
            a = assets.get(key)
            if not a or not a.get("href", "").startswith("http"):
                continue
            href = a["href"]
            try:
                with rasterio.Env(
                    AWS_NO_SIGN_REQUEST="YES",
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                ):
                    with rasterio.open(href) as ds:
                        arr = ds.read(1, window=Window(0, 0, 256, 256))
                return {
                    "ok": True,
                    "asset": key,
                    "href": href,
                    "scene_id": f.get("id"),
                    "full_shape": [ds.height, ds.width],
                    "full_res_m": list(ds.res),
                    "crs": str(ds.crs),
                    "window_shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "window_min": int(arr.min()),
                    "window_max": int(arr.max()),
                }
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                return {"ok": False, "asset": key, "href": href, "error": last}
    return {"ok": False, "error": "kein passendes Asset gefunden"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe-dates", nargs="*", default=["2023-06-05", "2021-07-15", "2019-07-10"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    results = []
    read_test = None
    for ds_ in args.probe_dates:
        D = dt.date.fromisoformat(ds_)
        lon, lat, r, c = find_fire_location(cfg, D)
        print(f"[s8] Brandpixel {D} -> lon={lon:.3f} lat={lat:.3f} (Zeile {r}, Spalte {c})", flush=True)
        entry = {"probe_date": ds_, "lon": lon, "lat": lat, "row": r, "col": c, "collections": {}}
        season_start = f"{D.year}-05-01"
        season_end = f"{D.year}-09-30"
        for coll, label in COLLECTIONS.items():
            try:
                feats = stac_search(coll, lon, lat, season_start, season_end)
            except Exception as exc:
                entry["collections"][coll] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"    {coll}: FEHLER {exc}")
                continue
            summ = summarise(feats, coll)
            summ["label"] = label
            entry["collections"][coll] = summ
            u30 = summ.get("usable_lt_30pct") or {}
            print(f"    {coll:18s} {summ['n_distinct_dates']:3d} Termine "
                  f"({season_start}..{season_end}), Median-Abstand {summ['median_gap_days']} d, "
                  f"Wolken p50 {(summ.get('cloud_cover') or {}).get('p50')}, "
                  f"<30% Wolken: {u30.get('n_dates')} Termine, Median-Abstand {u30.get('median_gap_days')} d",
                  flush=True)
            if read_test is None and coll == "sentinel-2-l2a" and feats:
                read_test = test_pixel_read(feats)
                print(f"    Pixel-Lesetest: {read_test.get('ok')} {read_test.get('asset')} "
                      f"{read_test.get('window_shape')} {read_test.get('full_res_m')}", flush=True)
        results.append(entry)

    # Abschaetzung eines ereigniszentrierten Chip-Datensatzes
    est = {}
    for n_events in (500, 2000, 10000):
        for chip in (128, 256):
            for n_t in (4, 8):
                bands = 6
                nbytes = n_events * n_t * chip * chip * bands * 2
                est[f"{n_events}ev_{chip}px_{n_t}t_{bands}b_uint16"] = {
                    "bytes": nbytes, "human": human_bytes(nbytes)
                }

    print("\n[s8] Groesse eines ereigniszentrierten Sentinel-2-Chipdatensatzes:")
    for k, v in est.items():
        print(f"    {k:34s} {v['human']}")

    write_manifest(
        cfg,
        "s8_sentinel_probe",
        args.config,
        {
            "stac_endpoint": STAC,
            "auth_required": False,
            "probes": results,
            "pixel_read_test": read_test,
            "chip_dataset_size_estimates": est,
            "note": (
                "Sentinel-2 beginnt mit voller 5-Tage-Wiederholung erst 2017 und deckt "
                "damit nur einen Teil des Feuerarchivs 2000-2025 ab."
            ),
        },
    )


if __name__ == "__main__":
    main()

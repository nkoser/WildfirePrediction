"""Ungekappte FWI-Indizes (FFMC, DMC, DC, ISI, BUI, FWI) auf dem Wuerfelgitter.

Die Rohraster des Datensatzes sind uint8 ohne scale/offset (audit/channel_dict.py). Extreme
Werte sind darin bei 255 abgeschnitten, und FWI / ISI nutzen 255 zugleich als NoData
(fwi255_check.py). Diese Datei klaert zuerst, was sich aus den vorhandenen Daten zurueckgewinnen
laesst (--mode probe), und baut dann die ungekappten Werte (--mode build, folgt).

probe  (nur lesen, Manifest p33_fwi_float_probe)
  roh      Datentyp, Baender, Tags, scale/offset, NoData der Roh-GeoTIFFs FWI / ERA5 / VPD
           an einigen Tagen; je Tag und Band Minimum / Maximum ueber das Jahr (Hinweis auf eine
           Skalierung je Datei statt einer festen Skalierung)
  hist     Haeufigkeit jedes uint8-Werts je FWI-Band (Land, Saison 2023, jeder 5. Tag)
  bezug    Die FWI-Komponenten haengen deterministisch voneinander ab (Van Wagner 1987):
             BUI = f(DMC, DC),  FWI = g(ISI, BUI)
           Mit unbekannten Faktoren u = s_X * X werden s_X per kleinster Quadrate auf Zelltagen
           ohne 0 / 255 geschaetzt. Passen die Gleichungen, sind die physikalischen Skalen bekannt.
  dc       Tageszuwachs von DC an trockenen Tagen gegen die ERA5-Temperatur (Van Wagner: linear
           in der Mittagstemperatur); R^2 zeigt, ob sich DC ueber die Kappung hinaus fortschreiben laesst

Befund der Proben (Jobs 4933, 4939): Die Roh-GeoTIFFs FWI / ERA5 / VPD sind selbst uint8 ohne
Metadaten, auf demselben Gitter wie der Wuerfel - es gibt lokal KEINE Gleitkomma-FWI. Auch ERA5 ist
uint8 mit unbekannter Kodierung (Tagesaggregate, keine Mittagswerte), eine physikalische Neuberechnung
des FWI-Systems ist daraus nicht sauber moeglich. Die Baender haengen aber in Code-Einheiten affin
an den CFFDRS-Gleichungen (BUI aus DMC / DC und FWI aus ISI / BUI je RMSE < 1 Code). Daraus folgt der
Weg hier: gekappte Werte werden IN CODE-EINHEITEN ueber 255 hinaus geschaetzt.

fit    (Manifest p33_fwi_float_fit, Parameter derived_data/fwi_float/params{tag}.json)
  Abbildungen in Code-Einheiten, u = s X + o je Band, X durch die CFFDRS-Gleichung:
    BUI  aus DMC, DC                     bui_eq
    FWI  aus ISI, BUI                    fwi_eq
    ISI  aus FFMC und Wind (u10, v10)    isi_eq, Wind W = a * |(u - cu, v - cv)|
  DC und DMC: Tageszuwachs als lineare Regression auf t2m_max, Niederschlag, Vorwert und Monat
  (gelernt auf ungekappten Paaren mit hohem Vorwert). Pruefung auf anderen Jahren mit einer
  kuenstlichen Kappung bei einem niedrigeren Code (fortgeschrieben gegen wahr).
build  (je Jahr, derived_data/fwi_float/uncapped_<jahr>.npz, nodata_<jahr>.npy, Manifest p33_fwi_float_build_<jahr>)
  Zelltage mit 255 (ausser NoData) werden geschaetzt, nie unter 255:
    DC, DMC   fortgeschrieben ab dem letzten ungekappten Tag (Zuwachsmodell)
    BUI       aus DMC / DC (beobachtet oder geschaetzt)
    ISI       aus FFMC und Wind
    FWI       aus ISI / BUI (beobachtet oder geschaetzt)
  NoData-Zellen (FWI und ISI an allen Saisontagen 255) bekommen ISI / FWI aus denselben
  Abbildungen, mit eigenem Flag. Gespeichert werden nur geaenderte Zelltage (duenn).
summary  (Manifest p33_fwi_float, derived_data/fwi_float/nodata_mask.npy)
  Jahresuebersicht, NoData-Maske ueber alle Jahre, Stabilitaet der NoData-Zellen.

Lesen: UncappedFWI(exp).window(cube, L, r, c) -> (T, 6, 13, 13) float32 FFMC..FWI in Code-Einheiten
(Kanaele 19-24 von Wuerfel und Cache), Werte > 255 moeglich; Normierung wie bisher (Code-Einheiten).
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile
from scipy.optimize import least_squares

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader

FWI_CH = {"FFMC": 19, "DMC": 20, "DC": 21, "ISI": 22, "BUI": 23, "FWI": 24}
ERA_CH = {"t2m": 1, "u10": 2, "v10": 3, "snow": 4, "precip": 5, "d2m": 7, "t2m_max": 13}


# ------------------------------------------------------------------ CFFDRS (Van Wagner 1987)
def bui_eq(P, D):
    """Build-up Index aus DMC (P) und DC (D)."""
    P, D = np.asarray(P, np.float64), np.asarray(D, np.float64)
    U = np.where(P <= 0.4 * D, 0.8 * P * D / np.maximum(P + 0.4 * D, 1e-9),
                 P - (1 - 0.8 * D / np.maximum(P + 0.4 * D, 1e-9)) * (0.92 + (0.0114 * P) ** 1.7))
    return np.maximum(U, 0.0)


def fwi_eq(R, U):
    """Fire Weather Index aus ISI (R) und BUI (U)."""
    R, U = np.asarray(R, np.float64), np.asarray(U, np.float64)
    fD = np.where(U <= 80, 0.626 * U ** 0.809 + 2, 1000 / (25 + 108.64 * np.exp(-0.023 * U)))
    B = 0.1 * R * fD
    return np.where(B > 1, np.exp(2.72 * (0.434 * np.log(np.maximum(B, 1.0 + 1e-9))) ** 0.647), B)


def isi_eq(Fm, W):
    """Initial Spread Index aus FFMC (Fm) und Wind W (km/h)."""
    Fm = np.clip(np.asarray(Fm, np.float64), 0.0, 101.0)
    m = 147.2 * (101 - Fm) / (59.5 + Fm)
    fF = 91.9 * np.exp(-0.1386 * m) * (1 + m ** 5.31 / 4.93e7)
    return 0.208 * np.exp(0.05039 * np.asarray(W, np.float64)) * fF


# ------------------------------------------------------------------ probe
def raw_probe(raw: Path, year: int, days: list[str]) -> dict:
    """Profil der Roh-GeoTIFFs und Minimum / Maximum je Tag und Band ueber das ganze Jahr."""
    out = {}
    for mod in ("FWI", "ERA5", "VPD"):
        tar = raw / "Entire_Canada_Maps" / mod / f"{mod}__{year}.tar"
        prof, mins, maxs, names = {}, [], [], []
        with tarfile.open(tar, "r:") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith((".tif", ".tiff")):
                    continue
                with MemoryFile(tf.extractfile(m).read()) as mem, mem.open() as ds:
                    a = ds.read()
                    stem = Path(m.name).stem
                    names.append(stem)
                    mins.append(a.reshape(a.shape[0], -1).min(1).tolist())
                    maxs.append(a.reshape(a.shape[0], -1).max(1).tolist())
                    if stem[5:] in days:
                        prof[stem] = {
                            "dtype": list(ds.dtypes), "count": ds.count, "shape": [ds.height, ds.width],
                            "nodata": ds.nodata, "scales": list(ds.scales), "offsets": list(ds.offsets),
                            "descriptions": list(ds.descriptions), "tags": ds.tags(),
                            "band_tags": [ds.tags(b) for b in range(1, ds.count + 1)],
                            "crs": str(ds.crs), "transform": list(ds.transform)[:6],
                            "bands": [{"min": int(b.min()), "max": int(b.max()),
                                       "p1": float(np.percentile(b, 1)), "p50": float(np.percentile(b, 50)),
                                       "p99": float(np.percentile(b, 99)),
                                       "share0": float((b == 0).mean()), "share255": float((b == 255).mean())}
                                      for b in a],
                        }
        mins, maxs = np.array(mins), np.array(maxs)
        out[mod] = {"n_files": len(names), "first": names[0] if names else None, "last": names[-1] if names else None,
                    "profiles": prof,
                    "per_day_min_range": [[int(mins[:, b].min()), int(mins[:, b].max())] for b in range(mins.shape[1])],
                    "per_day_max_range": [[int(maxs[:, b].min()), int(maxs[:, b].max())] for b in range(maxs.shape[1])],
                    "days_with_max255": [int((maxs[:, b] == 255).sum()) for b in range(maxs.shape[1])]}
        print(f"[fwif] {mod} {year}: {len(names)} Dateien, Max je Tag "
              f"{out[mod]['per_day_max_range']}, Min je Tag {out[mod]['per_day_min_range']}", flush=True)
        for k, p in prof.items():
            print(f"    {k}: {p['dtype'][0]} x{p['count']} {p['shape']} nodata {p['nodata']} scales {p['scales'][:3]} "
                  f"offsets {p['offsets'][:3]} tags {p['tags']} desc {p['descriptions'][:6]}", flush=True)
    return out


def cube_sample(cube: CubeReader, land: np.ndarray, year: str, step: int, n_cells: int, seed: int = 0):
    """Zufaellige Landzellen, alle Saisontage: (T, n, 42) uint8 sowie die Tage."""
    lr, lc = np.nonzero(land)
    pick = np.sort(np.random.default_rng(seed).choice(lr.size, min(n_cells, lr.size), replace=False))
    dyn, _ = cube._arrays(year)
    days = cube.per_year[year]["days"]
    rows = np.stack([np.asarray(dyn[d][lr[pick], lc[pick]]) for d in range(0, dyn.shape[0], step)])
    return rows, [days[d] for d in range(0, dyn.shape[0], step)]


def fit_scales(fun, ins: list[np.ndarray], out: np.ndarray, x0) -> dict:
    """u = s * X ohne Offset: min || u_out / s_out - fun(u_in / s_in) || ueber log s."""
    def res(p):
        s = np.exp(p)
        return fun(*[u / si for u, si in zip(ins, s[:-1])]) - out / s[-1]
    r = least_squares(res, np.log(x0), loss="soft_l1", f_scale=1.0)
    s = np.exp(r.x)
    pred = fun(*[u / si for u, si in zip(ins, s[:-1])]) * s[-1]
    return {"scales": s.tolist(), "rmse_code": float(np.sqrt(np.mean((pred - out) ** 2))),
            "mae_code": float(np.mean(np.abs(pred - out))), "share_within_1": float((np.abs(pred - out) <= 1.0).mean()),
            "corr": float(np.corrcoef(pred, out)[0, 1]), "n": int(out.size)}


def probe(args, cfg, exp) -> None:
    raw = Path(cfg["dataset_root"])
    res = {"raw": {str(y): raw_probe(raw, y, ["01_15", "04_01", "07_15", "08_20", "12_15"]) for y in args.raw_years}}
    cube = CubeReader(exp)
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    res["cube"] = {}
    for y in args.years:
        v, days = cube_sample(cube, land, y, 1, args.n_cells)                 # (T, n, 42)
        T = v.shape[0]
        hist = {k: np.bincount(v[::5, :, c].ravel(), minlength=256).tolist() for k, c in FWI_CH.items()}
        hist.update({k: np.bincount(v[::5, :, c].ravel(), minlength=256).tolist() for k, c in ERA_CH.items()})
        f = {k: v[:, :, c].astype(np.float64) for k, c in FWI_CH.items()}
        ok = lambda *ks: np.logical_and.reduce([(f[k] > 0) & (f[k] < 255) for k in ks])  # noqa: E731
        rs = np.random.default_rng(1)
        sub = lambda m: rs.choice(np.flatnonzero(m), min(args.n_fit, int(m.sum())), replace=False)  # noqa: E731
        m = sub(ok("DMC", "DC", "BUI").ravel())
        bui = fit_scales(bui_eq, [f["DMC"].ravel()[m], f["DC"].ravel()[m]], f["BUI"].ravel()[m], [1.0, 1.0, 1.0])
        m = sub(ok("ISI", "BUI", "FWI").ravel())
        fwi = fit_scales(fwi_eq, [f["ISI"].ravel()[m], f["BUI"].ravel()[m]], f["FWI"].ravel()[m], [1.0, 1.0, 1.0])
        # Identitaet zum Vergleich (u = X)
        ident = {
            "bui": float(np.sqrt(np.mean((bui_eq(f["DMC"], f["DC"])[ok("DMC", "DC", "BUI")] - f["BUI"][ok("DMC", "DC", "BUI")]) ** 2))),
            "fwi": float(np.sqrt(np.mean((fwi_eq(f["ISI"], f["BUI"])[ok("ISI", "BUI", "FWI")] - f["FWI"][ok("ISI", "BUI", "FWI")]) ** 2))),
        }
        # DC-Tageszuwachs an trockenen Tagen gegen Temperatur
        dc = f["DC"]
        pr = v[:, :, ERA_CH["precip"]].astype(np.float64)
        dry_code = int(np.argmax(np.bincount(pr.ravel().astype(np.int64), minlength=256)))
        inc = dc[1:] - dc[:-1]
        mm = (dc[1:] > 2) & (dc[1:] < 250) & (dc[:-1] > 2) & (dc[:-1] < 250) & (pr[1:] == dry_code)
        month = np.array([int(d[5:7]) for d in days])[1:, None].repeat(dc.shape[1], 1)
        dcfit = {}
        for tk in ("t2m", "t2m_max"):
            tt = v[1:, :, ERA_CH[tk]].astype(np.float64)
            Xd = np.column_stack([tt[mm], *[(month[mm] == mo).astype(float) for mo in range(4, 10)]])
            coef, *_ = np.linalg.lstsq(Xd, inc[mm], rcond=None)
            pred = Xd @ coef
            dcfit[tk] = {"slope_per_code": float(coef[0]), "month_terms": coef[1:].tolist(),
                         "r2": float(1 - np.var(inc[mm] - pred) / np.var(inc[mm])), "n": int(mm.sum())}
        # Laeufe bei 255: wie lange bleibt DC gekappt
        sat = dc == 255
        res["cube"][y] = {
            "n_cells": int(v.shape[1]), "n_days": T, "hist": hist, "fit_bui": bui, "fit_fwi": fwi,
            "rmse_identity": ident, "dc_dry_code": dry_code, "dc_increment": dcfit,
            "share_255": {k: float((f[k] == 255).mean()) for k in FWI_CH},
            "share_0": {k: float((f[k] == 0).mean()) for k in FWI_CH},
            "dc_sat_cells": float(sat.any(0).mean()),
        }
        print(f"[fwif] Wuerfel {y}: BUI-Fit Skalen (DMC, DC, BUI) {np.round(bui['scales'], 4)} RMSE {bui['rmse_code']:.2f} "
              f"Code, |Fehler|<=1 {bui['share_within_1']:.3f}; FWI-Fit (ISI, BUI, FWI) {np.round(fwi['scales'], 4)} "
              f"RMSE {fwi['rmse_code']:.2f}, <=1 {fwi['share_within_1']:.3f}; Identitaet RMSE {ident}", flush=True)
        print(f"[fwif]   DC-Zuwachs trocken (precip-Code {dry_code}): " + ", ".join(
            f"{k}: {d['slope_per_code']:.4f}/Code R2 {d['r2']:.3f} n {d['n']:,}" for k, d in dcfit.items()), flush=True)
        for k in FWI_CH:
            h = np.array(hist[k]); nz = np.flatnonzero(h)
            print(f"[fwif]   {k}: Werte {nz.min()}..{nz.max()}, Anteil 0 {h[0] / h.sum():.4f}, 255 {h[255] / h.sum():.4f}, "
                  f"p50 {int(np.searchsorted(np.cumsum(h), h.sum() / 2))}, p99 {int(np.searchsorted(np.cumsum(h), 0.99 * h.sum()))}",
                  flush=True)
    write_manifest(cfg, "p33_fwi_float_probe", args.config, {"years": args.years, "raw_years": args.raw_years, **res})


def fit_affine(fun, ins: list[np.ndarray], out: np.ndarray, s0, o0) -> dict:
    """Wie fit_scales, aber u = s * X + o: X = (u - o) / s, Startwerte s0, o0 je Band (Eingaenge, Ausgang)."""
    k = len(s0)

    def phys(u, s, o):
        return np.maximum((u - o) / s, 0.0)

    def res(p):
        s, o = np.exp(p[:k]), p[k:]
        return fun(*[phys(u, s[i], o[i]) for i, u in enumerate(ins)]) * s[-1] + o[-1] - out
    r = least_squares(res, np.concatenate([np.log(s0), o0]), loss="soft_l1", f_scale=1.0)
    s, o = np.exp(r.x[:k]), r.x[k:]
    pred = res(r.x) + out
    return {"scales": s.tolist(), "offsets": o.tolist(), "rmse_code": float(np.sqrt(np.mean((pred - out) ** 2))),
            "share_within_1": float((np.abs(pred - out) <= 1.0).mean()), "corr": float(np.corrcoef(pred, out)[0, 1]),
            "n": int(out.size)}


def probe2(args, cfg, exp) -> None:
    """Zweiter Durchgang: Skalen mit Offset und physikalisch plausiblen Startwerten; wie viele gekappte
    Zelltage lassen sich aus den uebrigen Baendern zurueckrechnen; mittlerer DC-Zuwachs an trockenen Tagen."""
    cube = CubeReader(exp)
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    res = {}
    for y in args.years:
        v, days = cube_sample(cube, land, y, 1, args.n_cells)
        f = {k: v[:, :, c].astype(np.float64) for k, c in FWI_CH.items()}
        ok = lambda *ks: np.logical_and.reduce([(f[k] > 0) & (f[k] < 255) for k in ks])  # noqa: E731
        rs = np.random.default_rng(1)
        sub = lambda m: rs.choice(np.flatnonzero(m), min(args.n_fit, int(m.sum())), replace=False)  # noqa: E731
        out = {"bui_scale": {}, "bui_affine": {}, "fwi_affine": {}}
        m = sub(ok("DMC", "DC", "BUI").ravel())
        ins, o = [f["DMC"].ravel()[m], f["DC"].ravel()[m]], f["BUI"].ravel()[m]
        for s0 in ([2.0, 0.3, 2.0], [1.0, 0.25, 1.0], [3.0, 0.5, 3.0]):
            out["bui_scale"][str(s0)] = fit_scales(bui_eq, ins, o, s0)
        out["bui_affine"] = fit_affine(bui_eq, ins, o, [2.0, 0.3, 2.0], [0.0, 0.0, 0.0])
        m = sub(ok("ISI", "BUI", "FWI").ravel())
        out["fwi_affine"] = fit_affine(fwi_eq, [f["ISI"].ravel()[m], f["BUI"].ravel()[m]], f["FWI"].ravel()[m],
                                       [21.0, 2.0, 7.8], [0.0, 0.0, 0.0])
        # NoData: Zellen mit FWI und ISI an allen Saisontagen 255
        nod = (f["FWI"] == 255).all(0) & (f["ISI"] == 255).all(0)
        cap = lambda k: (f[k] == 255) & ~nod[None]  # noqa: E731
        out["nodata_cells"] = float(nod.mean())
        out["capped_share"] = {k: float(cap(k).mean()) for k in FWI_CH}
        out["fwi_capped_recoverable"] = float(((f["ISI"] < 255) & (f["BUI"] < 255))[cap("FWI")].mean()) if cap("FWI").any() else None
        out["bui_capped_recoverable"] = float(((f["DMC"] < 255) & (f["DC"] < 255))[cap("BUI")].mean()) if cap("BUI").any() else None
        out["isi_capped_share_of_fwi_capped"] = float((f["ISI"] == 255)[cap("FWI")].mean()) if cap("FWI").any() else None
        dc = f["DC"]; pr = v[:, :, ERA_CH["precip"]]
        inc = dc[1:] - dc[:-1]
        mm = (dc[1:] > 2) & (dc[1:] < 250) & (dc[:-1] > 2) & (dc[:-1] < 250) & (pr[1:] == 0)
        mon = np.array([int(d[5:7]) for d in days])[1:, None].repeat(dc.shape[1], 1)
        out["dc_dry_mean_increment_code"] = {str(mo): float(inc[mm & (mon == mo)].mean()) for mo in range(4, 10)}
        dm = f["DMC"]; incm = dm[1:] - dm[:-1]
        mm2 = (dm[1:] > 2) & (dm[1:] < 250) & (dm[:-1] > 2) & (dm[:-1] < 250) & (pr[1:] == 0)
        out["dmc_dry_mean_increment_code"] = {str(mo): float(incm[mm2 & (mon == mo)].mean()) for mo in range(4, 10)}
        # Laenge der Laeufe mit DC = 255 (ohne NoData)
        c = cap("DC"); runs = []
        for j in np.flatnonzero(c.any(0)):
            r = 0
            for t in range(c.shape[0]):
                r = r + 1 if c[t, j] else (runs.append(r) or 0) if r else 0
            if r:
                runs.append(r)
        out["dc_cap_runs"] = {"n": len(runs), "median": float(np.median(runs)) if runs else None,
                              "p90": float(np.percentile(runs, 90)) if runs else None}
        res[y] = out
        print(f"[fwif2] {y}: NoData-Zellen {out['nodata_cells']:.4f}; gekappt (ohne NoData) "
              + ", ".join(f"{k} {s:.4f}" for k, s in out["capped_share"].items()), flush=True)
        for s0, r in out["bui_scale"].items():
            print(f"[fwif2]   BUI nur Skalen, Start {s0}: {np.round(r['scales'], 4)} RMSE {r['rmse_code']:.2f} <=1 {r['share_within_1']:.3f}", flush=True)
        for k in ("bui_affine", "fwi_affine"):
            r = out[k]
            print(f"[fwif2]   {k}: s {np.round(r['scales'], 4)} o {np.round(r['offsets'], 2)} RMSE {r['rmse_code']:.2f} "
                  f"<=1 {r['share_within_1']:.3f}", flush=True)
        print(f"[fwif2]   FWI gekappt, aber ISI und BUI nicht: {out['fwi_capped_recoverable']}; BUI gekappt, DMC und DC nicht: "
              f"{out['bui_capped_recoverable']}; ISI auch gekappt: {out['isi_capped_share_of_fwi_capped']}", flush=True)
        print(f"[fwif2]   DC-Zuwachs trocken je Monat (Code): {out['dc_dry_mean_increment_code']}; DMC: "
              f"{out['dmc_dry_mean_increment_code']}; DC-Kappungslaeufe {out['dc_cap_runs']}", flush=True)
    write_manifest(cfg, "p33_fwi_float_probe2", args.config, {"years": args.years, "results": res})


# ------------------------------------------------------------------ Abbildungen in Code-Einheiten
EST_BANDS = ("DMC", "DC", "ISI", "BUI", "FWI")                # Spalten von est in uncapped_<jahr>.npz
FLAG = {"DMC": 1, "DC": 2, "ISI": 4, "BUI": 8, "FWI": 16, "nodata_fill": 32, "run_from_start": 64}


def _phys(u, s, o):
    return np.maximum((np.asarray(u, np.float64) - o) / s, 0.0)


def map_bui(dmc, dc, p):
    s, o = p["scales"], p["offsets"]
    return s[2] * bui_eq(_phys(dmc, s[0], o[0]), _phys(dc, s[1], o[1])) + o[2]


def map_fwi(isi, bui, p):
    s, o = p["scales"], p["offsets"]
    return s[2] * fwi_eq(_phys(isi, s[0], o[0]), _phys(bui, s[1], o[1])) + o[2]


def map_isi(ffmc, u, v, p):
    W = p["a"] * np.hypot(np.asarray(u, np.float64) - p["cu"], np.asarray(v, np.float64) - p["cv"])
    return p["sI"] * isi_eq((np.asarray(ffmc, np.float64) - p["oF"]) / p["sF"], W) + p["oI"]


def inc_X(prev, tmax, pr, month: int):
    """Merkmale des Tageszuwachses von DC / DMC (Wetter des neuen Tags, Vorwert, Monat)."""
    prev, tmax, pr = (np.asarray(a, np.float64) for a in (prev, tmax, pr))
    mon = np.broadcast_to(np.asarray(month), prev.shape)
    return np.stack([np.ones_like(prev), tmax, pr, pr ** 2, prev, prev * pr, prev * tmax]
                    + [(mon == m).astype(np.float64) for m in range(5, 10)], -1)


def extrapolate(obs, cap, tmax, pr, months, coef, floor: float):
    """Gekappte Tage (cap) fortschreiben: est_t = max(floor, est_{t-1} + Zuwachs_t). Am ersten Saisontag
    gekappt -> floor (untere Schranke), Flag start. obs (T, n) -> est (T, n), start (T, n)."""
    est, start = obs.astype(np.float64).copy(), np.zeros(obs.shape, bool)
    coef = np.asarray(coef)
    for t in range(obs.shape[0]):
        c = cap[t]
        if not c.any():
            continue
        if t == 0:
            est[0, c], start[0, c] = floor, True
            continue
        prev = est[t - 1, c]
        est[t, c] = np.maximum(floor, prev + inc_X(prev, tmax[t, c], pr[t, c], months[t]) @ coef)
        start[t, c] = start[t - 1, c]
    return est, start


def err_stats(pred, true) -> dict:
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    if true.size < 2:
        return {"n": int(true.size)}
    return {"n": int(true.size), "rmse": float(np.sqrt(np.mean((pred - true) ** 2))), "mae": float(np.mean(np.abs(pred - true))),
            "bias": float(np.mean(pred - true)), "corr": float(np.corrcoef(pred, true)[0, 1]),
            "within_1": float((np.abs(pred - true) <= 1).mean()), "within_5": float((np.abs(pred - true) <= 5).mean())}


def sample_years(cube, land, years, n_cells, seed):
    vs = [cube_sample(cube, land, y, 1, n_cells, seed=seed + i)[0] for i, y in enumerate(years)]
    days = cube.per_year[years[0]]["days"]
    return np.concatenate(vs, 1), np.array([int(d[5:7]) for d in days])


def fit(args, cfg, exp) -> None:
    out_dir = exp / "derived_data" / "fwi_float"
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = out_dir / f"params{args.tag}.json"
    if pf.exists():
        raise SystemExit(f"ABBRUCH: {pf} existiert schon (nichts ueberschreiben; anderes --tag)")
    cube = CubeReader(exp)
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    t0 = time.time()
    V, months = sample_years(cube, land, args.fit_years, args.n_cells, 0)
    print(f"[fwif] Fit-Stichprobe {V.shape} aus {args.fit_years} ({time.time() - t0:.0f} s)", flush=True)
    f = {k: V[:, :, c].astype(np.float64) for k, c in FWI_CH.items()}
    e = {k: V[:, :, c].astype(np.float64) for k, c in ERA_CH.items()}
    nod = ((f["FWI"] == 255).all(0) & (f["ISI"] == 255).all(0))[None]
    ok = lambda *ks: np.logical_and.reduce([f[k] < 255 for k in ks]) & ~nod  # noqa: E731
    rs = np.random.default_rng(1)
    sub = lambda m: rs.choice(np.flatnonzero(m), min(args.n_fit, int(m.sum())), replace=False)  # noqa: E731
    R = lambda a, m: a.ravel()[m]  # noqa: E731
    P = {"fit_years": args.fit_years, "units": "Code-Einheiten des Wuerfels (uint8-Skala, ueber 255 fortgesetzt)"}

    m = sub(ok("DMC", "DC", "BUI").ravel() & (f["DMC"] > 0).ravel() & (f["DC"] > 0).ravel())
    P["bui"] = fit_affine(bui_eq, [R(f["DMC"], m), R(f["DC"], m)], R(f["BUI"], m), [1.9, 0.33, 1.5], [0.0, 0.0, 0.0])
    m = sub(ok("ISI", "BUI", "FWI").ravel() & (f["ISI"] > 0).ravel())
    P["fwi"] = fit_affine(fwi_eq, [R(f["ISI"], m), R(f["BUI"], m)], R(f["FWI"], m), [20.0, 2.0, 7.9], [0.0, 0.0, 0.0])
    m = sub(ok("FFMC", "ISI").ravel() & (f["FFMC"] > 0).ravel())
    FF, U, Vv, I = R(f["FFMC"], m), R(e["u10"], m), R(e["v10"], m), R(f["ISI"], m)
    unpack = lambda p: {"sF": float(np.exp(p[0])), "oF": float(p[1]), "a": float(np.exp(p[2])), "cu": float(p[3]),  # noqa: E731
                        "cv": float(p[4]), "sI": float(np.exp(p[5])), "oI": float(p[6])}
    x0 = [np.log(2.5), 0.0, np.log(0.3), float(np.median(U)), float(np.median(Vv)), np.log(19.0), 0.0]
    r = least_squares(lambda p: map_isi(FF, U, Vv, unpack(p)) - I, x0, loss="soft_l1", f_scale=1.0)
    P["isi"] = {**unpack(r.x), **{k: v for k, v in err_stats(map_isi(FF, U, Vv, unpack(r.x)), I).items()}}
    P["isi_ok"] = bool(P["isi"]["rmse"] <= args.isi_max_rmse)
    print(f"[fwif] BUI s {np.round(P['bui']['scales'], 4)} o {np.round(P['bui']['offsets'], 2)} RMSE {P['bui']['rmse_code']:.2f}; "
          f"FWI s {np.round(P['fwi']['scales'], 4)} o {np.round(P['fwi']['offsets'], 2)} RMSE {P['fwi']['rmse_code']:.2f}; "
          f"ISI {({k: round(v, 3) for k, v in P['isi'].items()})} -> verwendbar {P['isi_ok']}", flush=True)

    # Zuwachsmodelle DC / DMC auf ungekappten Paaren mit hohem Vorwert
    P["inc"] = {}
    for k, lo, hi in (("DC", args.dc_lo, 235), ("DMC", args.dmc_lo, 225)):
        x = f[k]
        mm = (x[:-1] >= lo) & (x[:-1] <= hi) & (x[1:] < 255)
        mon = np.broadcast_to(months[1:, None], x[1:].shape)
        Xd = np.concatenate([inc_X(x[:-1][mm & (mon == mo)], e["t2m_max"][1:][mm & (mon == mo)],
                                   e["precip"][1:][mm & (mon == mo)], mo) for mo in range(4, 10)])
        yd = np.concatenate([(x[1:] - x[:-1])[mm & (mon == mo)] for mo in range(4, 10)])
        coef, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
        P["inc"][k] = {"coef": coef.tolist(), "prev_range": [lo, hi], "n": int(yd.size),
                       "r2": float(1 - np.var(yd - Xd @ coef) / np.var(yd)),
                       "features": "1, t2m_max, precip, precip^2, prev, prev*precip, prev*t2m_max, Monat 5..9"}
        print(f"[fwif] Zuwachs {k}: R2 {P['inc'][k]['r2']:.3f} auf {yd.size:,} Paaren", flush=True)

    # Pruefung auf anderen Jahren
    Vv_, months_v = sample_years(cube, land, args.val_years, args.n_cells // 2, 100)
    fv = {k: Vv_[:, :, c].astype(np.float64) for k, c in FWI_CH.items()}
    ev = {k: Vv_[:, :, c].astype(np.float64) for k, c in ERA_CH.items()}
    nodv = ((fv["FWI"] == 255).all(0) & (fv["ISI"] == 255).all(0))[None]
    okv = lambda *ks: np.logical_and.reduce([fv[k] < 255 for k in ks]) & ~nodv  # noqa: E731
    val = {"years": args.val_years, "mapping": {}, "pseudo_cap": {}}
    for name, pred, true, m in (
            ("BUI", lambda: map_bui(fv["DMC"], fv["DC"], P["bui"]), fv["BUI"], okv("DMC", "DC", "BUI")),
            ("FWI", lambda: map_fwi(fv["ISI"], fv["BUI"], P["fwi"]), fv["FWI"], okv("ISI", "BUI", "FWI")),
            ("ISI", lambda: map_isi(fv["FFMC"], ev["u10"], ev["v10"], P["isi"]), fv["ISI"], okv("FFMC", "ISI"))):
        pv = pred()
        val["mapping"][name] = {"all": err_stats(pv[m], true[m]),
                                "true_ge_200": err_stats(pv[m & (true >= 200)], true[m & (true >= 200)])}
        print(f"[fwif] Pruefung {name}: alle {val['mapping'][name]['all']}; wahr >= 200 {val['mapping'][name]['true_ge_200']}",
              flush=True)
    for k, Pc in (("DC", args.pseudo_cap_dc), ("DMC", args.pseudo_cap_dmc)):
        x = fv[k]
        cap = x >= Pc
        est, _ = extrapolate(x, cap, ev["t2m_max"], ev["precip"], months_v, P["inc"][k]["coef"], float(Pc))
        run = np.zeros(x.shape, np.int32)                  # Tag im Kappungslauf
        for t in range(x.shape[0]):
            run[t] = np.where(cap[t], (run[t - 1] if t else 0) + 1, 0)
        good = cap & (x < 255) & (run > 0)
        res_k = {"cap_code": Pc, "all": err_stats(est[good], x[good])}
        for lo, hi in ((1, 5), (6, 20), (21, 60), (61, 183)):
            mm = good & (run >= lo) & (run <= hi)
            res_k[f"run_{lo}_{hi}"] = err_stats(est[mm], x[mm])
        val["pseudo_cap"][k] = res_k
        print(f"[fwif] Pseudo-Kappung {k} bei {Pc}: " + "; ".join(
            f"{b} MAE {s.get('mae', float('nan')):.1f} Bias {s.get('bias', float('nan')):+.1f} n {s['n']:,}"
            for b, s in res_k.items() if isinstance(s, dict)), flush=True)
    P["validation"] = val
    pf.write_text(json.dumps(P, indent=1))
    write_manifest(cfg, f"p33_fwi_float_fit{args.tag}", args.config, {"params_file": str(pf), **P})


def build(args, cfg, exp) -> None:
    out_dir = exp / "derived_data" / "fwi_float"
    P = json.loads((out_dir / f"params{args.tag}.json").read_text())
    y = str(args.year)
    of, onod = out_dir / f"uncapped_{y}{args.tag}.npz", out_dir / f"nodata_{y}{args.tag}.npy"
    if of.exists() or onod.exists():
        raise SystemExit(f"ABBRUCH: {of} existiert schon (nichts ueberschreiben)")
    cube = CubeReader(exp)
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    dyn, _ = cube._arrays(y)
    T, Hg, Wg, _ = dyn.shape
    months = np.array([int(d[5:7]) for d in cube.per_year[y]["days"]])
    nod = np.zeros((Hg, Wg), bool)
    keys, ests, flags = [], [], []
    rs = np.random.default_rng(int(y))
    st = {k: [[], []] for k in ("BUI", "FWI", "ISI")}            # Selbsttest: Vorhersage, Wuerfel
    capped_vals = {k: [] for k in EST_BANDS}
    fill_vals = {"ISI": [], "FWI": []}
    n_cd = n_cd_valid = 0
    n_cap = {k: 0 for k in EST_BANDS}
    t0 = time.time()
    for r0 in range(0, Hg, args.block):
        r1 = min(Hg, r0 + args.block)
        rr, cc = np.nonzero(land[r0:r1])
        if rr.size == 0:
            continue
        slab = np.asarray(dyn[:, r0:r1])                     # (T, b, W, 42)
        v = slab[:, rr, cc]                                  # (T, n, 42)
        del slab
        f = {k: v[:, :, c].astype(np.float64) for k, c in FWI_CH.items()}
        e = {k: v[:, :, c].astype(np.float64) for k, c in ERA_CH.items()}
        nodc = (f["FWI"] == 255).all(0) & (f["ISI"] == 255).all(0)
        nod[r0 + rr, cc] = nodc
        nd = nodc[None]
        cap = {"DMC": f["DMC"] == 255, "DC": f["DC"] == 255, "BUI": f["BUI"] == 255,
               "ISI": (f["ISI"] == 255) & ~nd, "FWI": (f["FWI"] == 255) & ~nd}
        DMe, DMs = extrapolate(f["DMC"], cap["DMC"], e["t2m_max"], e["precip"], months, P["inc"]["DMC"]["coef"], 255.0)
        DCe, DCs = extrapolate(f["DC"], cap["DC"], e["t2m_max"], e["precip"], months, P["inc"]["DC"]["coef"], 255.0)
        BUIe = f["BUI"].copy()
        m = cap["BUI"]
        BUIe[m] = np.maximum(255.0, map_bui(DMe[m], DCe[m], P["bui"]))
        ISIe = f["ISI"].copy()
        fill = (f["ISI"] == 255) & nd
        isi_pred = map_isi(f["FFMC"], e["u10"], e["v10"], P["isi"]) if P["isi_ok"] else np.full(f["ISI"].shape, np.nan)
        ISIe[cap["ISI"]] = np.maximum(255.0, isi_pred[cap["ISI"]])
        ISIe[fill] = isi_pred[fill]
        FWIe = f["FWI"].copy()
        m = cap["FWI"]
        FWIe[m] = np.maximum(255.0, map_fwi(ISIe[m], BUIe[m], P["fwi"]))
        fillf = (f["FWI"] == 255) & nd
        FWIe[fillf] = map_fwi(ISIe[fillf], BUIe[fillf], P["fwi"])
        fl = np.zeros(f["DC"].shape, np.uint8)
        for k in EST_BANDS:
            fl |= cap[k].astype(np.uint8) * FLAG[k]
        fl |= (fill | fillf).astype(np.uint8) * FLAG["nodata_fill"]
        fl |= ((DMs & cap["DMC"]) | (DCs & cap["DC"])).astype(np.uint8) * FLAG["run_from_start"]
        tt, jj = np.nonzero(fl)
        keys.append(tt.astype(np.int64) * Hg * Wg + (r0 + rr[jj]).astype(np.int64) * Wg + cc[jj])
        ests.append(np.stack([DMe[tt, jj], DCe[tt, jj], ISIe[tt, jj], BUIe[tt, jj], FWIe[tt, jj]], 1).astype(np.float32))
        flags.append(fl[tt, jj])
        n_cd += f["DC"].size
        n_cd_valid += int((~nd).sum()) * T
        for k, a in zip(EST_BANDS, (DMe, DCe, ISIe, BUIe, FWIe)):
            n_cap[k] += int(cap[k].sum())
            if cap[k].any():
                capped_vals[k].append(a[cap[k]].astype(np.float32))
        fill_vals["ISI"].append(ISIe[fill].astype(np.float32)); fill_vals["FWI"].append(FWIe[fillf].astype(np.float32))
        # Selbsttest gegen den Wuerfel: ungekappt, keine NoData
        ok = lambda *ks: np.logical_and.reduce([f[k] < 255 for k in ks]) & ~nd  # noqa: E731
        for name, pred, true, mm in (("BUI", lambda: map_bui(f["DMC"], f["DC"], P["bui"]), f["BUI"], ok("DMC", "DC", "BUI")),
                                     ("FWI", lambda: map_fwi(f["ISI"], f["BUI"], P["fwi"]), f["FWI"], ok("ISI", "BUI", "FWI")),
                                     ("ISI", lambda: isi_pred, f["ISI"], ok("FFMC", "ISI"))):
            idx = np.flatnonzero(mm)
            if idx.size:
                idx = rs.choice(idx, min(20000, idx.size), replace=False)
                st[name][0].append(pred().ravel()[idx]); st[name][1].append(true.ravel()[idx])
        print(f"    Zeilen {r0}-{r1}: {rr.size:,} Landzellen, {tt.size:,} geaenderte Zelltage ({time.time() - t0:.0f} s)", flush=True)
    keys = np.concatenate(keys); order = np.argsort(keys, kind="stable")
    ests, flags = np.concatenate(ests)[order], np.concatenate(flags)[order]
    keys = keys[order]
    np.savez_compressed(of, keys=keys, est=ests, flags=flags, bands=np.array(EST_BANDS), shape=np.array([T, Hg, Wg]))
    np.save(onod, nod)
    q = lambda a: ({"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),  # noqa: E731
                    "max": float(np.nanmax(a)), "share_above_256": float((a > 256).mean()), "share_nan": float(np.isnan(a).mean())}
                   if a.size else None)
    cv = {k: np.concatenate(v) if v else np.zeros(0, np.float32) for k, v in capped_vals.items()}
    fv = {k: np.concatenate(v) for k, v in fill_vals.items()}
    res = {
        "year": y, "n_cell_days": n_cd, "n_cell_days_valid": n_cd_valid, "nodata_cells": int(nod.sum()),
        "nodata_share_land": float(nod[land].mean()), "n_rows": int(keys.size),
        "capped_share": {k: n_cap[k] / max(n_cd_valid if k in ("ISI", "FWI") else n_cd, 1) for k in EST_BANDS},
        "capped_estimates": {k: q(cv[k][np.isfinite(cv[k])]) if cv[k].size else None for k in EST_BANDS},
        "capped_nan_share": {k: float(np.isnan(cv[k]).mean()) if cv[k].size else None for k in EST_BANDS},
        "nodata_fill": {k: q(a[np.isfinite(a)]) if np.isfinite(a).any() else None for k, a in fv.items()},
        "run_from_start_rows": int(((flags & FLAG["run_from_start"]) > 0).sum()),
        "self_test": {k: err_stats(np.concatenate(a[0]), np.concatenate(a[1])) if a[0] else None for k, a in st.items()},
    }
    print(f"[fwif] {y}: NoData {res['nodata_share_land']:.4f} der Landzellen; gekappt " + ", ".join(
        f"{k} {s:.4f}" for k, s in res["capped_share"].items()) + f"; Selbsttest {res['self_test']}", flush=True)
    for k, s in res["capped_estimates"].items():
        if s:
            print(f"    {k} gekappt -> Schaetzung p50 {s['p50']:.0f} p90 {s['p90']:.0f} p99 {s['p99']:.0f} max {s['max']:.0f}", flush=True)
    write_manifest(cfg, f"p33_fwi_float_build_{y}{args.tag}", args.config, {"params_file": f"params{args.tag}.json",
                                                                            "files": [str(of), str(onod)], **res})


def summary(args, cfg, exp) -> None:
    out_dir = exp / "derived_data" / "fwi_float"
    man = exp / "data_manifests"
    years = [str(y) for y in range(2000, 2026)]
    per, masks = {}, {}
    for y in years:
        p = man / f"p33_fwi_float_build_{y}{args.tag}.json"
        if not p.exists():
            print(f"[fwif] {y}: kein Build", flush=True)
            continue
        d = json.loads(p.read_text())
        per[y] = {k: d[k] for k in ("nodata_share_land", "capped_share", "capped_estimates", "capped_nan_share",
                                    "nodata_fill", "run_from_start_rows", "self_test")}
        masks[y] = np.load(out_dir / f"nodata_{y}{args.tag}.npy")
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    M = np.stack(list(masks.values()))
    allm, anym = M.all(0) & land, M.any(0) & land
    for nm, a in (("nodata_mask", allm), ("nodata_any", anym)):
        fp = out_dir / f"{nm}{args.tag}.npy"
        if fp.exists():
            raise SystemExit(f"ABBRUCH: {fp} existiert schon")
        np.save(fp, a)
    # Lesepfad pruefen: UncappedFWI.window gegen den Wuerfel an zufaelligen Stichtagen
    cube = CubeReader(exp)
    uf = UncappedFWI(exp, args.tag)
    rs = np.random.default_rng(7)
    lr, lc = np.nonzero(land)
    chk = {"n": 0, "equal_where_below_255": 0, "n_below_255": 0, "n_255": 0, "n_255_raised": 0, "n_255_nan_kept": 0}
    Ls = [L for L in cube._pos if cube._pos[L][0] in per and cube._pos[L][1] >= 29]
    for _ in range(300):
        L = int(rs.choice(Ls)); j = int(rs.integers(lr.size))
        w = uf.window(cube, L, int(lr[j]), int(lc[j]))
        year, li = cube._pos[L]
        dyn, _ = cube._arrays(year)
        raw = np.asarray(dyn[li - 29:li + 1, lr[j], lc[j], 19:25]).astype(np.float32)   # Mittelzelle (30, 6)
        cen = w[:, :, 6, 6]
        below = raw < 255
        chk["n"] += 1
        chk["n_below_255"] += int(below.sum()); chk["equal_where_below_255"] += int((cen[below] == raw[below]).sum())
        chk["n_255"] += int((~below).sum()); chk["n_255_raised"] += int((cen[~below] > 255).sum())
        chk["n_255_nan_kept"] += int((cen[~below] == 255).sum())
    print(f"[fwif] Lesepfad: {chk}", flush=True)
    res = {"years": list(per), "per_year": per, "read_check": chk,
           "nodata": {"all_years_share_land": float(allm[land].mean()), "any_year_share_land": float(anym[land].mean()),
                      "jaccard_all_any": float(allm.sum() / max(anym.sum(), 1))}}
    print(f"[fwif] NoData in allen Jahren {res['nodata']['all_years_share_land']:.4f}, in irgendeinem "
          f"{res['nodata']['any_year_share_land']:.4f} der Landzellen", flush=True)
    print("| Jahr | " + " | ".join(f"{k} gekappt / p99 / max" for k in EST_BANDS) + " |")
    for y, d in per.items():
        print(f"| {y} | " + " | ".join(
            f"{d['capped_share'][k]:.4f} / {d['capped_estimates'][k]['p99']:.0f} / {d['capped_estimates'][k]['max']:.0f}"
            if d["capped_estimates"][k] else f"{d['capped_share'][k]:.4f} / - / -" for k in EST_BANDS) + " |", flush=True)
    write_manifest(cfg, f"p33_fwi_float{args.tag}", args.config, res)


class UncappedFWI:
    """Fenster FFMC..FWI (Wuerfel-Kanaele 19-24) mit den ungekappten Schaetzungen, Code-Einheiten."""

    def __init__(self, exp: Path, tag: str = "", use_fill: bool = True):
        self.d = Path(exp) / "derived_data" / "fwi_float"
        self.tag, self.use_fill = tag, use_fill
        self._y: dict[str, tuple] = {}

    def _load(self, year: str):
        if year not in self._y:
            z = np.load(self.d / f"uncapped_{year}{self.tag}.npz")
            keys, est, fl = z["keys"], z["est"], z["flags"]
            if not self.use_fill:                       # NoData-Zellen: ISI und FWI bei 255 lassen
                est = est.copy()
                nf = (fl & FLAG["nodata_fill"]) > 0
                est[nf, EST_BANDS.index("ISI")] = 255.0
                est[nf, EST_BANDS.index("FWI")] = 255.0
            self._y[year] = (keys, est, tuple(int(s) for s in z["shape"]))
        return self._y[year]

    def window(self, cube: CubeReader, L: int, r: int, c: int, T: int = 30, half: int = 6) -> np.ndarray:
        """(T, 6, P, P) float32 bis einschliesslich L, Rand genullt wie surv_prepare.window."""
        year, li = cube._pos[L]
        dyn, _ = cube._arrays(year)
        keys, est, (_, Hg, Wg) = self._load(year)
        P = 2 * half + 1
        out = np.zeros((T, 6, P, P), np.float32)
        r0, r1 = max(0, r - half), min(Hg, r + half + 1)
        c0, c1 = max(0, c - half), min(Wg, c + half + 1)
        d = np.asarray(dyn[li - T + 1:li + 1, r0:r1, c0:c1, 19:25]).astype(np.float32)   # (T, h, w, 6)
        tt, rr, cc = np.meshgrid(np.arange(li - T + 1, li + 1), np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        k = (tt.astype(np.int64) * Hg + rr) * Wg + cc
        pos = np.minimum(np.searchsorted(keys, k.ravel()), keys.size - 1)
        hit = keys[pos] == k.ravel()
        if hit.any():
            e = est[pos[hit]]                                # (m, 5) DMC..FWI
            flat = d.reshape(-1, 6)
            sub = flat[hit]
            sub[:, 1:] = np.where(np.isfinite(e), e, sub[:, 1:])
            flat[hit] = sub
            d = flat.reshape(d.shape)
        out[:, :, r0 - (r - half):r1 - (r - half), c0 - (c - half):c1 - (c - half)] = d.transpose(0, 3, 1, 2)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["probe", "probe2", "fit", "build", "summary"], default="probe")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--tag", default="")
    ap.add_argument("--block", type=int, default=32, help="build: Zeilen je Block")
    ap.add_argument("--fit-years", nargs="+", default=["2000", "2003", "2006", "2009", "2012", "2015", "2018"])
    ap.add_argument("--val-years", nargs="+", default=["2001", "2010", "2016", "2019"])
    ap.add_argument("--dc-lo", type=float, default=120)
    ap.add_argument("--dmc-lo", type=float, default=100)
    ap.add_argument("--pseudo-cap-dc", type=float, default=180)
    ap.add_argument("--pseudo-cap-dmc", type=float, default=150)
    ap.add_argument("--isi-max-rmse", type=float, default=2.5, help="ISI-Abbildung nur bei RMSE <= so viele Codes verwenden")
    ap.add_argument("--years", nargs="+", default=["2010", "2023"])
    ap.add_argument("--raw-years", type=int, nargs="+", default=[2023])
    ap.add_argument("--n-cells", type=int, default=20000)
    ap.add_argument("--n-fit", type=int, default=200000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if args.mode == "probe":
        probe(args, cfg, exp)
    elif args.mode == "probe2":
        probe2(args, cfg, exp)
    elif args.mode == "fit":
        fit(args, cfg, exp)
    elif args.mode == "build":
        build(args, cfg, exp)
    else:
        summary(args, cfg, exp)


if __name__ == "__main__":
    main()

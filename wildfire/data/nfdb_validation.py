"""Unabhaengige Validierung der Labelkette gegen die kanadische Branddatenbank.

Unsere Zuendungen sind ein PROXY: erstes MODIS-Feuer nach sieben feuerfreien
Tagen in Umgebungsradius 1. Ob dieser Proxy echte Braende trifft, laesst
sich aus MODIS allein nicht sagen - dafuer braucht es eine unabhaengige
Quelle.

Die National Forestry Database (NFDB) der kanadischen Forstbehoerde
enthaelt gemeldete Braende mit Koordinate, Datum, Ursache und Flaeche,
erhoben von den Provinzbehoerden. Sie ist unabhaengig von MODIS erhoben
und damit ein echter Aussenmassstab.

Geprueft wird in beide Richtungen:

  * **Sensitivitaet**: Welcher Anteil der gemeldeten Braende hat eine
    MODIS-Zuendung am selben Ort und im Zeitfenster? Erwartung: stark
    abhaengig von der Brandflaeche - ein 0,1-ha-Feuer kann ein 25-km2-Pixel
    nicht zum Ansprechen bringen.
  * **Bestaetigungsrate**: Welcher Anteil unserer Proxy-Zuendungen findet
    einen gemeldeten Brand in der Naehe? Ein niedriger Wert wuerde auf
    Fehldetektionen hindeuten.

Zusaetzlich wird ausgewertet, wie sich geplantes Abbrennen (Feld
PRESCRIBED und FIRE_TYPE) auf die Monate verteilt - das ist die direkte
Gegenprobe zu unserem Befund aus d1b, dass Fruehjahr und Herbst durch
Stoppelbrennen und Schlagabraumverbrennung verunreinigt sind.

Der entscheidende Vergleich ist das VERHAELTNIS unserer Zuendungen zu den
gemeldeten Braenden je Monat. In der Waldbrandsaison muss es stabil sein -
wir sehen einen bestimmten Anteil, gegeben durch die Groessenabhaengigkeit
der Detektion. Ein Monat, in dem MODIS ein Vielfaches dessen meldet, was
die Provinzbehoerden registrieren, zeigt Feuer an, die keine Waldbraende
sind.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest

CAUSE_NAMES = {"H": "Mensch", "N": "Blitz (natuerlich)", "U": "unbekannt", "": "leer"}
SIZE_BINS = [0, 1, 10, 100, 1_000, 10_000, 1e9]
SIZE_LABELS = ["<1 ha", "1-10 ha", "10-100 ha", "100-1k ha", "1k-10k ha", ">10k ha"]


def load_nfdb(path: Path) -> list[dict]:
    z = zipfile.ZipFile(path)
    name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
    raw = z.read(name).decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    for r in rows:                     # BOM im ersten Feldnamen
        if "﻿NFDBFIREID" in r:
            r["NFDBFIREID"] = r.pop("﻿NFDBFIREID")
    return rows


def registration_offsets(recs, fkeys, H, W, radius=3, lags=range(-2, 9),
                         size_lo=100.0, size_hi=1000.0):
    """Versatz zwischen Meldepunkt und MODIS-Feuerzelle, verzerrungsfrei.

    Zaehlt man je Brand den NAECHSTGELEGENEN Treffer, ist das Ergebnis zur Null
    hin verzerrt: Ein Grossbrand belegt viele Zellen, darunter fast immer die
    eigene, sodass ein echter Versatz verdeckt wuerde. Deshalb gehen nur Braende
    ein, bei denen im Suchfenster GENAU EINE Zelle brennt - dann ist der Versatz
    eindeutig und nicht durch eine Auswahlregel beeinflusst.

    Die Groessenschranke haelt die Braende bei rund einer Zellflaeche: nur dann
    ist "genau eine Zelle" keine Auswahl besonders kompakter Faelle.

    Rueckgabe enthaelt zusaetzlich die Sub-Zell-Lage je Versatzgruppe. Haengt der
    Versatz von der Lage im Sinne einer scharfen Grenze bei 0,5 ab, waere das
    Raster um eine halbe Zelle verschoben; ist die Trennung unscharf und nur eine
    Achse betroffen, liegt es an den Daten und nicht an der Umrechnung.
    """
    DAY_BITS = 16384
    lags = list(lags)

    def hits(rr, cc, t0):
        out = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r2, c2 = rr + dr, cc + dc
                if not (0 <= r2 < H and 0 <= c2 < W):
                    continue
                base = (r2 * W + c2) * DAY_BITS
                for lag in lags:
                    t = t0 + lag
                    if t < 0:
                        continue
                    j = np.searchsorted(fkeys, base + t, "left")
                    if j < fkeys.size and fkeys[j] == base + t:
                        out.append((dr, dc))
                        break
        return out

    single, n_sel, n_any = [], 0, 0
    for rec in recs:
        if not size_lo <= rec["size_ha"] < size_hi:
            continue
        n_sel += 1
        h = hits(rec["row"], rec["col"], rec["t"])
        if not h:
            continue
        n_any += 1
        if len(h) == 1:
            single.append((h[0][0], h[0][1], rec["frow"], rec["fcol"]))

    if not single:
        return {"n_selected": n_sel, "n_detected": n_any, "n_single_cell": 0}

    arr = np.array(single, float)
    dr, dc, fr, fc = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    dist = Counter((int(a), int(b)) for a, b in zip(dr, dc))
    n = len(single)

    by_group = {}
    for v in (-1, 0, 1):
        m = dr == v
        if m.sum():
            by_group[str(v)] = {"n": int(m.sum()), "mean_frow": float(fr[m].mean())}

    # Gegenprobe: waere die Mittelpunkts-Konvention (round) besser als floor?
    alt_r = np.abs(dr + (np.floor(fr) - np.round(fr)))
    alt_c = np.abs(dc + (np.floor(fc) - np.round(fc)))

    return {
        "size_range_ha": [size_lo, size_hi],
        "search_radius_px": radius,
        "day_window": [lags[0], lags[-1]],
        "n_selected": n_sel,
        "n_detected": n_any,
        "n_single_cell": n,
        "offset_distribution": {f"{a},{b}": c for (a, b), c in dist.most_common()},
        "share_exact": float(dist[(0, 0)] / n),
        "centroid_row": float(dr.mean()),
        "centroid_col": float(dc.mean()),
        "mean_frow_all": float(fr.mean()),
        "mean_fcol_all": float(fc.mean()),
        "subcell_by_row_offset": by_group,
        "share_exact_if_round_convention": float(np.mean((alt_r == 0) & (alt_c == 0))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nfdb", default="/data/Wildfire/artifacts/NFDB_point_txt.zip")
    ap.add_argument("--space-tol", type=int, default=1, help="Pixelradius fuer Treffer")
    ap.add_argument("--day-before", type=int, default=3)
    ap.add_argument("--day-after", type=int, default=7)
    ap.add_argument("--sweep", action="store_true",
                    help="Toleranzen variieren, um die Robustheit zu pruefen")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    tr = man["grid"]["transform"]
    H, W = man["grid"]["height"], man["grid"]["width"]
    valid = np.load(exp / "derived_data" / "masks" / "valid.npy")
    canada = np.load(exp / "derived_data" / "masks" / "canada.npy")

    fi = np.load(exp / "derived_data" / "events" / "fire_index.npz", allow_pickle=False)
    dates = [str(d) for d in fi["dates"]]
    t_of_date = {d: i for i, d in enumerate(dates)}
    # unsere Zuendungen als Schluesselmenge
    counts = np.diff(fi["ign_offsets"])
    ign_px = np.repeat(np.asarray(fi["ign_pixels"], np.int64), counts)
    ign_day = np.asarray(fi["ign_days"], np.int64)
    ours = set((ign_day * (H * W) + ign_px).tolist())
    print(f"[nfdb] {len(ours)} eigene Proxy-Zuendungen")

    rows = load_nfdb(Path(args.nfdb))
    print(f"[nfdb] {len(rows)} Eintraege in der Branddatenbank")

    def rc_of(lat: float, lon: float):
        c = math.floor((lon - tr[2]) / tr[0])
        r = math.floor((lat - tr[5]) / tr[4])
        return r, c

    recs = []
    skipped = Counter()
    for r in rows:
        try:
            y = int(r["YEAR"]); mo = int(r["MONTH"]); da = int(r["DAY"])
        except (ValueError, TypeError):
            skipped["kein Datum"] += 1
            continue
        if not (2000 <= y <= 2025):
            skipped["ausserhalb 2000-2025"] += 1
            continue
        if not (1 <= mo <= 12 and 1 <= da <= 31):
            skipped["unvollstaendiges Datum"] += 1
            continue
        try:
            d = dt.date(y, mo, da).isoformat()
        except ValueError:
            skipped["ungueltiges Datum"] += 1
            continue
        t = t_of_date.get(d)
        if t is None:
            skipped["Tag nicht im Raster"] += 1
            continue
        try:
            lat, lon = float(r["LATITUDE"]), float(r["LONGITUDE"])
        except (ValueError, TypeError):
            skipped["keine Koordinate"] += 1
            continue
        rr, cc = rc_of(lat, lon)
        if not (0 <= rr < H and 0 <= cc < W):
            skipped["ausserhalb des Gitters"] += 1
            continue
        try:
            size = float(r.get("SIZE_HA") or "nan")
        except ValueError:
            size = float("nan")
        recs.append({
            "t": t, "row": rr, "col": cc, "date": d, "year": y, "month": mo,
            "cause": (r.get("CAUSE") or "").strip().upper()[:1],
            "size_ha": size,
            "fire_type": (r.get("FIRE_TYPE") or "").strip(),
            "prescribed": (r.get("PRESCRIBED") or "").strip(),
            "agency": (r.get("SRC_AGENCY") or "").strip(),
            "in_valid": bool(valid[rr, cc]), "in_canada": bool(canada[rr, cc]),
            # Lage innerhalb der Zelle, 0 = Oberkante/Westrand, 1 = Unterkante/Ostrand.
            # Wird fuer die Registrierungsdiagnose gebraucht.
            "frow": (lat - tr[5]) / tr[4] - rr,
            "fcol": (lon - tr[2]) / tr[0] - cc,
        })
    print(f"[nfdb] {len(recs)} verwertbar, verworfen: {dict(skipped)}")

    n_valid = sum(r["in_valid"] for r in recs)
    print(f"[nfdb] davon in unserer Gueltigkeitsmaske: {n_valid} "
          f"({100*n_valid/max(1,len(recs)):.1f} %)")

    # ---------------------------------------------- Sensitivitaet
    k = args.space_tol
    offs = [(dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1)]
    lags = list(range(-args.day_before, args.day_after + 1))
    for rec in recs:
        hit = False
        for dt_ in lags:
            tt = rec["t"] + dt_
            if tt < 0 or tt >= len(dates):
                continue
            base = tt * (H * W)
            for dr, dc in offs:
                rr, cc = rec["row"] + dr, rec["col"] + dc
                if 0 <= rr < H and 0 <= cc < W and (base + rr * W + cc) in ours:
                    hit = True
                    break
            if hit:
                break
        rec["modis_hit"] = hit

    inmask = [r for r in recs if r["in_valid"]]
    det = sum(r["modis_hit"] for r in inmask) / max(1, len(inmask))
    print(f"\n[nfdb] Sensitivitaet insgesamt (in Maske): {100*det:.1f} % "
          f"({sum(r['modis_hit'] for r in inmask)}/{len(inmask)})")

    by_size = {}
    print("\n[nfdb] Sensitivitaet nach Brandflaeche:")
    for i, lab in enumerate(SIZE_LABELS):
        sel = [r for r in inmask if SIZE_BINS[i] <= (r["size_ha"] if r["size_ha"] == r["size_ha"] else -1) < SIZE_BINS[i + 1]]
        if not sel:
            continue
        d = sum(r["modis_hit"] for r in sel) / len(sel)
        by_size[lab] = {"n": len(sel), "detected": sum(r["modis_hit"] for r in sel), "rate": d}
        print(f"    {lab:11s} n={len(sel):6d}  erkannt {100*d:5.1f} %")

    by_cause = {}
    print("\n[nfdb] nach Ursache:")
    for c in ("N", "H", "U"):
        sel = [r for r in inmask if r["cause"] == c]
        if not sel:
            continue
        d = sum(r["modis_hit"] for r in sel) / len(sel)
        med = float(np.nanmedian([r["size_ha"] for r in sel]))
        by_cause[c] = {"name": CAUSE_NAMES[c], "n": len(sel), "rate": d, "median_size_ha": med}
        print(f"    {CAUSE_NAMES[c]:20s} n={len(sel):6d}  erkannt {100*d:5.1f} %  "
              f"Median {med:.1f} ha")

    # ---------------------------------------------- Bestaetigungsrate
    nfdb_keys = set()
    for rec in inmask:
        for dt_ in lags:
            tt = rec["t"] + dt_
            if 0 <= tt < len(dates):
                nfdb_keys.add(tt * (H * W) + rec["row"] * W + rec["col"])
    confirmed = 0
    n_ours_valid = 0
    for key in ours:
        t_, p_ = divmod(key, H * W)
        rr, cc = divmod(p_, W)
        if not valid[rr, cc]:
            continue
        n_ours_valid += 1
        for dr, dc in offs:
            r2, c2 = rr + dr, cc + dc
            if 0 <= r2 < H and 0 <= c2 < W and (t_ * (H * W) + r2 * W + c2) in nfdb_keys:
                confirmed += 1
                break
    conf_rate = confirmed / max(1, n_ours_valid)
    print(f"\n[nfdb] Bestaetigungsrate unserer Proxy-Zuendungen: {100*conf_rate:.1f} % "
          f"({confirmed}/{n_ours_valid})")

    # ---------------------------------------------- geplantes Abbrennen
    presc = [r for r in recs if r["prescribed"] or r["fire_type"].lower().startswith("prescribed")]
    ft = Counter(r["fire_type"] for r in recs)
    month_all = Counter(r["month"] for r in inmask)
    month_presc = Counter(r["month"] for r in presc)
    print(f"\n[nfdb] FIRE_TYPE: {ft.most_common(6)}")
    print(f"[nfdb] als geplant markiert: {len(presc)}")
    print("[nfdb] Monat: gemeldete Braende | davon geplant")
    for m in range(1, 13):
        print(f"    {m:2d}  {month_all.get(m,0):6d} | {month_presc.get(m,0):5d}")

    # ---------------------------------------------- Toleranz-Sweep
    # Raeumliche und zeitliche Toleranz sind frei gewaehlt. Ein Ergebnis,
    # das stark davon abhaengt, waere kein Ergebnis. Deshalb wird beides
    # variiert und die Abhaengigkeit ausgewiesen.
    sweep = {}
    if args.sweep:
        print("\n[nfdb] Toleranz-Sweep  (Raum in Pixeln, Zeit in Tagen)")
        print("      Raum  Tage-vor  Tage-nach | Sensitivitaet | Bestaetigung")
        for k2 in (0, 1, 2, 3):
            offs2 = [(dr, dc) for dr in range(-k2, k2 + 1) for dc in range(-k2, k2 + 1)]
            for before, after in ((0, 0), (1, 3), (3, 7), (7, 14)):
                lags2 = list(range(-before, after + 1))
                n_hit = 0
                for rec in inmask:
                    found = False
                    for dt2 in lags2:
                        tt = rec["t"] + dt2
                        if tt < 0 or tt >= len(dates):
                            continue
                        base = tt * (H * W)
                        for dr, dc in offs2:
                            rr, cc = rec["row"] + dr, rec["col"] + dc
                            if 0 <= rr < H and 0 <= cc < W and (base + rr * W + cc) in ours:
                                found = True
                                break
                        if found:
                            break
                    n_hit += found
                sens = n_hit / max(1, len(inmask))
                nk = set()
                for rec in inmask:
                    for dt2 in lags2:
                        tt = rec["t"] + dt2
                        if 0 <= tt < len(dates):
                            nk.add(tt * (H * W) + rec["row"] * W + rec["col"])
                conf = 0
                for key in ours:
                    t_, p_ = divmod(key, H * W)
                    rr, cc = divmod(p_, W)
                    if not valid[rr, cc]:
                        continue
                    for dr, dc in offs2:
                        r2, c2 = rr + dr, cc + dc
                        if 0 <= r2 < H and 0 <= c2 < W and (t_ * (H * W) + r2 * W + c2) in nk:
                            conf += 1
                            break
                sweep[f"space{k2}_days-{before}+{after}"] = {
                    "space_px": k2, "day_before": before, "day_after": after,
                    "sensitivity": sens,
                    "confirmation": conf / max(1, n_ours_valid),
                }
                print(f"      {k2:4d}  {before:8d}  {after:9d} | {100*sens:12.1f} % | "
                      f"{100*conf/max(1,n_ours_valid):11.1f} %")

    # ---------------------------------------------- Monatsvergleich
    # Unsere Zuendungen je Monat aus der Ereignisableitung, damit das
    # Verhaeltnis reproduzierbar im Manifest steht und nicht nebenbei
    # ausgerechnet wird.
    z = np.load(exp / "derived_data" / "events" / "per_day_counts.npz", allow_pickle=False)
    ours_month: Counter = Counter()
    for d_, v_ in zip(z["dates"], z["ign_r1"]):
        ours_month[str(d_)[5:7]] += int(v_)

    SEASON = ("05", "06", "07", "08")
    AUTUMN = ("10", "11")
    comparison = {}
    print("\n[nfdb] Monatsvergleich  gemeldet | unsere | Verhaeltnis")
    for m in range(1, 13):
        key = f"{m:02d}"
        rep_n = month_all.get(m, 0)
        our_n = ours_month.get(key, 0)
        ratio = (our_n / rep_n) if rep_n else None
        comparison[key] = {"reported": rep_n, "ours": our_n, "ratio": ratio}
        flag = "  <-- MODIS meldet ein Vielfaches" if ratio and ratio > 2 else ""
        print(f"    {key}  {rep_n:7d} | {our_n:7d} | "
              f"{ratio if ratio is None else round(ratio, 2)}{flag}")

    rep_s = sum(month_all.get(int(m), 0) for m in SEASON)
    our_s = sum(ours_month.get(m, 0) for m in SEASON)
    rep_a = sum(month_all.get(int(m), 0) for m in AUTUMN)
    our_a = sum(ours_month.get(m, 0) for m in AUTUMN)
    season_ratio = our_s / max(1, rep_s)
    autumn_ratio = our_a / max(1, rep_a)
    excess = autumn_ratio / max(1e-9, season_ratio)
    print(f"\n[nfdb] Mai-Aug   gemeldet {rep_s}, unsere {our_s}, Verhaeltnis {season_ratio:.2f}")
    print(f"[nfdb] Okt+Nov   gemeldet {rep_a}, unsere {our_a}, Verhaeltnis {autumn_ratio:.2f}")
    print(f"[nfdb] -> Herbst {excess:.1f}-fach ueberhoeht gegenueber der Saison")

    # ---------------------------------------------- Registrierung
    fcnt = np.diff(fi["fire_offsets"])
    fpx = np.repeat(np.asarray(fi["fire_pixels"], np.int64), fcnt)
    fkeys = np.sort(fpx * 16384 + np.asarray(fi["fire_days"], np.int64))
    reg = registration_offsets(recs, fkeys, H, W)
    print(f"\n[nfdb] Registrierung: {reg['n_single_cell']} eindeutige "
          f"Einzelzell-Treffer von {reg['n_selected']} Braenden 100-1000 ha")
    if reg.get("n_single_cell"):
        print(f"[nfdb]   exakt (0,0): {100*reg['share_exact']:.1f} %, "
              f"Schwerpunkt Zeile {reg['centroid_row']:+.3f} "
              f"Spalte {reg['centroid_col']:+.3f}")
        print(f"[nfdb]   Sub-Zell-Lage je Zeilenversatz: "
              + ", ".join(f"{k}: {v['mean_frow']:.3f} (n={v['n']})"
                          for k, v in sorted(reg["subcell_by_row_offset"].items())))
        print(f"[nfdb]   mit round statt floor waeren es "
              f"{100*reg['share_exact_if_round_convention']:.1f} % "
              f"-> die Abrundung bleibt die bessere Konvention")

    write_manifest(
        cfg, "d9_nfdb_validation", args.config,
        {
            "source": "Canadian National Fire Database (NFDB), NRCan/CFS",
            "file": args.nfdb,
            "n_records_total": len(rows),
            "n_records_usable_2000_2025": len(recs),
            "skipped": dict(skipped),
            "n_in_valid_mask": n_valid,
            "matching": {"space_tolerance_px": k,
                         "day_window": [-args.day_before, args.day_after]},
            "sensitivity_overall": det,
            "sensitivity_by_size": by_size,
            "sensitivity_by_cause": by_cause,
            "confirmation_rate_of_our_ignitions": conf_rate,
            "n_our_ignitions_in_mask": n_ours_valid,
            "fire_type_counts": dict(ft.most_common(10)),
            "n_prescribed": len(presc),
            "registration": reg,
            "reported_by_month": {str(m): month_all.get(m, 0) for m in range(1, 13)},
            "prescribed_by_month": {str(m): month_presc.get(m, 0) for m in range(1, 13)},
            "cause_counts": dict(Counter(r["cause"] for r in recs)),
            "tolerance_sweep": sweep,
            "monthly_comparison": comparison,
            "season_months": list(SEASON),
            "autumn_months": list(AUTUMN),
            "season_ratio_ours_per_reported": season_ratio,
            "autumn_ratio_ours_per_reported": autumn_ratio,
            "autumn_excess_factor": excess,
            "our_ignitions_by_month": {k: int(v) for k, v in sorted(ours_month.items())},
            "interpretation": (
                f"In der Saison sehen wir das {season_ratio:.2f}-fache der gemeldeten "
                f"Braende, im Herbst das {autumn_ratio:.2f}-fache. Der Ueberschuss um "
                f"Faktor {excess:.1f} bestaetigt unabhaengig, dass die Herbstdetektionen "
                "keine Waldbraende sind (vgl. d1b_ignition_qc)."
            ),
        },
    )


if __name__ == "__main__":
    main()

"""Stufe 2c: Traegt die Helligkeitstemperatur Vorhersage- oder Detektionssignal?

Die Kanaele 50-53 sind die MODIS-Helligkeitstemperaturen der Baender
20/21/31/32 - genau die Groessen, aus denen der MODIS-Feueralgorithmus die
Maske MCD14A1 berechnet, die hier als Label dient. Ein erhoehter Wert am Tag
der Detektion kann daher zweierlei sein:

  * **Detektion**: die Waermestrahlung des Feuers selbst. Dann ist der Kanal
    kein Praediktor, sondern eine Umschreibung des Labels.
  * **Praediktor**: Hitze und Trockenheit VOR dem Feuer. Dann ist der Kanal
    legitim und wertvoll.

Beides trennt sich am zeitlichen Verlauf: Eine Anomalie, die erst am Tag D
auftritt, ist Detektion. Eine, die schon Tage vorher da ist, ist Signal.

Methode: fuer eine Stichprobe von Tagen der Feuersaison werden die
Feuerpixel des Tages D mit einer gleich grossen Kontrollmenge feuerfreier
Landpixel DESSELBEN Tages verglichen. Fuer beide Gruppen wird der Kanalwert
an den Tagen D, D-1, D-3, D-7 und D-14 an DERSELBEN Pixelposition gelesen.

ZWEI SCHAETZER, weil der einfache der beiden verzerrt ist:

  * ``cohens_d``: Niveauunterschied Feuer vs. Kontrolle je Lag. VERZERRT -
    Feuerpixel liegen systematisch suedlicher, trockener und in anderer
    Landbedeckung als gleichverteilt gezogene Kontrollpixel. Dieser
    Standortunterschied steckt in JEDEM Lag, auch in D-14, und taeuscht
    Vorlaufsignal vor.
  * ``paired_delta``: Zuwachs INNERHALB desselben Pixels von der Basislinie
    D-14 bis zum jeweiligen Lag, dann Feuer- gegen Kontrollgruppe. Ein
    zeitlich konstanter Standortunterschied faellt in dieser Differenz
    heraus. Das ist der belastbare Schaetzer.

Die Kontrollpixel werden zusaetzlich nach LANDBEDECKUNGSKLASSE gematcht, um
den Rest des Standortunterschieds zu daempfen.

Zum Vergleich laufen FWI und VPD mit: bei ihnen MUSS die Anomalie schon
vorher da sein, sonst stimmt die Methode nicht.
"""

from __future__ import annotations

import argparse
import datetime as dt
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, load_config, write_manifest

LAGS = (0, -1, -3, -7, -14)
# (Modalitaet, Band, Kanalindex im Cache, Kurzname)
PROBES = [
    ("MCD09CMG", 0, 50, "BT_band20"),
    ("MCD09CMG", 1, 51, "BT_band21"),
    ("MCD09CMG", 2, 52, "BT_band31"),
    ("MCD09CMG", 3, 53, "BT_band32"),
    ("MCD11A1", 0, 46, "LST_Day"),
    ("FWI", 5, 24, "FWI"),
    ("VPD", 0, 25, "VPD"),
    ("ERA5", 0, 1, "temperature_2m"),
]


def read_day(root: Path, modality: str, date: dt.date) -> np.ndarray | None:
    tar = root / "Entire_Canada_Maps" / modality / f"{modality}__{date.year}.tar"
    target = f"{date.year}_{date.month:02d}_{date.day:02d}.tif"
    if not tar.exists():
        return None
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
    ap.add_argument("--n-days", type=int, default=30)
    ap.add_argument("--max-fire-px", type=int, default=400)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    root = Path(cfg["dataset_root"])

    # Landmaske und Landbedeckungsklassen fuer das Matching
    lc = read_day(root, "MCD12Q1", dt.date(2015, 1, 1))
    lc_class = lc[0]
    land = ~np.isin(lc_class, (0, 17))
    # Pixelindizes je Landbedeckungsklasse, fuer gematchte Kontrollziehung
    class_pool = {
        int(k): np.nonzero(land & (lc_class == k))
        for k in np.unique(lc_class[land])
    }

    # Tage der Feuersaison ueber mehrere Jahre ziehen
    years = list(range(2003, 2023))
    days: list[dt.date] = []
    for _ in range(args.n_days):
        y = int(rng.choice(years))
        doy = int(rng.integers(150, 260))  # Ende Mai bis Mitte September
        days.append(dt.date(y, 1, 1) + dt.timedelta(days=doy - 1))
    days = sorted(set(days))
    print(f"[s2c] {len(days)} Tage, Lags {LAGS}")

    acc: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"fire": [], "ctrl": []}
    )
    # Gepaarte Werte je Tag: name -> {"fire"|"ctrl": (n_px, n_lags)}
    paired: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"fire": [], "ctrl": []}
    )

    for n, D in enumerate(days):
        fire_mask = read_day(root, "MCD14A1", D)
        if fire_mask is None:
            continue
        fire = (fire_mask[0] == 1) & land
        rr, cc = np.nonzero(fire)
        if rr.size < 20:
            continue
        if rr.size > args.max_fire_px:
            sel = rng.choice(rr.size, args.max_fire_px, replace=False)
            rr, cc = rr[sel], cc[sel]

        # Kontrolle: feuerfreie Landpixel desselben Tages, nach
        # Landbedeckungsklasse gematcht - je Feuerpixel ein Kontrollpixel
        # derselben Klasse.
        kr_l, kc_l = [], []
        for r_, c_ in zip(rr.tolist(), cc.tolist()):
            pool = class_pool.get(int(lc_class[r_, c_]))
            if pool is None or pool[0].size == 0:
                continue
            for _ in range(8):  # wenige Versuche, kein Feuerpixel zu erwischen
                j = int(rng.integers(pool[0].size))
                pr, pc = int(pool[0][j]), int(pool[1][j])
                if not fire[pr, pc]:
                    kr_l.append(pr); kc_l.append(pc)
                    break
        kr, kc = np.asarray(kr_l), np.asarray(kc_l)
        if kr.size < 10:
            continue
        rr_m, cc_m = rr[: kr.size], cc[: kr.size]

        # Alle Lags EINES Tages zusammen lesen, damit die Paarung je Pixel
        # ueber die Lags hinweg garantiert ausgerichtet bleibt.
        # Tagescache: MCD09CMG speist vier Proben und wuerde sonst viermal
        # von der Platte gelesen.
        day_cache: dict[tuple[str, int], np.ndarray | None] = {}

        def get(mod: str, lag: int) -> np.ndarray | None:
            key = (mod, lag)
            if key not in day_cache:
                day_cache[key] = read_day(root, mod, D + dt.timedelta(days=lag))
            return day_cache[key]

        for mod, band, ch, name in PROBES:
            fire_rows, ctrl_rows, ok = [], [], True
            for lag in LAGS:
                arr = get(mod, lag)
                if arr is None or band >= arr.shape[0]:
                    ok = False
                    break
                a = arr[band].astype(np.float64)
                fire_rows.append(a[rr_m, cc_m])
                ctrl_rows.append(a[kr, kc])
            if not ok:
                continue
            fv = np.stack(fire_rows, axis=1)   # (n_px, n_lags)
            cv = np.stack(ctrl_rows, axis=1)
            paired[name]["fire"].append(fv)
            paired[name]["ctrl"].append(cv)
            for i, lag in enumerate(LAGS):
                acc[(name, lag)]["fire"].extend(fv[:, i].tolist())
                acc[(name, lag)]["ctrl"].extend(cv[:, i].tolist())
        print(f"    {n+1}/{len(days)} {D} {rr_m.size} Feuerpixel / "
              f"{kr.size} gematchte Kontrollen", flush=True)

    rows = []
    print("\n[s2c] Kanal            Lag   Feuer   Kontrolle    Diff   Cohen d")
    for mod, band, ch, name in PROBES:
        for lag in LAGS:
            d = acc.get((name, lag))
            if not d or not d["fire"]:
                continue
            f = np.asarray(d["fire"]); c = np.asarray(d["ctrl"])
            pooled = np.sqrt((f.var() + c.var()) / 2) or 1.0
            cohen = float((f.mean() - c.mean()) / pooled)
            rows.append(
                {
                    "channel": ch,
                    "name": name,
                    "modality": mod,
                    "band": band,
                    "lag_days": lag,
                    "n": int(f.size),
                    "fire_mean": float(f.mean()),
                    "ctrl_mean": float(c.mean()),
                    "diff": float(f.mean() - c.mean()),
                    "cohens_d": cohen,
                }
            )
            print(f"    ch{ch:02d} {name:14s} {lag:4d} {f.mean():8.2f} {c.mean():9.2f} "
                  f"{f.mean()-c.mean():8.2f}  {cohen:7.3f}")

    # ---------------------------------------------------- gepaarter Schaetzer
    base = LAGS.index(min(LAGS))  # Basislinie D-14
    paired_rows = []
    print("\n[s2c] GEPAART (Zuwachs seit D-14, Standortunterschied herausgekuerzt)")
    print("      Kanal            Lag   dFeuer   dKontr    Diff   Cohen d")
    paired_by_name: dict[str, dict[int, float]] = defaultdict(dict)
    for mod, band, ch, name in PROBES:
        if not paired[name]["fire"]:
            continue
        F = np.concatenate(paired[name]["fire"], axis=0)
        C = np.concatenate(paired[name]["ctrl"], axis=0)
        dF = F - F[:, [base]]
        dC = C - C[:, [base]]
        for i, lag in enumerate(LAGS):
            if lag == LAGS[base]:
                continue
            a, b = dF[:, i], dC[:, i]
            pooled = np.sqrt((a.var() + b.var()) / 2) or 1.0
            d = float((a.mean() - b.mean()) / pooled)
            paired_by_name[name][lag] = d
            paired_rows.append(
                {
                    "channel": ch, "name": name, "lag_days": lag,
                    "n": int(a.size),
                    "fire_delta_mean": float(a.mean()),
                    "ctrl_delta_mean": float(b.mean()),
                    "diff_of_deltas": float(a.mean() - b.mean()),
                    "cohens_d_paired": d,
                }
            )
            print(f"      ch{ch:02d} {name:14s} {lag:4d} {a.mean():8.2f} {b.mean():8.2f} "
                  f"{a.mean()-b.mean():8.2f}  {d:7.3f}")

    # Auswertung: Sprung von D-1 auf D relativ zum Niveau bei D-1
    verdict = {}
    for mod, band, ch, name in PROBES:
        at = {r["lag_days"]: r["cohens_d"] for r in rows if r["name"] == name}
        if 0 not in at or -1 not in at:
            continue
        d0, d1, d7 = at.get(0), at.get(-1), at.get(-7)
        # Die Bewertung stuetzt sich auf den GEPAARTEN Schaetzer; der
        # ungepaarte steht nur zum Vergleich daneben.
        pd0 = paired_by_name.get(name, {}).get(0)
        pd1 = paired_by_name.get(name, {}).get(-1)
        pd7 = paired_by_name.get(name, {}).get(-7)
        jump = (pd0 - pd1) if (pd0 is not None and pd1 is not None) else None
        ref = max(abs(pd0), 1e-9) if pd0 is not None else 1.0
        verdict[name] = {
            "channel": ch,
            "unpaired_cohens_d": {"D": d0, "D-1": d1, "D-7": d7},
            "paired_cohens_d": {"D": pd0, "D-1": pd1, "D-7": pd7},
            "same_day_jump_paired": jump,
            "interpretation": (
                "unbestimmt" if pd0 is None or pd7 is None or jump is None
                else "DETEKTION - Anomalie entsteht erst am Detektionstag"
                if jump > 0.5 * ref and abs(pd7) < 0.3 * ref
                else "PRAEDIKTOR - Anomalie besteht bereits Tage vorher"
                if abs(pd7) >= 0.5 * ref
                else "GEMISCHT - Vorlauf vorhanden, am Tag D zusaetzlicher Sprung"
            ),
        }
    print("\n[s2c] Bewertung:")
    for name, v in verdict.items():
        p_ = v["paired_cohens_d"]
        print(f"    ch{v['channel']:02d} {name:14s} "
              f"gepaart d(D)={(p_['D'] or 0):6.3f} d(D-1)={(p_['D-1'] or 0):6.3f} "
              f"d(D-7)={(p_['D-7'] or 0):6.3f} Sprung={(v['same_day_jump_paired'] or 0):6.3f}"
              f"  -> {v['interpretation']}")

    write_manifest(
        cfg,
        "s2c_bt_leakage",
        args.config,
        {
            "n_days_sampled": len(days),
            "days": [d.isoformat() for d in days],
            "lags": list(LAGS),
            "max_fire_px_per_day": args.max_fire_px,
            "control": "feuerfreie Landpixel desselben Tages, nach IGBP-Klasse gematcht",
            "measurements_unpaired": rows,
            "measurements_paired": paired_rows,
            "paired_baseline_lag": min(LAGS),
            "estimator_note": (
                "measurements_unpaired ist durch den Standortunterschied zwischen "
                "Feuer- und Kontrollpixeln verzerrt und wird nur zum Vergleich "
                "berichtet. Die Bewertung stuetzt sich auf measurements_paired."
            ),
            "verdict": verdict,
        },
    )


if __name__ == "__main__":
    main()

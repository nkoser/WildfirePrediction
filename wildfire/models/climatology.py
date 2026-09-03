"""B0: Klimatologie-Baselines.

Der Arbeitsplan verlangt als erste Baseline die historische Feuerhaeufigkeit
nach Region und Jahreszeit. Sie braucht kein Training, keinen Wuerfel und
keine GPU - nur die Zuendungen der Trainingsjahre. Ihr Zweck ist nicht,
gut zu sein, sondern die Messlatte zu setzen: Jedes gelernte Modell muss
zeigen, dass es mehr kann als das saisonale Mittel seiner Umgebung.

Drei Varianten, absichtlich von grob nach fein:

  global    ein einziger Hazard fuer alles. Der Nullpunkt.
  region    je Provinz und Zweiwochenfenster. Faengt, dass Alberta im Juli
            anders brennt als Nunavut im Mai.
  pixel     je Pixel und Zweiwochenfenster, raeumlich geglaettet. Ein
            einzelnes Pixel hat ueber 18 Trainingsjahre oft null oder eine
            Zuendung - ohne Glaettung waere die Schaetzung nur Rauschen.

Alle drei liefern einen KONSTANTEN taeglichen Hazard ueber den Horizont.
Das ist der Punkt: Klimatologie kennt keine Dynamik. Was ein Modell darueber
hinaus leistet, ist genau der Beitrag der Kovariaten.

Geschaetzt wird ausschliesslich auf Trainingsjahren. Die Rate ist
Zuendungen geteilt durch Pixeltage unter Risiko.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from wildfire.audit.common import (assert_readonly, landmark_horizon,
                                   load_config, write_manifest)
from wildfire.data.splits import SPLIT_DEFS

BIN_DAYS = 14
SMOOTH_SIGMA = 3.0
# Staerke der Schrumpfung zur Regionalrate, in Pseudo-Ereignissen. Ein
# Pixel ohne beobachtete Zuendung darf NICHT gegen null geschaetzt werden -
# es hat in 18 Jahren schlicht zu wenig Exposition, um "kein Risiko" zu
# belegen. Die Gamma-Poisson-Posteriorerwartung zieht es stattdessen zur
# Rate seiner Region.
PRIOR_EVENTS = 1.0
LAPLACE = 0.5          # nur noch fuer die Regionalraten


def doy_bin(d: dt.date, season_start: str) -> int:
    a = dt.date(d.year, int(season_start[:2]), int(season_start[3:]))
    return max(0, (d - a).days // BIN_DAYS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="temporal")
    ap.add_argument("--horizon", type=int, default=None,
                    help="Standard: aus dem Landmark-Manifest")
    ap.add_argument("--season-start", default="05-01")
    ap.add_argument("--season-end", default="08-31")
    ap.add_argument("--label-source", default="nfdb", choices=("nfdb", "modis"))
    ap.add_argument("--mask", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    if args.horizon is None:
        args.horizon = landmark_horizon(cfg)
        print(f"[cfg] Horizont aus dem Landmark-Manifest: {args.horizon}")

    exp = Path(cfg["experiment_root"])
    H = args.horizon
    mask_name = args.mask or ("observed_nfdb.npy" if args.label_source == "nfdb"
                              else "valid.npy")
    valid = np.load(exp / "derived_data" / "masks" / mask_name)
    region = np.load(exp / "derived_data" / "masks" / "region.npy")
    fi = np.load(exp / "derived_data" / "events" / "fire_index.npz", allow_pickle=False)
    dates = [dt.date.fromisoformat(str(d)) for d in fi["dates"]]
    year_of = np.array([d.year for d in dates])
    in_season = np.array(
        [args.season_start <= d.isoformat()[5:] <= args.season_end for d in dates])
    Hh, W = valid.shape

    if args.label_source == "nfdb":
        DAY_BITS = 16384
        z = np.load(exp / "derived_data" / "events" / "nfdb_index.npz", allow_pickle=True)
        keys = np.asarray(z["keys"], np.int64)
        ign_px, ign_t = keys // DAY_BITS, keys % DAY_BITS
    else:
        counts = np.diff(fi["ign_offsets"])
        ign_px = np.repeat(np.asarray(fi["ign_pixels"], np.int64), counts)
        ign_t = np.asarray(fi["ign_days"], np.int64)
    print(f"[B0] Labelquelle {args.label_source}, Maske {mask_name} "
          f"({int(valid.sum())} Pixel), {ign_px.size} Zuendungen")

    train_years = set(SPLIT_DEFS[args.split]["train"]["years"])
    n_bins = (dt.date(2001, int(args.season_end[:2]), int(args.season_end[3:]))
              - dt.date(2001, int(args.season_start[:2]), int(args.season_start[3:]))).days // BIN_DAYS + 1
    print(f"[B0] Trainingsjahre {min(train_years)}-{max(train_years)}, "
          f"{n_bins} Zeitfenster à {BIN_DAYS} Tage")

    # ------------------------------------------------ Zaehlen auf Trainingsjahren
    sel = np.array([in_season[t] and year_of[t] in train_years for t in ign_t])
    tr_px, tr_t = ign_px[sel], ign_t[sel]
    bins = np.array([doy_bin(dates[t], args.season_start) for t in tr_t])
    bins = np.clip(bins, 0, n_bins - 1)
    n_train_days = np.array([
        sum(1 for t in range(len(dates))
            if in_season[t] and year_of[t] in train_years
            and doy_bin(dates[t], args.season_start) == b)
        for b in range(n_bins)])
    print(f"[B0] {tr_px.size} Trainingszuendungen, Expositionstage je Fenster: "
          f"{n_train_days.tolist()}")

    cnt = np.zeros((n_bins, Hh, W), np.float32)
    np.add.at(cnt, (bins, tr_px // W, tr_px % W), 1.0)

    n_valid = int(valid.sum())
    models = {}

    # ---- global
    total_exposure = float(n_valid * n_train_days.sum())
    models["global"] = {"rate": float(tr_px.size / total_exposure)}
    print(f"[B0] global: {models['global']['rate']:.3e} Zuendungen je Pixeltag")

    # ---- Region x Zeitfenster
    reg_codes = json.loads((exp / "data_manifests" / "d2_regions.json").read_text())["code_of"]
    reg_rate = np.zeros((n_bins, max(reg_codes.values()) + 1), np.float64)
    for code in reg_codes.values():
        m = valid & (region == code)
        npix = int(m.sum())
        if npix == 0:
            continue
        c = cnt[:, m].sum(axis=1)
        reg_rate[:, code] = (c + LAPLACE) / (npix * n_train_days + LAPLACE / 1e-6)
    models["region_bin"] = {"shape": list(reg_rate.shape)}

    # ---- Pixel x Zeitfenster, raeumlich geglaettet
    # Zaehler UND Exposition glaetten, damit die Rate an Maskenraendern nicht
    # nach oben verzerrt: ein Pixel am Rand bekaeme sonst geglaettete Treffer
    # aus der Nachbarschaft, aber nur seine eigene Exposition.
    num = np.stack([ndimage.gaussian_filter(cnt[b], SMOOTH_SIGMA, mode="constant")
                    for b in range(n_bins)])
    den_mask = ndimage.gaussian_filter(valid.astype(np.float32), SMOOTH_SIGMA,
                                       mode="constant")
    # Gamma-Poisson-Posterior mit Erwartungswert der Regionalrate als Prior:
    #     rate = (beobachtet + a) / (Exposition + a / prior)
    # Ohne Beobachtung faellt das auf den Prior zurueck, mit vielen
    # Beobachtungen auf die empirische Rate.
    prior_map = np.zeros_like(num)
    for code in reg_codes.values():
        m = region == code
        for b in range(n_bins):
            prior_map[b][m] = reg_rate[b, code]
    prior_map = np.maximum(prior_map, models["global"]["rate"] * 1e-3)

    pix_rate = np.zeros_like(num)
    for b in range(n_bins):
        expo = np.maximum(den_mask * n_train_days[b], 1e-9)
        pix_rate[b] = (num[b] + PRIOR_EVENTS) / (expo + PRIOR_EVENTS / prior_map[b])
    pix_rate = np.clip(pix_rate, 1e-9, 0.5)
    v = pix_rate[:, valid]
    print(f"[B0] Pixelrate: Median {np.median(v):.3e}, p99 {np.percentile(v, 99):.3e}, "
          f"max {v.max():.3e}  (global {models['global']['rate']:.3e})")
    shrunk = float((np.abs(v - prior_map[:, valid]) / prior_map[:, valid] < 0.1).mean())
    print(f"[B0] {100*shrunk:.1f} % der Pixelraten liegen innerhalb 10 % ihres Priors "
          f"(dort war zu wenig Evidenz fuer etwas anderes)")

    out_dir = exp / "derived_data" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "b0_climatology.npz",
                        pixel_rate=pix_rate.astype(np.float32),
                        region_rate=reg_rate.astype(np.float32),
                        global_rate=np.array([models["global"]["rate"]]),
                        n_train_days=n_train_days,
                        bin_days=np.array([BIN_DAYS]))

    # ------------------------------------------------ Vorhersagen erzeugen
    from wildfire.eval.metrics import evaluate

    report = {}
    for part in ("val", "test"):
        f = exp / "derived_data" / "landmarks" / f"{args.split}_{part}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        t, r, c = z["t"].astype(np.int64), z["row"].astype(np.int64), z["col"].astype(np.int64)
        b = np.clip([doy_bin(dates[tt], args.season_start) for tt in t], 0, n_bins - 1)
        ev, tev, w = z["event"], z["t_event"], z["weight"]

        preds = {
            "global": np.full(t.size, models["global"]["rate"]),
            "region_bin": reg_rate[b, region[r, c]],
            "pixel_bin": pix_rate[b, r, c],
        }
        report[part] = {}
        for name, rate in preds.items():
            # float32 und sofort wieder freigeben: bei 2,5 Mio. Zeilen und
            # H = 100 sind das 1 GB je Variante.
            h = np.repeat(np.clip(rate, 1e-9, 0.5).astype(np.float32)[:, None],
                          H, axis=1)
            m = evaluate(h, ev, tev, w)
            del h
            report[part][name] = m
            hz = m["horizons"]
            print(f"[B0] {part:4s} {name:11s} NLL={m['nll']:.5f} "
                  f"IBS={m['integrated_brier']:.3e} "
                  f"AUC@30={hz['30']['roc_auc']:.4f} "
                  f"PR@30={hz['30']['pr_auc']:.4f} "
                  f"ECE@30={hz['30']['ece']:.3e}")

    from wildfire.experiment import register
    register(exp, f"b0_climatology_{args.split}", "baseline", args.config,
             split=args.split, seed=int(cfg["seed"]),
             metrics_path=str(exp / "data_manifests" / "m0_climatology.json"),
             predictions_path=str(out_dir / "b0_climatology.npz"),
             notes=(f"{len(models)} Varianten: global, region_bin, pixel_bin; "
                    f"Bins à {BIN_DAYS} d, sigma {SMOOTH_SIGMA}, "
                    f"Prior {PRIOR_EVENTS} Pseudo-Ereignisse"),
             extra={"test_auc_at_30": {k: v["horizons"]["30"]["roc_auc"]
                                       for k, v in report.get("test", {}).items()}})

    write_manifest(
        cfg, "m0_climatology", args.config,
        {
            "split": args.split, "horizon": H, "bin_days": BIN_DAYS,
            "smooth_sigma": SMOOTH_SIGMA, "laplace": LAPLACE,
            "train_years": sorted(train_years),
            "n_train_ignitions": int(tr_px.size),
            "global_rate": models["global"]["rate"],
            "exposure_days_per_bin": n_train_days.tolist(),
            "model_file": str(out_dir / "b0_climatology.npz"),
            "results": report,
        },
    )


if __name__ == "__main__":
    main()

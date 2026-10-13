"""B1: tabellarische Baselines auf Pixelmerkmalen.

Je Horizont tau aus {1, 7, 14, 30, 60, 100} wird ein binaerer Klassifikator auf
"Ereignis bis tau" trainiert - logistische Regression und Gradient
Boosting mit identischen Eingaben, wie der Arbeitsplan es verlangt.

**Woher der Prior kommt.** Die Korrektur braucht die echte Praevalenz der
Zielpopulation. Sie vom Testsplit zu nehmen waere ein Leak: In einer realen
Anwendung kennt man die Feuerrate des kommenden Jahres nicht. Standard ist
deshalb ``--prior-source val`` - die Validierungsjahre stehen zum
Vorhersagezeitpunkt zur Verfuegung. Zum Vergleich wird ``own`` mitberichtet,
also die (unzulaessige) Verwendung der Testpraevalenz; die Differenz zeigt,
wie viel diese Annahme wert waere.

**Geometriemerkmale als Ablation.** Breite, Zellflaeche und Region lassen
das Modell Geographie statt Physik lernen - bei der ersten Fassung stand
``cell_km2`` auf Rang 6 der Wichtigkeit. Das ist auf einem zeitlichen Split
zulaessig, waere auf einem raeumlichen OOD-Split aber genau die falsche
Abkuerzung. Deshalb laufen beide Varianten.

**Kein Merkmal ``year``.** Bei einem zeitlichen Split sind die Testjahre
nie im Training gesehen worden. Ein Baum kann darauf nur extrapolieren,
indem er den letzten Trainingswert fortschreibt - das ist keine
Information, sondern eine Falle. Das Merkmal wird immer entfernt.

**Prior-Korrektur.** Die Trainingsdatei ist angereichert: alle Positiven,
Negative nur als Stichprobe. Trainiert wird UNGEWICHTET, weil die
Anreicherung sonst verschenkt waere - Gewichte von 178 gegen 1 wuerden die
effektive Stichprobengroesse zusammenbrechen lassen. Stattdessen wird die
Ausgabe analytisch auf den echten Prior zurueckgerechnet:

    logit_korrigiert = logit_roh + log(pi_wahr / (1 - pi_wahr))
                                 - log(pi_train / (1 - pi_train))

Das ist die uebliche Korrektur bei Label Shift: Die klassenbedingten
Verteilungen aendern sich durch die Ziehung nicht, nur der Prior.
Zusaetzlich wird auf dem Validierungssplit isotonisch nachkalibriert -
gewichtet, also gegen die echte Praevalenz.

**Von Horizonten zu Hazards.** Die Metriken erwarten diskrete Hazards ueber
alle 30 Tage. Aus den vier Ankerpunkten F(tau) wird eine vollstaendige
Kurve gebildet: monoton erzwungen, dazwischen konstanter Hazard. Das ist
keine Verlegenheitsloesung, sondern die sparsamste Annahme, die mit den
vier Messpunkten vertraeglich ist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wildfire.audit.common import (assert_readonly, landmark_horizon,
                                   load_config, write_manifest)
from wildfire.eval.metrics import evaluate, negative_log_likelihood

# Ankerhorizonte. HORIZONS wird in main() auf den tatsaechlichen Horizont
# des Landmark-Manifests beschnitten - eine Quelle statt mehrerer Vorgaben.
HORIZONS_ALL = (1, 7, 14, 30, 60, 100)
HORIZONS: tuple = HORIZONS_ALL
EPS = 1e-7


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prior_shift(p_raw, pi_train, pi_true):
    """Von der angereicherten Trainingsverteilung auf den echten Prior."""
    return sigmoid(logit(p_raw) + logit(pi_true) - logit(pi_train))


def hazards_from_anchors(F: np.ndarray, taus, H: int) -> np.ndarray:
    """(N, n_tau) kumulative Risiken -> (N, H) diskrete Hazards.

    Monotonie wird ueber ein laufendes Maximum erzwungen; zwischen zwei
    Ankern wird konstanter Hazard angenommen.
    """
    F = np.maximum.accumulate(np.clip(F, EPS, 1 - EPS), axis=1)
    S = 1.0 - F
    h = np.zeros((F.shape[0], H), np.float64)
    prev_tau, prev_S = 0, np.ones(F.shape[0])
    for j, tau in enumerate(taus):
        span = tau - prev_tau
        ratio = np.clip(S[:, j] / np.maximum(prev_S, EPS), EPS, 1.0)
        hz = 1.0 - ratio ** (1.0 / span)
        h[:, prev_tau:tau] = hz[:, None]
        prev_tau, prev_S = tau, S[:, j]
    if prev_tau < H:
        h[:, prev_tau:] = h[:, prev_tau - 1][:, None]
    # nur nach oben klemmen: ein Nullhazard ist eine gueltige Aussage
    # ("hier passiert nichts") und darf das Survival unveraendert lassen.
    return np.clip(h, 0.0, 1 - EPS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="temporal")
    ap.add_argument("--horizon", type=int, default=None,
                    help="Standard: aus dem Landmark-Manifest")
    ap.add_argument("--max-iter-lr", type=int, default=300)
    ap.add_argument("--prior-source", default="val", choices=("val", "own"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    if args.horizon is None:
        args.horizon = landmark_horizon(cfg)
        print(f"[cfg] Horizont aus dem Landmark-Manifest: {args.horizon}")
    global HORIZONS
    HORIZONS = tuple(t for t in HORIZONS_ALL if t <= args.horizon)

    exp = Path(cfg["experiment_root"])
    fdir = exp / "derived_data" / "features"
    data = {}
    for part in ("train", "val", "test"):
        z = np.load(fdir / f"{args.split}_{part}.npz", allow_pickle=False)
        data[part] = {k: z[k] for k in z.files}
    names = [str(x) for x in data["train"]["names"]]
    print(f"[B1] {len(names)} Merkmale, "
          f"train={data['train']['X'].shape[0]} val={data['val']['X'].shape[0]} "
          f"test={data['test']['X'].shape[0]}")

    # echte Praevalenzen aus den Landmark-Manifesten
    lm = json.loads((exp / "data_manifests" / "d7_landmarks.json").read_text())
    pi_true = {p: lm["parts"][p]["prevalence_weighted"] for p in ("val", "test")}

    # ``year`` immer entfernen, Geometrie je nach Variante
    drop_always = {"year"}
    geo = {"lat", "cell_km2", "region"}
    variants = {
        "full": [i for i, n in enumerate(names) if n not in drop_always],
        "no_geo": [i for i, n in enumerate(names) if n not in (drop_always | geo)],
    }
    print(f"[B1] Varianten: full={len(variants['full'])} Merkmale, "
          f"no_geo={len(variants['no_geo'])} (ohne {sorted(geo)})")

    # Prior der Zielpopulation: standardmaessig aus Validierung, nicht aus Test
    pi_for = {p: (pi_true[p] if args.prior_source == "own" else pi_true["val"])
              for p in ("val", "test")}
    print(f"[B1] Prior-Quelle: {args.prior_source} -> "
          f"val {pi_for['val']:.3e}, test {pi_for['test']:.3e} "
          f"(echte Testpraevalenz {pi_true['test']:.3e})")

    results = {f"{m}_{v}": {} for m in ("logreg", "hgb") for v in variants}
    preds = {k: {p: np.zeros((data[p]["X"].shape[0], len(HORIZONS)))
                 for p in ("val", "test")} for k in results}

    for vname, cols in variants.items():
        Xtr = np.nan_to_num(data["train"]["X"][:, cols])
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        for j, tau in enumerate(HORIZONS):
            ytr = (data["train"]["event"].astype(bool)
                   & (data["train"]["t_event"] <= tau)).astype(int)
            pi_train = float(ytr.mean())
            print(f"\n[B1] {vname} Horizont {tau} d: {ytr.sum()} Positive "
                  f"(Anteil {pi_train:.4f})")
            models = {
                "logreg": LogisticRegression(max_iter=args.max_iter_lr,
                                             C=1.0).fit(Xtr_s, ytr),
                "hgb": HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
                    min_samples_leaf=50, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.15,
                    random_state=int(cfg["seed"])).fit(Xtr, ytr),
            }
            for mname, mdl in models.items():
                for part in ("val", "test"):
                    Xp = np.nan_to_num(data[part]["X"][:, cols])
                    raw = mdl.predict_proba(
                        scaler.transform(Xp) if mname == "logreg" else Xp)[:, 1]
                    preds[f"{mname}_{vname}"][part][:, j] = prior_shift(
                        raw, pi_train, pi_for[part])
                print(f"    {mname:7s} trainiert")

    # ---------------------------------------------- isotonische Nachkalibrierung
    # Nur auf VAL gefittet, gewichtet - der Testsplit bleibt unberuehrt.
    calibrators = {}
    for mname in results:
        calibrators[mname] = []
        for j, tau in enumerate(HORIZONS):
            yv = (data["val"]["event"].astype(bool)
                  & (data["val"]["t_event"] <= tau)).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(preds[mname]["val"][:, j], yv, sample_weight=data["val"]["weight"])
            calibrators[mname].append(iso)

    # ---------------------------------------------- Auswertung
    # Untergrenze fuer den Hazard. Zwischen zwei Ankern mit gleichem F ist der
    # Hazard exakt 0 - eine Zuendung dort hat Wahrscheinlichkeit 0, und die NLL
    # haengt dann allein an der Klemme der Metrik. Stattdessen bekommt B1 eine
    # eigene Untergrenze, gewaehlt auf VAL (gewichtete NLL), unveraendert auf
    # TEST angewandt. 0 steht im Gitter: ohne Nutzen wird nicht geklemmt.
    floor_grid = (0.0, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
    floors = {m: {} for m in results}
    for mname in results:
        F_by = {}
        for part in ("val", "test"):
            F_raw = preds[mname][part]
            F_cal = np.column_stack([calibrators[mname][j].predict(F_raw[:, j])
                                     for j in range(len(HORIZONS))])
            F_by[part] = {"prior_shift": F_raw, "isotonic": F_cal}
        for tag in ("prior_shift", "isotonic"):
            h_val = hazards_from_anchors(F_by["val"][tag], HORIZONS, args.horizon)
            nll_by = {f: negative_log_likelihood(np.maximum(h_val, f), data["val"]["event"],
                                                 data["val"]["t_event"], data["val"]["weight"])
                      for f in floor_grid}
            floor = min(nll_by, key=nll_by.get)
            floors[mname][tag] = {"chosen": floor,
                                  "val_nll_by_floor": {f"{f:g}": v for f, v in nll_by.items()}}
            print(f"[B1] {mname:11s} {tag:11s} Hazard-Untergrenze {floor:g} "
                  f"(val NLL {nll_by[0.0]:.5f} -> {nll_by[floor]:.5f})")
            for part in ("val", "test"):
                h = np.maximum(hazards_from_anchors(F_by[part][tag], HORIZONS, args.horizon),
                               floor)
                m = evaluate(h, data[part]["event"], data[part]["t_event"],
                             data[part]["weight"])
                m["hazard_floor"] = floor
                results[mname].setdefault(part, {})[tag] = m
                hz = m["horizons"]
                print(f"[B1] {part:4s} {mname:7s} {tag:11s} NLL={m['nll']:.5f} "
                      f"IBS={m['integrated_brier']:.3e} "
                      f"AUC@30={hz['30']['roc_auc']:.4f} "
                      f"PR@30={hz['30']['pr_auc']:.4f} "
                      f"ECE@30={hz['30']['ece']:.3e}")

    # Merkmalsbedeutung des Gradient Boosting fuer den 30-Tage-Horizont
    from sklearn.inspection import permutation_importance
    cols = variants["full"]
    fnames = [names[i] for i in cols]
    sub = np.random.default_rng(0).choice(
        data["val"]["X"].shape[0], min(20000, data["val"]["X"].shape[0]), replace=False)
    y30 = (data["val"]["event"].astype(bool) & (data["val"]["t_event"] <= 30)).astype(int)
    mdl30 = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=50,
        random_state=0).fit(np.nan_to_num(data["train"]["X"][:, cols]),
                            (data["train"]["event"].astype(bool)
                             & (data["train"]["t_event"] <= 30)).astype(int))
    pi = permutation_importance(mdl30, np.nan_to_num(data["val"]["X"][sub][:, cols]),
                                y30[sub], n_repeats=3, random_state=0,
                                scoring="roc_auc", n_jobs=-1)
    order = np.argsort(-pi.importances_mean)[:20]
    top = [{"feature": fnames[i], "importance": float(pi.importances_mean[i])}
           for i in order]
    print("\n[B1] wichtigste Merkmale (Permutation, AUC@30 auf val):")
    for t_ in top[:12]:
        print(f"    {t_['feature']:16s} {t_['importance']:+.4f}")

    out = exp / "derived_data" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "b1_predictions.npz",
                        **{f"{m}_{p}": preds[m][p] for m in preds for p in preds[m]})

    from wildfire.experiment import register
    register(exp, f"b1_tabular_{args.split}", "baseline", args.config,
             split=args.split, seed=int(cfg["seed"]),
             metrics_path=str(exp / "data_manifests" / "m1_tabular.json"),
             predictions_path=str(out / "b1_predictions.npz"),
             notes=(f"Varianten {list(variants)}, Horizonte {HORIZONS}, "
                    f"Prior aus '{args.prior_source}'"),
             extra={f"test_auc_at_{HORIZONS[-1]}": {m: results[m]["test"]["isotonic"]["horizons"][str(HORIZONS[-1])]["roc_auc"]
                                       for m in results}})

    write_manifest(cfg, "m1_tabular", args.config,
                   {"split": args.split, "horizons": list(HORIZONS),
                    "n_features": {k: len(v) for k, v in variants.items()},
                    "feature_names": names,
                    "dropped_always": sorted(drop_always),
                    "geometry_features": sorted(geo),
                    "prior_source": args.prior_source,
                    "prior_used": pi_for,
                    "prevalence_true": pi_true,
                    "calibration": "Prior-Korrektur, dann isotonisch auf val (gewichtet)",
                    "hazard_floor": floors,
                    "top_features_auc30": top,
                    "results": results})


if __name__ == "__main__":
    main()

"""Paper-Protokoll, Schritt 5: unser Survival-Modell auf den Stichproben des Papers.

Unser B4 (diskreter Kopf) ist auf NFDB-Neuzuendungen bei echter Rate trainiert.
Hier wird es ohne Nachtraining auf die Test-Stichproben des Papers angewandt,
mit dem Verschiebe-Protokoll des Papers:

  Zieltag t = Sample-Datum, Label = Cache-Label (Feuer im 13x13-Ausschnitt an t)
  Eingabe   = 30 Tage, die bei t - dt - 1 enden  ->  Landmark L = t - dt - 1
  Vorlauf   = k = dt + 1 Tage von L bis t

Daraus vier Scores je Stichprobe:

  centre_h  Hazard h_k der Mittelzelle: neue Zuendung genau am Tag t
  centre_F  Risiko F(k) der Mittelzelle: Zuendung bis zum Tag t
  patch_h   1 - prod (1 - h_k) ueber die Zellen des Ausschnitts in observed_nfdb
  patch_F   1 - prod (1 - F(k)) ebenso

Unsere Wahrscheinlichkeiten gelten fuer die echte Rate (~1e-5 je Tag), die
Paper-Labels fuer 1 : 2. Fuer Precision/Recall/F1/ECE/MCE/Brier wird der Score
deshalb per Platt-Skalierung auf der VAL-Menge des Caches umgerechnet; PR-AUC
und ROC-AUC haengen davon nicht ab.

Fairer Vergleich: nur Stichproben, fuer die der Landmark bei ALLEN dt im
Wuerfel liegt (1.4.-30.9., 30 Tage Rueckblick im selben Jahr). Das 3D-U-Net
wird aus seinen gespeicherten Vorhersagen (42_paper_reconstruct, _paperdt) auf
genau dieser Teilmenge neu ausgewertet.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import LOCKED_CHANNELS, CubeReader
from wildfire.data.features import COOLDOWN, LOOKBACK, SHORT, feature_names, row_features
from wildfire.models.mlp_survival import MLPSurvival
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics

DTS = (1, 7, 15, 30)
HALF = 6
SCORES = ("centre_h", "centre_F", "patch_h", "patch_F")


def patch_features(cube, tt, r, c, lat, area, region, locked):
    """Merkmale aller Zellen des 13x13-Ausschnitts um (r, c) am Tag tt.

    Gleiche Rechnung wie features.row_features, nur ueber einen Block.
    Rueckgabe: X (n, 216) roh, rr, cc (n,) Zellkoordinaten im Gitter.
    """
    year, li = cube._pos[tt]
    dyn, stat = cube._arrays(year)
    r0, r1 = max(0, r - HALF), min(cube.H, r + HALF + 1)
    c0, c1 = max(0, c - HALF), min(cube.W, c + HALF + 1)
    s = dyn[li - LOOKBACK + 1:li + 1, r0:r1, c0:c1, :].astype(np.float32)   # (30, h, w, C)
    s[LOOKBACK - COOLDOWN:, :, :, locked] = np.nan
    with np.errstate(invalid="ignore"):
        last = s[-1]
        mean_s = np.nanmean(s[-SHORT:], axis=0)
        mean_l = np.nanmean(s, axis=0)
        mn = np.nanmin(s, axis=0)
        mx = np.nanmax(s, axis=0)
    h, w = last.shape[:2]
    st = stat[r0:r1, c0:c1, :].astype(np.float32)
    d0 = dt.date.fromisoformat(cube.dates[tt])
    doy = d0.timetuple().tm_yday
    extra = np.stack([
        lat[r0:r1, c0:c1], area[r0:r1, c0:c1] / 1e6, region[r0:r1, c0:c1],
        np.full((h, w), np.sin(2 * np.pi * doy / 365.25)),
        np.full((h, w), np.cos(2 * np.pi * doy / 365.25)),
        np.full((h, w), d0.year)], axis=-1).astype(np.float32)
    X = np.concatenate([last, mean_s, mean_l, mn, mx, st, extra], axis=-1).reshape(h * w, -1)
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    return X, rr.ravel(), cc.ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--head", default="discrete")
    ap.add_argument("--unet-run", default="_paperdt")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------- unser Modell
    ck = torch.load(exp / "checkpoints" / f"b4_{args.head}_best.pt", weights_only=False)
    train_names = [str(n) for n in np.load(exp / "derived_data" / "features" / "temporal_train.npz")["names"]]
    cols = ck["cols"]
    mu, sd = np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32)
    model = MLPSurvival(len(cols), 100, 256, head=args.head)
    model.load_state_dict(ck["model"])
    model.eval().to(dev)
    cube = CubeReader(exp)
    all_names = feature_names(cube.n_dyn, cube.n_stat)
    take = np.array([all_names.index(n) for n in train_names])[cols]
    locked = np.array(sorted(LOCKED_CHANNELS))
    mdir = exp / "derived_data" / "masks"
    lat, area, region = (np.load(mdir / f) for f in ("latitude.npy", "cell_area_m2.npy", "region.npy"))
    observed = np.load(mdir / "observed_nfdb.npy")
    print(f"[ours] B4/{args.head}: {len(cols)} Merkmale, Checkpoint Epoche {ck.get('epoch')}")

    @torch.no_grad()
    def hazards(Xraw):
        Xs = (np.nan_to_num(Xraw[:, take]) - mu) / sd
        return model.hazards(torch.from_numpy(Xs.astype(np.float32)).to(dev)).double().cpu().numpy()

    # ---------------------------------------------- Stichproben des Caches
    d = exp / "derived_data" / "paper" / args.cache
    meta, y, split = np.load(d / "meta.npy"), np.load(d / "y.npy").astype(int), np.load(d / "split.npy")
    t_of = {s: i for i, s in enumerate(cube.dates)}

    def landmark(i, dtv):
        D = dt.date(int(meta[i, 0] // 10000), int(meta[i, 0] // 100 % 100), int(meta[i, 0] % 100))
        L = t_of.get((D - dt.timedelta(days=dtv + 1)).isoformat())
        if L is None or L not in cube._pos or (L - LOOKBACK + 1) not in cube._pos:
            return None
        if cube._pos[L - LOOKBACK + 1][0] != cube._pos[L][0] or \
           cube._pos[L - LOOKBACK + 1][1] != cube._pos[L][1] - LOOKBACK + 1:
            return None
        return L

    idx = {s: np.where(split == s)[0] for s in ("val", "test")}
    Ls = {s: {dtv: [landmark(i, dtv) for i in idx[s]] for dtv in DTS} for s in idx}
    common = {s: np.array([all(Ls[s][dtv][k] is not None for dtv in DTS)
                           for k in range(idx[s].size)]) for s in idx}
    print(f"[ours] gemeinsame Teilmenge: val {common['val'].sum()}/{idx['val'].size}, "
          f"test {common['test'].sum()}/{idx['test'].size}")

    # Selbsttest: Mittelzelle des Blocks == row_features
    for k in range(5):
        i = idx["test"][np.nonzero(common["test"])[0][k]]
        L = Ls["test"][1][np.nonzero(common["test"])[0][k]]
        r, c = int(meta[i, 1]), int(meta[i, 2])
        Xp, rr, cc = patch_features(cube, L, r, c, lat, area, region, locked)
        j = np.nonzero((rr == r) & (cc == c))[0][0]
        ref = row_features(cube, L, r, c, lat, area, region, locked)
        if not np.allclose(Xp[j], ref, rtol=1e-5, atol=1e-4, equal_nan=True):
            raise SystemExit("ABBRUCH: Block-Merkmale weichen von row_features ab")
    print("[ours] Selbsttest: Block-Merkmale der Mittelzelle == row_features")

    # ---------------------------------------------- Scores
    scores = {s: {dtv: {k: np.full(idx[s].size, np.nan) for k in SCORES} for dtv in DTS} for s in idx}
    for s in idx:
        for dtv in DTS:
            k = dtv + 1
            for n, i in enumerate(idx[s]):
                if not common[s][n]:
                    continue
                r, c = int(meta[i, 1]), int(meta[i, 2])
                Xp, rr, cc = patch_features(cube, Ls[s][dtv][n], r, c, lat, area, region, locked)
                hz = hazards(Xp)                                        # (n_pix, 100)
                hk = hz[:, k - 1]
                Fk = 1 - np.prod(1 - hz[:, :k], axis=1)
                obs = observed[rr, cc]
                ctr = (rr == r) & (cc == c)
                sc = scores[s][dtv]
                sc["centre_h"][n], sc["centre_F"][n] = hk[ctr][0], Fk[ctr][0]
                sc["patch_h"][n] = -np.expm1(np.log1p(-hk[obs]).sum()) if obs.any() else 0.0
                sc["patch_F"][n] = -np.expm1(np.log1p(-Fk[obs]).sum()) if obs.any() else 0.0
            print(f"[ours] {s} dt={dtv} fertig", flush=True)

    # ---------------------------------------------- Auswertung
    def logit(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    yv, yt = y[idx["val"]][common["val"]], y[idx["test"]][common["test"]]
    res: dict = {"ours": {}, "unet3d": {}}
    for dtv in DTS:
        for name in SCORES:
            sv = scores["val"][dtv][name][common["val"]]
            st = scores["test"][dtv][name][common["test"]]
            platt = LogisticRegression(C=1e6).fit(logit(sv)[:, None], yv)
            p_t = platt.predict_proba(logit(st)[:, None])[:, 1]
            m = paper_metrics(p_t, yt)
            res["ours"].setdefault(name, {})[str(dtv)] = m
    unet_path = exp / "predictions" / "paper" / f"rec_{args.cache}_cache{args.unet_run}.npz"
    if unet_path.exists():
        z = np.load(unet_path)
        pos = {int(i): n for n, i in enumerate(z["idx_test"])}
        sel = np.array([pos[int(i)] for i in idx["test"][common["test"]]])
        for dtv in DTS:
            per_seed = [paper_metrics(z[k][sel], yt) for k in z.files
                        if k.startswith(f"paper55_T30_end{dtv + 1}_s") and k.endswith("_test")]
            res["unet3d"][str(dtv)] = {m: float(np.mean([p[m] for p in per_seed]))
                                       for m in per_seed[0] if isinstance(per_seed[0][m], float)}
            res["unet3d"][str(dtv)]["n_seeds"] = len(per_seed)
    else:
        print(f"[ours] WARNUNG: keine 3D-U-Net-Vorhersagen unter {unet_path}")

    keys = ("f1", "pr_auc", "roc_auc", "precision", "recall", "ece", "mce", "brier")
    print(f"\n[ours] {args.cache}, Test, gemeinsame Teilmenge n={yt.size} "
          f"(Positive {int(yt.sum())}):  " + "  ".join(keys))
    for dtv in DTS:
        if str(dtv) in res["unet3d"]:
            u = res["unet3d"][str(dtv)]
            print(f"    dt={dtv:2d} 3D-U-Net        " + "  ".join(f"{u[k]:.3f}" for k in keys))
        for name in SCORES:
            m = res["ours"][name][str(dtv)]
            print(f"    dt={dtv:2d} B4 {name:9s}    " + "  ".join(f"{m[k]:.3f}" for k in keys))

    write_manifest(cfg, f"p4_ours_on_paper_{args.cache}", args.config, {
        "cache": args.cache, "model": f"B4/{args.head}", "checkpoint_epoch": ck.get("epoch"),
        "protocol": "Landmark L = t - dt - 1, Vorlauf k = dt + 1; Platt-Skalierung auf val",
        "metric_assumptions": ASSUMPTIONS,
        "n_common": {"val": int(common["val"].sum()), "test": int(common["test"].sum())},
        "n_total": {s: int(v.size) for s, v in idx.items()},
        "results": res,
    })


if __name__ == "__main__":
    main()

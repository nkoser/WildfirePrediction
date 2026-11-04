"""Gruppen-Shapley getrennt nach harten und repraesentativen Negativen (FWI-HNM), Paper-Protokoll D+1.

Hypothese: Die Caches ziehen die Negativen je zur Haelfte ueber und unter dem Median des FWI-HNM-Gefahrenwerts
s = sum_i w_i min(x_i / P99_i, 1) (FFMC, DMC, DC, ISI, BUI, FWI). Harte Negative haben hohe Feuergefahr ohne Feuer,
dort trennt das Wetter schlecht und das Netz muss die Feuerhistorie nutzen. Erwartung: phi(fire) groesser und
phi(era5_weather + fire_danger) kleiner auf "Positive + harte Negative" als auf "Positive + repraesentative Negative".

Nur CPU, kein Neutraining: liest die Scores aus shapley_groups (metrics/shapley_groups<src>/scores_*.npz, 1024
Teilmengen x dt x Stichproben) und wertet sie auf neuen Stichproben-Teilmengen aus.

  s           am Zentrumspixel und Tag D (letzter Cache-Tag, dort ziehen die Caches), Gewichte des Papers
              (--weights paper, Standard) oder unsere Rekonstruktion (--weights ours, p3_hnm_score), P99 jeweils
              aus p3_hnm_score (unser Trainingsset-Nachbau)
  hart/repr.  Cache-Negative (Label an D = 0) ueber/unter dem Median von s unter ALLEN Test-Negativen des Caches
              (dort ist der 50/50-Schnitt per Konstruktion gesetzt); zusaetzlich protokolliert: Anteil ueber dem
              Pool-Median aus p3_hnm_score
  Teilmengen  pos_hard = Cache-Positive + harte Negative, pos_rep = Cache-Positive + repraesentative Negative,
              all = alle; Label jeweils y_next_patch bzw. Mitte an D+1 wie in shapley_groups

Ausgaben (neu): metrics/shapley_hnm<tag>/summary.md, shapley_hnm.csv, Manifest p54_shapley_hnm<tag>.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.hnm import FWI_CH, NAMES, score
from wildfire.paper.shapley_groups import GN, METRICS, SUBSETS, mask_of, setup, shapley

PAPER_W = np.array([0.152, 0.184, 0.127, 0.164, 0.191, 0.182])
WEATHER = ("era5_weather", "fire_danger")


def ap_rows(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Average Precision wie sklearn (Gleichstaende als eine Schwelle) fuer jede Zeile von S (M, n) auf einmal."""
    o = np.argsort(-S, axis=1, kind="stable")
    s = np.take_along_axis(S, o, 1)
    yy = y[o].astype(np.float64)
    tps = np.cumsum(yy, 1)
    k = np.arange(1, S.shape[1] + 1)[None]
    last = np.ones_like(s, bool)
    last[:, :-1] = s[:, :-1] != s[:, 1:]                                            # letzter Index je Schwelle
    tb = np.where(last, tps, 0.0)
    prev = np.zeros_like(tb)
    prev[:, 1:] = np.maximum.accumulate(tb, 1)[:, :-1]                             # tps an der vorigen Schwelle
    return np.sum(np.where(last, (tps - prev) * tps / k, 0.0), 1) / y.sum()


def auc_rows(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """ROC-AUC (Mann-Whitney, mittlere Raenge bei Gleichstand) je Zeile."""
    from scipy.stats import rankdata
    r = rankdata(S, axis=1)
    p = int(y.sum()); n = y.size - p
    return (r[:, y == 1].sum(1) - p * (p + 1) / 2) / (p * n)


FAST = {"pr_auc": ap_rows, "roc_auc": auc_rows}


def hnm_split(d: dict, cd: Path, p99: np.ndarray, w: np.ndarray, pool_med: float) -> dict:
    """s je Teststichprobe (Mitte, Tag D) und Schnitt der Cache-Negativen am Median aller Test-Negativen."""
    split = np.load(cd / "split.npy")
    y_d = targets.labels(cd, "cache")
    te_all = np.where(split == "test")[0]
    raw = np.asarray(d["Xp"][te_all, d["LC"] - 1][:, FWI_CH, 6, 6]).astype(np.float64)
    s_all = score(raw, p99, w)
    neg_all = y_d[te_all] == 0
    med = float(np.median(s_all[neg_all]))
    pos = np.searchsorted(te_all, d["te"])
    s, yd = s_all[pos], y_d[d["te"]]
    hard = (yd == 0) & (s > med)
    rep = (yd == 0) & (s <= med)
    return {"s": s, "y_d": yd, "hard": hard, "rep": rep, "median_neg": med,
            "share_neg_above_pool_median": float((s_all[neg_all] > pool_med).mean()),
            "s_quantiles": {lab: np.percentile(s_all[(y_d[te_all] == k)], [10, 50, 90]).tolist()
                            for lab, k in (("positive", 1), ("negative", 0))}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--n-eval", type=int, default=3000, help="muss zu den Scores aus shapley_groups passen")
    ap.add_argument("--src-tag", default="", help="Tag des shapley_groups-Laufs, dessen Scores gelesen werden")
    ap.add_argument("--weights", choices=["paper", "ours"], default="paper")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    t0 = time.time()
    d = setup(cfg, args.cache, args.n_eval, torch.device("cpu"))
    exp = d["exp"]
    cd = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    src = exp / "metrics" / f"shapley_groups{args.src_tag}"
    out_dir = exp / "metrics" / f"shapley_hnm{args.tag}"
    man_dir = exp / "data_manifests"
    stage = f"p54_shapley_hnm{args.tag}"
    if (man_dir / f"{stage}.json").exists():
        raise SystemExit(f"ABBRUCH: {stage} existiert schon")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"ABBRUCH: {out_dir} ist nicht leer")
    out_dir.mkdir(parents=True, exist_ok=True)

    hm = json.load(open(man_dir / "p3_hnm_score.json"))
    p99 = np.array(hm["p99"])
    w = PAPER_W if args.weights == "paper" else np.array(hm["w"])
    hs = hnm_split(d, cd, p99, w, hm["pool_median_nofire"])
    # Kontrolle: dieselbe Aufteilung mit der jeweils anderen Gewichtung
    w_alt = np.array(hm["w"]) if args.weights == "paper" else PAPER_W
    alt = hnm_split(d, cd, p99, w_alt, hm["pool_median_nofire"])
    neg = hs["y_d"] == 0
    agree = float((hs["hard"][neg] == alt["hard"][neg]).mean())
    print(f"[shh] Test: pos {int((hs['y_d'] == 1).sum())}, hart {int(hs['hard'].sum())}, repr. {int(hs['rep'].sum())}; "
          f"Median s (Negative) {hs['median_neg']:.4f}; Uebereinstimmung paper/ours {agree:.3f}", flush=True)

    keep = {"all": np.ones(d["te"].size, bool), "pos_hard": (hs["y_d"] == 1) | hs["hard"],
            "pos_rep": (hs["y_d"] == 1) | hs["rep"]}
    counts = {k: {h: {"n": int(m.sum()), "pos_next": int(d["y"][h][m].sum())} for h in ("patch_head", "centre")}
              for k, m in keep.items()}
    # Wie viele Cache-Negative brennen an D+1 (zaehlen dann als Positive)?
    neg_next = {k: int(d["y"]["patch_head"][hs[k]].sum()) for k in ("hard", "rep")}

    rows, V = [], {}
    max_dev = {met: 0.0 for met in FAST}
    for fpath in sorted(src.glob("scores_*.npz")):
        name, seed = fpath.stem[len("scores_"):].rsplit("_", 1)
        z = np.load(fpath)
        if not np.array_equal(z["rows"], d["te"]):
            raise SystemExit(f"ABBRUCH: {fpath.name} mit anderen Stichproben erzeugt")
        for h in ("patch_head", "centre"):
            for j, dt_ in enumerate(DTS):
                for sub, sel in keep.items():
                    y = d["y"][h][sel]
                    if y.min() == y.max():
                        continue
                    for met, fn in FAST.items():
                        Sm = z[h][:, j][:, sel].astype(np.float64)
                        v = fn(y, Sm)
                        # Selbsttest gegen sklearn an einigen Teilmengen
                        chk = [0, 1, 2 ** len(GN) - 1, 357]
                        dev = max(abs(v[m] - METRICS[met](y, Sm[m])) for m in chk)
                        if dev > 1e-9:
                            raise SystemExit(f"ABBRUCH: {met} weicht von sklearn ab ({dev:.2e}) {fpath.name} {h} {dt_} {sub}")
                        max_dev[met] = max(max_dev[met], dev)
                        V[(name, seed, h, dt_, sub, met)] = v
                        phi = shapley(v)
                        rows += [{"model": name, "seed": seed, "head": h, "dt": dt_, "subset": sub, "metric": met,
                                  "group": g, "phi": float(phi[i]), "v_full": float(v[-1]), "v_empty": float(v[0]),
                                  "v_only": float(v[1 << i]), "v_without": float(v[(2 ** len(GN) - 1) ^ (1 << i)])}
                                 for i, g in enumerate(GN)]
        print(f"[shh] {fpath.name} ausgewertet ({time.time() - t0:.0f} s); max |Abw. sklearn| "
              + " ".join(f"{k} {x:.1e}" for k, x in max_dev.items()), flush=True)
    if not rows:
        raise SystemExit(f"ABBRUCH: keine Scores in {src}")
    with open(out_dir / "shapley_hnm.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)

    # ------------------------------------------------------------------ Bericht
    models = sorted({r["model"] for r in rows})

    def stat(vals):
        return float(np.mean(vals)), float(np.std(vals))

    def phis(mdl, h, dt_, sub, met, g):
        return [r["phi"] for r in rows if (r["model"], r["head"], r["dt"], r["subset"], r["metric"], r["group"])
                == (mdl, h, dt_, sub, met, g)]

    def vs(mdl, h, dt_, sub, met, m):
        return [v[m] for k, v in V.items() if k[0] == mdl and k[2:] == (h, dt_, sub, met)]

    m_fire, m_weather, m_full = mask_of(["fire"]), mask_of(WEATHER), mask_of(GN)
    L = ["# Gruppen-Shapley nach Art der Negativen (FWI-HNM), Paper-Protokoll D+1", "",
         f"Cache {args.cache}, dieselben {d['te'].size:,} Test-Stichproben und Scores wie "
         f"metrics/shapley_groups{args.src_tag} (Gruppen-Dropout-Netze, 1024 Teilmengen exakt), Mittel ± Std ueber Seeds.",
         f"Gefahrenwert s am Zentrumspixel an D, Gewichte: {args.weights} "
         f"({', '.join(f'{n} {x:.3f}' for n, x in zip(NAMES, w))}), P99 aus p3_hnm_score.",
         f"Schnitt: Median von s unter allen Test-Negativen des Caches = {hs['median_neg']:.4f}. "
         f"Anteil der Test-Negativen ueber dem Pool-Median (p3_hnm_score): {hs['share_neg_above_pool_median']:.3f}. "
         f"Gleiche Zuordnung mit der anderen Gewichtung: {agree:.3f}.",
         f"s-Quantile 10/50/90 im Test: Positive {'/'.join(f'{x:.3f}' for x in hs['s_quantiles']['positive'])}, "
         f"Negative {'/'.join(f'{x:.3f}' for x in hs['s_quantiles']['negative'])}.", "",
         "Teilmengen (Label Feuer an D+1): " + "; ".join(
             f"{k} n {c['patch_head']['n']:,} (Positive Ausschnitt {c['patch_head']['pos_next']:,}, "
             f"Mitte {c['centre']['pos_next']:,})" for k, c in counts.items()) + ".",
         f"Cache-Negative mit Feuer an D+1 (Ausschnitt): hart {neg_next['hard']}, repr. {neg_next['rep']}.", ""]
    for met in ("pr_auc", "roc_auc"):
        for h, hn in (("patch_head", "Ausschnitt"), ("centre", "Mitte")):
            for mdl in models:
                L += [f"## {mdl}, {hn}, {met}", "",
                      "| Gruppe | " + " | ".join(f"{sub} dt {dt_}" for dt_ in DTS for sub in ("pos_hard", "pos_rep"))
                      + " |", "|---" * (1 + 2 * len(DTS)) + "|"]
                for g in GN + ["weather+danger"]:
                    cells = []
                    for dt_ in DTS:
                        for sub in ("pos_hard", "pos_rep"):
                            if g == "weather+danger":
                                per_seed = np.add(*[phis(mdl, h, dt_, sub, met, x) for x in WEATHER])
                            else:
                                per_seed = phis(mdl, h, dt_, sub, met, g)
                            mu, sd = stat(per_seed) if len(per_seed) else (np.nan, np.nan)
                            cells.append(f"{mu:+.4f} ± {sd:.4f}")
                    L.append(f"| {g} | " + " | ".join(cells) + " |")
                for lab, m in (("v(alle)", m_full), ("v(nur fire)", m_fire), ("v(nur Wetter+Gefahr)", m_weather),
                               ("v(leer)", 0)):
                    L.append(f"| {lab} | " + " | ".join(f"{stat(vs(mdl, h, dt_, sub, met, m))[0]:.3f}"
                                                         for dt_ in DTS for sub in ("pos_hard", "pos_rep")) + " |")
                L.append("")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")

    write_manifest(cfg, stage, args.config, {
        "cache": args.cache, "n_eval": int(d["te"].size), "src": str(src), "weights": args.weights,
        "w": w.tolist(), "p99": p99.tolist(), "median_neg": hs["median_neg"],
        "share_neg_above_pool_median": hs["share_neg_above_pool_median"], "agreement_paper_ours": agree,
        "s_quantiles_test": hs["s_quantiles"], "counts": counts, "neg_fire_next": neg_next,
        "max_dev_vs_sklearn": max_dev,
        "outputs": {"summary": str(out_dir / "summary.md"), "csv": str(out_dir / "shapley_hnm.csv")},
    })
    print(f"[shh] fertig ({time.time() - t0:.0f} s) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

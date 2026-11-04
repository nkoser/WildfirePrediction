"""Varianten-Ablation (sts_extreme / sts_focus) auf drei Splits: zeitlich, Jahre stratifiziert, Bloecke stratifiziert.

Liest nur vorhandene Ergebnisse, nichts wird neu trainiert:
  Paper-Protokoll  metrics/focus_eval*/results.json (PR-AUC Ausschnitt je dt, alle Test-Stichproben und ign30, Seed-Mittel)
  Survival         metrics/sts_extreme*/summary.csv (Ausschnitt, kalibriert: NLL, O/E@30; Seed-Mittel)
  U-Net-Nachbau    zeitlich aus focus_eval (unet_rec); fuer die neuen Splits aus predictions/paper/
                   rec_<cache>_cache_next_paperhp_w0.npz (226/230), ign30 wie focus_eval aus dem Feuerkanal.
Zeitlich: focus_eval + focus_eval_cgfdmono, sts_extreme_focus_all + sts_extreme_cgfdmono (Modelle ohne Praefix).
Neue Splits: focus_eval_<sp>, sts_extreme_<sp> (235/236), Modelle mit Praefix <sp>_.
Ausgabe NEU: metrics/split_ablation<tag>/summary.md, summary.csv; Manifest p51_split_ablation<tag>.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from wildfire.audit.common import load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.prepare import FIRE_CH

MODELS = ["lh_ref", "lh_fdrop", "lh_resid", "lh_ignw", "lh_fdropall", "lh_fdrop_mono", "lh_fdrop_mono_rf",
          "cg_ref", "cg_mono", "cg_fdrop", "cg_ignw", "cg_fdrop_mono", "cg_fdrop_mono_rf", "fire_logit", "unet_rec"]
LABEL = {"fire_logit": "Feuerhistorie (logistisch)", "unet_rec": "U-Net-Nachbau"}
SPLITS = {
    "zeitlich": {"prefix": "", "focus": ["focus_eval", "focus_eval_cgfdmono"],
                 "surv": ["sts_extreme_focus_all", "sts_extreme_cgfdmono"], "cache": "cache_G"},
    "Jahre": {"prefix": "ys_", "focus": ["focus_eval_ys"], "surv": ["sts_extreme_ys"], "cache": "cache_G_yearstrat"},
    "Bloecke": {"prefix": "bs_", "focus": ["focus_eval_bs"], "surv": ["sts_extreme_bs"], "cache": "cache_G_blockstrat"},
}


def unet_scores(exp: Path, cache: str) -> dict | None:
    rp = exp / "predictions" / "paper" / f"rec_{cache}_cache_next_paperhp_w0.npz"
    if not rp.exists():
        return None
    cd = exp / "derived_data" / "paper" / f"{cache}_L61"
    te = np.where(np.load(cd / "split.npy") == "test")[0]
    z = np.load(rp)
    pos = {int(i): n for n, i in enumerate(z["idx_test"])}
    sel = np.array([pos[int(i)] for i in te])
    y = targets.labels(cd, "next_patch")[te]
    Xp = np.load(cd / "X.npy", mmap_mode="r")
    LC = Xp.shape[1]
    fire = np.asarray(Xp[te][:, :, FIRE_CH]) > 0                       # (n, LC, 13, 13)
    out = {"all": {}, "ign30": {}}
    seeds = [s for s in range(3) if f"paper55_T30_end1_s{s}_test" in z.files]
    for dt_ in DTS:
        e = LC - 1 - dt_
        ign = ~fire[:, e - 29:e + 1].reshape(te.size, -1).any(1)
        for sn, m in (("all", np.ones(te.size, bool)), ("ign30", ign)):
            per = [average_precision_score(y[m], z[f"paper55_T30_end{dt_}_s{s}_test"][sel][m]) for s in seeds]
            out[sn][str(dt_)] = float(np.mean(per))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", choices=["next_patch", "cache"], default="next_patch",
                    help="cache = Ziel D: PR-AUC aus focus_eval_target_D_<t|ys|bs> (238), U-Net dort enthalten")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"split_ablation{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon")
    if args.target == "cache":
        for sp, t in zip(SPLITS, ("t", "ys", "bs")):
            SPLITS[sp] = dict(SPLITS[sp], focus=[f"focus_eval_target_D_{t}"], prefix_unet=False)

    table = {}                                                          # split -> model -> dict
    for sp, spec in SPLITS.items():
        res = {}
        for f in spec["focus"]:                                         # erste Quelle gewinnt
            p = exp / "metrics" / f / "results.json"
            if p.exists():
                for k, v in json.loads(p.read_text()).items():
                    res.setdefault(k, v)
        surv = {}
        for f in spec["surv"]:
            p = exp / "metrics" / f / "summary.csv"
            if p.exists():
                for r in csv.DictReader(p.open()):
                    if r["member"].startswith("s"):
                        surv.setdefault(r["name"], {}).setdefault(f, []).append(r)
        un = unet_scores(exp, spec["cache"]) if spec["prefix"] and spec.get("prefix_unet", True) else None
        rows = {}
        for m in MODELS:
            key = m if m in ("fire_logit", "unet_rec") else spec["prefix"] + m
            d = {}
            r = res.get(key, {}).get("patch", {})
            for sn in ("all", "ign30"):
                if un is not None and m == "unet_rec":
                    d[sn] = un[sn]
                elif sn in r:
                    d[sn] = {k: v["pr_auc_mean"] for k, v in r[sn].items()}
            src = next(iter(surv.get(key, {}).values()), None)          # erste Tabelle mit allen Seeds
            if src:
                d["nll"] = float(np.mean([float(x["patch_curve_nll"]) for x in src]))
                d["oe"] = float(np.mean([float(x["patch_oe30_curve"]) for x in src]))
                d["n_seeds"] = len(src)
            rows[m] = d
        table[sp] = rows

    f3 = lambda d, sn: "/".join(f"{d[sn][str(t)]:.3f}".lstrip("0") for t in DTS) if sn in d else "–"  # noqa: E731
    tl = "D+1" if args.target == "next_patch" else "D"
    md = [f"# Varianten-Ablation auf drei Splits, Ziel {tl}", "",
          f"Ausschnitt, Ziel {tl}, Seed-Mittel. PR-AUC je dt " + "/".join(map(str, DTS)) + " auf allen Test-Stichproben "
          "und auf neuen Zuendungen (ign30, kein Feuer im Ausschnitt an den 30 Fenstertagen). NLL und O/E@30 kalibriert "
          "auf zufaelligen Survival-Stichtagen des jeweiligen Val/Test-Teils (zeitlich: Test 2023-2025).", ""]
    for sp in SPLITS:
        md += [f"## {sp}", "", "| Modell | PR-AUC alle | PR-AUC ign30 | NLL kal. | O/E@30 kal. |", "|---|---|---|---|---|"]
        for m, d in table[sp].items():
            nll = f"{d['nll']:.4f}" if "nll" in d else "–"
            oe = f"{d['oe']:.2f}" if "oe" in d else "–"
            md.append(f"| {LABEL.get(m, m)} | {f3(d, 'all')} | {f3(d, 'ign30')} | {nll} | {oe} |")
        md.append("")
    md += ["## PR-AUC alle, dt 1 und dt 30 nebeneinander", "",
           "| Modell | " + " | ".join(f"{sp} dt1 | {sp} dt30" for sp in SPLITS) + " |", "|---" * (1 + 2 * len(SPLITS)) + "|"]
    for m in MODELS:
        cells = []
        for sp in SPLITS:
            d = table[sp][m].get("all")
            cells += [f"{d['1']:.3f}" if d else "–", f"{d['30']:.3f}" if d else "–"]
        md.append(f"| {LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    out.mkdir(parents=True)
    (out / "summary.md").write_text("\n".join(md) + "\n")
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "model"] + [f"{sn}_dt{t}" for sn in ("all", "ign30") for t in DTS] + ["nll_cal", "oe30_cal"])
        for sp, rows in table.items():
            for m, d in rows.items():
                w.writerow([sp, m] + [d.get(sn, {}).get(str(t), "") for sn in ("all", "ign30") for t in DTS]
                           + [d.get("nll", ""), d.get("oe", "")])
    write_manifest(cfg, f"p51_split_ablation{args.tag}", args.config, {"target": args.target, "splits": SPLITS, "table": table,
                                                                       "out": str(out)})
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()

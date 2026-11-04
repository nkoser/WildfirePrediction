"""Tabellen aus dem Manifest p31_extreme_eval (extreme_eval.py) als Markdown auf stdout und CSV unter
metrics/extreme_eval/.

  A  Eingaben je Jahr: p95 / p99 / Maximum, Anteil bei 255, Anteil ueber dem Zellmaximum 2000-2019
  B  Anteil der Test-Stichtage in den Extremgruppen je Testjahr
  C  je Modell und Ziel: O/E@7, O/E@30, PR-AUC@30, ROC-AUC@30 je Gruppe und Variante
  D  Saison-Nachkalibrierung: gewaehltes W, k und NLL je Variante
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from wildfire.audit.common import load_config, read_manifest

CH = ("FWI", "DC", "BUI", "ISI", "FFMC")
GROUPS = ("alle", "jahr_2023", "jahr_2024", "jahr_2025",
          "FWI_lt_p50", "FWI_p50_90", "FWI_p90_95", "FWI_p95_99", "FWI_ge_p99",
          "DC_lt_p50", "DC_p50_90", "DC_p90_95", "DC_p95_99", "DC_ge_p99",
          "FWI_und_DC_ge_p95", "FWI_ueber_zell_p99", "FWI_ueber_zell_max", "DC_ueber_zell_p99", "DC_ueber_zell_max",
          "DC_ge_p95_jahr_2023", "DC_ge_p95_jahr_2024", "DC_ge_p95_jahr_2025")
VARIANTS = ("raw", "curve", "season", "oracle_year")


def f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = read_manifest(cfg, f"p31_extreme_eval{args.tag}")
    out = Path(cfg["experiment_root"]) / "metrics" / "extreme_eval"
    out.mkdir(parents=True, exist_ok=True)

    print(f"# p31_extreme_eval{args.tag}  (day_step {m['day_step']}, Selbsttest {m['self_test_agree']:.4f}, "
          f"{m['n_cells']:,} Zellen)\n")
    print("Schwellen (uint8) der gepoolten Trainingsverteilung:", {c: m["train_rank_thresholds"][c] for c in CH}, "\n")

    rows = []
    print("## A Eingaben je Jahr (Zellen der Stichtage, Saison)\n")
    print("| Jahr | " + " | ".join(f"{c} p95/p99/max | {c} @255 | {c} >Zellmax" for c in CH) + " |")
    print("|---|" + "---|---|---|" * len(CH))
    for y, row in m["inputs_by_year"].items():
        cells = []
        for c in CH:
            r = row[c]
            cells += [f"{r['p95']}/{r['p99']}/{r['max']}", f(r["share_255"], 4), f(r["share_above_cell_max"], 4)]
            rows.append({"jahr": y, "kanal": c, **r})
        print(f"| {y} | " + " | ".join(cells) + " |")
    with open(out / f"inputs_by_year{args.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print("\n## B Anteil der Test-Stichtage je Gruppe und Jahr\n")
    sh = m["group_share_by_year"]["test"]
    yrs = sorted(next(iter(sh.values())))
    print("| Gruppe | " + " | ".join(yrs) + " |\n|---|" + "---|" * len(yrs))
    for g in ("FWI_p95_99", "FWI_ge_p99", "DC_p95_99", "DC_ge_p99", "FWI_und_DC_ge_p95",
              "FWI_ueber_zell_max", "DC_ueber_zell_max", "DC_ueber_zell_p99"):
        print(f"| {g} | " + " | ".join(f(sh[g][y], 4) for y in yrs) + " |")
    vs = m["group_share_by_year"]["val"]
    print("\nVal-Jahre zum Vergleich:")
    for g in ("FWI_ge_p99", "DC_ge_p99", "DC_ueber_zell_max"):
        print(f"  {g}: " + ", ".join(f"{y} {f(v, 4)}" for y, v in sorted(vs[g].items())))

    rows = []
    for name, r in m["results"].items():
        for tg in ("patch", "centre"):
            res = r[tg]
            sizes = res["group_sizes"]
            print(f"\n## C {name} / {tg}  (curve a, b = {f(res['curve_ab'][0])}, {f(res['curve_ab'][1])}; "
                  f"Saison {res['season_best']}; Orakel-c {', '.join(f'{y} {f(c)}' for y, c in res['oracle_c'].items())})\n")
            print("| Gruppe | n | Feuer | " + " | ".join(f"{v} O/E@7 | {v} O/E@30" for v in VARIANTS)
                  + " | PR@30 | AUC@30 |")
            print("|---|---|---|" + "---|---|" * len(VARIANTS) + "---|---|")
            for g in GROUPS:
                if g not in res["variants"]["raw"]:
                    continue
                cells = []
                for v in VARIANTS:
                    s = res["variants"][v][g]
                    cells += [f(s["o_e@7"]), f(s["o_e@30"])]
                    rows.append({"modell": name, "ziel": tg, "gruppe": g, "variante": v, "n": sizes[g]["n"],
                                 "feuer": sizes[g]["events"], **{k: s.get(k) for k in
                                 ("nll", "dcal_dev", "o_e@7", "o_e@30", "o_e@100", "pr@7", "pr@30", "auc@7", "auc@30")}})
                raw = res["variants"]["raw"][g]
                print(f"| {g} | {sizes[g]['n']} | {sizes[g]['events']} | " + " | ".join(cells)
                      + f" | {f(raw['pr@30'])} | {f(raw['auc@30'])} |")
            print("\nD NLL je Variante (alle): " + ", ".join(
                f"{v} {f(res['variants'][v]['alle']['nll'], 4)}" for v in VARIANTS))
            print("  Saisongitter (Val-NLL / Test-NLL): " + ", ".join(
                f"{k} {f(s['val_nll'], 4)}/{f(s['test_nll'], 4)}" for k, s in res["season_grid"].items()))
    with open(out / f"groups{args.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n[ext] CSV unter {out}")


if __name__ == "__main__":
    main()

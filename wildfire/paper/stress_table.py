"""Tabellen aus dem Manifest p32_stress_test (stress_test.py) als Markdown auf stdout und CSV unter
metrics/stress_test/.

  S1  Verschiebung je Gruppe: F30 relativ zum unveraenderten Stichtag je Delta, Anteil Stichtage mit
      Abfall irgendwo in der Kette (Toleranz tol)
  S2  feste Werte je Kanal (Stichtag bzw. ganzes Fenster): F30 relativ zur Basis je Stufe, Abfall-Anteil
  S3  FWI/ISI-Code 254 -> 255
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from wildfire.audit.common import load_config, read_manifest

GROUPS = ("alle_zufaellig", "DC_ge_p99", "DC_ueber_zell_max")


def f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--variant", default="ens", help="ens oder s0/s1/s2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = read_manifest(cfg, f"p32_stress_test{args.tag}")
    out = Path(cfg["experiment_root"]) / "metrics" / "stress_test"
    out.mkdir(parents=True, exist_ok=True)
    print(f"# p32_stress_test{args.tag}  (Schwellen uint8: {m['thresholds_uint8']}, tol {m['tol']})\n")

    rows, crow = [], []
    for name, by_v in m["results"].items():
        for vn, by_t in by_v.items():
            for tg, by_g in by_t.items():
                for gn, g in by_g.items():
                    for sn, s in g["scen"].items():
                        rows.append({"modell": name, "variante": vn, "ziel": tg, "gruppe": gn, "n": g["n"],
                                     "szenario": sn, **s})
                    for cn, c in g["chains"].items():
                        crow.append({"modell": name, "variante": vn, "ziel": tg, "gruppe": gn, "kette": cn,
                                     **{k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in c.items()}})
    for fn, rr in (("scenarios", rows), ("chains", crow)):
        keys = sorted({k for r in rr for k in r}, key=lambda k: list(rr[0]).index(k) if k in rr[0] else 99)
        with open(out / f"{fn}{args.tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rr)

    deltas = m["deltas"]
    for name, by_v in m["results"].items():
        vn = args.variant if args.variant in by_v else next(iter(by_v))
        for tg, by_g in by_v[vn].items():
            print(f"\n## {name} / {vn} / {tg}\n")
            print("### S1 Verschiebung (F30 / F30 Basis; Abfall-Anteil mit Toleranz)\n")
            dl = [d for d in deltas if d < 0] + [0] + [d for d in deltas if d > 0]
            print("| Gruppe | Kanaele | " + " | ".join(f"{d:+d}" for d in dl) + " | Abfall irgendwo | Basis F30 |")
            print("|---|---|" + "---|" * (len(dl) + 2))
            for gn in GROUPS:
                if gn not in by_g:
                    continue
                g = by_g[gn]
                for sg in m["shift_groups"]:
                    cells = [f(g["scen"]["base" if d == 0 else f"shift_{sg}_{d:+d}"]["ratio_F30"], 2) for d in dl]
                    print(f"| {gn} (n {g['n']}) | {sg} | " + " | ".join(cells)
                          + f" | {f(g['chains'][f'shift_{sg}']['any_tol'])} | {f(g['scen']['base']['F30'], 4)} |")
            print("\n### S2 feste Werte (F30 / F30 Basis je Stufe, aufsteigend; Abfall irgendwo)\n")
            print("| Gruppe | Kanal | Modus | Stufen (uint8) | F30-Verhaeltnis | Abfall irgendwo |")
            print("|---|---|---|---|---|---|")
            for gn in GROUPS:
                if gn not in by_g:
                    continue
                g = by_g[gn]
                for cn, c in g["chains"].items():
                    if not cn.startswith("level_"):
                        continue
                    ch, mode = cn[len("level_"):cn.rfind("_")], cn.rsplit("_", 1)[1]
                    rat = " / ".join(f(g["scen"][s]["ratio_F30"], 2) for s in c["order"])
                    vals = " / ".join(f"{v:.0f}" for v in c["values"])
                    print(f"| {gn} | {ch} | {mode} | {vals} | {rat} | {f(c['any_tol'])} |")
            print("\n### S3 Code 254 -> 255 (ganzes Fenster)\n")
            print("| Gruppe | Kanaele | F30(255) / F30(254) | Anteil F30(255) < F30(254) (tol) |")
            print("|---|---|---|---|")
            for gn in GROUPS:
                if gn not in by_g:
                    continue
                for cn, c in by_g[gn]["chains"].items():
                    if cn.startswith("code_"):
                        print(f"| {gn} | {cn[5:-11]} | {f(c['ratio_F30'], 2)} | {f(c['share_255_below_254_tol'])} |")
    print(f"\n[stress] CSV unter {out}")


if __name__ == "__main__":
    main()

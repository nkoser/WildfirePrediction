"""Zusammenfassung der Laeufe von sts_extreme.py (slurm 166/167): Mittel und Spannweite ueber die Seeds,
Ensemble (Mittel der Survival-Kurven) und Niveau O/E@30 gesamt und je Testjahr.

Liest die Manifeste p35_sts_extreme_<name>_s<seed> und die Hazards derived_data/calib_cache_ext/
sts_extreme_<name>_s<seed>_s<seed>.npz (Referenzen: sts_extreme_ref_<name>_s<seed>.npz). Schreibt NEU:
  derived_data/calib_cache_ext/ens_<name>.npz   Ensemble-Hazards (Format wie calib_cache, fuer extreme_eval)
  metrics/sts_extreme<tag>/summary.md, summary.csv, Manifest p36_sts_extreme_table<tag>
Bricht ab, wenn das Ausgabeverzeichnis existiert; vorhandene Ensemble-Dateien werden nicht ueberschrieben.
Kalibriert = Lambda' = e^a Lambda^b, angepasst auf Val 2020-2022 (wie surv_eval).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.eval.metrics import evaluate
from wildfire.paper.compare import DTS
from wildfire.paper.extreme_eval import year_of
from wildfire.paper.surv_eval import _cum_hazard, calib_apply, calib_fit, compact, ensemble

SURV_KEYS = ("nll", "ibs", "c_td", "auc@30", "pr@30", "ece@30")
TAU = 30


def o_e(h: np.ndarray, e: np.ndarray, t: np.ndarray, m: np.ndarray | None = None) -> float:
    """Beobachtete / erwartete Ereignisse bis TAU; erwartet = Summe F(TAU)."""
    m = np.ones(e.size, bool) if m is None else m
    F = -np.expm1(-_cum_hazard(h[m])[:, TAU - 1])
    return float((e[m] & (t[m] <= TAU)).sum() / max(F.sum(), 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--names", nargs="+", required=True, help="Varianten, z. B. lh_ref lh_reg ...")
    ap.add_argument("--refs", nargs="*", default=[], help="Referenzen aus 167, z. B. abl_lh abl_cg_scratch")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    mdir = exp / "data_manifests"
    hz_dir = exp / "derived_data" / "calib_cache_ext"
    out = exp / "metrics" / f"sts_extreme{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --tag waehlen")

    sd = exp / "derived_data" / args.surv_dir
    ev = {s: dict(np.load(sd / f"{f}_meta.npz")) for s, f in (("val", "val"), ("test", "eval"))}
    yt = year_of(ev["test"]["L_abs"])
    years = sorted(set(int(y) for y in yt))

    runs = [(n, f"p35_sts_extreme_{n}_s{s}", hz_dir / f"sts_extreme_{n}_s{s}_s{s}.npz", f"s{s}")
            for n in args.names for s in args.seeds]
    runs += [(f"ref_{n}", f"p35_sts_extreme_ref_{n}", hz_dir / f"sts_extreme_ref_{n}_s{s}.npz", f"s{s}")
             for n in args.refs for s in args.seeds]
    rows, ens_info = [], {}
    by_name: dict[str, list] = {}
    for name, man, hz_path, sk in runs:
        mp = mdir / f"{man}.json"
        if not mp.exists() or not hz_path.exists():
            print(f"[tab] fehlt: {mp.name} oder {hz_path.name} - uebersprungen", flush=True)
            continue
        res = json.loads(mp.read_text())["results"][sk]
        z = np.load(hz_path)
        by_name.setdefault(name, []).append((sk, res, {k: z[k].astype(np.float64) for k in z.files}))

    def level(hz: dict, tg: str) -> dict:
        e = {s: ev[s][f"ev_{tg}"].astype(bool) for s in ev}
        t = {s: ev[s][f"t_{tg}"].astype(int) for s in ev}
        a, b = calib_fit(hz[f"val_{tg}"], e["val"], t["val"])
        d = {}
        for v, h in (("raw", hz[f"test_{tg}"]), ("curve", calib_apply(hz[f"test_{tg}"], a, b))):
            d[f"oe30_{v}"] = o_e(h, e["test"], t["test"])
            for y in years:
                d[f"oe30_{v}_{y}"] = o_e(h, e["test"], t["test"], yt == y)
        return d

    for name, items in by_name.items():
        for sk, res, hz in items:
            r = {"name": name, "member": sk}
            for tg in ("centre", "patch"):
                for v in ("raw", "curve"):
                    for k in SURV_KEYS:
                        r[f"{tg}_{v}_{k}"] = res["surv"][tg][v][k]
                for k, v in level(hz, tg).items():
                    r[f"{tg}_{k}"] = v
            for dt_ in DTS:
                p = res["paper"][str(dt_)]
                r[f"paper_centre_prauc_dt{dt_}"] = p["centre"]["pr_auc"]
                r[f"paper_patch_prauc_dt{dt_}"] = p["patch_head"]["pr_auc"]
                r[f"paper_patch_f1max_dt{dt_}"] = p["patch_head"]["f1_max"]
            rows.append(r)
            print(f"[tab] {name} {sk}: Ausschnitt kal. NLL {r['patch_curve_nll']:.4f}  O/E@30 roh "
                  f"{r['patch_oe30_raw']:.2f}  2023 {r.get('patch_oe30_raw_2023', float('nan')):.2f}", flush=True)
        if len(items) < 2:
            continue
        ens = {k: ensemble([hz[k] for _, _, hz in items]) for k in items[0][2]}
        ep = hz_dir / f"ens_{name}.npz"
        if not ep.exists():
            np.savez(ep, **{k: v.astype(np.float32) for k, v in ens.items()})
        r = {"name": name, "member": f"ens{len(items)}"}
        for tg in ("centre", "patch"):
            e = {s: ev[s][f"ev_{tg}"].astype(bool) for s in ev}
            t = {s: ev[s][f"t_{tg}"].astype(int) for s in ev}
            a, b = calib_fit(ens[f"val_{tg}"], e["val"], t["val"])
            for v, h in (("raw", ens[f"test_{tg}"]), ("curve", calib_apply(ens[f"test_{tg}"], a, b))):
                m = compact(evaluate(h, e["test"], t["test"]))
                for k in SURV_KEYS:
                    r[f"{tg}_{v}_{k}"] = m[k]
            for k, v in level(ens, tg).items():
                r[f"{tg}_{k}"] = v
        rows.append(r)
        ens_info[name] = {"path": str(ep), "members": [sk for sk, _, _ in items]}
        print(f"[tab] {name} Ensemble: Ausschnitt kal. NLL {r['patch_curve_nll']:.4f}  O/E@30 roh "
              f"{r['patch_oe30_raw']:.2f}", flush=True)

    out.mkdir(parents=True)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("name", "member"), k))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Markdown: Seed-Mittel (Spannweite) und Ensemble
    def agg(name, key):
        v = [r[key] for r in rows if r["name"] == name and r["member"].startswith("s") and key in r]
        return (np.mean(v), np.min(v), np.max(v)) if v else None

    def ensv(name, key):
        v = [r[key] for r in rows if r["name"] == name and r["member"].startswith("ens") and key in r]
        return v[0] if v else None

    fmt = lambda x, p=4: "–" if x is None else f"{x:.{p}f}"  # noqa: E731
    md = ["# sts_extreme: Zusatzeingaben und Monotonie", "",
          "Seed-Mittel (min–max) und Ensemble. Test 2023–2025, kalibriert = Val 2020–2022. "
          "O/E@30 > 1 = Unterschätzung.", ""]
    for tg, lab in (("patch", "Ausschnitt"), ("centre", "Mittelpixel")):
        md += [f"## {lab}", "",
               "| Variante | NLL kal. | C-td | AUC@30 | PR@30 | ECE@30 kal. | O/E@30 roh | O/E@30 kal. | "
               + " | ".join(f"O/E kal. {y}" for y in years) + " | Ens. NLL kal. | Ens. O/E kal. |",
               "|---" * (10 + len(years)) + "|"]
        for name in by_name:
            cell = lambda k, p=4: fmt(agg(name, k)[0] if agg(name, k) else None, p)  # noqa: E731
            nll = agg(name, f"{tg}_curve_nll")
            md.append(f"| {name} | {fmt(nll[0])} ({fmt(nll[1])}–{fmt(nll[2])}) | {cell(f'{tg}_curve_c_td', 3)} | "
                      f"{cell(f'{tg}_curve_auc@30', 3)} | {cell(f'{tg}_curve_pr@30', 3)} | {cell(f'{tg}_curve_ece@30', 4)} | "
                      f"{cell(f'{tg}_oe30_raw', 2)} | {cell(f'{tg}_oe30_curve', 2)} | "
                      + " | ".join(cell(f"{tg}_oe30_curve_{y}", 2) for y in years)
                      + f" | {fmt(ensv(name, f'{tg}_curve_nll'))} | {fmt(ensv(name, f'{tg}_oe30_curve'), 2)} |")
        md.append("")
    md += ["## Paper-Protokoll (PR-AUC, Seed-Mittel)", "",
           "| Variante | " + " | ".join(f"Mitte dt {d}" for d in DTS) + " | "
           + " | ".join(f"Ausschnitt dt {d}" for d in DTS) + " |", "|---" * (1 + 2 * len(DTS)) + "|"]
    for name in by_name:
        c = [fmt(agg(name, f"paper_centre_prauc_dt{d}")[0], 3) if agg(name, f"paper_centre_prauc_dt{d}") else "–" for d in DTS]
        p = [fmt(agg(name, f"paper_patch_prauc_dt{d}")[0], 3) if agg(name, f"paper_patch_prauc_dt{d}") else "–" for d in DTS]
        md.append(f"| {name} | " + " | ".join(c) + " | " + " | ".join(p) + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p36_sts_extreme_table{args.tag}", args.config, {
        "names": args.names, "refs": args.refs, "seeds": args.seeds, "years": years, "tau": TAU,
        "ensembles": ens_info, "out": str(out), "n_rows": len(rows)})


if __name__ == "__main__":
    main()

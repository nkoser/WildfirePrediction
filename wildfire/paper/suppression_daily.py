"""Bekaempfungsstatus je Tag (CWFIS Reported Fires) gegen die Kalibrierung aller Modelle aller vier Splits, CPU.

Ergaenzt suppression_splits: dort nur die Bekaempfungsklasse der NFDB (eine Angabe je Brand), hier der taegliche
Lagebericht der Feuerbehoerden (artifacts/cwfis_reportedfires, wildfire.data.cwfis_reported_fetch):
stage_of_control OC (ausser Kontrolle) / BH (wird gehalten) / UC (unter Kontrolle) / EX (geloescht) und
response_type FUL / MOD / MON je Statusintervall [record_start, record_end).

Je Test-Stichprobe (D, Ausschnittsmitte) und Stichtag t wird der hoechste Status unter den gemeldeten Braenden
(>= --min-ha, keine Rx-Braende) im Ausschnitt (+ dp) bestimmt: OC > BH > UC > EX (nur wenn in [t - 30, t]
geloescht) > kein Bericht. Stichtage: Vorhersagetag e = D - dt (was zur Vorhersage bekannt waere) und Zieltag
D + 1. Scores = Platt-kalibrierte Seed-Mittel aus metrics/suppression_splits/scores.npz (Job 5697), keine
Neubewertung. Kennzahl wie dort: mean(p) / mean(y) mit Bootstrap-Intervall, dazu mean(y) je Gruppe.

Ausgabe NEU: metrics/suppression_daily<tag>/{summary.md, groups.csv, coverage.csv, links.npz},
Manifest p55_suppression_daily<tag>.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.suppression_eval import boot_ratio, fmt
from wildfire.paper.suppression_splits import SPLITS

STAGES = ("OC", "BH", "UC", "EX")
NONE = "kein Bericht"
RANK = {s: i for i, s in enumerate(STAGES)}


def t_of(s: str) -> float:
    """ISO-Zeitstempel -> Tage (proleptisches Ordinal + Tagesbruchteil); leer -> nan."""
    if not s:
        return math.nan
    x = dt.datetime.fromisoformat(s[:19])
    return x.toordinal() + (x.hour * 3600 + x.minute * 60 + x.second) / 86400


def load_reports(src: Path, tr, H: int, W: int, min_ha: float) -> tuple[dict, list[dict]]:
    R, seen, cov = [], set(), []
    for f in sorted(src.glob("status_*.csv.gz")):
        n_raw = 0
        per_fire = Counter()
        with gzip.open(f, "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                n_raw += 1
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                per_fire[r["national_fire_id"]] += 1
                if r.get("fire_was_prescribed") == "1" or r["stage_of_control_status"] not in RANK:
                    continue
                try:
                    lat, lon, size = float(r["latitude"]), float(r["longitude"]), float(r["fire_size"] or "nan")
                except ValueError:
                    continue
                c, rr = math.floor((lon - tr[2]) / tr[0]), math.floor((lat - tr[5]) / tr[4])
                if not (0 <= rr < H and 0 <= c < W) or not size >= min_ha:
                    continue
                R.append({"row": rr, "col": c, "stage": r["stage_of_control_status"], "resp": r["response_type"],
                          "t0": t_of(r["record_start"]), "t1": t_of(r["record_end"]), "ts": t_of(r["status_date"]),
                          "agency": r["agency_code"], "fire": r["national_fire_id"]})
        k = np.array(list(per_fire.values())) if per_fire else np.zeros(1)
        cov.append({"status_year": f.name.split("_")[1].split(".")[0], "rows": n_raw, "fires": len(per_fire),
                    "records_per_fire_median": float(np.median(k)), "share_fires_multi": float((k > 1).mean())})
    by_cell = defaultdict(list)
    for i, r in enumerate(R):
        by_cell[(r["row"], r["col"])].append(i)
    return {"R": R, "by_cell": by_cell}, cov


def status_at(cand: list[int], R: list[dict], t: float) -> tuple[str, str]:
    best, resp = NONE, ""
    for i in cand:
        r = R[i]
        if not (r["t0"] <= t < r["t1"]) or (r["stage"] == "EX" and not t - 30 <= r["ts"] <= t):
            continue
        if best == NONE or RANK[r["stage"]] < RANK[best]:
            best, resp = r["stage"], r["resp"]
    return best, resp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reports", default="/data/Wildfire/artifacts/cwfis_reportedfires")
    ap.add_argument("--scores", default="/data/Wildfire/metrics/suppression_splits/scores.npz")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--min-ha", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"suppression_daily{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --tag waehlen")
    man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    tr, H, W = man["grid"]["transform"], man["grid"]["height"], man["grid"]["width"]
    idx_rep, cov = load_reports(Path(args.reports), tr, H, W, args.min_ha)
    R, by_cell = idx_rep["R"], idx_rep["by_cell"]
    print(f"[sdl] {len(R):,} Statusintervalle im Gitter (>= {args.min_ha} ha)", flush=True)
    Z = np.load(args.scores)
    rng = np.random.default_rng(0)
    rows, links, intro = [], {}, []
    half = 6 + args.dp
    for sp in args.splits:
        cache = SPLITS[sp][0]
        cd = exp / "derived_data" / "paper" / f"{cache}_L61"
        Xp = np.load(cd / "X.npy", mmap_mode="r")
        LC = Xp.shape[1]
        split, meta = np.load(cd / "split.npy"), np.load(cd / "meta.npy")
        te = np.where(split == "test")[0]
        yt = targets.labels(cd, "next_patch")[te]
        models = sorted({k.split("__")[1] for k in Z.files if k.startswith(f"{sp}__")})
        if not models or Z[f"{sp}__{models[0]}__dt1__platt"].size != te.size:
            print(f"[sdl] {sp}: keine passenden Scores, uebersprungen", flush=True)
            continue
        d0 = np.array([dt.date(int(D // 10000), int(D // 100 % 100), int(D % 100)).toordinal()
                       for D in meta[te, 0]], dtype=np.float64)
        cand = [[i for rr in range(r0 - half, r0 + half + 1) for cc in range(c0 - half, c0 + half + 1)
                 for i in by_cell.get((rr, cc), ())] for _, r0, c0 in meta[te]]
        tgt = [status_at(c, R, d + 1.5) for c, d in zip(cand, d0)]
        st_t = np.array([s for s, _ in tgt])
        links[f"{sp}__target_stage"] = st_t
        years = meta[te, 0] // 10000
        yr_cov = {int(y): int((st_t[years == y] != NONE).sum()) for y in np.unique(years)}
        intro.append(f"- {sp} = {cache}: Test {te.size:,}; Ausschnitte mit gemeldetem Brand am Zieltag je Testjahr "
                     f"{yr_cov}; Modelle {', '.join(models)}")
        for dt_ in DTS:
            e = LC - 1 - dt_
            fw = np.concatenate([(np.asarray(Xp[te[a:a + 1000], e - 29:e + 1, FIRE_CH]) > 0)
                                 .reshape(min(1000, te.size - a), -1).any(1) for a in range(0, te.size, 1000)])
            iss = [status_at(c, R, d - dt_ + 0.5) for c, d in zip(cand, d0)]
            st_i = np.array([s for s, _ in iss])
            rs_i = np.array([r for _, r in iss])
            links[f"{sp}__dt{dt_}__issue_stage"] = st_i
            links[f"{sp}__dt{dt_}__issue_resp"] = rs_i
            groups = [("alle", np.ones(te.size, bool))]
            groups += [(f"Vorhersagetag {s}", st_i == s) for s in STAGES]
            groups += [(f"Vorhersagetag {NONE}, Feuer im Fenster", (st_i == NONE) & fw),
                       (f"Vorhersagetag {NONE}, kein Feuer im Fenster", (st_i == NONE) & ~fw)]
            groups += [(f"Vorhersagetag OC, {r}", (st_i == "OC") & (rs_i == r)) for r in ("FUL", "MOD", "MON")]
            groups += [(f"Vorhersagetag BH/UC, {r}", np.isin(st_i, ("BH", "UC")) & (rs_i == r))
                       for r in ("FUL", "MOD", "MON")]
            groups += [(f"Zieltag {s}", st_t == s) for s in STAGES + (NONE,)]
            groups += [(f"OC -> {s}", (st_i == "OC") & (st_t == s)) for s in ("OC", "BH", "UC", "EX")]
            for name in models:
                p = Z[f"{sp}__{name}__dt{dt_}__platt"].astype(np.float64)
                for g, m in groups:
                    r = {"split": sp, "model": name, "dt": dt_, "group": g, "n": int(m.sum()), "pos": int(yt[m].sum()),
                         "mean_y": float(yt[m].mean()) if m.any() else np.nan,
                         "mean_p": float(p[m].mean()) if m.any() else np.nan}
                    r["ratio"] = r["mean_p"] / r["mean_y"] if m.any() and yt[m].sum() else np.nan
                    r["ratio_lo"], r["ratio_hi"] = boot_ratio(p[m], yt[m].astype(np.float64), rng) \
                        if m.any() else (np.nan, np.nan)
                    rows.append(r)
        print(f"[sdl] {sp} fertig", flush=True)

    # ---- Bericht
    L = ["# Taeglicher Bekaempfungsstatus (CWFIS Reported Fires) und Kalibrierung, alle Splits", "",
         "Quotient = mean(p) / mean(y) (p Platt-kalibriert, Seed-Mittel, aus suppression_splits), [95 %-Bootstrap]; "
         "> 1 = Ueberschaetzung. Status = hoechste Stufe der gemeldeten Braende (>= "
         f"{args.min_ha:g} ha) im Ausschnitt (+ {args.dp} Zellen): OC ausser Kontrolle, BH wird gehalten, "
         "UC unter Kontrolle, EX in den letzten 30 Tagen geloescht. Vorhersagetag = D - dt, Zieltag = D + 1.", "",
         "## Abdeckung der Lageberichte je Status-Jahr", "",
         "| Jahr | Zeilen | Braende | Berichte je Brand (Median) | Anteil Braende mit > 1 Bericht |", "|---|---|---|---|---|"]
    L += [f"| {c['status_year']} | {c['rows']:,} | {c['fires']:,} | {c['records_per_fire_median']:.0f} | "
          f"{c['share_fires_multi']:.2f} |" for c in cov]
    L += [""] + intro + [""]
    show = ["alle"] + [f"Vorhersagetag {s}" for s in STAGES] + \
           [f"Vorhersagetag {NONE}, Feuer im Fenster", f"Vorhersagetag {NONE}, kein Feuer im Fenster"] + \
           [f"Vorhersagetag OC, {r}" for r in ("FUL", "MON")] + [f"Vorhersagetag BH/UC, {r}" for r in ("FUL", "MON")] + \
           [f"Zieltag {s}" for s in STAGES] + [f"OC -> {s}" for s in ("OC", "BH", "UC", "EX")]
    for dt_ in DTS:
        L += [f"## dt {dt_}", ""]
        for sp in args.splits:
            models = sorted({r["model"] for r in rows if r["split"] == sp})
            if not models:
                continue
            L += [f"### {sp} = {SPLITS[sp][0]} ({SPLITS[sp][1]})", "",
                  "| Gruppe | n | Pos. | mean(y) | " + " | ".join(models) + " |", "|---|---|---|---|" + "---|" * len(models)]
            for g in show:
                cell = {r["model"]: r for r in rows if r["split"] == sp and r["dt"] == dt_ and r["group"] == g}
                if not cell or next(iter(cell.values()))["n"] == 0:
                    continue
                a = next(iter(cell.values()))
                L.append(f"| {g} | {a['n']:,} | {a['pos']:,} | {fmt(a['mean_y'])} | " + " | ".join(
                    f"{fmt(cell[mn]['ratio'])} [{fmt(cell[mn]['ratio_lo'])}, {fmt(cell[mn]['ratio_hi'])}]"
                    if mn in cell else "–" for mn in models) + " |")
            L += [""]
    out.mkdir(parents=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    for fn, data in (("groups.csv", rows), ("coverage.csv", cov)):
        with (out / fn).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    np.savez_compressed(out / "links.npz", **links)
    write_manifest(cfg, f"p55_suppression_daily{args.tag}", args.config,
                   {"args": vars(args), "intro": intro, "coverage": cov,
                    "outputs": {k: str(out / k) for k in ("summary.md", "groups.csv", "coverage.csv", "links.npz")}})
    print(f"[sdl] {out}", flush=True)


if __name__ == "__main__":
    main()

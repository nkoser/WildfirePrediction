"""Brandbekaempfung als Stoerfaktor der Infrastruktur-Variablen (NFDB RESPONSE / ATTK_DATE / OUT_DATE).

Frage: Wird dort, wo viele Menschen leben, staerker geloescht, sodass Feuer kleiner bleiben und kuerzer
brennen? Dann wirken Bevoelkerung und Infrastruktur zweifach (mehr Zuendungen, weniger Ausbreitung), und ein
Gruppen-Shapley nahe null hiesse nicht "irrelevant". Nur Auswertung, kein Training.

Teil A  NFDB allein, alle Braende 2000-2025 im Gitter und in der beobachteten Maske (observed_nfdb.npy):
        Abdeckung von RESPONSE / ATTK_DATE / OUT_DATE je Behoerde und Jahr; Groesse, Dauer
        (OUT_DATE - Zuenddatum) und Angriffsverzug (ATTK_DATE - Zuenddatum) je Bekaempfungsklasse und je
        Bevoelkerungsklasse am Zuendort (3x3-Mittel des Wuerfel-Statikkanals), dazu je Ursache, je
        FWI-Klasse am Zuendtag (Wuerfel, nur Saisontage) und je Behoerde.
Teil B  Modell gegen NFDB: die Scores der Gruppen-Dropout-Netze aus shapley_groups (218, gleiche 3000
        Test-Stichproben von cache_G, Mittel ueber Seeds), Teilmenge "alle Gruppen" und "alle ohne
        Infrastruktur". Kalibrierung mean(p) / mean(y) und PR-AUC je Bevoelkerungsklasse des Ausschnitts
        (Stichproben mit Feuer im Eingabefenster) und je Bekaempfungsklasse des groessten NFDB-Brandes
        (>= 1 ha), der im Ausschnitt (+ dp Pixel Rand) in [D - 30, D] gezuendet hat.
        Erwartung bei Bekaempfungseffekt: Ueberschaetzung (Quotient > 1) steigt mit der Bevoelkerung bzw.
        ist bei bekaempften Braenden groesser als bei nur beobachteten.

Bekaempfungsklassen: bekaempft = FUL, SUP, Actioned; begrenzt = MON, MOD, MNP, MDP; sonst unbekannt
(leer, PRO, NA, ...). Die rohen Codes stehen zusaetzlich im JSON.

Ausgabe NEU: metrics/suppression_eval<tag>/{summary.md, fires.csv, model_groups.csv}, Manifest
p55_suppression_eval<tag>; bricht ab, wenn das Verzeichnis existiert. Nur gelesen: alle Eingaben.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.nfdb_validation import load_nfdb
from wildfire.paper import targets
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.shapley_groups import DTS, GN

SUPP = {"FUL", "SUP", "ACTIONED"}
LIMITED = {"MON", "MOD", "MNP", "MDP"}
RCLS = ("bekaempft", "begrenzt", "unbekannt")
PCLS = ("0", "niedrig", "mittel", "hoch")
STAT_POP, STAT_BLD = 4, 2          # Statikkanaele im Wuerfel (d4_cube.json)
DYN_FWI = 24                       # Dynamikkanal FWI im Wuerfel
X_POP, X_FWI = 30, 24              # Kanaele im 55er-Cache
FULL = 2 ** len(GN) - 1
NO_INFRA = FULL ^ (1 << GN.index("infrastructure"))


def rclass(code: str) -> str:
    c = code.strip().upper()
    return "bekaempft" if c in SUPP else "begrenzt" if c in LIMITED else "unbekannt"


def day_of(s: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(s.strip()[:10])
    except ValueError:
        return None


def pop_bins(v: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """0 = keine Bevoelkerung, sonst Terzile der Werte > 0 (aus ref, Standard v)."""
    ref = v if ref is None else ref
    q = np.quantile(ref[ref > 0], [1 / 3, 2 / 3]) if (ref > 0).any() else np.array([np.inf, np.inf])
    out = np.where(v <= 0, 0, 1 + (v > q[0]) + (v > q[1]))
    return np.array(PCLS)[out]


# ---------------------------------------------------------------------------------------------- Teil A
def fires(exp: Path, nfdb: Path) -> list[dict]:
    man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    tr, H, W = man["grid"]["transform"], man["grid"]["height"], man["grid"]["width"]
    observed = np.load(exp / "derived_data" / "masks" / "observed_nfdb.npy").astype(bool)
    cube = json.loads((exp / "data_manifests" / "d4_cube.json").read_text())["per_year"]
    years = sorted(cube)
    stat = {y: np.load(cube[y]["stat_file"], mmap_mode="r") for y in years}
    out, drop = [], Counter()
    for r in load_nfdb(nfdb):
        try:
            d = dt.date(int(r["YEAR"]), int(r["MONTH"]), int(r["DAY"]))
            lat, lon = float(r["LATITUDE"]), float(r["LONGITUDE"])
        except (TypeError, ValueError):
            drop["kein Datum / keine Koordinate"] += 1
            continue
        if not 2000 <= d.year <= 2025:
            drop["ausserhalb 2000-2025"] += 1
            continue
        c, rr = math.floor((lon - tr[2]) / tr[0]), math.floor((lat - tr[5]) / tr[4])
        if not (0 <= rr < H and 0 <= c < W) or not observed[rr, c]:
            drop["ausserhalb Gitter / Maske"] += 1
            continue
        if "prescribed" in (r.get("FIRE_TYPE") or "").lower() or \
                (r.get("PRESCRIBED") or "").strip().upper() in ("Y", "YES", "TRUE", "1"):
            drop["kontrolliertes Abbrennen"] += 1
            continue
        try:
            size = float(r.get("SIZE_HA") or "nan")
        except ValueError:
            size = float("nan")
        o, a = day_of(r.get("OUT_DATE") or ""), day_of(r.get("ATTK_DATE") or "")
        dur = (o - d).days if o and 0 <= (o - d).days < 366 else np.nan
        atk = (a - d).days if a and 0 <= (a - d).days < 366 else np.nan
        y = str(min(max(d.year, int(years[0])), int(years[-1])))
        s = stat[y][max(rr - 1, 0):rr + 2, max(c - 1, 0):c + 2]
        out.append({"date": d, "row": rr, "col": c, "agency": (r.get("SRC_AGENCY") or "").strip(),
                    "cause": (r.get("CAUSE") or "U").strip()[:1].upper() or "U",
                    "response_raw": (r.get("RESPONSE") or "").strip(), "rcls": rclass(r.get("RESPONSE") or ""),
                    "size_ha": size, "dur": dur, "atk": atk,
                    "pop": float(s[..., STAT_POP].mean()), "bld": float(s[..., STAT_BLD].mean())})
    print(f"[supp] {len(out):,} NFDB-Braende im Gitter, verworfen {dict(drop)}", flush=True)
    # FWI am Zuendtag (nur Saisontage im Wuerfel); 255 = NoData/Saettigung -> nan
    by_year = defaultdict(list)
    for i, f in enumerate(out):
        by_year[str(f["date"].year)].append(i)
    for y, idx in by_year.items():
        fwi = np.full(len(idx), np.nan)
        if y in cube:
            local = {s: k for k, s in enumerate(cube[y]["days"])}
            dyn = np.memmap(cube[y]["dyn_file"], dtype=np.uint8, mode="r", shape=tuple(cube[y]["dyn_shape"]))
            for k, i in enumerate(idx):
                t = local.get(out[i]["date"].isoformat())
                if t is not None:
                    v = int(dyn[t, out[i]["row"], out[i]["col"], DYN_FWI])
                    fwi[k] = np.nan if v == 255 else v
        for k, i in enumerate(idx):
            out[i]["fwi"] = fwi[k]
    pb = pop_bins(np.array([f["pop"] for f in out]))
    for f, b in zip(out, pb):
        f["pcls"] = b
    v = np.array([f["fwi"] for f in out])
    q = np.nanquantile(v, [1 / 3, 2 / 3])
    for f in out:
        f["fcls"] = "nan" if np.isnan(f["fwi"]) else ("niedrig", "mittel", "hoch")[int(f["fwi"] > q[0]) + int(f["fwi"] > q[1])]
    return out


def describe(sub: list[dict]) -> dict:
    sz = np.array([f["size_ha"] for f in sub], float)
    du = np.array([f["dur"] for f in sub], float)
    at = np.array([f["atk"] for f in sub], float)
    known = [f for f in sub if f["rcls"] != "unbekannt"]
    med = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan
    return {"n": len(sub), "n_resp": len(known),
            "supp_share": (sum(f["rcls"] == "bekaempft" for f in known) / len(known)) if known else np.nan,
            "size_med": med(sz), "gt1": float(np.nanmean(sz >= 1)) if sub else np.nan,
            "gt100": float(np.nanmean(sz >= 100)) if sub else np.nan,
            "n_dur": int(np.isfinite(du).sum()), "dur_med": med(du),
            "dur_mean": float(np.nanmean(du)) if np.isfinite(du).any() else np.nan,
            "n_atk": int(np.isfinite(at).sum()), "atk_med": med(at)}


def fmt(x, p=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}"


def table(title: str, groups: list[tuple[str, list[dict]]]) -> list[str]:
    L = [f"### {title}", "", "| Gruppe | n | n mit RESPONSE | Anteil bekaempft | Median ha | >= 1 ha | >= 100 ha "
         "| n Dauer | Dauer Median / Mittel (d) | n Angriff | Angriffsverzug Median (d) |", "|---|" + "---|" * 10]
    for name, sub in groups:
        s = describe(sub)
        L.append(f"| {name} | {s['n']:,} | {s['n_resp']:,} | {fmt(s['supp_share'], 3)} | {fmt(s['size_med'])} | "
                 f"{fmt(s['gt1'], 3)} | {fmt(s['gt100'], 3)} | {s['n_dur']:,} | {fmt(s['dur_med'], 1)} / "
                 f"{fmt(s['dur_mean'], 1)} | {s['n_atk']:,} | {fmt(s['atk_med'], 1)} |")
    return L + [""]


def part_a(F: list[dict], test_years: set[int]) -> list[str]:
    L = ["## Teil A - NFDB allein", ""]
    # Abdeckung
    L += ["### Abdeckung je Behoerde", "", "| Behoerde | n | RESPONSE | ATTK_DATE | OUT_DATE | davon Testjahre n | "
          "RESPONSE in Testjahren |", "|---|---|---|---|---|---|---|"]
    for a, c in Counter(f["agency"] for f in F).most_common():
        s = [f for f in F if f["agency"] == a]
        t = [f for f in s if f["date"].year in test_years]
        sh = lambda xs, k: sum(1 for f in xs if (f["response_raw"] if k == "r" else np.isfinite(f[k]))) / max(len(xs), 1)
        L.append(f"| {a} | {c:,} | {sh(s, 'r'):.3f} | {sh(s, 'atk'):.3f} | {sh(s, 'dur'):.3f} | {len(t):,} | "
                 f"{sh(t, 'r'):.3f} |")
    L += [""]
    L += ["### Abdeckung je Jahr", "", "| Jahr | n | RESPONSE | OUT_DATE |", "|---|---|---|---|"]
    for y in sorted({f["date"].year for f in F}):
        s = [f for f in F if f["date"].year == y]
        L.append(f"| {y} | {len(s):,} | {sum(bool(f['response_raw']) for f in s) / len(s):.3f} | "
                 f"{sum(np.isfinite(f['dur']) for f in s) / len(s):.3f} |")
    L += ["", "Rohe RESPONSE-Codes: " + ", ".join(f"{k or '<leer>'} {v:,}" for k, v in
                                              Counter(f["response_raw"] for f in F).most_common()), ""]
    L += table("Je Bekaempfungsklasse", [(c, [f for f in F if f["rcls"] == c]) for c in RCLS])
    L += table("Je Bevoelkerungsklasse am Zuendort (0 / Terzile > 0)", [(c, [f for f in F if f["pcls"] == c]) for c in PCLS])
    for c in RCLS[:2]:
        L += table(f"Je Bevoelkerungsklasse, nur {c}", [(p, [f for f in F if f["pcls"] == p and f["rcls"] == c]) for p in PCLS])
    for cause, name in (("H", "Mensch"), ("N", "Blitz")):
        L += table(f"Je Bevoelkerungsklasse, Ursache {name}", [(p, [f for f in F if f["pcls"] == p and f["cause"] == cause])
                                                             for p in PCLS])
    for fc in ("niedrig", "mittel", "hoch"):
        L += table(f"Je Bevoelkerungsklasse, FWI am Zuendtag {fc} (Terzil)",
                   [(p, [f for f in F if f["pcls"] == p and f["fcls"] == fc]) for p in PCLS])
    for a, c in Counter(f["agency"] for f in F if f["rcls"] != "unbekannt").most_common():
        if c < 300:
            break
        L += table(f"Je Bevoelkerungsklasse, Behoerde {a}", [(p, [f for f in F if f["pcls"] == p and f["agency"] == a])
                                                           for p in PCLS])
    return L


# ---------------------------------------------------------------------------------------------- Teil B
def boot_ratio(p: np.ndarray, y: np.ndarray, rng, B: int = 500) -> tuple[float, float]:
    if y.sum() == 0:
        return np.nan, np.nan
    i = rng.integers(0, p.size, (B, p.size))
    r = p[i].mean(1) / np.maximum(y[i].mean(1), 1e-9)
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def part_b(exp: Path, src: Path, F: list[dict], dp: int, rows_out: list[dict]) -> list[str]:
    cd = exp / "derived_data" / "paper" / "cache_G_L61"
    Xp = np.load(cd / "X.npy", mmap_mode="r")
    meta = np.load(cd / "meta.npy")
    LC = Xp.shape[1]
    y_all = targets.labels(cd, "next_patch")
    # NFDB-Braende >= 1 ha als Liste je Tag
    big = [f for f in F if f["size_ha"] >= 1]
    by_day = defaultdict(list)
    for f in big:
        by_day[f["date"].toordinal()].append(f)
    rng = np.random.default_rng(0)
    L = ["## Teil B - Modell (Gruppen-Dropout-Netze, cache_G Test, Ziel D+1) gegen Bevoelkerung und NFDB", "",
         "Quotient = mean(p) / mean(y) mit 95 %-Bootstrap-Intervall; > 1 = Ueberschaetzung. 'ohne Infra' = gleiches "
         "Netz, Infrastruktur-Gruppe ausgeblendet. PR-AUC nur bei >= 5 Positiven und Negativen.", ""]
    for name in ("cg_gdrop", "lh_gdrop"):
        files = sorted(glob.glob(str(src / f"scores_{name}_s*.npz")))
        if not files:
            L += [f"{name}: keine Scores in {src}", ""]
            continue
        zs = [np.load(f) for f in files]
        te = zs[0]["rows"]
        assert all((z["rows"] == te).all() for z in zs), "Stichproben der Seeds verschieden"
        P = {k: np.mean([z["patch_head"][k] for z in zs], 0) for k in (FULL, NO_INFRA)}    # (len(DTS), n)
        y = y_all[te]
        pop = np.asarray(Xp[te, LC - 1, X_POP], np.float32).reshape(te.size, -1).mean(1)
        fwi_raw = np.asarray(Xp[te, LC - 1, X_FWI], np.float32).reshape(te.size, -1)
        fwi = np.where(fwi_raw == 255, np.nan, fwi_raw)
        fwi = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in fwi])
        # groesster NFDB-Brand im Ausschnitt (+ dp) mit Zuendung in [D - 30, D]
        rc = []
        for D, r0, c0 in meta[te]:
            d = dt.date(int(D // 10000), int(D // 100 % 100), int(D % 100)).toordinal()
            cand = [f for k in range(d - 30, d + 1) for f in by_day.get(k, ())
                    if abs(f["row"] - r0) <= 6 + dp and abs(f["col"] - c0) <= 6 + dp]
            rc.append(max(cand, key=lambda f: f["size_ha"])["rcls"] if cand else "kein NFDB-Brand")
        rc = np.array(rc)
        print(f"[supp] {name}: {len(files)} Seeds, {te.size} Stichproben, NFDB-Verknuepfung {Counter(rc.tolist())}",
              flush=True)
        for j, dt_ in enumerate(DTS):
            e = LC - 1 - dt_
            fw = (np.asarray(Xp[te, e - 29:e + 1, FIRE_CH]) > 0.5).reshape(te.size, -1).any(1)
            pb = pop_bins(pop, pop[fw])
            fh = np.where(np.isnan(fwi), "nan", np.where(fwi > np.nanmedian(fwi[fw]), "hoch", "niedrig"))
            groups = [("Feuer im Fenster, alle", fw)]
            groups += [(f"Feuer im Fenster, Bevoelkerung {b}", fw & (pb == b)) for b in PCLS]
            groups += [(f"Feuer im Fenster, FWI {h}, Bevoelkerung {b}", fw & (fh == h) & (pb == b))
                       for h in ("niedrig", "hoch") for b in PCLS]
            groups += [(f"NFDB {c}", rc == c) for c in RCLS + ("kein NFDB-Brand",)]
            groups += [(f"NFDB {c}, Bevoelkerung {b}", (rc == c) & (pb == b)) for c in RCLS[:2] for b in PCLS]
            for g, m in groups:
                row = {"model": name, "dt": dt_, "group": g, "n": int(m.sum()), "pos": int(y[m].sum())}
                for k, tag in ((FULL, "alle"), (NO_INFRA, "ohne_infra")):
                    p = P[k][j][m].astype(np.float64)
                    row[f"mean_p_{tag}"] = float(p.mean()) if m.any() else np.nan
                    row[f"ratio_{tag}"] = float(p.mean() / y[m].mean()) if m.any() and y[m].sum() else np.nan
                    row[f"ratio_{tag}_lo"], row[f"ratio_{tag}_hi"] = boot_ratio(p, y[m].astype(np.float64), rng) \
                        if m.any() else (np.nan, np.nan)
                    ok = 5 <= y[m].sum() <= m.sum() - 5
                    row[f"prauc_{tag}"] = float(average_precision_score(y[m], p)) if ok else np.nan
                row["mean_y"] = float(y[m].mean()) if m.any() else np.nan
                rows_out.append(row)
        for dt_ in (1, 15):
            L += [f"### {name}, dt {dt_}", "", "| Gruppe | n | Positive | mean y | mean p | Quotient alle [95 %] | "
                  "Quotient ohne Infra | PR-AUC alle | PR-AUC ohne Infra |", "|---|---|---|---|---|---|---|---|---|"]
            for r in rows_out:
                if r["model"] == name and r["dt"] == dt_ and r["n"] >= 20:
                    L.append(f"| {r['group']} | {r['n']:,} | {r['pos']:,} | {fmt(r['mean_y'], 3)} | "
                             f"{fmt(r['mean_p_alle'], 3)} | {fmt(r['ratio_alle'])} [{fmt(r['ratio_alle_lo'])}, "
                             f"{fmt(r['ratio_alle_hi'])}] | {fmt(r['ratio_ohne_infra'])} | {fmt(r['prauc_alle'], 3)} | "
                             f"{fmt(r['prauc_ohne_infra'], 3)} |")
            L += ["", "(Gruppen mit n < 20 nur in model_groups.csv.)", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nfdb", default="/data/Wildfire/artifacts/NFDB_point_txt.zip")
    ap.add_argument("--src-tag", default="", help="Scores aus metrics/shapley_groups<src-tag>")
    ap.add_argument("--dp", type=int, default=2, help="Teil B: Rand um den 13x13-Ausschnitt in Pixeln")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"suppression_eval{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - wird nicht ueberschrieben")

    F = fires(exp, Path(args.nfdb))
    cd = exp / "derived_data" / "paper" / "cache_G_L61"
    meta, split = np.load(cd / "meta.npy"), np.load(cd / "split.npy")
    test_years = {int(m // 10000) for m in meta[split == "test", 0]}
    L = ["# Brandbekaempfung als Stoerfaktor der Infrastruktur-Variablen (NFDB)", "",
         f"NFDB {args.nfdb}, {len(F):,} Wildfeuer 2000-2025 im Gitter und in der beobachteten Maske. "
         f"Bevoelkerung = WorldPop-Statikkanal des Wuerfels, 3x3-Mittel am Zuendort (Teil A) bzw. Mittel ueber den "
         f"13x13-Ausschnitt (Teil B). Testjahre cache_G: {sorted(test_years)}.", ""]
    L += part_a(F, test_years)
    rows: list[dict] = []
    L += part_b(exp, exp / "metrics" / f"shapley_groups{args.src_tag}", F, args.dp, rows)

    out.mkdir(parents=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    keys = ["date", "row", "col", "agency", "cause", "response_raw", "rcls", "size_ha", "dur", "atk", "pop", "bld",
            "fwi", "pcls", "fcls"]
    with (out / "fires.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({k: f[k] for k in keys} for f in F)
    if rows:
        with (out / "model_groups.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write_manifest(cfg, f"p55_suppression_eval{args.tag}", args.config,
                   {"args": vars(args), "n_fires": len(F), "test_years": sorted(test_years),
                    "response_raw": dict(Counter(f["response_raw"] for f in F)),
                    "response_class": dict(Counter(f["rcls"] for f in F)),
                    "outputs": {k: str(out / k) for k in ("summary.md", "fires.csv", "model_groups.csv")}})
    print(f"[supp] {out}", flush=True)


if __name__ == "__main__":
    main()

"""Split-Audit: Leakage und Black-Swan-Verteilung der cache_G-Splits, dazu zwei leakage-freie Kandidaten. Nur CPU.

Geprueft werden die JSON-Splits unter /firedata/MPCFire/<cache>/samples_variant_G.json (--caches) und zwei hier
gebaute Kandidaten auf denselben 26.112 Stichproben:

  year_strat   ganze Jahre je Split. Testjahre und Valjahre per Zufallssuche so gewaehlt, dass Stichproben,
               Positive, Black-Swan-Ereignisse und deren Feuer-Pixeltage moeglichst 70/10/20 verteilt sind und
               jeder Split fruehe, mittlere und spaete Jahre enthaelt. Kein Pixel-Tag in zwei Splits (bis auf
               den Jahreswechsel, der nachgeprueft wird).
  block_strat  Einheiten = Jahr x raeumlicher Grossblock (--unit Pixel). Einheiten, die dasselbe Black-Swan-
               Ereignis beruehren, werden verschmolzen (ein Ereignis liegt nie in zwei Splits). Zuweisung gierig
               mit Zufallsneustarts, Ziel: 70/10/20 je Jahr UND fuer Black-Swan-Ereignisse. Danach Purge: jede
               Train-/Val-Stichprobe, deren Raum-Zeit-Fenster eine Stichprobe eines spaeteren Splits beruehrt,
               faellt weg (split = "drop").

Leakage-Masse (je Test- bzw. Val-Stichprobe: gibt es eine Stichprobe eines anderen Splits mit ...):
  overlap  |dr|, |dc| <= 12 und |dd| <= 60: die 13 x 13 x 61-Eingabefenster teilen Pixel-Tage.
  horizon  |dr|, |dc| <= 12 und |dd| <= 90: Fenster plus 30-Tage-Horizont der Survival-Ziele beruehren sich.
  fire     beide positiv, |dr|, |dc| <= 25 und |dd| <= 14: sehr wahrscheinlich dasselbe Feuer.
Black Swans: find_events aus black_swan.py, aber Rekord leave-one-year-out ueber ALLE Jahre 2000-2025 (sonst
haengt die Definition selbst vom zeitlichen Split ab). Stichprobe gehoert zu einem Ereignis, wenn Jahr, 16er-Block
und Datum in [Beginn - 30 d, Ende] passen. "gross" = Ereignis mit >= --major Feuer-Pixeltagen.

Ausgabe NEU: metrics/split_audit<tag>/summary.md, per_year.csv, events.csv;
             derived_data/splits_v2<tag>/<kandidat>/split.npy (Zeilen wie cache_G_L61) und samples_variant_G.json;
             Manifest p41_split_audit<tag>.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import wildfire.paper.black_swan as bs
from wildfire.audit.common import assert_readonly, load_config, write_manifest

SPLITS = ("train", "val", "test")
TARGET = {"train": 0.70, "val": 0.10, "test": 0.20}
EPOCH = dt.date(2000, 1, 1)
YEARS = list(range(2000, 2026))


def load_json(path: Path) -> dict:
    doc = json.loads(path.read_text())
    rec = doc["positive_samples"] + doc["negative_samples"]
    return {
        "key": np.array([(int(r["date"].replace("-", "")), int(r["pixel_row"]), int(r["pixel_col"])) for r in rec]),
        "day": np.array([(dt.date.fromisoformat(r["date"]) - EPOCH).days for r in rec]),
        "year": np.array([int(r["year"]) for r in rec]),
        "row": np.array([int(r["pixel_row"]) for r in rec]),
        "col": np.array([int(r["pixel_col"]) for r in rec]),
        "y": np.array([1 if r["sample_type"] == "positive" else 0 for r in rec]),
        "split": np.array([r["split"] for r in rec]),
        "fire_id": np.array([str(r.get("fire_id")) if r.get("fire_id") is not None else "" for r in rec]),
        "records": rec,
    }


def order_to(base_key: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Index je Zeile von base_key in key; bricht ab, wenn die Mengen nicht identisch sind."""
    pos = defaultdict(list)
    for i, k in enumerate(map(tuple, key)):
        pos[k].append(i)
    idx = []
    for k in map(tuple, base_key):
        if not pos.get(k):
            raise SystemExit(f"ABBRUCH: Stichprobe {k} fehlt")
        idx.append(pos[k].pop(0))
    return np.array(idx)


def near(a: np.ndarray, b: np.ndarray, dpx: float, dday: float) -> np.ndarray:
    """Fuer jeden Punkt aus a: gibt es in b einen mit |dr|, |dc| <= dpx und |dd| <= dday?"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    s = dpx / dday
    pa = np.c_[a[:, 0], a[:, 1], a[:, 2] * s]
    pb = np.c_[b[:, 0], b[:, 1], b[:, 2] * s]
    d, _ = cKDTree(pb).query(pa, k=1, p=np.inf, distance_upper_bound=dpx + 1e-6)
    return np.isfinite(d)


def leakage(D: dict, split: np.ndarray) -> dict:
    P = np.c_[D["row"], D["col"], D["day"]].astype(float)
    out = {}
    for s, others in (("test", ("train", "val")), ("val", ("train",))):
        m, o = split == s, np.isin(split, others)
        pos_m, pos_o = m & (D["y"] == 1), o & (D["y"] == 1)
        out[s] = {
            "n": int(m.sum()),
            "overlap": float(near(P[m], P[o], 12, 60).mean()) if m.any() else float("nan"),
            "horizon": float(near(P[m], P[o], 12, 90).mean()) if m.any() else float("nan"),
            "fire": float(near(P[pos_m], P[pos_o], 25, 14).mean()) if pos_m.any() else float("nan"),
        }
    fid = D["fire_id"]
    if (fid != "").any():
        sets = defaultdict(set)
        for f, s in zip(fid, split):
            if f and s in SPLITS:
                sets[f].add(s)
        out["fire_id_multi_split"] = int(sum(len(v) > 1 for v in sets.values()))
        out["fire_id_n"] = len(sets)
    return out


def sample_events(D: dict, events: list, blk: int, pre: int) -> np.ndarray:
    """Ereignis-id je Stichprobe (-1 = keins); bei mehreren das groesste (Liste ist nach Groesse sortiert)."""
    ev = np.full(len(D["y"]), -1)
    br, bc = D["row"] // blk, D["col"] // blk
    for e in reversed(events):
        blocks = set(map(tuple, e["blocks"]))
        m = (D["year"] == e["year"]) & (D["day"] >= e["start_day"] - pre) & (D["day"] <= e["end_day"])
        idx = np.nonzero(m)[0]
        hit = [i for i in idx if (br[i], bc[i]) in blocks]
        ev[hit] = e["id"]
    return ev


def event_split_table(ev: np.ndarray, split: np.ndarray, events: list, major: int) -> dict:
    by = defaultdict(Counter)
    for e, s in zip(ev, split):
        if e >= 0 and s in SPLITS:
            by[e][s] += 1
    fpd = {e["id"]: e["fire_pixel_days"] for e in events}
    res = {"all": Counter(), "major": Counter(), "fpd": Counter(), "multi": 0, "multi_major": 0, "n": 0, "n_major": 0}
    for e, c in by.items():
        owner = c.most_common(1)[0][0]
        big = fpd[e] >= major
        res["n"] += 1
        res["n_major"] += big
        res["all"][owner] += 1
        res["fpd"][owner] += fpd[e]
        if big:
            res["major"][owner] += 1
        if len(c) > 1:
            res["multi"] += 1
            res["multi_major"] += big
    return res


def score(shares: dict) -> float:
    """Summe der quadrierten Abweichungen von 70/10/20 ueber alle Kriterien."""
    return sum((shares[k][s] - TARGET[s]) ** 2 for k in shares for s in SPLITS)


# ------------------------------------------------------------------ Kandidat A: ganze Jahre
def build_year_strat(D: dict, ev: np.ndarray, events: list, major: int, n_test: int, n_val: int,
                     iters: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    ys = np.array(YEARS)
    big = {e["id"] for e in events if e["fire_pixel_days"] >= major}
    w = {
        "samples": np.array([(D["year"] == y).sum() for y in ys], float),
        "pos": np.array([((D["year"] == y) & (D["y"] == 1)).sum() for y in ys], float),
        "events": np.array([sum(e["year"] == y and e["id"] in big for e in events) for y in ys], float),
        "event_fpd": np.array([sum(e["fire_pixel_days"] for e in events if e["year"] == y and e["id"] in big)
                               for y in ys], float),
    }
    tot = {k: v.sum() for k, v in w.items()}
    period = np.digitize(ys, [2009, 2018])                                   # 0: 2000-08, 1: 2009-17, 2: 2018-25
    best, best_sc = None, np.inf
    for _ in range(iters):
        perm = rng.permutation(len(ys))
        te, va = perm[:n_test], perm[n_test:n_test + n_val]
        if len(set(period[te])) < 3 or len(set(period[va])) < min(3, n_val):
            continue
        lab = np.zeros(len(ys), int)
        lab[va], lab[te] = 1, 2
        shares = {k: {s: v[lab == j].sum() / tot[k] for j, s in enumerate(SPLITS)} for k, v in w.items()}
        sc = score(shares)
        if sc < best_sc:
            best, best_sc = lab.copy(), sc
    year_split = {int(y): SPLITS[j] for y, j in zip(ys, best)}
    split = np.array([year_split[int(y)] for y in D["year"]], dtype="<U5")
    return split, {"years": {s: [y for y, v in year_split.items() if v == s] for s in SPLITS}, "score": best_sc}


# ------------------------------------------------------------------ Kandidat B: Jahr x Grossblock
def build_block_strat(D: dict, ev: np.ndarray, events: list, major: int, unit: int, restarts: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    u_of = np.array([f"{y}_{r // unit}_{c // unit}" for y, r, c in zip(D["year"], D["row"], D["col"])])
    names = sorted(set(u_of))
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in set(ev[ev >= 0].tolist()):                                     # ein Ereignis = eine Gruppe
        us = sorted(set(u_of[ev == e]))
        for a in us[1:]:
            parent[find(a)] = find(us[0])
    g_of = np.array([find(u) for u in u_of])
    groups = sorted(set(g_of))
    gi = {g: i for i, g in enumerate(groups)}
    G = np.array([gi[g] for g in g_of])
    big = {e["id"] for e in events if e["fire_pixel_days"] >= major}
    fpd = {e["id"]: e["fire_pixel_days"] for e in events}

    # Gewichte je Gruppe: Stichproben je Jahr, Positive je Jahr, grosse Ereignisse, deren Feuer-Pixeltage
    ny = len(YEARS)
    W = np.zeros((len(groups), 2 * ny + 2))
    np.add.at(W, (G, D["year"] - 2000), 1.0)
    np.add.at(W, (G[D["y"] == 1], ny + D["year"][D["y"] == 1] - 2000), 1.0)
    for e in set(ev[ev >= 0].tolist()):
        if e in big:
            g = G[ev == e][0]
            W[g, 2 * ny] += 1
            W[g, 2 * ny + 1] += fpd[e]
    tot = W.sum(0)
    keep = tot > 0
    W, tot = W[:, keep], tot[keep]
    tgt = np.array([TARGET[s] for s in SPLITS])
    imp = np.ones(W.shape[1])
    imp[-2:] = 4.0                                                          # Black Swans staerker gewichten

    def assign(order):
        acc = np.zeros((3, W.shape[1]))
        lab = np.zeros(len(groups), int)
        for g in order:
            costs = []
            for j in range(3):
                acc[j] += W[g]
                sh = acc / tot
                costs.append((imp * (sh - tgt[:, None] * (acc.sum(0) / tot)) ** 2).sum())
                acc[j] -= W[g]
            j = int(np.argmin(costs))
            acc[j] += W[g]
            lab[g] = j
        sh = acc / tot
        return lab, float((imp * (sh - tgt[:, None]) ** 2).sum())

    size = W.sum(1)
    best, best_sc = None, np.inf
    for k in range(restarts):
        noise = rng.uniform(0.7, 1.3, len(groups)) if k else np.ones(len(groups))
        lab, sc = assign(np.argsort(-size * noise))
        if sc < best_sc:
            best, best_sc = lab, sc
    split = np.array([SPLITS[j] for j in best[G]], dtype="<U5")

    # Purge: Train/Val-Stichproben, deren Fenster + Horizont eine Stichprobe eines spaeteren Splits beruehrt
    P = np.c_[D["row"], D["col"], D["day"]].astype(float)
    n_drop = {}
    for s, later in (("val", ("test",)), ("train", ("val", "test"))):
        m = split == s
        hit = near(P[m], P[np.isin(split, later)], 12, 90)
        idx = np.nonzero(m)[0][hit]
        split[idx] = "drop"
        n_drop[s] = int(hit.sum())
    return split, {"unit_px": unit, "n_groups": len(groups), "n_units": len(names), "score": best_sc,
                   "dropped": n_drop}


def describe(name: str, D: dict, split: np.ndarray, ev: np.ndarray, events: list, major: int) -> dict:
    tab = {s: {"n": int((split == s).sum()), "pos": int(((split == s) & (D["y"] == 1)).sum())} for s in SPLITS}
    tab["drop"] = {"n": int((split == "drop").sum()), "pos": int(((split == "drop") & (D["y"] == 1)).sum())}
    years = {y: {s: int(((D["year"] == y) & (split == s)).sum()) for s in SPLITS} for y in YEARS}
    return {"name": name, "table": tab, "years": years, "leak": leakage(D, split),
            "events": event_split_table(ev, split, events, major), "_split": split}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--caches", nargs="*", default=["cache_G", "cache_G_stratified", "cache_G_extreme_fire"])
    ap.add_argument("--blk", type=int, default=16)
    ap.add_argument("--win", type=int, default=14)
    ap.add_argument("--nmin", type=int, default=30)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--pre", type=int, default=30)
    ap.add_argument("--major", type=int, default=300, help="Feuer-Pixeltage ab denen ein Ereignis 'gross' ist")
    ap.add_argument("--n-test-years", type=int, default=5)
    ap.add_argument("--n-val-years", type=int, default=3)
    ap.add_argument("--year-iters", type=int, default=200000)
    ap.add_argument("--unit", type=int, default=64, help="Grossblock in Pixeln (ca. 5 km je Pixel N-S)")
    ap.add_argument("--restarts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"split_audit{args.tag}"
    sp_dir = exp / "derived_data" / f"splits_v2{args.tag}"
    for p in (out, sp_dir):
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderen --tag waehlen")
    rng = np.random.default_rng(args.seed)

    # Black Swans unabhaengig vom Split: Rekord leave-one-year-out ueber alle Jahre
    bs.TRAIN = (YEARS[0], YEARS[-1])
    events, annual, _ = bs.find_events(args, exp)
    print(f"[audit] {len(events)} Black-Swan-Ereignisse, davon {sum(e['fire_pixel_days'] >= args.major for e in events)}"
          f" gross (>= {args.major} Feuer-Pixeltage)", flush=True)

    root = Path(cfg["dataset_root"])
    base = load_json(root / "cache_G" / "samples_variant_G.json")
    meta = np.load(exp / "derived_data" / "paper" / "cache_G_L61" / "meta.npy")
    if not np.array_equal(meta, base["key"]):
        raise SystemExit("ABBRUCH: cache_G JSON und cache_G_L61/meta.npy haben nicht dieselbe Reihenfolge")
    ev = sample_events(base, events, args.blk, args.pre)
    print(f"[audit] {int((ev >= 0).sum())} Stichproben liegen in einem Black-Swan-Ereignis", flush=True)

    results = []
    for c in args.caches:
        D = load_json(root / c / "samples_variant_G.json")
        idx = order_to(base["key"], D["key"])
        if not np.array_equal(D["y"][idx], base["y"]):
            raise SystemExit(f"ABBRUCH: Labels von {c} weichen von cache_G ab")
        Dc = dict(base, fire_id=D["fire_id"][idx])
        results.append(describe(c, Dc, D["split"][idx], ev, events, args.major))
        print(f"[audit] {c}: {results[-1]['leak']}", flush=True)

    cand = {}
    split_a, info_a = build_year_strat(base, ev, events, args.major, args.n_test_years, args.n_val_years,
                                       args.year_iters, rng)
    cand["year_strat"] = (split_a, info_a)
    split_b, info_b = build_block_strat(base, ev, events, args.major, args.unit, args.restarts, rng)
    cand["block_strat"] = (split_b, info_b)
    for n, (s, info) in cand.items():
        results.append(describe(n, base, s, ev, events, args.major) | {"info": info})
        print(f"[audit] {n}: {info} {results[-1]['leak']}", flush=True)

    # ---------------------------------------------------------------- schreiben
    out.mkdir(parents=True)
    for n, (s, info) in cand.items():
        d = sp_dir / n
        d.mkdir(parents=True)
        np.save(d / "split.npy", s)
        doc = {"positive_samples": [], "negative_samples": [], "info": info}
        for r, sv in zip(base["records"], s):
            if sv == "drop":
                continue
            doc["positive_samples" if r["sample_type"] == "positive" else "negative_samples"].append(r | {"split": str(sv)})
        (d / "samples_variant_G.json").write_text(json.dumps(doc))
    with open(out / "events.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "year", "start", "end", "lat", "lon", "fire_pixel_days", "ratio_max", "n_samples"]
                   + [r["name"] for r in results])
        for e in events:
            m = ev == e["id"]
            if not m.any():
                continue
            row = [e["id"], e["year"], e["start"], e["end"], f"{e['lat']:.2f}", f"{e['lon']:.2f}",
                   e["fire_pixel_days"], f"{e['ratio_max']:.2f}", int(m.sum())]
            for r in results:
                c = Counter(r["_split"][m].tolist())
                row.append(" ".join(f"{k}:{c[k]}" for k in SPLITS + ("drop",) if c[k]))
            w.writerow(row)
    with open(out / "per_year.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split_scheme", "year"] + list(SPLITS))
        for r in results:
            for y, v in r["years"].items():
                w.writerow([r["name"], y] + [v[s] for s in SPLITS])

    pct = lambda x: "–" if x != x else f"{100 * x:.1f} %"  # noqa: E731
    md = ["# Split-Audit cache_G", "",
          f"26.112 Stichproben (13 x 13 px, Fenster D-60..D). Black Swans: {len(events)} Ereignisse (Rekord "
          f"leave-one-year-out ueber 2000-2025), davon {sum(e['fire_pixel_days'] >= args.major for e in events)} gross "
          f"(>= {args.major} Feuer-Pixeltage); {int((ev >= 0).sum())} Stichproben liegen in einem Ereignis "
          f"(Block {args.blk} px, [Beginn - {args.pre} d, Ende]).", "",
          "## Groesse und Leakage", "",
          "Leakage = Anteil der Test- (Val-) Stichproben mit einer Stichprobe aus einem anderen Split in der Naehe: "
          "overlap = Eingabefenster teilen Pixel-Tage (|dr|,|dc| <= 12, |dd| <= 60); horizon = mit 30-d-Horizont "
          "(|dd| <= 90); fire = Positiv mit Positiv im Abstand <= 25 px und <= 14 d (dasselbe Feuer).", "",
          "| Split | train n/pos | val n/pos | test n/pos | drop | Test overlap | Test horizon | Test fire | "
          "Val overlap | Val fire | fire_id in >1 Split |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        t, L = r["table"], r["leak"]
        fid = f"{L['fire_id_multi_split']} / {L['fire_id_n']}" if "fire_id_n" in L else "–"
        md.append(f"| {r['name']} | {t['train']['n']:,}/{t['train']['pos']:,} | {t['val']['n']:,}/{t['val']['pos']:,} | "
                  f"{t['test']['n']:,}/{t['test']['pos']:,} | {t['drop']['n']:,} | {pct(L['test']['overlap'])} | "
                  f"{pct(L['test']['horizon'])} | {pct(L['test']['fire'])} | {pct(L['val']['overlap'])} | "
                  f"{pct(L['val']['fire'])} | {fid} |")
    md += ["", "## Black-Swan-Ereignisse je Split (Ereignis gehoert dem Split mit den meisten seiner Stichproben)", "",
           "| Split | Ereignisse train/val/test | grosse train/val/test | Feuer-Pixeltage train/val/test | "
           "Ereignisse in >1 Split (gross) |", "|---|---|---|---|---|"]
    for r in results:
        E = r["events"]
        tf = sum(E["fpd"].values()) or 1
        md.append(f"| {r['name']} | {'/'.join(str(E['all'][s]) for s in SPLITS)} | "
                  f"{'/'.join(str(E['major'][s]) for s in SPLITS)} | "
                  f"{'/'.join(str(round(100 * E['fpd'][s] / tf)) + ' %' for s in SPLITS)} | "
                  f"{E['multi']} / {E['n']} ({E['multi_major']} / {E['n_major']}) |")
    md += ["", "## Kandidaten", ""]
    for n, (_, info) in cand.items():
        md.append(f"- **{n}**: `{json.dumps(info)}`")
    md += ["", "## Testanteil je Jahr", "", "| Split | " + " | ".join(str(y)[2:] for y in YEARS) + " |",
           "|---" * (len(YEARS) + 1) + "|"]
    for r in results:
        cells = []
        for y in YEARS:
            v = r["years"][y]
            n = sum(v.values())
            cells.append(f"{100 * v['test'] / n:.0f}" if n else "–")
        md.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p41_split_audit{args.tag}", args.config, {
        "args": vars(args), "results": [{k: v for k, v in r.items() if k != "_split"} for r in results],
        "outputs": {"summary": str(out / "summary.md"), "splits": str(sp_dir)}})


if __name__ == "__main__":
    main()

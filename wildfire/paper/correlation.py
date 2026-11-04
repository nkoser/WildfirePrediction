"""Wie haengen die 55 Eingabekanaele zusammen? Korrelationsanalyse im Paper-Protokoll.

Grundlage ist das Zentrumspixel (oder mit --reduce mean das Ausschnittsmittel)
jeder Stichprobe an allen 61 Cache-Tagen (D-60 .. D), standardmaessig nur der
Train-Split. Alle Kanaele sind uint8 ohne scale/offset (Audit s7); Spearman
haengt nur von der Reihenfolge ab und ist gegen diese Kodierung unempfindlich.

  roh          Spearman ueber alle Stichproben-Tage gepoolt. Enthaelt raeumliche
               Unterschiede (DEM gegen t2m) und den Jahresgang.
  im Fenster   Spearman auf x - Fenstermittel je Stichprobe und Kanal. Ort und
               langsamer Jahresgang fallen weg, uebrig bleibt die Kopplung von
               Tag zu Tag - die Groesse, die ein Was-waere-wenn-Szenario
               konsistent halten muss. Statische Kanaele haben hier keine
               Varianz und fehlen.
  Jahreszeiten roh, getrennt nach DJF/MAM/JJA/SON des jeweiligen Tages.
  nur negativ  roh, nur Stichproben ohne Feuer an D+1 - zeigt, wie stark die
               Ueberrepraesentation der Feuertage die Matrix verschiebt.
  Verzoegerung corr(a_t, b_{t+k}) auf den Fenster-Anomalien, k = 0 .. --max-lag.
               Durch den Abzug des Fenstermittels sind lange Lags leicht
               negativ verzerrt; bei 61 Tagen und k <= 14 ist das klein.
  Cluster      hierarchisch (average linkage) auf 1 - |rho_roh|, geschnitten bei
               |rho| >= --cluster-rho; Vorschlag fuer datengetriebene Gruppen
               der Permutation Importance (importance.GROUPS).
  Teilkorr.    aus der Inversen der rohen Spearman-Matrix; VIF = diag(R^-1),
               R^2 eines Kanals aus allen anderen = 1 - 1/VIF.
  Determ.      R^2 bekannter Rechenbeziehungen auf Rangen (VPD aus t2m und
               Taupunkt, FWI aus ISI und BUI, ...). Nahe 1 heisst: der Kanal
               traegt nichts Eigenes und muss im Szenario mitgerechnet werden.
  Label        ROC-AUC je Kanal gegen Feuer im Ausschnitt an D+1, einmal fuer den
               Wert an D und einmal fuer das Mittel D-6 .. D; zusaetzlich je
               Monat gerechnet und gemittelt, damit der Jahresgang nicht als
               Trennschaerfe zaehlt.

Landbedeckung (ch54) ist kategorial und fehlt in allen Rangmatrizen; fuer sie
steht je Kanal das Korrelationsverhaeltnis eta (Anteil der Varianz, den die
Klasse erklaert). Der Anteil exakter Nullen je Kanal wird mitberichtet: ohne
NoData-Wert sind fehlende MODIS-Beobachtungen und gueltige 0 nicht zu trennen.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.prepare import CHANNEL_NAMES, LANDCOVER_CH

N_CH = len(CHANNEL_NAMES)
CENTRE = 6
STATIC = list(range(26, 38)) + [LANDCOVER_CH]      # OSM, Worldpop, DEM-Ableitungen, Landbedeckung
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
LAG_DRIVERS = ("precip", "t2m", "t2m_max", "dewpoint", "u10", "net_solar")
LAG_TARGETS = ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "VPD", "swvl1", "skin_reservoir",
               "LST_day", "NDVI", "MCD14A1_fire")
# Kanal <- Eingaben seiner Berechnung (Van Wagner 1987 fuer das FWI-System; VPD aus
# Temperatur und Taupunkt). Wind liegt nur als u10/v10 vor, Betrag nicht rekonstruierbar.
DETERMINISTIC = {
    "VPD": ("t2m", "dewpoint"),
    "FWI": ("ISI", "BUI"),
    "BUI": ("DMC", "DC"),
    "ISI": ("FFMC", "u10", "v10"),
    "FFMC": ("t2m", "dewpoint", "u10", "v10", "precip"),
    "DMC": ("t2m", "dewpoint", "precip"),
    "t2m_max": ("t2m",),
    "skin_t_max": ("t2m_max", "net_solar"),
    "EVI": ("NDVI",),
    "FPAR": ("LAI",),
}


# ---------------------------------------------------------------- Bausteine


def load_pixels(X: np.ndarray, rows: np.ndarray, reduce: str, chunk: int = 1024) -> np.ndarray:
    """(N, L, C, 13, 13) uint8 -> (n, L, C) float32 fuer die Zeilen rows."""
    out = np.empty((rows.size, X.shape[1], X.shape[2]), np.float32)
    for a in range(0, rows.size, chunk):
        r = rows[a:a + chunk]
        if reduce == "centre":
            out[a:a + r.size] = X[r, :, :, CENTRE, CENTRE]
        else:
            out[a:a + r.size] = X[r].mean(axis=(3, 4), dtype=np.float32)
        if (a // chunk) % 5 == 0:
            print(f"    {a + r.size}/{rows.size}", flush=True)
    return out


def day_months(dates: np.ndarray, L: int) -> np.ndarray:
    """JJJJMMTT des Stichtags D -> (n, L) Monat jedes Fenstertags D-(L-1) .. D."""
    base = np.array([np.datetime64(dt.date(int(d) // 10000, int(d) // 100 % 100, int(d) % 100))
                     for d in dates])
    days = base[:, None] - np.arange(L - 1, -1, -1).astype("timedelta64[D]")[None]
    return days.astype("datetime64[M]").astype(int) % 12 + 1


def ranks(Z: np.ndarray) -> np.ndarray:
    """Spaltenweise Rangtransformation (Bindungen gemittelt), standardisiert; konstante Spalten -> NaN."""
    R = np.empty(Z.shape, np.float64)
    for j in range(Z.shape[1]):
        R[:, j] = rankdata(Z[:, j])
    R -= R.mean(0)
    s = R.std(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return R / np.where(s > 0, s, np.nan)


def spearman(Z: np.ndarray) -> np.ndarray:
    """(m, c) -> (c, c) Spearman; Spalten ohne Varianz geben NaN-Zeilen."""
    R = ranks(Z)
    return np.clip(R.T @ R / R.shape[0], -1.0, 1.0)


def finite(R: np.ndarray) -> np.ndarray:
    """NaN-Eintraege (Kanal ohne Varianz) -> 0, Diagonale 1."""
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    return R


def lagged(Zr: np.ndarray, max_lag: int) -> np.ndarray:
    """Zr (n, L, c) standardisierte Range -> (K+1, c, c) mit [k, a, b] = corr(a_t, b_{t+k})."""
    n, L, c = Zr.shape
    out = np.empty((max_lag + 1, c, c))
    for k in range(max_lag + 1):
        A = Zr[:, :L - k].reshape(-1, c)
        B = Zr[:, k:].reshape(-1, c)
        with np.errstate(invalid="ignore", divide="ignore"):
            A = (A - A.mean(0)) / A.std(0)
            B = (B - B.mean(0)) / B.std(0)
        out[k] = A.T @ B / A.shape[0]
    return out


def partial_and_vif(R: np.ndarray, ridge: float = 1e-6):
    """Korrelationsmatrix -> (Teilkorrelationen, VIF); ridge haelt fast singulaere Matrizen invertierbar."""
    P = np.linalg.inv(R + ridge * np.eye(len(R)))
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pc, np.diag(P)


def r2_from(R: np.ndarray, y: int, xs: list[int]) -> float:
    """R^2 der linearen Regression von y auf xs, nur aus der Korrelationsmatrix."""
    Rxx = R[np.ix_(xs, xs)]
    rxy = R[xs, y]
    return float(rxy @ np.linalg.solve(Rxx + 1e-9 * np.eye(len(xs)), rxy))


def clusters(R: np.ndarray, names: list[str], rho: float):
    """Average linkage auf 1 - |rho|; Schnitt bei Abstand 1 - rho. -> (Gruppen, Blattreihenfolge, Linkage)."""
    D = 1.0 - np.abs(R)
    np.fill_diagonal(D, 0.0)
    Lk = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Lk, t=1.0 - rho, criterion="distance")
    groups = {}
    for n, g in zip(names, lab):
        groups.setdefault(int(g), []).append(n)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), names.index(g[0])))
    return ordered, leaves_list(Lk), Lk


def eta(x: np.ndarray, cls: np.ndarray) -> float:
    """Korrelationsverhaeltnis: Anteil der Varianz von x, den die Klassen erklaeren (Wurzel)."""
    tot = x.var()
    if tot == 0:
        return float("nan")
    between = sum((cls == k).mean() * (x[cls == k].mean() - x.mean()) ** 2 for k in np.unique(cls))
    return float(np.sqrt(between / tot))


def auc_by_month(x: np.ndarray, y: np.ndarray, month: np.ndarray, min_n: int = 50) -> float:
    """ROC-AUC je Monat, gewichtet nach Stichproben; Monate mit nur einer Klasse oder zu wenig Daten fallen weg."""
    num = den = 0.0
    for m in np.unique(month):
        s = month == m
        if s.sum() >= min_n and 0 < y[s].sum() < s.sum():
            num += s.sum() * roc_auc_score(y[s], x[s])
            den += s.sum()
    return num / den if den else float("nan")


# ---------------------------------------------------------------- Ausgabe


def write_matrix(path: Path, M: np.ndarray, names: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + names)
        for n, row in zip(names, M):
            w.writerow([n] + [("" if np.isnan(v) else round(float(v), 4)) for v in row])


def top_pairs(M: np.ndarray, names: list[str], thr: float) -> list[dict]:
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(M[i, j]) and abs(M[i, j]) >= thr:
                out.append({"a": names[i], "b": names[j], "rho": round(float(M[i, j]), 4)})
    return sorted(out, key=lambda p: -abs(p["rho"]))


def plot_all(fig_dir: Path, tag: str, res: dict, names: list[str], within_names: list[str],
             order: np.ndarray, Lk, lag: np.ndarray, label_rows: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    paths = []

    def heat(M, nm, title, fname, idx=None):
        idx = np.arange(len(nm)) if idx is None else idx
        fig, ax = plt.subplots(figsize=(13, 11.5))
        im = ax.imshow(M[np.ix_(idx, idx)], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(idx)), [nm[i] for i in idx], rotation=90, fontsize=7)
        ax.set_yticks(range(len(idx)), [nm[i] for i in idx], fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
        fig.tight_layout()
        p = fig_dir / fname
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(str(p))

    heat(res["raw"], names, "Spearman roh (gepoolt), nach Clustern sortiert", f"corr_raw_{tag}.png", order)
    wi = [names.index(n) for n in within_names]
    sub_order = [i for i in order if i in wi]
    heat(res["raw"], names, "roh - nur dynamische Kanaele", f"corr_raw_dyn_{tag}.png", np.array(sub_order))
    w_order = np.array([within_names.index(names[i]) for i in sub_order])
    heat(res["within"], within_names, "Spearman im Fenster (Fenstermittel abgezogen)",
         f"corr_within_{tag}.png", w_order)

    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Lk, labels=names, leaf_rotation=90, leaf_font_size=7, ax=ax, color_threshold=None)
    ax.set_ylabel("1 - |rho|")
    ax.set_title("Hierarchisches Clustering (average linkage) der rohen Spearman-Matrix")
    fig.tight_layout()
    p = fig_dir / f"corr_dendrogram_{tag}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))

    drivers = [d for d in LAG_DRIVERS if d in within_names]
    tg = [t for t in LAG_TARGETS if t in within_names]
    fig, axes = plt.subplots(2, (len(drivers) + 1) // 2, figsize=(5 * ((len(drivers) + 1) // 2), 8),
                             squeeze=False, sharey=True)
    ks = np.arange(lag.shape[0])
    cmap = plt.get_cmap("tab20")
    for ax, d in zip(axes.ravel(), drivers):
        a = within_names.index(d)
        for j, t in enumerate(tg):
            if t != d:
                ax.plot(ks, lag[:, a, within_names.index(t)], color=cmap(j), lw=1.3, label=t)
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_title(f"{d} an t  gegen  X an t + k")
        ax.set_xlabel("k [Tage]")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Spearman (Fenster-Anomalien)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = fig_dir / f"corr_lag_{tag}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))

    rows_ = sorted(label_rows, key=lambda r: -abs(r["auc_D"] - 0.5))
    fig, ax = plt.subplots(figsize=(9, 12))
    y = np.arange(len(rows_))[::-1]
    ax.barh(y + 0.2, [r["auc_D"] - 0.5 for r in rows_], 0.4, label="Wert an D", color="#2a6fdb")
    ax.barh(y - 0.2, [r["auc_D_month"] - 0.5 for r in rows_], 0.4, label="Wert an D, je Monat", color="#e8a33d")
    ax.set_yticks(y, [r["channel"] for r in rows_], fontsize=7)
    ax.axvline(0, color="#555", lw=0.6)
    ax.set_xlabel("ROC-AUC - 0,5 gegen Feuer im Ausschnitt an D+1")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = fig_dir / f"corr_label_{tag}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))
    return paths


def report_md(path: Path, args, info: dict, res: dict, names: list[str], within_names: list[str],
              groups: list[list[str]], vif: dict, det: dict, lag_peaks: list[dict],
              label_rows: list[dict], zeros: dict, eta_lc: dict, figs: list[str]) -> None:
    A = []
    A.append(f"# Korrelationsanalyse der Eingabekanaele ({args.cache}, {args.split}, {args.reduce})")
    A.append("")
    A.append(f"{info['n_samples']} Stichproben ({info['n_pos']} mit Feuer an D+1) x {info['L']} Tage = "
             f"{info['n_rows']} Stichproben-Tage. Spearman; Landbedeckung kategorial, nur eta.")
    A.append("")
    A.append(f"## Stark gekoppelte Paare, roh (|rho| >= {args.pair_rho})")
    A.append("")
    A.append("| a | b | rho roh | rho im Fenster |")
    A.append("|---|---|---|---|")
    for p in top_pairs(res["raw"], names, args.pair_rho)[:40]:
        w = "-"
        if p["a"] in within_names and p["b"] in within_names:
            w = f"{res['within'][within_names.index(p['a']), within_names.index(p['b'])]:+.2f}"
        A.append(f"| {p['a']} | {p['b']} | {p['rho']:+.2f} | {w} |")
    A.append("")
    A.append(f"## Cluster (average linkage, |rho| >= {args.cluster_rho})")
    A.append("")
    for g in groups:
        if len(g) > 1:
            A.append(f"- {', '.join(g)}")
    A.append(f"- Einzeln: {', '.join(g[0] for g in groups if len(g) == 1)}")
    A.append("")
    A.append("## Redundanz: R^2 aus allen anderen Kanaelen (1 - 1/VIF), roh")
    A.append("")
    A.append("| Kanal | VIF | R^2 |")
    A.append("|---|---|---|")
    for n, v in sorted(vif.items(), key=lambda kv: -kv[1])[:20]:
        A.append(f"| {n} | {v:.1f} | {1 - 1 / v:.3f} |")
    A.append("")
    A.append("## Rechenbeziehungen: R^2 auf Rangen (roh / im Fenster)")
    A.append("")
    A.append("| Kanal | aus | R^2 roh | R^2 im Fenster |")
    A.append("|---|---|---|---|")
    for n, d in det.items():
        rw = "-" if d["r2_within"] is None else f"{d['r2_within']:.3f}"
        A.append(f"| {n} | {', '.join(d['from'])} | {d['r2_raw']:.3f} | {rw} |")
    A.append("")
    A.append("## Verzoegerte Kopplung (Fenster-Anomalien): staerkster Lag je Paar")
    A.append("")
    A.append("| Treiber an t | Kanal an t+k | rho bei k=0 | bester k | rho bei bestem k | rho bei k=max |")
    A.append("|---|---|---|---|---|---|")
    for r in lag_peaks:
        A.append(f"| {r['driver']} | {r['target']} | {r['rho_0']:+.2f} | {r['k_best']} | "
                 f"{r['rho_best']:+.2f} | {r['rho_max_lag']:+.2f} |")
    A.append("")
    A.append("## Trennschaerfe gegen Feuer an D+1 (ROC-AUC, Train)")
    A.append("")
    A.append("Je Monat = AUC innerhalb jedes Monats, gewichtet gemittelt. Kanal 0 und die "
             "Helligkeitstemperaturen sind Feuer-Rohmaterial (Audit s7), keine Vorhersage.")
    A.append("")
    A.append("| Kanal | AUC an D | AUC D-6..D | AUC an D je Monat |")
    A.append("|---|---|---|---|")
    for r in sorted(label_rows, key=lambda r: -abs(r["auc_D"] - 0.5))[:25]:
        A.append(f"| {r['channel']} | {r['auc_D']:.3f} | {r['auc_7d']:.3f} | {r['auc_D_month']:.3f} |")
    A.append("")
    A.append("## Datenqualitaet")
    A.append("")
    hi = {n: z for n, z in zeros.items() if z >= 0.05}
    A.append("Anteil exakter Nullen >= 5 % (fehlende Beobachtung nicht von 0 unterscheidbar): "
             + (", ".join(f"{n} {z:.0%}" for n, z in sorted(hi.items(), key=lambda kv: -kv[1])) or "keine"))
    A.append("")
    A.append(f"Groesste Verschiebung roh, nur negative Stichproben gegen alle: max |delta rho| = "
             f"{info['max_shift_neg']:.2f} ({info['max_shift_pair']}).")
    A.append("")
    A.append("Landbedeckung, eta (staerkste 10): "
             + ", ".join(f"{n} {v:.2f}" for n, v in sorted(eta_lc.items(), key=lambda kv: -kv[1])[:10]))
    A.append("")
    A.append("## Abbildungen")
    A.append("")
    for f in figs:
        A.append(f"- `{f}`")
    path.write_text("\n".join(A) + "\n")


# ---------------------------------------------------------------- Ablauf


def run(args, cfg) -> None:
    exp = Path(cfg["experiment_root"])
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy", mmap_mode="r")
    split = np.load(d / "split.npy")
    meta = np.load(d / "meta.npy")
    y_all = targets.labels(d, args.target)
    rows = np.arange(split.size) if args.split == "all" else np.where(split == args.split)[0]
    if args.max_samples and rows.size > args.max_samples:
        rows = np.sort(np.random.default_rng(args.seed).choice(rows, args.max_samples, replace=False))
    y = y_all[rows]
    print(f"[corr] {args.cache} {args.split}: {rows.size} Stichproben ({int(y.sum())} positiv), "
          f"Fenster {X.shape[1]} Tage, reduce={args.reduce}", flush=True)

    V = load_pixels(X, rows, args.reduce)                      # (n, L, C)
    n, L, C = V.shape
    month = day_months(meta[rows, 0], L)                       # (n, L)
    flat = V.reshape(-1, C)
    tag = f"{args.cache}{targets.tag(args.target)}_{args.split}_{args.reduce}"
    out = exp / "reports" / f"correlation_{tag}"
    fig_dir = exp / "figures" / f"correlation_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    num = [c for c in range(C) if c != LANDCOVER_CH]
    names = [CHANNEL_NAMES[c] for c in num]
    dyn = [c for c in num if c not in STATIC]
    within_names = [CHANNEL_NAMES[c] for c in dyn]

    zeros = {CHANNEL_NAMES[c]: float((flat[:, c] == 0).mean()) for c in range(C)}
    lc = flat[:, LANDCOVER_CH]
    eta_lc = {CHANNEL_NAMES[c]: eta(flat[:, c], lc) for c in num}

    res = {}
    print("[corr] roh", flush=True)
    res["raw"] = spearman(flat[:, num])
    print("[corr] im Fenster", flush=True)
    anom = V[:, :, dyn] - V[:, :, dyn].mean(1, keepdims=True)
    res["within"] = spearman(anom.reshape(-1, len(dyn)))
    for s, ms in SEASONS.items():
        sel = np.isin(month.ravel(), ms)
        res[f"raw_{s}"] = spearman(flat[sel][:, num]) if sel.sum() > 1000 else np.full((len(num),) * 2, np.nan)
        print(f"[corr] {s}: {int(sel.sum())} Tage", flush=True)
    neg = np.repeat(y == 0, L)
    res["raw_neg"] = spearman(flat[neg][:, num])

    print("[corr] Verzoegerung", flush=True)
    Zr = ranks(anom.reshape(-1, len(dyn))).reshape(n, L, len(dyn))
    lag = lagged(Zr, args.max_lag)
    del Zr

    # Kanaele ohne Varianz (NaN) zaehlen fuer Inverse, Clustering und R^2 als unkorreliert
    R0, W0 = finite(res["raw"]), finite(res["within"])
    pc, vif_arr = partial_and_vif(R0)
    vif = {nm: float(v) for nm, v in zip(names, vif_arr)}
    groups, order, Lk = clusters(R0, names, args.cluster_rho)

    det = {}
    for tgt, src in DETERMINISTIC.items():
        det[tgt] = {"from": list(src),
                    "r2_raw": r2_from(R0, names.index(tgt), [names.index(s) for s in src]),
                    "r2_within": (r2_from(W0, within_names.index(tgt),
                                          [within_names.index(s) for s in src])
                                  if all(x in within_names for x in (tgt, *src)) else None)}

    lag_peaks = []
    for dr in LAG_DRIVERS:
        a = within_names.index(dr)
        for t in LAG_TARGETS:
            if t == dr:
                continue
            curve = lag[:, a, within_names.index(t)]
            kb = int(np.argmax(np.nan_to_num(np.abs(curve))))
            lag_peaks.append({"driver": dr, "target": t, "rho_0": float(curve[0]), "k_best": kb,
                              "rho_best": float(curve[kb]), "rho_max_lag": float(curve[-1]),
                              "curve": [round(float(v), 4) for v in curve]})

    m_D = month[:, -1]
    label_rows = []
    for c in range(C):
        xD, x7 = V[:, -1, c], V[:, -7:, c].mean(1)
        label_rows.append({"channel": CHANNEL_NAMES[c],
                           "auc_D": float(roc_auc_score(y, xD)),
                           "auc_7d": float(roc_auc_score(y, x7)),
                           "auc_D_month": float(auc_by_month(xD, y, m_D)),
                           "zero_frac": round(zeros[CHANNEL_NAMES[c]], 4)})

    diff = np.abs(res["raw_neg"] - res["raw"])
    i, j = np.unravel_index(np.nanargmax(diff), diff.shape)
    info = {"n_samples": int(n), "n_pos": int(y.sum()), "L": int(L), "n_rows": int(n * L),
            "max_shift_neg": float(diff[i, j]), "max_shift_pair": f"{names[i]} / {names[j]}"}

    for k, M in res.items():
        write_matrix(out / f"spearman_{k}.csv", M, within_names if k == "within" else names)
    write_matrix(out / "partial_raw.csv", pc, names)
    np.savez_compressed(out / "lag_within.npz", lag=lag.astype(np.float32), names=np.array(within_names))
    with (out / "label_assoc.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(label_rows[0]))
        w.writeheader(); w.writerows(label_rows)
    (out / "clusters.json").write_text(json.dumps(
        {"rho": args.cluster_rho, "groups": groups,
         "importance_groups": {f"C_{'_'.join(g[:2])}": [CHANNEL_NAMES.index(x) for x in g]
                               for g in groups if len(g) > 1}}, indent=2))
    figs = plot_all(fig_dir, tag, res, names, within_names, order, Lk, lag, label_rows)
    report_md(out / "report.md", args, info, res, names, within_names, groups, vif, det,
              [{k: v for k, v in r.items() if k != "curve"} for r in lag_peaks],
              label_rows, zeros, eta_lc, figs)
    print(f"[corr] {out}/report.md", flush=True)
    print((out / "report.md").read_text(), flush=True)

    write_manifest(cfg, f"p19_correlation_{tag}", args.config, {
        "cache": args.cache, "target": args.target, "split": args.split, "reduce": args.reduce,
        "rows": info, "max_lag": args.max_lag, "cluster_rho": args.cluster_rho,
        "outputs": {"dir": str(out), "figures": figs},
        "clusters": groups, "vif": vif, "deterministic": det, "lag_peaks": lag_peaks,
        "label": label_rows, "zero_frac": zeros, "eta_landcover": eta_lc,
        "top_pairs_raw": top_pairs(res["raw"], names, args.pair_rho),
        "top_pairs_within": top_pairs(res["within"], within_names, args.pair_rho),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    ap.add_argument("--reduce", choices=("centre", "mean"), default="centre",
                    help="Zentrumspixel oder Mittel ueber den 13x13-Ausschnitt")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = alle Stichproben des Splits")
    ap.add_argument("--max-lag", type=int, default=14)
    ap.add_argument("--cluster-rho", type=float, default=0.7)
    ap.add_argument("--pair-rho", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    run(args, cfg)


if __name__ == "__main__":
    main()

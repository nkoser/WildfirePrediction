"""Permutation Importance fuer die neuen Modelle, auch nur fuer neue Zuendungen.

Wie importance.py (Kanalfenster zwischen Test-Stichproben vertauscht, Wichtigkeit = Metrik ohne minus mit
Vertauschung, PR-AUC und ROC-AUC, ohne neues Training), aber:

  Modelle   alles, was msv2_eval.run_paper auswertet: STS-LH beta = 1 (_abl_lhbce10), LH + Zustandsmerkmale,
            Multi-State v2, Contagion (von Grund auf und warm gestartet). Laden und Score wie dort:
            F(dt + 1) am Fensterende e = L - dt - 2, Ausschnitt = Ausschnittskopf, Mitte = Pixel (6, 6).
  Merkmale  Die Zustandsmerkmale (state_features, 183 Tage Feuer-Historie je Pixel) sind fuer Modelle, die sie
            nutzen, eine eigene Variable "state_feat"; die Gruppe G_fire_history_all vertauscht Feuerkanal und
            Merkmale zusammen. Sonst bliebe die Historie trotz vertauschtem Feuerkanal im Modell.
  Teilmengen
            all     alle Test-Stichproben (wie importance.py)
            ign30   kein Feuer im Ausschnitt in den 30 Fenstertagen bis e; Positive sind damit neue Zuendungen
            ign183  zusaetzlich kein Feuer in 183 Tagen (Zustandsmerkmale is_O, is_B in keinem Pixel)
            Vertauscht wird innerhalb der Teilmenge, ihre Randverteilung bleibt. In ign30/ign183 ist der
            Feuerkanal im Fenster ueberall 0, seine Wichtigkeit dort also per Konstruktion 0.

Je Aufruf ein Checkpoint (--index in die Liste aller NAME=GLOB-Dateien), Ergebnis
metrics/importance_v2/<name>_s<seed><tag>.json; --aggregate fasst zu CSV und Abbildung zusammen.

  python -m wildfire.paper.importance_v2 --config ... --models lhb10=GLOB msv2=GLOB ... --index 0
  python -m wildfire.paper.importance_v2 --config ... --aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import multistate_v2 as msv2
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.importance import GROUPS, SEASON
from wildfire.paper.msv2_eval import load_net, model_files, seed_of
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import CHANNEL_NAMES, FIRE_CH
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import doy_of

HALF = 6
STATE = "state_feat"
SUBSETS = ("all", "ign30", "ign183")
METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}
OUT = "importance_v2"


def variables(use_feat: bool) -> dict[str, dict]:
    """Name -> {"ch": Kanaele, "doy": Kalendertag vertauschen, "feat": Zustandsmerkmale vertauschen}."""
    v = {n: {"ch": [c], "doy": False, "feat": False} for c, n in enumerate(CHANNEL_NAMES)}
    v.update({g: {"ch": list(c), "doy": False, "feat": False} for g, c in GROUPS.items()})
    v[SEASON] = {"ch": [], "doy": True, "feat": False}
    if use_feat:
        v[STATE] = {"ch": [], "doy": False, "feat": True}
        v["G_fire_history_all"] = {"ch": [FIRE_CH], "doy": False, "feat": True}
    return v


def subset_masks(X, fz, rows, e: int, j: int) -> dict[str, np.ndarray]:
    """Boolesche Masken ueber rows: Feuer im Ausschnitt in den 30 Fenstertagen bis e bzw. in 183 Tagen."""
    win = np.asarray(X[rows, e - 29:e + 1, FIRE_CH]).reshape(rows.size, -1).max(1) > 0
    f = np.asarray(fz[rows, j], np.float32)                                    # (n, 169, NF)
    f183 = (f[..., 3] + f[..., 4]).max(1) > 0.5
    return {"all": np.ones(rows.size, bool), "ign30": ~win, "ign183": ~win & ~f183}


def metric(y, s) -> dict:
    if y.sum() == 0 or y.sum() == y.size:
        return {m: float("nan") for m in METRICS}
    return {m: float(fn(y, s)) for m, fn in METRICS.items()}


def run(args, cfg) -> None:
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    todo = [(nm, f) for nm, fl in model_files(args.models).items() for f in fl]
    if args.index >= len(todo):
        raise SystemExit(f"[imp2] Index {args.index} ausserhalb von {len(todo)} Checkpoints")
    name, ckpt = todo[args.index]
    out = exp / "metrics" / OUT / f"{name}_s{seed_of(ckpt)}{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert")
    out.parent.mkdir(parents=True, exist_ok=True)

    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")
    split = np.load(d / "split.npy")
    doyD = doy_of(np.load(d / "meta.npy")[:, 0])
    LC = X.shape[1]
    L = LC + targets.shift(args.target)
    rows = np.where(split == "test")[0]
    ok = targets.ok_mask(d)
    if ok is not None:
        rows = rows[ok[rows]]
    fz = np.load(exp / "derived_data" / "state_features" / f"paper{args.feat_tag}.npy", mmap_mode="r")
    info = json.loads((exp / "derived_data" / "state_features" / f"paper{args.feat_tag}.json").read_text())
    assert list(info["dts"]) == list(DTS), "Merkmale passen nicht zum Paper-Cache"
    rows = rows[rows < fz.shape[0]]
    if args.max_test:
        rows = np.sort(np.random.default_rng(args.seed).choice(rows, min(args.max_test, rows.size), replace=False))
    fire_t = targets.fire_map(d, X, args.target)
    lab = {"patch": targets.labels(d, args.target)[rows], "centre": fire_t[rows, HALF, HALF].astype(int)}

    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    net = load_net(ckpt, device)
    use_feat = msv2.needs_features(net)
    var = variables(use_feat)
    print(f"[imp2] {name} {Path(ckpt).name}: {rows.size} Test-Stichproben, Merkmale {use_feat}, {len(var)} Variablen, "
          f"dt {list(args.dts)}, {args.repeats} Wiederholung(en)", flush=True)

    @torch.no_grad()
    def scores(rr, j, e, spec=None, src=None, bs=64):
        """F(dt + 1) fuer Ausschnitt und Mitte der Zeilen rr; spec/src: Variable, deren Eingabe aus src kommt."""
        k = torch.tensor([float(L - e - 1)], device=device)                     # dt + 1
        sp, sc = [], []
        for a in range(0, rr.size, bs):
            r = rr[a:a + bs]
            rs = r if src is None else src[a:a + bs]
            x = windows(Xg, r, np.full(r.size, e))
            doy = doyD[r]
            if spec is not None:
                if spec["ch"]:
                    ci = torch.tensor(spec["ch"], device=device)
                    x[:, :, ci] = windows(Xg, rs, np.full(rs.size, e))[:, :, ci]
                if spec["doy"]:
                    doy = doyD[rs]
            x = (x.float() - m_t) / s_t
            if use_feat:
                fr = rs if spec is not None and spec["feat"] else r
                net.set_features(torch.from_numpy(np.asarray(fz[fr, j], np.float32)).to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw_px, raw_pa = net(x, torch.from_numpy(doy - (LC - 1 - e)).to(device))
            sc.append((-torch.expm1(net.px_head.log_survival(raw_px[:, HALF, HALF].float(), k))[:, 0]).float().cpu().numpy())
            sp.append((-torch.expm1(net.pa_head.log_survival(raw_pa.float(), k))[:, 0]).float().cpu().numpy())
        return {"patch": np.concatenate(sp), "centre": np.concatenate(sc)}

    rng = np.random.default_rng(args.seed)
    res: dict = {}
    for dt_ in args.dts:
        j = list(DTS).index(dt_)
        e = L - dt_ - 2
        masks = subset_masks(X, fz, rows, e, j)
        for sub in args.subsets:
            t0 = time.time()
            m = masks[sub]
            rr = rows[m]
            y = {tg: v[m] for tg, v in lab.items()}
            evaluate = lambda s: {tg: metric(y[tg], s[tg]) for tg in y}     # noqa: E731
            base = evaluate(scores(rr, j, e))
            perms = [rr[rng.permutation(rr.size)] for _ in range(args.repeats)]
            drops = {}
            for vn, spec in var.items():
                if sub != "all" and spec["ch"] == [FIRE_CH] and not spec["feat"]:
                    continue                                                # Feuerkanal dort konstant 0
                ms = [evaluate(scores(rr, j, e, spec, src)) for src in perms]
                drops[vn] = {tg: {mt: [base[tg][mt] - x_[tg][mt] for x_ in ms] for mt in METRICS} for tg in y}
            res.setdefault(str(dt_), {})[sub] = {
                "n": int(rr.size), "n_pos": {tg: int(v.sum()) for tg, v in y.items()}, "base": base, "drop": drops}
            top = sorted(drops, key=lambda n: -np.nanmean(drops[n]["patch"]["pr_auc"]))[:8]
            print(f"[imp2] dt={dt_:2d} {sub:6s} n={rr.size} pos={int(y['patch'].sum())} Basis PR-AUC "
                  f"{base['patch']['pr_auc']:.3f} in {time.time() - t0:.0f} s; groesster Abfall: "
                  + ", ".join(f"{n} {np.nanmean(drops[n]['patch']['pr_auc']):+.3f}" for n in top), flush=True)

    doc = {"name": name, "checkpoint": ckpt, "cache": args.cache, "target": args.target, "use_feat": use_feat,
           "n_test": int(rows.size), "repeats": args.repeats, "seed": args.seed,
           "method": "Eingabe der Variable innerhalb der Teilmenge vertauscht; drop = Basis - vertauscht",
           "score": "F(dt + 1) am Fensterende e = L - dt - 2; Ausschnitt = Ausschnittskopf, Mitte = Pixel (6, 6)",
           "subsets": {"all": "alle Test-Stichproben", "ign30": "kein Feuer im Ausschnitt in 30 Fenstertagen",
                       "ign183": "ign30 und kein Feuer im Ausschnitt in 183 Tagen (Zustandsmerkmale)"},
           "variables": {n: {"channels": [CHANNEL_NAMES[c] for c in s["ch"]], "season": s["doy"], "state_feat": s["feat"]}
                         for n, s in var.items()},
           "results": res}
    out.write_text(json.dumps(doc, indent=1, default=float))
    write_manifest(cfg, f"p42_{OUT}_{name}_s{seed_of(ckpt)}{args.tag}", args.config, {"output": str(out)})
    print(f"[imp2] -> {out}", flush=True)


def aggregate(args, cfg) -> None:
    """Alle JSON unter metrics/importance_v2 (mit --tag) -> CSV (Mittel, Streuung ueber Seeds und Wiederholungen)."""
    exp = Path(cfg["experiment_root"])
    files = sorted((exp / "metrics" / OUT).glob(f"*_s[0-9]{args.tag}.json"))
    docs = [json.loads(p.read_text()) for p in files]
    if not docs:
        raise SystemExit(f"[imp2] keine Ergebnisse unter {exp / 'metrics' / OUT}")
    rows_out = []
    for name in sorted({doc["name"] for doc in docs}):
        ds = [doc for doc in docs if doc["name"] == name]
        for dt_ in ds[0]["results"]:
            for sub, r0 in ds[0]["results"][dt_].items():
                for tg in ("patch", "centre"):
                    for vn in r0["drop"]:
                        r = {"model": name, "dt": int(dt_), "subset": sub, "label": tg, "variable": vn,
                             "group": vn.startswith("G_") or vn in (SEASON, STATE), "n_ckpt": len(ds),
                             "n": r0["n"], "n_pos": r0["n_pos"][tg]}
                        for mt in METRICS:
                            v = np.concatenate([doc["results"][dt_][sub]["drop"][vn][tg][mt] for doc in ds])
                            r[f"base_{mt}"] = round(float(np.nanmean([doc["results"][dt_][sub]["base"][tg][mt]
                                                                      for doc in ds])), 4)
                            r[f"drop_{mt}"] = round(float(np.nanmean(v)), 4)
                            r[f"drop_{mt}_std"] = round(float(np.nanstd(v)), 4)
                        rows_out.append(r)
    out = exp / "reports" / f"{OUT}_{args.cache}{targets.tag(args.target)}{args.tag}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    print(f"[imp2] {out} ({len(rows_out)} Zeilen)", flush=True)
    for sub in sorted({r["subset"] for r in rows_out}):
        plot(rows_out, sub, exp / "figures" / f"{OUT}_{args.cache}{targets.tag(args.target)}{args.tag}_{sub}.png")


DT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]         # Kategorien 1-4 der Referenzpalette, fest je dt


def plot(rows_out: list[dict], sub: str, path: Path) -> None:
    """Je Modell eine Zeile: links Gruppen, rechts die 15 wichtigsten Einzelkanaele; Balken je dt (Ausschnitt)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [r for r in rows_out if r["label"] == "patch" and r["subset"] == sub]
    models = sorted({r["model"] for r in sel})
    dts = sorted({r["dt"] for r in sel})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#0b0b0b"})
    fig, axes = plt.subplots(len(models), 2, figsize=(14, 5.2 * len(models)), squeeze=False, facecolor="#fcfcfb")
    for i, mdl in enumerate(models):
        for jx, grp in enumerate((True, False)):
            rs = [r for r in sel if r["model"] == mdl and r["group"] == grp]
            score: dict = {}
            for r in rs:
                score[r["variable"]] = score.get(r["variable"], 0.0) + r["drop_pr_auc"]
            names = sorted(score, key=lambda n: -score[n])[: (None if grp else 15)][::-1]
            ax, h = axes[i, jx], 0.8 / len(dts)
            ax.set_facecolor("#fcfcfb")
            for kk, dt_ in enumerate(dts):
                val = {r["variable"]: r for r in rs if r["dt"] == dt_}
                y = np.arange(len(names)) + (kk - (len(dts) - 1) / 2) * h
                ax.barh(y, [val[n]["drop_pr_auc"] for n in names], h * 0.9,
                        xerr=[val[n]["drop_pr_auc_std"] for n in names], color=DT_COLORS[kk % len(DT_COLORS)],
                        label=f"Δt = {dt_}", error_kw={"lw": 0.6, "ecolor": "#52514e"})
            ax.set_yticks(np.arange(len(names)), names, fontsize=8)
            ax.axvline(0, color="#52514e", lw=0.6)
            ax.set_xlabel("Drop in patch PR-AUC when permuted")
            n_ = rs[0]["n"] if rs else 0
            pos = rs[0]["n_pos"] if rs else 0
            ax.set_title(f"{mdl} · {sub} (n = {n_}, positives = {pos}): "
                         f"{'groups' if grp else 'top 15 single channels'}", fontsize=10, loc="left")
            ax.grid(axis="x", alpha=0.25)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    print(f"[imp2] {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=[], help="NAME=GLOB")
    ap.add_argument("--index", type=int, default=0, help="Checkpoint in der Liste aller NAME=GLOB-Dateien")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap.add_argument("--feat-tag", default="")
    ap.add_argument("--dts", type=int, nargs="+", default=list(DTS))
    ap.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-test", type=int, default=0, help="Rauchtest: zufaellige Teilmenge des Test-Splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    (aggregate if args.aggregate else run)(args, cfg)


if __name__ == "__main__":
    main()

"""Attribution auf die Eingabe: welches Pixel an welchem Tag ueber welchen Kanal die Vorhersage traegt.

Attention-Karten (attention_maps.py) gewichten nur ganze Token; der Stem hat zu dem Zeitpunkt alle 55 Kanaele eines
Pixels und seiner 3 x 3-Nachbarn schon gemischt. Kanalgenau ist erst die Attribution auf die Eingabe:

  Integrated Gradients   A = (x - b) * mittlerer Gradient von F(dt + 1) entlang b + a (x - b), a in (0, 1],
                         Riemann-Mittelpunkte (--steps), gemittelt ueber --n-refs Referenzen b = Fenster zufaelliger
                         anderer Test-Stichproben am selben Stichtag (wie bei Expected Gradients; dieselbe Logik wie die
                         Vertauschung in importance_v2). Keine Null-Baseline: die GroupNorm im Stem macht das Netz
                         nahezu skaleninvariant, der Pfad a x waere fast flach.
                         A hat die Form 30 Tage x 55 Kanaele x 13 x 13, mit Vorzeichen (+ erhoeht das Risiko).
  Vollstaendigkeit       sum A = F(x) - mittleres F(b) bis auf den Diskretisierungsfehler; gemeldet als relativer Fehler.
                         Die Restluecke je Stichprobe wird gespeichert (resid) und in den Kanal-Anteilen als eigene
                         Spalte gezeigt: bei Contagion / Multi-State der Beitrag des beobachteten Feuers ueber die Schwelle.
                         Contagion / Multi-State lesen den Brennzustand ueber eine Schwelle aus dem Feuerkanal - durch
                         die fliesst kein Gradient, der Feuerkanal wird dort unterschaetzt und die Luecke ist groesser.
                         Zustandsmerkmale (msv2, lhf) bleiben fest und werden nicht attribuiert.
  Score                  wie msv2_eval.run_paper: F(dt + 1) am Fensterende e = L - dt - 2, Ausschnittskopf und Mitte.

Verdichtet je Modell (Seeds gemittelt), dt und Gruppe (all · Feuer am Zieltag, all · kein Feuer, neue Zuendung =
ign30 · Feuer am Zieltag): Anteil je Kanal, Karte je Kanal (Summe ueber Tage), Profil je Kanal ueber die Tage;
dazu je Stichprobe die Kanalsummen und fuer Beispiele der volle Tensor (Seed 0).

  python -m wildfire.paper.attribution_maps --config ... --models lhb10=GLOB --tag _lhb10
  python -m wildfire.paper.attribution_maps --config ... --plot-only --tags _lhb10 _msv2 ...
  -> metrics/attribution_maps/attribution<tag>.npz (+ .json), figures/attribution_maps_*.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import multistate_v2 as msv2
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.msv2_eval import load_net, model_files
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import CHANNEL_NAMES, FIRE_CH
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import doy_of

P, HALF = 13, 6
HEADS = ("patch", "centre")
GROUPS = ("all · fire next day", "all · no fire", "new ignition · fire next day")
OUT = "attribution_maps"


def groups(y: np.ndarray, ign: np.ndarray) -> dict[str, np.ndarray]:
    return {GROUPS[0]: y, GROUPS[1]: ~y, GROUPS[2]: ign & y}


def scores(net, x, doy, k):
    """-> F(dt + 1) Ausschnitt, Mitte (B,) float32, differenzierbar."""
    raw_px, raw_pa = net(x, doy)
    fp = -torch.expm1(net.pa_head.log_survival(raw_pa.float(), k))[:, 0]
    fc = -torch.expm1(net.px_head.log_survival(raw_px[:, HALF, HALF].float(), k))[:, 0]
    return fp, fc


def integrated_gradients(net, x, refs, doy, k, steps: int):
    """x (B, T, C, 13, 13) normiert, refs Liste von Referenzen gleicher Form -> A je Kopf (gemittelt ueber die
    Referenzen), F(x), mittleres F(Referenz) je Kopf (B,). Pfad b + a (x - b): mit b = 0 waere er eine reine
    Skalierung, und die GroupNorm im Stem macht das Netz nahezu skaleninvariant - die Gradienten entlang des Pfads
    verschwinden dann und sum A erklaert F(x) - F(0) nicht (Rauchtest: Fehler ~1)."""
    A = {h: torch.zeros_like(x) for h in HEADS}
    f0 = {h: torch.zeros(x.shape[0], device=x.device) for h in HEADS}
    for b in refs:
        g = {h: torch.zeros_like(x) for h in HEADS}
        for s in range(steps):
            xi = (b + (x - b) * ((s + 0.5) / steps)).requires_grad_(True)
            fp, fc = scores(net, xi, doy, k)
            g["patch"] += torch.autograd.grad(fp.sum(), xi, retain_graph=True)[0]
            g["centre"] += torch.autograd.grad(fc.sum(), xi)[0]
        with torch.no_grad():
            fb = scores(net, b, doy, k)
        for h, v in zip(HEADS, fb):
            A[h] += (x - b) * g[h] / (steps * len(refs))
            f0[h] += v / len(refs)
    with torch.no_grad():
        f1 = scores(net, x, doy, k)
    return A, dict(zip(HEADS, f1)), f0


def run(args, cfg) -> None:
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = exp / "metrics" / OUT / f"attribution{args.tag}.npz"
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
    rows = rows[rows < fz.shape[0]]
    pool = rows.copy()                                                 # Referenzen: alle Test-Stichproben
    y_all = targets.labels(d, args.target).astype(bool)
    rng = np.random.default_rng(args.seed)
    # Positive ueberrepraesentiert (Haelfte der Stichprobe), damit neue Zuendungen nicht zu wenige sind
    pos, neg = rows[y_all[rows]], rows[~y_all[rows]]
    npos = min(args.n // 2, pos.size)
    rows = np.sort(np.concatenate([rng.choice(pos, npos, replace=False),
                                   rng.choice(neg, min(args.n - npos, neg.size), replace=False)]))
    y = y_all[rows]
    fire_t = targets.fire_map(d, X, args.target)[rows]
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    files = [f for fl in model_files(args.models).values() for f in fl][:args.max_seeds]
    names = list(model_files(args.models))
    if len(names) != 1:
        raise SystemExit("[attr] genau ein Modell je Aufruf (--models NAME=GLOB)")
    name = names[0]
    store: dict[str, np.ndarray] = {"rows": rows, "label": y, "fire_target": fire_t}
    meta: dict = {"name": name, "files": files, "steps": args.steps, "n_refs": args.n_refs, "dts": list(args.dts), "n": int(rows.size),
                  "n_pos": int(y.sum()), "channels": list(CHANNEL_NAMES), "completeness": {}}
    for dt_ in args.dts:
        j = list(DTS).index(dt_)
        e = L - dt_ - 2
        k = torch.tensor([float(dt_ + 1)], device=device)
        ign = np.asarray(X[rows, e - 29:e + 1, FIRE_CH]).reshape(rows.size, -1).max(1) == 0
        G = groups(y, ign)
        store[f"ign30|dt{dt_}"] = ign
        store[f"fire_days|dt{dt_}"] = np.asarray(X[rows, e - 29:e + 1, FIRE_CH] > 0).sum(1).astype(np.int16)
        acc = {h: {"ch_abs": [], "ch_sig": [], "space": {g: 0.0 for g in G}, "time": {g: 0.0 for g in G}} for h in HEADS}
        comp = {h: [] for h in HEADS}
        ex_idx = np.concatenate([np.where(y & ~ign)[0][:args.n_examples // 2], np.where(y & ign)[0][:args.n_examples // 2]])
        examples = {h: [] for h in HEADS}
        for si, f in enumerate(files):
            t0 = time.time()
            net = load_net(f, device)
            for p_ in net.parameters():
                p_.requires_grad_(False)
            use_feat = msv2.needs_features(net)
            ref_rng = np.random.default_rng(args.seed + 1)             # gleiche Referenzen fuer alle Seeds
            ch_abs = {h: [] for h in HEADS}
            ch_sig = {h: [] for h in HEADS}
            space = {h: [] for h in HEADS}
            time_ = {h: [] for h in HEADS}
            ex = {h: {} for h in HEADS}
            for a in range(0, rows.size, args.bs):
                r = rows[a:a + args.bs]
                x = (windows(Xg, r, np.full(r.size, e)).float() - m_t) / s_t
                x = x[:, -net.hist:]
                if use_feat:
                    net.set_features(torch.from_numpy(np.asarray(fz[r, j], np.float32)).to(device))
                doy = torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device)
                refs = []
                for _ in range(args.n_refs):
                    rb = ref_rng.choice(pool, r.size)
                    refs.append(((windows(Xg, rb, np.full(rb.size, e)).float() - m_t) / s_t)[:, -net.hist:])
                A, f1, f0 = integrated_gradients(net, x, refs, doy, k, args.steps)
                for h in HEADS:
                    Ah = A[h]
                    comp[h].append(torch.stack([Ah.sum((1, 2, 3, 4)), f1[h] - f0[h]], 1).cpu().numpy())
                    ab = Ah.abs()
                    ch_abs[h].append(ab.sum((1, 3, 4)).cpu().numpy())                       # (B, C)
                    ch_sig[h].append(Ah.sum((1, 3, 4)).cpu().numpy())
                    space[h].append(ab.sum(1).flatten(2).cpu().numpy().astype(np.float16))  # (B, C, 169)
                    time_[h].append(ab.sum((3, 4)).cpu().numpy().astype(np.float16))       # (B, T, C)
                    if si == 0:
                        for b in np.where(np.isin(np.arange(a, a + r.size), ex_idx))[0]:
                            ex[h][a + b] = Ah[b].flatten(2).cpu().numpy().astype(np.float16)    # (T, C, 169)
            for h in HEADS:
                ch_abs[h] = np.concatenate(ch_abs[h]); ch_sig[h] = np.concatenate(ch_sig[h])
                sp = np.concatenate(space[h]).astype(np.float32); tm = np.concatenate(time_[h]).astype(np.float32)
                acc[h]["ch_abs"].append(ch_abs[h]); acc[h]["ch_sig"].append(ch_sig[h])
                for g, msk in G.items():
                    if msk.sum():
                        acc[h]["space"][g] = acc[h]["space"][g] + sp[msk].mean(0) / len(files)
                        acc[h]["time"][g] = acc[h]["time"][g] + tm[msk].mean(0) / len(files)
                if si == 0:
                    examples[h] = [ex[h][i] for i in ex_idx]
            del net
            torch.cuda.empty_cache()
            c = np.concatenate(comp["patch"][-int(np.ceil(rows.size / args.bs)):])
            rel = np.abs(c[:, 0] - c[:, 1]).sum() / max(np.abs(c[:, 1]).sum(), 1e-12)
            print(f"[attr] {name} {Path(f).name} dt={dt_}: {rows.size} Stichproben in {time.time() - t0:.0f} s, "
                  f"Vollstaendigkeit Ausschnitt rel. Fehler {rel:.3f}", flush=True)
        for h in HEADS:
            c = np.concatenate(comp[h])
            meta["completeness"][f"{h}|dt{dt_}"] = {
                "rel_abs_error": float(np.abs(c[:, 0] - c[:, 1]).sum() / max(np.abs(c[:, 1]).sum(), 1e-12)),
                "corr": float(np.corrcoef(c[:, 0], c[:, 1])[0, 1]) if c.shape[0] > 1 else float("nan")}
            store[f"{h}|dt{dt_}|resid"] = (c[:, 1] - c[:, 0]).reshape(len(files), -1).mean(0).astype(np.float32)
            store[f"{h}|dt{dt_}|ch_abs"] = np.mean(acc[h]["ch_abs"], 0).astype(np.float32)   # (n, C)
            store[f"{h}|dt{dt_}|ch_sig"] = np.mean(acc[h]["ch_sig"], 0).astype(np.float32)
            for gi, g in enumerate(G):
                store[f"{h}|dt{dt_}|space|g{gi}"] = np.asarray(acc[h]["space"][g], np.float32)  # (C, 169)
                store[f"{h}|dt{dt_}|time|g{gi}"] = np.asarray(acc[h]["time"][g], np.float32)    # (T, C)
            if examples[h]:
                store[f"{h}|dt{dt_}|examples"] = np.stack(examples[h])                            # (E, T, C, 169)
        store[f"examples_idx|dt{dt_}"] = ex_idx
    np.savez_compressed(out, **store)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    write_manifest(cfg, f"p44_{OUT}{args.tag}", args.config, {"output": str(out), **meta})
    print(f"[attr] -> {out}", flush=True)


# ------------------------------------------------------------------ Abbildungen
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
GROUP_COLORS = dict(zip(GROUPS, ("#2a78d6", "#eb6834", "#1baf7a")))


def plot(paths: list[Path], fig_dir: Path, head: str = "patch") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blue = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    orange = LinearSegmentedColormap.from_list("seq_orange", ["#fcfcfb", "#f7d2c0", "#f19a72", "#eb6834", "#a8441d"])
    div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ee8f8e", "#b52f2f"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK, "axes.titlecolor": INK})
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = [(json.loads(p.with_suffix(".json").read_text()), dict(np.load(p))) for p in paths]
    C = list(CHANNEL_NAMES)

    def save(fig, nm):
        p = fig_dir / f"{OUT}_{head}_{nm}.png"
        fig.savefig(p, dpi=140, facecolor=SURF, bbox_inches="tight")
        plt.close(fig)
        print(f"[attr] {p}", flush=True)

    def clean(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def share(S, dt_, msk, resid: bool = False):
        """Anteil je Kanal; mit resid zusaetzlich die Restluecke F(x) - F(b) - sum A als Spalte C (bei Contagion /
        Multi-State der Beitrag des beobachteten Feuers ueber die Schwelle, sonst Diskretisierungsfehler)."""
        v = S[f"{head}|dt{dt_}|ch_abs"][msk]
        if resid:
            v = np.concatenate([v, np.abs(S[f"{head}|dt{dt_}|resid"][msk])[:, None]], 1)
        return (v / np.maximum(v.sum(1, keepdims=True), 1e-12)).mean(0)

    dts = runs[0][0]["dts"]
    # 1 Kanal-Anteile je Modell und dt
    fig, axes = plt.subplots(len(dts), len(runs), figsize=(4.6 * len(runs), 6.2 * len(dts)), squeeze=False, facecolor=SURF)
    for i, dt_ in enumerate(dts):
        for jx, (m, S) in enumerate(runs):
            ax = axes[i, jx]; clean(ax)
            G = groups(S["label"], S[f"ign30|dt{dt_}"])
            sh = {g: share(S, dt_, msk, resid=True) for g, msk in G.items() if msk.sum()}
            order = np.argsort(-sh[GROUPS[0]])[:20][::-1]
            lab = C + ["fire: spread path (residual)"]
            hh = 0.8 / len(sh)
            for kk, (g, v) in enumerate(sh.items()):
                ax.barh(np.arange(order.size) + (kk - (len(sh) - 1) / 2) * hh, v[order] * 100, hh * 0.9,
                        color=GROUP_COLORS[g], label=f"{g} (n={int(G[g].sum())})")
            ax.set_yticks(np.arange(order.size), [lab[c] for c in order], fontsize=8)
            ax.set_xlabel("Share of total |attribution| (%)")
            comp = m["completeness"][f"{head}|dt{dt_}"]["rel_abs_error"]
            ax.set_title(f"{m['name']} · Δt = {dt_}  (completeness error {comp:.2f})", loc="left", fontsize=9)
            ax.grid(axis="x", alpha=0.25)
        axes[i, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"Which channels carry the prediction ({head} head, Integrated Gradients)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, "channels")

    for m, S in runs:
        nm = m["name"]
        # 2 Karten je Kanal: 8 wichtigste Kanaele + alle Kanaele zusammen
        rows_ = [(dt_, gi) for dt_ in dts for gi in (0, 2)]
        top = np.argsort(-share(S, dts[0], S["label"]))[:8]
        fig, axes = plt.subplots(len(rows_), top.size + 1, figsize=(1.9 * (top.size + 1), 2.1 * len(rows_)),
                                 squeeze=False, facecolor=SURF)
        for i, (dt_, gi) in enumerate(rows_):
            sp = S[f"{head}|dt{dt_}|space|g{gi}"]                                     # (C, 169)
            vmax = max(float(sp[top].max()), 1e-12)
            for jx in range(top.size + 1):
                ax = axes[i, jx]
                img = (sp.sum(0) if jx == top.size else sp[top[jx]]).reshape(P, P)
                ax.imshow(img, cmap=blue, vmin=0, vmax=img.max() if jx == top.size else vmax)
                ax.plot(HALF, HALF, "+", color=INK, ms=5)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title("all channels" if jx == top.size else C[top[jx]], fontsize=8)
            axes[i, 0].set_ylabel(f"Δt={dt_}\n{GROUPS[gi]}", fontsize=7)
        fig.suptitle(f"{nm}: where each channel matters (|IG| summed over days; channels share one scale per row)",
                     x=0.01, ha="left", color=INK)
        fig.tight_layout()
        save(fig, f"{nm}_space")

        # 3 Kanal x Tag
        top15 = np.argsort(-share(S, dts[0], S["label"]))[:15]
        fig, axes = plt.subplots(1, len(rows_), figsize=(4.0 * len(rows_), 4.4), squeeze=False, facecolor=SURF)
        for i, (dt_, gi) in enumerate(rows_):
            tm = S[f"{head}|dt{dt_}|time|g{gi}"][:, top15].T                          # (15, T)
            ax = axes[0, i]
            ax.imshow(tm, cmap=blue, aspect="auto", vmin=0)
            ax.set_yticks(np.arange(top15.size), [C[c] for c in top15] if i == 0 else [], fontsize=7)
            T = tm.shape[1]
            ax.set_xticks([0, T // 2, T - 1], [f"{-T + 1}", f"{-T + 1 + T // 2}", "0"])
            ax.set_xlabel("Day relative to window end")
            ax.set_title(f"Δt={dt_} · {GROUPS[gi]}", fontsize=8, loc="left")
        fig.suptitle(f"{nm}: when each channel matters (|IG| summed over the patch)", x=0.01, ha="left", color=INK)
        fig.tight_layout()
        save(fig, f"{nm}_time")

        # 4 Beispiele mit Vorzeichen
        dt_ = dts[0]
        if f"{head}|dt{dt_}|examples" in S:
            E = S[f"{head}|dt{dt_}|examples"].astype(np.float32)                    # (E, T, C, 169)
            idx = S[f"examples_idx|dt{dt_}"]
            fig, axes = plt.subplots(len(idx), 8, figsize=(16, 2.1 * len(idx)), squeeze=False, facecolor=SURF)
            for i, s in enumerate(idx):
                sp = E[i].sum(0)                                                     # (C, 169) mit Vorzeichen
                tc = np.argsort(-np.abs(sp).sum(1))[:6]
                axes[i, 0].imshow(S[f"fire_days|dt{dt_}"][s], cmap=orange, vmin=0)
                axes[i, 1].imshow(S["fire_target"][s].astype(float), cmap=orange, vmin=0, vmax=1)
                lim = max(float(np.abs(sp[tc]).max()), 1e-12)
                for jx, c in enumerate(tc):
                    axes[i, jx + 2].imshow(sp[c].reshape(P, P), cmap=div, vmin=-lim, vmax=lim)
                    axes[i, jx + 2].set_title(C[c], fontsize=7)
                for ax in axes[i]:
                    ax.set_xticks([]); ax.set_yticks([]); ax.plot(HALF, HALF, "+", color=INK, ms=5)
                axes[i, 0].set_ylabel("new ignition" if S[f"ign30|dt{dt_}"][s] else "fire in window", fontsize=7)
                if i == 0:
                    axes[i, 0].set_title("fire days in window", fontsize=7)
                    axes[i, 1].set_title("fire on target day", fontsize=7)
            fig.suptitle(f"{nm}, Δt = {dt_}: signed attribution of the 6 strongest channels per example "
                         "(red raises risk, blue lowers it)", x=0.01, ha="left", color=INK)
            fig.tight_layout()
            save(fig, f"{nm}_examples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=[], help="genau ein NAME=GLOB")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap.add_argument("--feat-tag", default="")
    ap.add_argument("--dts", type=int, nargs="+", default=[1, 30])
    ap.add_argument("--n", type=int, default=1200, help="Stichproben, Haelfte Positive")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--n-refs", type=int, default=4, help="Referenz-Stichproben je Stichprobe")
    ap.add_argument("--max-seeds", type=int, default=3)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--tags", nargs="*", default=[], help="mit --plot-only: welche attribution<tag>.npz")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if args.plot_only:
        paths = [exp / "metrics" / OUT / f"attribution{t}.npz" for t in args.tags]
        for h in HEADS:
            plot(paths, exp / "figures", h)
    else:
        run(args, cfg)


if __name__ == "__main__":
    main()

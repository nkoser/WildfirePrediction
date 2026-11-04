"""Attention-Karten der STS-Modelle im FireMPC-Protokoll (Test-Split), ohne Eingriff in die Modellklassen.

Die Bloecke rufen nn.MultiheadAttention mit need_weights=False auf. Ein Pre-Hook setzt need_weights=True
(gemittelt ueber die Koepfe), ein Forward-Hook liest die Gewichte und verdichtet sie sofort:

  raeumlich   Block-Attention ueber die 169 Pixel eines Tages; Anfrage = Mittelpixel (6, 6) am letzten
              Fenstertag -> Karte 13 x 13; dazu die mittlere Attention-Distanz (Pixel) aller Anfragen am letzten Tag
  zeitlich    Block-Attention ueber die 30 Tage eines Pixels; Anfrage = letzter Tag -> Profil ueber 30 Tage,
              fuer das Mittelpixel und gemittelt ueber alle Pixel
  Pooling     nur STS-LH: Attention-Pooling des Ausschnittskopfs (CLS -> 169 Pixel) -> Karte 13 x 13

Gemittelt je Modell (alle Seeds), dt, Teilmenge (all, ign30 = kein Feuer im Ausschnitt im Fenster) und Label
(Feuer im Ausschnitt am Zieltag ja/nein); dazu Beispiele. Attention zeigt, wohin das Modell schaut, nicht, was
die Vorhersage traegt - das misst importance_v2.

  python -m wildfire.paper.attention_maps --config ... --models lhb10=GLOB msv2=GLOB cg=GLOB [--tag _x]
  -> metrics/attention_maps/attention<tag>.npz, figures/attention_maps<tag>_*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import multistate_v2 as msv2
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.msv2_eval import load_net, model_files
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import doy_of

P, HALF = 13, 6
C_IDX = HALF * P + HALF
GRID = np.stack(np.meshgrid(np.arange(P), np.arange(P), indexing="ij"), -1).reshape(-1, 2).astype(np.float32)
DIST = np.sqrt(((GRID[:, None] - GRID[None]) ** 2).sum(-1))                     # (169, 169) Pixel
SUBS = ("all", "ign30")


class Recorder:
    """Hooks an alle Attention-Module eines Netzes; sammelt verdichtete Gewichte je Batch."""

    def __init__(self, net, device):
        self.B, self.T = 0, 0
        self.rec: dict[str, list] = {}
        self.dist = torch.from_numpy(DIST).to(device)
        self.handles = []
        for i, blk in enumerate(net.blocks):
            if getattr(blk, "attn", "factorized") != "factorized":
                raise SystemExit(f"[attn] Block {i}: nur faktorisierte Attention unterstuetzt ({blk.attn})")
            self._hook(blk.sa, f"sp{i}")
            self._hook(blk.ta, f"tp{i}")
        if isinstance(getattr(net, "pool", None), nn.MultiheadAttention):
            self._hook(net.pool, "pool")
        self.n_layers = len(net.blocks)

    def _hook(self, mod, key):
        def pre(_m, a, kw):
            kw["need_weights"] = True
            kw["average_attn_weights"] = True
            return a, kw

        def post(_m, _a, _kw, out):
            self._store(key, out[1].float())
        self.handles += [mod.register_forward_pre_hook(pre, with_kwargs=True),
                         mod.register_forward_hook(post, with_kwargs=True)]

    def _store(self, key, w):
        B, T, N = self.B, self.T, P * P
        if key.startswith("sp"):
            w = w.view(B, T, N, N)[:, -1]                                              # letzter Tag (B, N, N)
            self.rec.setdefault(key + "_centre", []).append(w[:, C_IDX].cpu().numpy())
            self.rec.setdefault(key + "_dist", []).append((w * self.dist).sum(-1).mean(-1).cpu().numpy())
        elif key.startswith("tp"):
            w = w.view(B, N, T, T)[:, :, -1]                                            # Anfrage letzter Tag (B, N, T)
            self.rec.setdefault(key + "_centre", []).append(w[:, C_IDX].cpu().numpy())
            self.rec.setdefault(key + "_all", []).append(w.mean(1).cpu().numpy())
        else:
            self.rec.setdefault("pool", []).append(w[:, 0].cpu().numpy())

    def take(self) -> dict[str, np.ndarray]:
        out = {k: np.concatenate(v) for k, v in self.rec.items()}
        self.rec = {}
        return out

    def close(self):
        for h in self.handles:
            h.remove()


def run(args, cfg) -> None:
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = exp / "metrics" / "attention_maps" / f"attention{args.tag}.npz"
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
    rows = np.sort(np.random.default_rng(args.seed).choice(rows, min(args.max_n, rows.size), replace=False))
    y = targets.labels(d, args.target)[rows].astype(bool)
    fire_t = targets.fire_map(d, X, args.target)[rows]
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    ign = {}
    for dt_ in args.dts:
        e = L - dt_ - 2
        ign[dt_] = np.asarray(X[rows, e - 29:e + 1, FIRE_CH]).reshape(rows.size, -1).max(1) == 0

    store: dict[str, np.ndarray] = {"rows": rows, "label": y, "fire_target": fire_t}
    for dt_ in args.dts:
        e = L - dt_ - 2
        store[f"ign30|dt{dt_}"] = ign[dt_]
        store[f"fire_days|dt{dt_}"] = np.asarray(X[rows, e - 29:e + 1, FIRE_CH] > 0).sum(1).astype(np.int16)

    n_layers = {}
    for name, files in model_files(args.models).items():
        per_seed: dict[int, list[dict]] = {dt_: [] for dt_ in args.dts}
        for f in files[:args.max_seeds]:
            net = load_net(f, device)
            rec = Recorder(net, device)
            n_layers[name] = rec.n_layers
            use_feat = msv2.needs_features(net)
            for dt_ in args.dts:
                j = list(DTS).index(dt_)
                e = L - dt_ - 2
                with torch.no_grad():
                    for a in range(0, rows.size, args.bs):
                        r = rows[a:a + args.bs]
                        x = (windows(Xg, r, np.full(r.size, e)).float() - m_t) / s_t
                        if use_feat:
                            net.set_features(torch.from_numpy(np.asarray(fz[r, j], np.float32)).to(device))
                        rec.B, rec.T = r.size, net.hist
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            net(x, torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device))
                per_seed[dt_].append(rec.take())
            rec.close()
            del net
            torch.cuda.empty_cache()
            print(f"[attn] {name} {Path(f).name}: {rows.size} Stichproben, dt {list(args.dts)}", flush=True)
        for dt_ in args.dts:
            for k in per_seed[dt_][0]:
                v = np.mean([s[k] for s in per_seed[dt_]], 0)                      # Mittel ueber Seeds je Stichprobe
                store[f"{name}|dt{dt_}|{k}"] = v.astype(np.float32)
                if (k.startswith("sp") and k.endswith("_centre")) or k == "pool":
                    store[f"{name}|dt{dt_}|{k}|seed0"] = per_seed[dt_][0][k].astype(np.float32)
    np.savez_compressed(out, **store)
    meta = {"models": {n: fl[:args.max_seeds] for n, fl in model_files(args.models).items()}, "n_layers": n_layers,
            "dts": list(args.dts), "n": int(rows.size), "n_pos": int(y.sum()),
            "n_ign30": {str(k): int(v.sum()) for k, v in ign.items()},
            "n_ign30_pos": {str(k): int((v & y).sum()) for k, v in ign.items()}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    write_manifest(cfg, f"p43_attention_maps{args.tag}", args.config, {"output": str(out), **meta})
    print(f"[attn] -> {out}", flush=True)
    plot(out, exp / "figures", args.tag)


# ------------------------------------------------------------------ Abbildungen
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
GROUP_COLORS = {"all · fire next day": "#2a78d6", "all · no fire": "#eb6834", "new ignition · fire next day": "#1baf7a"}


def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    blue = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    orange = LinearSegmentedColormap.from_list("seq_orange", ["#fcfcfb", "#f7d2c0", "#f19a72", "#eb6834", "#a8441d"])
    return blue, orange


def _groups(S, dt_):
    y, ig = S["label"], S[f"ign30|dt{dt_}"]
    return {"all · fire next day": y, "all · no fire": ~y, "new ignition · fire next day": ig & y}


def plot(npz: Path, fig_dir: Path, tag: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = dict(np.load(npz))
    meta = json.loads(npz.with_suffix(".json").read_text())
    models, dts, nl = list(meta["n_layers"]), meta["dts"], meta["n_layers"]
    blue, orange = _cmaps()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK, "axes.titlecolor": INK})
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, nm):
        p = fig_dir / f"attention_maps{tag}_{nm}.png"
        fig.savefig(p, dpi=140, facecolor=SURF, bbox_inches="tight")
        plt.close(fig)
        print(f"[attn] {p}", flush=True)

    def clean(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # 1 zeitlich: Anfrage letzter Tag, Mittelpixel, letzter Block
    fig, axes = plt.subplots(len(dts), len(models), figsize=(4.2 * len(models), 3.2 * len(dts)), squeeze=False,
                             facecolor=SURF, sharey=True)
    days = np.arange(-29, 1)
    for i, dt_ in enumerate(dts):
        for jx, m in enumerate(models):
            ax = axes[i, jx]; clean(ax)
            v = S[f"{m}|dt{dt_}|tp{nl[m] - 1}_centre"]
            for g, msk in _groups(S, dt_).items():
                if msk.sum():
                    ax.plot(days[-v.shape[1]:], v[msk].mean(0), lw=2, color=GROUP_COLORS[g], label=f"{g} (n={msk.sum()})")
            ax.axhline(1 / v.shape[1], color=INK2, lw=0.8, ls=":")
            ax.set_title(f"{m} · Δt = {dt_}", loc="left", fontsize=10)
            ax.set_xlabel("Day relative to window end")
            if jx == 0:
                ax.set_ylabel("Temporal attention (last block)")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Where in time the centre pixel looks (query = last window day; dotted = uniform)", x=0.01, ha="left",
                 color=INK)
    fig.tight_layout()
    save(fig, "temporal")

    # 2 raeumlich: Anfrage Mittelpixel, letzter Tag, letzter Block (relativ zu gleichverteilt)
    def map_grid(key_of, title, nm, models_):
        cols = [(dt_, g) for dt_ in dts for g in GROUP_COLORS]
        fig, axes = plt.subplots(len(models_), len(cols), figsize=(2.3 * len(cols), 2.5 * len(models_)),
                                 squeeze=False, facecolor=SURF)
        for i, m in enumerate(models_):
            for jx, (dt_, g) in enumerate(cols):
                ax = axes[i, jx]
                msk = _groups(S, dt_)[g]
                v = S[key_of(m, dt_)]
                img = (v[msk].mean(0) if msk.sum() else np.zeros(P * P)).reshape(P, P) * P * P
                im = ax.imshow(img, cmap=blue, vmin=0, vmax=max(img.max(), 1.0))
                ax.plot(HALF, HALF, "+", color=INK, ms=6)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{m} · Δt={dt_}\n{g}", fontsize=7, loc="left")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cb.ax.tick_params(labelsize=6)
        fig.suptitle(title + " (colour = × uniform; + = centre pixel)", x=0.01, ha="left", color=INK)
        fig.tight_layout()
        save(fig, nm)

    map_grid(lambda m, dt_: f"{m}|dt{dt_}|sp{nl[m] - 1}_centre",
             "Spatial attention of the centre pixel on the last window day (last block)", "spatial", models)
    pooled = [m for m in models if f"{m}|dt{dts[0]}|pool" in S]
    if pooled:
        map_grid(lambda m, dt_: f"{m}|dt{dt_}|pool", "Patch-head attention pooling over pixels", "pool", pooled)

    # 3 mittlere raeumliche Attention-Distanz je Block
    fig, axes = plt.subplots(1, len(dts), figsize=(5.2 * len(dts), 3.4), squeeze=False, facecolor=SURF, sharey=True)
    cat = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for i, dt_ in enumerate(dts):
        ax = axes[0, i]; clean(ax)
        for k, m in enumerate(models):
            v = [S[f"{m}|dt{dt_}|sp{b}_dist"].mean() for b in range(nl[m])]
            ax.plot(np.arange(1, nl[m] + 1), v, "-o", lw=2, ms=8, color=cat[k % len(cat)], label=m)
        ax.axhline(DIST.mean(), color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("Block"); ax.set_title(f"Δt = {dt_}", loc="left")
        ax.set_xticks(np.arange(1, max(nl.values()) + 1))
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Mean spatial attention distance (pixels)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("How far pixels look in space, per block (dotted = uniform)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, "distance")

    # 4 Beispiele (Seed 0): Feuer-Historie, Ziel, Attention je Modell
    dt_ = dts[0]
    y, ig = S["label"], S[f"ign30|dt{dt_}"]
    rng = np.random.default_rng(0)
    pick = [*rng.choice(np.where(y & ~ig)[0], min(3, int((y & ~ig).sum())), replace=False),
            *rng.choice(np.where(y & ig)[0], min(3, int((y & ig).sum())), replace=False)]
    cols = ["fire days in window", "fire on target day"] + [f"{m}: spatial" for m in models] + [f"{m}: pool" for m in pooled]
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(2.1 * len(cols), 2.2 * len(pick)), squeeze=False,
                             facecolor=SURF)
    for i, s in enumerate(pick):
        imgs = [(S[f"fire_days|dt{dt_}"][s], orange), (S["fire_target"][s].astype(float), orange)]
        imgs += [(S[f"{m}|dt{dt_}|sp{nl[m] - 1}_centre|seed0"][s].reshape(P, P) * P * P, blue) for m in models]
        imgs += [(S[f"{m}|dt{dt_}|pool|seed0"][s].reshape(P, P) * P * P, blue) for m in pooled]
        for jx, (img, cm) in enumerate(imgs):
            ax = axes[i, jx]
            ax.imshow(img, cmap=cm, vmin=0)
            ax.plot(HALF, HALF, "+", color=INK, ms=5)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[jx], fontsize=7)
        axes[i, 0].set_ylabel("new ignition" if ig[s] else "fire in window", fontsize=7)
    fig.suptitle(f"Examples, Δt = {dt_}, seed 0", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, "examples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", default=[], help="NAME=GLOB")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap.add_argument("--feat-tag", default="")
    ap.add_argument("--dts", type=int, nargs="+", default=[1, 30])
    ap.add_argument("--max-n", type=int, default=6000)
    ap.add_argument("--max-seeds", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--plot-only", action="store_true", help="nur Abbildungen aus vorhandenem npz")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    if args.plot_only:
        plot(exp / "metrics" / "attention_maps" / f"attention{args.tag}.npz", exp / "figures", args.tag)
    else:
        run(args, cfg)


if __name__ == "__main__":
    main()

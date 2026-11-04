"""Randdiagnose, GPU: Haengt das Risiko des Pixelkopfs von der Position im 13 x 13-Ausschnitt ab?

Anlass: Attention und Integrated Gradients (Treiberanalyse) zeigen mehr Gewicht an Raendern und Ecken. Der Stem
(2 x Conv 3 x 3, Zero-Padding, sts_dctm.py) gibt den aeusseren zwei Ringen kuenstliche Nachbarn (normiert 0 =
Kanalmittel). Zwei Tests, Score F(30) des Pixelkopfs, roh (ohne Kalibrierung), Mittel ueber die Seeds:
  oe     O/E-Karte je Position: --n-oe zufaellige Test-Stichtage aus surv_eval_test (2023-2025), je Position das
         Label "erstes Feuer bis 30 Tage" (pixel_events aus den MODIS-Rohdaten, wie surv_eval), nur Landpixel.
         Die Stichtage liegen zufaellig im Raum, die beobachtete Rate ist deshalb je Position gleich (bis auf
         Rauschen); eine Struktur in der mittleren Vorhersage ist ein Positionseffekt des Modells.
  shift  Verschiebungstest: --n-shift Bodenpixel (Land, Abstand >= 12 Pixel zum Gitterrand; Haelfte mit Feuer im
         Ausschnitt-Fenster, Haelfte ohne). Je Bodenpixel 169 Fenster aus einem 25 x 25-Kasten, sodass derselbe
         Bodenpixel an jeder Position (i, j) sitzt; F30 am Bodenpixel. Gleiche Wahrheit, nur die Position wandert.
         Kennzahlen je Position: mittleres log(F_p / F_Mitte), Faktor mean(F_p) / mean(F_Mitte), Anteil mit
         Faktor > 1.5 bzw. < 1/1.5.
Orientierung: Zeile 0 = Norden, Spalte 0 = Westen (Gitter EPSG:4326, transform[4] < 0).
Modelle (--models): sts_extreme-Checkpoints (lh_ref, cg_ref, lh_mono, cg_mono; Bias wie im Training aus dem
Checkpoint, wie black_swan_score), Ablationslaeufe mit beta = 1 (lhbce10, utae_bce10, convlstm_bce10; U-TAE und
ConvLSTM mit Reflect-Padding als Kontrolle ohne Zero-Padding; Bias = Standard von load_sts, nur relativ
lesen) und sts_inputs-Laeufe (name:inputs:pad, z. B. lh_reflect:all:reflect).
Ausgabe NEU: metrics/edge_diag<tag>/ (summary.md, maps.npz), figures/edge_diag<tag>/, Manifest p42_edge_diag<tag>.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from wildfire.audit.common import assert_readonly, load_config, write_manifest  # noqa: E402
from wildfire.data.dataset import CubeReader  # noqa: E402
from wildfire.paper import sts_extreme  # noqa: E402
from wildfire.paper.black_swan_score import event_box  # noqa: E402
from wildfire.paper.pixel_landmarks import pixel_events  # noqa: E402
from wildfire.paper.reconstruct import cache_norm  # noqa: E402
from wildfire.paper.surv_eval import raw_fire_keys  # noqa: E402
from wildfire.paper.surv_prepare import HALF, T_WIN  # noqa: E402

P = 2 * HALF + 1
TAU = 30.0
ABL = {"lhbce10": "abl_lhbce10", "utae_bce10": "abl_utae_bce10", "convlstm_bce10": "abl_convlstm_bce10"}
DEFAULT_MODELS = ["lh_ref", "cg_ref", "lh_mono", "cg_mono", "lhbce10", "utae_bce10", "convlstm_bce10"]


def load_models(spec: str, ck: Path, seeds, device):
    """-> Liste der Netze (eval) und Beschreibung."""
    nets = []
    if spec in ABL:
        from wildfire.paper.vision_backbones import load_any as vload
        for s in seeds:
            nets.append(vload(str(ck / f"sts_cache_G_next_cube_{ABL[spec]}_s{s}_s{s}.pt"), device))
        return nets, f"{ABL[spec]} (Bias Standard load_sts)"
    name, inputs, pad = (spec.split(":") + ["", ""])[:3] if ":" in spec else (spec, "", "")
    if inputs:
        from wildfire.paper.sts_inputs import build_masked
        sts_extreme.build = build_masked(inputs, pad or "zeros")
    try:
        for s in seeds:
            p = ck / f"sts_extreme_{name}_s{s}_s{s}.pt"
            c = torch.load(p, map_location="cpu")
            if c.get("args", {}).get("ctx", "none") != "none":
                raise SystemExit(f"ABBRUCH: {p.name} hat Zusatzeingaben (ctx)")
            net, _ = sts_extreme.load_any(str(p), c["init_px"], c["init_pa"], device)
            nets.append(net)
    finally:
        from wildfire.paper.sts_inputs import _ORIG_BUILD
        sts_extreme.build = _ORIG_BUILD
    return nets, f"sts_extreme_{name} (Bias wie im Training){' inputs ' + inputs if inputs else ''}{' pad ' + pad if pad else ''}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-oe", type=int, default=20000)
    ap.add_argument("--n-shift", type=int, default=2000)
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    mout, fout = exp / "metrics" / f"edge_diag{args.tag}", exp / "figures" / f"edge_diag{args.tag}"
    for o in (mout, fout):
        if o.exists():
            raise SystemExit(f"ABBRUCH: {o} existiert schon - anderen --tag waehlen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    norm = lambda x: (torch.from_numpy(np.ascontiguousarray(x)).to(device).float() - m_t) / s_t  # noqa: E731
    tau = torch.tensor([TAU], device=device)
    ck = exp / "checkpoints" / "paper"
    t0 = time.time()

    @torch.no_grad()
    def f30(net, x, doy):
        """-> F(30) je Pixel (B, 13, 13)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_px, _ = net(norm(x), torch.from_numpy(np.asarray(doy, np.float32)).to(device))
        raw_px = raw_px.float()
        B = raw_px.shape[0]
        ls = net.px_head.log_survival(raw_px.reshape(B * P * P, -1), tau)[:, 0]
        return (-torch.expm1(ls)).view(B, P, P).float().cpu().numpy()

    # ---- Stichtage
    rng = np.random.default_rng(args.seed)
    sd = exp / "derived_data" / args.surv_dir
    meta = dict(np.load(sd / "eval_meta.npz"))
    Xs = np.load(sd / "eval_X.npy", mmap_mode="r")
    n_all = meta["row"].size
    cube = CubeReader(exp)
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    keys, _ = raw_fire_keys(Path(cfg["dataset_root"]))
    io = np.sort(rng.choice(n_all, min(args.n_oe, n_all), replace=False))
    rr, cc, La = meta["row"][io], meta["col"][io], meta["L_abs"][io]
    ev, tt = pixel_events(keys, La, rr, cc, cube.H, cube.W)
    y_oe = ev & (tt <= TAU)
    dr = np.arange(-HALF, HALF + 1)
    R, C = rr[:, None, None] + dr[None, :, None], cc[:, None, None] + dr[None, None, :]
    ok = (R >= 0) & (R < cube.H) & (C >= 0) & (C < cube.W)
    land_oe = ok & land[np.clip(R, 0, cube.H - 1), np.clip(C, 0, cube.W - 1)]
    X_oe = np.asarray(Xs[io])
    print(f"[edge] O/E: {io.size:,} Stichtage, Feuer bis 30 d je Landpixel {y_oe[land_oe].mean():.4f} "
          f"({time.time() - t0:.0f} s)", flush=True)

    M = 2 * HALF                                            # Kasten 25 x 25: Bodenpixel +- 12
    cand = np.where((meta["row"] >= M) & (meta["row"] < cube.H - M) & (meta["col"] >= M) & (meta["col"] < cube.W - M)
                    & land[meta["row"], meta["col"]])[0]
    fw = meta["fire_win_patch"][cand].astype(bool)
    h = args.n_shift // 2
    is_ = np.concatenate([rng.choice(cand[fw], min(h, fw.sum()), replace=False),
                          rng.choice(cand[~fw], args.n_shift - min(h, fw.sum()), replace=False)])
    fire_sh = meta["fire_win_patch"][is_].astype(bool)
    print(f"[edge] Verschiebung: {is_.size:,} Bodenpixel, davon {fire_sh.sum():,} mit Feuer im Fenster", flush=True)
    ii, jj = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()

    res, maps = {}, {}
    for spec in args.models:
        nets, desc = load_models(spec, ck, args.seeds, device)
        key = spec.split(":")[0]
        t1 = time.time()
        # O/E
        F_oe = np.zeros((len(nets), io.size, P, P), np.float32)
        for a in range(0, io.size, args.bs):
            j = slice(a, a + args.bs)
            for k, net in enumerate(nets):
                F_oe[k, j] = f30(net, X_oe[j], meta["doy"][io][j])
        Fm = F_oe.mean(0)
        pred = np.where(land_oe, Fm, 0).sum(0) / land_oe.sum(0).clip(1)
        obs = np.where(land_oe, y_oe, 0).sum(0) / land_oe.sum(0).clip(1)
        # Verschiebung
        F_sh = np.zeros((len(nets), is_.size, P, P), np.float32)
        for q, r_ in enumerate(is_):
            L, r, c = int(meta["L_cube"][r_]), int(meta["row"][r_]), int(meta["col"][r_])
            box, li0, R0, C0, _ = event_box(cube, np.array([L]), np.array([r - HALF, r + HALF]), np.array([c - HALF, c + HALF]))
            li = cube._pos[L][1] - li0
            xb = box[li - T_WIN + 1:li + 1]                                    # (30, 55, 25, 25)
            x = np.stack([xb[:, :, M - i:M - i + P, M - j:M - j + P] for i, j in zip(ii, jj)])
            for k, net in enumerate(nets):
                Fp = f30(net, x, np.full(x.shape[0], meta["doy"][r_]))
                F_sh[k, q] = Fp[np.arange(P * P), ii, jj].reshape(P, P)
            if q % 500 == 0:
                print(f"[edge]   {key}: Verschiebung {q:,}/{is_.size:,} ({time.time() - t1:.0f} s)", flush=True)
        Fs = np.clip(F_sh.mean(0), 1e-7, 1)                                    # (n, 13, 13), Seed-Mittel
        lr = np.log(Fs) - np.log(Fs[:, HALF, HALF])[:, None, None]
        sub = {"alle": np.ones(is_.size, bool), "feuer": fire_sh, "ohne": ~fire_sh}
        sh = {s: {"logratio": lr[m].mean(0), "factor": Fs[m].mean(0) / Fs[m, HALF, HALF].mean(),
                  "up": (lr[m] > np.log(1.5)).mean(0), "down": (lr[m] < -np.log(1.5)).mean(0)} for s, m in sub.items()}
        ring = np.maximum(abs(ii - HALF), abs(jj - HALF)).reshape(P, P)
        inner = ring <= 3
        corners = {"NW": (0, 0), "NO": (0, P - 1), "SW": (P - 1, 0), "SO": (P - 1, P - 1)}
        r_ = {"desc": desc, "oe": {}, "shift": {}}
        rel = pred / pred[inner].mean()
        r_["oe"] = {"pred_mean": float(pred.mean()), "obs_mean": float(obs.mean()),
                    "oe_all": float(obs.sum() / pred.sum()),
                    "pred_rel_ring": [float(rel[ring == k].mean()) for k in range(HALF + 1)],
                    "oe_ring": [float(obs[ring == k].mean() / pred[ring == k].mean()) for k in range(HALF + 1)],
                    "obs_rel_ring": [float((obs / obs[inner].mean())[ring == k].mean()) for k in range(HALF + 1)],
                    "pred_rel_corner": {k: float(rel[v]) for k, v in corners.items()},
                    "pred_rel_side": {"N": float(rel[0, 1:-1].mean()), "S": float(rel[-1, 1:-1].mean()),
                                      "W": float(rel[1:-1, 0].mean()), "O": float(rel[1:-1, -1].mean())}}
        for s, d in sh.items():
            fac = d["factor"]
            r_["shift"][s] = {"factor_ring": [float(fac[ring == k].mean()) for k in range(HALF + 1)],
                              "geo_ring": [float(np.exp(d["logratio"][ring == k].mean())) for k in range(HALF + 1)],
                              "factor_corner": {k: float(fac[v]) for k, v in corners.items()},
                              "factor_side": {"N": float(fac[0, 1:-1].mean()), "S": float(fac[-1, 1:-1].mean()),
                                              "W": float(fac[1:-1, 0].mean()), "O": float(fac[1:-1, -1].mean())},
                              "share_up_ring": [float(d["up"][ring == k].mean()) for k in range(HALF + 1)],
                              "share_down_ring": [float(d["down"][ring == k].mean()) for k in range(HALF + 1)]}
        res[key] = r_
        maps[key] = {"pred": pred, "obs": obs, **{f"{s}_{q}": v for s, d in sh.items() for q, v in d.items()}}
        print(f"[edge] {key}: O/E {r_['oe']['oe_all']:.2f}; Vorhersage relativ je Ring 0..6 "
              + " ".join(f"{v:.2f}" for v in r_["oe"]["pred_rel_ring"]) + "; Verschiebung Faktor je Ring "
              + " ".join(f"{v:.2f}" for v in r_["shift"]["alle"]["factor_ring"])
              + f"; Ecken {r_['shift']['alle']['factor_corner']} ({time.time() - t1:.0f} s)", flush=True)
        del nets
        torch.cuda.empty_cache()

    # ---- Ausgaben
    mout.mkdir(parents=True)
    fout.mkdir(parents=True)
    np.savez(mout / "maps.npz", **{f"{m}__{k}": v for m, d in maps.items() for k, v in d.items()},
             oe_rows=io, shift_rows=is_)
    md = ["# Randdiagnose: Positionseffekt des Pixelkopfs (F30 roh, Seed-Mittel)", "",
          "Zeile 0 = Norden, Spalte 0 = Westen. Ring k = max(|i-6|, |j-6|); Ring 6 = aeusserster Rand.", "",
          f"O/E: {io.size:,} zufaellige Test-Stichtage 2023-2025, nur Landpixel. Verschiebung: {is_.size:,} Bodenpixel "
          f"({fire_sh.sum():,} mit Feuer im Ausschnitt-Fenster).", "",
          "## O/E-Karte: mittlere Vorhersage je Ring relativ zum Inneren (Ring <= 3); Beobachtung zum Vergleich", "",
          "| Modell | O/E gesamt | Vorhersage rel. Ring 0..6 | Beobachtung rel. Ring 0..6 | O/E je Ring 0..6 | Ecken NW/NO/SW/SO | Seiten N/S/W/O |",
          "|---|---|---|---|---|---|---|"]
    for m, r_ in res.items():
        o = r_["oe"]
        md.append(f"| {m} | {o['oe_all']:.2f} | " + " ".join(f"{v:.2f}" for v in o["pred_rel_ring"]) + " | "
                  + " ".join(f"{v:.2f}" for v in o["obs_rel_ring"]) + " | " + " ".join(f"{v:.2f}" for v in o["oe_ring"])
                  + " | " + "/".join(f"{v:.2f}" for v in o["pred_rel_corner"].values()) + " | "
                  + "/".join(f"{v:.2f}" for v in o["pred_rel_side"].values()) + " |")
    for s, lab in (("alle", "alle Bodenpixel"), ("feuer", "mit Feuer im Fenster"), ("ohne", "ohne Feuer im Fenster")):
        md += ["", f"## Verschiebung, {lab}: Faktor mean(F_p) / mean(F_Mitte)", "",
               "| Modell | Faktor Ring 0..6 | geom. Mittel Ring 0..6 | Anteil > 1.5x Ring 6 | Anteil < 1/1.5 Ring 6 | Ecken NW/NO/SW/SO | Seiten N/S/W/O |",
               "|---|---|---|---|---|---|---|"]
        for m, r_ in res.items():
            d = r_["shift"][s]
            md.append(f"| {m} | " + " ".join(f"{v:.2f}" for v in d["factor_ring"]) + " | "
                      + " ".join(f"{v:.2f}" for v in d["geo_ring"]) + f" | {d['share_up_ring'][-1]:.2f} | "
                      f"{d['share_down_ring'][-1]:.2f} | " + "/".join(f"{v:.2f}" for v in d["factor_corner"].values())
                      + " | " + "/".join(f"{v:.2f}" for v in d["factor_side"].values()) + " |")
    (mout / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)

    # Abbildung: je Modell eine Zeile: Vorhersage rel., Beobachtung rel., Verschiebung alle / Feuer / ohne
    plt.rcParams["font.family"] = "DejaVu Sans"
    cols = [("pred", "Mean prediction F30\n(rel. to inner 7×7)"), ("obs", "Observed rate\n(rel. to inner 7×7)"),
            ("alle_factor", "Shift test, all\nmean F(p) / mean F(centre)"),
            ("feuer_factor", "Shift test, fire in window"), ("ohne_factor", "Shift test, no fire in window")]
    fig, axs = plt.subplots(len(maps), len(cols), figsize=(3.0 * len(cols), 2.9 * len(maps)), squeeze=False)
    inner = (np.maximum(abs(ii - HALF), abs(jj - HALF)) <= 3).reshape(P, P)
    for r, (m, d) in enumerate(maps.items()):
        for c, (k, lab) in enumerate(cols):
            v = d[k] / d[k][inner].mean() if k in ("pred", "obs") else d[k]
            lv = np.log2(np.clip(v, 1e-3, None))
            im = axs[r, c].imshow(lv, cmap="RdBu_r", vmin=-1, vmax=1)
            axs[r, c].plot(HALF, HALF, "k+", ms=8)
            axs[r, c].set_xticks([0, HALF, P - 1], ["W", "", "E"])
            axs[r, c].set_yticks([0, HALF, P - 1], ["N", "", "S"])
            if r == 0:
                axs[r, c].set_title(lab, fontsize=9)
            if c == 0:
                axs[r, c].set_ylabel(m, fontsize=10)
    cb = fig.colorbar(im, ax=axs, shrink=0.6, ticks=[-1, -0.585, 0, 0.585, 1])
    cb.ax.set_yticklabels(["0.5×", "0.67×", "1×", "1.5×", "2×"])
    fig.suptitle("Position effect of the pixel head (F30, raw, seed mean); row 0 = north", fontsize=11)
    fig.savefig(fout / "edge_diag_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    write_manifest(cfg, f"p42_edge_diag{args.tag}", args.config, {
        "models": args.models, "seeds": args.seeds, "n_oe": int(io.size), "n_shift": int(is_.size), "tau": TAU,
        "orientation": "Zeile 0 = Norden, Spalte 0 = Westen", "results": res, "out": [str(mout), str(fout)]})


if __name__ == "__main__":
    main()

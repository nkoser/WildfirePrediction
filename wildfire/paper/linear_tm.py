"""Lineares Transformationsmodell mit cloglog-Link = Cox-Modell: Hazard Ratios je Variable.

Das interpretierbare Gegenstueck zum STS-DCTM, mit derselben Likelihood:

    log Lambda(tau | x) = log(e^a(tau) - e^a(0)) + x'beta,   a(tau) Bernstein vom Grad 12 auf [0, 100]

Mit cloglog-Link (Minimum-Extremwert) ist das genau Proportional Hazards: der
Basis-Hazard ist glatt statt Breslow-Treppe, e^beta_j ist die Hazard Ratio fuer
+1 Standardabweichung der Variable j. Mit --link logistic wird daraus ein
Proportional-Odds-Modell, e^beta sind dann Odds Ratios fuer F(tau).

Daten (beide bei natuerlicher Haeufigkeit, keine Anreicherung):
  Anpassung   zufaellige Stichtage des Wuerfels 2000-2019 (pixel_landmarks,
              Strom "random"; Ereignisse je Pixel bis 100 Tage)
  Auswertung  zufaellige Stichtage 2023-2025 aus surv_eval (dieselben wie fuer
              STS-DCTM und pxst), gleicher Metriksatz

Merkmale je Stichtag, je auf Trainings-Mittel und -Streuung standardisiert:
  <Kanal>          Mittelzelle, Mittel der letzten 7 Fenstertage, fuer die 53
                   Kanaele ohne Feuermaske und Landbedeckung (kategorial)
  fire_patch_30d   log(1 + Feuer-Pixeltage im Ausschnitt ueber 30 Tage)
  fire_centre_7d   Feuer in der Mittelzelle in den letzten 7 Tagen (0/1)
  season_sin/cos   Kalendertag des Stichtags

Berichtet wird univariabel (je ein Merkmal) und multivariabel, mit Wald-
Intervallen aus der beobachteten Information (Hesse-Matrix der NLL). Mit allen
57 Merkmalen ist das multivariable Modell nicht deutbar: t2m, t2m_max und
skin_t_max korrelieren mit r = 1,00, ebenso latent_heat und tot_evap; die HR
liefen auf 900 und 0,01 auseinander. Multivariabel geht deshalb je Gruppe
korrelierter Merkmale nur eines ein (select, |r| < --max-corr, gierig nach dem
univariablen Effekt); "represented_by" nennt den Vertreter. Das volle Modell
wird nur als Vorhersage mitbewertet (linear_all).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.eval.metrics import evaluate
from wildfire.models.bernstein import BernsteinSurvivalHead
from wildfire.models.surv_heads import interval_nll
from wildfire.paper.prepare import CHANNEL_NAMES, FIRE_CH, LANDCOVER_CH

H = 100
DEGREE = 12
C = 6
CH = [c for c in range(len(CHANNEL_NAMES)) if c not in (FIRE_CH, LANDCOVER_CH)]
NAMES = [CHANNEL_NAMES[c] for c in CH] + ["fire_patch_30d", "fire_centre_7d", "season_sin", "season_cos"]


def features(X, doy: np.ndarray, rows: np.ndarray | None = None, chunk: int = 4096) -> np.ndarray:
    """X (N, 30, 55, 13, 13) uint8 (auch memmap), Zeilen rows, doy je Zeile aus rows -> (n, len(NAMES)) float64.

    Gelesen werden nur die benoetigten Ausschnitte, nicht die ganzen Fenster."""
    rows = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
    out = []
    for a in range(0, rows.size, chunk):
        r = rows[a:a + chunk]
        xc = np.asarray(X[r, -7:, :, C, C], np.float64)                    # (b, 7, 55)
        fire = np.asarray(X[r, :, FIRE_CH]) > 0                            # (b, 30, 13, 13)
        out.append(np.column_stack([xc[:, :, CH].mean(1), np.log1p(fire.sum((1, 2, 3))),
                                    fire[:, -7:, C, C].any(1)]))
    ang = 2 * np.pi * np.asarray(doy, np.float64) / 365.25
    return np.column_stack([np.concatenate(out), np.sin(ang), np.cos(ang)])


class LinearTM(torch.nn.Module):
    def __init__(self, p: int, link: str):
        super().__init__()
        self.head = BernsteinSurvivalHead(DEGREE, float(H), init_offset=-8.0, init_cif=0.05, link=link)
        self.theta = torch.nn.Parameter(torch.zeros(DEGREE + 1, dtype=torch.float64))
        self.beta = torch.nn.Parameter(torch.zeros(p, dtype=torch.float64))

    def raw(self, x):
        r = self.theta.expand(x.shape[0], -1).clone()
        r[:, 0] = r[:, 0] + x @ self.beta                   # Shift: alle theta gleich, Kurven parallel in h
        return r

    def nll_sum(self, x, ev, t):
        return interval_nll(self.head, self.raw(x), ev, t) * x.shape[0]


def fit(x, ev, t, link, init=None, iters=200):
    m = LinearTM(x.shape[1], link).to(x.device)
    if init is not None:
        m.theta.data.copy_(init)
    opt = torch.optim.LBFGS(m.parameters(), lr=1, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = m.nll_sum(x, ev, t) / x.shape[0]
        loss.backward()
        return loss

    opt.step(closure)
    return m


def wald(m: LinearTM, x, ev, t) -> np.ndarray:
    """Standardfehler von beta aus der inversen Hesse-Matrix der Summen-NLL (theta und beta gemeinsam)."""
    n_t = m.theta.numel()

    def f(v):
        r = v[:n_t].expand(x.shape[0], -1).clone()
        r[:, 0] = r[:, 0] + x @ v[n_t:]
        return interval_nll(m.head, r, ev, t) * x.shape[0]

    v = torch.cat([m.theta.detach(), m.beta.detach()])
    Hm = torch.autograd.functional.hessian(f, v)
    Hm = Hm.double().cpu().numpy()                     # cusolver-SVD konvergiert bei schlecht konditionierter H nicht
    try:
        cov = np.linalg.pinv(Hm, hermitian=True)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(Hm + 1e-8 * np.abs(Hm).max() * np.eye(len(Hm)), hermitian=True)
    return np.sqrt(np.clip(np.diag(cov)[n_t:], 0, None))


def select(strength: np.ndarray, corr: np.ndarray, max_corr: float) -> np.ndarray:
    """Je Gruppe korrelierter Merkmale einer: gierig nach |beta| univariabel, aufgenommen wird nur,
    was mit allen bereits gewaehlten |r| < max_corr hat."""
    keep: list[int] = []
    for j in np.argsort(-np.abs(strength)):
        if all(abs(corr[j, k]) < max_corr for k in keep):
            keep.append(int(j))
    return np.array(sorted(keep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--link", choices=["cloglog", "logistic"], default="cloglog")
    ap.add_argument("--max-corr", type=float, default=0.7,
                    help="multivariabel nur Merkmale, die untereinander |r| < max-corr haben")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ---- Anpassung: zufaellige Wuerfel-Stichtage
    cdir = exp / "derived_data" / args.cube_dir
    meta = np.load(cdir / "train_meta.npz")
    rows = np.where(meta["kind"] == 1)[0]
    dates = CubeReader(exp).dates
    doy = np.array([dt.date.fromisoformat(dates[int(L)]).timetuple().tm_yday for L in meta["L"][rows]])
    Z_tr = features(np.load(cdir / "train_X.npy", mmap_mode="r"), doy, rows)
    ev_px, t_px = np.load(cdir / "train_event.npy")[rows], np.load(cdir / "train_t.npy")[rows].astype(np.int64)
    y_tr = {"centre": (ev_px[:, C, C], t_px[:, C, C]),
            "patch": (ev_px.reshape(rows.size, -1).any(1), t_px.reshape(rows.size, -1).min(1))}
    mu, sd = Z_tr.mean(0), Z_tr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    print(f"[ltm] Training: {rows.size:,} zufaellige Stichtage, {len(NAMES)} Merkmale, Ereignisrate 100 d: "
          f"Mitte {y_tr['centre'][0].mean():.4f}, Ausschnitt {y_tr['patch'][0].mean():.4f} ({time.time() - t0:.0f} s)",
          flush=True)

    # ---- Auswertung: zufaellige Stichtage der Testjahre (surv_eval)
    sdir = exp / "derived_data" / args.surv_dir
    ev_meta = dict(np.load(sdir / "eval_meta.npz"))
    Z_te = features(np.load(sdir / "eval_X.npy", mmap_mode="r"), ev_meta["doy"])
    y_te = {"centre": (ev_meta["ev_centre"].astype(bool), ev_meta["t_centre"].astype(int)),
            "patch": (ev_meta["ev_patch"].astype(bool), ev_meta["t_patch"].astype(int))}
    print(f"[ltm] Test: {Z_te.shape[0]:,} Stichtage ({time.time() - t0:.0f} s)", flush=True)

    tt = lambda a: torch.from_numpy(np.asarray(a)).to(device)  # noqa: E731
    x_tr, x_te = tt((Z_tr - mu) / sd), tt((Z_te - mu) / sd)
    corr = np.corrcoef(((Z_tr - mu) / sd).T)
    np.fill_diagonal(corr, 0)
    res, rows_csv = {}, []
    z = norm.ppf(0.975)
    for target in ("centre", "patch"):
        ev, t = tt(y_tr[target][0]), tt(y_tr[target][1])
        m0 = fit(x_tr[:, :0], ev, t, args.link)                              # nur Basis-Hazard
        uni = np.zeros((len(NAMES), 2))
        for j in range(len(NAMES)):
            mj = fit(x_tr[:, j:j + 1], ev, t, args.link, init=m0.theta.detach(), iters=100)
            uni[j] = [mj.beta.item(), wald(mj, x_tr[:, j:j + 1], ev, t)[0]]
        sel = select(uni[:, 0], corr, args.max_corr)
        m_all = fit(x_tr, ev, t, args.link, init=m0.theta.detach())         # alle Merkmale, nur fuer die Vorhersage
        m = fit(x_tr[:, sel], ev, t, args.link, init=m0.theta.detach())
        se, beta = np.full(len(NAMES), np.nan), np.full(len(NAMES), np.nan)
        se[sel], beta[sel] = wald(m, x_tr[:, sel], ev, t), m.beta.detach().cpu().numpy()
        with torch.no_grad():
            h = {"base": m0.head.hazards(m0.raw(x_te[:, :0]), H).cpu().numpy(),
                 "linear_all": m_all.head.hazards(m_all.raw(x_te), H).cpu().numpy(),
                 "linear_sel": m.head.hazards(m.raw(x_te[:, sel]), H).cpu().numpy()}
        metr = {}
        for k, hk in h.items():
            e = evaluate(hk, *y_te[target])
            metr[k] = {"nll": e["nll"], "ibs": e["integrated_brier"], "c_index": e["concordance_index"],
                       "c_td": e["c_index_td"], "dcal_dev": e["d_cal_max_dev"], "dcal_p": e["d_cal_p"],
                       **{f"{q}@{tau}": e["horizons"][str(tau)][key] for tau in (1, 7, 30, 100)
                          for q, key in (("auc", "roc_auc"), ("pr", "pr_auc"), ("brier", "brier"), ("ece", "ece"))}}
            print(f"[ltm] {target:6s} {k:10s} Test: C {metr[k]['c_index']:.3f}  Ctd {metr[k]['c_td']:.3f}  "
                  f"IBS {metr[k]['ibs']:.5f}  NLL {metr[k]['nll']:.4f}  AUC@7 {metr[k]['auc@7']:.3f}  "
                  f"PR@7 {metr[k]['pr@7']:.3f}  PR@30 {metr[k]['pr@30']:.3f}", flush=True)
        table = []
        for j, nm in enumerate(NAMES):
            pj = int(np.argmax(np.abs(corr[j])))
            rep = nm if j in sel else NAMES[int(sel[np.argmax(np.abs(corr[j, sel]))])]
            table.append({
                "target": target, "link": args.link, "variable": nm, "selected": bool(j in sel), "represented_by": rep,
                "ratio": float(np.exp(beta[j])), "lo": float(np.exp(beta[j] - z * se[j])),
                "hi": float(np.exp(beta[j] + z * se[j])),
                "p": float(2 * norm.sf(abs(beta[j]) / se[j])) if j in sel else float("nan"),
                "uni_ratio": float(np.exp(uni[j, 0])), "uni_lo": float(np.exp(uni[j, 0] - z * uni[j, 1])),
                "uni_hi": float(np.exp(uni[j, 0] + z * uni[j, 1])),
                "max_corr": float(corr[j, pj]), "max_corr_with": NAMES[pj], "sd_raw": float(sd[j])})
        table.sort(key=lambda r: (not r["selected"], -abs(np.log(r["ratio"] if r["selected"] else r["uni_ratio"]))))
        print(f"[ltm] {target}: {len(sel)} von {len(NAMES)} Merkmalen (|r| < {args.max_corr}); "
              f"{'Hazard' if args.link == 'cloglog' else 'Odds'} Ratios je +1 SD:", flush=True)
        for r in table:
            if r["selected"]:
                vertr = [x["variable"] for x in table if not x["selected"] and x["represented_by"] == r["variable"]]
                print(f"      {r['variable']:16s} {r['ratio']:6.3f} [{r['lo']:.3f}, {r['hi']:.3f}]   univariabel "
                      f"{r['uni_ratio']:6.3f}" + (f"   vertritt {', '.join(vertr)}" if vertr else ""), flush=True)
        res[target] = {"metrics": metr, "ratios": table, "selected": [NAMES[j] for j in sel],
                       "n_train": int(rows.size), "n_events_train": int(y_tr[target][0].sum())}
        rows_csv += table

    stem = f"linear_tm_{args.link}{args.tag}"
    out = exp / "reports" / f"{stem}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        w.writeheader(); w.writerows(rows_csv)
    print(f"[ltm] {out}", flush=True)
    plot(res, args.link, exp / "figures" / f"{stem}.png")
    write_manifest(cfg, f"p19_{stem}", args.config, {
        "link": args.link, "model": f"Bernstein Grad {DEGREE} auf [0, {H}] + linearer Shift x'beta",
        "features": NAMES, "standardization": {"mean": mu.tolist(), "sd": sd.tolist()},
        "train": "zufaellige Wuerfel-Stichtage 2000-2019 (pixel_landmarks random)",
        "test": f"{args.surv_dir}/eval (zufaellige Stichtage der Testjahre)", "results": res,
    })


def plot(res: dict, link: str, path: Path) -> None:
    """Forest-Plot je Ziel: multivariabel (Punkt mit 95-%-Intervall) und univariabel (Raute)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(res), figsize=(7 * len(res), 13), squeeze=False)
    for ax, (target, r) in zip(axes[0], res.items()):
        tab = sorted(r["ratios"], key=lambda x: (x["selected"], np.log(x["ratio"] if x["selected"] else x["uni_ratio"])))
        y = np.arange(len(tab))
        ax.errorbar([x["ratio"] for x in tab], y, xerr=[[x["ratio"] - x["lo"] for x in tab],
                                                         [x["hi"] - x["ratio"] for x in tab]],
                    fmt="o", color="#2a6fdb", ms=4, lw=1, label="multivariabel")
        ax.plot([x["uni_ratio"] for x in tab], y, "D", color="#e8a33d", ms=3.5, alpha=0.8, label="univariabel")
        ax.axvline(1, color="#555", lw=0.7)
        ax.set_xscale("log")
        ax.set_yticks(y, [x["variable"] for x in tab], fontsize=7)
        ax.set_xlabel(f"{'Hazard' if link == 'cloglog' else 'Odds'} Ratio je +1 SD (log)")
        ax.set_title(f"{target}: Test C-td {r['metrics']['linear_sel']['c_td']:.3f} ({len(r['selected'])} Merkmale)")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"[ltm] {path}", flush=True)


if __name__ == "__main__":
    main()

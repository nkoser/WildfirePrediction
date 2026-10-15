"""Survival auf den Paper-Daten, Schritt 2: Modelle trainieren und auswerten.

Landmarks aus surv_prepare (Stichtag L, Zeit bis zum ersten Feuer im 13 x 13-
Ausschnitt, zensiert bei 100 Tagen), Eingabe 30 Tage x 55 Kanaele x 13 x 13 bis L.

  km          Kaplan-Meier aus dem Training, fuer alle gleich
  km_fire     Kaplan-Meier getrennt nach "Feuer im Ausschnitt im Fenster ja/nein"
              (das Survival-Gegenstueck zur Persistenz)
  mlp_dctm    MLP auf Zusammenfassungen der Mittelzelle + DCTM-Kopf
  st_dctm     Tages-CNN -> Gedaechtnis -> Cross-Attention + DCTM-Kopf

DCTM-Kopf: F(tau | x) = 1 - sigma(-h(tau)) / sigma(-h(0)), h in Bernstein-Basis vom
Grad 12 auf [0, 100], monoton per Konstruktion; Verlust ist die exakte
intervallzensierte Likelihood. Jede Abfrage tau in [0, 100] ist moeglich.

Zwei Varianten:
  all   alle Stichtage
  new   nur Stichtage ohne Feuer im Ausschnitt in L-6 .. L ("neue Braende")
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.eval.metrics import evaluate
from sklearn.linear_model import LogisticRegression

from wildfire.models.bernstein import BernsteinSurvivalHead, bernstein_nll
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.compare import MLP
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.stdctm import CrossBlock, DayEncoder

H = 100
HZ = (1, 7, 14, 30, 60, 100)


# ------------------------------------------------------------------ Kaplan-Meier
def km_hazard(event, t_event):
    """Diskreter Hazard je Tag 1..H aus (event, t_event)."""
    h = np.zeros(H)
    for k in range(1, H + 1):
        at_risk = (t_event >= k).sum()
        d = ((t_event == k) & (event == 1)).sum()
        h[k - 1] = d / at_risk if at_risk else 0.0
    return np.clip(h, 1e-6, 1 - 1e-6)


# ------------------------------------------------------------------ Modelle
class STEncoder(nn.Module):
    def __init__(self, c_in: int = 55, d: int = 128):
        super().__init__()
        self.enc, self.q_enc = DayEncoder(c_in, d=d), DayEncoder(c_in, d=d)
        self.time = nn.Embedding(30, d)
        self.mem_block = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True)
        self.cross = nn.ModuleList([CrossBlock(d) for _ in range(2)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                                   # (B, 30, C, 13, 13)
        B, T = x.shape[:2]
        mem = self.enc(x[:, :-1].flatten(0, 1)).view(B, T - 1, -1)
        lag = torch.arange(T - 1, 0, -1, device=x.device)
        mem = self.mem_block(mem + self.time(lag)[None])
        q = (self.q_enc(x[:, -1]) + self.time.weight[0])[:, None]
        for blk in self.cross:
            q = blk(q, mem)
        return self.norm(q[:, 0])


class SurvNet(nn.Module):
    def __init__(self, body: nn.Module, d: int, init_cif: float):
        super().__init__()
        self.body = body
        self.dctm = BernsteinSurvivalHead(degree=12, tau_max=float(H), init_offset=-8.0,
                                          init_cif=init_cif)
        self.lin = nn.Linear(d, self.dctm.n_out)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return self.lin(self.body(x))


class MLPBody(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.m = MLP(n_in, 1)

    def forward(self, x):
        return self.m.body(x)


# ------------------------------------------------------------------ Daten
def summaries(X):
    s = X[:, :, :, 6, 6].astype(np.float32)                 # (n, 30, 55)
    return np.concatenate([s[:, -1], s[:, -7:].mean(1), s.mean(1), s.min(1), s.max(1)], 1)


def norm_batch(xb, mean, std, device):
    t = torch.from_numpy(np.ascontiguousarray(xb)).to(device).float()
    return (t - torch.from_numpy(mean).to(device)[None, None, :, None, None]) / \
        torch.from_numpy(std).to(device)[None, None, :, None, None]


def train(net, get_x, n_tr, ev, te, val_fn, device, seed, epochs, patience, bs, lr):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    best, state, bad, hist = np.inf, None, 0, []
    ev_t, te_t = torch.from_numpy(ev).float().to(device), torch.from_numpy(te).long().to(device)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(n_tr)
        tl = tn = 0
        for a in range(0, n_tr, bs):
            b = np.sort(order[a:a + bs])
            raw = net(get_x(b))
            bt = torch.from_numpy(b).to(device)
            loss = bernstein_nll(net.dctm, raw, ev_t[bt], te_t[bt], float(H))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tl += loss.item() * b.size; tn += b.size
        vl = val_fn(net)
        hist.append({"epoch": ep, "train_nll": tl / tn, "val_nll": vl})
        print(f"      Epoche {ep:2d}: train NLL {tl / tn:.4f}  val NLL {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return hist


@torch.no_grad()
def hazards(net, get_x, n, device, bs=256):
    net.eval()
    out = []
    for a in range(0, n, bs):
        b = np.arange(a, min(a + bs, n))
        out.append(net.dctm.hazards(net(get_x(b)), H).double().cpu().numpy())
    return np.concatenate(out)


def summarize(m):
    return {"nll": m["nll"], "c_index": m["concordance_index"], "ibs": m["integrated_brier"],
            **{f"auc@{k}": m["horizons"][str(k)]["roc_auc"] for k in HZ},
            **{f"pr@{k}": m["horizons"][str(k)]["pr_auc"] for k in HZ},
            **{f"ece@{k}": m["horizons"][str(k)]["ece"] for k in HZ}}


# ------------------------------------------------------------------ Hauptteil
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--variants", nargs="+", default=["all", "new"])
    ap.add_argument("--methods", nargs="+", default=["km", "km_fire", "mlp_dctm", "st_dctm"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--paper-eval", action="store_true",
                    help="zusaetzlich im Verschiebe-Protokoll des Papers auswerten (Variante all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = exp / "derived_data" / "paper_surv" / args.cache
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")

    t0 = time.time()
    data = {}
    for part in ("train", "val", "test"):
        z = np.load(d / f"{part}_meta.npz")
        data[part] = {k: z[k] for k in z.files}
        data[part]["X"] = np.load(d / f"{part}_X.npy")
        data[part]["fire_win"] = (data[part]["X"][:, :, FIRE_CH] > 0).any(axis=(1, 2, 3))
    print(f"[surv] geladen in {time.time() - t0:.0f} s: " + ", ".join(
        f"{p} {v['X'].shape[0]} (Ereignisse {v['event'].mean():.2f})" for p, v in data.items()), flush=True)

    # ---------------------------------------------- Auswertung im Protokoll des Papers
    # Gleiche Test-Stichproben, gleiche Eingabefenster (30 Tage bis t - dt - 1)
    # wie das 3D-U-Net. Fuer uns ist das ein Stichtag L = t - dt - 1; abgelesen
    # wird die Kurve beim Vorlauf k = dt + 1, als F(k) und als S(k-1) - S(k).
    PDTS = (1, 7, 15, 30)
    paper, pres = None, {}
    if args.paper_eval:
        pdir = exp / "derived_data" / "paper" / f"{args.cache}_L61"
        sp = np.load(pdir / "split.npy")
        paper = {"X": np.load(pdir / "X.npy", mmap_mode="r"), "y": np.load(pdir / "y.npy").astype(int),
                 "idx": {s: np.where(sp == s)[0] for s in ("val", "test")}}
        paper["L"] = paper["X"].shape[1]

    def pwin(rows, dt_):
        t1 = paper["L"] - dt_ - 1
        return np.asarray(paper["X"][rows, t1 - 30:t1])          # (n, 30, 55, 13, 13)

    def paper_curves(haz_fn):
        out = {}
        for s, rows in paper["idx"].items():
            out[s] = {}
            for dt_ in PDTS:
                k = dt_ + 1
                Fk, dk = [], []
                for a in range(0, rows.size, 256):
                    r = rows[a:a + 256]
                    Sv = np.cumprod(1 - haz_fn(pwin(r, dt_)), axis=1)
                    Fk.append(1 - Sv[:, k - 1]); dk.append(Sv[:, k - 2] - Sv[:, k - 1])
                out[s][dt_] = (np.concatenate(Fk), np.concatenate(dk))
        return out

    def paper_record(name, curves_per_seed):
        yv, yt = paper["y"][paper["idx"]["val"]], paper["y"][paper["idx"]["test"]]
        lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12)) - np.log(np.clip(1 - p, 1e-12, 1))
        pres[name] = {}
        for dt_ in PDTS:
            for si, score in enumerate(("F_k", "first_on_day_k")):
                ms = []
                for c in curves_per_seed:
                    pl = LogisticRegression(C=1e6).fit(lg(c["val"][dt_][si])[:, None], yv)
                    ms.append(paper_metrics(pl.predict_proba(lg(c["test"][dt_][si])[:, None])[:, 1], yt))
                pres[name].setdefault(score, {})[str(dt_)] = {
                    k: float(np.mean([m[k] for m in ms])) for k in ms[0] if isinstance(ms[0][k], float)}
            f = pres[name]["F_k"][str(dt_)]
            print(f"[paper] {name:9s} dt={dt_:2d} F(k): PR-AUC {f['pr_auc']:.3f} ROC {f['roc_auc']:.3f} "
                  f"F1 {f['f1']:.3f} ECE {f['ece']:.3f} Brier {f['brier']:.3f}", flush=True)

    @torch.no_grad()
    def net_haz(net, x):
        net.eval()
        return net.dctm.hazards(net(x), H).double().cpu().numpy()

    res: dict = {}
    for var in args.variants:
        sel = {p: (np.ones(len(v["event"]), bool) if var == "all" else ~v["recent_fire"].astype(bool))
               for p, v in data.items()}
        D = {p: {k: (a[sel[p]] if isinstance(a, np.ndarray) and a.shape[:1] == sel[p].shape else a)
                 for k, a in v.items()} for p, v in data.items()}
        tr, va, te = D["train"], D["val"], D["test"]
        print(f"\n[surv] Variante {var}: train {len(tr['event'])}, val {len(va['event'])}, "
              f"test {len(te['event'])}; Ereignisrate test {te['event'].mean():.3f}", flush=True)
        res[var] = {"n": {p: int(len(D[p]["event"])) for p in D},
                    "event_rate": {p: float(D[p]["event"].mean()) for p in D}}
        ev_te, tt_te = te["event"].astype(bool), te["t_event"].astype(int)

        def report(name, per_seed_h):
            ms = [summarize(evaluate(h, ev_te, tt_te)) for h in per_seed_h]
            agg = {k: {"mean": float(np.mean([m[k] for m in ms])), "std": float(np.std([m[k] for m in ms]))}
                   for k in ms[0]}
            res[var][name] = agg
            print(f"[surv] {var:3s} {name:9s} NLL {agg['nll']['mean']:.4f}  C {agg['c_index']['mean']:.3f}  "
                  + "  ".join(f"AUC@{k} {agg[f'auc@{k}']['mean']:.3f}" for k in HZ)
                  + f"  IBS {agg['ibs']['mean']:.4f}", flush=True)

        if "km" in args.methods:
            h = km_hazard(tr["event"], tr["t_event"])
            report("km", [np.tile(h, (len(ev_te), 1))])
        if "km_fire" in args.methods:
            hs = {f: km_hazard(tr["event"][tr["fire_win"] == f], tr["t_event"][tr["fire_win"] == f])
                  for f in (False, True)}
            report("km_fire", [np.stack([hs[bool(f)] for f in te["fire_win"]])])
            if paper and var == "all":
                paper_record("km_fire", [paper_curves(lambda w: np.stack(
                    [hs[bool(f)] for f in (w[:, :, FIRE_CH] > 0).any(axis=(1, 2, 3))]))])

        init_cif = float(1 - np.prod(1 - km_hazard(tr["event"], tr["t_event"])))
        if "mlp_dctm" in args.methods:
            S = {p: summaries(D[p]["X"]) for p in D}
            mu, sd = S["train"].mean(0), S["train"].std(0) + 1e-6
            S = {p: torch.from_numpy(((v - mu) / sd).astype(np.float32)).to(device) for p, v in S.items()}
            per, pc = [], []
            for seed in args.seeds:
                print(f"    mlp_dctm seed {seed}", flush=True)
                net = SurvNet(MLPBody(S["train"].shape[1]), 256, init_cif).to(device)
                ev_v, tt_v = torch.from_numpy(va["event"]).float().to(device), torch.from_numpy(va["t_event"]).long().to(device)
                def vf(n):
                    n.eval()
                    with torch.no_grad():
                        return bernstein_nll(n.dctm, n(S["val"]), ev_v, tt_v, float(H)).item()
                train(net, lambda b: S["train"][torch.from_numpy(b).to(device)], len(tr["event"]),
                      tr["event"], tr["t_event"], vf, device, seed, 200, 15, 512, 1e-3)
                per.append(hazards(net, lambda b: S["test"][torch.from_numpy(b).to(device)], len(te["event"]), device))
                if paper and var == "all":
                    pc.append(paper_curves(lambda w, net=net: net_haz(net, torch.from_numpy(
                        ((summaries(w) - mu) / sd).astype(np.float32)).to(device))))
            report("mlp_dctm", per)
            if pc:
                paper_record("mlp_dctm", pc)

        if "st_dctm" in args.methods:
            per, pc = [], []
            for seed in args.seeds:
                print(f"    st_dctm seed {seed}", flush=True)
                net = SurvNet(STEncoder(), 128, init_cif).to(device)
                gx = {p: (lambda b, p=p: norm_batch(D[p]["X"][b], mean, std, device)) for p in D}

                def vf(n):
                    n.eval()
                    with torch.no_grad():
                        tot = 0.0
                        for a in range(0, len(va["event"]), 256):
                            b = np.arange(a, min(a + 256, len(va["event"])))
                            raw = n(gx["val"](b))
                            tot += bernstein_nll(n.dctm, raw, torch.from_numpy(va["event"][b]).float().to(device),
                                                 torch.from_numpy(va["t_event"][b]).long().to(device),
                                                 float(H)).item() * b.size
                    return tot / len(va["event"])
                train(net, gx["train"], len(tr["event"]), tr["event"], tr["t_event"], vf,
                      device, seed, 40, 6, 64, 5e-4)
                per.append(hazards(net, gx["test"], len(te["event"]), device))
                if paper and var == "all":
                    pc.append(paper_curves(lambda w, net=net: net_haz(net, norm_batch(w, mean, std, device))))
            report("st_dctm", per)
            if pc:
                paper_record("st_dctm", pc)

    write_manifest(cfg, f"p8_surv_{args.cache}", args.config, {
        "cache": args.cache, "follow_up_days": H, "horizons": list(HZ), "seeds": args.seeds,
        "dctm_head": "Bernstein Grad 12 auf [0,100], monoton, S(0)=1, intervallzensierte Likelihood",
        "note": "Stichtage an Paper-Stichproben gebunden (1/3 Feuertage); Ereignisrate weit ueber der echten",
        "results": res,
        "paper_protocol": pres,
    })

    if pres:
        ref = json.loads((exp / "data_manifests" / f"p5_compare_{args.cache}.json").read_text())["results"]
        print(f"\n[paper] {args.cache} TEST, PR-AUC / ROC-AUC im Verschiebe-Protokoll (Survival: F(k))")
        for dt_ in PDTS:
            row = [f"dt={dt_:2d}"]
            for m in ("unet3d_dt", "hgb_dt", "mlp_multi"):
                a = ref[m][str(dt_)]["summary"]
                row.append(f"{m} {a['pr_auc']['mean']:.3f}/{a['roc_auc']['mean']:.3f}")
            for m in pres:
                a = pres[m]["F_k"][str(dt_)]
                row.append(f"{m} {a['pr_auc']:.3f}/{a['roc_auc']:.3f}")
            print("    " + "   ".join(row))


if __name__ == "__main__":
    main()

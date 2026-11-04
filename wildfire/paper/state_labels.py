"""Zustaende am Stichtag und Typen des ersten Feuers danach: gemeinsame Definition fuer Multi-State-Auswertungen.

Feuerhistorie aus dem CSR-Index aller MODIS-Feuertage je Pixel (derived_data/events/fire_index.npz, nur Pixel der
Gueltigkeitsmaske; Tagesindex -> Tag seit 1.1.2000 ueber dessen dates), als sortierte Schluessel
Pixel * DAY_BITS + Tag wie in surv_eval / cause_labels. Alternativ (--keys raw) die rohen MCD14A1-Feuer wie bei den
Survival-Zielen (surv_eval.raw_fire_keys, ganzes Gitter).

Pixel am Stichtag L (Tage absolut):
  B "brennend"     Feuer in [L-6, L]
  O "erloschen"    kein Feuer in [L-6, L], aber Feuer in [L-183, L-7]
  U "unverbrannt"  sonst
  days_since       L - letzter Feuertag <= L, 0 .. 365; 366 = keins im letzten Jahr
  n365             Feuertage in [L-364, L]
Ausschnitt (13 x 13): B, wenn irgendein Pixel B; O, wenn kein B aber ein O; sonst U; days_since = Minimum,
n365 = Summe ueber die Pixel.

Typ des ersten Feuers an Tag k = L + t (nur bei Ereignis bis H; t aus den Survival-Zielen):
  continuation  Zustand B
  return        Zustand O
  spread        Zustand U und ein ANDERES Pixel im (2r+1)^2-Kern um ein Erstfeuer-Pixel brannte in [k-3, k-1]
  ignition      Zustand U, sonst
r = 1 (3 x 3) und r = 3 (7 x 7) werden beide ausgegeben. Erstfeuer-Pixel: fuer die Mitte das Mittelpixel, fuer den
Ausschnitt alle Pixel des Ausschnitts mit Feuer an Tag k. Findet der Schluesselsatz an Tag k kein Feuer im
Ausschnitt (Ziel aus anderer Quelle), ist der Typ bei Zustand U "unresolved".
NFDB-Ursache (cause_labels, nur Test-Teil "eval"): Code des Mittelpixels bzw. die hoechste Zuendungs-Ursache
(1 Blitz, 2 Mensch, 3 unbekannt) unter den Erstfeuer-Pixeln, sonst 4 Ausbreitung / 5 ohne Label / 0 kein Ereignis.

  python -m wildfire.paper.state_labels --config ... [--part eval|val|km] [--keys index|raw] [--max-n N --tag _smoke]
  -> derived_data/state_labels/<part>_<keys><tag>.npz (+ .json mit Anteilen); bricht ab, wenn vorhanden.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config

DAY_BITS = 1 << 15
DAY0 = dt.date(2000, 1, 1)
HALF, P = 6, 13
STATES = ("U", "O", "B")
TYPES = ("none", "continuation", "return", "spread", "ignition", "unresolved")
T_CODE = {k: i for i, k in enumerate(TYPES)}


def index_keys(exp: Path) -> np.ndarray:
    """fire_index.npz -> sortierte Schluessel Pixel * DAY_BITS + Tag seit 1.1.2000."""
    z = np.load(exp / "derived_data" / "events" / "fire_index.npz")
    absd = np.array([(dt.date.fromisoformat(str(d)) - DAY0).days for d in z["dates"]], np.int64)
    assert np.all(np.diff(absd) > 0), "fire_index: Daten nicht streng aufsteigend"
    pix = np.repeat(z["fire_pixels"].astype(np.int64), np.diff(z["fire_offsets"]))
    return np.sort(pix * DAY_BITS + absd[z["fire_days"].astype(np.int64)])


def raw_keys(cfg) -> np.ndarray:
    from wildfire.paper.surv_eval import raw_fire_keys
    return raw_fire_keys(Path(cfg["dataset_root"]))[0]


def count_in(keys: np.ndarray, flat: np.ndarray, lo, hi) -> np.ndarray:
    """Zahl der Feuertage in [lo, hi] je Pixel (Arrays gleicher Form)."""
    flat = np.asarray(flat, np.int64)
    a = np.searchsorted(keys, flat * DAY_BITS + np.asarray(lo, np.int64), "left")
    b = np.searchsorted(keys, flat * DAY_BITS + np.asarray(hi, np.int64), "right")
    return b - a


def last_upto(keys: np.ndarray, flat: np.ndarray, L) -> np.ndarray:
    """Letzter Feuertag <= L je Pixel, -1 wenn keiner."""
    flat = np.asarray(flat, np.int64)
    pos = np.searchsorted(keys, flat * DAY_BITS + np.asarray(L, np.int64), "right") - 1
    k = keys[np.maximum(pos, 0)]
    return np.where((pos >= 0) & (k // DAY_BITS == flat), k % DAY_BITS, -1)


def pixel_state(keys, flat, L):
    """-> state (0 U, 1 O, 2 B), days_since (0..366), n365 je Pixel; flat und L broadcastbar."""
    flat, L = np.broadcast_arrays(np.asarray(flat, np.int64), np.asarray(L, np.int64))
    nb = count_in(keys, flat, L - 6, L)
    no = count_in(keys, flat, L - 183, L - 7)
    state = np.where(nb > 0, 2, np.where(no > 0, 1, 0)).astype(np.int8)
    last = last_upto(keys, flat, L)
    ds = np.where((last >= 0) & (L - last <= 365), L - last, 366).astype(np.int16)
    return state, ds, count_in(keys, flat, L - 364, L).astype(np.int16)


def patch_flat(row, col, H, W):
    """(n, 169) flache Pixel-IDs des 13 x 13-Ausschnitts und Maske innerhalb des Gitters."""
    dr, dc = np.meshgrid(np.arange(-HALF, HALF + 1), np.arange(-HALF, HALF + 1), indexing="ij")
    R, C = row[:, None] + dr.ravel()[None], col[:, None] + dc.ravel()[None]
    ok = (R >= 0) & (R < H) & (C >= 0) & (C < W)
    return np.where(ok, R * W + C, 0).astype(np.int64), ok, R, C


def neighbour_fire(keys, R, C, k, H, W, r: int) -> np.ndarray:
    """Brannte ein ANDERES Pixel im (2r+1)^2-Kern um (R, C) in [k-3, k-1]? R, C, k gleicher Form."""
    hit = np.zeros(np.shape(R), bool)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = R + dr, C + dc
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            n = count_in(keys, np.where(ok, rr * W + cc, 0), k - 3, k - 1)
            hit |= ok & (n > 0)
    return hit


def centre_labels(keys, row, col, L, ev, t, H, W):
    flat = row.astype(np.int64) * W + col
    st, ds, n365 = pixel_state(keys, flat, L)
    k = L + t.astype(np.int64)
    out = {"state_centre": st, "days_since_centre": ds, "n365_centre": n365}
    for r in (1, 3):
        nf = neighbour_fire(keys, row.astype(np.int64), col.astype(np.int64), k, H, W, r)
        typ = np.where(st == 2, T_CODE["continuation"], np.where(st == 1, T_CODE["return"],
                       np.where(nf, T_CODE["spread"], T_CODE["ignition"])))
        out[f"type_centre_r{r}"] = np.where(ev, typ, T_CODE["none"]).astype(np.int8)
    out["keys_hit_centre"] = ev & (count_in(keys, flat, k, k) > 0)      # Ziel-Ereignis im Schluesselsatz?
    return out


def patch_labels(keys, row, col, L, ev, t, H, W, chunk=5000):
    n = row.size
    out = {"state_patch": np.zeros(n, np.int8), "days_since_patch": np.zeros(n, np.int16),
           "n365_patch": np.zeros(n, np.int32), "n_first_px_patch": np.zeros(n, np.int16),
           "type_patch_r1": np.zeros(n, np.int8), "type_patch_r3": np.zeros(n, np.int8),
           "first_px_mask": np.zeros((n, P * P), bool)}
    for a in range(0, n, chunk):
        s = slice(a, min(a + chunk, n))
        flat, ok, R, C = patch_flat(row[s].astype(np.int64), col[s].astype(np.int64), H, W)
        Ls = L[s][:, None].astype(np.int64)
        st, ds, n365 = pixel_state(keys, flat, Ls)
        st = np.where(ok, st, 0)
        ds = np.where(ok, ds, 366)
        pst = st.max(1)
        out["state_patch"][s] = pst
        out["days_since_patch"][s] = ds.min(1)
        out["n365_patch"][s] = np.where(ok, n365, 0).sum(1)
        k = Ls + t[s][:, None].astype(np.int64)
        first = ok & (count_in(keys, flat, k, k) > 0) & ev[s][:, None]
        out["first_px_mask"][s] = first
        out["n_first_px_patch"][s] = first.sum(1)
        kk = np.broadcast_to(k, R.shape)
        fi = np.nonzero(first)                                          # nur Erstfeuer-Pixel pruefen
        for r in (1, 3):
            nf = np.zeros(first.shape[0], bool)
            nf[fi[0][neighbour_fire(keys, R[fi], C[fi], kk[fi], H, W, r)]] = True
            typ = np.where(pst == 2, T_CODE["continuation"], np.where(pst == 1, T_CODE["return"],
                           np.where(first.any(1), np.where(nf, T_CODE["spread"], T_CODE["ignition"]),
                                    T_CODE["unresolved"])))
            out[f"type_patch_r{r}"][s] = np.where(ev[s], typ, T_CODE["none"])
    return out


def cause_codes(exp: Path, meta: dict, first_mask: np.ndarray):
    """NFDB-Codes aus cause_labels (Teil eval); None, wenn die Datei fehlt oder nicht zu den Stichtagen passt."""
    p = exp / "derived_data" / "cause_labels" / "eval_cause.npz"
    if not p.exists():
        return None
    z = np.load(p)
    if not (np.array_equal(z["row"], meta["row"]) and np.array_equal(z["col"], meta["col"])
            and np.array_equal(z["L_abs"], meta["L_abs"])):
        print("[state] eval_cause.npz passt nicht zu eval_meta - Ursachen entfallen", flush=True)
        return None
    c = z["cause"].reshape(len(meta["row"]), -1)
    centre = c[:, (P * P) // 2]
    ign = np.where(first_mask & np.isin(c, (1, 2, 3)), c, 0)
    # Prioritaet Blitz < Mensch < unbekannt nach Code; vorhandene Zuendung schlaegt Ausbreitung/ohne Label
    best = ign.max(1)
    rest = np.where((first_mask & (c == 4)).any(1), 4, np.where(first_mask.any(1), 5, 0))
    return centre.astype(np.int8), np.where(best > 0, best, rest).astype(np.int8)


def shares(code: np.ndarray, names) -> dict:
    return {nm: float((code == i).mean()) for i, nm in enumerate(names)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--part", default="eval", choices=("eval", "val", "km"))
    ap.add_argument("--keys", default="index", choices=("index", "raw"))
    ap.add_argument("--max-n", type=int, default=0, help="Rauchtest: nur die ersten N Stichtage")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out_dir = exp / "derived_data" / "state_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.part}_{args.keys}{args.tag}.npz"
    if out.exists() or out.with_suffix(".json").exists():
        raise SystemExit(f"ABBRUCH: {out} existiert")
    meta = dict(np.load(exp / "derived_data" / args.surv_dir / f"{args.part}_meta.npz"))
    if args.max_n:
        meta = {k: v[:args.max_n] for k, v in meta.items()}
    H, W = (int(v) for v in np.load(exp / "derived_data" / "events" / "fire_index.npz")["grid"])
    keys = index_keys(exp) if args.keys == "index" else raw_keys(cfg)
    print(f"[state] {args.part}: {meta['row'].size:,} Stichtage, {keys.size:,} Feuer-Pixeltage ({args.keys})", flush=True)
    row, col, L = meta["row"], meta["col"], meta["L_abs"].astype(np.int64)
    res = centre_labels(keys, row, col, L, meta["ev_centre"].astype(bool), meta["t_centre"].astype(np.int64), H, W)
    res.update(patch_labels(keys, row, col, L, meta["ev_patch"].astype(bool), meta["t_patch"].astype(np.int64), H, W))
    if args.part == "eval":
        cc = cause_codes(exp, meta, res["first_px_mask"])
        if cc is not None and not args.max_n:
            res["cause_centre"], res["cause_patch"] = cc
    summ = {"args": vars(args), "n": int(row.size), "keys": int(keys.size),
            "definition": __doc__.split("\n\n")[1:4]}
    for tgt in ("centre", "patch"):
        ev = meta[f"ev_{tgt}"].astype(bool)
        st = res[f"state_{tgt}"]
        d = {"state_share": shares(st, STATES),
             "event_rate_by_state": {s: float(ev[st == i].mean()) if (st == i).any() else None
                                     for i, s in enumerate(STATES)},
             "n_events": int(ev.sum())}
        for r in (1, 3):
            ty = res[f"type_{tgt}_r{r}"][ev]
            d[f"type_share_events_r{r}"] = shares(ty, TYPES)
        if tgt == "centre":
            d["event_in_keys"] = float(res["keys_hit_centre"][ev].mean()) if ev.any() else None
            o = (st == 1)
            d["O_days_since_le30_share"] = float((res["days_since_centre"][o] <= 30).mean()) if o.any() else None
        else:
            d["event_first_px_found"] = float((res["n_first_px_patch"][ev] > 0).mean()) if ev.any() else None
        summ[tgt] = d
        print(f"[state] {tgt}: Zustaende " + ", ".join(f"{k} {v:.4f}" for k, v in d["state_share"].items())
              + "; Ereignisrate je Zustand " + ", ".join(f"{k} {v:.4f}" for k, v in d["event_rate_by_state"].items()
                                                       if v is not None), flush=True)
        for r in (1, 3):
            print(f"[state]   Ereignistypen r{r} ({d['n_events']:,} Ereignisse): " + ", ".join(
                f"{k} {v:.3f}" for k, v in d[f"type_share_events_r{r}"].items() if k != "none"), flush=True)
    np.savez_compressed(out, **res, row=row, col=col, L_abs=meta["L_abs"])
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1, default=float))
    print(f"[state] -> {out}", flush=True)


if __name__ == "__main__":
    main()

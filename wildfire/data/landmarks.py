"""Erzeugt den Landmark-Datensatz fuer Dense Spatiotemporal Survival.

Fuer ein Pixel i und einen Landmark-Zeitpunkt t wird die Zeit bis zur
naechsten Proxy-Zuendung modelliert:

    T_i - t   mit Ereignisindikator, rechtszensiert bei Horizont H.

Festgelegte Regeln - jede davon ist eine bewusste Entscheidung und steht
im Manifest:

  Beobachtungsfenster L   30 Tage Historie, t-L+1 .. t
  Prognosehorizont H      100 Tage, Ereignisfenster t+1 .. t+H.
                          Der fruehere Deckel von 30 Tagen kam von der
                          Herbstkontamination der MODIS-Maske. Mit
                          gemeldeten Braenden faellt er weg: gemessen
                          betraegt der Hazard 1,5e-5 je Tag in Tagen 1-30,
                          1,1e-5 in 31-60, 6,3e-6 in 61-90 und 1,9e-6 in
                          91-120. Erst danach faellt er in den Winter, wo
                          er exakt null wird - was ein Modell aus dem
                          Kalendertag trivial lernt und was Metriken bei
                          H > 150 nur noch aufblaeht.
  Cooldown C              7 Tage. Eine Zuendung ist ein Feuer, dem in
                          Umgebungsradius 1 sieben feuerfreie Tage
                          vorausgingen (siehe wildfire.data.events).
  Aktive Feuer            Landmarks an Pixeln, die in t-C+1 .. t brennen,
                          werden AUSGESCHLOSSEN. "Zeit bis zur naechsten
                          Zuendung" ist fuer ein brennendes Pixel nicht
                          definiert.
  Wiederkehrende Feuer    Ein Pixel darf mehrfach als Landmark auftreten,
                          zu verschiedenen t. Das ist der Kern der
                          Survival-Formulierung, kein Duplikat.
  Zensierung              Rechtszensierung bei H. Das EREIGNISFENSTER ist
                          nicht auf die Saison beschraenkt - nur der
                          Landmark ist es. Ein Landmark im August darf in
                          den November schauen; dort ist der Hazard klein,
                          aber das ist eine Aussage ueber die Welt und kein
                          Datenproblem. Bei Landmarks bis zum 31. August
                          und H = 100 bleibt t+H im selben Kalenderjahr,
                          die zeitlichen Splits bleiben also sauber.
  Gesperrte Kanaele       Feuermaske und die 4-um-Helligkeitstemperaturen
                          sind nur bis t-C sichtbar. Sie sind Rohmaterial
                          des Labels (siehe Audit-Stufen 6 und 2c).
  Saison                  Landmarks 1. Mai bis 31. August. Ausserhalb
                          dominieren Stoppelbrennen und
                          Schlagabraumverbrennung (Stufe d1b).

Ziehung: In ALLEN Splits werden alle Positiven behalten und Negative mit
bekannter Rate gezogen. Jede Zeile traegt ein Gewicht
``w = N_stratum / n_gezogen``; die Positiven haben Gewicht 1.

Warum nicht gleichmaessig ziehen? Bei einer Praevalenz von rund 1 zu 2500
liefert eine gleichmaessige Ziehung von 1 Mio. Zeilen nur ein paar hundert
Positive - zu wenig fuer Kalibrierung und horizontweise Metriken. Mit
Gewichten sind Brier-Score, PR-AUC und Kalibrierung erwartungstreu fuer
die echte Population, bei einem Bruchteil der Zeilen.

Die Gewichte werden gegen eine gleichmaessige Kontrollziehung geprueft:
die gewichtete Praevalenz muss deren beobachteter Praevalenz entsprechen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest
from wildfire.data.splits import SPLIT_DEFS, assign

# Wuerfelindizes der Kanaele, die dem Label zu nahe sind.
LOCKED_CHANNELS = {
    0: "MCD14A1 active_fire - ist das Label selbst",
    38: "BT_band20 (3.7 um) - Feuerband, aus dem MCD14A1 berechnet wird",
    39: "BT_band21 (3.9 um) - Feuerband, aus dem MCD14A1 berechnet wird",
    40: "BT_band31 (11 um) - Referenzkanal des Feueralgorithmus",
    41: "BT_band32 (12 um) - Referenzkanal des Feueralgorithmus",
}


DAY_BITS = 16384          # > 9497 Tage, Zweierpotenz fuer schnelle Division


class FireIndex:
    """Feuer- und Zuendungstage je Pixel, als EIN global sortierter Schluessel.

    Statt einer Ragged-Struktur mit Offsets je Pixel wird jeder Eintrag als
    ``pixel * DAY_BITS + tag`` kodiert. Weil Pixel-ID und Tag beide
    aufsteigend sortiert sind, ist das Ergebnis ein einziges sortiertes
    int64-Array - und damit ist jede Abfrage ein vektorisiertes
    ``np.searchsorted`` ueber beliebig viele Kandidaten gleichzeitig.

    Das ersetzt Python-Schleifen ueber Millionen Kandidaten, die vorher
    sowohl an Laufzeit als auch an Speicher gescheitert sind.
    """

    def __init__(self, path: Path, nfdb_path: Path | None = None) -> None:
        z = np.load(path, allow_pickle=False)
        self.H, self.W = (int(x) for x in z["grid"])
        self.dates = [str(d) for d in z["dates"]]
        # Die MODIS-Maske bleibt IMMER die Quelle fuer "brennt gerade" -
        # sie ist tagesaufgeloest und flaechendeckend, waehrend die
        # Branddatenbank nur Zuendungen kennt, keine Brenndauer.
        self.fire_keys = self._keys(z["fire_pixels"], z["fire_offsets"], z["fire_days"])
        if nfdb_path is not None:
            n = np.load(nfdb_path, allow_pickle=True)
            self.ign_keys = np.asarray(n["keys"], np.int64)
            self.ign_events = np.asarray(n["event_id"], np.int32)
            self.ign_cause = np.asarray(n["cause"], np.int8)
            self.ign_size = np.asarray(n["size_ha"], np.float32)
            self.label_source = "nfdb"
        else:
            self.ign_keys = self._keys(z["ign_pixels"], z["ign_offsets"], z["ign_days"])
            self.ign_events = np.asarray(z["ign_events"], np.int32)
            self.ign_cause = np.zeros(self.ign_keys.size, np.int8)
            self.ign_size = np.full(self.ign_keys.size, np.nan, np.float32)
            self.label_source = "modis_proxy"

    @staticmethod
    def _keys(pixels, offsets, days) -> np.ndarray:
        counts = np.diff(offsets)
        px = np.repeat(np.asarray(pixels, np.int64), counts)
        return px * DAY_BITS + np.asarray(days, np.int64)

    def has_fire_between(self, flat: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Brennt Pixel flat[i] an irgendeinem Tag in [lo[i], hi[i]]?"""
        lo = np.maximum(lo, 0)
        j = np.searchsorted(self.fire_keys, flat * DAY_BITS + lo, "left")
        ok = j < self.fire_keys.size
        out = np.zeros(flat.shape, bool)
        out[ok] = self.fire_keys[j[ok]] <= (flat[ok] * DAY_BITS + hi[ok])
        return out

    def next_ignition(self, flat: np.ndarray, t: np.ndarray):
        """Erste Zuendung mit Tag > t. (Tag, Ereignis-ID, Ursache, gefunden)."""
        j = np.searchsorted(self.ign_keys, flat * DAY_BITS + (t + 1), "left")
        ok = j < self.ign_keys.size
        day = np.full(flat.shape, -1, np.int64)
        eid = np.full(flat.shape, -1, np.int32)
        cau = np.zeros(flat.shape, np.int8)
        jj = j[ok]
        same = np.zeros(flat.shape, bool)
        same[ok] = (self.ign_keys[jj] // DAY_BITS) == flat[ok]
        sel = np.nonzero(same)[0]
        day[sel] = self.ign_keys[j[sel]] % DAY_BITS
        eid[sel] = self.ign_events[j[sel]]
        cau[sel] = self.ign_cause[j[sel]]
        return day, eid, cau, same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="temporal", choices=sorted(SPLIT_DEFS))
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--cooldown", type=int, default=7)
    ap.add_argument("--season-start", default="05-01")
    ap.add_argument("--season-end", default="08-31")
    ap.add_argument("--neg-per-pos-train", type=int, default=10)
    ap.add_argument("--neg-per-split-eval", type=int, default=2_000_000)
    ap.add_argument("--prevalence-probe", type=int, default=8_000_000,
                    help="gleichmaessige Kontrollziehung zur Pruefung der Gewichte")
    ap.add_argument("--label-source", default="nfdb", choices=("nfdb", "modis"),
                    help="nfdb = gemeldete Zuendungen, modis = abgeleiteter Proxy")
    ap.add_argument("--nfdb-index",
                    default="derived_data/events/nfdb_index.npz")
    ap.add_argument("--mask", default=None,
                    help="Gueltigkeitsmaske; Standard folgt der Labelquelle. "
                         "Bei NFDB ist es observed_nfdb.npy - Regionen ohne "
                         "jede Meldung liefern keine Negative, sondern "
                         "unbeobachtete Zeit.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))

    exp = Path(cfg["experiment_root"])
    L, H, C = args.lookback, args.horizon, args.cooldown
    nfdb = (exp / args.nfdb_index) if args.label_source == "nfdb" else None
    idx = FireIndex(exp / "derived_data" / "events" / "fire_index.npz", nfdb)
    print(f"[landmarks] Labelquelle: {idx.label_source}, "
          f"{idx.ign_keys.size} Zuendungen")
    mask_name = args.mask or ("observed_nfdb.npy" if args.label_source == "nfdb"
                              else "valid.npy")
    valid = np.load(exp / "derived_data" / "masks" / mask_name)
    print(f"[landmarks] Maske {mask_name}: {int(valid.sum())} Pixel")
    region = np.load(exp / "derived_data" / "masks" / "region.npy")
    reg_man = json.loads((exp / "data_manifests" / "d2_regions.json").read_text())
    code_of = reg_man["code_of"]

    dates = [dt.date.fromisoformat(d) for d in idx.dates]
    year_of = np.array([d.year for d in dates], np.int32)
    in_season = np.array(
        [args.season_start <= d.isoformat()[5:] <= args.season_end for d in dates], bool)
    season_ts = np.nonzero(in_season)[0].astype(np.int64)
    print(f"[landmarks] {season_ts.size} Landmark-Tage in der Saison "
          f"{args.season_start}..{args.season_end}, L={L} H={H} C={C}")

    valid_flat = np.nonzero(valid.ravel())[0].astype(np.int64)
    n_valid = valid_flat.size
    region_flat = region.ravel()
    print(f"[landmarks] {n_valid} gueltige Pixel, Kandidatenraum "
          f"{n_valid * season_ts.size / 1e9:.2f} Mrd. Pixel-Tage")

    spec = SPLIT_DEFS[args.split]
    buf_flat = None
    bpath = exp / "derived_data" / "masks" / f"buffer_{args.split}.npy"
    if bpath.exists():
        buf_flat = np.load(bpath).ravel()

    def part_masks(t: np.ndarray, flat: np.ndarray) -> dict[str, np.ndarray]:
        """Vektorisierte Splitzuweisung; test schlaegt val schlaegt train."""
        yr = year_of[t]
        rc = region_flat[flat]
        out, taken = {}, np.zeros(t.shape, bool)
        for p in ("test", "val", "train"):
            sp = spec[p]
            m = np.isin(yr, np.asarray(sp["years"]))
            if "regions" in sp:
                codes = np.asarray([code_of[n] for n in sp["regions"]], region_flat.dtype)
                m &= np.isin(rc, codes)
            if p in ("train", "val") and buf_flat is not None:
                m &= ~buf_flat[flat]
            m &= ~taken
            out[p] = m
            taken |= m
        return out

    # ---------------------------------------------- Positive Landmarks
    ign_flat = idx.ign_keys // DAY_BITS
    ign_day = idx.ign_keys % DAY_BITS
    ign_cause = idx.ign_cause
    k = np.arange(1, H + 1, dtype=np.int64)
    P = np.repeat(ign_flat, H)
    T = np.repeat(ign_day, H)
    E = np.repeat(idx.ign_events.astype(np.int64), H)
    t_l = T - np.tile(k, ign_flat.size)
    keep = (t_l >= 0) & (t_l < len(dates))
    P, T, E, t_l = P[keep], T[keep], E[keep], t_l[keep]
    # Nur der LANDMARK muss in der Saison liegen, nicht das Ereignis.
    keep = in_season[t_l]
    P, T, E, t_l = P[keep], T[keep], E[keep], t_l[keep]
    print(f"[landmarks] {t_l.size} Kandidaten aus Zuendungen nach Saisonfilter")

    keep = ~idx.has_fire_between(P, t_l - C + 1, t_l)
    P, T, E, t_l = P[keep], T[keep], E[keep], t_l[keep]
    nd, ne, nc, found = idx.next_ignition(P, t_l)
    keep = found & (nd == T)          # nur wenn T wirklich die NAECHSTE ist
    P, T, E, t_l, C_ = P[keep], T[keep], E[keep], t_l[keep], nc[keep]
    pos_tev = (T - t_l).astype(np.int64)
    print(f"[landmarks] {t_l.size} positive Landmarks")

    # ---------------------------------------------- Negative Ziehung
    def draw(n: int):
        ts = season_ts[rng.integers(season_ts.size, size=n)]
        px = valid_flat[rng.integers(n_valid, size=n)]
        return ts, px

    def label(ts: np.ndarray, px: np.ndarray):
        alive = ~idx.has_fire_between(px, ts - C + 1, ts)
        nd, ne, nc, found = idx.next_ignition(px, ts)
        ev = found & ((nd - ts) <= H) & (nd > ts)
        tev = np.where(ev, nd - ts, H).astype(np.int64)
        eid = np.where(ev, ne, -1).astype(np.int64)
        cau = np.where(ev, nc, 0).astype(np.int64)
        return alive, ev, tev, eid, cau

    pm = part_masks(t_l, P)
    rows = {}
    for part in ("train", "val", "test"):
        m = pm[part]
        rows[part] = [t_l[m], P[m], np.ones(m.sum(), np.int64), pos_tev[m],
                      E[m], C_[m].astype(np.int64)]
    print(f"[landmarks] positive je Split: "
          f"{ {k2: int(v[0].size) for k2, v in rows.items()} }")

    # ---------------------------------------------- Groesse der Straten
    # Wie viele zulaessige Kandidaten hat jeder Split ueberhaupt? Nur so
    # laesst sich das Gewicht der gezogenen Negative bestimmen.
    probe_ts, probe_px = draw(args.prevalence_probe)
    probe_alive, probe_ev, _, _, _ = label(probe_ts, probe_px)
    probe_parts = part_masks(probe_ts, probe_px)
    strata = {}
    for part in ("train", "val", "test"):
        inp = probe_parts[part]
        n_in = int(inp.sum())
        if n_in == 0:
            continue
        share = n_in / args.prevalence_probe
        alive_rate = float(probe_alive[inp].mean())
        n_cand = int(round(n_valid * season_ts.size * share))
        n_alive = int(round(n_cand * alive_rate))
        n_pos_exact = int(rows[part][0].size)
        strata[part] = {
            "probe_rows": n_in,
            "share_of_candidate_space": share,
            "alive_rate": alive_rate,
            "n_candidates": n_cand,
            "n_alive": n_alive,
            "n_positive_exact": n_pos_exact,
            "n_negative": max(1, n_alive - n_pos_exact),
            "probe_prevalence": float(probe_ev[inp & probe_alive].mean()),
        }
        print(f"[landmarks] {part:5s} Kandidaten={n_cand} zulaessig={n_alive} "
              f"positiv={n_pos_exact} -> Negative={strata[part]['n_negative']} "
              f"Kontrollpraevalenz={strata[part]['probe_prevalence']:.3e}")

    # ---------------------------------------------- Negative je Split ziehen
    targets = {
        "train": args.neg_per_pos_train * max(1, rows["train"][0].size),
        "val": args.neg_per_split_eval,
        "test": args.neg_per_split_eval,
    }
    for part, n_neg in targets.items():
        n_neg = min(n_neg, strata[part]["n_negative"])
        got = 0
        acc = [[] for _ in range(6)]
        while got < n_neg:
            ts, px = draw(min(4_000_000, (n_neg - got) * 3 + 200_000))
            alive, ev, tev, eid, cau = label(ts, px)
            m = alive & ~ev & part_masks(ts, px)[part]
            take = np.nonzero(m)[0][: n_neg - got]
            for arr, val in zip(acc, (ts[take], px[take], np.zeros(take.size, np.int64),
                                      tev[take], eid[take], cau[take])):
                arr.append(val)
            got += take.size
        for i in range(6):
            rows[part][i] = np.concatenate([rows[part][i]] + acc[i])
        strata[part]["n_negative_sampled"] = int(got)
        strata[part]["negative_weight"] = strata[part]["n_negative"] / max(1, got)
        print(f"[landmarks] {part:5s} {got} Negative gezogen, "
              f"Gewicht {strata[part]['negative_weight']:.1f}")

    # ---------------------------------------------- Ablage
    out_dir = exp / "derived_data" / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for part, (t_, p_, ev_, tev_, eid_, cau_) in rows.items():
        if t_.size == 0:
            continue
        r_, c_ = np.divmod(p_, idx.W)
        pos_mask = ev_ == 1
        w_ = np.where(pos_mask, 1.0, strata[part]["negative_weight"]).astype(np.float32)
        f = out_dir / f"{args.split}_{part}.npz"
        np.savez_compressed(f, t=t_.astype(np.int32), row=r_.astype(np.int16),
                            col=c_.astype(np.int16), event=ev_.astype(np.int8),
                            t_event=tev_.astype(np.int16), event_id=eid_.astype(np.int32),
                            year=year_of[t_].astype(np.int16), weight=w_,
                            cause=cau_.astype(np.int8))
        w_prev = float((w_ * ev_).sum() / w_.sum())
        summary[part] = {
            "n": int(t_.size), "n_positive": int(pos_mask.sum()),
            "prevalence_in_file": float(ev_.mean()),
            "prevalence_weighted": w_prev,
            "prevalence_uniform_probe": strata[part]["probe_prevalence"],
            "weight_check_relative_error": abs(w_prev - strata[part]["probe_prevalence"])
                                           / max(1e-12, strata[part]["probe_prevalence"]),
            "negative_weight": strata[part]["negative_weight"],
            "t_event_positive": {
                "p10": float(np.percentile(tev_[pos_mask], 10)) if pos_mask.any() else None,
                "p50": float(np.median(tev_[pos_mask])) if pos_mask.any() else None,
                "p90": float(np.percentile(tev_[pos_mask], 90)) if pos_mask.any() else None,
            },
            "n_unique_events": int(np.unique(eid_[pos_mask]).size),
            "n_unique_pixels": int(np.unique(p_).size),
            "cause_counts": {str(int(k)): int(v) for k, v in
                             zip(*np.unique(cau_[pos_mask], return_counts=True))},
            "years": {str(int(y)): int(n) for y, n in
                      zip(*np.unique(year_of[t_], return_counts=True))},
            "file": str(f), "bytes": int(f.stat().st_size),
            "sampling": ("alle Positiven + gezogene Negative" if part == "train"
                         else "gleichmaessige Ziehung, Praevalenz erhalten"),
        }
        print(f"[landmarks] {part:5s} n={t_.size:9d} positiv={int(pos_mask.sum()):8d} "
              f"Datei-Praev={ev_.mean():.3e} gewichtet={w_prev:.3e} "
              f"Kontrolle={strata[part]['probe_prevalence']:.3e} "
              f"(Abw. {100*summary[part]['weight_check_relative_error']:.1f} %)  "
              f"{human_bytes(f.stat().st_size)}")

    # Leakage-Kontrolle: keine Ereignis-ID ueber Splits
    ev_sets = {p: set(np.unique(rows[p][4][rows[p][2] == 1]).tolist()) - {-1}
               for p in rows if rows[p][0].size}
    cross = {f"{a}|{b}": len(ev_sets[a] & ev_sets[b])
             for a in ev_sets for b in ev_sets if a < b}
    print(f"[landmarks] gemeinsame Ereignis-IDs ueber Splits: {cross}")

    write_manifest(
        cfg, "d7_landmarks", args.config,
        {
            "split": args.split,
            "lookback": L, "horizon": H, "cooldown": C,
            "season": {"start": args.season_start, "end": args.season_end},
            "label_source": idx.label_source,
            "mask": mask_name,
            "cause_codes": {"0": "unbekannt", "1": "Blitz", "2": "Mensch"},
            "locked_channels": {str(k2): v for k2, v in LOCKED_CHANNELS.items()},
            "locked_until": "t - C",
            "n_valid_pixels": int(n_valid),
            "n_season_days": int(season_ts.size),
            "candidate_space": int(n_valid) * int(season_ts.size),
            "n_positive_landmarks_total": int(pos_tev.size),
            "train_neg_per_pos": args.neg_per_pos_train,
            "strata": strata,
            "prevalence_probe_size": args.prevalence_probe,
            "shared_event_ids_across_splits": cross,
            "prior_correction_note": (
                "Train ist durch Negativziehung angereichert. Fuer Kalibrierung "
                "und absolute Risiken muss der Prior korrigiert werden; Val und "
                "Test tragen die echte Praevalenz."
            ),
            "parts": summary,
        },
    )


if __name__ == "__main__":
    main()

"""Metriken fuer diskrete Survival-Vorhersage mit Rechtszensierung.

Alle Modelle geben dasselbe aus: diskrete Hazards ``h[k]`` fuer
``k = 1..H``, also die bedingte Wahrscheinlichkeit, dass das Ereignis am
Tag ``k`` eintritt, gegeben dass es bis dahin nicht eingetreten ist.
Daraus folgen Survival und kumulatives Risiko:

    S(k) = prod_{j<=k} (1 - h[j])          F(k) = 1 - S(k)

**Zensierung.** In unserem Landmark-Aufbau wird ausschliesslich
administrativ bei H zensiert - die Saison ist so gewaehlt, dass das
Ereignisfenster nie ueber den 30. September hinausreicht. Daraus folgt
etwas Angenehmes: fuer jedes ``tau <= H`` ist ``1{T <= tau}`` bei ALLEN
Beispielen bekannt. Brier-Score und AUC brauchen deshalb keine
IPCW-Gewichtung; die uebliche Fehlerquelle zensierter Auswertung entfaellt.

**Gewichte.** Die Landmark-Tabelle ist geschichtet gezogen (alle Positiven,
Bruchteil der Negativen). Jede Metrik nimmt deshalb ein Gewicht je Zeile
entgegen. Ohne das waeren alle praevalenzabhaengigen Groessen - PR-AUC,
Brier, Kalibrierung - um Groessenordnungen daneben.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-7


# numpy hat np.trapz in 2.0 zu np.trapezoid umbenannt; die alte Schreibweise
# entfaellt in 3.0. Welche Version im Container liegt, haengt am Image - Job
# 5847 starb an genau dieser Zeile, weil dort numpy < 2 lief. Einmal aufloesen
# statt an jeder Aufrufstelle zu verzweigen.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def survival_from_hazard(h: np.ndarray) -> np.ndarray:
    """(N, H) Hazards -> (N, H) Survival S(k).

    Nach unten wird NICHT geklemmt: Ein vorhergesagter Nullhazard ist
    zulaessig und soll das Survival unveraendert lassen. Eine untere
    Klemme auf EPS wuerde sich ueber 30 Schritte zu einem Fehler von rund
    3e-6 aufsummieren - klein, aber unnoetig, denn hier wird kein
    Logarithmus gebildet. Die NLL klemmt separat, dort ist es noetig.
    """
    return np.cumprod(1.0 - np.clip(h, 0.0, 1 - EPS), axis=1)


def cif_from_hazard(h: np.ndarray) -> np.ndarray:
    """Kumulatives Risiko F(k) = 1 - S(k). Per Konstruktion monoton."""
    return 1.0 - survival_from_hazard(h)


def negative_log_likelihood(h, event, t_event, weight=None,
                            chunk: int = 250_000) -> float:
    """Diskrete Survival-NLL, rechtszensiert bei H.

    In Bloecken gerechnet. Die Zwischenschritte ``log1p`` und ``cumsum``
    haben je die volle Groesse der Hazardmatrix; bei 2,5 Mio. Zeilen und
    H = 100 waeren das in float64 zweimal 2 GB gleichzeitig. Blockweise
    bleibt der Spitzenbedarf bei wenigen hundert Megabyte, und das Ergebnis
    ist identisch - es ist eine gewichtete Summe.
    """
    h = np.asarray(h)
    N, H = h.shape
    event = np.asarray(event).astype(bool)
    k = np.asarray(t_event).astype(int)
    w = np.ones(N) if weight is None else np.asarray(weight, float)

    acc, wsum = 0.0, 0.0
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        hb = np.clip(np.asarray(h[a:b], np.float64), EPS, 1 - EPS)
        kb, eb, wb = k[a:b], event[a:b], w[a:b]
        csum = np.cumsum(np.log1p(-hb), axis=1)
        idx = np.arange(b - a)
        prior = np.where(kb > 1, csum[idx, np.maximum(kb - 2, 0)], 0.0)
        ll = np.where(eb, np.log(hb[idx, kb - 1]) + prior, csum[:, H - 1])
        acc += float((wb * ll).sum())
        wsum += float(wb.sum())
        del hb, csum
    return -acc / wsum


def brier_at(h, event, t_event, tau: int, weight=None, cif=None) -> float:
    """Brier-Score fuer 'Ereignis bis Tag tau'. Ohne IPCW - siehe Modulkopf.

    ``cif`` erlaubt es, die einmal berechnete Risikomatrix durchzureichen.
    Ohne das berechnete jede Metrik sie neu - bei 2,2 Mio. Zeilen und
    100 Tagen sind das 1,8 GB je Aufruf, und ``evaluate`` ruft rund zwei
    Dutzend Mal auf.
    """
    F = (cif_from_hazard(h) if cif is None else cif)[:, tau - 1]
    y = (np.asarray(event).astype(bool) & (np.asarray(t_event) <= tau)).astype(float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    return float((w * (F - y) ** 2).sum() / w.sum())


def integrated_brier(h, event, t_event, weight=None, cif=None) -> float:
    H = h.shape[1]
    F = cif_from_hazard(h) if cif is None else cif
    return float(np.mean([brier_at(h, event, t_event, t, weight, F)
                          for t in range(1, H + 1)]))


def _roc_pr(y, score, w):
    """ROC-AUC und Average Precision, BINDUNGSSICHER.

    Bei gleichen Scores darf die Reihenfolge im Array das Ergebnis nicht
    bestimmen. Genau das passierte vorher: Die Landmark-Tabellen fuehren
    alle Positiven zuerst, ``argsort`` ist stabil, und ein konstanter
    Praediktor bekam dadurch AUC 1,0 statt 0,5. Deshalb werden Punkte mit
    identischem Score zu EINEM Schwellwert zusammengefasst, bevor die
    Kurve aufgebaut wird.
    """
    order = np.argsort(-score, kind="mergesort")
    s_, y_, w_ = score[order], y[order], w[order]
    # Gruppengrenzen: letzter Index je eindeutigem Score
    last = np.r_[np.nonzero(np.diff(s_))[0], s_.size - 1]
    tp = np.cumsum(w_ * y_)[last]
    fp = np.cumsum(w_ * (1 - y_))[last]
    P, N = tp[-1], fp[-1]
    if P <= 0 or N <= 0:
        return float("nan"), float("nan"), None
    tpr, fpr = tp / P, fp / N
    auc = float(_trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    prec = tp / np.maximum(tp + fp, EPS)
    ap = float(np.sum(np.diff(np.r_[0.0, tpr]) * prec))
    return auc, ap, (fpr, tpr, prec)


def auc_and_pr_at(h, event, t_event, tau: int, weight=None, cif=None):
    """ROC-AUC und PR-AUC (Average Precision) fuer den Horizont tau."""
    score = (cif_from_hazard(h) if cif is None else cif)[:, tau - 1]
    y = (np.asarray(event).astype(bool) & (np.asarray(t_event) <= tau)).astype(float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    auc, ap, _ = _roc_pr(y, score, w)
    return auc, ap


def calibration(h, event, t_event, tau: int, weight=None, n_bins: int = 15,
                cif=None):
    """Zuverlaessigkeitsdiagramm und erwarteter Kalibrierungsfehler.

    Gebinnt wird nach VORHERGESAGTER Wahrscheinlichkeit auf Quantilen, weil
    bei einer Praevalenz um 1e-3 aequidistante Bins fast leer waeren.
    """
    p = (cif_from_hazard(h) if cif is None else cif)[:, tau - 1]
    y = (np.asarray(event).astype(bool) & (np.asarray(t_event) <= tau)).astype(float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if qs.size < 2:
        # Konstanter Praediktor: genau ein Bin. Der Kalibrierungsfehler ist
        # dann wohldefiniert - die Differenz zwischen Vorhersage und
        # beobachteter Rate - und NICHT undefiniert.
        pred = float((w * p).sum() / w.sum())
        obs = float((w * y).sum() / w.sum())
        return {"bins": [{"lo": pred, "hi": pred, "predicted": pred,
                          "observed": obs, "weight": float(w.sum()),
                          "n": int(y.size)}],
                "ece": abs(pred - obs)}
    bins, ece, wtot = [], 0.0, w.sum()
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (p >= lo) & (p < hi if hi < qs[-1] else p <= hi)
        if not m.any():
            continue
        ww = w[m].sum()
        pred = float((w[m] * p[m]).sum() / ww)
        obs = float((w[m] * y[m]).sum() / ww)
        bins.append({"lo": float(lo), "hi": float(hi), "predicted": pred,
                     "observed": obs, "weight": float(ww), "n": int(m.sum())})
        ece += ww * abs(pred - obs)
    return {"bins": bins, "ece": float(ece / wtot)}


def recall_at_false_alarm_rate(h, event, t_event, tau: int, far: float,
                               weight=None, cif=None):
    """Ereignis-Recall bei fest vorgegebener Falschalarmrate.

    Der operativ relevante Betriebspunkt: Wie viele Braende faengt man,
    wenn man bereit ist, auf einem festen Anteil der Flaeche zu alarmieren?
    """
    p = (cif_from_hazard(h) if cif is None else cif)[:, tau - 1]
    y = (np.asarray(event).astype(bool) & (np.asarray(t_event) <= tau)).astype(float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
    order = np.argsort(-p)
    ys, ws = y[order], w[order]
    fp = np.cumsum(ws * (1 - ys))
    tp = np.cumsum(ws * ys)
    N, P = fp[-1], tp[-1]
    if P <= 0 or N <= 0:
        return float("nan"), float("nan")
    j = int(np.searchsorted(fp / N, far, "right")) - 1
    if j < 0:
        return 0.0, float(p[order][0])
    return float(tp[j] / P), float(p[order][j])


def lead_time(h, event, t_event, threshold: float, weight=None, cif=None):
    """Vorwarnzeit: Tage zwischen erstem Alarm und Ereignis.

    Alarm ist der erste Tag k, an dem das kumulative Risiko den Schwellwert
    ueberschreitet. Nur ueber tatsaechliche Ereignisse gemittelt.
    """
    F = cif_from_hazard(h) if cif is None else cif
    ev = np.asarray(event).astype(bool)
    k = np.asarray(t_event).astype(int)
    w = np.ones(len(k)) if weight is None else np.asarray(weight, float)
    if not ev.any():
        return {"mean": float("nan"), "median": float("nan"), "detected_share": 0.0}
    Fe, ke, we = F[ev], k[ev], w[ev]
    above = Fe >= threshold
    first = np.where(above.any(1), above.argmax(1) + 1, np.iinfo(np.int32).max)
    hit = first <= ke
    if not hit.any():
        return {"mean": float("nan"), "median": float("nan"), "detected_share": 0.0}
    lt = (ke - first)[hit].astype(float)
    ww = we[hit]
    order = np.argsort(lt)
    csum = np.cumsum(ww[order])
    med = float(lt[order][int(np.searchsorted(csum, csum[-1] / 2))])
    return {"mean": float((ww * lt).sum() / ww.sum()), "median": med,
            "detected_share": float(ww.sum() / we.sum())}


def concordance_index(risk, event, t_event, weight=None) -> tuple[float, float]:
    """Gewichteter Harrell-C ueber die gesamte Zeitachse.

    Die AUC(tau) beurteilt die Rangordnung an EINEM Stichtag. Der C-Index
    fasst sie ueber alle Zeitpunkte zusammen: Von je zwei Landmarks, bei denen
    einer nachweislich frueher gezuendet hat, soll genau dieser das hoehere
    Risiko tragen.

    Vergleichbar sind Paare mit ``T_i < T_j`` und ``event_i = 1``; konkordant
    sind sie, wenn ``risk_i > risk_j``. Bindungen im Risiko zaehlen halb,
    Bindungen in der Zeit sind nicht vergleichbar.

    Zensierung ist hier rein administrativ bei H, also fuer alle gleich. Eine
    IPCW-Korrektur (Uno-C) waere damit die Identitaet - siehe die Begruendung
    im Metrikabschnitt.

    Gewichte gehen als Produkt ins Paar ein: Ohne sie misst man den C-Index
    der angereicherten STICHPROBE, nicht den der Grundgesamtheit.

    Rueckgabe: (C, Summe der vergleichbaren Paargewichte).
    """
    risk = np.asarray(risk, float)
    n = risk.size
    w = np.ones(n) if weight is None else np.asarray(weight, float)
    ev = np.asarray(event).astype(bool)
    times = np.asarray(t_event, np.int64)

    order = np.argsort(risk, kind="mergesort")
    rs, ws = risk[order], w[order]
    pos = np.empty(n, np.int64)
    pos[order] = np.arange(n)
    lo = np.searchsorted(rs, rs, "left")[pos]      # Grenzen der Bindungsgruppe
    hi = np.searchsorted(rs, rs, "right")[pos]

    seen = np.zeros(n, bool)                       # bereits einsortiert, risikosortiert
    conc = disc = tie = 0.0
    for t in range(int(times.max()), 0, -1):
        grp = np.flatnonzero(times == t)
        if grp.size == 0:
            continue
        e = grp[ev[grp]]
        if e.size and seen.any():
            cum = np.concatenate(([0.0], np.cumsum(np.where(seen, ws, 0.0))))
            tot = cum[-1]
            conc += float((w[e] * cum[lo[e]]).sum())
            tie += float((w[e] * (cum[hi[e]] - cum[lo[e]])).sum())
            disc += float((w[e] * (tot - cum[hi[e]])).sum())
        seen[pos[grp]] = True

    denom = conc + disc + tie
    return ((conc + 0.5 * tie) / denom if denom > 0 else float("nan")), denom


def evaluate(h, event, t_event, weight=None, horizons=(1, 7, 14, 30, 60, 100),
             far=0.01, alarm_threshold=0.01) -> dict:
    """Vollstaendiger Metriksatz nach Phase 8.1 des Arbeitsplans."""
    H = h.shape[1]
    hs = [t for t in horizons if t <= H]
    # EINMAL berechnen und durchreichen. float32 genuegt: die Risiken liegen
    # um 1e-3, und der Rundungsfehler ist um Groessenordnungen kleiner als
    # der statistische Fehler bei diesen Fallzahlen.
    F = cif_from_hazard(np.asarray(h, np.float32)).astype(np.float32)
    out = {
        "n": int(len(event)),
        "n_events": int(np.asarray(event).sum()),
        "nll": negative_log_likelihood(h, event, t_event, weight),
        "integrated_brier": integrated_brier(h, event, t_event, weight, F),
        "horizons": {},
        "lead_time": lead_time(h, event, t_event, alarm_threshold, weight, F),
    }
    # Rangordnung ueber die GANZE Zeitachse, als Ergaenzung zu AUC(tau).
    # Risiko ist F am Horizont - die Groesse, nach der auch sortiert wuerde.
    c_idx, c_pairs = concordance_index(F[:, H - 1], event, t_event, weight)
    out["concordance_index"] = c_idx
    out["concordance_pairs"] = c_pairs
    for t in hs:
        auc, ap = auc_and_pr_at(h, event, t_event, t, weight, F)
        cal = calibration(h, event, t_event, t, weight, cif=F)
        rec, thr = recall_at_false_alarm_rate(h, event, t_event, t, far, weight, F)
        out["horizons"][str(t)] = {
            "brier": brier_at(h, event, t_event, t, weight, F),
            "roc_auc": auc, "pr_auc": ap, "ece": cal["ece"],
            "reliability": cal["bins"],
            f"recall_at_far_{far}": rec, "threshold_at_far": thr,
        }
    return out

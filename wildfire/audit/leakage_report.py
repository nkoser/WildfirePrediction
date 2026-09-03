"""Erzeugt ``leakage_report.md`` aus den Pruefungen aller Stufen.

Der Arbeitsplan verlangt neben den Split-Dateien einen eigenen
Leakage-Bericht. Er ist bewusst getrennt vom Datenaudit: Das Audit
beschreibt, WAS in den Daten steht; dieser Bericht beantwortet die eine
Frage, ob ein Modell die Antwort auf einem Weg sehen kann, den es zum
Vorhersagezeitpunkt nicht haette.

Behandelt werden vier Ebenen:

  1. Die gelieferten Fenster-Caches - dort ist Leakage nachgewiesen und
     quantifiziert. Deshalb werden sie nicht verwendet.
  2. Kanaele, die Rohmaterial des Labels sind, und ihre Sperre.
  3. Die Splits: gemeinsame Ereignisse, Tage, Pixel, Pufferzonen.
  4. Der Landmark-Aufbau: Zensierung, Zeitfenster, Gewichte.

Alle Zahlen stammen aus den Stufen-Manifesten. Fehlt eine Stufe, wird das
im Bericht ausgewiesen statt uebergangen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.audit.common import load_config, provenance

STAGES = ["s5_alignment", "s6_label_def", "s2c_bt_leakage",
          "d3_splits", "d7_landmarks", "d10_sanity"]


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "ja" if x else "nein"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    if isinstance(x, int):
        return f"{x:,}".replace(",", " ")
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])

    m, missing = {}, []
    for st in STAGES:
        p = exp / "data_manifests" / f"{st}.json"
        if p.exists():
            m[st] = json.loads(p.read_text())
        else:
            missing.append(st)

    L: list[str] = []
    A = L.append
    p = provenance(args.config, "leakage_report")

    A("# Leakage-Bericht")
    A("")
    A(f"- Erzeugt: `{p['timestamp_utc']}` von `wildfire/audit/leakage_report.py`")
    A(f"- Git-Commit: `{p['git_commit'] or 'n/a'}` (dirty: {p['git_dirty']})")
    A(f"- Config-Hash: `{p['config_hash']}`")
    if missing:
        A(f"- **Fehlende Stufen:** `{missing}` - die zugehoerigen Abschnitte sind unvollstaendig.")
    A("")
    A("Dieser Bericht beantwortet eine einzige Frage: Kann ein Modell die")
    A("Antwort auf einem Weg sehen, den es zum Vorhersagezeitpunkt nicht haette?")
    A("")

    # ---------------------------------------------------------------- 1
    A("## 1. Die gelieferten Caches sind unbrauchbar")
    A("")
    ali, lab = m.get("s5_alignment", {}), m.get("s6_label_def", {})
    lk = ali.get("leakage") or {}
    A("Der Datensatz bringt vier vorgebaute Fenster-Caches mit. Sie werden in")
    A("diesem Projekt **nicht verwendet**. Grund:")
    A("")
    A(f"- Das Eingabefenster endet am Sample-Datum D selbst "
      f"(Zeitversatz {lk.get('last_timestep_offsets_observed')} Tage).")
    A(f"- Die Feuermaske ist Eingabekanal 0: `{fmt(lk.get('fire_mask_in_input_channels'))}`.")
    A(f"- Die Labelregel ist empirisch `{(lab.get('best_rule') or {}).get('rule')}` "
      f"mit {fmt(100*(lab.get('best_rule') or {}).get('agreement', 0), 3)} % Uebereinstimmung -")
    A("  also **Feuer im Patch am letzten Eingabezeitschritt**.")
    A("")
    sc = (lab.get("input_fire_shortcut") or {}).get("trivial_input_fire_classifier") or {}
    if sc:
        A("Messung des Kurzschlusses - eine einzige Boolesche Abfrage auf Kanal 0,")
        A("ohne Modell, ohne Lernen:")
        A("")
        A("| Groesse | Wert |")
        A("|---|---|")
        for k, v in sc.items():
            A(f"| {k} | {fmt(v, 4)} |")
        A("")
        A(f"Bei sauberem Aufbau muesste diese Regel bei der Praevalenz landen "
          f"(rund 33 %). Sie erreicht {fmt(100*sc.get('accuracy', 0), 3)} %.")
        A("")

    # ---------------------------------------------------------------- 2
    A("## 2. Kanaele, die Rohmaterial des Labels sind")
    A("")
    bt = m.get("s2c_bt_leakage", {})
    A("Das Label MCD14A1 wird von MODIS aus den Helligkeitstemperaturen der")
    A("Baender 20/21/31/32 berechnet. Diese Baender sind als Kanaele 38-41 im")
    A("Datensatz enthalten - sie sind nicht bloss korreliert mit dem Ziel,")
    A("sondern dessen Ausgangsmaterial.")
    A("")
    if bt.get("measurements_paired"):
        by = {}
        for r in bt["measurements_paired"]:
            by.setdefault(r["name"], {})[r["lag_days"]] = r["cohens_d_paired"]
        A("Gepaarter Effektstaerkenverlauf (Zuwachs seit D-14, Standortunterschied")
        A("herausgekuerzt), Cohen's d:")
        A("")
        A("| Kanal | D-7 | D-3 | D-1 | **D** |")
        A("|---|---|---|---|---|")
        seen = []
        for r in bt["measurements_paired"]:
            if r["name"] in seen:
                continue
            seen.append(r["name"])
            d = by[r["name"]]
            A(f"| ch{r['channel']} `{r['name']}` | {fmt(d.get(-7))} | {fmt(d.get(-3))} | "
              f"{fmt(d.get(-1))} | **{fmt(d.get(0))}** |")
        A("")
        A("Die 4-um-Feuerbaender erreichen am Detektionstag die hoechsten Werte")
        A("aller gemessenen Kanaele - hoeher als FWI und VPD, also hoeher als")
        A("Kanadas operationeller Gefahrenindex. Ihr Vorlauf bei D-7 ist dagegen")
        A("nicht groesser als der der unauffaelligen Referenzbaender. Das Muster")
        A("passt zur Strahlung des Feuers selbst, nicht zu seiner Ursache.")
        A("")
    lm = m.get("d7_landmarks", {})
    if lm.get("locked_channels"):
        A("**Sperre im eigenen Aufbau.** Diese Kanaele sind nur bis `t - C` sichtbar,")
        A(f"mit C = {fmt(lm.get('cooldown'))} Tagen. Die letzten C Zeitschritte werden")
        A("genullt UND in der Beobachtungsmaske als unbeobachtet markiert - eine Null")
        A("allein waere ein gueltiger Messwert und nicht von Abwesenheit zu unterscheiden.")
        A("")
        A("| Kanal | Begruendung |")
        A("|---|---|")
        for k, v in lm["locked_channels"].items():
            A(f"| {k} | {v} |")
        A("")

    # ---------------------------------------------------------------- 3
    A("## 3. Splits")
    A("")
    sp = m.get("d3_splits", {})
    if sp.get("splits"):
        A(f"Alle Pruefungen bestanden: **{fmt(sp.get('all_checks_passed'))}**")
        A("")
        A("| Split | Train | Val | Test | gemeinsame Ereignisse | gemeinsame Pixel-Zeitpunkte | Mindestabstand |")
        A("|---|---|---|---|---|---|---|")
        for name, s in sp["splits"].items():
            c, ch = s["counts"], s["checks"]
            ev = max(ch["shared_event_ids"].values())
            px = max(ch["shared_pixel_timepoints"].values())
            md = ch.get("min_train_test_pixel_distance")
            A(f"| `{name}` | {fmt(c['train'])} | {fmt(c['val'])} | {fmt(c['test'])} | "
              f"{ev} | {px} | {fmt(md)} |")
        A("")
        A("**Ereignisgruppierung schlaegt Regionenregel.** Ein Brand kann eine")
        A("Provinzgrenze ueberschreiten; das Pufferband faengt das nicht ab, weil")
        A("ein Ereignis weiter reichen kann als der Puffer. Beruehrt ein Ereignis")
        A("den Test, gehoert es ganz dorthin.")
        A("")
        for name, s in sp["splits"].items():
            if s.get("dropped_by_event_regrouping"):
                A(f"- `{name}`: {s['dropped_by_event_regrouping']} Landmarks deswegen verworfen")
        A("")
        if sp.get("buffers"):
            A("**Raeumliche Pufferzonen** verhindern ueberlappende Eingabepatches:")
            A("")
            A("| Split | Pufferbreite | Pixel im Puffer | Anteil der Flaeche |")
            A("|---|---|---|---|")
            for name, b in sp["buffers"].items():
                A(f"| `{name}` | {b['half_patch']} px | {fmt(b['n_buffer_pixels'])} | "
                  f"{fmt(100*b['share_of_valid'], 3)} % |")
            A("")

    # ---------------------------------------------------------------- 4
    A("## 4. Landmark-Aufbau")
    A("")
    if lm:
        A(f"- Beobachtungsfenster L = {fmt(lm.get('lookback'))} Tage, Horizont H = "
          f"{fmt(lm.get('horizon'))} Tage, Cooldown C = {fmt(lm.get('cooldown'))} Tage")
        A(f"- Saison {lm.get('season', {}).get('start')} bis {lm.get('season', {}).get('end')}")
        A(f"- Gemeinsame Ereignis-IDs ueber Splits: `{lm.get('shared_event_ids_across_splits')}`")
        A("")
        A("**Zensierung ausschliesslich administrativ bei H.** Die Saison ist so")
        A("gewaehlt, dass das Ereignisfenster nie ueber den 30. September")
        A("hinausreicht. Daraus folgt, dass fuer jedes tau <= H der Indikator")
        A("1{T <= tau} bei allen Beispielen bekannt ist - Brier-Score und AUC")
        A("brauchen deshalb keine IPCW-Gewichtung.")
        A("")
        A("**Geschichtete Ziehung.** Alle Positiven, Negative mit bekannter Rate,")
        A("Gewicht je Zeile. Ohne das waeren alle praevalenzabhaengigen Metriken")
        A("um Groessenordnungen daneben.")
        A("")
        if lm.get("parts"):
            A("| Split | Zeilen | Positive | Praevalenz in der Datei | gewichtet | Kontrollziehung |")
            A("|---|---|---|---|---|---|")
            for part, s in lm["parts"].items():
                A(f"| {part} | {fmt(s['n'])} | {fmt(s['n_positive'])} | "
                  f"{fmt(s.get('prevalence_in_file'))} | {fmt(s.get('prevalence_weighted'))} | "
                  f"{fmt(s.get('prevalence_uniform_probe'))} |")
            A("")

    # ---------------------------------------------------------------- 5
    A("## 5. Unabhaengige Kontrolle an echten Daten")
    A("")
    sa = m.get("d10_sanity")
    if sa:
        A(f"- Labelkette gegen Rohdaten neu gerechnet: "
          f"{fmt(sa.get('n_label_ok'))}/{fmt(sa.get('n_label_checks'))} identisch")
        A(f"- Zeitausrichtung byte-genau: "
          f"{fmt(sa.get('n_align_ok'))}/{fmt(sa.get('n_align_checks'))}")
        A(f"- Verletzungen der Kanalsperre: {len(sa.get('lock_violations') or [])}")
        A(f"- Gesamt: **{fmt(sa.get('passed'))}**")
    else:
        A("Stufe `d10_sanity` liegt noch nicht vor.")
    A("")

    A("## 6. Was offen bleibt")
    A("")
    A("- **Gapfilling und Interpolation.** Die Kanaele 40/41 (LAI/FPAR) sind aus")
    A("  4-Tage-Kompositen interpoliert, 42-47 laut Dokumentation gapfilled. Ob")
    A("  dabei ueber den Zuendzeitpunkt hinweg interpoliert wurde, ist aus den")
    A("  Dateien nicht pruefbar - die Methode ist nirgends dokumentiert.")
    A("- **Detektionsverzoegerung.** Bricht ein Feuer an t aus und meldet MODIS es")
    A("  erst an t+2, steckt seine Waerme schon in den Eingabekanaelen, waehrend")
    A("  das Label spaeter liegt. Der Cooldown daempft das, beseitigt es nicht.")
    A("- **Bestaetigungsrate gegen die Branddatenbank** haengt an der gewaehlten")
    A("  Toleranz und ist ohne Brandperimeter nicht interpretierbar.")
    A("")

    out = exp / "reports" / "leakage_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"[leakage] {out}  ({len(''.join(L))/1000:.1f} kB)")
    if missing:
        print(f"[leakage] fehlende Stufen: {missing}")


if __name__ == "__main__":
    main()

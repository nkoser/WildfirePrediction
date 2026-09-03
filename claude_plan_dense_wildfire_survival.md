# Arbeitsplan für Claude: Dense Spatiotemporal Survival Forecasting für Waldbrände

## Rolle und übergeordnetes Ziel

Du arbeitest als wissenschaftlicher ML-Engineer an einem Forschungsprototyp für eine potenzielle CVPR-Einreichung. Ziel ist nicht, lediglich ein bestehendes Deep-DCTM auf neue Daten anzuwenden. Es soll untersucht werden, ob sich aus multimodalen räumlich-zeitlichen Kanada-Karten für jeden 1-km-Pixel eine kalibrierte Verteilung der Zeit bis zu einer zukünftigen Waldbrandzündung vorhersagen lässt.

Die allgemeine Aufgabe soll als **Dense Spatiotemporal Survival Forecasting** formuliert werden:

- Input: eine Folge räumlich ausgerichteter Karten mit mehreren dynamischen und statischen Kanälen;
- Output: für jeden Pixel eine Survival-Kurve beziehungsweise Hazard-Werte für mehrere zukünftige Zeitschritte;
- Hauptereignis: möglichst die **Entstehung eines neuen Feuers (Ignition)** und nicht lediglich das Vorhandensein verbrannter Fläche;
- wissenschaftliches Ziel: prüfen, ob ein Survival-Modell gegenüber einfacher Next-Day-Klassifikation einen messbaren Vorteil bei mehreren Prognosehorizonten, Zensierung und Kalibrierung besitzt.

## Wichtige Arbeitsregeln

1. Beginne nicht sofort mit der Implementierung eines großen Modells. Führe zuerst einen vollständigen Daten- und Leakage-Audit durch.
2. Triff keine stillen Annahmen. Dokumentiere unklare Datenfelder, Einheiten, Zeitauflösungen, Koordinatensysteme und Definitionen.
3. Verändere bestehende Daten niemals in-place. Abgeleitete Daten, Splits, Logs und Checkpoints müssen getrennt gespeichert werden.
4. Jeder Lauf muss reproduzierbar sein: Konfiguration, Git-Commit, Seed, Datenversion, Split-Version, Umgebung, Kommando, Log und Metriken speichern.
5. **Ausnahmslos alle ausführbaren Arbeitsschritte müssen auf dem Cluster über `sbatch` gestartet werden.** Dies gilt nicht nur für rechenintensive Verarbeitung und Training, sondern auch für Dateninventarisierung, Datenanalyse, Preprocessing, Split-Erzeugung, Sanity Checks, Tests, Inferenz, Evaluation, Metrikberechnung und Figure-Generierung. Führe keine Python-Skripte, Analyseprogramme, Notebooks oder sonstigen Pipeline-Schritte direkt auf dem Login-Node aus. Auf dem Login-Node sind nur nicht-rechenintensive Shell-Aktionen wie Navigation, das Lesen beziehungsweise Editieren kleiner Textdateien, `git status` und das Einreichen oder Überprüfen von Slurm-Jobs erlaubt. Orientiere dich bei Ressourcen, Modulen, Umgebung und Logging an vorhandenen Slurm-Skripten im Repository.
6. Schreibe modularen, getesteten Code. Keine Logik ausschließlich in Notebooks. Notebooks sind nur für Exploration und Visualisierung zulässig.
7. Implementiere zuerst kleine, überprüfbare Baselines. Ein komplexer Ansatz darf erst gestartet werden, wenn Datenpipeline, Splits und Evaluation anhand eines kleinen Ausschnitts validiert sind.
8. Vermeide räumliche, zeitliche und ereignisbasierte Leakage. Ein zufälliger Split einzelner Pixel-Zeitpunkte ist für die Hauptergebnisse nicht zulässig.
9. Erfinde keine Ergebnisse. Markiere klar, was implementiert, erfolgreich ausgeführt, nur geplant oder derzeit blockiert ist.

## Vom Nutzer bereitzustellender Datenpfad

Der Nutzer wird einen konkreten Pfad angeben, unter dem die vorhandenen Daten zu finden sind. **Warte auf diesen Pfad und erfinde keinen Datenort.** Behandle den angegebenen Pfad zunächst als read-only. Verschiebe, lösche, überschreibe oder reorganisiere dort keine Dateien.

Nachdem der Pfad angegeben wurde, ist die erste fachliche Aufgabe ausschließlich die Analyse der vorhandenen Daten. Auch diese Analyse muss vollständig über einen oder mehrere `sbatch`-Jobs erfolgen. Erstelle dafür ein kleines, reproduzierbares Audit-Skript und ein passendes Slurm-Skript. Starte zu diesem Zeitpunkt noch kein Preprocessing für das Training und kein Modelltraining.

Untersuche rekursiv und ressourcenschonend:

- vorhandene Dateien, Ordnerstruktur und Dateiformate;
- Anzahl und Größe der Dateien, ohne unnötig alle großen Dateien vollständig in den Speicher zu laden;
- räumliche Dimensionen, Auflösung, Projektion und geographische Abdeckung;
- Zeitpunkte, Zeitauflösung, Zeitraum, Lücken und Duplikate;
- vorhandene Variablen beziehungsweise Kanäle einschließlich Einheiten und Datentypen;
- Missing-/NoData-Werte und offensichtliche Qualitätsprobleme;
- Feuerlabels, Zündpunkte, Brandmasken, Perimeter, Ursachen und Ereignis-IDs;
- Möglichkeiten, einzelne Modalitäten räumlich und zeitlich miteinander zu verknüpfen.

Gib dem Nutzer nach Abschluss der Audit-Jobs **zunächst eine verständliche Zusammenfassung**, bevor du weitere Schritte ausführst. Diese Zusammenfassung muss mindestens enthalten:

1. welche Datensätze und Dateiformate vorhanden sind;
2. welche Informationen beziehungsweise Variablen pro Pixel und Zeitpunkt verfügbar sind;
3. räumliche und zeitliche Auflösung sowie abgedeckter Zeitraum;
4. welche Zielinformationen zu Waldbränden vorhanden sind;
5. ob echte Zündungen von bloßer Brandfläche beziehungsweise Ausbreitung unterschieden werden können;
6. Umfang, Klassenverteilung, fehlende Werte und erkennbare Qualitätsprobleme;
7. mögliche Leakage-Risiken oder zeitlich erst nach dem Ereignis verfügbare Variablen;
8. welche Informationen für das geplante Survival-Setup noch fehlen oder unklar sind;
9. eine konkrete Empfehlung für die sinnvollste Zieldefinition und die nächsten Schritte.

Speichere zusätzlich einen detaillierten `data_audit.md` sowie ein maschinenlesbares Datenmanifest. **Stoppe danach und warte auf die Rückmeldung des Nutzers.** Implementiere noch keine Trainingspipeline und starte kein Training, bevor der Nutzer die Zusammenfassung gesehen und das weitere Vorgehen bestätigt hat.

## Vor Beginn einzusetzende Pfade

Bitte frage nach fehlenden Angaben oder trage sie nach einer Repository-Inspektion ein:

```text
REPOSITORY=<PATH_TO_REPOSITORY>
RAW_DATA=<PATH_WILL_BE_PROVIDED_BY_USER>
EXPERIMENT_ROOT=/home/nik/scratch/<EXPERIMENT_NAME>
CODE_DIR=<NEW_SUBDIRECTORY_INSIDE_REPOSITORY>
```

Erzeuge unter `EXPERIMENT_ROOT` mindestens:

```text
configs/
data_manifests/
derived_data/
splits/
logs/
checkpoints/
predictions/
metrics/
figures/
artifacts/
reports/
```

Der Code selbst gehört in `CODE_DIR` innerhalb des Repositorys. Große Daten, Checkpoints und Resultate gehören nach `EXPERIMENT_ROOT`, nicht in Git.

---

## Phase 1: Repository- und Daten-Audit

### 1.1 Repository verstehen

- Lies vorhandene `README`-, Konfigurations-, Dataset- und Slurm-Dateien.
- Identifiziere vorhandene Umgebungen, Coding-Konventionen und Datenloader.
- Prüfe den aktuellen Git-Status, ohne bestehende Änderungen zu überschreiben.
- Erstelle anschließend einen kurzen Implementierungsplan mit den Dateien, die neu angelegt oder verändert werden sollen.

### 1.2 Dateninventar am bereitgestellten Pfad erstellen

Verwende hierfür ausschließlich den vom Nutzer angegebenen `RAW_DATA`-Pfad. Ermittle und dokumentiere über `sbatch`:

- räumliche Abdeckung, Rastergröße und Projektion;
- räumliche Auflösung und ob tatsächlich jeder Pixel 1 km × 1 km repräsentiert;
- Zeitauflösung, Zeitraum, fehlende Zeitpunkte und unterschiedliche Messfrequenzen;
- dynamische Kanäle, z. B. Temperatur, Niederschlag, Wind, Bodenfeuchte, Dürreindizes und Biomasse;
- statische Kanäle, z. B. Höhe, Hangneigung, Landbedeckung, Straßen, Siedlungen und Gewässer;
- Feuerdaten: Zündpunkt, Startzeitpunkt, Ursache, Brandperimeter, verbrannte Fläche und Ereignis-ID;
- Einheiten, NoData-Werte, Qualitätsflags und Quellen jedes Kanals.

Erzeuge ein maschinenlesbares Manifest und einen Audit-Bericht. Der Bericht soll insbesondere beantworten:

1. Sind echte Zündpunkte mit hinreichend genauer Zeit- und Ortsinformation vorhanden?
2. Enthalten die Labels nur Brandperimeter oder tatsächlich neue Ignitions?
3. Können einzelne verbrannte Pixel einer gemeinsamen Brandereignis-ID zugeordnet werden?
4. Welche Kovariaten wären zum Vorhersagezeitpunkt real verfügbar?
5. Gibt es Kanäle, die Informationen aus der Zukunft oder aus nachträglicher Rekonstruktion enthalten?
6. Wie häufig treten Feuer pro Pixel, Jahr und Region auf?
7. Wie viele Pixel brennen wiederholt?

### 1.3 Frühe Go-/No-Go-Entscheidung

- **Wenn Zündpunkte vorhanden sind:** Primäraufgabe ist Ignition Forecasting.
- **Wenn nur Brandmasken/Perimeter vorhanden sind:** Bezeichne das Ziel nicht als Zündung. Formuliere zunächst Fire Occurrence beziehungsweise Spread Forecasting und dokumentiere diese Einschränkung.
- **Wenn zeitliche Informationen zu grob sind:** Passe Vorhersageintervalle und Horizonte an die tatsächliche Labelauflösung an.

Liefere am Ende dieser Phase eine kompakte Entscheidungsvorlage. Noch kein großes Training starten.

---

## Phase 2: Zielvariable und statistische Formulierung

### 2.1 Primäre Zieldefinition

Für Pixel \(i\) und Landmark-Zeitpunkt \(t\) soll aus einer Historie der Länge \(L\)

\[
X_{i,t-L+1:t}
\]

die Zeit bis zur nächsten Zündung \(T_i-t\) modelliert werden. Die gewünschte Ausgabe ist

\[
S_i(\tau\mid t)=P(T_i-t>\tau\mid X_{t-L+1:t})
\]

oder äquivalent ein diskreter Hazard

\[
h_i(k\mid t)=P(T_i-t=k\mid T_i-t\geq k, X_{t-L+1:t}).
\]

Aus dem Hazard müssen monotone kumulative Risiken folgen:

\[
F_i(H\mid t)=1-\prod_{k=1}^{H}(1-h_i(k\mid t)).
\]

### 2.2 Landmark-Datensatz

Erzeuge Beispiele an zulässigen Zeitpunkten \(t\). Jedes Beispiel enthält:

- die vergangenen \(L\) Zeitschritte der räumlichen Umgebung;
- statische Merkmale;
- Zeit bis zum nächsten Ereignis oder zur Zensierung;
- Ereignisindikator;
- Pixel-, Region-, Jahr- und Ereignis-ID;
- optional die Brandursache.

Definiere exakt:

- Beobachtungsfenster \(L\);
- Prognosehorizont \(H\);
- Umgang mit bereits aktiven Feuern;
- Cooldown-/Reset-Regel nach einem Feuer;
- wiederkehrende Ereignisse;
- administrative oder datenbedingte Rechtszensierung;
- NoData- und Ausschlussmasken.

Wichtig: Landmark-Beispiele desselben realen Brandereignisses dürfen nicht über Trainings- und Testdaten verteilt werden.

### 2.3 Zusätzliche Aufgaben

Nur wenn die Labels dies erlauben:

- getrennte Heads für Zündung und Ausbreitung;
- Competing Risks für Blitz, menschliche Ursache und andere Ursachen;
- Vorhersage der Brandgröße als sekundäre, klar getrennte Aufgabe.

---

## Phase 3: Leakage-sichere Splits

Implementiere und versioniere mindestens:

1. **Temporal Split:** Training auf früheren Jahren, Validierung und Test auf späteren Jahren.
2. **Spatial OOD Split:** vollständige Regionen oder Provinzen ausschließlich im Test.
3. **Extreme-Season Split:** mindestens eine außergewöhnliche Feuersaison als schwieriger Test, sofern die Datenmenge dies erlaubt.
4. **Event Grouping:** alle Pixel und Zeitpunkte desselben Brandereignisses bleiben in genau einem Split.

Führe automatische Leakage-Checks aus:

- keine identischen Zeit-Raum-Samples über Splits;
- keine gemeinsame Event-ID über Splits;
- keine überlappenden Patches, die dasselbe Ereignis unmittelbar abbilden;
- Normalisierungsstatistiken ausschließlich aus dem Training;
- keine Zukunftskanäle im Input.

Speichere Split-Manifeste mit Hashes und einer kurzen Begründung.

---

## Phase 4: Datenpipeline und Sanity Checks

### 4.1 Pipeline

- räumliche und zeitliche Ausrichtung aller Kanäle;
- konservatives Resampling je nach kontinuierlichem oder kategorialem Kanal;
- train-only Normalisierung;
- explizite Missingness-Masken statt unbemerkter Nullauffüllung;
- Patch-Sampling mit ausreichendem räumlichem Kontext;
- effizientes Lazy Loading beziehungsweise Chunking;
- kontrolliertes Negativ-Sampling für das Training.

Die reale Klassenprävalenz muss für Validierung, Test und Kalibrierung erhalten bleiben. Falls Negativ-Sampling die Trainingsverteilung verändert, muss dies bei Loss, Kalibrierung oder Prior-Korrektur berücksichtigt werden.

### 4.2 Sanity Checks

Vor dem ersten vollständigen Lauf:

- visualisiere mehrere Inputsequenzen mit allen wichtigen Kanälen;
- überlagere Zündpunkte und Brandperimeter;
- prüfe zufällig ausgewählte Landmark-Labels manuell;
- overfitte ein sehr kleines Dataset;
- kontrolliere, dass die Survival-Kurve monoton ist;
- vergleiche Ereignis- und Zensierungsverteilungen zwischen den Splits;
- prüfe die Pipeline auf NaNs, falsche Einheiten und zeitliche Verschiebungen.

---

## Phase 5: Baselines

Implementiere die Modelle schrittweise in folgender Reihenfolge.

### B0: Klimatologie und Persistenz

- historische Feuerhäufigkeit pro Region und Jahreszeit;
- letzter bekannter Risikozustand beziehungsweise saisonales Mittel;
- falls vorhanden: Canadian Fire Weather Index als externe fachliche Baseline.

### B1: Tabellarische Baselines

- logistische Regression für Ereignis innerhalb von 1, 7 und 30 Tagen;
- XGBoost oder LightGBM mit identischen Inputs;
- optional diskrete Hazard-Regression.

### B2: Räumliche Klassifikationsbaseline

- U-Net oder vergleichbares CNN;
- Input: aktueller Zustand oder kurze gestapelte Historie;
- Output: Next-Step-Hazard beziehungsweise feste Horizonte.

### B3: Räumlich-zeitliche Baseline

- ConvLSTM, Temporal CNN oder kleiner Temporal Transformer;
- Ausgabe diskreter Hazards für \(1,\ldots,H\);
- konsistente kumulative Risiken aus den Hazards ableiten.

### B4: Nicht-räumliche Survival-Baselines

- DeepHit oder diskretes Survival-Modell auf Pixelmerkmalen;
- DCTM ohne räumlichen Kontext;
- dadurch wird der Nutzen des räumlichen Encoders isoliert.

Jede Baseline muss dieselben Splits, Horizonte und Evaluationsroutinen nutzen.

---

## Phase 6: Vorgeschlagenes Hauptmodell

Implementiere erst nach erfolgreicher Baseline-Validierung ein **Dense Spatiotemporal DCTM**.

### 6.1 Architektur

Eine mögliche erste Version:

```text
multimodale Rastersequenz
    -> räumlicher Encoder pro Zeitpunkt
    -> temporale Fusion
    -> Multi-Scale-Decoder
    -> dichter Survival-/DCTM-Head pro Pixel
    -> S(tau), F(tau) und h(tau)
```

Der räumliche Encoder kann zunächst ein CNN oder U-Net-Encoder sein. Die temporale Komponente kann mit einer kleinen Temporal Attention oder einem Transformer beginnen. Vermeide unnötige Architekturkomplexität, bevor der Nutzen einzelner Komponenten nachgewiesen ist.

### 6.2 DCTM-Head

Prüfe eine monotone Parametrisierung der bedingten Transformationsfunktion, beispielsweise mit Bernstein-Basis. Stelle numerisch sicher, dass die resultierende CDF gültig und monoton ist. Implementiere:

- rechtszensierte Likelihood;
- Ereignis-Likelihood;
- mehrere Horizonte aus derselben Verteilung;
- stabile Berechnung im Log-Raum;
- Tests für Monotonie, endliche Gradienten und Grenzfälle.

Falls der kontinuierliche DCTM-Head instabil oder für die zeitliche Auflösung nicht sinnvoll ist, implementiere als belastbare Hauptalternative einen diskreten Multi-Horizon-Hazard-Head. Diese Alternative ist kein Scheitern, sondern eine notwendige methodische Ablation.

### 6.3 Räumliche Abhängigkeit

Untersuche nacheinander:

1. räumlichen Kontext nur über den Encoder;
2. zusätzliche Glättungs- oder Konsistenzregularisierung;
3. optional ereignisbewusste Loss-Gewichtung.

Vermeide eine naive Annahme unabhängiger Pixel bei statistischer Unsicherheitsanalyse. Konfidenzintervalle sollten über Jahre, Regionen oder Brandereignisse gebootstrapped werden, nicht über einzelne Pixel.

---

## Phase 7: Training

- Konfigurationsgetriebene Läufe, beispielsweise per YAML.
- Zunächst kleine Patches, kurze Historie und wenige Epochen für Pipeline-Tests.
- Danach systematische Skalierung.
- Klassenungleichgewicht mit gewichteter Likelihood, Focal-Variante oder kontrolliertem Sampling untersuchen.
- Early Stopping anhand einer vorab festgelegten Validierungsmetrik.
- Seeds für mindestens drei Wiederholungen der wichtigsten Modelle.
- Gemischte Präzision nur nach einem numerischen Vergleich mit FP32 auf kleinem Datensatz.
- Checkpoints müssen Modell, Optimizer, Scheduler, Epoch, Konfiguration und Daten-/Split-Version enthalten.

Erstelle Slurm-Skripte für **sämtliche** ausführbaren Schritte. Dazu gehören mindestens:

- Dateninventarisierung und Daten-Audit;
- Preprocessing;
- Sanity-Check/Smoke-Test;
- Unit- und Integrationstests;
- Split-Erzeugung und Leakage-Prüfung;
- Baseline-Training;
- Hauptmodell-Training;
- Inferenz;
- Evaluation und Figure-Generierung.

Auch kurze Testläufe und kleine Python-Hilfsskripte werden über `sbatch` ausgeführt. Verwende für kleine Jobs angemessene CPU-, RAM- und Zeitlimits und fordere nur dann eine GPU an, wenn sie tatsächlich benötigt wird.

Schätze vor großen Runs Laufzeit, GPU-Speicher und Speicherbedarf. Starte keine große Sweep-Suche ohne bestätigte funktionierende Einzelkonfiguration.

---

## Phase 8: Evaluation

### 8.1 Primäre Metriken

- Negative Log-Likelihood;
- Integrated Brier Score;
- zeitabhängige AUC;
- PR-AUC für feste Horizonte;
- Brier Score für 1, 7, 14 und 30 Tage beziehungsweise datenangemessene Horizonte;
- Kalibrierungsfehler und Reliability Diagrams;
- Event-Recall bei festgelegter False-Alarm-Rate;
- durchschnittliche und mediane Vorwarnzeit.

C-Index allein ist wegen seltener, räumlich abhängiger Ereignisse nicht ausreichend.

### 8.2 Räumlich-ereignisbasierte Evaluation

Bewerte zusätzlich:

- ob mindestens ein Pixel in der Umgebung eines späteren Zündpunkts rechtzeitig alarmiert wurde;
- Distanz zwischen vorhergesagtem Risikomaximum und tatsächlicher Zündung;
- zusammenhängende False-Alarm-Regionen;
- Ergebnisse pro Provinz, Saison, Vegetationstyp und Ursache;
- Generalisierung auf ungesehene Regionen und Jahre.

### 8.3 Kalibrierung

- Kalibrierung ausschließlich auf dem Validierungssplit fitten;
- Testsplit nur einmal für finale Ergebnisse verwenden;
- sowohl globale als auch regionale Kalibrierung berichten;
- Auswirkungen von Negativ-Sampling beziehungsweise Prior Shift prüfen.

### 8.4 Unsicherheit

- Cluster-Bootstrap über Brandereignisse, Jahre oder Regionen;
- 95-%-Konfidenzintervalle für zentrale Metriken;
- keine Pixel als unabhängige Bootstrap-Einheiten behandeln.

---

## Phase 9: Ablationen

Mindestens folgende Vergleiche vorsehen:

1. Klassifikation fester Horizonte vs. diskreter Hazard vs. DCTM;
2. ohne vs. mit räumlichem Kontext;
3. aktueller Zeitpunkt vs. zeitliche Historie;
4. Wetterkanäle vs. Wetter + Vegetation/Biomasse vs. alle Modalitäten;
5. unterschiedliche Historienlängen;
6. unterschiedliche räumliche Patchgrößen;
7. ohne vs. mit Behandlung wiederkehrender Ereignisse;
8. zufälliges Negativ-Sampling vs. korrigiertes Sampling;
9. In-Distribution vs. zeitliches und räumliches OOD;
10. Ignition-Labels vs. Occurrence-Labels, falls beide verfügbar sind.

Die wichtigste Forschungsfrage lautet nicht nur, welches Modell die höchste AUC erreicht, sondern ob die Survival-Formulierung bessere Likelihood, Kalibrierung und horizonübergreifende Konsistenz liefert.

---

## Phase 10: CVPR-taugliche wissenschaftliche Positionierung

Prüfe nach den ersten belastbaren Experimenten kritisch, welche Contribution tatsächlich getragen wird:

1. neue Aufgabe: Dense Spatiotemporal Survival Forecasting;
2. kuratierter multimodaler Benchmark mit zeitlichen, räumlichen und Event-Splits;
3. neue dichte Survival-Architektur für zeitabhängige Kovariaten und wiederkehrende Ereignisse;
4. kalibrierte Risikokarten für mehrere Horizonte;
5. robuste OOD-Evaluation auf ungesehenen Jahren, Regionen und extremen Saisons.

Für eine CVPR-Main-Track-Positionierung sollten möglichst echte visuelle Earth-Observation-Daten wie Sentinel- oder Landsat-Sequenzen enthalten sein. Ausschließlich handkuratierte tabellarische Pixelmerkmale würden die Vision-Story schwächen.

Verallgemeinere die Methode, soweit möglich, über Waldbrände hinaus. Diskutiere beispielsweise Überflutung, Erdrutsche, Dürre oder andere räumlich-zeitliche Ereignisse. Eine zweite Aufgabe oder ein zweiter Datensatz wäre ein starker Nachweis der methodischen Allgemeinheit, ist aber erst nach einem funktionierenden Hauptsetup sinnvoll.

---

## Erwartete Ergebnisse und Dateien

Liefere schrittweise:

1. `data_audit.md` mit Datenqualität, Labeldefinition und Go-/No-Go-Entscheidung;
2. `dataset_manifest.json` oder äquivalentes maschinenlesbares Manifest;
3. versionierte Split-Dateien und `leakage_report.md`;
4. modulare Datenpipeline mit Tests;
5. reproduzierbare Baselines;
6. Dense-Survival-Hauptmodell mit Unit-Tests;
7. Slurm-Skripte für alle rechenintensiven Schritte;
8. Evaluationsskripte, Tabellen und Abbildungen;
9. `experiment_registry.csv` oder JSONL mit allen Läufen;
10. `final_report.md` mit Ergebnissen, Limitationen, offenen Fragen und Empfehlung für das weitere Vorgehen.

## Rückverfolgbarkeit jedes Experiments

Für jeden Lauf speichere mindestens:

```text
run_id
timestamp
git_commit
git_dirty_state
config_path
config_hash
dataset_manifest_hash
split_hash
seed
slurm_job_id
hostname/GPU
environment information
checkpoint path
prediction path
metrics path
status
notes
```

Ergebnisse dürfen nicht manuell aus Logs übertragen werden. Tabellen und Abbildungen müssen aus gespeicherten Vorhersagen und Metrikdateien reproduzierbar erzeugt werden.

---

## Konkrete Reihenfolge der ersten Arbeitsschritte

Arbeite zunächst ausschließlich diese Punkte ab. **Alle Analysen und Skriptausführungen in dieser Liste müssen über `sbatch` erfolgen:**

1. Auf den vom Nutzer angegebenen Datenpfad warten.
2. Repository und vorhandene Slurm-Struktur inspizieren.
3. Ein Audit-Skript und ein `sbatch`-Skript für die Datenanalyse erstellen.
4. Den angegebenen Datenpfad read-only analysieren.
5. Dateninventar und Labeldefinition erstellen.
6. Prüfen, ob echte Ignitions oder nur verbrannte Pixel verfügbar sind.
7. Umfang, zeitliche und räumliche Abdeckung, Variablen, Missingness und Klassenverteilung zusammenfassen.
8. Einen Leakage-sicheren Splitvorschlag ausarbeiten.
9. Eine kleine Landmark-Dataset-Spezifikation mit Beispiel-Tensorformen schreiben.
10. Speicher- und Rechenaufwand abschätzen.
11. Geplante Code- und Dateistruktur zeigen.
12. Offene Fragen und Blocker nennen.

Stoppe danach und berichte dem Nutzer zuerst verständlich, welche Informationen in den Daten tatsächlich enthalten sind. Warte auf seine Bestätigung, bevor du Preprocessing oder Training startest. Der erste Bericht soll insbesondere eine Empfehlung enthalten, ob

- Ignition Survival Forecasting,
- Fire Occurrence/Spread Forecasting oder
- zunächst einfache Multi-Horizon-Klassifikation

mit den tatsächlich vorhandenen Daten am saubersten umsetzbar ist.

# AGENTS.md

## Projektauftrag

Dieses Repository vergleicht praxistaugliche Ansatze aus dem Longstaff-Schwartz-Monte-Carlo-Umfeld mit Reinforcement Learning und verwandten Verfahren. Ziel ist kein Toy-Beispiel, sondern ein reproduzierbarer Forschungs- und Engineering-Vergleich auf Basis der im Repository enthaltenen Marktdaten.

Die fachliche Reihenfolge ist bewusst gestuft:

1. Zuerst einfache amerikanische Optionen untersuchen, um Datenzugriff, Simulation, Exercise-Logik, Bewertungsmetriken und Baselines sauber aufzubauen.
2. Danach eine Swing-Option modellieren, wie sie bei Gasvertraegen vorkommt, mit realistischen Volumen-, Ausuebungs- und Vertragsrestriktionen.

Alle Implementierungen sollen so gebaut werden, dass sie spaeter realistisch bewertet, erweitert und gegen neue Daten laufen gelassen werden konnen.

## Datenbasis

Die primaere Datenquelle ist:

- `ttf_klines_5m_from_1m.sqlite`

Bekannte Struktur:

- Tabelle: `klines`
- Spalten: `symbol`, `interval`, `open_time`, `close_time`, `open`, `high`, `low`, `close`, `volume`
- Zeitstempel: Unix-Millisekunden, als UTC behandeln
- Aktuell vorhandene Reihen: `FRONT`, `FRONT+1`, `FRONT+2`
- Intervall: `5m`

Regeln fur Datenzugriff:

- Die originale SQLite-Datei nicht ueberschreiben, migrieren oder in-place bereinigen.
- Verbindungen standardmaessig read-only oeffnen, sofern kein expliziter Grund fur Schreibzugriff besteht.
- Abgeleitete Daten, Feature-Caches, Backtest-Ergebnisse und Modellartefakte in separaten Verzeichnissen ablegen, z. B. `data/processed/`, `outputs/`, `runs/`, `models/` oder `artifacts/`.
- Bei jeder Datenverarbeitung klar dokumentieren, welche Symbole, Zeitbereiche, Filter und Aggregationen genutzt wurden.
- Keine Datenlecks: Splits muessen strikt zeitlich sein. Validierung und Test duerfen keine Informationen aus spaeteren Zeitpunkten in Training oder Feature-Engineering einspeisen.

## Vergleichsrahmen

Jeder Ansatz soll denselben Vergleichsrahmen nutzen:

- Gleiche Daten-Splits und gleiche Marktannahmen.
- Identische Kostenannahmen fur Transaktionskosten, Slippage, Liquiditaets-/Volumenlimits und Roll-/Ausfuehrungslogik.
- Klare Definition von Zustand, Aktion, Reward/PnL und Risikolimits.
- Out-of-sample-Auswertung als Standard, nicht nur In-sample-Fit.
- Reproduzierbarkeit uber feste Seeds, gespeicherte Konfigurationen und versionierte Experiment-Metadaten.

Bevor neue komplexe Modelle gebaut werden, muessen starke Baselines vorhanden sein:

- Naive und regelbasierte Strategien.
- Einfache statistische oder lineare Modelle.
- Longstaff-Schwartz/Least-Squares-Monte-Carlo-Varianten.
- Mindestens ein einfacher RL-Ansatz erst dann, wenn Environment, Kostenmodell und Baselines stabil sind.

## Fachliche Roadmap

Phase 1: Amerikanische Optionen

- Mit einfachen amerikanischen Puts und Calls beginnen.
- Zunaechst bekannte Testfaelle mit synthetischen Pfaden nutzen, damit LSMC, Exercise-Logik und Regression gegen erwartbare Ergebnisse validiert werden koennen.
- Danach die im Repo enthaltenen Marktzeitreihen als empirische Grundlage fuer Pfade, Returns, Volatilitaetsschaetzung oder Szenarioerzeugung nutzen.
- Bewertungslogik trennen von Modelllogik: Payoff, Exercise-Entscheidung, Discounting, Pfadgenerierung und Regression sollen separat testbar sein.
- Baselines umfassen mindestens europaeischen Payoff, einfache Hold-/Exercise-Regeln und klassische LSMC mit wenigen Basisfunktionen.

Phase 2: Swing-Option fuer Gasvertraege

- Erst starten, wenn Datenladepfad, Backtest-/Bewertungslogik und amerikanische Option stabil getestet sind.
- Vertragsparameter explizit modellieren: Laufzeit, Nominierungsfrequenz, minimale/maximale Tages- oder Periodenmengen, Gesamtvolumenband, Strike/Vertragspreis, Ausuebungsfenster und ggf. Take-or-Pay- oder Make-up-Regeln.
- Zustand muss mindestens Zeit, Restlaufzeit, verbleibendes Volumen, Preis-/Forward-Informationen und relevante historische Features enthalten.
- Aktionen sollen realistische Nominierungen oder Exercise-Entscheidungen abbilden, nicht nur binaeres Ausueben.
- Bewertet wird gegen nachvollziehbare Heuristiken, LSMC-/ADP-Varianten und spaeter RL- oder Offline-RL-Ansaetze.
- Ergebnisse muessen wirtschaftlich interpretierbar sein: Optionswert, Nutzungsprofil, Volumenpfade, Kosten, Risikokennzahlen und Sensitivitaeten gegen Vertragsparameter.

## Methodische Leitplanken

Longstaff-Schwartz/LSMC:

- Regressionsbasis, Zielvariable, Discounting und Exercise-/Stopping-Logik explizit machen.
- Basisfunktionen oder Regressoren nicht beliebig waehlen; Auswahl begruenden und gegen robuste Alternativen testen.
- Pfadgenerierung, Bootstrapping oder Resampling darf die Zeitstruktur der Marktdaten nicht zerstoeren.

Reinforcement Learning:

- Environment zuerst sauber spezifizieren und testen, bevor Agenten optimiert werden.
- Rewards muessen reale Handlungsziele abbilden: PnL nach Kosten, Risiko, Drawdown, Positionsgrenzen und ggf. Carry/Roll-Effekte.
- Offline-RL, Fitted Q Iteration, Approximate Dynamic Programming oder contextual bandits sind gleichwertige Kandidaten, wenn sie besser zur Datenlage passen als Online-RL.
- Keine unrealistischen Annahmen wie perfekte Ausfuehrung zum zukuenftigen Close oder Training auf Testperioden.

Praxisbezug:

- Jede Strategie braucht explizite Annahmen zu Ausfuehrung, Handelbarkeit, Positionsgroesse, Kosten und Risikobegrenzung.
- Ergebnisse muessen neben Rendite auch Risiko und Stabilitaet zeigen: z. B. Sharpe/Sortino, Max Drawdown, Turnover, Hit Rate, Tail Loss, Kostenanteil und Periodenstabilitaet.
- Sensitivitaetsanalysen sind wichtiger als ein einzelnes optimales Ergebnis.

## Erwartete Repo-Struktur

Falls neue Dateien angelegt werden, diese Struktur bevorzugen:

```text
src/                  Wiederverwendbarer Projektcode
tests/                Unit- und Integrationstests
configs/              YAML/TOML/JSON-Konfigurationen fuer Experimente
notebooks/            Explorative Analysen, keine produktive Logik
data/processed/       Abgeleitete Datensaetze und Feature-Caches
outputs/              Reports, Tabellen, Plots
runs/                 Experimentlaeufe und Logs
models/               Trainierte Modelle und Checkpoints
artifacts/            Sonstige generierte Artefakte
```

Produktive Logik gehoert in `src/`, nicht nur in Notebooks. Notebooks duerfen Ergebnisse erklaeren, sollen aber importierbaren Code nutzen.

## Coding-Standards

- Python bevorzugen, sofern kein anderer Stack im Repo etabliert wird.
- Kleine, testbare Module statt grosser Skripte.
- Konfiguration von Experimenten nicht im Code verstecken; CLI-Argumente oder Konfigurationsdateien nutzen.
- Zufallsquellen zentral seeden und Seeds in Ergebnis-Metadaten speichern.
- Datums-/Zeitlogik immer timezone-bewusst behandeln.
- Finanzmathematische Annahmen im Code oder in begleitenden Docs knapp begruenden.
- Keine Secrets, API-Keys oder privaten Zugangsdaten committen.

## Tests und Validierung

Bei neuen Implementierungen mindestens pruefen:

- Datenladepfad und Schemaannahmen.
- Zeitliche Sortierung, fehlende Werte, Duplikate und monotone Timestamps.
- Korrekte Train/Validation/Test-Splits ohne Leakage.
- Backtest-PnL inklusive Kosten und Positionsgrenzen.
- Determinismus bei festen Seeds.
- Smoke-Test fuer jeden neuen Modell- oder Strategiepfad.

Teure Experimente sollten eine schnelle Testkonfiguration haben, die lokal in kurzer Zeit laeuft.

## Analyse- und Reportregeln

- All new project documentation, root README content, analysis reports, report tables, plot titles, figure captions, and prompt files must be written in English unless the user explicitly requests another language for a specific artifact.
- Analysen muessen reproduzierbar sein: genutzte Datenquelle, Symbol, Zeitraum, Split-Logik, Modellkonfiguration, Seed und Ausfuehrungsbefehl dokumentieren.
- Analyseergebnisse nicht nur als lose Artefakte unter `outputs/` ablegen. Zentrale Ergebnisse gehoeren zusaetzlich in das Root-`README.md`, sofern sie fuer den Projektstand relevant sind.
- Das Root-`README.md` soll bei relevanten Analysen Kennzahlen, kurze fachliche Interpretation, Grenzen der Aussagekraft und konkrete naechste Validierungsschritte enthalten.
- Grafiken aus Analysen sollen im Root-`README.md` eingebunden werden, wenn sie das Ergebnis verstaendlich machen. Die Bilddateien duerfen als generierte Artefakte unter `outputs/` bleiben und muessen nicht versioniert werden.
- Tabellen und Plots muessen klar zwischen In-sample, Validation und Test unterscheiden. Keine Ergebnisgrafik darf suggerieren, dass Testdaten fuer Training oder Feature-Erzeugung genutzt wurden.
- Path-quality evaluation must not rely on simple point-forecast or marginal diagnostics such as RMSE, correlation, R2, MSE skill, or VaR hit rates as the main score. These may be reported only as auxiliary model diagnostics. Primary path-quality reports should use distributional scores and two-sample criteria such as Energy Score and multiband maximum mean discrepancy (MMD), computed on path-level features across multiple horizons.
- Schlechte oder gemischte Ergebnisse gehoeren ausdruecklich in die Dokumentation. Modelle nicht schoenreden; klar benennen, welche Diagnosen gegen eine produktive Nutzung sprechen.

## Git- und Artefaktregeln

- Die Datei `ttf_klines_5m_from_1m.sqlite` ist eine kuratierte Eingangsdatenbank und darf versioniert bleiben.
- SQLite-Nebendateien wie `*.sqlite-wal`, `*.sqlite-shm` und Backups nicht committen.
- Generierte Modelle, Runs, Plots, Cache-Dateien und grosse Zwischenprodukte nicht committen.
- Wenn ein Ergebnis fuer eine Auswertung wichtig ist, lieber die Konfiguration und eine kompakte Ergebniszusammenfassung committen als rohe Artefaktberge.

## Arbeitsweise fuer Agenten

- Vor Aenderungen kurz die vorhandene Struktur und Datenannahmen pruefen.
- Bestehende Nutzerdateien nicht ueberschreiben oder entfernen, wenn das nicht explizit verlangt wurde.
- Bei Modellvergleichen erst die gemeinsame Bewertungslogik staerken, dann einzelne Modelle optimieren.
- Jede neue Methode muss gegen Baselines laufen koennen.
- Ergebnisse nicht ueberinterpretieren; klar zwischen In-sample, Validation und Test unterscheiden.
- Wenn eine Annahme unklar ist, konservativ und transparent entscheiden.

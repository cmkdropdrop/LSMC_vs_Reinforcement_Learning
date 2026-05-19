# LSMC vs Reinforcement Learning

Dieses Repository baut eine reproduzierbare Vergleichsbasis fuer LSMC-, ADP-
und spaetere Reinforcement-Learning-Verfahren auf. Der aktuelle Stand ist noch
keine Options- oder RL-Engine, sondern ein sauberer Daten-, Volatilitaets- und
Pfadgenerierungsbaustein auf Basis der enthaltenen Marktdaten.

## Datenbasis

Primaere Quelle:

```text
ttf_klines_5m_from_1m.sqlite
```

Die SQLite-Datei wird read-only gelesen. Aktuell werden die `klines`-Daten fuer
`FRONT`, `FRONT+1` und `FRONT+2` mit 5-Minuten-Intervall unterstuetzt. Die
Zeitstempel werden als UTC interpretiert.

Die aktuelle Analyse arbeitet mit TTF-Gas-Daten, also einem zentralen
europaeischen Gas-Benchmark. Der untersuchte Zeitraum enthaelt die Marktphase
rund um den Iran-Konflikt 2026, in der TTF- und europaeische Gaspreise laut
Marktberichten stark reagierten. Diese Daten werden deshalb bewusst als
Haertetest fuer spaetere LSMC- und Reinforcement-Learning-Ansaetze verstanden:
Die Modelle sollen nicht nur auf glatten Toy-Daten funktionieren, sondern auch
unter sprunghaften, geopolitisch getriebenen Volatilitaetsregimen.

## Implementierter Stand

- Read-only-Datenloader mit Schema-, OHLC-, Duplikat- und Gap-Pruefung
- Log-Return-Berechnung ohne Lookahead
- GJR-GARCH(1,1) fuer Intraday-Volatilitaet
- HAR-RV fuer Tages-RV aus daily/weekly/monthly Features
- Monte-Carlo-Pfad-Schnittstelle mit `path`, `step`, `time`, `price`, `return`,
  `variance`, `volatility`, `model`
- Analyse-CLI fuer rollierende GARCH-Refits und Tages-RV-Diagnostik

## Ausfuehrung

```powershell
python -m pip install -e .
python -m lsmc_rl.simulation.paths --config configs/mc_paths_front.yaml
```

HAR-RV-Pfade:

```powershell
python -m lsmc_rl.simulation.paths --config configs/mc_paths_front.yaml --model-type har_rv --output-path outputs/mc_paths_front_har_rv.csv
```

Aktuelle GARCH-Analyse mit regelmaessigem Refit:

```powershell
python -m lsmc_rl.analysis.garch_refit_report --output-dir outputs/garch_refit_report_front --report-path docs/garch_refit_analysis.md
```

## Aktuelle Analyse: Rolling-Refit GJR-GARCH

Die detaillierte Analyse steht in
[docs/garch_refit_analysis.md](docs/garch_refit_analysis.md). Sie bewertet
GJR-GARCH nicht mehr primar auf einzelnen, sehr verrauschten 5-Minuten-Quadraten,
sondern aggregiert sequenzielle Intraday-One-Step-Varianzprognosen zu Tages-RV.
Zusaetzlich wird das Modell im Testzeitraum regelmaessig neu gefittet.

Diese Auswertung ist als TTF-Gas-Stresstest einzuordnen: Der Iran-Konflikt 2026
fuehrte zu stark erhoehten Energiepreis- und Lieferkettenrisiken, sodass die
Pfadgeneratoren explizit unter schwierigen Marktbedingungen vorbereitet werden.

Setup:

| Groesse | Wert |
| --- | --- |
| Symbol | `FRONT` |
| Train | `2025-04-01T07:00:00+00:00` bis `2025-11-14T08:40:00+00:00` |
| Test | `2025-11-14T09:15:00+00:00` bis `2026-03-26T15:35:00+00:00` |
| Testtage | `84` |
| Refit-Intervall | `288` Test-Returns |
| Refits | `9` |

Kompakte Ergebnisse:

| Metrik | Rolling-Refit GJR | Persistence |
| --- | ---: | ---: |
| Daily-RV RMSE | `0.002137` | `0.013529` |
| Daily-RV Korrelation | `0.987597` | `0.499442` |
| Daily-RV R2 vs. Mittelwert | `0.975002` | `-0.001539` |
| MSE-Skill vs. Persistence | `0.975039` | - |
| 5%-VaR-Hitrate Tagesreturns | `0.035714` | - |
| 1%-VaR-Hitrate Tagesreturns | `0.011905` | - |

![Daily RV forecast](outputs/garch_refit_report_front/garch_daily_rv_forecast.png)

![Daily return bands](outputs/garch_refit_report_front/garch_daily_return_bands.png)

Kurzfazit: Die neue GARCH-Diagnostik ist deutlich fairer fuer den spaeteren
Pfadgenerator. Auf Tages-RV erreicht Rolling-Refit-GJR eine hohe Korrelation und
ein hohes R2 gegenueber realisierter Volatilitaet. Die 1%-VaR-Hitrate liegt nahe
am nominalen Niveau, die 5%-Hitrate ist eher konservativ. Offen bleiben
Normalinnovationen, Intraday-Saisonalitaet und eine systematische Suche ueber
Refit-Frequenzen.

## Tests

```powershell
pytest
```

Die Tests decken Datenladepfad, Return-Berechnung, Volatilitaetsmodelle,
Pfadsimulation und die Analyse-Aggregationen ab.

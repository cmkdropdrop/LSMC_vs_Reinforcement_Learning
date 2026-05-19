# GARCH Refit Analyse

Diese Analyse ersetzt die fruehere stark kleinteilige 5-Minuten-Diagnostik nicht, bewertet GJR-GARCH aber auf einer fuer Pfadgenerierung faireren Ebene: sequenzielle Intraday-One-Step-Varianzprognosen werden zu Tages-RV aggregiert. Zusaetzlich wird das Modell im Testzeitraum regelmaessig neu gefittet.

## Setup

| Groesse | Wert |
| --- | --- |
| Symbol | `FRONT` |
| Intervall | `5m` |
| Train | `2025-04-01T07:00:00+00:00` bis `2025-11-14T08:40:00+00:00` |
| Test | `2025-11-14T09:15:00+00:00` bis `2026-03-26T15:35:00+00:00` |
| Train-Returns | 5551 |
| Test-Returns | 2379 |
| Test-Tage | 84 |
| Refit-Intervall | 288 Test-Returns |
| Refits | 9 |
| Fallback-Refits | 0 |

![Price split](../outputs/garch_refit_report_front/garch_refit_price_split.png)

## Ergebnisuebersicht

| Metrik | Statisches GJR | Rolling-Refit GJR | Persistence |
| --- | --- | --- | --- |
| Daily-RV RMSE | 0.001909 | 0.002137 | 0.013529 |
| Daily-RV MAE | 6.901474e-04 | 8.798250e-04 | 0.005194 |
| Daily-RV Korrelation | 0.990185 | 0.987597 | 0.499442 |
| Daily-RV R2 vs. Mittelwert | 0.980057 | 0.975002 | -0.001539 |
| QLIKE | -4.709384 | -5.188923 | -1.749987 |
| MSE-Skill vs. Persistence | 0.980086 | 0.975039 | - |

Die aggregierte Tages-RV-Diagnostik ist deutlich stabiler als die reine 5-Minuten-Auswertung. Das Rolling-Refit-GJR erreicht eine Daily-RV-Korrelation von `0.987597` und ein R2 von `0.975002`. Gegenueber der Persistence-Baseline liegt der MSE-Skill bei `0.975039`.

![Daily RV forecast](../outputs/garch_refit_report_front/garch_daily_rv_forecast.png)

## Rolling-Refit Und Kalibrierung

Der Refit erfolgt auf einem expandierenden historischen Fenster. Dadurch kann das Modell spaeter im Testzeitraum neue Volatilitaetsregime aufnehmen, ohne zukuenftige Daten zu verwenden. In dieser Auswertung gab es keine notwendigen Fallbacks auf alte Parameter.

| Kalibrierungsmetrik | Wert |
| --- | --- |
| 5%-VaR-Hitrate auf Tagesreturns | 0.035714 |
| 1%-VaR-Hitrate auf Tagesreturns | 0.011905 |
| Mittelwert standardisierter Tagesresiduen | 0.173075 |
| Std. standardisierter Tagesresiduen | 1.169122 |
| Excess Kurtosis standardisierter Tagesresiduen | 5.697768 |

![Daily return bands](../outputs/garch_refit_report_front/garch_daily_return_bands.png)

![Standardized residuals](../outputs/garch_refit_report_front/garch_standardized_residuals.png)

Die 1%-VaR-Hitrate liegt nahe am nominalen Niveau. Die 5%-Hitrate ist konservativ niedrig, und die standardisierten Residuen bleiben etwas zu breit und leptokurtisch. Das ist deutlich besser interpretierbar als die alte 5-Minuten-Tail-Diagnostik, aber noch kein Freifahrtschein fuer Optionsbewertung.

## Parameterpfad

![Refit parameters](../outputs/garch_refit_report_front/garch_refit_parameters.png)

Der Parameterpfad zeigt, ob der Refit stabile Parameter liefert oder einzelne Testabschnitte stark andere Dynamiken erzwingen. Fuer spaetere Experimente sollte dieser Pfad zusammen mit Optionswert-Sensitivitaeten beobachtet werden.

## Fazit

Die neue GARCH-Analyse liefert deutlich bessere und fachlich fairere Ergebnisse, weil sie Volatilitaet auf Tagesebene bewertet und das Modell im Testzeitraum regelmaessig neu kalibriert. Fuer die naechste Stufe ist Rolling-Refit-GJR daher ein besserer Kandidat als die alte statische 5-Minuten-Diagnose.

Trotzdem bleiben Annahmen offen: Normalinnovationen, keine explizite Intraday-Saisonalitaet und keine Rolling-Origin-Grid-Suche ueber Refit-Frequenzen. Diese Punkte sollten vor einer produktiven LSMC-/RL-Nutzung weiter getestet werden.

## Reproduktion

```powershell
python -m lsmc_rl.analysis.garch_refit_report --output-dir outputs/garch_refit_report_front --report-path docs/garch_refit_analysis.md
```

Die numerischen Rohdaten stehen in `outputs/garch_refit_report_front/garch_refit_metrics.json`.

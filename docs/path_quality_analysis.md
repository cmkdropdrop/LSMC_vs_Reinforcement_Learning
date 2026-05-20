# Path Quality Diagnostics

This report evaluates the GJR-GARCH and HAR-RV path generators with distributional scores rather than point-forecast metrics. Lower values are better for QLIKE, Gaussian NLL, Energy Score, and multiband MMD. The path-quality diagnostic is intentionally table-only; plots are not needed for the model decision at this stage.

## Scores

| model | QLIKE | Gaussian NLL | Energy Score | multiband MMD^2 |
| --- | --- | --- | --- | --- |
| gjr_garch | -4.668053 | -1.388606 | 3.920006 | 0.356907 |
| har_rv | 231.368793 | 40.844925 | 135.499740 | 0.430470 |

Ranking by score:

| score | best model |
| --- | --- |
| qlike | gjr_garch |
| normal_nll | gjr_garch |
| energy_score | gjr_garch |
| aggregate_mmd2 | gjr_garch |

QLIKE and Gaussian NLL are computed on aligned out-of-sample daily return and variance forecasts. Energy Score and multiband MMD are computed on robustly standardized daily path-feature vectors built from returns, realized volatility, downside/upside variation, drawdown, intraday range, and multi-horizon return and variance bands.

## MMD By Feature Band

| model | core | extremes | multi_horizon | aggregate |
| --- | --- | --- | --- | --- |
| gjr_garch | 0.482634 | 0.361964 | 0.226124 | 0.356907 |
| har_rv | 0.453582 | 0.473121 | 0.364707 | 0.430470 |

## Interpretation

- QLIKE winner: `gjr_garch`.
- Gaussian NLL winner: `gjr_garch`.
- Energy Score winner: `gjr_garch`.
- Multiband MMD winner: `gjr_garch`.
- Practical assessment: `gjr_garch` currently produces the more realistic paths.

A model is more credible for option work only if it performs well on both forecast likelihood scores and distributional path scores. QLIKE/NLL evaluate the one-day return and variance forecasts; Energy Score and multiband MMD test whether simulated daily path features look like observed daily path features across several bands. Disagreement between these scores is a warning that a model may forecast variance acceptably while still generating unrealistic path shapes.

## Reproduce

```powershell
$env:PYTHONPATH='src'
python -m lsmc_rl.analysis.path_quality_report --output-dir outputs/path_quality_report_front --report-path docs/path_quality_analysis.md
```

The full numeric output is stored in `outputs/path_quality_report_front/metrics.json`.

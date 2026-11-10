# Gold LightGBM experiment and handoff

Target: GLD ETF adjusted closing price at the next recorded observation, in the dataset's price units.
No gram, IDR, or physical-gold interpretation is assigned.

## Procedure

Approved Gold manifest replay; 1367 training, 342 testing, one holdout purge.
TimeSeriesSplit(5, gap=1), training only. RandomizedSearchCV uses 12 candidates,
negative RMSE, random_state=42, n_jobs=1, refit=True, error_score='raise'.
The CPU estimator uses LGBMRegressor with parameters:
objective="regression", random_state=42, n_jobs=1, deterministic=True,
force_row_wise=True, subsample_freq=1, importance_type="gain", verbose=-1.
No learned preprocessing, scaling, or feature changes were introduced.

Fit accounting: 60 search CV fits + 5 fixed-parameter descriptive CV fits + 1 final training refit.
The additional 5 fits report all metrics before testing; they are not another search.
Final test and persistence metrics were computed only after training-only selection.
Reload/audit prediction calls do not change or retune the fitted model.

## Best parameters

```json
{
  "model__subsample": 1.0,
  "model__reg_lambda": 5.0,
  "model__num_leaves": 8,
  "model__n_estimators": 200,
  "model__min_child_samples": 10,
  "model__max_depth": 3,
  "model__learning_rate": 0.05,
  "model__colsample_bytree": 1.0
}
```

## Actual metrics

| Model / subset | MAE | MAPE (%) | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| LightGBM training | 0.779815076 | 0.607158678 | 1.159619746 | 1.076856418 | 0.996815546 |
| LightGBM CV mean | 4.181892029 | 3.377142843 | 75.841660087 | 5.509848611 | 0.483193606 |
| LightGBM testing | 0.647531209 | 0.536290075 | 0.671981204 | 0.819744597 | 0.968823759 |
| Persistence testing | 0.567221865 | 0.468659243 | 0.550177837 | 0.741739736 | 0.974474767 |

MAE/RMSE use dataset price units; MSE uses squared price units; R2 is unitless.
CV values are unweighted fold means. Mean fold RMSE is not sqrt(mean fold MSE).
MAPE is sklearn MAPE multiplied by 100; zero targets are rejected.

CV RMSE by fold: 18.90208228755004, 2.851969883248238, 3.1713760750764997, 1.6900774268868068, 0.9337373817327482
CV RMSE standard deviation (ddof=0): 6.744125471276942

## Quality assessment

LightGBM does NOT improve upon persistence on this test period.
Persistence predicts the next adjusted close using observed Close(t), on identical targets.
LightGBM test RMSE is 10.516473% higher than persistence; all five reported test metrics favor persistence.
No retuning was performed after observing test/baseline performance.

Test minus train RMSE: -0.25711182143422584
Test RMSE is lower than train RMSE, so this simple gap does not show the usual
higher-test-error pattern. It is not proof against overfitting or distribution shift.
Temporal CV is unstable across periods: the first fold RMSE is much larger (18.902082) and its R2 is negative (-0.588347).
Read-only diagnostics show distribution shift across temporal regimes is a plausible contributor, not a proven sole cause.
High test R2 alone is insufficient evidence of added forecasting value.
Comparative evaluation against XGBoost is delegated to Tahap 3 (Azzam).

## Outputs

- models/gold/lightgbm_pipeline.joblib: fitted Gold LightGBM pipeline only.
- results/gold/lightgbm_metrics.json / .csv: train, CV, test, and persistence metrics.
- results/gold/lightgbm_cv_results.csv: all 12 candidates and per-fold scores.
- results/gold/lightgbm_metadata.json: schema, timing, folds, versions, hashes, and quality metrics.
- results/gold/lightgbm_test_predictions.csv: source/target rows/dates, predictions, residuals, baseline.
- results/gold/lightgbm_execution_log.json: execution configuration, events, warnings, artifact hashes.
- figures/gold/lightgbm_actual_vs_predicted.png
- figures/gold/lightgbm_residuals.png
- figures/gold/lightgbm_feature_importance.png
- results/gold/lightgbm_training_source.py.txt: byte-identical copy of trainer source at time of execution.

Feature importance is normalized gain (High: 39.47%, Close: 45.21%, Low: 14.44%).
Model SHA-256: cbf223e3d5733817f7dd849a52a95ca7a999a4c2d32bd1114d02a124de8cf43b

## Commands and interfaces

From Week-11, the training command is:

    & './.venv/Scripts/python.exe' -B -m src.train_lightgbm_gold

The trainer refuses existing Gold output artifacts; inspect rather than overwrite.
Do not rerun training to optimize against this final test set.

Relevant tests:

    & './.venv/Scripts/python.exe' -B -m pytest tests -v -p no:cacheprovider

Load without training, from Week-11:

```python
import joblib
from src.gold_data_pipeline import replay_gold_manifest
split = replay_gold_manifest()
model = joblib.load("models/gold/lightgbm_pipeline.joblib")
predictions = model.predict(split.X_test)
```

## Unresolved limitations

Historical data end in 2018. Next recorded observation is not a verified consecutive exchange session.
Currency/unit details, license, class-wide adoption, and point-in-time adjustment provenance
remain external documentation items. Five CV folds span unequal calendar durations.
Large CV variance exists between temporal folds (fold 1 vs fold 5).
One random seed (42) and bounded search of 12 candidates were evaluated.
Test set was never used for model tuning or feature selection.
Full model comparison and Streamlit UI implementation will be performed by Azzam in Tahap 3.
Warnings captured during search/CV/final prediction: []

# Gold XGBoost experiment and handoff

Target: GLD ETF adjusted closing price at the next recorded observation, in the dataset's price units.
No gram, IDR, or physical-gold interpretation is assigned.

## Procedure

Approved Gold manifest replay; 1367 training, 342 testing, one holdout purge.
TimeSeriesSplit(5, gap=1), training only. RandomizedSearchCV uses 12 candidates,
negative RMSE, random_state=42, n_jobs=1, refit=True, error_score='raise'.
The CPU estimator uses reg:squarederror and tree_method='hist'.
No learned preprocessing, scaling, or feature changes were introduced.

Fit accounting: 60 search CV fits + 5 fixed-parameter descriptive CV fits + 1 final training refit.
The additional 5 fits report all metrics before testing; they are not another search.
Final test and persistence metrics were computed only after training-only selection.
Reload/audit prediction calls do not change or retune the fitted model.

## Best parameters

{
  "model__subsample": 1.0,
  "model__reg_lambda": 1,
  "model__n_estimators": 100,
  "model__max_depth": 2,
  "model__learning_rate": 0.1,
  "model__colsample_bytree": 0.8
}

## Actual metrics

| Model / subset | MAE | MAPE (%) | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| XGBoost training | 0.847445769 | 0.658597558 | 1.395211869 | 1.181190869 | 0.996168582 |
| XGBoost CV mean | 4.099111051 | 3.313673037 | 71.938840521 | 5.406863028 | 0.502982133 |
| XGBoost testing | 0.636764737 | 0.526921073 | 0.646805375 | 0.804242112 | 0.969991780 |
| Persistence testing | 0.567221865 | 0.468659243 | 0.550177837 | 0.741739736 | 0.974474767 |

MAE/RMSE use dataset price units; MSE uses squared price units; R2 is unitless.
CV values are unweighted fold means. Mean fold RMSE is not sqrt(mean fold MSE).
MAPE is sklearn MAPE multiplied by 100; zero targets are rejected.

CV RMSE by fold: 18.384214403, 2.869771104, 3.099724435, 1.713239257, 0.967365941
CV RMSE standard deviation (ddof=0): 6.534881231953991

## Quality assessment

XGBoost does NOT improve upon persistence on this test period.
Persistence predicts the next adjusted close using observed Close(t), on identical targets.
XGBoost test RMSE is 8.426456% higher; all five reported test metrics favor persistence.
No retuning was performed after observing test/baseline performance.

Test minus train RMSE: -0.37694875653689885
Test RMSE is lower than train RMSE, so this simple gap does not show the usual
higher-test-error pattern. It is not proof against overfitting or distribution shift.
Temporal CV is unstable: the first fold RMSE is much larger and its R2 is negative.
Read-only diagnostics show 140/227 first-fold validation targets outside its
training target range; distribution change is a plausible contributor, not a proven sole cause.
High test R2 alone is insufficient evidence of added forecasting value.
No comparison with LightGBM has been performed.

## Outputs

- models/gold/xgboost_pipeline.joblib: fitted Gold pipeline only.
- results/gold/xgboost_metrics.json / .csv: train, CV, test, and persistence metrics.
- results/gold/xgboost_cv_results.csv: all 12 candidates and per-fold scores.
- results/gold/xgboost_metadata.json: schema, timing, folds, versions, hashes, and quality metrics.
- results/gold/xgboost_test_predictions.csv: source/target rows/dates, predictions, residuals, baseline.
- results/gold/xgboost_execution_log.json: execution configuration, events, warnings, artifact hashes.
- figures/gold/xgboost_actual_vs_predicted.png
- figures/gold/xgboost_residuals.png
- figures/gold/xgboost_feature_importance.png

Feature importance is normalized gain, predictive contribution rather than causality.
Model SHA-256: fe5f7a2cc001bc32ae336311c85730e404154ce13ecd1af1a66f3f3f0f3d8425

## Commands and interfaces

From Week-11, the training command is:

    & './.venv/Scripts/python.exe' -B -m src.train_xgboost_gold

The trainer refuses existing Gold output artifacts; inspect rather than overwrite.
Do not rerun training to optimize against this final test set.

Relevant tests:

    & './.venv/Scripts/python.exe' -B -m pytest tests -v -p no:cacheprovider

The original experiment stage recorded 62 tests including legacy regression checks.
Use the active Gold-only suite above for the integrated project.
Small synthetic test fixtures are not research experiment results.

Load without training, from Week-11:

    import joblib
    from src.gold_data_pipeline import replay_gold_manifest
    split = replay_gold_manifest()
    model = joblib.load("models/gold/xgboost_pipeline.joblib")
    predictions = model.predict(split.X_test)

Jamal should reuse the Gold manifest, 11-feature schema, training-only temporal CV,
and src.evaluation.evaluate_regression. Shared metrics use dataset price units.
Preserve the same persistence comparison and explicitly distinguish train/CV/test metrics.

## Unresolved limitations

Historical data end in 2018. Next recorded observation is not a verified exchange session.
Currency/unit details, license, class-wide adoption, and point-in-time adjustment provenance
remain external documentation items. Five CV folds span unequal calendar durations.
No claim of current-market performance or superiority over LightGBM is made.
This experiment evidence predates the local integration commit; no push/PR or Google Docs edits were performed.
Warnings captured during search/CV/final prediction: []

Integration note: README documents safe --output-root reproduction. The original
trainer source is preserved as results/gold/xgboost_training_source.py.txt and
matches trainer_source_sha256 in the immutable experiment metadata.

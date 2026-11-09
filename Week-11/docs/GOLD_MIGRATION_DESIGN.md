# Gold migration design — Week 11

Scope: approved data design. Completed XGBoost results are documented in GOLD_XGBOOST_RESULTS.md.

## Dataset provenance and interpretation

Official source:
https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset

The prior audit downloaded the public official Kaggle archive. Only FINAL_USO.csv
was saved to Week-11/data/raw/FINAL_USO.csv. It is read-only input.
Original schema: 1718 rows, 81 columns; 2011-12-15 through 2018-12-31.

SHA-256:
f1a366b4edfbdf074136727f3653058be6d91456e0128a462220797400ae6087

This is GLD ETF price data, NOT gold price per gram. Currency/unit and licensing
details still need source confirmation; never label these values IDR/gram.
The prior prototype remains in Git history; the active Week-11 project is Gold-only.

## Target, cutoff, and identifiers

At the end of observation t, predict Adj Close at the next recorded observation
t+1. This is not a claim of a verified next exchange session or fixed calendar day.

y(t) = Adj Close(t+1).

The original zero-based CSV source row positions remain DataFrame indices.
Source date, target date, target row, and gap days are audit metadata, not features.

Close and Adj Close are exactly equal in this CSV. Same-day Close(t+1) directly
leaks the target; historical Close(t) is permitted for this next-observation target.
Negative shifts occur only in labels/audit metadata, never in predictors.
Date is not converted into an arbitrary numeric ordinal feature.

## Exact feature engineering

| Feature | Definition |
|---|---|
| Open, High, Low, Close, Volume | Observed at t; available by the stated end-of-observation cutoff |
| close_lag_1 / close_lag_5 | Close(t-1) / Close(t-5), measured in recorded-observation positions |
| close_mean_prev_5 | Mean Close(t-5) through Close(t-1); min_periods=5 |
| close_std_prev_5 | Sample std of those preceding five prices; ddof=1 |
| close_return_1 / close_return_5 | Close(t)/Close(t-k)-1 for k=1 or 5; no fill |

All 11 features are numeric. Rolling windows exclude t and never center.
No future values, backfill, global fit, imputation, scaling, or external asset
features are used. Unexpected missing/nonfinite raw numeric values are rejected.

The loader rejects unsorted input instead of silently changing original source
identities. All retained samples are ascending. Historical lags may span calendar
gaps; five recorded observations do not necessarily mean five calendar days.

## Date-gap policy and exclusions

Exclude a prediction pair if its source-to-target gap exceeds 6 calendar days.
No missing observations are fabricated. Excluding a pair does not remove its
target observation from use as a later source or previously observed history.

| Source row | Source date | Target row/date | Gap | Reason |
|---:|---|---|---:|---|
| 0 | 2011-12-15 | 1 / 2011-12-16 | 1 | leading_history_unavailable |
| 1 | 2011-12-16 | 2 / 2011-12-19 | 3 | leading_history_unavailable |
| 2 | 2011-12-19 | 3 / 2011-12-20 | 1 | leading_history_unavailable |
| 3 | 2011-12-20 | 4 / 2011-12-21 | 1 | leading_history_unavailable |
| 4 | 2011-12-21 | 5 / 2011-12-22 | 1 | leading_history_unavailable |
| 326 | 2013-04-15 | 327 / 2013-05-01 | 16 | source_target_gap_exceeds_6_calendar_days |
| 478 | 2013-12-13 | 479 / 2014-01-02 | 20 | source_target_gap_exceeds_6_calendar_days |
| 1717 | 2018-12-31 | None / None | None | no_next_recorded_observation |

Eight excluded rows leave 1710 eligible pairs. These comprise five leading
history rows, two long-gap pairs, and the final row with no next target.

## Chronological holdout and purge

Reserve the latest ceil(20% of eligible samples) for untouched testing.
Purge one eligible sample immediately before test; never shuffle.
Train target timestamps must be strictly earlier than the first test source cutoff.
Purge is recorded separately from raw-data exclusions.

- training: 1367 samples; source 2011-12-22 to 2017-08-10; targets 2011-12-23 to 2017-08-11.
- testing: 342 samples; source 2017-08-14 to 2018-12-28; targets 2017-08-16 to 2018-12-31.
- Purged source indices: [1374].

## Training-only cross-validation

TimeSeriesSplit(n_splits=5, gap=1) applies ONLY to X_train.
Each fold verifies disjoint indices, valid sizes, source ordering, and strict
training-target timestamps before the first validation source cutoff.
No model is fitted. Equal validation row counts do not imply equal calendar durations.

| Fold | Train | Validation | Train target end | Validation source start/end |
|---:|---:|---:|---|---|
| 1 | 231 | 227 | 2012-12-03 | 2012-12-04 / 2013-11-22 |
| 2 | 458 | 227 | 2013-11-22 | 2013-11-25 / 2014-11-14 |
| 3 | 685 | 227 | 2014-11-14 | 2014-11-17 / 2015-10-14 |
| 4 | 912 | 227 | 2015-10-14 | 2015-10-15 / 2016-09-12 |
| 5 | 1139 | 227 | 2016-09-12 | 2016-09-13 / 2017-08-10 |

Any learned preprocessing added later must be inside the estimator pipeline and
fitted on each CV training fold only.

## Reproducibility and public interfaces

Gold manifest: results/gold/split_manifest.json.
This Gold manifest is the sole active split manifest.

The split fingerprint hashes canonical sorted JSON excluding only split_sha256.
It includes dataset hash, policies, exact indices, exclusions, date ranges, and CV
boundaries. No timestamp or machine-specific absolute path is included.
The manifest byte hash is a separate audit value.

Replay rebuilds deterministic features and selects stored indices without
resplitting. It rejects fingerprint, dataset, configuration, or content mismatch.
Repeated CLI runs accept an identical manifest, but refuse to overwrite a
different one.

Public functions in src/gold_data_pipeline.py:

    load_gold_data(path=DATASET_PATH, *, expected_sha256=EXPECTED_SHA256,
                   expected_shape=(1718,81)) -> GoldData
    make_gold_samples(dataset: GoldData) -> GoldSamples
    chronological_split(samples: GoldSamples) -> GoldSplit
    make_time_series_cv() -> TimeSeriesSplit
    validate_cv_folds(split: GoldSplit) -> list[dict]
    make_gold_manifest(split: GoldSplit) -> dict
    replay_gold_manifest(path=MANIFEST_PATH, dataset=None) -> GoldSplit

Hash/shape overrides are for explicit small test fixtures; production uses the
approved defaults. GoldSplit exposes X_train, X_test, y_train, y_test;
GoldSamples.timeline retains source/target dates and original row identities.

Run from Week-11:

    & './.venv/Scripts/python.exe' -B -m src.gold_data_pipeline
    & './.venv/Scripts/python.exe' -B -m pytest tests/test_gold_data_pipeline.py tests/test_xgboost_gold.py tests/test_evaluation.py -v -p no:cacheprovider

## Handoff for Falah and Jamal

    from src.gold_data_pipeline import replay_gold_manifest
    from src.gold_data_pipeline import make_time_series_cv, validate_cv_folds
    split = replay_gold_manifest()
    fold_audit = validate_cv_folds(split)
    cv = make_time_series_cv()
    # Later search.fit receives split.X_train and split.y_train only.

Both estimators must use the same 11-feature schema, manifest, and training-only
CV. No imputation/scaling is currently required.
All current unit-test fixtures are for behavior verification only.

## Limitations

- Next recorded observation is not guaranteed to be the next exchange session.
- Long-gap pair exclusion does not repair an incomplete exchange calendar.
- Lags/rolling windows may span older observed history across calendar gaps.
- Adjusted values have not been verified as point-in-time, revision-free data.
- Data end in 2018; no current-market accuracy claim is made.
- Unit/currency, license details, and class-wide adoption remain external items.
- External OHLC anomalies from audit are outside the selected GLD-only features.

Split fingerprint: 07224b5958086dc8c76fca249c26e960e7748cb21e70c94650ce8d9459f259f2

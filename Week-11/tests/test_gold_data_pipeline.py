"""Synthetic fixtures validate time alignment, never model performance."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.gold_data_pipeline import (
    GoldData, FEATURES, DATASET_PATH, WEEK_DIR, load_gold_data, make_gold_samples,
    chronological_split, make_gold_manifest, replay_gold_manifest,
    make_time_series_cv, validate_cv_folds,
)

@pytest.fixture
def csv_factory(tmp_path):
    def make(frame=None):
        if frame is None:
            close=np.arange(1,81,dtype=float)+100
            frame=pd.DataFrame({
                "Date":pd.bdate_range("2020-01-01",periods=80).strftime("%Y-%m-%d"),
                "Open":close-0.5,"High":close+1,"Low":close-1,
                "Close":close,"Adj Close":close+0.25,"Volume":1000,
            })
        p=tmp_path/"fixture.csv"
        frame.to_csv(p,index=False,encoding="utf-8")
        return p,hashlib.sha256(p.read_bytes()).hexdigest()
    return make

def fixture_data(factory,frame=None):
    p,h=factory(frame)
    return load_gold_data(p,expected_sha256=h,expected_shape=None)

def fixture_frame(factory):
    return fixture_data(factory).frame.assign(
        Date=lambda d:d.Date.dt.strftime("%Y-%m-%d")).copy()

def test_hash_validation(csv_factory):
    p,_=csv_factory()
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        load_gold_data(p,expected_sha256="0"*64,expected_shape=None)

def test_required_columns(csv_factory):
    with pytest.raises(ValueError,match="Missing required"):
        fixture_data(csv_factory,fixture_frame(csv_factory).drop(columns="Volume"))

@pytest.mark.parametrize("kind",["invalid","duplicate","unsorted","wrong_format"])
def test_invalid_dates(csv_factory,kind):
    d=fixture_frame(csv_factory)
    if kind=="invalid": d.loc[5,"Date"]="2020-02-30"
    elif kind=="duplicate": d.loc[5,"Date"]=d.loc[4,"Date"]
    elif kind=="unsorted": d.loc[[4,5],"Date"]=d.loc[[5,4],"Date"].to_numpy()
    else: d.loc[5,"Date"]="01/01/2020"
    with pytest.raises(ValueError): fixture_data(csv_factory,d)

@pytest.mark.parametrize("column,value",[("Adj Close",np.nan),("High",np.inf),
                                         ("Close",0),("Adj Close","bad")])
def test_numeric_validation(csv_factory,column,value):
    d=fixture_frame(csv_factory); d[column]=d[column].astype(object)
    d.loc[10,column]=value
    with pytest.raises(ValueError): fixture_data(csv_factory,d)

def test_next_observation_alignment(csv_factory):
    d=fixture_data(csv_factory); s=make_gold_samples(d)
    assert s.y.loc[10]==d.frame.loc[11,"Adj Close"]
    assert s.X.loc[10,"Close"]==d.frame.loc[10,"Close"]
    assert s.timeline.loc[10,"target_row"]==11
    assert s.timeline.loc[10,"target_date"]>s.timeline.loc[10,"source_date"]
    assert "Adj Close" not in s.X and "Date" not in s.X
    assert tuple(s.X.columns)==FEATURES

def test_future_changes_cannot_change_predictors(csv_factory):
    d=fixture_data(csv_factory); before=make_gold_samples(d)
    frame=d.frame.copy()
    frame.loc[11:,["Open","High","Low","Close","Adj Close","Volume"]]*=3
    after=make_gold_samples(GoldData(frame,d.path,d.sha256))
    pd.testing.assert_series_equal(before.X.loc[10],after.X.loc[10])
    assert before.y.loc[10]!=after.y.loc[10]

def test_historical_lags_rolling_returns(csv_factory):
    d=fixture_data(csv_factory); row=make_gold_samples(d).X.loc[10]; c=d.frame.Close
    assert row.close_lag_1==c.loc[9] and row.close_lag_5==c.loc[5]
    assert row.close_mean_prev_5==pytest.approx(c.loc[5:9].mean())
    assert row.close_std_prev_5==pytest.approx(c.loc[5:9].std(ddof=1))
    assert row.close_return_1==pytest.approx(c.loc[10]/c.loc[9]-1)
    assert row.close_return_5==pytest.approx(c.loc[10]/c.loc[5]-1)

def test_leading_history_and_terminal_exclusions(csv_factory):
    s=make_gold_samples(fixture_data(csv_factory))
    assert s.X.index.tolist()==list(range(5,79))
    assert [r["source_row"] for r in s.exclusions]==[0,1,2,3,4,79]
    assert all(r["reasons"]==["leading_history_unavailable"] for r in s.exclusions[:5])
    assert s.exclusions[-1]["reasons"]==["no_next_recorded_observation"]

def test_long_gap_excludes_only_pair(csv_factory):
    d=fixture_frame(csv_factory)
    d.loc[20:,"Date"]=(pd.to_datetime(d.loc[20:,"Date"])+pd.Timedelta(days=20)).dt.strftime("%Y-%m-%d")
    s=make_gold_samples(fixture_data(csv_factory,d))
    assert 19 not in s.X.index and 20 in s.X.index and 18 in s.X.index
    e=next(r for r in s.exclusions if r["source_row"]==19)
    assert e["gap_calendar_days"]>6
    assert e["reasons"]==["source_target_gap_exceeds_6_calendar_days"]

def test_chronological_split_and_purge(csv_factory):
    s=make_gold_samples(fixture_data(csv_factory)); part=chronological_split(s)
    assert len(part.test_indices)==int(np.ceil(len(s.X)*0.2))
    assert len(part.purged_indices)==1
    assert not set(part.train_indices)&set(part.test_indices)
    assert part.train_indices+part.purged_indices+part.test_indices==s.X.index.tolist()
    assert s.timeline.loc[part.train_indices,"target_date"].max()<s.timeline.loc[part.test_indices,"source_date"].min()

def test_cv_training_only_and_ordered(csv_factory):
    part=chronological_split(make_gold_samples(fixture_data(csv_factory)))
    folds=validate_cv_folds(part)
    assert len(folds)==5 and make_time_series_cv().gap==1
    for f in folds:
        assert f["train_target_end"]<f["validation_source_start"]
        assert set(f["train_source_indices"]).issubset(part.train_indices)
        assert set(f["validation_source_indices"]).issubset(part.train_indices)
        assert not set(f["validation_source_indices"])&set(part.test_indices)
        assert len(f["gap_source_indices"])==1

def test_reproducibility_and_replay(csv_factory,tmp_path):
    d=fixture_data(csv_factory)
    first=chronological_split(make_gold_samples(d))
    m=make_gold_manifest(first)
    assert m==make_gold_manifest(chronological_split(make_gold_samples(d)))
    p=tmp_path/"manifest.json"; p.write_text(json.dumps(m),encoding="utf-8")
    replay=replay_gold_manifest(p,d)
    pd.testing.assert_frame_equal(replay.X_train,first.X_train)
    pd.testing.assert_frame_equal(replay.X_test,first.X_test)
    pd.testing.assert_series_equal(replay.y_train,first.y_train)

def test_manifest_tampering(csv_factory,tmp_path):
    d=fixture_data(csv_factory); m=make_gold_manifest(chronological_split(make_gold_samples(d)))
    m["test_source_indices"][0]=m["train_source_indices"][0]
    p=tmp_path/"manifest.json"; p.write_text(json.dumps(m),encoding="utf-8")
    with pytest.raises(ValueError,match="fingerprint"): replay_gold_manifest(p,d)

def test_real_csv_unchanged():
    before=DATASET_PATH.read_bytes()
    d=load_gold_data(); validate_cv_folds(chronological_split(make_gold_samples(d)))
    assert DATASET_PATH.read_bytes()==before
    assert d.frame.shape==(1718,81)

def test_gold_manifest_and_source_contract():
    from src.gold_data_pipeline import MANIFEST_PATH
    part=replay_gold_manifest()
    manifest=json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"]==hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    assert part.X_train.shape==(1367,11) and part.X_test.shape==(342,11)
    assert tuple(part.X_train.columns)==FEATURES

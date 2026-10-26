import numpy as np
import pandas as pd
import pytest

from autonomis.data import load_table
from autonomis.features import engineer, make_preprocessor
from autonomis.splits import create_splits, read_splits, save_splits

SOURCE = "data/reference/ibm_telco/Telco-Customer-Churn.csv"


def test_saved_splits_are_disjoint_reproducible_and_bound_to_input(tmp_path):
    data = load_table(SOURCE)
    splits = create_splits(data)
    assert splits == create_splits(data)
    groups = [set(splits[name]) for name in ("train", "search", "audit")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == set(data.ids)
    assert set(splits["probe_train"]) | set(splits["probe_valid"]) == groups[0] | groups[1]
    assert not set(splits["probe_train"]) & set(splits["probe_valid"])
    path = tmp_path / "splits.json"
    save_splits(path, splits)
    assert read_splits(path, data) == splits
    with pytest.raises(ValueError):
        save_splits(path, {**splits, "seed": 1})


def test_corrupt_membership_is_rejected_even_with_same_source_hash(tmp_path):
    data = load_table(SOURCE)
    splits = create_splits(data)
    splits["train"][0] = splits["audit"][0]
    path = tmp_path / "corrupt.json"
    save_splits(path, splits)
    with pytest.raises(ValueError):
        read_splits(path, data)


def test_row_features_do_not_divide_zero_or_imputed_charges():
    X = load_table(SOURCE).X.head(3).copy()
    X["tenure"] = [0.0, 2.0, 4.0]
    X["TotalCharges"] = [np.nan, 80.0, 200.0]
    result = engineer(X, "V1")
    assert result.historical_monthly_charge.isna().iloc[0]
    assert result.historical_monthly_charge.iloc[1] == 40
    assert result.historical_monthly_charge.iloc[2] == 50
    assert np.isfinite(result.tenure_log1p).all()
    assert "historical_monthly_charge" not in X


def test_preprocessing_uses_training_statistics_and_accepts_unknown_categories():
    data = load_table(SOURCE)
    train = data.X.head(3).copy()
    train["TotalCharges"] = [10.0, 20.0, np.nan]
    valid = train.iloc[[0]].copy()
    valid["TotalCharges"] = np.nan
    valid["Contract"] = "previously unseen"
    transform = make_preprocessor(train, scale_numeric=False)
    transform.fit(train)  # Preprocessing only: this is not predictive model training.
    values = transform.transform(valid)
    dense = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    assert dense[0, 2] == 15.0
    assert np.isfinite(dense).all()
    assert list(transform.get_feature_names_out())[2] == "numeric__TotalCharges"


def test_missing_engineered_sources_fail_instead_of_guessing():
    with pytest.raises(ValueError):
        engineer(pd.DataFrame({"tenure": [1]}), "V1")

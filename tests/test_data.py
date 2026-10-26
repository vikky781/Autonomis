from pathlib import Path

import pandas as pd
import pytest

from autonomis.data import CATEGORICAL, NUMERIC, load_table, validate_table

SOURCE = Path(__file__).resolve().parents[1] / "data/reference/ibm_telco/Telco-Customer-Churn.csv"


def test_reference_schema_preserves_missing_charges_and_drops_target_from_features():
    data = load_table(SOURCE)
    assert data.X.shape == (7043, 19)
    assert data.y.sum() == 1869
    assert data.X.TotalCharges.isna().sum() == 11
    assert data.ids.is_unique
    assert "Churn" not in data.X and "customerID" not in data.X
    assert len(CATEGORICAL) == 16 and len(NUMERIC) == 3


@pytest.mark.parametrize("column,value", [("Churn", "Maybe"), ("tenure", -1),
    ("MonthlyCharges", "broken"), ("SeniorCitizen", 2), ("customerID", "")])
def test_bad_inputs_fail_closed(column, value):
    frame = pd.read_csv(SOURCE, dtype=str).head(4)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_table(frame)


def test_duplicate_identity_and_missing_features_are_rejected():
    frame = pd.read_csv(SOURCE, dtype=str).head(4)
    frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    with pytest.raises(ValueError, match="unique"):
        validate_table(frame)
    with pytest.raises(ValueError, match="Missing"):
        validate_table(frame.drop(columns="Contract"))


def test_inference_requires_features_but_does_not_require_labels():
    raw = pd.read_csv(SOURCE, dtype=str).head(4).drop(columns="Churn")
    data = validate_table(raw, require_target=False)
    assert data.y is None and len(data.X) == 4


def test_dataset_fingerprint_is_verified():
    with pytest.raises(ValueError, match="fingerprint"):
        load_table(SOURCE, expected_sha256="0" * 64)

"""Reference schema validation without learning from any target values."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
)
FEATURES = NUMERIC + CATEGORICAL


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series | None
    ids: pd.Series
    sha256: str = ""


def validate_table(frame: pd.DataFrame, *, require_target: bool = True) -> Dataset:
    required = {*FEATURES, "customerID"} | ({"Churn"} if require_target else set())
    if frame.columns.duplicated().any() or required - set(frame.columns):
        raise ValueError(f"Missing or duplicate columns: {sorted(required - set(frame.columns))}")
    if len(frame) == 0:
        raise ValueError("Empty table")
    ids = frame.customerID.astype("string").str.strip().reset_index(drop=True)
    if ids.isna().any() or ids.eq("").any() or not ids.is_unique:
        raise ValueError("customerID must be nonempty and unique")
    X = frame.loc[:, FEATURES].copy().reset_index(drop=True)
    for name in NUMERIC:
        raw = X[name].astype("string").str.strip()
        missing = raw.isna() | raw.eq("")
        if name != "TotalCharges" and missing.any():
            raise ValueError(f"Missing required numeric values: {name}")
        try:
            values = pd.to_numeric(raw.mask(missing), errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid numeric values: {name}") from error
        if ((~np.isfinite(values) & values.notna()) | values.lt(0)).any():
            raise ValueError(f"Invalid numeric range: {name}")
        if name == "tenure" and values.mod(1).ne(0).any():
            raise ValueError("tenure must be integral")
        X[name] = values
    for name in CATEGORICAL:
        X[name] = X[name].astype("string").str.strip().replace("", pd.NA)
        if name == "SeniorCitizen" and not X[name].isin(["0", "1"]).all():
            raise ValueError("SeniorCitizen must contain 0 or 1")
        X[name] = X[name].fillna("__MISSING__").astype(str)
    y = None
    if require_target:
        target = frame.Churn.astype("string").str.strip()
        if not target.isin(["Yes", "No"]).all():
            raise ValueError("Churn must contain only Yes or No")
        y = target.map({"Yes": 1, "No": 0}).astype(int).reset_index(drop=True)
    return Dataset(X, y, ids)


def load_table(path: str | Path, *, expected_sha256: str | None = None,
               require_target: bool = True) -> Dataset:
    digest = file_hash(path)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError("Dataset fingerprint does not match the frozen input")
    data = validate_table(pd.read_csv(path, dtype=str, keep_default_na=False),
                          require_target=require_target)
    data.sha256 = digest
    return data

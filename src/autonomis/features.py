"""Row-local feature design and train-only preprocessing; no predictors are fitted here."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autonomis.data import CATEGORICAL, FEATURES, NUMERIC

ADDONS = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies")
ENGINEERED = ("tenure_log1p", "active_addons_count", "historical_monthly_charge")


def engineer(X, version="V0"):
    if version not in {"V0", "V1"} or set(FEATURES) - set(X):
        raise ValueError("Missing feature sources or unknown feature version")
    result = X.loc[:, FEATURES].copy()
    if version == "V1":
        result["tenure_log1p"] = np.log1p(result.tenure)
        result["active_addons_count"] = result.loc[:, ADDONS].eq("Yes").sum(axis=1)
        result["historical_monthly_charge"] = result.TotalCharges.div(
            result.tenure.where(result.tenure.gt(0)))
    return result


def make_preprocessor(X, *, scale_numeric=True):
    numeric = list(NUMERIC) + [name for name in ENGINEERED if name in X]
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True,
                                        keep_empty_features=True))]
    if scale_numeric:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                       dtype=np.float32), list(CATEGORICAL)),
    ], sparse_threshold=1.0, remainder="drop")

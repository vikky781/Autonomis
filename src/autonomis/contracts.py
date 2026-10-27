"""Immutable experiment inputs and explicit training authorization."""

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


DEFAULTS = {
    "lr": {"C": 1.0, "max_iter": 3000, "tol": 0.0001, "solver": "lbfgs"},
    "catboost": {
        "depth": 6, "l2_leaf_reg": 3.0, "learning_rate": 0.05, "iterations": 1000,
        "loss_function": "Logloss", "eval_metric": "AUC", "boosting_type": "Ordered",
        "bootstrap_type": "Bayesian", "bagging_temperature": 1.0, "border_count": 254,
        "max_ctr_complexity": 2, "one_hot_max_size": 2, "counter_calc_method": "SkipTest",
    },
    "lightgbm": {
        "num_leaves": 15, "min_child_samples": 50, "reg_lambda": 1.0,
        "learning_rate": 0.05, "n_estimators": 1000, "objective": "binary",
        "deterministic": True, "force_col_wise": True, "verbosity": -1,
    },
}

CATBOOST_GPU_DEFAULTS = {
    "depth": 6, "l2_leaf_reg": 3.0, "learning_rate": 0.05, "iterations": 1000,
    "loss_function": "Logloss", "eval_metric": "AUC", "boosting_type": "Plain",
    "border_count": 128, "max_ctr_complexity": 2, "devices": "0", "gpu_ram_part": 0.75,
}


def recipe_defaults(family, backend):
    if family == "catboost" and backend == "gpu":
        return CATBOOST_GPU_DEFAULTS
    return DEFAULTS[family]


@dataclass(frozen=True)
class Recipe:
    family: str
    params: Mapping[str, Any] = field(default_factory=dict)
    feature_version: str = "V0"
    backend: str = "cpu"
    seed: int = 20261026
    threads: int = 4

    def __post_init__(self):
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError("Backend must be CPU or GPU")
        if self.family not in DEFAULTS or self.feature_version not in {"V0", "V1"}:
            raise ValueError("Unknown family or feature version")
        if self.backend == "gpu" and self.family != "catboost":
            raise ValueError("GPU recipes are supported only for CatBoost")
        if not 1 <= self.threads <= 4 or not isinstance(self.seed, int):
            raise ValueError("Invalid thread count or seed")
        defaults = recipe_defaults(self.family, self.backend)
        if not isinstance(self.params, Mapping):
            raise ValueError("Recipe parameters must be a mapping")
        if set(self.params) - defaults.keys():
            raise ValueError("Unsupported model parameter for this family/backend")
        params = {**defaults, **self.params}
        for name, value in params.items():
            expected = defaults[name]
            if isinstance(expected, bool):
                valid_type = type(value) is bool
            elif isinstance(expected, int):
                valid_type = type(value) is int
            elif isinstance(expected, float):
                valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
            else:
                valid_type = type(value) is type(expected)
            if not valid_type:
                raise ValueError(f"Invalid parameter type: {name}")
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"Nonfinite parameter: {name}")
        for name in ("C", "max_iter", "tol", "depth", "l2_leaf_reg", "learning_rate",
                     "iterations", "num_leaves", "min_child_samples", "n_estimators"):
            if name in params and params[name] <= 0:
                raise ValueError(f"Nonpositive parameter: {name}")
        if self.backend == "gpu" and (params["devices"] != "0" or
                                      params["gpu_ram_part"] != 0.75):
            raise ValueError("GPU profile is fixed to device 0 and RAM fraction 0.75")
        object.__setattr__(self, "params", MappingProxyType(params))
        fingerprint(self.to_dict())

    def to_dict(self):
        return {"family": self.family, "params": dict(self.params),
                "feature_version": self.feature_version, "backend": self.backend,
                "seed": self.seed, "threads": self.threads}

    @property
    def key(self):
        return fingerprint(self.to_dict())

    def changed(self, **parameters):
        return replace(self, params={**self.params, **parameters})


@dataclass(frozen=True)
class Observation:
    attempt_id: int
    recipe: Recipe
    roc_auc: float
    average_precision: float
    train_roc_auc: float = 0.0
    fit_seconds: float = 0.0
    tree_count: int | None = None
    phase: str = "search"
    artifact: str | None = None
    parent_id: int | None = None

    def __post_init__(self):
        for value in (self.roc_auc, self.average_precision, self.train_roc_auc):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Scores must be finite and in [0, 1]")
        if not math.isfinite(self.fit_seconds) or self.fit_seconds < 0:
            raise ValueError("Invalid fit duration")

    def to_dict(self):
        return {**self.__dict__, "recipe": self.recipe.to_dict()}


class TrainingDisabled(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingPermit:
    enabled: bool = False

    def check(self):
        if self.enabled is not True:
            raise TrainingDisabled("Training is disabled in the campaign configuration")

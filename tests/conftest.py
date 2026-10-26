"""No predictive learning is permitted anywhere in the verification suite."""

import pytest
from sklearn.linear_model import LogisticRegression


@pytest.fixture(autouse=True)
def forbid_predictive_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A real predictive estimator must never fit in this test suite")

    monkeypatch.setattr(LogisticRegression, "fit", forbidden)

import numpy as np
import pytest

from autonomis.artifacts import load_bundle, save_bundle
from autonomis.contracts import Recipe
from autonomis.data import load_table
from autonomis.evaluation import positive_scores, score_predictions
from autonomis.predict import OutputContract, export_predictions, predict_bundle


class FixedPredictor:
    """A stored response fixture, not a fitted or learning model."""

    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        return np.tile([0.8, 0.2], (len(X), 1))


def test_auc_and_ap_are_computed_from_supplied_scores():
    scores = score_predictions([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert scores["roc_auc"] == 0.75
    assert scores["average_precision"] == pytest.approx(5 / 6)
    assert positive_scores(FixedPredictor(), np.zeros((3, 2))).tolist() == [0.8] * 3
    with pytest.raises(ValueError):
        score_predictions([0, 0], [0.1, 0.2])


def test_artifacts_reload_without_training_and_detect_corruption(tmp_path):
    directory = tmp_path / "model"
    saved = save_bundle(directory, model=FixedPredictor(), preprocessor=None,
                        recipe=Recipe("catboost"), provenance={"synthetic_fixture": True})
    data = load_table("data/reference/ibm_telco/Telco-Customer-Churn.csv")
    loaded = load_bundle(directory)
    assert predict_bundle(loaded, data.X.head(3)).tolist() == [0.8] * 3
    assert loaded.manifest["model_hash"] == saved["model_hash"]
    (directory / "model.joblib").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        load_bundle(directory)


def test_export_preserves_identity_and_refuses_overwrite_or_invalid_probabilities(tmp_path):
    target = tmp_path / "predictions.csv"
    contract = OutputContract()
    export_predictions(target, ["b", "a"], [0.8, 0.2], model_hash="model",
                       input_hash="input", contract=contract)
    assert target.read_text().splitlines() == ["customerID,churn_probability", "b,0.8", "a,0.2"]
    with pytest.raises(ValueError):
        export_predictions(target, ["b", "a"], [0.1, 0.2], model_hash="model",
                           input_hash="input", contract=contract)
    with pytest.raises(ValueError):
        export_predictions(tmp_path / "bad.csv", ["a"], [float("nan")],
                           model_hash="model", input_hash="input", contract=contract)
    assert not (tmp_path / "bad.csv").exists()

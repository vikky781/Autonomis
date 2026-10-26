import hashlib
import json
import socket
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from test_campaign import inline_fixture_attempt

import autonomis.campaign as campaign
from autonomis.config import load_config
from autonomis.contracts import Recipe, TrainingDisabled, TrainingPermit
from autonomis.evaluation import positive_scores
from autonomis.predict import OutputContract, export_predictions


class SparseResponse:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.tile([0.25, 0.75], (X.shape[0], 1))


def config_at(tmp_path):
    return replace(load_config("configs/campaign.yaml"), output_dir=tmp_path / "campaign",
                   permit=TrainingPermit(True, socket.gethostname()))


def test_sparse_scoring_uses_row_count_not_ambiguous_sparse_length():
    assert positive_scores(SparseResponse(), csr_matrix(np.zeros((2, 3)))).tolist() == [0.75, 0.75]


@pytest.mark.parametrize("enabled", ["false", 1])
def test_truthy_nonboolean_cannot_authorize_training(enabled):
    with pytest.raises(TrainingDisabled):
        TrainingPermit(enabled, socket.gethostname()).check()


def test_prepare_cannot_recreate_a_lost_ledger(tmp_path):
    config = config_at(tmp_path)
    _, _, ledger = campaign.prepare_campaign(config, kind="simulation")
    attempt = ledger.reserve("search", "one", "s")
    ledger.finish(attempt, success=False)
    ledger.path.unlink()
    with pytest.raises(ValueError, match="[Ll]edger"):
        campaign.prepare_campaign(config, kind="simulation")
    assert not ledger.path.exists()


def test_missing_runtime_after_consumed_attempt_cannot_restart_deadlines(tmp_path):
    config = config_at(tmp_path)
    _, _, ledger = campaign.prepare_campaign(config, kind="simulation")
    attempt = ledger.reserve("search", Recipe("lr").key, "s", details={"recipe": Recipe("lr").to_dict()})
    ledger.finish(attempt, success=False)
    with pytest.raises(ValueError, match="runtime"):
        campaign.run_campaign(config, kind="simulation", families=("lr",))
    assert not (config.output_dir / "runtime.json").exists()


def test_short_terminal_window_preserves_fallback_and_reports_its_holdout(tmp_path, monkeypatch):
    config = replace(config_at(tmp_path), timeout_seconds=600,
                     exploration_seconds=1000, terminal_seconds=1300)
    clock = [10000.0]
    monkeypatch.setattr(campaign.time, "time", lambda: clock[0])

    def slow_fixture(ledger, recipe, phase, membership_hash, job, **kwargs):
        row = inline_fixture_attempt(ledger, recipe, phase, membership_hash,
                                     {**job, "fixture_seconds": 600}, **kwargs)
        clock[0] = 11100.0  # Only 200 seconds remain; a 600-second fit must not start.
        return row

    monkeypatch.setattr(campaign, "run_attempt", slow_fixture)
    summary = campaign.run_campaign(config, kind="simulation", families=("lr",))
    assert summary["attempts_used"] == 1
    assert summary["submission_path"] == summary["proxy_path"]
    assert summary["final_has_internal_holdout_score"] is True
    assert summary["submission_internal_metrics"] == summary["audit_metrics"]


def test_prediction_sidecar_binds_actual_csv_bytes_and_detects_tampering(tmp_path):
    from autonomis.predict import verify_export

    path = tmp_path / "pred.csv"
    export_predictions(path, ["a"], [0.75], model_hash="model", input_hash="input",
                       contract=OutputContract())
    metadata = json.loads(path.with_suffix(".csv.metadata.json").read_text())
    assert metadata["output_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_export(path)["output_hash"] == metadata["output_hash"]
    path.write_text("customerID,churn_probability\na,0.1\n")
    with pytest.raises(ValueError, match="hash"):
        verify_export(path)

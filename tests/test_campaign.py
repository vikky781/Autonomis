from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

import autonomis.campaign as campaign
from autonomis.artifacts import load_bundle, save_bundle
from autonomis.config import load_config
from autonomis.contracts import TrainingDisabled, TrainingPermit
from autonomis.data import load_table
from autonomis.evaluation import score_predictions
from autonomis.reporting import render_report


class StoredResponse:
    classes_ = np.array([0, 1])

    def __init__(self, identities):
        self.identities = identities

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


def inline_fixture_attempt(ledger, recipe, phase, membership_hash, job, **kwargs):
    """Replace only the learning/process boundary; retain real budget and controller behavior."""
    attempt = ledger.reserve(phase, recipe.key, membership_hash, details=kwargs["details"])
    folder = kwargs["output_dir"] / "attempts" / str(attempt) / "artifact"
    model = StoredResponse(job["train_ids"])
    manifest = save_bundle(folder, model=model, preprocessor=None, recipe=recipe,
        provenance={"synthetic_fixture": True, "train_ids": job["train_ids"], "phase": phase,
                    "dataset_hash": job["dataset_hash"]})
    data = load_table(job["dataset_path"])
    lookup = dict(zip(data.ids, data.y, strict=True))
    valid = job["valid_ids"]
    metrics = score_predictions([lookup[i] for i in valid], [0.5] * len(valid)) if phase in {"search", "probe"} else {}
    if metrics:
        metrics["train_roc_auc"] = 0.5
    result = {"metrics": metrics, "fit_seconds": job.get("fixture_seconds", 0.01), "tree_count": 100,
              "artifact": str(folder), "model_hash": manifest["model_hash"],
              "validation_predictions": [0.5] * len(valid), "validation_ids": valid,
              "training_fraction": len(job["train_ids"]) / len(data.ids)}
    ledger.finish(attempt, success=True, result=result)
    return ledger.attempts()[-1]


def test_disabled_campaign_does_not_create_output_or_consume_budget(tmp_path):
    config = replace(load_config("configs/campaign.yaml"), output_dir=tmp_path / "denied",
                     permit=TrainingPermit())
    with pytest.raises(TrainingDisabled):
        campaign.run_campaign(config)
    assert not config.output_dir.exists()


def test_gpu_campaign_rejects_small_reference_before_reservation(tmp_path):
    config = replace(load_config("configs/campaign.yaml"), output_dir=tmp_path / "gpu", backend="gpu",
                     permit=TrainingPermit(True))
    with pytest.raises(ValueError, match="50,000"):
        campaign.run_campaign(config, kind="simulation")
    _, _, ledger = campaign.prepare_campaign(config, kind="simulation")
    assert ledger.used == 0


def test_start_new_round_preserves_history_and_shared_budget(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    raw = yaml.safe_load(Path("configs/campaign.yaml").read_text(encoding="utf-8"))
    raw["dataset"] = str(Path(raw["dataset"]).resolve())
    raw["output_dir"] = "campaigns/reference"
    config_path = config_dir / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    _, _, ledger = campaign.prepare_campaign(config)
    for index, phase in enumerate(["search"] * 5 + ["probe"] * 2 + ["audit", "final"]):
        attempt = ledger.reserve(phase, f"recipe-{index}", f"membership-{index}")
        ledger.finish(attempt, success=True, result={"synthetic": True})
    ledger.lock_audit({"submission": "model-p", "proxy": "model-a",
                       "audit_predictions": "predictions-a"})

    next_config, next_ledger = campaign.start_new_round(config, ledger)

    assert ledger.path.is_file()
    assert next_config.output_dir != config.output_dir
    assert next_config.seed == config.seed + 1
    assert next_ledger.audit_lock is None
    assert next_ledger.used == 9 and next_ledger.remaining == 6
    assert next_ledger.phase_counts() == {}
    resumed = load_config(config_path)
    assert resumed.output_dir == next_config.output_dir
    assert resumed.seed == next_config.seed
    report = render_report(next_ledger, next_config.output_dir / "report.html",
                           training_controls=True).read_text(encoding="utf-8")
    assert "Earlier campaign history" in report
    assert "reference" in report and "9 attempt(s) · Audit locked" in report


def test_complete_simulated_campaign_locks_distinct_proxy_and_final_roles(tmp_path, monkeypatch):
    config = replace(load_config("configs/campaign.yaml"), output_dir=tmp_path / "simulation",
                     permit=TrainingPermit(True))
    monkeypatch.setattr(campaign, "run_attempt", inline_fixture_attempt)
    final = campaign.run_campaign(config, kind="simulation", families=("lr", "catboost", "lightgbm"))
    assert final["kind"] == "simulation" and final["official_submission"] is False
    assert final["final_has_internal_holdout_score"] is False
    proxy = load_bundle(final["proxy_path"])
    submitted = load_bundle(final["submission_path"])
    assert len(submitted.manifest["provenance"]["train_ids"]) == 7043
    assert len(proxy.manifest["provenance"]["train_ids"]) < 7043
    assert final["audit_metrics"]["roc_auc"] == 0.5
    _, _, ledger = campaign.prepare_campaign(config, kind="simulation")
    assert ledger.used <= 15 and ledger.audit_lock is not None
    assert sum(row["phase"] == "audit" for row in ledger.attempts()) == 1
    assert sum(row["phase"] == "final" for row in ledger.attempts()) == 1
    before = ledger.used
    assert campaign.run_campaign(config, kind="simulation")["submission_hash"] == final["submission_hash"]
    assert ledger.used == before

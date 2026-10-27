import json
from pathlib import Path

import yaml

from autonomis.cli import main
from autonomis.ledger import Ledger
from autonomis.reporting import render_report


def configuration(tmp_path):
    raw = yaml.safe_load(Path("configs/campaign.yaml").read_text())
    raw["dataset"] = str(Path(raw["dataset"]).resolve())
    raw["output_dir"] = "campaigns/campaign"
    raw["enable_training"] = False
    path = tmp_path / "configs" / "campaign.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw))
    return ["--config", str(path)]


def test_profile_prepare_plan_and_report_never_reserve_a_fit(tmp_path, capsys):
    prefix = configuration(tmp_path)
    assert main([*prefix, "profile"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 7043
    assert main([*prefix, "preflight"]) == 0
    assert json.loads(capsys.readouterr().out)["training_performed"] is False
    assert main([*prefix, "prepare"]) == 0
    capsys.readouterr()
    assert main([*prefix, "plan"]) == 0
    proposed = json.loads(capsys.readouterr().out)
    assert proposed["proposal"]["recipe"]["family"] == "lr"
    assert proposed["attempts_used"] == 0
    assert main([*prefix, "report"]) == 0
    capsys.readouterr()
    assert main([*prefix, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["attempts_used"] == 0


def test_training_is_refused_even_with_execute_flag(tmp_path, capsys):
    prefix = configuration(tmp_path)
    assert main([*prefix, "train", "--execute"]) == 2
    assert "Training is disabled" in capsys.readouterr().err
    assert not (tmp_path / "campaigns" / "campaign").exists()


def test_reconcile_requires_explicit_operator_confirmation(tmp_path, capsys):
    prefix = configuration(tmp_path)
    assert main([*prefix, "prepare"]) == 0
    capsys.readouterr()
    assert main([*prefix, "reconcile"]) == 2
    assert "Verify the recorded worker" in capsys.readouterr().err
    assert main([*prefix, "reconcile", "--confirm-no-active-worker"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["interrupted"] is True and result["attempts_used"] == 0


def test_report_escapes_history_and_labels_simulations(tmp_path):
    ledger = Ledger.initialize(tmp_path / "sim.sqlite", dataset_hash="d", split_hash="s",
                               kind="simulation")
    attempt = ledger.reserve("search", "r", "s", details={"reason": "<script>alert(1)</script>"})
    ledger.finish(attempt, success=False, error="<bad>")
    path = render_report(ledger, tmp_path / "report.html")
    content = path.read_text()
    assert "SIMULATION" in content and "not model performance" in content
    assert "<script>" not in content and "&lt;script&gt;" in content
    assert "No trained submission artifact" in content


def test_report_zero_attempts_shows_empty_read_only_dashboard(tmp_path):
    ledger = Ledger.initialize(tmp_path / "empty.sqlite", dataset_hash="d", split_hash="s")

    content = render_report(ledger, tmp_path / "empty.html").read_text()

    assert "Autonomis" in content
    assert '<span id="shared-attempts-used">0</span> <span>/ 15</span>' in content
    assert "15 available" in content
    assert "No Search results recorded" in content
    assert "No experiment attempts have been recorded" in content
    assert "Hidden-test metrics: unavailable" in content
    assert "does not train or reserve attempts" in content


def test_report_separates_search_audit_and_hidden_test_evidence(tmp_path):
    ledger = Ledger.initialize(tmp_path / "campaign.sqlite", dataset_hash="d", split_hash="s")
    search_id = ledger.reserve("search", "search-key", "search-split", details={
        "reason": "Depth trial from anchor", "parent_id": 1,
        "recipe": {"family": "catboost", "backend": "cpu"},
    })
    ledger.finish(search_id, success=True, result={
        "metrics": {"roc_auc": 0.8123, "average_precision": 0.6021,
                    "train_roc_auc": 0.91},
        "fit_seconds": 12.5, "tree_count": 450, "artifact": "search.cbm",
    })
    audit_id = ledger.reserve("audit", "audit-key", "audit-split", details={
        "reason": "Frozen terminal fit", "parent_id": search_id,
        "recipe": {"family": "catboost", "backend": "cpu"},
    })
    ledger.finish(audit_id, success=True, result={
        "metrics": {"roc_auc": 0.997, "average_precision": 0.99},
        "fit_seconds": 18.0, "tree_count": 450, "artifact": "audit.cbm",
    })
    ledger.lock_audit({"submission": "submission-hash", "proxy": "proxy-hash",
                       "audit_predictions": "predictions-hash"})
    summary = {
        "kind": "real", "submission_role": "full_data_final",
        "submission_hash": "submission-hash", "submission_path": "models/final",
        "submission_training_fraction": 1.0, "proxy_path": "models/audit",
        "proxy_training_fraction": 0.8, "audit_metrics": {
            "roc_auc": 0.701, "average_precision": 0.311,
        }, "audit_metric_applies_to": "proxy-hash",
        "hidden_test_score": None,
    }

    content = render_report(ledger, tmp_path / "campaign.html", summary=summary).read_text()

    assert "Recorded campaign evidence" in content
    assert "Best Search ROC-AUC" in content and "0.812300" in content
    assert "Audit proxy metrics" in content and "0.701000" in content
    assert "0.311000" in content
    assert "proxy-hash" in content
    assert "submission-hash" in content
    assert "100% of training data" in content
    assert "80% of training data" in content
    assert "Hidden-test metrics: unavailable" in content
    assert "0.997000" not in content
    assert "0.990000" not in content
    assert "450" in content and "12.50 s" in content
    assert "Depth trial from anchor" in content
    assert "locked" in content.lower()
    assert "shared-budget-bar" in content and "search-trajectory" in content
    assert "SIMULATION" not in content


def test_report_flags_artifact_summary_that_disagrees_with_audit_lock(tmp_path):
    ledger = Ledger.initialize(tmp_path / "locked.sqlite", dataset_hash="d", split_hash="s")
    ledger.lock_audit({"submission": "locked-submission", "proxy": "locked-proxy",
                       "audit_predictions": "locked-predictions"})
    summary = {
        "submission_hash": "stale-submission", "submission_path": "models/final",
        "submission_training_fraction": 1.0, "proxy_path": "models/audit",
        "proxy_training_fraction": 0.8, "audit_metric_applies_to": "stale-proxy",
    }

    content = render_report(ledger, tmp_path / "mismatch.html", summary=summary).read_text()

    assert "Artifact identity mismatch" in content
    assert "not verified against the ledger lock" in content
    assert "stale-submission" in content and "locked-submission" in content

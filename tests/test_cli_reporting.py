import json
from pathlib import Path

import yaml

from autonomis.cli import main
from autonomis.ledger import Ledger
from autonomis.reporting import render_report


def configuration(tmp_path):
    raw = yaml.safe_load(Path("configs/campaign.yaml").read_text())
    raw["dataset"] = str(Path(raw["dataset"]).resolve())
    raw["output_dir"] = "campaign"
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
    assert not (tmp_path / "campaign").exists()


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

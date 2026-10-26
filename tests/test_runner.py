import socket
import sys
from types import SimpleNamespace

import pytest

from autonomis.contracts import Recipe, TrainingDisabled, TrainingPermit
from autonomis.ledger import Ledger
from autonomis.runner import run_attempt
from autonomis.training import build_cpu_estimator, build_estimator, fit_cpu_job, gpu_preflight


def fixed_worker(job):
    return {"synthetic_fixture": True, "value": job["attempt_id"]}


def failing_worker(job):
    raise RuntimeError("deliberate fake-worker failure")


def never_finishes(job):
    import time
    time.sleep(20)


def ledger_at(tmp_path):
    return Ledger.initialize(tmp_path / "campaign.sqlite", dataset_hash="data",
                             split_hash="split", kind="simulation")


def test_training_denial_happens_before_reservation_or_worker(tmp_path):
    ledger = ledger_at(tmp_path)
    with pytest.raises(TrainingDisabled):
        run_attempt(ledger, Recipe("lr"), "search", "part", {},
                    output_dir=tmp_path, permit=TrainingPermit(), worker=fixed_worker)
    assert ledger.used == 0
    with pytest.raises(TrainingDisabled):
        fit_cpu_job({})


def test_cpu_constructor_is_inspected_without_fitting():
    model = build_cpu_estimator(Recipe("lr"))
    assert model.get_params()["C"] == 1.0
    assert model.get_params()["max_iter"] == 3000
    assert not hasattr(model, "coef_")


def test_gpu_constructor_is_inspected_without_fitting(monkeypatch):
    class FakeCatBoostClassifier:
        def __init__(self, **params):
            self.params = params

    monkeypatch.setitem(sys.modules, "catboost", SimpleNamespace(
        CatBoostClassifier=FakeCatBoostClassifier))
    model = build_estimator(Recipe("catboost", backend="gpu"))
    assert model.params["task_type"] == "GPU"
    assert model.params["devices"] == "0"
    assert model.params["gpu_ram_part"] == 0.75
    assert model.params["boosting_type"] == "Plain"
    with pytest.raises(ValueError, match="CPU constructor"):
        build_cpu_estimator(Recipe("catboost", backend="gpu"))


def test_gpu_preflight_checks_rows_driver_memory_and_package_without_training(monkeypatch):
    with pytest.raises(ValueError, match="50,000"):
        gpu_preflight(50_000)
    monkeypatch.setattr("autonomis.training.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("autonomis.training.version", lambda name: "1.2.8")
    completed = SimpleNamespace(returncode=0, stderr="",
        stdout="0, NVIDIA RTX 4060 Laptop GPU, 7800, 575.57.08\n")
    monkeypatch.setattr("autonomis.training.subprocess.run", lambda *args, **kwargs: completed)
    result = gpu_preflight(50_001)
    assert result["device_index"] == 0
    assert result["free_vram_mib"] == 7800
    assert result["catboost_version"] == "1.2.8"
    assert result["training_performed"] is False


def test_gpu_preflight_rejects_insufficient_free_vram(monkeypatch):
    monkeypatch.setattr("autonomis.training.importlib.util.find_spec", lambda name: object())
    completed = SimpleNamespace(returncode=0, stderr="",
                                stdout="0, GPU, 6000, driver\n")
    monkeypatch.setattr("autonomis.training.subprocess.run", lambda *args, **kwargs: completed)
    with pytest.raises(RuntimeError, match="6144 MiB"):
        gpu_preflight(50_001)


def test_fake_worker_result_is_durable_and_duplicate_does_not_execute_again(tmp_path):
    ledger = ledger_at(tmp_path)
    kwargs = dict(output_dir=tmp_path, permit=TrainingPermit(True, socket.gethostname()),
                  worker=fixed_worker, timeout_seconds=30)
    first = run_attempt(ledger, Recipe("lr"), "search", "part", {}, **kwargs)
    assert first["status"] == "succeeded" and first["result"]["value"] == 1
    again = run_attempt(ledger, Recipe("lr"), "search", "part", {}, **kwargs)
    assert again["id"] == first["id"] and ledger.used == 1


def test_worker_failure_and_timeout_never_refund_tokens(tmp_path):
    ledger = ledger_at(tmp_path)
    kwargs = dict(output_dir=tmp_path, permit=TrainingPermit(True, socket.gethostname()))
    failed = run_attempt(ledger, Recipe("lr"), "search", "part", {},
                         worker=failing_worker, timeout_seconds=30, **kwargs)
    assert failed["status"] == "failed" and "deliberate" in failed["error"]
    timed = run_attempt(ledger, Recipe("lr", {"C": 10}), "search", "part", {},
                        worker=never_finishes, timeout_seconds=0.2, **kwargs)
    assert timed["status"] == "failed" and "timed out" in timed["error"]
    assert ledger.used == 2

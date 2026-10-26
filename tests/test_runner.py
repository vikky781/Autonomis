import socket

import pytest

from autonomis.contracts import Recipe, TrainingDisabled, TrainingPermit
from autonomis.ledger import Ledger
from autonomis.runner import run_attempt
from autonomis.training import build_cpu_estimator, fit_cpu_job


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

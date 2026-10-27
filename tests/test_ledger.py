from concurrent.futures import ThreadPoolExecutor

import pytest

from autonomis.ledger import BudgetError, Ledger


def ledger_at(path):
    return Ledger.initialize(path, dataset_hash="data", split_hash="split", kind="simulation")


def complete(ledger, phase, number, success=True):
    attempt = ledger.reserve(phase, f"recipe-{number}", f"split-{number}")
    ledger.finish(attempt, success=success, result={"synthetic": True})
    return attempt


def test_failures_remain_consumed_and_attempt_sixteen_is_refused(tmp_path):
    ledger = ledger_at(tmp_path / "runs.sqlite")
    for index, phase in enumerate(["search"] * 11 + ["probe"] * 2 + ["audit", "final"]):
        complete(ledger, phase, index, success=index != 0)
    reopened = Ledger(ledger.path, dataset_hash="data", split_hash="split")
    assert reopened.used == 15
    assert reopened.attempts()[0]["status"] == "failed"
    with pytest.raises(BudgetError):
        reopened.reserve("final", "sixteenth", "full")


def test_phase_limits_and_active_worker_prevent_overspending(tmp_path):
    ledger = ledger_at(tmp_path / "runs.sqlite")
    first = ledger.reserve("search", "one", "search")
    with pytest.raises(BudgetError, match="active"):
        ledger.reserve("search", "two", "search")
    ledger.finish(first, success=False)
    for index in range(1, 11):
        complete(ledger, "search", index)
    with pytest.raises(BudgetError):
        ledger.reserve("search", "extra", "search")
    assert ledger.remaining == 4


def test_concurrent_callers_cannot_both_reserve_last_token(tmp_path):
    ledger = ledger_at(tmp_path / "runs.sqlite")
    for index, phase in enumerate(["search"] * 11 + ["probe"] * 2 + ["audit"]):
        complete(ledger, phase, index)

    def reserve(_):
        try:
            return ledger.reserve("final", "last", "full")
        except BudgetError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, range(2)))
    assert sum(value is not None for value in results) == 1
    assert ledger.used == 15


def test_audit_lock_survives_restart_and_cannot_change_artifacts(tmp_path):
    ledger = ledger_at(tmp_path / "runs.sqlite")
    lock = {"submission": "model-p", "proxy": "model-a", "audit_predictions": "pred-a"}
    ledger.lock_audit(lock)
    reopened = Ledger(ledger.path, dataset_hash="data", split_hash="split")
    assert reopened.audit_lock == lock
    with pytest.raises(BudgetError, match="audit"):
        reopened.reserve("search", "later", "search")
    with pytest.raises(ValueError, match="locked"):
        reopened.lock_audit({**lock, "submission": "different"})


def test_new_campaign_keeps_old_audit_locked_and_uses_only_global_remaining(tmp_path):
    root = tmp_path / "campaigns"
    first = Ledger.initialize(root / "reference" / "ledger.sqlite", dataset_hash="data",
                               split_hash="split-one", kind="real")
    for index, phase in enumerate(["search"] * 5 + ["probe"] * 2 + ["audit", "final"]):
        attempt = first.reserve(phase, f"recipe-{index}", f"split-{index}")
        first.finish(attempt, success=True, result={"synthetic": True})
    lock = {"submission": "model-p", "proxy": "model-a", "audit_predictions": "pred-a"}
    first.lock_audit(lock)

    second = Ledger.initialize(root / "reference-round-02" / "ledger.sqlite",
                               dataset_hash="data", split_hash="split-two", kind="real")

    assert first.audit_lock == lock
    assert second.audit_lock is None
    assert second.used == 9
    assert second.remaining == 6
    assert second.phase_counts() == {}
    next_attempt = second.reserve("search", "new-round-recipe", "new-round-split")
    second.finish(next_attempt, success=True, result={"synthetic": True})
    assert second.used == 10
    with pytest.raises(BudgetError, match="Audit lock"):
        first.reserve("search", "old-round-recipe", "old-round-split")


def test_missing_corrupt_or_changed_campaign_is_not_silently_reset(tmp_path):
    with pytest.raises(ValueError):
        Ledger(tmp_path / "absent.sqlite", dataset_hash="data", split_hash="split")
    ledger = ledger_at(tmp_path / "runs.sqlite")
    with pytest.raises(ValueError, match="fingerprint"):
        Ledger(ledger.path, dataset_hash="changed", split_hash="split")
    with pytest.raises(ValueError):
        Ledger.initialize(ledger.path, dataset_hash="data", split_hash="split")


def test_interruption_requires_explicit_worker_reconciliation(tmp_path):
    ledger = ledger_at(tmp_path / "runs.sqlite")
    ledger.reserve("search", "interrupted", "search")
    with pytest.raises(ValueError, match="worker"):
        ledger.reconcile_interrupted()
    ledger.reconcile_interrupted(confirm_no_active_worker=True)
    assert ledger.used == 1
    assert ledger.attempts()[0]["status"] == "interrupted"

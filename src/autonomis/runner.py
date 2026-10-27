"""Supervise one explicitly requested worker; failures never refund reservations."""

import multiprocessing
import time
import traceback
from pathlib import Path

from autonomis.contracts import TrainingDisabled
from autonomis.storage import read_json, write_json


def _worker_entry(function, job, destination, start_event):
    try:
        if not start_event.wait(30):
            raise RuntimeError("Parent did not authorize worker start")
        result = {"ok": True, "result": function(job)}
    except BaseException:
        result = {"ok": False, "error": traceback.format_exc(limit=8)}
    write_json(destination, result)


def run_attempt(ledger, recipe, phase, membership_hash, job, *, output_dir,
                permit, timeout_seconds=1800, worker=None, details=None):
    permit.check()  # Must precede reservation, construction, process creation and cache access.
    simulation = ledger.identity["kind"] == "simulation"
    if worker is not None and not simulation:
        raise ValueError("Injected workers are allowed only in explicitly marked simulations")
    if simulation and worker is None:
        raise TrainingDisabled("A simulation cannot invoke real estimator training")
    if timeout_seconds <= 0:
        raise ValueError("Worker timeout must be positive")
    for row in ledger.attempts():
        if (row["phase"], row["recipe_key"], row["split_hash"]) == (
                phase, recipe.key, membership_hash) and row["status"] == "succeeded":
            return row
    if worker is None:
        from autonomis.training import fit_job
        worker = fit_job
    attempt_id = ledger.reserve(phase, recipe.key, membership_hash,
                                 details={"recipe": recipe.to_dict(), **(details or {})})
    start = time.monotonic()
    process = None
    try:
        folder = Path(output_dir) / "attempts" / f"{attempt_id:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        destination = folder / "worker-result.json"
        payload = {**job, "recipe": recipe.to_dict(), "phase": phase,
                   "attempt_id": attempt_id, "artifact_dir": str(folder / "artifact"),
                   "permit": {"enabled": permit.enabled}}
        context = multiprocessing.get_context("spawn")
        start_event = context.Event()
        process = context.Process(target=_worker_entry,
                                  args=(worker, payload, str(destination), start_event))
        process.start()
        ledger.mark_running(attempt_id, process.pid)
        start_event.set()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            if process.is_alive():
                raise RuntimeError("Worker could not be stopped; manual reconciliation required")
            ledger.finish(attempt_id, success=False, error="Worker timed out; token remains consumed")
        elif not destination.exists() or process.exitcode != 0:
            ledger.finish(attempt_id, success=False, error=f"Worker exited without result: {process.exitcode}")
        else:
            outcome = read_json(destination)
            result = outcome.get("result", {})
            result["wall_seconds"] = time.monotonic() - start
            ledger.finish(attempt_id, success=outcome["ok"], result=result,
                          error=outcome.get("error"))
    except BaseException as error:
        if process is not None and process.is_alive():
            # Leave the reservation active rather than authorize another fit beside an unknown worker.
            raise RuntimeError("Active worker must be reconciled before continuing") from error
        current = next(row for row in ledger.attempts() if row["id"] == attempt_id)
        if current["status"] in {"reserved", "running"}:
            ledger.finish(attempt_id, success=False, error=str(error))
        if not isinstance(error, Exception):
            raise
    return next(row for row in ledger.attempts() if row["id"] == attempt_id)

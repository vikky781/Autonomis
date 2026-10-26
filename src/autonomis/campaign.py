"""Sequential CPU orchestration; execution requires the named training authority."""

import importlib.util
import socket
import time
from pathlib import Path

from autonomis.artifacts import load_bundle
from autonomis.contracts import Observation, Recipe, fingerprint
from autonomis.data import file_hash, load_table
from autonomis.evaluation import score_predictions
from autonomis.ledger import Ledger
from autonomis.planner import Planner, finalists, paired_winner
from autonomis.predict import OutputContract, export_predictions, predict_bundle
from autonomis.runner import run_attempt
from autonomis.splits import create_splits, read_splits, save_splits, split_hash
from autonomis.storage import read_json, write_json


def prepare_campaign(config, *, kind="real"):
    data = load_table(config.dataset, expected_sha256=config.expected_sha256)
    folder = config.output_dir
    manifest_path = folder / "splits.json"
    if manifest_path.exists():
        splits = read_splits(manifest_path, data)
    else:
        splits = create_splits(data, config.seed)
        save_splits(manifest_path, splits)
    frozen = {"dataset_role": config.dataset_role, "dataset_hash": data.sha256,
              "split_hash": split_hash(splits), "seed": config.seed, "threads": config.threads,
              "feature_ablation": config.feature_ablation, "kind": kind,
              "official_submission": config.official_submission,
              "timeout_seconds": config.timeout_seconds,
              "exploration_seconds": config.exploration_seconds,
              "terminal_seconds": config.terminal_seconds}
    write_json(folder / "campaign.json", frozen)
    path = folder / "ledger.sqlite"
    if path.exists():
        ledger = Ledger(path, dataset_hash=data.sha256, split_hash=split_hash(splits))
        if ledger.identity["kind"] != kind:
            raise ValueError("Real and simulated campaigns cannot share a ledger")
    else:
        ledger = Ledger.initialize(path, dataset_hash=data.sha256,
                                   split_hash=split_hash(splits), kind=kind)
    return data, splits, ledger


def observations(ledger, phase="search"):
    result = []
    for row in ledger.attempts():
        if row["phase"] != phase or row["status"] != "succeeded":
            continue
        metrics = row["result"]["metrics"]
        result.append(Observation(row["id"], Recipe(**row["details"]["recipe"]),
            metrics["roc_auc"], metrics["average_precision"], metrics.get("train_roc_auc", 0),
            row["result"]["fit_seconds"], row["result"]["tree_count"], phase,
            row["result"]["artifact"], row["details"].get("parent_id")))
    return result


def _verify(directory, data, splits, *, proxy=False):
    bundle = load_bundle(directory)
    if bundle.manifest["provenance"]["dataset_hash"] != data.sha256:
        raise ValueError("Artifact dataset fingerprint differs")
    sample = data.X.head(8)
    export_predictions(Path(directory) / "functional-check.csv", data.ids.head(8),
                       predict_bundle(bundle, sample), model_hash=bundle.manifest["model_hash"],
                       input_hash=fingerprint([data.sha256, data.ids.head(8).tolist()]),
                       contract=OutputContract())
    checked = {"path": str(directory), "model_hash": bundle.manifest["model_hash"],
               "training_fraction": len(bundle.manifest["provenance"]["train_ids"]) / len(data.ids)}
    if proxy:
        lookup = {value: index for index, value in enumerate(data.ids)}
        indices = [lookup[value] for value in splits["audit"]]
        predictions = predict_bundle(bundle, data.X.iloc[indices]).tolist()
        path = Path(directory) / "audit-predictions.json"
        write_json(path, {"ids": splits["audit"], "scores": predictions})
        checked.update(audit_path=str(path), audit_hash=file_hash(path))
    return checked


def _publish_audit(config, data, ledger, selection):
    proxy = selection["proxy"]
    for candidate in (selection["submission"], proxy):
        if load_bundle(candidate["path"]).manifest["model_hash"] != candidate["model_hash"]:
            raise ValueError("Selected artifact changed")
    if file_hash(proxy["audit_path"]) != proxy["audit_hash"]:
        raise ValueError("Audit predictions changed")
    ledger.lock_audit({"submission": selection["submission"]["model_hash"],
                       "proxy": proxy["model_hash"], "audit_predictions": proxy["audit_hash"]})
    saved = read_json(proxy["audit_path"])
    lookup = dict(zip(data.ids, data.y, strict=True))
    metrics = score_predictions([lookup[value] for value in saved["ids"]], saved["scores"])
    summary = {"kind": ledger.identity["kind"], "dataset_role": config.dataset_role,
               "official_submission": False, "official_output_contract": "unavailable",
               "attempts_used": ledger.used, "submission_path": selection["submission"]["path"],
               "submission_hash": selection["submission"]["model_hash"],
               "submission_training_fraction": selection["submission"]["training_fraction"],
               "proxy_path": proxy["path"], "proxy_training_fraction": proxy["training_fraction"],
               "audit_metrics": metrics, "audit_metric_applies_to": proxy["model_hash"],
               "final_has_internal_holdout_score": False, "hidden_test_score": None,
               "efficiency_factor": None, "technical_errors": selection["errors"]}
    write_json(config.output_dir / "summary.json", summary)
    return summary


def run_campaign(config, *, kind="real", families=None):
    config.permit.check()
    data, splits, ledger = prepare_campaign(config, kind=kind)
    selection_path = config.output_dir / "selection.json"
    if selection_path.exists():
        return _publish_audit(config, data, ledger, read_json(selection_path))
    available = tuple(families) if families is not None else tuple(
        family for family in ("lr", "catboost", "lightgbm")
        if family == "lr" or importlib.util.find_spec(family) is not None)
    runtime_path = config.output_dir / "runtime.json"
    if runtime_path.exists():
        runtime = read_json(runtime_path)
        if runtime["host"] != socket.gethostname() or runtime["families"] != list(available):
            raise ValueError("Training host or supported families changed on resume")
    else:
        runtime = {"host": socket.gethostname(), "families": list(available), "started_at": time.time()}
        write_json(runtime_path, runtime)
    exploration_deadline = runtime["started_at"] + config.exploration_seconds
    terminal_deadline = runtime["started_at"] + config.terminal_seconds
    planner = Planner(families=available, feature_ablation=config.feature_ablation,
                      seed=config.seed, threads=config.threads)

    def execute(recipe, phase, train_ids, valid_ids, *, details, fixed_count=None):
        deadline = exploration_deadline if phase in {"search", "probe"} else terminal_deadline
        timeout = min(config.timeout_seconds, deadline - time.time())
        if timeout <= 2:
            return None
        job = {"dataset_path": str(config.dataset), "dataset_hash": data.sha256,
               "train_ids": train_ids, "valid_ids": valid_ids, "fixed_tree_count": fixed_count}
        membership = fingerprint({"train": train_ids, "valid": valid_ids, "trees": fixed_count})
        return run_attempt(ledger, recipe, phase, membership, job, output_dir=config.output_dir,
                           permit=config.permit, timeout_seconds=timeout,
                           details={"recipe": recipe.to_dict(), **details})

    freeze_path = config.output_dir / "frozen-recipe.json"
    if not freeze_path.exists():
        while time.time() < exploration_deadline:
            attempted = [Recipe(**row["details"]["recipe"]) for row in ledger.attempts()
                         if row["phase"] == "search"]
            proposal = planner.propose(observations(ledger), attempted)
            if proposal is None:
                break
            row = execute(proposal.recipe, "search", splits["train"], splits["search"],
                          details=proposal.to_dict())
            if row is None:
                break
        choices = finalists(observations(ledger))
        if not choices:
            raise RuntimeError("No verified search artifact is available")
        slowest = max(o.fit_seconds for o in choices)
        now = time.time()
        can_probe = (len(choices) == 2 and ledger.remaining >= 4 and
                     choices[0].roc_auc - choices[1].roc_auc <= 0.01 and
                     now + 3 * slowest < exploration_deadline and
                     now + 7.5 * slowest + 2 < terminal_deadline)
        if can_probe:
            for choice in choices:
                previous = [r for r in ledger.attempts() if r["phase"] == "probe"
                            and r["recipe_key"] == choice.recipe.key]
                if previous:
                    continue
                if ledger.remaining <= 2 or time.time() + 6 * slowest + 2 >= terminal_deadline:
                    break
                execute(choice.recipe, "probe", splits["probe_train"], splits["probe_valid"],
                        details={"parent_id": choice.attempt_id, "reason": "Frozen paired check"},
                        fixed_count=choice.tree_count)
        winner = paired_winner(choices, observations(ledger, "probe"))
        original = next(o for o in observations(ledger) if o.recipe.key == winner.recipe.key)
        fallback = _verify(original.artifact, data, splits, proxy=True)
        frozen = {"recipe": original.recipe.to_dict(), "tree_count": original.tree_count,
                  "fallback": fallback, "source_attempt": original.attempt_id}
        write_json(freeze_path, frozen)
    frozen = read_json(freeze_path)
    recipe = Recipe(**frozen["recipe"])
    fallback = _verify(frozen["fallback"]["path"], data, splits, proxy=True)
    candidates, errors = {}, []
    for phase, train_ids, valid_ids in (
            ("audit", [*splits["train"], *splits["search"]], splits["audit"]),
            ("final", data.ids.tolist(), [])):
        existing = [row for row in ledger.attempts() if row["phase"] == phase]
        row = existing[0] if existing else execute(recipe, phase, train_ids, valid_ids,
                details={"parent_id": frozen["source_attempt"], "reason": "Frozen terminal fit"},
                fixed_count=frozen["tree_count"])
        if row and row["status"] == "succeeded":
            try:
                candidates[phase] = _verify(row["result"]["artifact"], data, splits,
                                             proxy=phase == "audit")
            except Exception as error:
                errors.append(f"{phase} artifact validation: {error}")
        else:
            errors.append(f"{phase}: {row['error'] if row else 'skipped for deadline'}")
    selection = {"submission": candidates.get("final", candidates.get("audit", fallback)),
                 "proxy": candidates.get("audit", fallback), "errors": errors}
    write_json(selection_path, selection)
    return _publish_audit(config, data, ledger, selection)

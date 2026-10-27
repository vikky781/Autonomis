"""Safe-by-default local tools. No command implicitly trains a model."""

import argparse
import json
import sys

from autonomis.artifacts import load_bundle
from autonomis.campaign import observations, prepare_campaign, run_campaign
from autonomis.config import load_config
from autonomis.contracts import Recipe, TrainingDisabled
from autonomis.data import load_table
from autonomis.ledger import Ledger
from autonomis.planner import Planner
from autonomis.predict import OutputContract, export_predictions, predict_bundle
from autonomis.reporting import render_report
from autonomis.storage import read_json
from autonomis.training import gpu_preflight


def _open_ledger(config):
    frozen = read_json(config.output_dir / "campaign.json")
    return Ledger(config.output_dir / "ledger.sqlite", dataset_hash=config.expected_sha256,
                  split_hash=frozen["split_hash"], history_ledgers=config.history_ledgers)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Budgeted churn tooling; training disabled by default")
    parser.add_argument("--config", default="configs/campaign.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("profile", "prepare", "preflight", "status", "plan", "report"):
        commands.add_parser(name)
    reconcile = commands.add_parser("reconcile", help="Mark a verified stopped worker interrupted")
    reconcile.add_argument("--confirm-no-active-worker", action="store_true",
                           help="Required operator confirmation before changing ledger state")
    training = commands.add_parser("train", help="Requires explicit host authorization and --execute")
    training.add_argument("--execute", action="store_true")
    prediction = commands.add_parser("predict", help="Load a trusted bundle; never fit")
    prediction.add_argument("--artifact", required=True)
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "train":
            if not args.execute:
                raise TrainingDisabled("Training requires --execute and a configured training host")
            config.permit.check()
            value = run_campaign(config)
        elif args.command == "profile":
            data = load_table(config.dataset, expected_sha256=config.expected_sha256)
            value = {"dataset_role": config.dataset_role, "sha256": data.sha256,
                     "rows": len(data.ids), "predictors": len(data.X.columns),
                     "positive_labels": int(data.y.sum()),
                     "missing_TotalCharges": int(data.X.TotalCharges.isna().sum()),
                     "official_equivalence": "unverified", "training_performed": False}
        elif args.command == "preflight":
            value = {"backend": config.backend, "training_performed": False}
            if config.backend == "gpu":
                data = load_table(config.dataset, expected_sha256=config.expected_sha256)
                value.update(gpu_preflight(len(data.ids)))
        elif args.command == "prepare":
            _, splits, ledger = prepare_campaign(config)
            value = {"attempts_used": ledger.used, "output_dir": str(config.output_dir),
                     "partition_sizes": {k: len(splits[k]) for k in
                         ("train", "search", "audit", "probe_train", "probe_valid")}}
        elif args.command == "predict":
            bundle = load_bundle(args.artifact)
            data = load_table(args.input, require_target=False)
            value = export_predictions(args.output, data.ids, predict_bundle(bundle, data.X),
                                       model_hash=bundle.manifest["model_hash"], input_hash=data.sha256,
                                       contract=OutputContract())
        elif args.command == "reconcile":
            ledger = _open_ledger(config)
            ledger.reconcile_interrupted(confirm_no_active_worker=args.confirm_no_active_worker)
            value = {"interrupted": True, "attempts_used": ledger.used,
                     "remaining": ledger.remaining}
        else:
            ledger = _open_ledger(config)
            value = {"attempts_used": ledger.used, "remaining": ledger.remaining,
                     "audit_locked": ledger.audit_lock is not None, "kind": ledger.identity["kind"]}
            if args.command == "plan":
                if ledger.audit_lock is not None:
                    value.update(proposal=None, executed=False,
                                 note="Audit identities are locked; no further training is allowed")
                    print(json.dumps(value, indent=2, allow_nan=False))
                    return 0
                if (config.output_dir / "selection.json").exists():
                    value.update(proposal=None, executed=False,
                                 note="Terminal selection is already frozen; use report or predict")
                    print(json.dumps(value, indent=2, allow_nan=False))
                    return 0
                tried = [Recipe(**row["details"]["recipe"]) for row in ledger.attempts()
                         if row["phase"] == "search"]
                runtime = config.output_dir / "runtime.json"
                runtime_data = read_json(runtime) if runtime.exists() else None
                if runtime_data is not None and runtime_data.get("backend") != config.backend:
                    raise ValueError("Configured backend differs from the frozen campaign runtime")
                families = tuple(runtime_data["families"]) if runtime_data else (
                    "lr", "catboost", "lightgbm")
                proposal = Planner(families=families, seed=config.seed, threads=config.threads,
                                   backend=config.backend,
                                   feature_ablation=config.feature_ablation).propose(
                                       observations(ledger), tried)
                value.update(proposal=proposal.to_dict() if proposal else None,
                             executed=False, note="Proposal only; no fit or reservation")
            elif args.command == "report":
                summary = config.output_dir / "summary.json"
                path = render_report(ledger, config.output_dir / "report.html",
                                     summary=read_json(summary) if summary.exists() else None)
                value["report"] = str(path)
        print(json.dumps(value, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

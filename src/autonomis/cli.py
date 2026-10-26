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


def _open_ledger(config):
    frozen = read_json(config.output_dir / "campaign.json")
    return Ledger(config.output_dir / "ledger.sqlite", dataset_hash=config.expected_sha256,
                  split_hash=frozen["split_hash"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Budgeted churn tooling; training disabled by default")
    parser.add_argument("--config", default="configs/campaign.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("profile", "prepare", "status", "plan", "report"):
        commands.add_parser(name)
    training = commands.add_parser("train", help="CPU only; requires explicit host authorization")
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
        else:
            ledger = _open_ledger(config)
            value = {"attempts_used": ledger.used, "remaining": ledger.remaining,
                     "audit_locked": ledger.audit_lock is not None, "kind": ledger.identity["kind"]}
            if args.command == "plan":
                tried = [Recipe(**row["details"]["recipe"]) for row in ledger.attempts()
                         if row["phase"] == "search"]
                runtime = config.output_dir / "runtime.json"
                families = tuple(read_json(runtime)["families"]) if runtime.exists() else (
                    "lr", "catboost", "lightgbm")
                proposal = Planner(families=families, seed=config.seed, threads=config.threads,
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

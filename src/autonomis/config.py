"""Strict campaign configuration with an explicit training enable switch."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from autonomis.contracts import TrainingPermit


@dataclass(frozen=True)
class Config:
    root: Path
    dataset: Path
    output_dir: Path
    history_ledgers: tuple[Path, ...]
    dataset_role: str
    expected_sha256: str
    seed: int
    threads: int
    timeout_seconds: int
    exploration_seconds: int
    terminal_seconds: int
    feature_ablation: bool
    official_submission: bool
    backend: str
    permit: TrainingPermit


def load_config(path):
    path = Path(path).resolve()
    root = path.parent.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"dataset", "dataset_role", "expected_sha256", "output_dir", "history_ledgers", "budget",
                "search_limit", "probe_limit", "seed", "threads", "timeout_seconds",
                "exploration_seconds", "terminal_seconds", "enable_training",
                "backend", "feature_ablation", "official_submission"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("Configuration has missing or unknown keys")
    if (raw["budget"], raw["search_limit"], raw["probe_limit"]) != (15, 11, 2):
        raise ValueError("Budget allocation must be 15 / 11 / 2")
    if raw["backend"] not in {"cpu", "gpu"}:
        raise ValueError("Backend must be cpu or gpu")
    if raw["dataset_role"] not in {"public_reference", "official"}:
        raise ValueError("Unknown dataset role")
    for key in ("enable_training", "feature_ablation", "official_submission"):
        if type(raw[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key in ("seed", "threads", "timeout_seconds", "exploration_seconds", "terminal_seconds"):
        if type(raw[key]) is not int or raw[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if raw["threads"] > 4 or raw["terminal_seconds"] <= raw["exploration_seconds"]:
        raise ValueError("Invalid thread/time allocation")
    digest = raw["expected_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid dataset SHA-256")
    output = (root / raw["output_dir"]).resolve()
    campaign_root = (root / "campaigns").resolve()
    if not output.is_relative_to(campaign_root) or output == campaign_root:
        raise ValueError("Real campaign ledgers must stay under the shared campaigns directory")
    if (not isinstance(raw["history_ledgers"], list)
            or any(not isinstance(item, str) or not item.strip()
                   for item in raw["history_ledgers"])):
        raise ValueError("history_ledgers must be a list of paths")
    history_ledgers = tuple((root / item).resolve() for item in raw["history_ledgers"])
    return Config(root, (root / raw["dataset"]).resolve(), output, history_ledgers, raw["dataset_role"],
                  digest, raw["seed"], raw["threads"], raw["timeout_seconds"],
                  raw["exploration_seconds"], raw["terminal_seconds"],
                  raw["feature_ablation"], raw["official_submission"], raw["backend"],
                  TrainingPermit(raw["enable_training"]))

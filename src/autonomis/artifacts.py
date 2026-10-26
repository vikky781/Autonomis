"""Own-process model bundles; manifests detect corruption, not hostile replacement."""

import os
import platform
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import joblib

from autonomis.contracts import Recipe
from autonomis.data import file_hash
from autonomis.storage import read_json, write_json


@dataclass
class Bundle:
    model: object
    preprocessor: object
    recipe: Recipe
    manifest: dict


def save_bundle(directory, *, model, preprocessor, recipe, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "model.joblib"
    if target.exists() or (directory / "manifest.json").exists():
        raise ValueError("Artifact directory already contains a model")
    descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(descriptor)
    try:
        joblib.dump({"model": model, "preprocessor": preprocessor}, temporary, compress=3)
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    manifest = {"schema_version": 1, "model_hash": file_hash(target),
                "recipe": recipe.to_dict(), "provenance": provenance,
                "python": platform.python_version(),
                "versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")}}
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_bundle(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    target = directory / "model.joblib"
    if file_hash(target) != manifest["model_hash"]:
        raise ValueError("Artifact hash mismatch")
    for name, expected in manifest["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Artifact environment differs: {name}; use its recorded environment")
    # Only load bundles produced by this project from a trusted local location.
    payload = joblib.load(target)
    return Bundle(payload["model"], payload["preprocessor"],
                  Recipe(**manifest["recipe"]), manifest)

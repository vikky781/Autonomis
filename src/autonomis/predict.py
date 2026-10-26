"""Artifact-only inference and append-only, contract-labelled CSV exports."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from autonomis.contracts import fingerprint
from autonomis.data import file_hash
from autonomis.evaluation import positive_scores, validate_scores
from autonomis.features import engineer
from autonomis.storage import write_bytes, write_json


@dataclass(frozen=True)
class OutputContract:
    id_column: str = "customerID"
    score_column: str = "churn_probability"
    official: bool = False

    def __post_init__(self):
        if not self.id_column or not self.score_column or self.id_column == self.score_column:
            raise ValueError("Output column names must be distinct and nonempty")


def predict_bundle(bundle, X):
    frame = engineer(X, bundle.recipe.feature_version)
    encoded = bundle.preprocessor.transform(frame) if bundle.preprocessor is not None else frame
    return positive_scores(bundle.model, encoded)


def export_predictions(path, identities, values, *, model_hash, input_hash, contract):
    ids = pd.Series(list(identities), dtype="string")
    if ids.isna().any() or ids.str.strip().eq("").any() or not ids.is_unique:
        raise ValueError("Prediction identities must be nonempty and unique")
    scores = validate_scores(values, len(ids))
    frame = pd.DataFrame({contract.id_column: ids, contract.score_column: scores})
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    path = Path(path)
    write_bytes(path, payload)
    checked = pd.read_csv(path, dtype={contract.id_column: str})
    if checked[contract.id_column].tolist() != ids.tolist() or list(checked) != list(frame):
        raise ValueError("Saved prediction row/column contract failed")
    validate_scores(checked[contract.score_column], len(ids))
    record = {"model_hash": model_hash, "input_hash": input_hash,
              "contract": asdict(contract), "contract_hash": fingerprint(asdict(contract)),
              "output_hash": file_hash(path),
              "row_count": len(ids), "status": "official" if contract.official else "provisional"}
    write_json(path.with_suffix(path.suffix + ".metadata.json"), record)
    return verify_export(path)


def verify_export(path):
    from autonomis.storage import read_json

    path = Path(path)
    record = read_json(path.with_suffix(path.suffix + ".metadata.json"))
    if (file_hash(path) != record["output_hash"] or
            fingerprint(record["contract"]) != record["contract_hash"]):
        raise ValueError("Prediction output/contract hash mismatch")
    contract = OutputContract(**record["contract"])
    saved = pd.read_csv(path, dtype={contract.id_column: str})
    if list(saved) != [contract.id_column, contract.score_column]:
        raise ValueError("Prediction columns differ from recorded contract")
    if len(saved) != record["row_count"] or not saved[contract.id_column].is_unique:
        raise ValueError("Prediction row identity/count differs from recorded contract")
    validate_scores(saved[contract.score_column], record["row_count"])
    return record

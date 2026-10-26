"""Guarded CPU/GPU adapters. Native fitting never runs in the no-training tests."""

import importlib.util
import subprocess
import time
from importlib.metadata import version

from autonomis.artifacts import load_bundle, save_bundle
from autonomis.contracts import Recipe, TrainingPermit
from autonomis.data import CATEGORICAL, load_table
from autonomis.evaluation import positive_scores, score_predictions
from autonomis.features import engineer, make_preprocessor

GPU_MIN_ROWS = 50_000
GPU_MIN_FREE_MIB = 6_144


def gpu_preflight(row_count):
    """Check the declared GPU gate without constructing or fitting an estimator."""
    if row_count <= GPU_MIN_ROWS:
        raise ValueError("CatBoost GPU requires more than 50,000 labelled rows; use the CPU profile")
    if importlib.util.find_spec("catboost") is None:
        raise RuntimeError("CatBoost is required for the GPU profile; install the gpu extra")
    command = ["nvidia-smi", "--query-gpu=index,name,memory.free,driver_version",
               "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Unable to execute nvidia-smi for GPU preflight") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "nvidia-smi failed"
        raise RuntimeError(f"GPU preflight failed: {detail}")
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            devices.append(fields)
    device = next((fields for fields in devices if fields[0] == "0"), None)
    if device is None:
        raise RuntimeError("GPU preflight could not identify device 0")
    try:
        free_mib = int(device[2])
    except ValueError as error:
        raise RuntimeError("GPU preflight returned invalid free VRAM") from error
    if free_mib < GPU_MIN_FREE_MIB:
        raise RuntimeError("CatBoost GPU requires at least 6144 MiB free VRAM on device 0")
    return {"device_index": 0, "device_name": device[1], "free_vram_mib": free_mib,
            "minimum_free_vram_mib": GPU_MIN_FREE_MIB, "driver_version": device[3],
            "catboost_version": version("catboost"), "row_count": row_count,
            "training_performed": False}


def build_estimator(recipe):
    params = dict(recipe.params)
    if recipe.family == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params, random_state=recipe.seed)
    if recipe.family == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params, task_type=recipe.backend.upper(),
                                  thread_count=recipe.threads, random_seed=recipe.seed,
                                  allow_writing_files=False)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params, random_state=recipe.seed, n_jobs=recipe.threads, metric="auc")


def build_cpu_estimator(recipe):
    if recipe.backend != "cpu":
        raise ValueError("CPU constructor received a GPU recipe")
    return build_estimator(recipe)


def fit_job(job):
    TrainingPermit(**job.get("permit", {})).check()
    from threadpoolctl import threadpool_limits

    recipe = Recipe(**job["recipe"])
    data = load_table(job["dataset_path"], expected_sha256=job["dataset_hash"])
    lookup = {value: index for index, value in enumerate(data.ids)}
    train = [lookup[value] for value in job["train_ids"]]
    valid = [lookup[value] for value in job.get("valid_ids", [])]
    if not train or set(train) & set(valid):
        raise ValueError("Training/validation membership is invalid")
    frame = engineer(data.X, recipe.feature_version)
    X_train, y_train = frame.iloc[train], data.y.iloc[train]
    X_valid = frame.iloc[valid] if valid else None
    processor = None
    if recipe.family != "catboost":
        processor = make_preprocessor(X_train, scale_numeric=recipe.family == "lr")
        X_train = processor.fit_transform(X_train)
        X_valid = processor.transform(X_valid) if valid else None
    model_recipe = recipe
    fixed_count = job.get("fixed_tree_count")
    if fixed_count is not None and recipe.family != "lr":
        key = "iterations" if recipe.family == "catboost" else "n_estimators"
        model_recipe = recipe.changed(**{key: int(fixed_count)})
    estimator = build_estimator(model_recipe)
    search = job["phase"] == "search"
    if search and not valid:
        raise ValueError("Search requires validation rows")
    start = time.monotonic()
    with threadpool_limits(limits=recipe.threads):
        if recipe.family == "catboost":
            options = {"cat_features": list(CATEGORICAL), "verbose": False,
                       "use_best_model": search}
            if search:
                options.update(eval_set=(X_valid, data.y.iloc[valid]), early_stopping_rounds=100)
            estimator.fit(X_train, y_train, **options)
            tree_count = int(estimator.tree_count_)
        elif recipe.family == "lightgbm":
            options = {}
            if search:
                from lightgbm import early_stopping
                options = {"eval_set": [(X_valid, data.y.iloc[valid])], "eval_metric": "auc",
                           "callbacks": [early_stopping(100, first_metric_only=True, verbose=False)]}
            estimator.fit(X_train, y_train, **options)
            tree_count = int(estimator.best_iteration_ or model_recipe.params["n_estimators"])
        else:
            estimator.fit(X_train, y_train)
            tree_count = None
        fit_seconds = time.monotonic() - start
        metrics = {}
        predictions = positive_scores(estimator, X_valid).tolist() if valid else []
        if job["phase"] in {"search", "probe"}:
            metrics = score_predictions(data.y.iloc[valid], predictions)
            metrics["train_roc_auc"] = score_predictions(
                y_train, positive_scores(estimator, X_train))["roc_auc"]
    provenance = {"dataset_hash": data.sha256, "train_ids": job["train_ids"],
                  "phase": job["phase"], "attempt_id": job["attempt_id"],
                  "resolved_parameters": estimator.get_params()}
    manifest = save_bundle(job["artifact_dir"], model=estimator, preprocessor=processor,
                           recipe=recipe, provenance=provenance)
    reloaded = load_bundle(job["artifact_dir"])
    from autonomis.predict import predict_bundle
    verification = predict_bundle(reloaded, data.X.iloc[train[:8]])
    return {"metrics": metrics, "fit_seconds": fit_seconds, "tree_count": tree_count,
            "artifact": job["artifact_dir"], "model_hash": manifest["model_hash"],
            "validation_predictions": predictions, "validation_ids": job.get("valid_ids", []),
            "verification_rows": len(verification), "training_fraction": len(train) / len(data.ids)}


fit_cpu_job = fit_job

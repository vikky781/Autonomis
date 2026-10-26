"""CPU-only adapter definitions. Real fitting is guarded and is not run during development."""

import time

from autonomis.artifacts import load_bundle, save_bundle
from autonomis.contracts import Recipe, TrainingPermit
from autonomis.data import CATEGORICAL, load_table
from autonomis.evaluation import positive_scores, score_predictions
from autonomis.features import engineer, make_preprocessor


def build_cpu_estimator(recipe):
    params = dict(recipe.params)
    if recipe.family == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params, random_state=recipe.seed)
    if recipe.family == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params, task_type="CPU", thread_count=recipe.threads,
                                  random_seed=recipe.seed, allow_writing_files=False)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params, random_state=recipe.seed, n_jobs=recipe.threads, metric="auc")


def fit_cpu_job(job):
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
    estimator = build_cpu_estimator(model_recipe)
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

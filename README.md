# Autonomis

Budget-constrained customer churn experimentation for Utkrishta 2026. The controller proposes experiments from prior development results, with a persistent maximum of 15 training attempts.

**Current boundary:** CPU and CatBoost GPU execution paths are implemented. No predictive model has been trained in this workspace. The local dashboard and CLI allow an explicit training action on any machine where the project and its required packages are available. Native-model round-trips and GPU execution still need verification. Test fixtures do not represent measured model performance.

## Safe local setup

Python 3.12 is the tested interpreter series. Use a new virtual environment. The development lock has been verified for no-training tests on Windows and Ubuntu; native tree-model/GPU compatibility still needs verification on the training laptop.

```bash
python -m venv .venv
# Ubuntu:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install -e .
python -m pytest
python -m ruff check src tests
```

The test suite forbids real Logistic Regression fitting. CatBoost and LightGBM packages are not needed for no-training development tests. Static prediction fixtures and fake workers test model boundaries without learning parameters.

## Commands that do not train

Run from the repository root:

```bash
python -m autonomis profile
python -m autonomis prepare
python -m autonomis preflight
python -m autonomis status
python -m autonomis plan
python -m autonomis report
python -m autonomis serve
# after verifying a worker has stopped:
python -m autonomis reconcile --confirm-no-active-worker
```

- `profile` verifies the raw CSV fingerprint, validates the schema, and reports data counts only.
- `prepare` creates immutable customer partitions, campaign metadata and an empty SQLite ledger. It fits no preprocessing or predictive estimator and consumes zero attempts.
- `preflight` checks the configured backend without fitting or reserving an attempt. GPU mode checks dataset size, CatBoost availability, NVIDIA driver/device visibility and free VRAM.
- `status` reads the consumed/remaining budget and audit-lock state.
- `plan` prints the next proposed recipe and rationale without reserving or running it.
- `report` renders a local read-only HTML history. With no fits it explicitly says no trained artifact/score is available. Simulated histories are labelled synthetic.
- `serve` hosts the judge-facing campaign demo on `http://127.0.0.1:8502`. It shows two explicit controls: one manually selected Search fit, or the sequential campaign runner (up to 15 shared attempts). Training is enabled in the checked-in campaign config and is not tied to a particular machine. The sequential runner may stop early when phase gates or runtime deadlines require it. Run `prepare` once first to create the empty campaign ledger.
- `reconcile` marks a reserved/running worker as interrupted only after explicit operator confirmation; the reservation remains consumed.

The browser demo is generated from the campaign ledger and uses Python's standard-library HTTP server. It needs no GPU, JavaScript build, or extra web package. The dashboard shows the current phase and why the next step follows, while retaining detailed run cards and a clear separation between Search, Audit-proxy, final-fit, and hidden-test evidence.

Generated state is under `campaigns/reference/` and ignored by Git. A shared budget index under `campaigns/.global-budget.sqlite` counts real attempts across campaign ledgers, including archived campaign folders. Optional `history_ledgers` can import external historical runs; configured sources must remain available or ledger setup fails closed. Never delete or replace a ledger to obtain more fits.

The local generated run records and model artifacts were cleared for a fresh workspace start. The local attempt counter now starts at zero and allows at most 15 newly recorded attempts. Clearing local files does not change any attempt allocation maintained by the organizer; confirm the applicable allowance before real training.

## Starting training

`configs/campaign.yaml` has `enable_training: true`. The dashboard offers one manually selected Search fit or the complete sequential campaign. The CLI still requires the explicit `--execute` flag:

```bash
python -m autonomis train --execute
```

Training can run from any machine using this project. The dashboard binds to loopback by default; each action requires a button click and confirmation. Every reservation, including failed or interrupted work, consumes one of the same 15 shared attempts. Adapters are isolated in `training.py`; CatBoost and LightGBM are optional packages declared in the `cpu` and `gpu` extras.

## CatBoost GPU profile

GPU mode is opt-in with `backend: gpu` and changes only CatBoost; Logistic Regression and LightGBM remain on CPU. It uses the frozen recipe profile from the plan: device 0, GPU RAM fraction 0.75, Plain boosting, border count 128 and maximum CTR complexity 2. Backend and resolved parameters are part of recipe/artifact identity. No automatic CPU retry or backend switch occurs after a reserved GPU attempt fails.

Before the first GPU attempt, preflight requires more than 50,000 labelled rows, CatBoost from the supported `gpu` extra, visible NVIDIA device 0 with driver information, and at least 6,144 MiB free VRAM. The 7,043-row IBM reference therefore remains a CPU campaign. Preflight does not train and does not prove native GPU fitting/serialization compatibility. CatBoost GPU retraining is not promised to be bitwise deterministic.

After verifying the accepted dataset and local package/driver compatibility, install the desired native extras in the isolated environment and run the read-only check:

```bash
python -m pip install -e '.[cpu,gpu]'
python -m autonomis preflight
```

Select `backend: gpu` before the campaign starts only when its data gate is met; backend changes are rejected on resume.

## Experiment protocol

| Stage | Maximum counted attempts |
|---|---:|
| Initial LR / CatBoost / LightGBM anchors | 3 |
| Adaptive search and one feature ablation | 8 |
| Two frozen finalists on one shared development split | 2 |
| Audit proxy A trained on development 80% | 1 |
| Submission candidate P trained on all labelled training rows | 1 |
| **Total** | **15** |

Reservations are consumed even when a worker fails, times out or crashes. One SQLite transaction protects capacity; only one reservation/worker may be active. Search/checks cannot consume the two reserved terminal tokens. A duplicate successful recipe/split returns its existing result; there is no automatic failed-fit retry. Native fitting runs in a supervised spawned worker with a configured timeout.

Search uses Train 60% / Search 20%, with Audit 20% held out of search. Train-only learned preprocessing prevents leakage. The finite planner mutates one knob at a time, prioritizes capacity/regularization from feedback, switches families on stagnation, and records every actual decision. The fixed V1 ablation adds log-tenure, an active-add-on count, and historical monthly charge with safe zero-tenure handling.

The winner and tree count are frozen before terminal fitting. A is fitted on development 80%; its Audit predictions are saved before P uses all labels. Submission selection is technical-only: P, otherwise A, otherwise verified cached search artifact F. Audit proxy is A, otherwise F. Both choices lock before Audit metrics are revealed. **The full-data P model has no internal held-out score; Audit metrics describe A/F only.**

Interrupted workers require explicit reconciliation after verifying the recorded process is stopped. Do not blindly relaunch a second campaign or retry an unknown active worker. Keep failed attempts visible. Replaying reports/inference does not fit or spend tokens.

## Artifact-only inference

After a verified artifact is produced on the training machine, inference uses the saved model/preprocessing and never fits:

```bash
python -m autonomis predict --artifact PATH_TO_BUNDLE --input INPUT.csv --output predictions.csv
```

Only load trusted bundles produced by this project: joblib is not a sandbox for untrusted files. The loader checks model-file hashes and recorded core package versions before deserializing. Input must include the 19 reference predictors and unique customerID values. Missing required features fail clearly; legitimate unknown categories use the saved encoder policy. Positive scores are selected by class label, not column-position guessing.

Exports preserve customer order, validate finite probabilities, are read back from disk and have sidecar model/input/contract provenance. Existing exports cannot be overwritten with different content. The current two-column output is **provisional**, not an invented organizer submission format.

## Data and score honesty

The downloaded public IBM file has 7,043 rows, 21 columns, 1,869 positive labels, 11 blank TotalCharges values and no duplicated customer IDs. The byte fingerprint is recorded in [the provenance note](data/reference/ibm_telco/README.md). Git preserves CSV bytes across Windows/Ubuntu checkouts.

The official portal files, hidden-test membership and sample submission are not available. Public-data equivalence is unverified. Do not look up public labels for organizer-held-out customers, silently augment training with them, or call an internal holdout the organizer's hidden test.

Before an official run, replace the provisional output contract with the organizer's exact sample-submission schema and verify row identity, ordering, score semantics, and header policy. `official_submission` records that intent but does not claim compatibility by itself.

ROC-AUC and Average Precision are named separately; trapezoidal PR area is not silently substituted for AP. Search/probe results are selection estimates. Hidden-test score and the PDF's undefined Efficiency Factor remain unavailable, never zero or invented. No performance improvement or winning score has been established.

## Architecture and plan

- [Architecture and module contracts](docs/architecture.md)
- [Current implementation plan](docs/superpowers/plans/2026-10-26-autonomous-ai-data-scientist-v2.md)
- [Execution status and explicit deviations](docs/implementation-status.md)

The repository is developed directly on main with one-file commits at the user's request. CI runs lint and no-training tests. No slides are generated.

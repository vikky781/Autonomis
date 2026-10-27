# Autonomis

Budget-constrained customer churn experimentation for Utkrishta 2026. The controller proposes experiments from prior development results, with a persistent maximum of 15 training attempts.

**Current boundary:** guarded CPU and CatBoost GPU execution paths are implemented. No predictive model has been trained on the development PC. Real fitting, native-model round-trips and GPU execution have not been verified here. Tests use non-learning fixtures and mocked GPU inspection; their scores are not measured model performance.

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
# after verifying a worker has stopped:
python -m autonomis reconcile --confirm-no-active-worker
```

- `profile` verifies the raw CSV fingerprint, validates the schema, and reports data counts only.
- `prepare` creates immutable customer partitions, campaign metadata and an empty SQLite ledger. It fits no preprocessing or predictive estimator and consumes zero attempts.
- `preflight` checks the configured backend without fitting or reserving an attempt. GPU mode checks dataset size, CatBoost availability, NVIDIA driver/device visibility and free VRAM.
- `status` reads the consumed/remaining budget and audit-lock state.
- `plan` prints the next proposed recipe and rationale without reserving or running it.
- `report` renders a local read-only HTML history. With no fits it explicitly says no trained artifact/score is available. Simulated histories are labelled synthetic.
- `reconcile` marks a reserved/running worker as interrupted only after explicit operator confirmation; the reservation remains consumed.

Generated state is under `campaigns/reference/` and ignored by Git. A shared budget index under `campaigns/.global-budget.sqlite` counts real attempts across campaign ledgers, including archived campaign folders. The configured `history_ledgers` also imports the previously recorded parent-project runs. If either history source is missing, conflicting, or inaccessible, real ledger setup fails closed. Never delete or replace a ledger to obtain more fits; preserve campaign ledgers and the historical sources together.

The configured history contains 26 prior recorded fits (15 active and 11 archived in the parent project). Until the organizer confirms an exemption or grants a new allocation, the shared counter treats all 26 as consumed and refuses additional real attempts. The 15-attempt cap applies across campaigns; a new ledger does not create a new allowance.

## Training is disabled here

`configs/campaign.yaml` has `enable_training: false` and `training_host: null`. The training gateway checks both explicit permission and an exact host match before reservation, process startup or estimator construction. The CLI additionally requires `--execute`. A default-config invocation is rejected and consumes nothing:

```bash
python -m autonomis train --execute
```

**Do not enable training or run a baseline on the development PC.** The user's explicit instruction is to defer all real fitting to the friend's laptop. No training command has been used successfully in development.

Adapters are isolated in `training.py`, selected only after permission checks, and optional native dependencies are declared in the `cpu` and `gpu` extras. They are not installed by the safe setup above. Model-library installation and real fitting verification belong on the training laptop.

## CatBoost GPU profile

GPU mode is opt-in with `backend: gpu` and changes only CatBoost; Logistic Regression and LightGBM remain on CPU. It uses the frozen recipe profile from the plan: device 0, GPU RAM fraction 0.75, Plain boosting, border count 128 and maximum CTR complexity 2. Backend and resolved parameters are part of recipe/artifact identity. No automatic CPU retry or backend switch occurs after a reserved GPU attempt fails.

Before the first GPU attempt, preflight requires more than 50,000 labelled rows, CatBoost from the supported `gpu` extra, visible NVIDIA device 0 with driver information, and at least 6,144 MiB free VRAM. The 7,043-row IBM reference therefore remains a CPU campaign. Preflight does not train and does not prove native GPU fitting/serialization compatibility. Any native smoke fit on campaign data is a counted experiment and must run only on the authorized laptop. CatBoost GPU retraining is not promised to be bitwise deterministic.

On the authorized laptop, after verifying the accepted dataset and local package/driver compatibility, install the native extras in its isolated environment and run the read-only check:

```bash
python -m pip install -e '.[cpu,gpu]'
python -m autonomis preflight
```

Keep `enable_training: false` during setup/preflight. Select `backend: gpu` before the campaign starts only when its data gate is met; backend changes are rejected on resume.

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

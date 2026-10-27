# Autonomis

**A budget-aware experiment controller for customer churn prediction.**

Autonomis is our Challenge 1 project for the Utkrishta 2026 AI/ML Hackathon. It turns model experimentation into an auditable campaign: each fit is proposed, reserved against a shared limit, evaluated on a named split, and recorded with its recipe and artifacts. The goal is to make each experiment informative while keeping the full campaign within the 15-fit limit.

**Presentation:** [Autonomis — Utkrishta 2026 (PowerPoint)](slides/output/Autonomis_Utkrishta_2026.pptx). The deck reports local reference-data evidence and marks organizer results as unavailable.

## At a glance

| | Current project evidence |
|---|---|
| Challenge | Challenge 1 — Autonomous AI Data Scientist |
| Dataset currently configured | Public IBM Telco Customer Churn reference data |
| Campaign allocation | 15 attempts shared across rounds |
| Recorded real fits | 15 successful CPU fits across two rounds (9 + 6) |
| Current round | `reference-round-02`, four Search fits, Audit and final fit; Audit locked |
| Remaining allocation | 0 of 15 attempts |
| Best recorded Search ROC-AUC | 0.869660 (Search split; not hidden-test performance) |
| Current Audit proxy ROC-AUC / Average Precision | 0.853652 / 0.672107 |
| Official organizer data match | Not verified |
| Official submission schema / hidden-test score | Not available |

The original round is preserved in `campaigns/reference/`; the active-round pointer is local, ignored state in `campaigns/current.json`. The global ledger counts attempts across rounds. Starting another round preserves history and **does not reset** the allocation. The current round's six fits and the global 15/15 count are separate counters. All attempts in this allocation are now used.

## What it does

- Validates the input dataset and records its SHA-256 fingerprint.
- Creates deterministic, immutable row splits for Train, Search, and Audit.
- Applies training-only preprocessing to avoid leakage.
- Proposes a finite sequence of experiments from previous results and records the reason for each proposal.
- Enforces one durable 15-attempt budget across campaign rounds. Failed and interrupted reservations count.
- Runs either one manually selected Search experiment or a sequential campaign from the local dashboard.
- Stores model bundles, run metadata, metrics, and prediction-export provenance.
- Presents the campaign as a judge-facing local dashboard and a read-only report.

The implemented model families are Logistic Regression, CatBoost, and LightGBM. All 15 saved fits in the current evidence are Logistic Regression on CPU; the other model families are available in the code but are not represented in those results. CatBoost GPU support has preflight checks, but a native GPU fit and serialization round-trip have not been verified.

## Run the local demo

The dashboard uses Python's standard-library HTTP server and runs on a CPU. No GPU or JavaScript build is needed to view the interface.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install -e .
python -m autonomis serve
```

Open [http://127.0.0.1:8502](http://127.0.0.1:8502). The server binds to loopback by default. Training actions require an explicit dashboard action and confirmation. Keep the server running while a worker is active.

### Start an experiment

The checked-in configuration enables training. The dashboard presents two actions:

1. **Train one Search attempt** runs one selected model family.
2. **Run full campaign** runs the sequential campaign, subject to phase gates, remaining budget, and deadlines.

The CLI requires an explicit execution flag:

```bash
python -m autonomis status      # read-only budget and lock status
python -m autonomis plan        # read-only next proposal
python -m autonomis preflight   # read-only backend readiness
python -m autonomis train --execute
```

When at least three shared attempts remain, use **Archive this round and reset Audit** in the dashboard to preserve a round and create a fresh split. A brand-new empty round explains that its Audit is already fresh. Starting a new round resets only the round-local counter; the shared allocation total never resets. In the completed run documented here, the last two attempts went to the current round's Audit and final fit; the 15-attempt allocation is now exhausted.

### Read-only commands

```bash
python -m autonomis profile
python -m autonomis prepare
python -m autonomis preflight
python -m autonomis status
python -m autonomis plan
python -m autonomis report
```

`prepare` creates campaign metadata and splits but does not fit an estimator or consume an attempt. `report` reads saved evidence and does not train. If a worker is interrupted, first verify that it has stopped; only then reconcile its reservation:

```bash
python -m autonomis reconcile --confirm-no-active-worker
```

## Experiment strategy

Each round separates exploratory selection from terminal evaluation:

| Phase | Purpose | Per-round cap |
|---|---|---:|
| Search | Explore recipes using Train / Search partitions while keeping Audit held out | 11 |
| Probe | Compare frozen finalists on a shared additional development split | 2 |
| Audit | Evaluate the frozen audit proxy once | 1 |
| Final | Fit the selected submission candidate on all labelled development rows | 1 |
| **Per-round maximum** |  | **15** |

The global allocation enforces 15 attempts total across rounds. The finite planner uses prior results, changes one knob at a time, can switch model families on stagnation, and records its rationale. Duplicate successful recipe/split pairs return their existing result instead of triggering another fit. Reservations are irreversible, including failures and interruptions.

The Search partition is used for selection. Audit metrics belong to a separately identified proxy model trained on development data, not the full-data final model. The full-data fit has no internal held-out score by design. Search and Audit values are not hidden-test results.

## Saved results

The following values come from the saved reference-round ledger and its report:

| Evidence | Result | Interpretation |
|---|---:|---|
| Best Search ROC-AUC | 0.869660 | Best of five successful Search fits |
| Best Search Average Precision | 0.695799 | Same Search result; not a hidden-test score |
| Current round best Search ROC-AUC / Average Precision | 0.859300 / 0.702953 | Best of four Search fits on a new split |
| Round 1 Audit proxy ROC-AUC / Average Precision | 0.851950 / 0.668741 | Applies to the locked round 1 proxy |
| Current round Audit proxy ROC-AUC / Average Precision | 0.853652 / 0.672107 | Applies to the locked `reference-round-02` proxy |
| Full-data final model | Saved | No held-out metric is recorded for this fit |
| Hidden-test metric / Efficiency Factor | Unavailable | Organizer evaluation data and a verified formula input are not present |

The official challenge defines the primary metric as hidden-test ROC-AUC and the final score as ROC-AUC multiplied by an Efficiency Factor. This repository does not define that factor or contain an organizer evaluation result, so neither is estimated here.

## Data and submission status

The configured dataset is the public IBM Telco Customer Churn reference file with 7,043 rows and 21 columns. Its recorded SHA-256 is:

```text
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

This fingerprint shows which public file the local campaign used; it does not prove equivalence to the dataset supplied through the organizer portal. Do not obtain or use labels for organizer-held-out customers.

The prediction export currently uses a **provisional** two-column contract (`customerID`, `churn_probability`). The organizer's sample submission and exact output schema have not been verified. Treat the saved model as a fitted candidate, not a completed or validated hackathon submission. Before submission, confirm the official data, row identities and order, score semantics, required header, and output format.

## Setup for development

Python 3.12 is the verified interpreter series. The locked development dependencies support linting and no-training tests; native tree-model and GPU compatibility should be checked separately on the intended machine.

```bash
python -m pip install -r requirements-dev.lock
python -m pip install -e .
python -m pytest
python -m ruff check src tests
```

The no-training test suite uses fake workers and prediction fixtures. It does not establish model performance. CatBoost and LightGBM are optional CPU extras. GPU mode is opt-in and is limited to CatBoost; preflight checks data size, package, driver/device visibility, and free VRAM. A preflight pass does not itself verify a native GPU fit.

## Repository map

```text
src/autonomis/             campaign, planner, ledger, training, prediction, reporting
configs/campaign.yaml      checked-in campaign configuration
data/reference/            public IBM reference dataset and provenance note
docs/architecture.md       execution flow and component boundaries
docs/implementation-status.md  verified state and outstanding gaps
campaigns/                 ignored local ledgers, splits, and saved artifacts
slides/output/             generated hackathon presentation
```

See [architecture](docs/architecture.md) for module contracts and [implementation status](docs/implementation-status.md) for verified behavior and open work.

## Known gaps before judging

1. Verify that the organizer portal dataset matches the public reference file and record the evidence.
2. Confirm the official sample submission schema and export a compatible file.
3. Verify organizer data equivalence and the official output contract before preparing a submission.
4. Report the official hidden-test ROC-AUC and Efficiency Factor only when the organizer supplies them.
5. Verify native CatBoost/LightGBM serialization and GPU behavior before presenting those paths as operational results.
6. Add the verified team-member roster to the cover slide; it was not present in this repository.

The project is designed for transparent demonstration: it distinguishes implementation capability, completed local fits, and organizer-verified outcomes.

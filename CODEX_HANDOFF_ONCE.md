# Autonomis — One-Time Codex Handoff

> Read this file fully before taking any action. This is a temporary context handoff for the Codex session on the training laptop. The user will provide the next task after you have absorbed it. Do not begin training or change the project direction on your own.

## Project and challenge

This repository implements the **Autonomous AI Data Scientist** hackathon problem: an auditable, budget-constrained experiment controller for Telco Customer Churn prediction. The user is building this with an 18-hour hackathon window, no paid API budget, and the intent to use a teammate's Ubuntu laptop for model training.

The challenge document is `Utkrishta 2026 Participant Problem Statements.pdf` in the repository root. Its relevant requirements:

- At most 15 model-training experiments; a fit-and-validation cycle counts as one experiment.
- The system should use feedback from completed experiments to choose later experiments.
- Primary metric is ROC-AUC; Average Precision (PR-AUC) is secondary.
- The official data and submission schema come from the participant portal. They have not yet been obtained or verified.
- The scoring formula mentions an Efficiency Factor but does not define it. Do not invent its formula or claim a score improvement.
- Deliverables eventually include runnable code, reproducibility documentation, predictions matching organizer schema, evidence of actual experiment history, a presentation, and a demo. The user has asked for no presentation work at this point.

The exact meaning of “15 runs” in a presentation phrase versus “maximum 15” is ambiguous. The implementation treats 15 as a hard cap and never fabricates runs.

## User's non-negotiable instructions

- **Never train any predictive model on the development computer.** It has no GPU and is for development only.
- Train only on the teammate's Ubuntu laptop: NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM, 24 GB RAM, Ryzen 7 7335HS (as reported by the user). Teammate has prior PyTorch experience.
- **GPU integration is intentionally deferred.** Do not probe, install, implement, or execute GPU/CUDA integration until the user explicitly asks. If the next task reaches that boundary, stop and explain what remains.
- The user authorized work directly on `main`, pushes to `origin/main`, and a separate commit for every changed file. Continue that convention unless the user changes it.
- Be direct and precise. Do not bluff. Distinguish verified results from assumptions.
- No API spend is planned. If genuinely needed, the user mentioned OpenRouter or Groq for a few calls; the current design is deterministic and does not depend on an LLM/API.
- The user may first provide the official data or organizer clarifications; do not assume the public reference equals the official dataset.

## Data status and caveats

A public IBM Telco Customer Churn CSV is stored at `data/reference/ibm_telco/Telco-Customer-Churn.csv`, with provenance in its adjacent `README.md`. Source URL:

`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

Verified public-reference facts:

- 7,043 rows, 21 columns: customer identifier, 19 predictors, churn target.
- Target: 1,869 churn (26.54%); 5,174 retained (73.46%).
- 16 categorical predictors (including `SeniorCitizen`) and 3 numeric (`tenure`, `MonthlyCharges`, `TotalCharges`).
- 11 blank `TotalCharges` values, all at zero tenure; no duplicate customer IDs.
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
- Public descriptive associations were inspected for checkpoint discussion. They are not causal effects, model performance, or independent validation. Because full-table target summaries were inspected, do not describe this public reference as an untouched audit set.

**Critical limitation:** organizers' portal dataset, train/test split, and sample submission are still unavailable/unverified. Do not use public labels to claim official hidden-test performance or infer that IBM's CSV is the portal data. When official files arrive, inspect their schema and provenance before adapting anything.

## Current repository state at handoff

The source machine last reported a clean `main` checkout at commit `afe8fedafb29a1f97ea666b32a5b809ed539bdb0` (`ci: verify against the tested development dependency lock`), pushed to `origin/main`. Verify the actual checkout before acting; this handoff itself will add a newer commit.

Implementation is in `src/autonomis/`, configuration in `configs/`, plans in `docs/superpowers/plans/`, architecture in `docs/architecture.md`, and execution boundaries in `docs/implementation-status.md`.

Implemented, CPU-side orchestration and safety features include:

- strict tabular input and target/ID handling;
- immutable split manifests and train-only preprocessing;
- a SQLite experiment ledger with irreversible reservations and attempt caps;
- deterministic, feedback-based experiment proposal logic;
- isolated worker execution with host permission checks and timeout/failure accounting;
- model/artifact and prediction export contracts with integrity checks;
- report/status/plan/profile/predict CLI paths;
- simulation-safe tests and documentation.

The default development host is explicitly not authorized for training. Native CatBoost/LightGBM fit/serialization behavior, actual laptop execution, GPU behavior, model performance, organizer schema, and hidden-test score remain unverified. No predictive model has been trained in this repository. A prior local check reported 58 tests passing and Ruff clean; remote CI was not verified. Re-run/inspect the current state rather than assuming it is unchanged.

## Experiment protocol encoded in the plan

Read `docs/superpowers/plans/2026-10-26-autonomous-ai-data-scientist-v2.md` and `docs/implementation-status.md` for exact details. Important design decisions:

- Deterministic local planner; no paid LLM/API dependency.
- Hard cap of 15 attempts, with failed/interrupted reserved attempts consuming budget.
- Plan allocation: 11 search (3 model-family anchors + 8 adaptive search), up to 2 paired probes, then terminal audit/final artifacts, with 2 reserved terminal slots as detailed in the plan. Do not change allocations casually.
- Planned model families: regularized logistic regression, CatBoost, LightGBM, all controlled by one shared search/terminal protocol.
- Strict separation of development Search, sealed Audit, and final full-data fit. Audit metrics are revealed only after final recipe and identity are frozen/committed. No post-audit tuning.
- The described split seeds/sizes and model settings are in the versioned plan. Follow the plan or record an explicit, justified change before execution.
- GPU use is not in the current implementation and is a deliberate stop boundary.

## Checkpoint presentation context (theory only so far)

The user has already received a short checkpoint-1 talk track and theoretical dataset analysis. Do not assert that any model was trained or any ROC-AUC achieved. Safe status statement: engineering/orchestration is implemented and locally tested; actual predictive experiments have not started because official data is not available and the execution laptop boundary is being respected.

## What the receiving Codex should do now

1. Read this handoff completely.
2. Inspect `git status`, current branch/commit, and the repository docs named above; do not assume the source machine's state is identical.
3. After you have captured the context in the current conversation, remove **this handoff file from the working tree** and commit/push that deletion to `main` as a one-file commit, if network/auth permits. Do not rewrite Git history or force-push: deleting the file in a later commit removes it from the current repository tree, but the earlier commit remains in Git history. If user requires history erasure, stop and ask because that is a history rewrite with coordination implications.
4. Then stop and wait for the user's next prompt. Do not train, implement GPU integration, download/replace official data, create a PPT, or perform unrelated work without an explicit next instruction.

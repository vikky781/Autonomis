# Autonomous AI Data Scientist Implementation Plan

> Superseded by [revision 2](2026-10-26-autonomous-ai-data-scientist-v2.md), which incorporates the Ubuntu RTX 4060 laptop, downloaded IBM reference data, and a revised 11-search + 2-probe + audit-model + full-data-final allocation. Retained as the original analysis, not the current execution plan.

> Planning only. No implementation, model training, dependency installation, notebook, or presentation has been created. All proposed parameter values and engineering thresholds below are our decisions, not organizer requirements. Implementation requires a subsequent user instruction.

**Goal:** Build a local, reproducible churn prediction system that chooses experiments using previous results, stays within a maximum of 15 training attempts under our conservative accounting, and produces an auditable final model and organizer-compatible predictions.

**Architecture:** A deterministic sequential experiment controller proposes a bounded candidate, a persistent budget ledger authorizes one training attempt, and an evaluator returns development-set feedback. After search and optional paired robustness checks, freeze one recipe, train it on development data, audit the exact persisted artifact once, and submit it unchanged.

**Proposed stack:** Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, SQLite, pytest, and Matplotlib. Local CPU execution; no paid API, remote agent framework, database service, or GPU dependency. Resolve compatible stable versions and lock the actual environment at implementation time; this planning exercise has not tested package installation.

**Source specification:** Utkrishta 2026 Participant Problem Statements.pdf, repository root, pages 1 and 3. All three pages were rendered and visually inspected. Page 2 concerns the other track and does not supply our evaluation metrics.

**Current repository:** Only the problem-statement PDF and Git metadata were present when inspected. There is no existing application, official dataset, sample submission, or empirical result to build on.

**Resources:** User reports 18 hours remaining and no paid API budget. Team size and hardware are unspecified. Schedule assumes one primary implementer and ordinary CPU access; runtimes are not yet measured.

## 1. What the brief actually requires

| Requirement | Exact practical consequence | Source |
|---|---|---|
| Predict customer churn on the portal-provided Telco Customer Churn single table | Obtain the official files; do not assume a public copy has the same schema or rows | PDF p. 1 |
| Maximum 15 model-training experiments | Budget all estimator training; do not hide CV, seed repeats, or final refits | PDF p. 1 |
| One fit-and-validation cycle counts as one experiment | The unit is a trained estimator evaluated on held-out data, not an optimization-framework trial or notebook cell | PDF p. 1 |
| Loading, cleaning, profiling, and feature design do not count | These operations may precede training; fitting a predictor to test a feature package still counts | PDF p. 1 |
| Next experiments must use feedback from previous runs | A static 15-configuration list alone does not demonstrate the requested intelligence | PDF p. 1 |
| Primary metric: hidden-test ROC-AUC | Produce positive-class scores; optimize ranking, not classification accuracy | PDF p. 1 |
| Secondary metric: PR-AUC / Average Precision | Record Average Precision explicitly; retain PR curves and clarify the organizer's exact convention | PDF p. 1 |
| Final Score = ROC-AUC multiplied by Efficiency Factor | The factor is undefined; we cannot compute or promise a final score | PDF p. 1 |
| Planner strategy, chronological run history, trajectory, and hidden-test result | Persist evidence during execution; hidden-test results remain unavailable until released | PDF p. 1 |
| Runnable source/notebooks, reproducibility README, standardized predictions, 10-slide deck, working demonstration | Build a small complete pipeline and retain evidence for the later deck; no slides now | PDF p. 3 |

### Organizer ambiguities that affect the plan

1. What is the Efficiency Factor, including behavior for fewer than 15 runs and failed attempts?
2. Is 15 a maximum only, or does the phrase about a table of all 15 executed runs imply an expectation to execute exactly 15?
3. Are each CV fold, seed repetition, failed/interrupted fit, final refit, and early-stopped training attempt counted as assumed here?
4. Is early stopping within one fitted boosting model permitted as one experiment? Does choosing the number of trees internally require any special reporting?
5. What are the official target, positive label, row identifier, submission columns/order, test features, and submission limits?
6. Are there official split rules, timestamps/customer groups, external-data restrictions, environment limits, or later portal instructions?

**Working policy:** Never exceed 15; charge failed or interrupted reserved training attempts; count every CV estimator separately; reserve final training; report actual executed attempts rather than inventing missing rows. Do not assume fewer runs earn more points. If clarification changes the policy, update and freeze it before the first official fit. An incompatible later rule may require organizer guidance; deleting the ledger does not undo consumed experiments.

## 2. Judge the design against all 100 points

The leaderboard formula and the overall judging weights are distinct. Do not manufacture an arithmetic combination between them.

| Judging criterion | Weight | What our build must demonstrate | Acceptance evidence |
|---|---:|---|---|
| Technical Performance | 40% | Strong, defensible churn ranking on the organizer's data | ROC-AUC and AP for every supported successful anchor (three on the default path), search trajectory, paired robustness table when executed, separately labelled final audit, correctly formatted hidden-test predictions; unavailable/failed anchors explicitly recorded |
| Approach & Innovation | 20% | The next experiment changes because of previous observations | Logged incumbent, diagnostics, considered alternatives, chosen mutation and rationale, rejected duplicates, remaining budget, and a replayable decision sequence |
| Efficiency & Constraints | 15% | Compliance is enforced by the system | Durable lifetime fit ledger, refusal of attempt 16, failure/crash accounting, no hidden CV/refit, actual elapsed time per attempt |
| Presentation Quality | 15% | Claims are clear, supported, and defensible | Saved EDA, pipeline description, actual chronological history, curve/table exports, metric definitions, limitations and Q&A notes for the later required deck |
| Code & Demo Quality | 10% | Someone can reproduce and demonstrate the pipeline | Locked dependencies, simple local entry points, artifact load/predict verification, README, deterministic controller replay, inference without training |

No numerical AUC target is justified yet. Acceptance is correct measurement and a complete compliant system; predictive quality must be established by actual experiments and ultimately hidden evaluation.

## 3. Decisions reached through adversarial review

Three subagents independently covered optimization architecture, statistical red-teaming, and judging/compliance, then challenged one another's proposals.

| Dispute | Rejected or qualified position | Resolution |
|---|---|---|
| More algorithm families versus useful feedback | Four or five anchors leave too little budget to improve any one family | Three complementary anchors; focus subsequent search on at most two families |
| TPE or Bayesian optimization versus a heuristic | A large mixed search space is difficult to justify with so few observations; default Optuna TPE uses 10 startup trials | Use an explicit local feedback-driven policy; do not claim a measured advantage over TPE |
| LLM autonomy versus deterministic optimization | API-dependent planning adds cost, latency, and reproducibility work | Core requires zero API calls; LLM narration is deferred and cannot authorize training |
| Reused validation versus an honest final evaluation | Repeatedly selected development scores are not unbiased generalization estimates | Isolate an untouched audit partition; label search and robustness results as selection evidence |
| Audit an 80%-trained model then submit a 100%-trained replacement | The audit metric would describe a different model | Default submits exactly the artifact audited; acknowledge the 20% training-data opportunity cost |
| Four robustness fits for everyone | This can waste budget or manufacture a second contender | Conditional block for two close finalists, subject to time and remaining tokens |
| One failed robustness fit | Comparing unequal numbers of successful splits is misleading | Rank only common successfully completed pairs; disclose the number of pairs |
| Refit when a demo opens | Can exhaust or violate the experiment cap | Load persisted artifacts; demo replay and inference never train |

This is a defensible design choice under the present constraints, not proof that this allocation or model family is globally optimal. In particular, 12 search + 2 robustness + 1 final, or a precommitted full-data final-refit protocol, may outperform it. The data and missing rules do not establish a unique optimum.

## 4. Data contract and leakage controls

Complete this before attempt 1:

- Obtain the official training/test tables, sample submission, and any portal addenda. Keep raw inputs immutable and record content hashes.
- Confirm the target and map churn to positive class 1 explicitly. Verify model score columns by class labels, not by guessing column position.
- Establish stable row keys and retain original test order. An identifier is retained for output/joining and excluded from predictor features.
- Freeze the official output contract: exact column names/order, row count, identifier or row-order rule, positive-class score meaning, required types, and header/index policy. Read every saved submission back from disk and compare it against this contract and the official test-row manifest. If the official contract is unavailable, submission compatibility remains unresolved; a provisional CSV is not an accepted submission.
- Profile schema, missingness, numeric parsing, categorical levels, class counts, duplicated identifiers/records, and possible temporal/group dependence. Never assume 7,043 rows, a particular churn rate, or the familiar public Telco schema.
- Review feature availability at the intended prediction time. Exclude direct label copies and post-outcome information. Columns such as churn reason or customer status are examples to investigate if present, not assertions that these files contain them.
- Treat whitespace-only numeric entries as missing after documenting the conversion. A missing charge is not automatically zero. Keep real zero-tenure customers.
- Do not delete two legitimate customers merely because their feature values match. True repeated records or repeated customer observations require an identity/group policy.
- Split before learned imputation, scaling, category dictionaries, feature selection, target encoding, or resampling. Refit these transforms only on each attempt's training partition. This follows the official [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
- The audit evaluator may use labels to construct a stratified split, but the planner receives no audit labels, feature-target associations, metric feedback, or error analysis before artifact freeze. This is an enforced workflow boundary, not a claim of cryptographic isolation from the developer.

**Feature package V0:** Validated raw predictors, documented type/missing-value handling, no identifier or outcome leakage. Keep meaningful categorical states such as no service separate when the schema distinguishes them.

**Optional V1:** One small, predeclared row-wise feature package only if the official schema supports it: log of one plus nonnegative tenure, a clearly defined count of active services, and average historical charge where tenure is positive. Zero-tenure division becomes missing plus an indicator, never infinity. Feature definitions require business plausibility and are not causal claims. An unchanged-recipe V0/V1 model comparison consumes a normal search fit. Do not let speculative feature invention delay the working pipeline.

## 5. Validation protocol: separate selection from audit

### Default for independent customer rows

| Partition | Fraction of labelled data | Allowed use |
|---|---:|---|
| Train | 60% | Learn model parameters and preprocessing during search |
| Search | 20% | Adaptive configuration selection, diagnostics, and permitted early stopping |
| Audit | 20% | One final evaluation after selecting and persisting the submission artifact |

First hold out 20% Audit using stratification and seed 20261026. Split the remaining development 80% into 75% Train and 25% Search using seed 20261027. Save exact row membership. Integer rounding makes fractions approximate. Estimator seed is 20261026 throughout unless a separately declared experiment changes it; no seed shopping.

If customer groups or prediction timestamps require it, replace random stratification with group-disjoint or chronological partitions before attempt 1. Apply the same dependence-aware logic to robustness checks. Exact group/time boundaries depend on data that are currently absent. Stop before fitting if valid partitions with both target classes cannot be formed; do not manufacture meaningful AUC from a one-class partition.

**Search scores are reused selection estimates.** Boosting early stopping may use Search if the organizers allow it; this makes Search even more clearly part of optimization. Never use Audit as an early-stopping evaluation set. CatBoost explicitly documents best-model selection using a supplied validation set and allows AUC selection while optimizing Logloss: [CatBoost tuning guide](https://catboost.ai/docs/en/concepts/parameter-tuning).

### Paired robustness checks

Before fitting any model, save two additional 75/25 partitions of development data, seeds 20261028 and 20261029. After search, freeze up to two finalist recipes, including feature version, preprocessing, hyperparameters, seed, and tree count. Fit both finalists on the same training rows and evaluate on the same validation rows for each partition.

Execute A/split 1, B/split 1, A/split 2, B/split 2. These are four budgeted fits, not one CV call. Fixed rounds replace early stopping during these checks. Select by mean ROC-AUC over shared successful pairs, then mean AP, then mean fit time, then lower canonical recipe ID; numerical equality tolerance is 0.00000001. Log each paired delta separately.

These overlapping splits reuse development labels. Call this robustness reranking, not an independent confirmation study. Do not compute a t-test or report an independence-based standard error from two splits. Adaptive validation overfitting is a known model-selection risk, not cured merely by adding CV: [Cawley and Talbot, JMLR](https://www.jmlr.org/papers/v11/cawley10a.html).

### Final fitting and audit

Freeze the winner, fixed iteration count, preprocessing recipe, and a verified cached fallback before final fitting. Train once on the development 80%, learn preprocessing on those rows only, and do not pass Audit to training or early stopping. For a boosting winner, use the best tree count from its original search artifact, converting zero-based best-iteration indices to a tree count where necessary. If early stopping was disabled, use that search artifact's fixed trained tree count. Robustness fits and final fitting use the same frozen count, with early-stopping callbacks disabled and CatBoost best-model selection disabled.

Finalization success means training, persistence, reload, valid Audit predictions, and valid official-test predictions/export all succeed. Read the test export back and check the frozen output contract. Record the selected model, Audit predictions, and test-export hashes before revealing Audit labels; a successful training return alone is insufficient.

If training or any of those technical gates fails, choose the previously frozen search artifact for that winner on technical status alone, before Audit access. Validate that fallback's Audit predictions and official-test export by the same gates without another fit. If the fallback also fails, leave Audit closed and report that no verified submission is ready. Otherwise audit and submit that exact fallback, reporting its real training fraction and original run ID rather than claiming a successful 80%-trained final model.

Before label access, durably commit an Audit-open record bound to the selected model, Audit prediction, and test-export hashes. The ledger then forbids all further training and rejects evaluation of a different artifact or prediction hash. A restart may recompute metrics or regenerate reports from the same frozen predictions; it may not select another model. This is one immutable final evaluation, not a restriction on reading the same stored result twice. Submit the locked artifact unchanged; an Audit disappointment does not permit reselection, calibration, feature changes, or further training.

The default deliberately sacrifices 20% of labelled training rows to audit the actual submitted model. A full-100% refit is an alternative protocol, not an uncounted extra fit or a decision made after seeing Audit.

## 6. Metrics and uncertainty

- Compute ROC-AUC from churn scores/probabilities. Hard labels and a tuned 0.5 threshold discard ranking information; threshold optimization is outside the primary objective. [Official ROC-AUC definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
- Compute Average Precision using its named definition. Trapezoidal PR-curve area is not identical to AP; label any separately computed PR area precisely and follow the organizer's eventual convention. [Official AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- Search-stage attempts log Train/Search ROC-AUC and Search AP; only those attempts contribute to the Search trajectory. Robustness metrics identify their own held-out split IDs. Final development fitting records no held-out Search metric because Search rows now belong to training; the final artifact's evaluation metric is Audit only. Label all training metrics as in-sample. Also log tree count, convergence/warnings, runtime, and feature count. Train-to-Search gaps are diagnostics, not proof of a particular failure cause.
- Optionally compute a 95% percentile interval from 2,000 stratified bootstrap resamples of the frozen Audit predictions, seed 20261030. Resample label-prediction row pairs together within each class, preserving class counts, and use the 2.5th/97.5th percentiles. This scores existing predictions and consumes no fit. It describes sampling uncertainty conditional on the fitted artifact, class counts, and assumptions, not training variability, selection uncertainty, or future distribution shift. Paired resampling and percentile intervals follow the general mechanics in the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html); class-stratified resampling is our specified procedure, not a claim that SciPy enables stratification automatically.
- Use customer-cluster or temporal-block resampling where dependence requires it; otherwise omit a misleading interval and disclose the limitation. Report actual class counts and interval width without inventing a universal sample-size guarantee.
- Do not publish a hidden-test score until the organizers supply it. Use unavailable, not zero or an estimated future result.

## 7. Initial models and narrow candidate catalog

These are starting hypotheses, not expected winners. CPU threads are limited to at most four available cores; attempts run sequentially. Record actual package versions, resolved defaults, parameters, and hardware.

| Anchor | Initial proposed settings | Preprocessing and purpose |
|---|---|---|
| Logistic Regression | L2 penalty, C 1, solver lbfgs, maximum iterations 3,000, tolerance 0.0001, no class weights | Train-only numeric median imputation with missing indicators and scaling; categorical missing sentinel and one-hot encoding that tolerates unseen levels. Cheap, interpretable linear baseline and early persisted fallback |
| CatBoost | Binary Logloss, evaluation AUC, depth 6, learning rate 0.05, L2 leaf regularization 3, maximum 1,000 trees, seed fixed, CPU | Native categorical features, explicit categorical missing sentinel; documented numeric missing handling. Nonlinear candidate suited to mixed tables without manual target encoding |
| LightGBM | Binary objective, evaluation AUC, 15 leaves, minimum child samples 50, L2 regularization 1, learning rate 0.05, maximum 1,000 trees, seed fixed, deterministic CPU settings | Train-only numeric imputation and categorical one-hot encoding. A distinct tree-growth candidate using a bounded capacity search |

For permitted boosting early stopping, use patience 100 with Search as the sole monitored selection set. Retain best-tree artifacts and save learning curves. If early-stopping accounting is not accepted, use fixed tree counts frozen before fit 1 and record that Search still drives configuration selection. Do not silently add extra fits to select an iteration count.

CatBoost's categorical transformation and ordered statistics are documented by its authors; this motivates trying it, not predicting victory: [official categorical processing](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic), [CatBoost paper](https://arxiv.org/abs/1706.09516). LightGBM documents the relationship among leaves, minimum leaf observations, and overfitting: [official parameter tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html).

### Allowed adaptive moves

| Family | Discrete choices | Move restriction |
|---|---|---|
| Logistic Regression | C 0.1, 1, 10 | One adjacent C change from the current family incumbent |
| CatBoost | Depth 4, 6, 8; L2 leaf regularization 3, 10, 30 | One adjacent parameter change at a time |
| LightGBM | Leaves 7, 15, 31; minimum child samples 20, 50, 100; L2 regularization 1, 10, 30 | One adjacent parameter change at a time |
| Boosting horizon | Maximum trees 1,000 or 2,000 | Expand once only if the best tree count reached at least 90% of the cap; otherwise leave the cap fixed |
| Feature package | V0, optionally V1 | At most one V0/V1 ablation in search, copying the current incumbent's other settings |

Learning rate and class weights remain fixed. No default SMOTE, external target encoding, RFECV, calibration CV, stacking, bagging seeds, broad AutoML, or neural network search. A class-weighting experiment can be a later explicit replacement for a catalog move only if decided before the campaign, not an unlogged extra fit.

If a library is unavailable, do not add a fourth family midway. Before fit 1, reduce the planned families to the supported ones and re-freeze the catalog; retain the same global budget. Measure actual fit times on the first counted attempts, not on extra unlogged pilot fits.

## 8. Deterministic feedback-driven controller

### Budget allocation

| Stage | Maximum fits | Decision |
|---|---:|---|
| Anchors | 3 | One each: Logistic Regression, CatBoost, LightGBM |
| Adaptive search | 7 | Choose each next move from prior logs; anchors plus adaptive search total at most 10 attempts |
| Paired robustness | 4 | Conditional comparison of two frozen close finalists |
| Final development fit | 1 | Reserved until finalization; uses frozen winner recipe |
| Total successful full path | 15 | Failed/reserved attempts consume these same slots; never add replacements beyond the cap |

This table defines a maximum full path, not a promise of exactly 15 successful models. If fewer search attempts are used, later run IDs follow actual execution order; do not fabricate runs 11 through 15.

### Exact proposal policy

1. Rank valid anchors by Search ROC-AUC, then AP, then lower fit time, then canonical recipe ID. Use numerical equality tolerance 0.00000001. Keep the best two valid families for search. If only one remains valid, narrow to that family; if none is valid, stop and report failure before consuming more fits. Record every unsupported/failed anchor without inventing a score.
2. Give every shortlisted family one local adaptive move in anchor-rank order. Later proposals normally use the family with the highest current Search incumbent score. With one family, continue its untried neighbors without a switching requirement.
3. Enumerate untried one-step neighbors of that family's incumbent. A canonical recipe includes family, every resolved parameter, feature version, training split, and seed. Never retrain an identical recipe/split/seed accidentally.
4. Prioritize a boosting-cap expansion if best tree count reached at least 90% of the existing cap. Otherwise, if train minus Search ROC-AUC exceeds 0.05, prefer lower capacity/stronger regularization. Otherwise prefer a modest capacity increase/weaker regularization. These are heuristic diagnostics, not causal inferences.
5. For CatBoost, capacity order is depth then L2. For LightGBM, it is leaves then minimum child samples then L2. For Logistic Regression it is C. Within a direction, choose the adjacent catalog value; skip exhausted/invalid neighbors. Try the opposite direction after the preferred direction is exhausted. Break remaining ties lexicographically by canonical recipe ID. An exhausted family hands priority to the other available family; finalize when every available family is exhausted.
6. After each completed adaptive fit, update the family incumbent if the ranking improves. Count an improvement below 0.002 ROC-AUC relative to that family's incumbent at proposal time as non-material for scheduling, even if it becomes the new best score. With two shortlisted families, two consecutive non-material moves in one family switch priority to the other for its next untried move; if the other is exhausted, continue the available family. With one family, do not invent another or require a switch.
7. If V1 is available and explicitly enabled before fit 1, evaluate it once on the global incumbent after every shortlisted family received its first local move, within the same seven adaptive slots. It changes only the feature package. Disable this ablation first if delivery time is tight.
8. Stop search at 10 reserved search attempts including failed attempts, on exhaustion of every available family's useful catalog, or when every shortlisted family has had at least two adaptive attempts and the latest four valid adaptive results brought no individual global-incumbent improvement of at least 0.002. The frozen exploration deadline also stops search. The 0.05 and 0.002 thresholds are engineering choices, not significance thresholds or claimed optimal constants.
9. At search end, take the top two distinct valid recipes, which may share a family. Run robustness only if their Search ROC-AUC gap is at most 0.01 and at least five tokens remain. Estimate a probe duration as 1.5 times the slowest observed successful finalist search fit and final-fit duration as twice that observed duration: the latter combines 80/60 training-row growth with a 1.5 safety factor. Require four estimated probes to finish before the exploration deadline and those probes plus estimated final fitting and a two-minute margin to finish before the finalization deadline. These are planning estimates, not runtime guarantees. Recheck time and budget before each probe. If admission fails, select the search incumbent or use the already completed common-pair rule, then finalize early.
10. Freeze both recipes before any robustness fit. Rank only common successful split pairs. Two pairs means mean-of-two ranking; one pair means a single paired comparison labelled fragile; zero pairs falls back to the original search ranking. Do not include an unpaired score in the mean. Stop starting robustness work if it would consume the final reserved token.
11. Lock the winner and its original-search artifact as the verified fallback. Use the reserved final fit. Then follow the one-time audit procedure in section 5.

The 0.01 near-tie band is another predeclared engineering choice, not a confidence interval. Freeze exploration deadline at implementation hour 10, finalization deadline at hour 12, and per-attempt timeout at 30 minutes. A fit's effective timeout is the shorter of 30 minutes and the remaining phase window. A supervised training worker records its process identity; a timeout terminates that worker, retains the consumed token, and follows the failure policy. Do not start another fit until the prior worker is confirmed stopped. A finalization timeout selects the pre-frozen fallback without further training. These timeout choices are ours, not organizer runtime rules.

The planner is autonomous because its family, direction, next configuration, and stop decision respond to observations. It is not a learned policy or a claim of optimal sequential experimental design.

### Required decision record

Every proposal records the preceding run IDs, remaining budget, incumbent ID, train/Search gap, best-tree/cap diagnostic, candidate neighbors considered, chosen canonical recipe, rejected duplicates/invalid choices, reason for the move, expected tradeoff, and stop/finalization decision. After the result arrives, record observed deltas and whether the proposed hypothesis was supported. A poor result is evidence to retain, not a row to erase.

### Why not default TPE here?

The official Optuna TPE sampler uses 10 startup trials by default. With a 10-attempt search stage, leaving that unchanged produces no TPE-informed suggestions. Reducing startup trials is possible, but does not establish a better surrogate with three conditional families and very few observations. The local heuristic has fewer moving parts and a clearer demonstration within 18 hours. This is a feasibility judgment, not a benchmark result. [Official TPESampler documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

## 9. Budget ledger and failure semantics

SQLite is the authoritative ledger; CSV/JSON exports are derived evidence. One process owns training. The trainer adapter is the only path permitted to construct and fit a predictive model during an official campaign.

- Perform file/schema/config/partition checks before reserving a token. These checks may not call a predictive estimator's training method.
- Use a transaction to check remaining capacity and insert a unique monotonically numbered reservation. At 15 consumed reservations, reject another attempt before estimator invocation. Keep at least one token for finalization during search and robustness.
- Mark the reservation running before model training; later mark it succeeded, failed, or interrupted. A crash after reservation is conservatively consumed even if it is unclear whether training began. This may count more strictly than the organizers; document it.
- Never delete a consumed reservation or reset the counter after a process restart. Never retry a failed attempt under the same ID. An explicitly approved later retry is a new counted attempt with a new ID and the same global cap. A missing/corrupt ledger or changed input/split hash prevents training rather than initializing a fresh campaign silently.
- An interrupted process is not silently resumed into a fresh fit. Reconcile its state conservatively on restart; use verified completed artifacts when possible. No automatic library-level training retry or full refit is permitted outside the adapter.
- Store dataset/split/config hashes, role, seed, preprocessing version, model parameters, timestamps, status, scores, warning/error text, feature count, best tree count, rationale, and artifact path/hash.
- A duplicate successful recipe/split/seed returns its existing artifact/result. Repetition with a deliberately different seed or split is a new counted fit, not a cache hit.
- An invalid prediction artifact or failed metric evaluation invalidates the result but does not refund training. A convergence warning is recorded; do not silently increase iterations and refit.
- Bootstrap, plots, threshold inspection, loading a trained model, prediction, and decision replay are scoring/read operations. Learned preprocessing that trains an auxiliary predictive model is outside this minimal design.

Count estimator fits inside wrappers rather than public method calls. Calibration CV and stacking can trigger additional base-estimator fits and are deliberately excluded: [calibration documentation](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html), [stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html).

Permutation importance of an already fitted model re-scores shuffled data without training another estimator. If used for explanation, calculate it on development data, not as feedback from Audit. Do not claim to have performed an additional feature-selection experiment without a counted subsequent model fit. [Official permutation-importance documentation](https://scikit-learn.org/stable/modules/permutation_importance.html).

## 10. Proposed repository responsibilities

These are future files, not files created by this planning task. Keep one local application; no microservices or agent orchestra.

| Proposed path | Responsibility |
|---|---|
| pyproject.toml and dependency lock | Reproducible local environment |
| configs/campaign.yaml | Frozen budget, seeds, split policy, candidate catalog, cutoffs, target/identifier mapping |
| src/autonomis/data.py | Official data contract, safe type parsing, immutable input checks |
| src/autonomis/splits.py | Saved Train/Search/Audit and paired robustness memberships |
| src/autonomis/features.py | V0/V1 definitions and per-training-partition preprocessing |
| src/autonomis/ledger.py | Atomic token reservation, statuses, persistence, cache keys, evidence export |
| src/autonomis/training.py | Sole estimator adapters, counted fitting, diagnostics, artifact persistence |
| src/autonomis/planner.py | Deterministic family/mutation/stop decisions from logged observations |
| src/autonomis/campaign.py | Sequential search, paired robustness, recipe/fallback freeze, finalization |
| src/autonomis/evaluation.py | Named metrics and one-time audit entry point |
| src/autonomis/predict.py | Artifact-only inference and organizer-schema export |
| src/autonomis/reporting.py | History, trajectories, metric tables, curves, read-only HTML evidence/demo |
| tests/ | Target mapping, split/leakage, planner, ledger, resume, artifacts, output contracts |
| artifacts/ | Versioned model bundles, preprocessing, label/schema metadata, manifests and checksums |
| reports/ | Actual run history, decisions, split summaries, separate audit report and figures |
| README.md | Setup, allowed entry points, experiment-count semantics, replay, inference, limitations |

The module contracts exchange a dataset manifest, split manifest, candidate recipe, reserved attempt, run result, and frozen final-artifact reference. Every result links to one consumed ledger attempt; every final prediction export links to the final-artifact hash. Reporting receives stored results and predictions and has no training capability. Do not duplicate training logic in notebooks or a demo.

## 11. Implementation sequence and acceptance gates

### Task 1: Freeze rules, inputs, and validation

- Obtain official artifacts and resolve accessible portal instructions; document remaining ambiguities.
- Freeze target mapping, allowed features, split membership, experiment accounting and campaign parameters.
- Acceptance: hashes and row memberships persist; no record/customer group crosses disallowed boundaries; both classes exist in evaluation partitions; the target and identifier cannot enter predictor features; Audit cannot flow to the planner.

### Task 2: Budget enforcement and one working baseline

- Implement the ledger, preprocessing, LR adapter, artifact serialization and prediction export together as the first vertical slice.
- Before official model fitting, test ledger/controller behavior with a fake trainer and synthetic returned metrics. Do not spend official dataset fits on infrastructure tests.
- Then execute the counted LR baseline as attempt 1, load its persisted artifact, and generate sample output without another fit. This becomes the earliest verified fallback.
- Acceptance: initial attempt is durable; load/predict reproduces stored scores within the recorded numerical tolerance; export preserves identifiers/order, has finite values in the probability range, and matches the actual organizer schema.

### Task 3: Tree adapters and adaptive planner

- Add CatBoost and LightGBM behind the same training gateway, then implement the finite catalog and deterministic controller.
- Test synthetic histories that trigger capacity reduction, horizon extension, family switching, duplicate rejection, invalid candidate rejection, and early finalization.
- Acceptance: the same history produces the same next recipe; changed diagnostics produce the expected different move; the controller cannot bypass the budget gateway; early stopping never receives Audit.

### Task 4: Execute one official campaign

- Complete the remaining counted anchors and adaptive fits using the same persistent ledger as attempt 1.
- Persist valid artifacts immediately. Generate plots from stored predictions and metrics, without refitting.
- Run only the permitted optional robustness block and reserve finalization.
- Acceptance: each executed attempt has a real status, reason and artifact/error; every adaptive proposal references prior feedback; all successful artifacts reload; the actual count remains at most 15.

### Task 5: Freeze, final fit, audit, predictions

- Select using the fixed search/paired rule, lock the fallback, and perform the reserved final fit.
- Persist and verify the selected artifact, Audit predictions, and read-back-validated official-test export before accessing Audit labels. Commit the hash-bound Audit-open record, then evaluate the locked predictions.
- Acceptance: audit score belongs to the submitted artifact; any finalization technical failure selects only the pre-frozen fallback before label access; no Audit-dependent model change; no training occurs while exporting predictions; restart cannot replace the audited artifact.

### Task 6: Reproducible evidence and working demo

- Build a compact local read-only report: budget used/remaining, chronological results, selected decisions, best-so-far Search curve, paired deltas, separate Audit metrics, and sample churn scores.
- Provide a local prediction action that loads the final artifact and validates the input schema. No account setup or API key is required.
- Write setup/replay/inference instructions and limitations. Retain evidence for the later mandatory deck; do not create that deck until requested.
- Acceptance: in a fresh process, report replay and inference work with zero new consumed fits; all displayed numbers originate in persisted results; no claimed hidden-test score appears before release.

## 12. Critical verification scenarios

| Scenario | Required behavior |
|---|---|
| 15 slots already consumed | Attempt 16 rejected before model training |
| Two callers request the last slot | At most one reservation succeeds |
| Crash after reservation or while fitting | Slot remains consumed; restart does not reset or secretly repeat training |
| Final training or persistence/reload/export gate fails | Before Audit access, use and validate the pre-frozen fallback without training; report the failure and actual fallback training fraction |
| Restart after Audit opens | Reuse only the bound model and prediction hashes; reject a replacement model and any further training |
| Attempt exceeds its timeout | Terminate the recorded worker, keep its token consumed, verify it stopped before further work |
| Ledger is corrupt or data/split hash changed | Refuse training; no reset or silent fresh campaign |
| Only one robustness pair completes | Rank on that common pair only and disclose fragility |
| No common robustness pair completes | Use precommitted search ranking; do not average unpaired results |
| Unknown categorical level at inference | Handle under the saved pipeline policy; no fitting or schema guessing |
| Empty numeric value or zero tenure | Documented missing handling; no division by zero or dropped test row |
| Duplicate customer observations | Respect pre-frozen grouping policy; no cross-partition identity leakage |
| Positive class stored second/first in estimator metadata | Correctly identify churn score from class labels |
| One-class evaluation partition | Fail validation before fitting; no fabricated AUC |
| Shuffled input order | Export restores organizer-required order and identifiers |
| Missing/extra feature column | Enforce the saved schema; fail clearly on required missing inputs |
| Saved model reloaded | Predictions match saved-reference predictions within documented numerical tolerance |
| Report/demo reopened repeatedly | Ledger count does not change |
| Low final Audit score | Report it honestly; no model reselection or training restart |

Infrastructure tests use fake training behavior. Any real predictive estimator trained on the official dataset, including a smoke test or notebook fit, must pass through and consume the same official ledger. Re-running the entire campaign as a reproducibility demonstration would spend more fits and is not part of this plan.

## 13. Eighteen-hour delivery schedule

Approximate allocation, measured from implementation authorization. Revise after inspecting official dataset size and actual hardware. Do not interpret these estimates as measured runtimes.

| Hours | Deliverable | Gate |
|---|---|---|
| 0-1.5 | Official files/rules, target/features, EDA, frozen split manifest | No training until contract and accounting are frozen |
| 1.5-4 | Ledger, fake-trainer tests, preprocessing and LR vertical slice | Budget refusal, persistence and output contracts work |
| 4-5 | Counted LR attempt 1, artifact load/predict, first valid output | Working fallback exists early |
| 5-7 | Counted tree anchors, controller, synthetic-history verification | Actual feedback can drive the next candidate |
| 7-10 | Remaining search and eligible robustness fits | Every fit is counted; preserve final token |
| 10-12 | Final selection/fit, audit, official prediction export | Submission artifact and file are verified |
| 12-14 | Read-only demo, reproducible README, figures and evidence | Replay/inference consume no fits |
| 14-16 | Later presentation preparation/rehearsal when requested | Evidence covers the brief's required structure; no invented results |
| 16-18 | Submission checks and recovery buffer | Keep a verified artifact and prediction file ready |

Hard operational priorities: first working persisted model before elaborate optimization; stop starting exploratory work by hour 10; aim to freeze a valid submission by hour 12. If a model's observed runtime threatens these cutoffs, shorten search/robustness under the predeclared time rule rather than jeopardizing a working submission.

Cut in this order if time is tight: optional V1 features, optional importance/uncertainty plots, extra cosmetic reporting, second robustness pair, then remaining adaptive moves. Preserve accounting, basic leakage checks, final predictions, README, and a working inference demo. If only one robustness pair is completed, use its paired-only rule and disclosure. Do not quietly change a frozen model recipe to speed the final fit; choose any speed-related recipe changes through counted search before freeze.

## 14. Evidence to retain for the later presentation

This is an evidence checklist, not a slide deck or a slide-content draft.

- Problem and objective: churn ranking under experiment constraints, with the exact metric/budget interpretation.
- Data: actual row/column counts, class balance, missingness, leakage exclusions, feature definitions and split memberships.
- Pipeline: one clear end-to-end diagram can be produced later from the implemented module boundaries.
- Algorithm: exact adaptive rules and two real examples where feedback changed the next decision.
- Experiment history: chronological records of all actual attempts, including failures, configuration, metrics, runtime and rationale.
- Evaluation: separate selection estimates, paired robustness evidence, final Audit result, and eventual official hidden-test result.
- Comparison: actual results for every supported successful anchor versus the chosen final recipe, with unavailable or failed anchors explicitly recorded. Three anchor results are the default successful path, not a reason to invent metrics. No unsupported claim that our planner beat random or Bayesian search.
- Demo: final model artifact, sample probabilities, output validation and replay without fitting.
- Limitations: unknown Efficiency Factor until clarified, small experiment budget, noisy development selection, held-out-data opportunity cost, actual class/group limitations, possible distribution shift, and any failure fallback.

## 15. Definition of done and honest limits

Implementation is complete only when the official prediction schema is satisfied, the final artifact can be loaded and used locally, every consumed fit is accounted for within 15, all required evidence is traceable, and the demo can run without training. The later deck and live/video demo remain organizer deliverables, even though neither is produced in this planning task.

This plan does not establish a performance result, a guaranteed winning technique, or an optimal Efficiency Factor tradeoff. Those depend on the absent official dataset, missing rules, and measured results. It does establish a specific executable design with bounded search, conservative accounting, a feedback-dependent controller, an audit that belongs to the submitted artifact, and a realistic zero-API build scope.

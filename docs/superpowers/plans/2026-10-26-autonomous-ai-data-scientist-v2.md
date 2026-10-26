# Autonomous AI Data Scientist Implementation Plan, Revision 2

> Planning and dataset acquisition only. No source code, notebook, dependency installation, model fit, or slide deck has been produced. This revision supersedes revision 1 and incorporates the user's Ubuntu training laptop, working GPU experience, zero API budget, and permission to obtain public reference data.

**Goal:** Deliver a feedback-driven, auditable churn experiment system within a maximum of 15 counted training attempts, using the friend's laptop as the single training authority and producing a reproducible inference/demo package.

**Architecture:** One local controller, one durable SQLite budget ledger, three estimator adapters, immutable split manifests, and saved model/evidence bundles. Search and optional paired sensitivity checks choose a frozen recipe. A development-trained audit model and a full-data final model are then fitted before audit results are revealed.

**Stack proposed for later implementation:** CPython with available compatible release wheels, pandas, NumPy, scikit-learn, CatBoost, LightGBM, SQLite, pytest, Matplotlib, and ordinary local files. Use an isolated environment alongside the existing PyTorch setup. Lock actual versions after verifying imports on Ubuntu; no version compatibility has been tested here. No API keys, LLM service, distributed training, or additional GPU framework is required.

**Spec:** Utkrishta 2026 Participant Problem Statements.pdf, repository root, pages 1 and 3. Revision 1 contains the original source analysis; this document is the current execution proposal.

## 1. Confirmed facts and remaining uncertainty

| Item | Confirmed state | Consequence |
|---|---|---|
| Training machine | Friend's Ubuntu laptop; RTX 4060, 8 GB VRAM, 24 GB RAM, Ryzen HS CPU reported as 7335HS | Plan for one local worker; exact CPU/core count and driver/package compatibility remain preflight checks |
| Existing GPU experience | Friend has used PyTorch GPU training | Useful readiness evidence; does not itself verify CatBoost's GPU binary |
| Prior work on this dataset | None reported | Zero experiment budget consumed so far |
| Available build window | Originally 18 hours; discussion time consumes the same overall window | Use milestones and preserve contingency; do not assume another 18 hours automatically starts |
| Paid API budget | None | Entire pipeline and planner operate offline after package/data acquisition |
| Organizer clarifications | None supplied; user authorizes us to choose operational policies | Freeze our accounting/split protocol, but do not invent an organizer scoring formula |
| Working data | Public IBM file downloaded with user authorization | Concrete schema and small-table plan now possible; official equivalence remains unverified |
| Official portal/test/submission | Unavailable; user will check later | Build to the working contract, mark official submission compatibility pending |

**PDF requirements:** Maximum 15 model-training experiments; one fit-and-validation cycle is one experiment; loading/cleaning/profiling/feature design are excluded. Primary hidden-test ROC-AUC, secondary PR-AUC/Average Precision. The next experiment must use feedback from previous runs. The leaderboard formula is ROC-AUC multiplied by an undefined Efficiency Factor. Overall judging: performance 40%, approach/innovation 20%, efficiency/constraints 15%, presentation 15%, code/demo 10%.

The undefined factor cannot be chosen by us. Record ROC-AUC, AP, exact attempt count and runtime separately, and leave the organizer-scaled score unavailable. The maximum-versus-all-15 wording is handled as a cap: never invent attempts or run useless fits to fill a table.

## 2. Concrete working dataset

Source: [IBM's Telco Customer Churn CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).

Local file: data/reference/ibm_telco/Telco-Customer-Churn.csv. Provenance and fingerprint: data/reference/ibm_telco/README.md.

Verified without model training:

- 7,043 rows; 21 columns; 970,457 bytes.
- SHA-256: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91.
- Target Churn: Yes 1,869; No 5,174. Map Yes to 1 and No to 0.
- customerID has no repeated groups; exclude it from predictors and retain it for identity/output.
- 19 raw predictors remain: numeric tenure, MonthlyCharges, TotalCharges; 16 categorical predictors including SeniorCitizen treated as a categorical indicator.
- TotalCharges has 11 whitespace-only entries; parse them as missing, preserve the customers, and learn any imputation only from each fit's training rows.

Preserve no internet service and no phone service states where present. They encode structural service availability and are not automatically equivalent to an ordinary No. The reference schema requires all 19 named predictors; trim outer whitespace, accept only Yes/No target labels and 0/1 SeniorCitizen values, and reject unknown target labels. Require finite nonnegative numeric charges when present and nonnegative integer tenure. Treat blank TotalCharges as missing, but reject other unrecognized numeric strings instead of silently coercing them. Validate the accepted official schema separately if it differs. Legitimate unseen levels in other categorical features follow the saved unknown-category policy and are logged; do not reinterpret them as target labels or fit a new encoder.

**Data boundary:** Public reference data are the authorized working input, not a verified organizer split. No official hidden-test file has been downloaded. A local audit is never labelled hidden-test performance. When official files arrive, use their train/test membership; do not look up public labels for their hidden-test IDs or silently merge those labels into training. If a dataset change follows counted training, preserve the existing budget/history and assess whether the submitted model has already seen any official test rows. Do not reset the counter.

Implementation and data-contract tests can proceed using the reference schema. Before consuming the 15-fit campaign, choose and record the accepted input fingerprint; if the organizer's files are still unavailable, the campaign is explicitly a reference-data campaign and organizer compatibility remains provisional. This prevents accidentally running one undocumented campaign now and another later.

## 3. Hardware policy: use the laptop, choose the backend deliberately

### Default for the downloaded table

| Component | Execution | Proposed control |
|---|---|---|
| Data preparation/metrics | CPU | Keep categorical one-hot output sparse; no unnecessary dense copies |
| Logistic Regression | CPU | Single fit at a time; limit BLAS/OpenMP pools to four threads or fewer if fewer cores are available |
| CatBoost | CPU | Four training threads; explicit backend/profile in every recipe |
| LightGBM | CPU | Four threads, deterministic mode enabled, force column-wise training |
| Report/demo/inference | CPU | Load persisted models; never fit from the demo |

This uses the friend's training machine without assuming the GPU is the best engine for 7,043 rows. CatBoost documents GPU speedups for sufficiently large datasets, beginning around tens of thousands of objects; this is not a measured crossover on this laptop. [CatBoost speed guide](https://catboost.ai/docs/en/concepts/speed-up-training). LightGBM notes GPU overhead and dataset-size effects in its [GPU performance guide](https://lightgbm.readthedocs.io/en/stable/GPU-Performance.html).

Use a conservative working host-memory target below approximately 16 GB and record peak worker memory where available, leaving headroom for Ubuntu and development tools. This is an operational target, not an estimator-enforced cap or a memory guarantee. The raw CSV is small; categorical expansion and fitted structures still need inspection.

### Conditional GPU profile

If organizer data are materially larger, CatBoost GPU becomes eligible before its first counted fit. Engineering gate: more than 50,000 labelled rows, supported released wheel, compatible detected driver, and approximately 6 GB free VRAM. The row cutoff is our policy, not an empirical optimum. Cap optional backend setup at 30 minutes; retain CPU if prerequisites are not ready.

Proposed GPU profile: device 0, GPU RAM fraction 0.75, Plain boosting, border count 128, maximum categorical combination complexity 2, initial depth 6 with catalog maximum 8. On an 8 GB device the fraction targets roughly 6 GB of model allocation; it does not guarantee total process memory or arbitrary-table fit. [CatBoost memory guidance](https://catboost.ai/docs/en/concepts/faq), [performance parameters](https://catboost.ai/docs/en/references/training-parameters/performance).

The CPU profile uses Ordered boosting, border count 254, Bayesian bootstrap with bagging temperature 1, categorical combination complexity 2, one-hot maximum category size 2, and SkipTest counter calculation. These are explicit starting hypotheses for the small mixed table, not claims of superiority. The GPU profile uses its supported settings and does not copy CPU-only parameters such as random_strength or binary-classification rsm tuning. Record all resolved defaults as well as supplied values. [CatBoost common parameters](https://catboost.ai/docs/en/references/training-parameters/common), [categorical counter settings](https://catboost.ai/docs/en/references/training-parameters/ctr).

CPU and GPU can use different algorithms/defaults, and CatBoost GPU training is nondeterministic even with a fixed seed. Backend/profile is part of recipe identity and remains fixed across search, paired checks, audit fitting and full-data fitting. Saved-artifact inference/replay is reproducible; identical GPU retraining is not promised. [Official GPU documentation](https://catboost.ai/docs/en/features/training-on-gpu).

Keep LightGBM on CPU even though Ubuntu permits GPU backends. Its OpenCL gpu and separate cuda implementations are different configurations; adding a build/backend project has no demonstrated benefit here. [LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html). Do not add XGBoost as a fourth anchor merely to occupy the RTX 4060.

### One authoritative execution machine

All real fits on the chosen dataset run on the friend's laptop through the same ledger. Current-workspace development, unit tests, report generation, and artifact-only inference do not create a second training campaign. Copying the repository does not create another 15-fit allowance.

Before implementation execution, inspect the laptop's Ubuntu/Python versions, exact CPU, available cores/RAM/disk, NVIDIA driver, device identity and available VRAM. Import/version/device inspection may not train a predictor. Any real model smoke test or CPU/GPU comparison on campaign data is a counted fit. In this plan a GPU initialization/driver failure makes that backend's family unavailable; use already valid other-family artifacts rather than silently switching it to CPU. A candidate-specific out-of-memory failure invalidates that candidate and suppresses larger-capacity neighbors, while lower-capacity candidates on the same backend may still be considered as new counted attempts. No event triggers an automatic CPU retry or backend change inside one token.

## 4. Revised experiment allocation

| Stage | Maximum attempts | Purpose |
|---|---:|---|
| Three initial anchors | 3 | Linear baseline, native-category booster, leaf-wise booster |
| Feedback-driven search | 8 | Local candidate moves and one feature ablation; total search at most 11 |
| Paired sensitivity check | 2 | Two frozen finalists on one additional shared development split |
| Audit-model fit A | 1 | Development 80% training; untouched 20% evaluation for this model |
| Full-data final fit P | 1 | All accepted labelled training rows; official test remains separate |
| Total maximum | 15 | Failures/interruption reservations consume these same slots |

This replaces revision 1's four-fit robustness block and 80%-only final model. It allocates one more fit to search and permits full-data final training, which better matches the hidden-test objective. It is a justified tradeoff, not demonstrated better AUC. One paired split provides limited sensitivity evidence and cannot support a significance claim.

Reserve two terminal tokens throughout search/checks. On the complete no-failure path, the audit and full-data fits are attempts 14 and 15. With early stopping or skipped checks they have earlier actual IDs; never manufacture missing history rows.

## 5. Data splits and what each score means

### Before attempt 1

Save stable row identities and the dataset hash. For the current one-row-per-customer reference table, make stratified Audit 20% with seed 20261026, then split the remaining 80% development pool into Train 75%/Search 25% with seed 20261027. This produces approximately Train 60%, Search 20%, Audit 20% of all labelled rows. Save one additional stratified 75/25 development partition with seed 20261028 for paired sensitivity checks.

The actual saved row/class counts, not rounded expected counts, are authoritative. If official data reveal repeated customers or a real prediction timestamp, choose group-disjoint or chronological splitting before fitting. Fail preflight if an evaluation partition has only one target class. No seed shopping.

Learn imputation, scaling, category dictionaries, and any selected transformation only from the training rows used by that specific fit. Never fit a target encoder on all labels. [Scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

### Phase-specific interpretation

| Phase | Training rows | Evaluation rows | Interpretation |
|---|---|---|---|
| Search | Train 60% | Search 20% | Reused optimization/selection estimate |
| Paired check | Same saved 60%-equivalent subset for both finalists | Same saved 20%-equivalent subset | Additional sensitivity/reranking evidence on already exposed development data |
| Audit model A | Development 80% | Audit 20% | Held-out score for A, used as a recipe-performance proxy |
| Final model P | All accepted labelled training rows | Organizer hidden test, when supplied | P has no internal held-out score; official result is unavailable until returned |

Do not show Search metrics as validation of A/P: Search is in their training data. Audit metrics/intervals belong only to A, or the explicitly named cached fallback if A fails. They are never labelled as P's score.

## 6. Models, preprocessing, and feature ablation

Common estimator seed: 20261026. No default class weighting or SMOTE. The objective is ROC-AUC ranking, with AP as the declared secondary comparator. Threshold tuning and calibration CV are outside this budgeted build.

| Family | Initial configuration | Allowed search moves |
|---|---|---|
| Logistic Regression | L2, C 1, lbfgs, maximum 3,000 iterations, tolerance 0.0001 | C 0.1, 1, 10 |
| CatBoost | Logloss, evaluation AUC, depth 6, learning rate 0.05, L2 leaf regularization 3, maximum 1,000 trees; frozen CPU/GPU profile from section 3 | Depth 4, 6, 8; L2 3, 10, 30 |
| LightGBM | Binary objective, evaluation AUC, 15 leaves, minimum child samples 50, L2 regularization 1, learning rate 0.05, maximum 1,000 trees, no class weights; CPU deterministic/column-wise profile | Leaves 7, 15, 31; minimum child samples 20, 50, 100; L2 1, 10, 30 |

Boosting can extend its cap from 1,000 to 2,000 trees only through another counted search fit when the best tree count reached at least 90% of its cap. During search, use Search-only early stopping with patience 100 and persist the best-tree artifact. Our declared accounting treats internal stopping as one fit. Record this policy as ours, not an organizer confirmation. Fixed-round search is the alternative if later rules disallow it. [CatBoost tuning](https://catboost.ai/docs/en/concepts/parameter-tuning), [LightGBM tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html).

**Logistic preprocessing:** Numeric median imputation plus missing indicators and numeric scaling, all trained on the fit's training rows; categorical missing sentinel and sparse one-hot encoding with unknown-category handling.

**CatBoost preprocessing:** Preserve the declared categorical columns as canonical strings, including categorical missing sentinel; parse numeric blanks as missing and use documented native numeric missing handling. No external target encoding. Preserve feature order and categorical indexes in the artifact metadata.

**LightGBM preprocessing:** Train-only numeric median imputation/missing indicators and sparse categorical one-hot encoding. Preserve category dictionaries and exact output-column order. Numeric scaling is unnecessary for this adapter.

**V0:** The 19 validated raw predictors with the preprocessing above.

**V1:** V0 plus exactly three predeclared row-wise features supported by the downloaded schema: log of one plus tenure; count of Yes across OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV and StreamingMovies; TotalCharges divided by positive tenure. Nonpositive tenure gives missing for the ratio, with missingness preserved. Do not divide imputed TotalCharges by tenure or silently replace missing charges with zero.

Spend one of the eight adaptive slots comparing V1 to the current V0 incumbent with all other settings unchanged, after every shortlisted family received its first local move. Freeze and record that exact parent recipe before making the V1 copy. Change only its feature package; a simultaneous parameter/backend change is not this ablation. Retain the measured comparison. If the attempt fails, its token remains consumed, mark the ablation attempted-failed, keep V0, and do not retry it automatically. If the accepted official schema lacks any required V1 source column, mark V1 not applicable and skip without a fit; never guess column mappings or substitute definitions. A time-based skip is also recorded before the slot is reserved. V1 is a hypothesis and may lose; do not claim engineered features help before testing.

## 7. Precise feedback-driven search policy

1. Execute the supported anchors in LR, CatBoost, LightGBM order. Rank by Search ROC-AUC, then AP, then lower fit time, then canonical recipe ID. For deterministic comparisons, round metric ranking keys to eight decimal places. Record unsupported/failed anchors without invented metrics.
2. Shortlist the best two valid families, or the sole valid family if only one exists. If no valid artifact exists, stop instead of consuming the remaining budget blindly.
3. Give every shortlisted family one adaptive local move in anchor-rank order. Then normally select the family with the highest incumbent Search score.
4. Enumerate unique, untried one-step neighbors of that family's incumbent. First consider a justified tree-cap extension. Otherwise, when in-sample Train ROC-AUC minus Search ROC-AUC exceeds 0.05, prioritize lower capacity then stronger regularization; otherwise prioritize a modest capacity increase then weaker regularization. These signals suggest hypotheses, not diagnosed causes.
5. Clone the selected family's exact incumbent and change one knob only. In the lower-capacity/stronger-regularization direction, try CatBoost depth down then L2 up; LightGBM leaves down then minimum child samples up then L2 up; LR C down. In the opposite direction invert those signs. Move to the adjacent catalog value, skipping invalid/executed recipes, then exhaust the opposite direction. Break remaining ties by canonical recipe ID. Feature version and backend remain in the recipe key. Exhausted families hand over to another available shortlisted family.
6. Perform the single V1 comparison at the first adaptive slot after the initial local move for every shortlisted family; this scheduled ablation takes priority over another local-neighbor proposal. It changes only feature version. An eligible successful V1 result becomes its family's incumbent only under the same ranking rule; a loss or failed attempt retains V0. Thereafter clone the selected family's exact incumbent, preserving its winning feature version. Not-applicable/time-skipped ablations consume no fit.
7. Update each incumbent by the ranking rule. For scheduling, call an improvement below 0.002 ROC-AUC relative to the parent's score non-material. Two consecutive non-material local moves in a family give the other available family priority for its next move. With one family, continue its remaining neighbors.
8. Stop at 11 reserved search attempts including failures, exhausted useful neighborhoods, the exploration deadline, or four consecutive valid adaptive results with no individual global improvement of at least 0.002 after every shortlisted family had at least two local attempts and the feature ablation was resolved as succeeded, lost, attempted-failed, not applicable, or time-skipped. Tiny genuine improvements can update the incumbent without resetting this scheduling rule.
9. Choose the two top distinct valid recipes, including two from the same family if appropriate. The optional paired check requires Search AUC gap at most 0.01, four remaining tokens, and time for two probes plus both terminal fits. Freeze recipes, actual trained tree counts, preprocessing and backend before either check. Run each once on the same pre-saved development split, with no early stopping.
10. If both checks succeed, rank by paired-split ROC-AUC, AP, lower fit time and recipe ID. If either fails or is skipped, use the original Search ranking. Do not compare one candidate's paired score against another candidate's original Search score, or invent a replacement probe.

The 0.05/0.002/0.01 values are declared engineering heuristics, not significance thresholds. This is an explicit sequential heuristic, not a learned policy or claim of optimal experimental design. Its innovation evidence is the causal chain from observed feedback to the next action and an enforceable budget.

Each decision records parent run IDs, incumbent, remaining tokens/time, candidate alternatives, rejected duplicates, selected mutation, rationale, observed result and hypothesis outcome. Replay uses stored observations and does not train.

Default TPE would use ten startup trials; reducing that default is possible, but we have no evidence that a mixed-family surrogate with so few observations improves this campaign. Keep the simpler controller. [Official TPESampler documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

## 8. Terminal protocol: more training data without misleading audit claims

Freeze the winner, its actual search-trained tree count, backend and features before either terminal fit. Use that count unchanged for the paired check, A and P, with early stopping and best-model selection disabled. Transferring the count is a budget-conscious heuristic, not proof it is optimal for larger samples; do not invent proportional tree-count scaling or run a learning-curve helper with hidden extra fits.

1. Verify cached winner-search artifact F can reload and make valid Audit predictions and schema-valid inference/export. Preserve its actual training fraction and source attempt ID. This is the predeclared fallback.
2. Fit A once on development 80%, with fresh preprocessing learned only on those rows. Persist/reload it and save Audit predictions. Do not calculate or show Audit metrics yet. If any technical gate fails, A is unavailable and F remains the audit proxy.
3. Fit P once on all accepted labelled training rows with the same frozen recipe, fresh preprocessing on those rows, no local validation set and no early stopping. This fit is charged even without local held-out labels. Verify persistence, reload and inference/export. A technical failure does not permit a retry beyond the planned terminal attempt.
4. Before seeing any Audit result, choose the submission by technical success P, otherwise A, otherwise F. Choose the audit proxy A, otherwise F. A failed A does not automatically prevent the one frozen P attempt if inputs are valid and time permits. A skipped P leaves its token unused.
5. Commit both selected model hashes and the proxy's Audit-prediction hash to a durable audit-open record. Retain the hashes of all inference exports already generated. Then compute Audit metrics for the proxy only. The ledger rejects any further training or replacement of those locked identities/predictions. A restart may recompute reports from the same predictions. Later inference exports are append-only records keyed by the locked model hash, input-file hash and output-contract hash; they do not replace provisional exports or reopen Audit.

**Important access distinction:** P necessarily trains on the former Audit labels. The final trainer may consume them only for the already frozen full-data fit. The planner receives no Audit metrics or label/error analyses before both artifact choices are locked. Audit predictions for A/F are persisted before P trains. This prevents result-driven retuning while retaining an honest held-out metric for A/F.

If both terminal fits fail, submit verified F when possible and report the actual artifact/fraction. If F cannot pass technical validation, report no verified deliverable; do not fabricate a score or silently train again.

Currently no official test file exists in the workspace. A provisional output uses customerID and churn_probability for functional tests and is explicitly labelled provisional. It does not count as organizer-format acceptance. Once the official test/sample arrives, freeze exact columns/order, types, row identity/order, score semantics and header/index policy. Generate a new append-only official export by inference from the locked artifact and read it back to verify all rows, preserving the provisional file and recording the model/input/contract/output hashes. No extra training or Audit reopening is required. If official data provenance would invalidate the model, surface that issue instead of pretending output formatting fixes it.

## 9. Ledger, failure handling, and timing

- SQLite on the friend's laptop is the source of truth. One training gateway reserves an irrevocable token transactionally before invoking an estimator. Count reservations, not successful models. At 15, another fit is rejected before estimator invocation.
- Search/probes preserve two terminal tokens. Failures, interrupted fits, serialization/metric failures after training, and uncertain crash-after-reservation states remain consumed. No reset, deletion, hidden retry or new-machine allowance.
- Planned candidates and preflight rejections consume nothing, provided no predictor trains. A successful duplicate recipe/split/seed returns cached results; a deliberate different split/backend/seed is a different counted attempt.
- Include attempt role, data/split/recipe hashes, parent evidence, seed, environment, backend, timestamps, warnings/errors, metrics, actual tree count, training fraction, process identity and artifact hashes. Corrupt/missing campaign state or changed data/split hashes prevents silent resume.
- A supervised worker has a 30-minute maximum per attempt and a shorter timeout when a phase deadline is closer. On timeout, stop the worker, keep the token consumed, and verify termination before another fit starts.
- Freeze an exploration cutoff and terminal-fit cutoff before launch, derived from the actual event deadline. Suggested milestones below stop exploration by implementation hour 8 and terminal fitting by hour 10; the original 18-hour overall deadline still governs.
- For admission planning, estimate probe time as 1.5 times the slowest successful finalist search fit, A time as twice that duration, and P time as 2.5 times it. The latter two combine row growth with a 1.5 safety factor. Require estimated remaining work plus a two-minute margin to fit the phase windows and recheck before every attempt. These are estimates, not measured scaling laws or guarantees. If needed, skip probes or P and retain the verified fallback.
- Calibration CV, stacking, RFECV, repeated seeds, AutoML and optimizer wrappers can contain extra estimator fits; exclude them from this build. Bootstrap, scoring saved predictions, plots, artifact loading and replay do not train.

Expose three separate clocks in evidence: fit wall time, full attempt wall time including preprocessing/metrics/persistence, and cumulative campaign time. GPU startup/transfer overhead belongs in measured wall time; do not report only kernel time.

## 10. Metrics and judge-facing evidence

Compute ROC-AUC and Average Precision on positive-class scores; verify the estimator's class mapping. AP and trapezoidal PR-curve area are not interchangeable, so label any additional PR-area calculation explicitly. [ROC-AUC definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

Search trajectory contains search-stage scores only. Paired scores identify their own split. Audit panel names proxy artifact A/F and its training fraction; final P has no internal held-out score. Hidden-test and efficiency-adjusted scores remain unavailable until actually supplied.

Optionally derive a 95% interval for the audit proxy from 2,000 stratified bootstrap samples of its saved label/prediction pairs, seed 20261030, using percentile endpoints. This is conditional on that fitted proxy, class counts and sampling assumptions; it does not quantify P's score, training variability or future shift. If accepted data require customer grouping or temporal dependence, use a justified cluster/block bootstrap matching that dependence; otherwise omit the interval and disclose the limitation. Do not apply the default independent-row bootstrap in that case. Do not run significance tests on a single paired probe. Reused validation selection itself can overfit: [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html).

| Rubric | Required artifact/evidence |
|---|---|
| Performance 40% | Actual anchor/search results, paired deltas when available, proxy Audit ROC-AUC/AP, valid final probabilities, eventual organizer result |
| Innovation 20% | Real feedback-to-next-action examples, rejected alternatives, early-stop/switch decisions, deterministic replay |
| Efficiency 15% | At-most-15 ledger including failures, budget trajectory, full wall times, enforced refusal and resume behavior |
| Presentation 15% | EDA and missingness, pipeline design, chronological actual attempts, labelled comparison tables, limitations and later Q&A material |
| Code/demo 10% | Locked environment, saved native models/preprocessing, README, local inference and report replay with training disabled |

No promise of a particular AUC, GPU speedup, improved V1 features, planner superiority or winning score is justified before measurements. AUC 0.90 or 0.95 is not an acceptance target invented from the brief.

## 11. Future file responsibilities and implementation gates

No files in this table are implemented by this planning task, except the explicitly named downloaded data/provenance documents.

| Future path | Responsibility |
|---|---|
| configs/campaign.yaml | Dataset fingerprint/role, target, backend profiles, seeds, catalog, budget, clock cutoffs and provisional/official output contract |
| src/autonomis/data.py and splits.py | Raw input validation, row identity, per-phase immutable memberships |
| src/autonomis/features.py | V0/V1 row transforms and training-only fitted preprocessing |
| src/autonomis/ledger.py | Atomic reservations, irreversible consumption, statuses, audit lock and resume |
| src/autonomis/training.py | Sole model training entry, CPU/GPU adapters, timeouts, diagnostics and artifacts |
| src/autonomis/planner.py and campaign.py | Finite feedback policy, paired check, two terminal roles and fallback choices |
| src/autonomis/evaluation.py and predict.py | Named metrics, proxy Audit evaluation, artifact-only inference and output validation |
| src/autonomis/reporting.py | Stored-evidence charts and read-only local demo/report |
| artifacts/ and reports/ | Versioned bundles, metadata, ledger exports, decisions, plots and separate Audit report |
| tests/ and README.md | Contract/ledger/planner verification, installation, inference, replay and limitations |

Implement in six reviewable steps:

1. **Input and environment contract:** inspect Ubuntu/runtime, freeze working/official data role and exact split manifests; imports/device inspection only. Gate: correct target, no ID leakage, valid partitions, reproducible data hash.
2. **Vertical slice:** ledger, LR preprocessing, counted attempt 1, saved model, reload and provisional prediction output. Gate: a verified fallback exists before elaborate optimization. Use fake trainers to verify infrastructure before touching campaign fits.
3. **Tree adapters and controller:** add CatBoost/LightGBM under the same gateway, execute counted anchors, implement feedback policy against synthetic histories. Gate: same stored feedback gives same decision, changed evidence changes appropriate actions, no Audit feedback reaches search.
4. **Search and optional paired check:** execute sequentially, saving every successful artifact immediately. Gate: at most 11 search attempts, reserved terminal tokens, complete paired results or original-ranking fallback.
5. **A/P terminal stages:** freeze recipe/F, fit and verify A, fit and verify P, lock both choices, then reveal proxy Audit metrics. Gate: no metric-driven selection and correct artifact/score labelling.
6. **Packaging and demonstration:** bundle native model plus preprocessing, class/schema metadata, environment and provenance; read-only history/demo; README and later presentation evidence. Gate: inference/reporting work without training and output reload matches the saved contract.

Artifact bundles include raw-data fingerprint, stage training memberships, exact feature ordering/category definitions, target mapping, all resolved parameters, backend, package versions, native estimator/preprocessor files, and reference predictions/checksums. Use the same supported environment to verify CPU inference on the saved bundle; do not assume cross-version or GPU/CPU predictions are bitwise identical.

## 12. Verification that matters

Use a fake trainer for ledger/controller tests; these do not fit a predictive model. Any actual predictor trained on campaign data, including a notebook smoke test, is counted.

- Attempt 16 never reaches the trainer; concurrent last-token requests cannot both succeed.
- Search/probes cannot consume either reserved terminal token. Failed/timeout/crashed reservations stay consumed across restarts.
- Changed data/split hashes, corrupt ledger, or an active unresolved worker prevent a silent fresh run.
- One failed paired probe restores the search ranking; no mixed-split comparison or replacement beyond the two slots.
- A failure still permits the one frozen P attempt when inputs/time are valid; P failure selects verified A/F before Audit metrics appear.
- A/P/fallback technical validation covers persistence, reload, required columns, positive-class score, finite probability range and saved-output read-back.
- Audit lock binds both submission/proxy model identities and the proxy's Audit predictions; restart cannot swap them or authorize more training. Later official inference exports append their input/contract/output hashes without replacing provisional records or reopening Audit.
- Unknown categories and blank charges use the persisted preprocessing policy; zero tenure cannot generate infinity; missing required columns fail clearly.
- Test row identity/order and output headers/index match the selected provisional/official contract; no target column enters inference.
- Search, paired, proxy Audit and P results cannot be silently mixed in the report. P has no internal holdout metric.
- Demo replay and prediction run with training disabled and zero tokens remaining. No service/API key or new fit is needed.

## 13. Delivery schedule

Approximate work allocations, not measured training times. Deduct elapsed discussion/preparation from contingency and bind the real clock cutoffs before launch.

| Build hours | Deliverable |
|---|---|
| 0-0.5 | Ubuntu/import/device inspection, data provenance/role, frozen accounting and split protocol |
| 0.5-2 | Ledger, fake-trainer tests, preprocessing and inference/export contracts |
| 2-3 | Counted LR baseline; saved/reloaded model and functional output |
| 3-5 | Counted tree anchors and tested deterministic controller |
| 5-8 | Remaining adaptive moves and eligible paired check; stop exploration by hour 8 |
| 8-10 | A/P fits or verified fallbacks, artifact lock, proxy Audit and prediction packaging |
| 10-12 | README, evidence report, inference/demo verification |
| 12-15 | Later requested presentation preparation/rehearsal and event submission checks |
| 15-18 | Contingency for environment issues, packaging and actual submission |

Cut optional V1 ablation, optional importance/bootstrap plots, cosmetic reporting, paired checks and remaining search moves before compromising a working artifact or budget controls. Any optional feature experiment skipped for time is logged. If the deadline arrives before P, submit verified A/F with honest training-fraction disclosure rather than starting an unsafe last-minute fit.

## 14. Completion and unresolved inputs

Planning output now consists of this document plus the downloaded public CSV and provenance note. Implementation remains unstarted.

The system is ready for a reference-data demonstration when its saved artifact, score labels, durable budget, replay, inference and provisional output validate. It is ready for official submission only after the actual organizer input/output contracts and train/test provenance are verified. A working public-data pipeline does not prove that compatibility.

Still unknown: portal data identity/membership, actual official test features and output schema, the Efficiency Factor, any later organizer interpretation of counting, and measured runtime on the Ubuntu laptop. These do not justify invented numbers. If any answer changes the dataset, counting or accepted final-training protocol, resolve that change explicitly before continuing the counted campaign.

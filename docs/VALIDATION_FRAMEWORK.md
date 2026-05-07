# Biometric Engine Validation Framework

## Status
Placeholder scaffolding has been replaced by real executable validation logic. All four scripts now contain production-grade implementations that call the actual backend pipeline functions or perform real statistical analysis on pipeline output data.

**Full validation must only run with a real ground-truth manifest** containing actual probe/gallery image pairs with verified identity labels. The example manifest (`validation_pairs.example.csv`) uses placeholder GCS paths and must not be used for real evaluation.

## Why Validation is Needed
The Fallen facial verification pipeline implements a multi-stage biometric system comprising neural face embeddings (ArcFace + Facenet512 60/40 ensemble) and classical computer vision mark extraction. Currently, thresholds like the `structural_sim < 0.40` safety rule and the naive multiplication of mark Likelihood Ratios (LRs) are heuristically defined. Before any math, threshold, or policy changes can be deployed, we must construct an empirical baseline to measure the Equal Error Rate (EER), False Acceptance Rate (FAR), and False Rejection Rate (FRR).

## Approved Execution Environments & The No-Local-Execution Rule
**STRICT RULE:** Validation scripts must NEVER be executed on local developer machines. Processing biometric data requires a controlled, auditable, and production-mirrored environment.

Approved execution environments are:
1. **GitHub Actions** (via `workflow_dispatch` — manual trigger only)
2. A dedicated **Cloud Run validation job/service**
3. The existing **backend container image** executing in a secure cloud shell

**No private images may be committed to the repository.** All image references must be GCS URIs (`gs://...`).

## Dataset Format
Validation requires a structured dataset mapped via a CSV manifest (`validation_pairs.csv`).

**Required Columns:** `pair_id`, `image1_url_or_gcs_path`, `image2_url_or_gcs_path`, `label_same_person`, `category`, `notes`, `expected_challenge_type`

**Categories:**
* `same_person_normal`
* `same_person_age_gap`
* `same_person_lighting_pose`
* `different_person_random`
* `different_person_lookalike`
* `mark_heavy_same_person`
* `mark_heavy_different_person`
* `twins_or_high_similarity_imposters`

## Scripts

### 1. Full-Pipeline Validation (`scripts/validate_full_pipeline.py`)
**Production-equivalent.** Calls the exact same backend functions used by `/verify/fuse`:
- `fetch_image_from_url` → GCS image loading
- `apply_clahe` → CLAHE preprocessing
- `align_face_crop` → MediaPipe face alignment to 256×256
- `extract_ensemble_embeddings` → ArcFace + Facenet512 extraction
- `compute_ensemble_similarity` → 60/40 weighted cosine similarity
- `_run_mark_evidence_pipeline` → mark detection + matching + LR
- `score_to_lr_ensemble` → calibrated LR from ensemble score
- Bayesian posterior fusion → `LR_total → posterior`
- Veto protocol → `structural_sim < 0.40`

Does NOT write to the database, create payment jobs, call the frontend, or mutate production data.

Supports `--dry-run` to validate manifest structure without processing images.

**Outputs:** `validation_results.jsonl`, `summary_metrics.json`, `false_positives.csv`, `false_negatives.csv`, `conflicting_evidence_cases.csv`

### 2. Embedding-Only Diagnostics (`scripts/validate_embeddings.py`)
**Non-production-equivalent (clearly labeled).** Designed for rapid threshold sweeps across ArcFace, Facenet512, and ensemble configurations. Extracts real embeddings using the same backend functions but bypasses mark detection and Bayesian fusion.

Tests: ArcFace alone, Facenet512 alone, 60/40 ensemble, 50/50, 70/30, 80/20 alternatives, and full threshold sweep from 0.00 to 1.00 at 0.01 increments.

**Outputs:** `threshold_sweep.csv`, `embedding_metrics.json`

### 3. Policy Simulation (`scripts/simulate_decision_policies.py`)
Reads `validation_results.jsonl` and simulates 5 alternative safety rule / decision policies:
- Current hard safety rule (`structural_sim < 0.40 → score = 0`)
- Soft cap (cap at 50 instead of zeroing)
- Human review flag (keep score, flag for review)
- Separate evidence channels (match if face OR marks strong)
- Strict mark override (override only if 3+ marks AND LR ≥ 100)

**Outputs:** `policy_comparison.json`, `policy_confusion_matrices.csv`, `recovered_true_positives.csv`, `new_false_positives.csv`

### 4. Mark LR Correlation (`scripts/evaluate_mark_lr_correlation.py`)
Reads `validation_results.jsonl` and evaluates 5 alternative mark LR aggregation models:
- Naive multiplication (current production)
- Region cap (cap LR per facial quadrant)
- Distance decay (diminishing returns for successive marks)
- Log-LR cap (cap `log10(total_LR)` at configurable maximum)
- Composite region clustering (cluster adjacent marks, take max LR)

**Outputs:** `mark_lr_comparison.json`, `mark_lr_distribution.csv`, `overcounting_cases.csv`

## Metrics Produced
The framework outputs JSON lines and CSV reports calculating:
* **AUC (Area Under Curve):** Overall model separability
* **EER (Equal Error Rate):** Where FAR meets FRR
* **FAR / FRR:** False Accept / False Reject rates
* **Confusion Matrix:** True/False Positives/Negatives
* **Precision / Recall**
* **Conflicting Evidence Cases:** Face weak / marks strong, and vice versa

## Interpretation & Constraints
* **What decisions cannot be made yet?** We cannot alter the 60/40 ensemble weights, change the `< 0.40` veto threshold, or implement mark dependency penalties until this framework produces a statistically significant output report over a real ground-truth dataset.
* **How to interpret results:** If removing the hard veto recovers 50 True Positives but introduces 10 False Positives (imposters), the business logic must determine if that tradeoff is acceptable based on the deployment's security posture.
* **No local execution.** All scripts are designed to run inside the backend container or via GitHub Actions only.

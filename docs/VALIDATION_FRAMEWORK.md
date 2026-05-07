# Biometric Engine Validation Framework

## Why Validation is Needed
The Fallen facial verification pipeline implements a multi-stage biometric system comprising neural face embeddings and classical computer vision mark extraction. Currently, thresholds like the `structural_sim < 0.40` safety rule and the naive multiplication of mark Likelihood Ratios (LRs) are heuristically defined. Before any math, threshold, or policy changes can be deployed, we must construct an empirical baseline to measure the Exact Error Rate (EER), False Acceptance Rate (FAR), and False Rejection Rate (FRR). This validation framework establishes the methodology to gather this data without altering production behaviors.

## Approved Execution Environments & The No-Local-Execution Rule
**STRICT RULE:** Validation scripts must NEVER be executed on local developer machines. Processing biometric data requires a controlled, auditable, and production-mirrored environment.

Approved execution environments are:
1. GitHub Actions (via workflow dispatch)
2. A dedicated Cloud Run validation job/service
3. The existing backend container image executing in a secure cloud shell.

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

> Note: All image paths must be secure GCS URLs (e.g., `gs://...`). No private images may be committed to the repository.

## Execution Modes
### 1. Full-Pipeline Validation (`validate_full_pipeline.py`)
This is the primary ground-truth evaluation. It does not bypass any part of the system. It passes image pairs through the exact same alignment, bounding, embedding, and mark detection logic utilized by the live `/verify/fuse` endpoint.

### 2. Embedding-Only Diagnostics (`validate_embeddings.py`)
This is a secondary, *non-production-equivalent* mode. It is designed to rapidly sweep thousands of threshold permutations across pre-extracted ArcFace and FaceNet512 vectors to find the optimal mathematical crossover point (EER). It bypasses the mark pipeline entirely.

## Metrics Produced
The framework outputs JSON lines and CSV reports calculating:
* **AUC (Area Under Curve):** Overall model separability.
* **EER (Equal Error Rate):** Where FAR meets FRR.
* **FAR / FRR:** False Accept / False Reject rates.
* **Confusion Matrix:** True/False Positives/Negatives.
* **Precision / Recall**
* **Conflicting Evidence Cases:** E.g., Face similarity is weak but mark evidence is strong.

## Interpretation & Constraints
* **What decisions cannot be made yet?** We cannot alter the 60/40 ensemble weights, change the `< 0.40` veto threshold, or implement mark dependency penalties until this framework produces a statistically significant output report over the dataset.
* **How to interpret results:** If removing the hard veto recovers 50 True Positives but introduces 10 False Positives (imposters), the business logic must determine if that tradeoff is acceptable based on the deployment's security posture.

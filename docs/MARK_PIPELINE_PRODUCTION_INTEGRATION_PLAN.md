# Mark Pipeline Production Integration Plan

This document outlines the strategy for integrating the deterministic V2 Mark Detection and Matching Pipeline into the production scoring routes (`/verify/fuse` and `/vault/search`). 

## 1. Current Stable Diagnostic Path
The `/marks/analyze` endpoint currently acts as a forensic-grade, deterministic diagnostic baseline. Its flow operates as follows:
1. **Raw Image Input**
2. **`align_face_crop(raw)`**: Initial alignment at 256x256.
3. **`preprocess_for_mark_detection(aligned, target_size=1024)`**: Deterministic single-CLAHE enhancement and sizing.
4. **`detect_facial_marks(..., input_is_preprocessed=True)`**: Skips internal double-CLAHE, leveraging the preprocessed input.
5. **`mark_matcher.match_facial_marks(...)`**: V2 isolated matcher applying 1:1 Hungarian assignment via normalized canonical coordinates.
6. **Diagnostics/Provenance Response**: Full telemetry, deterministic hashing, Bayesian LR values, rejected candidate tracking, and calibration statuses returned.

## 2. Current Legacy Production Path
The production verification (`/verify/fuse`) and search (`/vault/search`) endpoints currently utilize the legacy mark flow:
1. **`apply_clahe(raw)`**: Applies CLAHE prematurely prior to alignment.
2. **`align_face_crop(clahe_image)`**: Aligns the already-enhanced image.
3. **`detect_facial_marks()`**: Default legacy mode with `input_is_preprocessed=False`, causing a redundant internal CLAHE pass (double-CLAHE).
4. **Old In-Main `match_facial_marks()`**: Legacy threshold-based pairing logic embedded in `main.py`.
5. **`compute_mark_correspondence()`**: Legacy statistical computation for LR.

## 3. Exact Shared Helper Proposal
To safely transition without duplicating code, we will consolidate the V2 mark pipeline into a dedicated helper function.

**Function Signature:**
```python
def _run_mark_evidence_pipeline(
    probe_img: np.ndarray,
    gallery_img: Optional[np.ndarray],
    probe_file_hash: Optional[str],
    gallery_file_hash: Optional[str],
    mode: str, # "diagnostic" | "production"
    target_size: int = 1024
) -> dict:
```

**Inputs:**
- Raw OpenCV image arrays for probe (and optionally gallery).
- SHA256 hashes representing the original bytes of the probe/gallery for self-match logic.
- Operational mode (`"diagnostic"` or `"production"`).
- Target resolution size for mark detection (default 1024).

**Outputs:**
A comprehensive dictionary encapsulating all required diagnostic, match, and evidence states:
- `mark_match_status`
- `matcher_status`
- `lr_marks`
- `individual_mark_lrs`
- `accepted_correspondences`
- `rejected_correspondences`
- `probe_preprocessing` & `gallery_preprocessing` (including deterministic hashes)
- `detector_thresholds` & `mark_detector_trace`
- `mark_detector_version`, `mark_matcher_version`, `preprocessor_version`

**Behavioral Requirements:**
- **Exact Self-Match Handling**: If `probe_file_hash == gallery_file_hash`, the function will bypass the matcher, yield `EXACT_SELF_MATCH`, neutralize the statistical `lr_marks` to `1.0`, and populate the rejection summary explicitly indicating circular evidence avoidance.
- **Calibration Missing Handling**: If `TIER4_CALIBRATION` is absent, the LR will safely default to `1.0`, and `calibration_status` will reflect `"MISSING"`.
- **Probe-Only Mode Handling**: If `gallery_img` is absent, the matcher is bypassed, `mark_match_status` is `"NOT_RUN_SINGLE_IMAGE"`, and probe extraction metadata is returned.
- **Face Detection Failure Handling**: If `align_face_crop` yields `None` for landmarks, the pipeline early-exits, returning `FACE_NOT_DETECTED` statuses while preserving threshold and version transparency.
- **Preprocessor Hash Propagation**: The preprocessor's generated `decoded_hash` and intermediate payload must directly pipe into the response for auditability.
- **Threshold Propagation**: Detection and matcher thresholds must be strictly derived from `get_detector_thresholds()` and `get_matcher_thresholds()` modules.

## 4. Risk Controls
To ensure zero degradation to production stability, the following constraints must be strictly observed:
- **Preserve the Baseline**: Keep `/marks/analyze` active and untouched as the known-good diagnostic baseline against which the helper logic will be validated.
- **Phased Integration**: Integrate the helper into `/verify/fuse` first. `/vault/search` will only follow once `/verify/fuse` proves fully stable.
- **Isolate Embeddings**: The production face embedding and Bayesian baseline paths must remain completely unaffected. Mark evidence swapping must occur only after alignment/preprocessing boundaries are verified.
- **No Premature Schema/Frontend Changes**: Do not mutate the `VerificationEvent` database schema or SymmetryMerge frontend UI until the backend JSON response contract proves stable in production routes.
- **Feature Flag Safeties**: The migration of `/verify/fuse` to the new helper must be initially gated behind an environment variable feature flag (e.g., `USE_MARK_PIPELINE_V2=true`).

## 5. Tests Required Before Production Route Integration
Before pointing production routes to the helper, the following backend endpoint contract tests must be implemented:
1. `/verify/fuse` contract test with mocked images and mocked face embeddings to ensure end-to-end traversal.
2. `/verify/fuse` exact self-match mark neutrality test to guarantee that duplicate ingestion halts false LR inflation.
3. `/verify/fuse` calibration-missing neutral LR test to guarantee fallback scoring behavior.
4. `/verify/fuse` pre/post CLAHE hash population test to audit image integrity traces.
5. `/vault/search` traversal test (deferred until Phase F).
6. Regression test confirming that when the `USE_MARK_PIPELINE_V2` flag is enabled, production routes no longer execute double-CLAHE.

## 6. Implementation Order
We will proceed via a strictly ordered, risk-averse execution pipeline:
1. **Step A**: Add the shared helper `_run_mark_evidence_pipeline` in `backend/main.py` (or a dedicated module) using the `/marks/analyze` logic, but *do not hook any endpoints to it yet*.
2. **Step B**: Write direct unit/contract tests for the helper.
3. **Step C**: Wire the existing `/marks/analyze` endpoint to the new helper. Run tests to prove zero regression on the diagnostic contract.
4. **Step D**: Feature-flag `/verify/fuse` to intercept its mark scoring flow and route it through the new helper.
5. **Step E**: Execute the required `/verify/fuse` tests (from Section 5).
6. **Step F**: Consider routing `/vault/search` to the helper only after `fuse` is validated.
7. **Step G**: Update frontend diagnostic visualizations strictly after backend API consistency is secured.

## 7. Explicit Non-Goals
This integration plan *strictly prohibits* the following until specifically requested:
- No frontend UI work or React/SymmetryMerge modifications.
- No modifications to the database schema.
- No tuning, tweaking, or altering of detector/matcher thresholds or confidence logic.
- No implementation of generative AI enhancements (GFPGAN, CodeFormer, etc.).
- No production scoring changes beyond the fundamental plumbing of the deterministic mark evidence.

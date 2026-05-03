# Scoring Pipeline Audit — Forensic Facial Marks

This document serves as the institutional-grade audit trail for the Bayesian Likelihood Ratio (LR) pipeline for forensic facial mark matching.

**Version:** v2.0.0  
**Effective:** v1.138.0

## 1. Mathematical Source of Truth: Bayesian LR Formula

The scoring pipeline operates on strict Bayesian principles to fuse facial geometry (ensemble model) with secondary forensic mark evidence.

The overarching mathematical fusion formula is:
```text
LR_total = LR_ensemble * lr_marks
```

### 1.1 `lr_marks` Calculation
The `lr_marks` product is strictly the aggregate of valid, probabilistically independent mark correspondences. It is only included when sufficient high-quality mark correspondences exist and are independent of facial shape structure.

## 2. Exact Byte-Identical Image Self-Match Policy

### Policy

For exact byte-identical image comparisons (where `probe_file_hash == gallery_file_hash`):

- `exact_image_match = true`
- `mark_match_status = "EXACT_SELF_MATCH"`
- `mark_correspondence_score = 100.0`
- `marks_matched = min(len(valid_probe_marks), len(valid_gallery_marks))`
- Correspondences are identity-mapped by index (mark `i` ↔ mark `i`)
- `lr_marks = 1.0` (neutral multiplier)
- `mark_lrs = []` (no individual LRs computed)

### Rationale

For exact byte-identical image comparisons, mark LR is **not used as independent evidence** because the probe and gallery are the exact same image. The forensic mark correspondence is trivially true, meaning computing Bayesian Likelihood Ratios would be mathematically meaningless. It would inflate confidence based on circular evidence.

Identity confidence for self-matches is handled by the **exact-image sanity path** (hash comparison), not by the mark evidence system. The backend ensures the neutral LR (`1.0`) prevents statistical distortion.

## 3. Insufficient Marks Logic

Both `/verify/fuse` and `/vault/search` enforce identical logic for mark viability using a strict **OR** condition:

```python
if len(valid_probe_marks) < 2 or len(valid_gallery_marks) < 2:
    mark_match_status = "INSUFFICIENT_MARKS"
```

If **either** side has fewer than 2 reliable marks, the system asserts that it cannot make a statistically meaningful mark comparison.

## 4. Logical Decoupling: Detector vs Calibration

- **`mark_match_status`**: Reflects the status of the mark *detector* subsystem. (e.g., `MATCHED`, `INSUFFICIENT_MARKS`, `EXACT_SELF_MATCH`, `NO_MATCHES`, `DETECTOR_UNAVAILABLE`).
- **`calibration_status` (`TIER4_CALIBRATION`)**: Reflects whether the *Bayesian population dataset* is available to compute statistical prevalence (LR probabilities) of a given mark type in a given region.

These domains are strictly decoupled. Missing calibration data will fallback gracefully to a neutral mark score (`lr_marks = 1.0`) without erroneously flagging the detector hardware/software as broken.

## 5. Audit Ledger (`VerificationEvent`) Schema Integrity

The `VerificationEvent` schema is strictly enforced. The 8 core mark evidence columns represent the immutable audit trail for forensic compliance:

- `mark_match_status`
- `marks_detected_probe`
- `marks_detected_gallery`
- `mark_lrs_json`
- `accepted_mark_correspondences_json`
- `mark_detector_version`
- `mark_matcher_version`
- `mark_overlay_url`

All endpoints (`/verify/fuse` and `/vault/search`) serialize to this identical ledger structure.

## 6. CI/CD Validation & Production Constraints

- **No Local Execution**: The Verification algorithm relies entirely on hosted Cloud Run environments.
- **Hosted Pipeline is Truth**: The hosted GitHub Actions and Cloud Run CI/CD pipeline represent the absolute source of truth.
- **Forensic Telemetry Toggles**: Deep debugging telemetry is activated strictly via `DEBUG_FORENSIC=true`.

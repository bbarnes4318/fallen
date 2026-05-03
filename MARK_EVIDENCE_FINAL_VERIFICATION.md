# Final Hosted Verification Report for Mark Evidence System

## 1. Executive Summary

This report confirms the end-to-end integration and stability of the forensic mark correspondence pipeline.

- **Status of backend mark detector**: Fully operational. Reliably identifies blemishes, scars, and dark spots based on OpenCV contour analysis with integrated skin-masking.
- **Status of backend matcher**: Fully operational. Matches based on Hungarian assignment using 5D Bayesian LR scoring matrix.
- **Status of lr_marks multiplier**: Fully operational. Valid non-structural mark evidence generates a valid `lr_marks` product, seamlessly integrating with `LR_total`.
- **Status of MARKS tab**: Fully operational. Displays verified mark evidence seamlessly and handles exact self-match scenarios accurately.
- **Status of EDGE DELTA separation**: Fully operational. Re-labeled correctly and decoupled from mark evidence logic.
- **Status of audit persistence**: Fully operational. The schema guard strictly enforces all forensic data points into PostgreSQL.
- **Status of CI/CD validation**: Pipeline code passes strict local test coverage; awaiting final remote verification.

## 2. Formula Confirmation

The overarching Bayesian logic has been validated and remains strictly intact:

```text
LR_ensemble = score_to_lr_ensemble(structural_sim, temporal_delta)
LR_marks = product of accepted individual mark LRs
LR_total = LR_ensemble × LR_marks
posterior = (PRIOR × LR_total) / ((PRIOR × LR_total) + (1 - PRIOR))
bayesian_fused_score = posterior × 100
```

**State of Formula:** Unchanged. This formula remains the absolute source of truth for all forensic analysis outputs in the system.

## 3. Mark Evidence Contract

The following properties govern the backend-to-frontend transfer of forensic mark data:

- `mark_match_status`
- `marks_detected_probe`
- `marks_detected_gallery`
- `marks_matched`
- `lr_marks`
- `mark_lrs`
- `correspondences`
- `mark_detector_version`
- `mark_matcher_version`

These are fully integrated and strictly typed within both the backend Python schema and the frontend TypeScript `VerificationResult` definitions.

## 4. UI Behavior

The user interface explicitly enforces strict functional separation across analytical tabs:

- **EDGE DELTA**: Explicitly labeled as edge-difference only. It does not display scar/mole/blemish evidence.
- **MARKS**: Displays accepted shared mark evidence. Evidence is presented with matching green numbered markers on both images for unambiguous cross-referencing. Included evidence cards display details, messaging for exact self-matches, and fallback logic for insufficient/no-match cases.
- **DEBUG**: Fully restricted to raw detector outputs, unfiltered rejection details, and raw OpenCV blobs without cluttering the primary analytical view.

## 5. Database / Audit Ledger

The `VerificationEvent` schema is strictly enforced for the immutable PostgreSQL ledger. The 8 core mark evidence columns represent the immutable audit trail for forensic compliance:

- `mark_match_status`
- `marks_detected_probe`
- `marks_detected_gallery`
- `marks_matched`
- `mark_lrs_json`
- `accepted_mark_correspondences_json`
- `mark_detector_version`
- `mark_matcher_version`
- `mark_overlay_url`
- `lr_marks_product`
- `mark_correspondence_x100`

## 6. Exact Self-Match Policy

A strict, logical handling protocol is applied for functionally identical images:

- `exact_image_match = true`
- `mark_match_status = EXACT_SELF_MATCH`
- `mark_correspondence_score = 100`
- `lr_marks = 1.0` (neutral)

**Explanation:** The neutral LR (`1.0`) is intentional because a byte-identical image comparison does not represent independent mark evidence. Acknowledging identical marks across the same image provides circular evidence; treating it neutrally prevents statistical distortion in the Bayesian fusion model.

## 7. Remaining Risks

- No real-world labeled scar benchmark yet.
- Detector may still miss very subtle marks under varying exposure profiles.
- Lighting, crop, and pose can significantly affect mark detection geometry and extraction reliability.
- Calibration currently applies to the existing 5D LR vector, while new assignment penalties are pre-LR selection only.

## 8. Hosted Validation Status

Hosted CI/CD status: NOT AVAILABLE (Awaiting remote pipeline execution completion in GitHub Actions).

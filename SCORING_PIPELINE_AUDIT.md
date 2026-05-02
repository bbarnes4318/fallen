# SCORING_PIPELINE_AUDIT.md
## Forensic Audit Report — Bayesian Scoring Pipeline

**Audit Date:** 2026-05-02
**Pipeline Version:** v4.0 (Scientific Bayesian LR Framework)
**Auditor:** Fallen Engineering

---

## 1. Live Scoring Formula

The production pipeline uses a multiplicative Bayesian Likelihood Ratio framework:

```python
lr_ensemble = score_to_lr_ensemble(structural_sim, temporal_delta)
lr_marks    = mark_result.get("lr_marks", 1.0)
lr_total    = lr_ensemble * lr_marks
posterior   = (PRIOR * lr_total) / ((PRIOR * lr_total) + (1 - PRIOR))
bayesian_fused_score = posterior * 100.0
```

**Prior:** `PRIOR = 0.5` (neutral, agnostic)

**Simplification:** With Prior = 0.5, posterior = LR_total / (LR_total + 1)

---

## 2. Functions and Files

| Function / Component | File | Line (approx) | Status |
|---|---|---|---|
| `score_to_lr_ensemble()` | `backend/main.py` | ~607 | ✅ ACTIVE |
| `compute_mark_correspondence()` | `backend/main.py` | ~1800 | ✅ ACTIVE |
| `finite_or_none()` | `backend/main.py` | ~591 | ✅ NEW |
| Mark Override Protocol | `backend/main.py` | ~2337, ~2910 | ✅ NEW |
| `scoring_trace` builder | `backend/main.py` | ~2538, ~3098 | ✅ NEW |
| `VerificationResponse` model | `backend/main.py` | ~415 | ✅ EXTENDED |
| `VerificationEvent` ledger | `backend/models.py` | ~84 | ✅ EXTENDED |
| `ScoringTrace` interface | `frontend/types/verification.ts` | ~117 | ✅ NEW |
| `VerificationResult` interface | `frontend/types/verification.ts` | ~139 | ✅ EXTENDED |
| Bayesian Debug Panel | `frontend/components/SymmetryMerge.tsx` | ~738 | ✅ NEW |

---

## 3. Prior Commit Features — Status

| Commit | Feature | Status |
|---|---|---|
| `b25327f` | RetinaFace backend / correct ArcFace alignment | ✅ ACTIVE |
| `6e3a40a` | Mark override protocol | 🔧 WAS MISSING → NOW IMPLEMENTED |
| — | Bayesian LR formula | ✅ ACTIVE (confirmed identical) |
| — | `score_to_lr_ensemble()` calibration | ✅ ACTIVE with arcface fallback |
| — | `compute_mark_correspondence()` Tier 4 | ✅ ACTIVE |
| — | Temporal delta conditioning | ✅ ACTIVE |

---

## 4. Issues Found and Fixed

### A. Mark Override Protocol (HIGH — FIXED)
**Problem:** The ArcFace veto at `structural_sim < 0.40` was unconditional. Commit `6e3a40a` specified a mark-override pathway, but it was never implemented.

**Fix:** Implemented v1.0 mark override protocol with dual-gate threshold:
- 3+ individual mark LRs > 1.0
- Aggregate `lr_marks >= 100.0`
- Override lifts the hard-zero but does NOT claim ArcFace passed

### B. Scoring Trace (MEDIUM — FIXED)
**Problem:** No structured debug telemetry in the response JSON.

**Fix:** `scoring_trace` dict added to response when `DEBUG_FORENSIC=true`. Contains all LR components, calibration status, veto state, and mark override eligibility with safe serialization via `finite_or_none()`.

### C. METHODOLOGY.md (HIGH — FIXED)
**Problem:** Section 6 documented `fused_score = w1*tier1 + w2*tier2 + w3*tier3` — dead code formula.

**Fix:** Rewrote Section 6 to document the live Bayesian LR framework, including mark override protocol and veto behavior.

### D. Dead Code (LOW — FIXED)
**Problem:** `base_fused_score` was computed using the legacy weighted-average formula but never used.

**Fix:** Removed from both `/verify/fuse` and `/vault/search` endpoints.

### E. Calibration Status (MEDIUM — FIXED)
**Problem:** Calibration state was not surfaced to the frontend.

**Fix:** `calibration_status` field added to `VerificationResponse` and persisted to the immutable audit ledger.

### F. Frontend Debug Panel (MEDIUM — FIXED)
**Problem:** No Bayesian scoring visibility in the UI.

**Fix:** Added `BAYESIAN SCORING TRACE` panel in `SymmetryMerge.tsx`, visible only in forensic debug mode. Shows all LR components, calibration status, veto state, and mark override status.

---

## 5. Scoring Trace Shape

```typescript
interface ScoringTrace {
  calibration_status: string;       // "LOADED" | "MISSING"
  calibration_source: string;
  calibration_benchmark: string;
  tier4_calibration_status: string;
  lr_ensemble_raw: number | null;
  lr_marks_raw: number | null;
  lr_total_raw: number | null;
  lr_ensemble_display: string;      // Scientific notation "{:.6e}"
  lr_marks_display: string;
  lr_total_display: string;
  posterior_raw: number | null;
  fused_score_pre_veto: number | null;
  fused_score_post_veto: number | null;
  veto_triggered: boolean;
  veto_reason: string | null;       // "ARCFACE_VETO" | "ARCFACE_VETO_MARK_OVERRIDE"
  veto_override_applied: boolean;
  veto_override_reason: string | null;
  mark_override_eligible: boolean;
  temporal_delta_years: number | null;
  ensemble_thresholds_key: string;  // "ensemble" | "arcface" | "NONE"
}
```

**Visibility:** Response JSON only when `DEBUG_FORENSIC=true`.

**Immutable Ledger (always persisted):** `lr_arcface`, `lr_marks_product`, `lr_total`, `posterior_probability`, `bayesian_fused_score_x100`, `marks_matched`, `calibration_status`, `veto_reason`, `veto_override_applied`.

---

## 6. Test Summary

| Test | Description | Status |
|---|---|---|
| `test_bayesian_identity()` | Posterior = LR/(LR+1) for Prior=0.5 | ✅ PASS |
| `test_veto_without_override()` | ArcFace < 0.40, marks < 3 → score=0, preserves pre-veto | ✅ PASS |
| `test_veto_with_mark_override()` | ArcFace < 0.40, 3+ positive marks, LR≥100 → override | ✅ PASS |
| `test_calibration_missing()` | CALIBRATION=None → status=MISSING | ✅ PASS |
| `test_mark_override_safeguards()` | Single extreme LR, low aggregate, malformed values | ✅ PASS |
| `main()` (E2E) | Full pipeline with real LFW images | ✅ PASS (requires dataset) |

---

## 7. Files Modified

| File | Change Type |
|---|---|
| `backend/main.py` | MODIFIED — mark override, scoring_trace, dead code removal, safe serialization |
| `backend/models.py` | MODIFIED — extended VerificationEvent ledger schema |
| `frontend/types/verification.ts` | MODIFIED — ScoringTrace interface, new VerificationResult fields |
| `frontend/components/SymmetryMerge.tsx` | MODIFIED — Bayesian Scoring Trace debug panel |
| `METHODOLOGY.md` | MODIFIED — rewrote Section 6 (Bayesian LR framework) |
| `test_e2e_bayesian.py` | MODIFIED — 5 new unit tests for override protocol |
| `SCORING_PIPELINE_AUDIT.md` | NEW — this document |

# Court-Survivable Forensic Scoring Refactor v2 — Implementation Audit

**Audit Date:** 2026-05-03T08:54:00-04:00
**Current Commit SHA:** `266e19f48f21e2fc08c79eb5a351b4ce10840278`
**Commit Message:** `chore: bump version to 1.141.0 for deployment`

---

## 1. Search Results — Required v2 Terms

| # | Required v2 Term | Status | Files Found |
|---|---|---|---|
| 1 | `mark_diagnostics` | **EXISTS** | `backend/main.py` (model field L558, payload build L2944/L3662, response wire L3068/L3786), `frontend/types/verification.ts` (L231), `frontend/app/page.tsx` (L1208–L1279), `frontend/components/SymmetryMerge.tsx` (L717–L732) |
| 2 | `rejection_summary` | **EXISTS** | `backend/main.py` (builder fn L392, payload wire L2953/L3671), `frontend/types/verification.ts` (L164), `frontend/app/page.tsx` (L1273–L1279), `frontend/components/SymmetryMerge.tsx` (L732) |
| 3 | `/marks/analyze` | **MISSING** | No endpoint found. No `@app.post` or `@app.get` route matching `marks` or `analyze` exists in `backend/main.py`. |
| 4 | `SHARED MARK EVIDENCE` | **EXISTS — page.tsx ONLY** | `frontend/app/page.tsx` (L1203 comment, L1206 UI label). **NOT present** in `frontend/components/SymmetryMerge.tsx`. |
| 5 | `MarkDiagnostics` | **EXISTS** | `frontend/types/verification.ts` (L155 interface definition, L231 usage) |
| 6 | `raw_arcface_similarity` | **EXISTS — model/backend only** | `backend/main.py` (model L584, wire L3086), `frontend/types/verification.ts` (L250). **NOT rendered** in any frontend component — no usage in `page.tsx` or `SymmetryMerge.tsx`. |
| 7 | `raw_secondary_similarity` | **EXISTS — model/backend only** | `backend/main.py` (model L585, wire L3087), `frontend/types/verification.ts` (L251). **NOT rendered** in any frontend component. |
| 8 | `fused_face_model_similarity` | **EXISTS — model/backend only** | `backend/main.py` (model L586, wire L3088), `frontend/types/verification.ts` (L252). **NOT rendered** in any frontend component. |
| 9 | `lr_face_model` | **EXISTS — model/backend only** | `backend/main.py` (model L587, wire L3089), `frontend/types/verification.ts` (L253). **NOT rendered** in any frontend component. |
| 10 | `probe_aligned_crop_hash_pre_clahe` | **EXISTS — model only, NOT populated** | `backend/main.py` (AuditLog model field L513), `frontend/types/verification.ts` (L31). No assignment line `probe_aligned_crop_hash_pre_clahe=` found in backend pipeline code. Field declared but never written. |
| 11 | `probe_aligned_crop_hash_post_clahe` | **EXISTS — model only, NOT populated** | `backend/main.py` (AuditLog model field L515), `frontend/types/verification.ts` (L33). No assignment found. Field declared but never written. |
| 12 | `gallery_aligned_crop_hash_pre_clahe` | **EXISTS — model only, NOT populated** | `backend/main.py` (AuditLog model field L514), `frontend/types/verification.ts` (L32). No assignment found. Field declared but never written. |
| 13 | `gallery_aligned_crop_hash_post_clahe` | **EXISTS — model only, NOT populated** | `backend/main.py` (AuditLog model field L516), `frontend/types/verification.ts` (L34). No assignment found. Field declared but never written. |
| 14 | `FACE EMBEDDING ANALYSIS` | **EXISTS — page.tsx ONLY** | `frontend/app/page.tsx` (L1297). **NOT present** in `frontend/components/SymmetryMerge.tsx`. |
| 15 | `BAYESIAN POSTERIOR PROBABILITY` | **EXISTS — page.tsx ONLY** | `frontend/app/page.tsx` (L1176). **NOT present** in `frontend/components/SymmetryMerge.tsx`. |
| 16 | `FACE MODEL VETO TRIGGERED` | **EXISTS** | `frontend/app/page.tsx` (L1438), `frontend/components/SymmetryMerge.tsx` (L546, L553 — partial text variants `FACE MODEL VETO` present in verdict/pre-veto rows). |

---

## 2. Summary of Missing / Incomplete Items

### Fully Missing
- **`/marks/analyze` endpoint** — No dedicated standalone mark analysis endpoint exists anywhere in the backend.

### Declared But Not Populated (Dead Fields)
- `probe_aligned_crop_hash_pre_clahe` — model field exists, never assigned in pipeline
- `probe_aligned_crop_hash_post_clahe` — model field exists, never assigned in pipeline
- `gallery_aligned_crop_hash_pre_clahe` — model field exists, never assigned in pipeline
- `gallery_aligned_crop_hash_post_clahe` — model field exists, never assigned in pipeline

> **Note:** The non-CLAHE variants (`probe_aligned_crop_hash`, `gallery_aligned_crop_hash`) ARE populated at L2878–L2879 and L3599–L3600 of `backend/main.py`, but these are vector hashes, not image hashes at the pre/post CLAHE preprocessing stage. The four CLAHE-granular fields are provenance gaps.

### Exists in page.tsx but NOT in SymmetryMerge.tsx
- `SHARED MARK EVIDENCE` panel — present in `page.tsx` Intelligence Panel only; `SymmetryMerge.tsx` has its own marks-mode card but does not replicate the "SHARED MARK EVIDENCE" first-class panel.
- `FACE EMBEDDING ANALYSIS` label — `page.tsx` only
- `BAYESIAN POSTERIOR PROBABILITY` label — `page.tsx` only

### Backend Fields That Exist But Are Never Rendered on Frontend
- `raw_arcface_similarity` — wired from backend but never displayed in any UI component
- `raw_secondary_similarity` — wired from backend but never displayed
- `fused_face_model_similarity` — wired from backend but never displayed
- `lr_face_model` — wired from backend but never displayed

### Test Infrastructure
- No `test_e2e_bayesian.py` or any E2E test file exists in the repository. The `backend/tests/` directory does not exist. Only `backend/test_security_helpers.py` exists as a test file.

---

## 3. Files That Need Modification

| File | Required Changes |
|---|---|
| `backend/main.py` | Add `/marks/analyze` endpoint; populate pre/post CLAHE hash fields in both verification pipelines (1:1 and 1:N); wire `raw_arcface_similarity` / `raw_secondary_similarity` / `fused_face_model_similarity` / `lr_face_model` to audit log persistence |
| `backend/models.py` | Add columns for pre/post CLAHE hashes if DB persistence is required |
| `frontend/components/SymmetryMerge.tsx` | Add `SHARED MARK EVIDENCE` first-class diagnostic panel (currently only has marks-mode card); add `FACE EMBEDDING ANALYSIS` section; add `BAYESIAN POSTERIOR PROBABILITY` display; render `raw_arcface_similarity`, `raw_secondary_similarity`, `fused_face_model_similarity`, `lr_face_model` |
| `frontend/app/page.tsx` | Render the four face-model decomposition fields (`raw_arcface_similarity`, `raw_secondary_similarity`, `fused_face_model_similarity`, `lr_face_model`) in the Intelligence Panel breakdown section |
| `frontend/types/verification.ts` | No changes needed — all required types already declared |
| `backend/test_e2e_bayesian.py` | **NEW FILE** — E2E test suite for Bayesian mark pipeline |

---

## 4. Phase-by-Phase Implementation Checklist

### Phase 1: Backend Mark Diagnostics Hardening

- [ ] Verify `mark_diagnostics` payload includes all required fields in both 1:1 (`/verify/fuse`) and 1:N (`/verify/vault-sweep`) pipelines
- [ ] Populate `probe_aligned_crop_hash_pre_clahe` at the preprocessing stage BEFORE CLAHE equalization is applied
- [ ] Populate `probe_aligned_crop_hash_post_clahe` at the preprocessing stage AFTER CLAHE equalization is applied
- [ ] Populate `gallery_aligned_crop_hash_pre_clahe` at the corresponding gallery preprocessing stage
- [ ] Populate `gallery_aligned_crop_hash_post_clahe` at the corresponding gallery preprocessing stage
- [ ] Wire all four CLAHE hash fields into both the `AuditLog` response AND the `VerificationEvent` database persistence (if `models.py` columns are added)
- [ ] Ensure `raw_arcface_similarity`, `raw_secondary_similarity`, `fused_face_model_similarity`, and `lr_face_model` are wired into the `AuditLog` response object (currently wired into `VerificationResponse` at L3086–L3089, but verify they are also in the nested `audit_log` dict for persistence)

### Phase 2: `/marks/analyze` Endpoint

- [ ] Create new `POST /marks/analyze` endpoint in `backend/main.py`
- [ ] Accept two images (probe + gallery) or two GCS URIs
- [ ] Run the mark detection pipeline in isolation (no full verification)
- [ ] Return `mark_diagnostics` payload, raw mark arrays, correspondences, and `rejection_summary`
- [ ] Gate behind operator authentication
- [ ] Include `mark_detector_version` and `mark_matcher_version` in response

### Phase 3: Frontend Shared Mark Evidence Panel

- [ ] Add `SHARED MARK EVIDENCE` first-class panel to `SymmetryMerge.tsx` (currently only in `page.tsx`)
- [ ] Render `raw_arcface_similarity` in `FACE EMBEDDING ANALYSIS` section of `page.tsx` Intelligence Panel
- [ ] Render `raw_secondary_similarity` alongside ArcFace in the same section
- [ ] Render `fused_face_model_similarity` as the fused face-model score
- [ ] Render `lr_face_model` as the face-model Likelihood Ratio
- [ ] Add `BAYESIAN POSTERIOR PROBABILITY` display to `SymmetryMerge.tsx` (currently only in `page.tsx`)
- [ ] Ensure all rendered values handle `null` / `undefined` gracefully with "N/A" fallbacks

### Phase 4: SymmetryMerge MARKS Hardening

- [ ] Audit the SymmetryMerge marks-mode card (L714–L858) for completeness against the `MarkDiagnostics` interface
- [ ] Ensure `rejection_summary` is displayed in SymmetryMerge marks card (currently rendered at L777)
- [ ] Verify `mark_match_status` transitions (`EXACT_SELF_MATCH`, `MATCHED`, `INSUFFICIENT_MARKS`, `NO_MATCHES`, `DETECTOR_UNAVAILABLE`) are all handled with correct labels and colors
- [ ] Validate that mark overlay circles (probe/gallery) correctly render matched vs. unmatched states using the `ForensicPoint.isMatched` flag
- [ ] Ensure `FACE MODEL VETO` language in SymmetryMerge (L546, L553) is consistent with the `FACE MODEL VETO TRIGGERED` language in `page.tsx` (L1438)

### Phase 5: Forensic Language and Audit Provenance

- [ ] Verify all forensic language strings are consistent across `page.tsx` and `SymmetryMerge.tsx`:
  - "FACE MODEL VETO TRIGGERED" vs "FACE MODEL VETO" (partial mismatch)
  - "BAYESIAN POSTERIOR PROBABILITY" label exists in `page.tsx` but not `SymmetryMerge.tsx`
  - "SHARED MARK EVIDENCE" exists in `page.tsx` but not `SymmetryMerge.tsx`
  - "FACE EMBEDDING ANALYSIS" exists in `page.tsx` but not `SymmetryMerge.tsx`
- [ ] Ensure the pre/post CLAHE hashes are displayed in the Technical Breakdown overlay (L1484–L1587 of `page.tsx`) once populated
- [ ] Confirm `mark_overlay_url` from GCS receipt generation (commit `42bbb6a`) is surfaced in the audit trail
- [ ] Verify `code_commit_hash`, `docker_image_digest`, `arcface_weight_hash`, and other provenance fields in `AuditLog` are actually populated at runtime (model fields exist but population not verified in this audit)

### Phase 6: Tests and Verification

- [ ] Create `backend/test_e2e_bayesian.py` — E2E test suite
- [ ] Test: `/verify/fuse` returns `mark_diagnostics` with all required fields
- [ ] Test: `/verify/fuse` returns `raw_arcface_similarity`, `raw_secondary_similarity`, `fused_face_model_similarity`, `lr_face_model`
- [ ] Test: `mark_diagnostics.rejection_summary` is populated correctly for each mark state
- [ ] Test: Pre/post CLAHE hashes are non-null when images are processed
- [ ] Test: `/marks/analyze` endpoint returns valid diagnostics
- [ ] Test: Exact self-match detection triggers `EXACT_SELF_MATCH` status and neutralizes `lr_marks` to 1.0
- [ ] Test: Veto override logic (`evaluate_mark_veto_override`) correctly gates on 3+ positive mark LRs and aggregate >= 100.0
- [ ] Test: `mark_lrs` array in response matches individual correspondence LRs
- [ ] Verify CI/CD pipeline passes with all new code (hosted, not local — per project rules)

---

## 5. Verdict

The previous deployment summary claiming "Court-Survivable Forensic Scoring Refactor v2 complete" was **inaccurate**. The visible commit range (`42bbb6a`..`266e19f`) added:

1. Mark overlay GCS receipt generation (`42bbb6a`)
2. A version bump to 1.141.0 (`266e19f`)

The following v2 requirements **do not exist** in the codebase:

- `/marks/analyze` endpoint — **completely absent**
- Pre/post CLAHE hash population — **fields declared but never assigned**
- Face-model decomposition fields — **wired from backend but never rendered on frontend**
- `SHARED MARK EVIDENCE` in SymmetryMerge — **absent** (only in `page.tsx`)
- E2E test suite — **no test file exists**

The `mark_diagnostics` infrastructure, `rejection_summary` builder, `MarkDiagnostics` TypeScript interface, and basic forensic language labels do exist and appear functional. These represent approximately **40-50%** of the v2 scope. The remaining work is concentrated in the `/marks/analyze` endpoint, CLAHE provenance chain, frontend rendering of face-model decomposition fields, SymmetryMerge panel parity, and test infrastructure.

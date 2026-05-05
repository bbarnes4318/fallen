# MARK PIPELINE FAILURE MAP

> Forensic audit of the facial mark detection pipeline.
> Produced: 2026-05-04
> Auditor: Antigravity (automated code audit)
> Scope: `backend/mark_detector.py`, `backend/main.py`, `frontend/components/SymmetryMerge.tsx`, `frontend/app/page.tsx`

---

## Summary

The mark detection pipeline has **7 confirmed failure points** and **1 architectural constraint violation** that collectively explain why production comparisons (e.g. Job ID 17: 86.2% score, 0 marks detected) return zero forensic mark evidence. The root cause is a **double CLAHE application** that washes out the luminance contrast the detector depends on, compounded by code triplication that makes consistent fixes impossible.

---

## F1: DOUBLE CLAHE — Primary Detection Killer

**Severity:** CRITICAL
**Impact:** Destroys subtle luminance contrast before detection channels can operate.

`main.py` applies `apply_clahe()` to the raw image **before** alignment. The CLAHE-enhanced image is then aligned and passed to `detect_facial_marks()`. Inside the detector, `_run_channels()` creates a **second** CLAHE instance and applies it again to the L channel (LAB) and grayscale.

```
main.py:2314     gallery_clahe = apply_clahe(gallery_img)        ← 1st CLAHE (full BGR→LAB→CLAHE→BGR)
main.py:2318     gallery_aligned, gallery_landmarks = align_face_crop(gallery_clahe)
main.py:2398     detect_facial_marks(gallery_aligned, ...)
                   ↓
mark_detector.py:281   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  ← 2nd CLAHE
mark_detector.py:282   L_clahe = clahe.apply(L)          ← Applied to already-CLAHE'd luminance
mark_detector.py:296   gray_clahe = clahe.apply(gray)     ← Applied to already-CLAHE'd grayscale
```

**Affected channels:** `dark_lesion` (L282-286), `bright_scar` (L289-293), `linear_scar_v2` (L296-309). The legacy channels (`dark`, `light`, `linear_scar`, `texture_cluster`) operate on raw grayscale of the aligned crop, which is ALSO already CLAHE'd from the first pass.

**Evidence files:**
- `backend/main.py` — `apply_clahe()` at line 1033, called at lines 2314-2315, 3057-3058, 3366
- `backend/mark_detector.py` — `_run_channels()` at line 249, internal CLAHE at lines 281-282, 296

**Verdict:** This is a bug. Double CLAHE must be eliminated. The detector should receive either raw-aligned OR single-CLAHE'd input, never double.

---

## F2: Alignment Inconsistency — Provenance Hash Mismatch

**Severity:** HIGH
**Impact:** Pre-CLAHE provenance hashes document pixels that were never used for detection.

The pipeline runs **two separate alignment passes** producing potentially different crops:

```
main.py:2308   gallery_pre_clahe_crop, _gpc_lm = align_face_crop(gallery_img)     ← Alignment A (raw)
main.py:2310   gallery_aligned_crop_hash_pre_clahe = compute_image_hash(...)
main.py:2314   gallery_clahe = apply_clahe(gallery_img)
main.py:2318   gallery_aligned, gallery_landmarks = align_face_crop(gallery_clahe)  ← Alignment B (CLAHE'd)
main.py:2322   gallery_aligned_crop_hash_post_clahe = compute_image_hash(...)
```

`align_face_crop()` (line 1118) calls `face_mesh.process()` internally. MediaPipe landmark detection is sensitive to contrast changes — CLAHE shifts intensity, which can shift detected eye centers, which changes the rotation angle, which changes the crop. **Alignment A and Alignment B can produce different 256×256 crops from the same source image.**

The pre-CLAHE hash therefore documents a crop that doesn't correspond to the actual detection input. The provenance chain is broken.

**Evidence files:**
- `backend/main.py` — `align_face_crop()` at line 1118, dual calls at lines 2308-2309 vs 2318-2319

**Fix constraint:** One canonical alignment transform must be computed once and applied consistently to all derivative crops.

---

## F3: Tripled Code — 3 Copy-Pasted Detection Blocks

**Severity:** HIGH
**Impact:** Impossible to fix mark detection consistently; any threshold change risks drift.

The full mark pipeline (detect → filter valid marks → self-match check → match → compute LR → build diagnostics) is copy-pasted at three locations:

| Endpoint | Detection call | Post-processing block |
|---|---|---|
| Verification (`/verify`) | `main.py:2398-2399` | Lines ~2397-2780 |
| Mark Analyze (`/marks/analyze`) | `main.py:3109-3110` | Lines ~3108-3310 |
| Vault Search (`/vault/search`) | `main.py:3530-3531` | Lines ~3529-3900+ |

Each block independently:
1. Calls `detect_facial_marks()` for both gallery and probe
2. Filters valid marks via occlusion mask check
3. Detects exact self-match via file hash comparison
4. Calls `match_facial_marks()` and `compute_mark_correspondence()`
5. Builds `mark_diagnostics` payload
6. Builds `lr_calculation_trace`
7. Builds debug overlays (if DEBUG_FORENSIC)

**Evidence files:**
- `backend/main.py` — three `detect_facial_marks` call pairs at lines 2398-2399, 3109-3110, 3530-3531

**Fix constraint:** Extract to a single shared function. Do NOT consolidate yet — document only.

---

## F4: Debug Threshold Reporting — Wrong Values

**Severity:** MEDIUM
**Impact:** Misleading diagnostic output for anyone debugging detection failures.

The `/marks/analyze` endpoint's `detector_thresholds` block (gated behind `DEBUG_FORENSIC`) reports **incorrect** values:

```python
# main.py:3292-3299 — REPORTED (WRONG)
result["detector_thresholds"] = {
    "min_contour_area": 8,        # Correct
    "max_contour_area": 500,      # Correct
    "min_contrast": 3.0,          # WRONG — actual is 1.5
    "min_overlap_ratio": 0.70,    # WRONG — actual is 0.50
    "adaptive_block_size": 15,    # Correct
    "adaptive_c": 5,              # WRONG — actual is 3
}
```

Actual values from `mark_detector.py`:

```python
# mark_detector.py:24-27 — ACTUAL
_MIN_AREA = 8
_MAX_AREA = 500
_MIN_OVERLAP_RATIO = 0.50
_MIN_CONTRAST = 1.5
```

**Evidence files:**
- `backend/main.py` — lines 3292-3299
- `backend/mark_detector.py` — lines 23-27
- Discrepancies: `min_contrast` (3.0 vs 1.5), `min_overlap_ratio` (0.70 vs 0.50), `adaptive_c` (5 vs 3)

---

## F5: Calibration Short-Circuit

**Severity:** HIGH
**Impact:** When calibration data is unavailable, mark evidence produces zero forensic value.

```python
# main.py:1811-1816
if TIER4_CALIBRATION is None:
    return {
        "score": None, "matched": 0,
        "total_gallery": n_gal, "total_probe": n_pro,
        "matches": [], "lr_marks": 1.0, "mark_lrs": [],
    }
```

When `TIER4_CALIBRATION` is `None`:
- `compute_mark_correspondence()` returns `lr_marks=1.0` (neutral) regardless of detected marks
- `matched` is forced to 0 even if the matcher found pairs
- All individual mark LRs are discarded

The calibration file (`calibration_data/tier4_population_model.pkl`, 12MB) exists locally and loads from GCS as fallback. Load failure is silent except for a print statement.

**Evidence files:**
- `backend/main.py` — `_load_tier4_calibration()` at line 669, assignment at line 712
- `backend/main.py` — `compute_mark_correspondence()` at line 1779, null check at line 1811
- `backend/calibration_data/tier4_population_model.pkl` — exists (12MB)

---

## F6: Skin Mask Erosion vs. Overlap Threshold

**Severity:** MEDIUM
**Impact:** Marks near jawline, hairline, and temples are systematically rejected.

The skin mask construction applies an 11×11 elliptical erosion after building the face oval polygon:

```python
# mark_detector.py:54-55
ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skin_mask = cv2.erode(skin_mask, ek, iterations=1)
```

Then the overlap test requires 50% of the mark contour to fall inside this **already-eroded** mask:

```python
# mark_detector.py:366-369
op = np.count_nonzero(cv2.bitwise_and(mm, skin_mask))
if op / cp < min_overlap:   # min_overlap = 0.50
    rej = f"insufficient_face_overlap ({op/cp:.2f})"
```

On a 256×256 crop, the 11×11 erosion removes ~5px from every edge of the face oval. Marks that overlap the face boundary — exactly where forensically significant scars and birthmarks frequently appear (temple, jawline, ear region) — will have <50% overlap and be rejected.

The fallback path (min_overlap=0.30) partially compensates but caps LR at 3.0, making those marks forensically near-useless.

**Evidence files:**
- `backend/mark_detector.py` — erosion at line 55, overlap test at lines 366-369
- `backend/mark_detector.py` — fallback thresholds at lines 29-33

---

## F7: No Per-Channel Stage Diagnostics

**Severity:** MEDIUM
**Impact:** Impossible to determine which detection channel is failing without DEBUG_FORENSIC enabled.

The `mark_diagnostics` payload includes aggregate counts but does NOT break down per-channel candidate production:

```python
# mark_detector.py:413-424 (trace dict)
trace = {
    "initial_candidates": sum(len(c) for c in channels.values()),  # Aggregate only
    "dark_lesion_initial_candidates": len(channels.get("dark_lesion", [])),
    "bright_scar_initial_candidates": len(channels.get("bright_scar", [])),
    # ...but NO per-channel pass/fail at area, shape, region, contrast stages
}
```

The `_filter_contours()` function tracks stage counts (`area_pass`, `shape_pass`, `region_pass`, `contrast_pass`) but only as **aggregate** totals across all channels — not per-channel.

When production returns 0 marks, we cannot distinguish between:
- "dark_lesion found 50 candidates but all failed contrast" vs.
- "dark_lesion found 0 candidates because thresholding produced nothing"

**Evidence files:**
- `backend/mark_detector.py` — `_filter_contours()` at line 328, aggregate trace at lines 413-424
- `backend/main.py` — `mark_diagnostics` payload at lines 3204-3224

---

## F8: Resolution Constraint Violation — 256×256 Is Insufficient

**Severity:** HIGH
**Impact:** Small marks (moles, freckles, fine scars) are below the pixel threshold at 256×256.

`align_face_crop()` produces a 256×256 canonical crop:

```python
# main.py:1118
def align_face_crop(image: np.ndarray, target_size: int = 256):
```

At 256×256, `_MIN_AREA = 8` pixels means a mark must be at least ~3×3 pixels. On a typical portrait where the face occupies 70% of frame, a 2mm mole maps to approximately 1-2 pixels at 256×256 — below the detection threshold.

At 768×768 (minimum) or 1024×1024 (preferred), the same 2mm mole maps to 4-8 pixels, comfortably above the area threshold.

**Evidence files:**
- `backend/main.py` — `align_face_crop()` default `target_size=256` at line 1118
- `backend/mark_detector.py` — `_MIN_AREA = 8` at line 24

---

## Architectural Constraints (Mandated)

The following constraints are **non-negotiable** for any future implementation:

| Constraint | Rationale |
|---|---|
| **No double CLAHE.** Eliminate the bug. Do not preserve backward compatibility. | Double CLAHE is a bug, not a feature. |
| **Mark detection at 768×768 minimum, 1024×1024 preferred.** | 256×256 is insufficient for small marks. |
| **One canonical alignment transform.** Compute once, apply to all derivatives. | Prevents provenance hash mismatch (F2). |
| **No generative face restoration (GFPGAN, CodeFormer, etc.).** | Can hallucinate marks, destroying forensic integrity. |
| **No job_id support on `/marks/analyze` yet.** | Keeps endpoint standalone and DB-free. |
| **No frontend changes until backend stabilizes.** | Frontend already has the rendering infrastructure. |
| **No tripled-code consolidation yet.** | Document first, consolidate after detector is validated. |

---

## Detection Channel Inventory

| Channel | Source | Method | Lines (mark_detector.py) |
|---|---|---|---|
| `dark` | Legacy | Adaptive threshold on grayscale (inverse) | 254-256 |
| `light` | Legacy | Adaptive threshold on inverted grayscale | 259-261 |
| `linear_scar` | Legacy | Canny edge detection | 264-268 |
| `texture_cluster` | Legacy | Bilateral filter absdiff | 271-274 |
| `dark_lesion` | v2.1.0 | LAB L-channel CLAHE + median blur + Otsu | 279-286 |
| `bright_scar` | v2.1.0 | LAB L-channel CLAHE + percentile threshold | 289-293 |
| `linear_scar_v2` | v2.1.0 | Scharr gradient + directional morphology | 296-309 |
| `texture_anomaly` | v2.1.0 | Local variance z-score > 2.5 | 312-322 |

**Note:** Legacy channels operate on `gray` (grayscale of the aligned crop). v2.1.0 channels operate on `L_clahe` / `gray_clahe` (internally CLAHE'd). With the double-CLAHE bug (F1), the v2.1.0 channels receive **triple**-normalized luminance: raw → main.py CLAHE → mark_detector.py CLAHE.

---

## Filter Pipeline (per contour)

```
Contour from channel
  → Area check: _MIN_AREA (8) ≤ area ≤ _MAX_AREA (500)
    → Shape/descriptor build: moments, circularity, bbox, eccentricity
      → Centroid bounds check: within image, not in used_mask
        → Skin mask check: centroid must be inside face oval
          → Border check: not within 10px of crop edge
            → Overlap check: ≥50% of contour pixels inside eroded skin mask
              → Contrast check: ≥1.5 absolute intensity difference from local mean
                → ACCEPT as mark candidate
```

If strict pass yields 0 marks, fallback runs with relaxed thresholds:
- `_FB_MIN_AREA = 4`, `_FB_MIN_OVERLAP = 0.30`, `_FB_MIN_CONTRAST = 0.5`
- Fallback marks capped at 15, LR capped at 3.0, confidence halved

---

## Call Graph

```
/verify (or /vault/search or /marks/analyze)
  → fetch_image_from_url() or _decode_b64_image()
  → apply_clahe()                          ← 1st CLAHE
  → align_face_crop(clahe_image)           ← Alignment + MediaPipe landmarks
  → detect_facial_marks(aligned, landmarks)
      → _build_skin_mask()                 ← Face oval polygon + erosion + feature exclusion
      → _run_channels()                    ← 8 detection channels, internal 2nd CLAHE
      → _filter_contours() [strict]        ← Area/shape/region/contrast filtering
      → _filter_contours() [fallback]      ← If strict yields 0
      → _dedup_candidates()               ← Cross-channel NMS
  → match_facial_marks()                   ← Hungarian assignment (scipy)
  → compute_mark_correspondence()          ← Bayesian LR per matched pair
  → _build_rejection_summary()             ← Human-readable explanation
```

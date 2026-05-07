# LFW Baseline Validation — Limitations

## What the LFW Baseline Proves

The Labeled Faces in the Wild (LFW) dataset is a standard academic benchmark for unconstrained face verification. It provides **truthfully labeled** same-person and different-person image pairs collected from news photos.

When used with the validation framework, LFW pairs confirm:

1. **Basic embedding separability** — Whether ArcFace + Facenet512 ensemble cosine similarity produces higher scores for genuine pairs than impostor pairs.
2. **Threshold operating point** — The FAR / FRR tradeoff at specific cosine similarity thresholds (0.30, 0.40, 0.50, 0.60, etc.).
3. **Equal Error Rate (EER)** — The operating point where FAR = FRR, providing a single-number summary of discriminative ability.
4. **AUC** — Area Under the ROC Curve, measuring overall separability.
5. **Safety rule impact** — Whether the current `structural_sim < 0.40` hard veto rule correctly rejects impostors without incorrectly rejecting genuine pairs.

## What the LFW Baseline Does NOT Prove

> [!WARNING]
> LFW alone is insufficient to justify any changes to production thresholds, safety rules, or mark evidence math.

### 1. Facial Mark / Scar / Crater Evidence

LFW does not contain annotated facial marks. The dataset was designed for face **recognition**, not **forensic identification**. Validation of the mark evidence pipeline (mark detection, matching, LR calculation, overcounting) requires a separate curated dataset with:
- Subjects with visible marks, scars, moles, acne, or craters
- Multiple images of the same subject showing mark consistency
- Different subjects with coincidentally similar mark patterns

**Status:** Not covered by LFW. Requires manual curation.

### 2. Twins and High-Similarity Imposters

LFW mismatched pairs are **randomly sampled** from different identities. They do not specifically target challenging cases where two different people look very similar (identical twins, siblings, doppelgängers). The False Accept Rate measured on random LFW pairs will **underestimate** the FAR against deliberately similar imposters.

**Status:** Not covered by LFW. Requires a dedicated twin/sibling dataset or manual curation.

### 3. Age Gap / Temporal Invariance

LFW images are mostly from a narrow time window (2002-2007 news photos). They do not test whether the pipeline correctly identifies the same person across significant aging (10+ years). The temporal invariance curve in the production pipeline is not validated by LFW.

**Status:** Not covered by LFW. Requires a cross-age dataset (e.g., MORPH, CACD, or manually curated pairs).

### 4. Extreme Lighting and Pose Variation

While LFW includes some natural lighting and pose variation, it does not specifically benchmark extreme cases (strong side-lighting, heavy shadow, extreme angles, partial occlusion). LFW images are predominantly frontal news photos.

**Status:** Partially covered. LFW provides some variation but not targeted extreme cases.

### 5. Different-Person Lookalikes

LFW mismatched pairs are random. The dataset does not curate pairs of different people who look unusually similar. A lookalike-specific evaluation requires manual identification of visually similar but genetically distinct individuals.

**Status:** Not covered by LFW.

### 6. Production Safety Rule Optimization

The hard veto at `structural_sim < 0.40` was chosen heuristically. LFW can show **how many LFW pairs** fall above or below this threshold, but LFW's population distribution does not match real production input (which includes user-uploaded selfies, documents, and highly variable image quality).

**Status:** LFW provides directional evidence only. Production threshold optimization requires a holdout set drawn from actual production image quality distributions.

## Minimum Viable Validation Matrix

| Category | LFW Covers? | Minimum Pairs Needed | Next Source |
|----------|-------------|---------------------|-------------|
| `same_person_normal` | ✅ Yes | 50+ (available) | LFW matched pairs |
| `same_person_age_gap` | ❌ No | 25 | MORPH, CACD, or manual |
| `same_person_lighting_pose` | ⚠️ Partial | 25 | LFW subset + manual tag |
| `different_person_random` | ✅ Yes | 50+ (available) | LFW mismatched pairs |
| `different_person_lookalike` | ❌ No | 25 | Manual curation |
| `mark_heavy_same_person` | ❌ No | 25 | Manual curation |
| `mark_heavy_different_person` | ❌ No | 25 | Manual curation |
| `twins_or_high_similarity_imposters` | ❌ No | 10+ | Twin dataset or manual |

## Decision Rules

> [!IMPORTANT]
> No production math, threshold, or safety rule changes should be made based on LFW results alone.

1. **LFW results are necessary but not sufficient.** If the pipeline fails on LFW, it has a fundamental problem. If it passes on LFW, it may still fail on harder cases.
2. **LFW AUC / EER establishes a floor**, not a ceiling. Production accuracy depends on the actual input distribution.
3. **Mark evidence and safety rule changes** require category-specific evaluation data that LFW does not provide.
4. **The hard veto policy** should not be removed or softened until twin/lookalike and mark-heavy evaluations demonstrate an acceptable false acceptance rate under the proposed alternative.

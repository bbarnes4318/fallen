# Mark-Heavy Dataset Strategy

**Phase**: 3A planning document (no code execution, no image uploads, no ingestion)  
**Status**: Draft  
**Purpose**: Define a curated validation dataset that oversamples distinctive mark types for robust Phase 3B+ experiments.

---

## Manifest Format

Use the same CSV format as `validation_pairs_lfw_50_50_sample.csv`, extended with mark-specific columns:

```
pair_id,image1_url_or_gcs_path,image2_url_or_gcs_path,label_same_person,category,notes,expected_challenge_type,primary_mark_type,secondary_mark_type,mark_region,manual_label_required,quality_notes
```

### Column definitions

| Column | Required | Description |
|---|---|---|
| `pair_id` | Yes | Unique identifier, e.g. `mh_same_001` |
| `image1_url_or_gcs_path` | Yes | GCS path to first image |
| `image2_url_or_gcs_path` | Yes | GCS path to second image |
| `label_same_person` | Yes | `true` or `false` |
| `category` | Yes | One of the categories below |
| `notes` | Yes | Human-readable description |
| `expected_challenge_type` | Yes | `mark_heavy`, `age_gap`, `lookalike`, `twins`, `baseline` |
| `primary_mark_type` | Yes | Dominant mark type expected: `dark_spot`, `depression_scar`, `linear_scar`, `dark_mole`, `mole`, `mixed` |
| `secondary_mark_type` | Optional | Second most common mark type if mixed |
| `mark_region` | Optional | Primary facial region of interest |
| `manual_label_required` | Yes | `true` if human annotation still needed, `false` if verified |
| `quality_notes` | Optional | Resolution, lighting, pose observations |

---

## Minimum Pair Counts

| Category | Code | Min Pairs | Priority |
|---|---|---|---|
| **Same-person, mark-heavy** | `mark_heavy_same_person` | 50 | Critical |
| — dark_spot dominant | | 15 | High |
| — depression_scar / structural_crater | | 10 | High |
| — linear_scar / scar | | 10 | High |
| — dark_mole / mole | | 10 | Medium |
| — mixed distinctive types | | 5 | Medium |
| **Same-person, age-gap** | `same_person_age_gap` | 20 | High |
| **Different-person, mark-heavy impostors** | `mark_heavy_different_person` | 50 | Critical |
| — lookalike pairs with similar marks | `different_person_lookalike` | 15 | High |
| — random pairs with many marks | | 20 | Medium |
| — twins or high-similarity impostors | `twins_or_high_similarity` | 10 | High |
| — mark-heavy + similar embeddings | | 5 | Critical |
| **Total** | | **120** | |

---

## How to Fill the CSV

### Step 1: Source selection
1. Run the mark detector on candidate images.
2. Filter for images with ≥5 distinctive marks (types: `dark_spot`, `depression_scar`, `linear_scar`, `dark_mole`, `mole`, `structural_crater`).
3. For same-person pairs: find subjects with multiple images where distinctive marks are visible.
4. For different-person pairs: find pairs where both subjects have ≥5 distinctive marks.

### Step 2: Manual annotation
- Run the detector, then have a human verify/correct:
  - Mark types (is the detected `dark_spot` really a dark spot?)
  - Mark regions (is it in the right periocular area?)
  - Image quality (resolution ≥ 250×250, frontal-ish pose, adequate lighting)
- Estimated effort: ~2 minutes per image.

### Step 3: Populate the CSV
- Set `pair_id` with prefix: `mh_same_` for same-person, `mh_diff_` for different-person.
- Set `manual_label_required = false` only after human verification.
- Upload images to `gs://hoppwhistle-facial-uploads/gallery/mark_heavy/`.

### Step 4: Validate
- Run the standard validation workflow:
  ```bash
  gh workflow run validation.yml \
    --ref <tag> \
    -f mode=full-pipeline-sharded \
    -f manifest_path=validation/validation_pairs_mark_heavy_template.csv \
    -f validation_shard_count=5 \
    -f strict_mark_matcher=true
  ```
- Check that detected mark counts align with manual annotations.

---

## Same-Person Categories

| Category | Definition | Selection criteria |
|---|---|---|
| `mark_heavy_same_person` | Subject has ≥5 distinctive marks visible | Detector output + human verification |
| `same_person_age_gap` | Same subject, photos ≥3 years apart | Metadata or dataset labels |
| `same_person_lighting_pose` | Same subject, significant variation | Visual inspection |

## Different-Person Categories

| Category | Definition | Selection criteria |
|---|---|---|
| `mark_heavy_different_person` | Both subjects have ≥5 distinctive marks | Detector output |
| `different_person_lookalike` | Face embedding cosine similarity > 0.65 | Pre-screen with embedding model |
| `twins_or_high_similarity` | Known twin pairs or visual doppelgangers | Dataset labels or manual identification |

---

## Public Dataset Assessment

| Dataset | Useful for | Limitation |
|---|---|---|
| LFW | General baseline | Low-res (250×250), few marks per subject, skewed to `light_scar` |
| CelebA | Attribute-labeled, large scale | No mark annotations; needs detector + verification |
| MORPH | Age-gap same-person | Restricted access, mostly mugshot-style |
| FG-NET | Extreme age gap | Very small (82 subjects), low resolution |
| VGGFace2 | Pose/age variation | No mark annotations, needs detection pipeline |
| Custom collection | Full control | Manual effort required, privacy considerations |

## What LFW Cannot Solve

- **Mark diversity**: LFW subjects rarely have >3 distinctive marks. Dominated by `light_scar` (77% of correspondences).
- **Age gap**: LFW images are from a narrow time window (~2002–2008 news photos).
- **Twins/lookalikes**: No labeled twin pairs.
- **Mark-heavy impostors**: Cannot guarantee both subjects have many distinctive marks.
- **Resolution**: 250×250 is marginal for patch descriptor comparison.

---

## Manual Labeling Requirements

Each image in the mark-heavy dataset requires:
1. **Mark type verification**: Confirm or correct detector output for each mark.
2. **Mark position verification**: Confirm centroid (x, y) in normalized coordinates.
3. **Region assignment**: Verify the canonical region assignment.
4. **Image quality assessment**: Note resolution, pose angle, lighting conditions.

Estimated effort per image: ~2 minutes for a trained annotator.
Estimated total effort for 120-pair dataset (240 images): ~8 hours.

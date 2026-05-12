# HQ Mark Visual Labeling Guide

**Source Run ID:** `25698453012`  
**Source Artifact:** `hq_phase2d_clean_visual_audit_evidence`

> [!IMPORTANT]
> **Facial mark evidence remains review-support-only and does not independently confirm identity.**

## Overview
This labeling guide provides explicit instructions for human reviewers manually auditing the Phase 2D forensic evidence cards. The purpose of this dataset is to create ground truth for distinguishing genuine biometric marks from environmental and sensor noise.

## File Structure
The template `hq_mark_visual_labels_template.csv` contains 152 pre-populated rows spanning `light_scar` and `dark_spot` detections across same-person and impostor comparisons.
- You must review each correspondence using its provided `evidence_card_512_path` and `evidence_card_768_path` imagery.
- Do NOT add new rows unless explicitly approved.

## Required Values for Label Fields

### Yes/No/Uncertain Fields
For the fields `real_visible_mark_probe`, `real_visible_mark_gallery`, `same_physical_mark`, `mark_type_correct`, and `anatomical_region_correct`, you must strictly use one of the following values:
*   `yes`
*   `no`
*   `uncertain`

### Yes/No Fields
For the environmental and texture flags (`texture_noise`, `lighting_glare`, `hair_or_stubble`, `wrinkle_or_expression_line`, `pore_or_skin_texture`, `makeup_or_shadow`), you must use:
*   `yes`
*   `no`

### Correspondence Confidence
For the `correspondence_confidence_1_to_5` field, use the following integer scale:
*   **1** = clearly not same mark
*   **2** = probably not same mark
*   **3** = uncertain / ambiguous
*   **4** = probably same mark
*   **5** = clearly same physical mark

## Definitions

- **Real Visible Mark:** A distinct, persistent anatomical feature (e.g., mole, permanent scar, structural crater) that is clearly visible and not an artifact of the image capture.
- **Texture Noise:** Granularity, sensor noise, or generalized skin roughness that the detector mistakenly bounded as a distinct mark.
- **Lighting Glare:** Specular highlights or reflections off the skin (often on the nose, forehead, or cheekbones) mistaken for a light mark or scar.
- **Hair or Stubble:** Single strands of hair, eyelash shadows, or facial stubble that mimic linear or dark point features.
- **Wrinkle or Expression Line:** A temporary or permanent crease in the skin caused by facial expression or aging, rather than injury or pigmentation (often confused with `linear_scar`).
- **Pore or Skin Texture:** A standard, non-distinctive anatomical pore or natural skin deviation, often hyper-visible in HQ crops, but not a unique biometric identifier.
- **Makeup or Shadow:** Cosmetic alterations or structural shadows mimicking dark spots or scars.
- **Same Physical Mark:** The mark identified in the probe image is unequivocally the exact same anatomical mark identified in the gallery image, considering placement, shape, and surrounding landmarks.
- **Mark Type Correct:** The detector's classification (e.g., `light_scar` vs `dark_spot`) accurately describes the physical reality of the mark.
- **Anatomical Region Correct:** The mark is located on the correct general facial region as mapped.

Please fill in the reviewer ID, the date of review, and any specific free-text notes in `reviewer_notes` for edge cases or anomalies.

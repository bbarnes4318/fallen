# HQ Mark Detector Visual Audit Decision Record

## Background
- LFW/low-quality results are treated as non-decisive.
- HQ dataset exposed detector over-detection.
- Threshold sweeps failed the wiring bar.
- Phase 2D clean visual evidence cards were generated.

## Visual Audit Findings
- `light_scar` is not reliable as identity evidence in the current detector.
- It frequently captures glare, pores, hair, wrinkles, stubble, and skin texture.
- `dark_spot` is more promising but still requires validation.
- Current detector is not approved for production mark scoring.

## Decision
- Suppress or downgrade `light_scar` in future validation experiments.
- Do not suppress/change it in production yet unless explicitly approved later.
- Continue testing `dark_spot` under strict validation-only rules.
- Mark evidence remains review-only / explanatory only.
- Marks do not override face similarity.
- No scoring change.
- No Bayesian fusion change.
- No `/vault/search` change.
- No frontend change.
- No deployment.

## Required Next Experiment: HQ Phase 2E: Validation-Only Detector Policy Experiment
Prepare a proposal only to test the following policies:

**A. light_scar_disabled_validation_only**
Remove `light_scar` from scoring-eligible mark evidence in validation only.

**B. dark_spot_strict_validation_only**
Preserve only visually strong dark spots/moles.

**C. light_scar_disabled_plus_dark_spot_strict**
Combine both validation-only policies.

**D. permanent_marks_only_validation_only**
Keep only `dark_mole`, `structural_crater`, `depression_scar`, strong `linear_scar`.

### Validation Requirements
- Use GitHub Actions only.
- Use: `validation/validation_pairs_hq_headshots.csv`
- Report:
  - total pairs
  - evaluated pairs
  - TP / FP / TN / FN
  - false support correspondences
  - same-person correspondence retention
  - different-person suppression
  - average high-value marks per image
  - mark-type distribution
  - whether any policy improves safety without killing useful signal

## Safety Language
“Facial mark evidence remains review-support-only and does not independently confirm identity.”

## Explicit Non-Changes
- production scoring unchanged
- Bayesian fusion unchanged
- `/vault/search` untouched
- frontend unchanged
- no deployment
- no images committed

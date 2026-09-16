# Fallen V3 Architecture Audit

**Audit target:** `main` at `cffd45598f1fd0fb211c4900c886e5ccad571266`

**Audit scope:** executable backend pipeline, mark subsystem, calibration/scoring concepts, test/deployment evidence, and frontend evidence exposure.

## 1. Current executable architecture

The current 1:1 verification path in `backend/main.py` performs:

1. image retrieval and raw-byte hashing
2. pre-CLAHE face alignment
3. CLAHE preprocessing
4. canonical 256x256 alignment
5. apparent-age estimation
6. cross-spectral normalization
7. ArcFace + Facenet512 embeddings combined as a fixed 60/40 ensemble
8. 3-D landmark-derived geometric ratios
9. LBP texture comparison
10. mark detection / correspondence through the V2 helper when enabled
11. face-model veto logic
12. Bayesian-style posterior generation
13. frontend/audit payload generation

The repository also contains a substantial standalone mark subsystem (`mark_detector.py`, `mark_matcher.py`, `mark_matcher_strict.py`, `mark_anatomy.py`) and a `/marks/analyze` endpoint on the current `main` branch despite an older audit document predating its addition.

## 2. Current strengths

- Existing ArcFace baseline is preserved and available for comparison.
- Facenet512 provides a second global representation.
- High-value chain-of-custody hashing and provenance concepts already exist.
- Mark detection is already a first-class code path rather than a UI-only concept.
- Mark detector has anatomy-aware regioning, exclusion zones, channel-specific candidate generation, and deterministic local descriptors.
- Existing code already records substantial raw evidence and diagnostic structures.
- Existing tests cover mark-analysis and several pipeline contracts.

## 3. Critical architectural shortcomings

### 3.1 Raw score -> LR translation is not a calibrated joint identity model

`pipeline_core.py::score_to_lr_ensemble` converts an ensemble similarity into an LR by looking up an LFW-derived FAR/FRR threshold bucket. This is a threshold lookup, not a fitted density model over Fallen's target population.

### 3.2 Temporal correction is hand-specified

The current implementation applies an exponential `exp(0.01 * temporal_delta)` correction to TPR. This is explicitly inconsistent with the V3 requirement to learn temporal effects from longitudinal data.

### 3.3 Evidence streams are multiplied as if independent

The verification path calculates `lr_total = lr_ensemble * lr_marks`. Global face evidence, geometry, local texture, and mark location are not statistically independent in general. This must be replaced with a calibrated joint fusion model.

### 3.4 Hard ArcFace veto still controls the production conclusion

The current verification path sets `veto_triggered = structural_sim < 0.40` and forces `fused_score = 0.0` unless a future mark-override path becomes eligible. The mark override is currently safety-disabled in `pipeline_core.py`.

This means the production decision is still structurally dominated by one global threshold rather than a calibrated multi-evidence decision.

### 3.5 Image quality is not yet an identity-resolution gate

The pipeline computes multiple image properties, but the architecture does not consistently distinguish poor observability from contradictory identity evidence. V3 introduces an explicit quality evidence object and an INDETERMINATE outcome.

### 3.6 The mark representation is still primarily classical

The detector contains useful multi-channel CV and deterministic local descriptors, but the current implementation is not yet a learned, high-resolution local representation system. LBP/Hu/intensity/gradient features are treated as telemetry in the current detector implementation and are not a learned mark identity representation.

### 3.7 Identity is still primarily pairwise

The production path is principally image A vs image B. The V3 target requires identity profiles that accumulate only high-confidence observations and represent persistent marks, local appearance, global embedding distributions, and temporal history.

### 3.8 No demonstrated V3 quantitative benchmark exists yet

The repository contains validation scripts and prior audit documents, but this branch deliberately does not claim improvement without a properly partitioned Fallen evaluation population. The new evaluation package exists to support that work.

## 4. V3 target architecture

```text
IMAGE
  |
  +--> IMAGE QUALITY
  |
  +--> FACE LOCALIZATION / ALIGNMENT
  |
  +--> GLOBAL FACE REPRESENTATIONS
  |
  +--> HIGH-RES LOCAL REPRESENTATIONS
  |       |
  |       +--> MARK CANDIDATES
  |       +--> LOCAL FEATURE CORRESPONDENCE
  |       +--> MARK CONSTELLATION
  |
  +--> ANATOMICAL NORMALIZATION
  +--> FACIAL ASYMMETRY
  |
  +--> IDENTITY PROFILE RETRIEVAL
  |
  +--> CALIBRATED JOINT FUSION
          |
          +--> SAME PERSON
          +--> DIFFERENT PERSON
          +--> INDETERMINATE
```

## 5. Migration rule

Existing production behavior remains the baseline until the V3 replacement wins an explicitly defined evaluation. No legacy module is deleted merely because a newer implementation exists.

## 6. Required next implementation phases

1. Build the partitioned evaluation harness and pair/cluster manifests.
2. Add high-resolution mark candidate extraction without replacing the legacy detector.
3. Benchmark learned local descriptors against deterministic descriptors.
4. Add anatomical canonicalization and visibility states.
5. Build graph/constellation correspondence baseline.
6. Train/fit a calibrated evidence-fusion model on a training/calibration split and evaluate on identities withheld from development.
7. Add identity-profile retrieval and conservative profile admission.
8. Replace hard-veto production decisions only after the new fusion model demonstrates target operating behavior.

## 7. Evidence policy

Raw similarities remain raw similarities. A calibrated probability/confidence must be produced only by a model fitted to properly partitioned data. A low-quality image is not a negative identity observation. An unobserved mark is not an absent mark.

## 8. Audit limitations

This audit is based on repository source and existing documentation. It is not a claim of biometric performance. No conclusion about the contribution of facial marks is made until Fallen-specific evaluation data demonstrates it.

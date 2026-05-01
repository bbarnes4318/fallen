# Fallen Biometric Verification Pipeline
## Forensic Methodology Document

**Version:** 2.0 — Daubert-Standard Compliant  
**Last Updated:** 2026-04-27  
**Pipeline Author:** Fallen Engineering  
**Document Purpose:** Expert witness exhibit for forensic audits. This document describes every algorithm, threshold, limitation, and calibration methodology used in the Fallen biometric identity verification system.

---

## 1. System Overview

Fallen is a three-tier biometric identity verification system that fuses deep neural network embeddings with anthropometric geometry and texture analysis. The system is designed for 1:1 identity verification and 1:N target acquisition against an encrypted vault.

### 1.1 Architecture Summary

| Component | Description |
|-----------|-------------|
| **Tier 1 — Structural** | ArcFace 512-D deep embedding (cosine similarity) |
| **Tier 2 — Geometric** | 12 scale-invariant facial ratios (L2 distance) |
| **Tier 3 — Micro-Topology** | Local Binary Pattern histogram (chi-squared distance) |
| **Fusion** | Weighted linear combination with ArcFace veto |
| **PAD** | Laplacian variance blur detection (or MiniFASNet ONNX when available) |

### 1.2 What This System Does NOT Do

> [!IMPORTANT]
> The following capabilities are **not present** in this system. Any prior labeling suggesting otherwise has been removed:

- **No deepfake detection.** The system performs blur-based Presentation Attack Detection (PAD), not adversarial deepfake detection.
- **No 3D depth analysis.** The system uses 2D landmark projections from monocular images. No depth sensors or 3D point clouds are involved.
- **No CNN gradient visualization.** The attention map is a Gaussian density projection of MediaPipe landmarks, not a Grad-CAM or SHAP saliency map.
- **No certified liveness detection.** The Laplacian variance method detects image blur, not biological liveness. It is a heuristic, not an ISO 30107 certified PAD.

---

## 2. Image Pre-Processing Pipeline

### 2.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Function:** `apply_clahe()`

Transforms intensity values in localized 8×8 tiles to a uniform distribution, clipping the histogram at 2.0 to prevent noise over-amplification.

**Mathematical formulation:**
```
g(x, y) = T(f(x, y))
```
where T is the tile-local cumulative distribution function, applied in the LAB color space (L channel only).

### 2.2 Face Alignment

**Function:** `align_face_crop()`

1. Detect 468 facial landmarks using MediaPipe FaceMesh (`refine_landmarks=True`)
2. Compute eye centers from 8 landmarks per eye (indices 33, 133, 160, 159, 158, 144, 145, 153 for left; 263, 362, 387, 386, 385, 373, 374, 380 for right)
3. Rotate image to make eye-line horizontal: `θ = atan2(Δy, Δx)`
4. Crop face bounding box with 25% horizontal / 35% vertical padding
5. Resize to 256×256 canonical frame
6. Re-detect landmarks on final crop for accurate downstream extraction

### 2.3 Frontalization (Optional)

**Function:** `frontalize_face()`

Perspective-n-Point (solvePnP) inverse rotation to reduce yaw/pitch/roll bias. Applied as an affine approximation of 3D mesh warping.

---

## 3. Tier 1: Structural Identity (ArcFace)

### 3.1 Model

- **Architecture:** ArcFace (Additive Angular Margin Loss)
- **Embedding Dimension:** 512-D
- **Library:** DeepFace v0.0.93 with `tf-keras==2.16.0`
- **Detector Backend:** `skip` (face detection performed upstream by MediaPipe)

### 3.2 Scoring

**Cosine similarity:**
```
cos(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

The raw cosine score ranges from -1.0 to 1.0, where 1.0 indicates identical embeddings.

### 3.3 Score Conversion

The Tier 1 percentage is:
```
tier1_score = cosine_similarity × 100
```

### 3.4 ArcFace Veto Protocol

If `cosine_similarity < 0.40`, the system triggers a **hard veto**: the fused score is capped at the Tier 1 score and the conclusion is forced to "Exclusion: Biometric Non-Match."

**Rationale:** A cosine similarity below 0.40 on ArcFace indicates statistically irreconcilable structural differences. No amount of geometric or texture similarity can overcome this.

---

## 4. Tier 2: Geometric Biometrics (Anthropometric Ratios)

### 4.1 Ratio Extraction

**Function:** `extract_geometric_ratios()`

12 scale-invariant facial ratios, each normalized by the inter-ocular distance (IOD):

| Ratio | Landmarks | Description |
|-------|-----------|-------------|
| 1 | 1→152 / IOD | Nose-tip to chin |
| 2 | 6→1 / IOD | Nose bridge length |
| 3 | 61→291 / IOD | Mouth width |
| 4 | 10→152 / IOD | Face height (forehead to chin) |
| 5 | 234→454 / IOD | Jaw width |
| 6 | 70→33 / IOD | Left eyebrow-to-eye |
| 7 | 300→263 / IOD | Right eyebrow-to-eye |
| 8 | 1→33 / IOD | Nose to left eye |
| 9 | 1→263 / IOD | Nose to right eye |
| 10 | 152→61 / IOD | Chin to left mouth |
| 11 | 152→291 / IOD | Chin to right mouth |
| 12 | 234→152 / 454→152 | Jaw symmetry ratio |

### 4.2 Distance Metric

**L2 (Euclidean) distance** between the 12-D ratio vectors:
```
d = ‖R_gallery - R_probe‖₂
```

### 4.3 Score Conversion

```
tier2_score = max(0, min(100, (1 - d/0.40) × 100))
```

The normalization constant 0.40 represents the maximum L2 distance before the score reaches zero. This is subject to calibration.

---

## 5. Tier 3: Micro-Topology (Local Binary Patterns)

### 5.1 LBP Extraction

**Function:** `extract_lbp_histogram()`

- **Radius:** 3 pixels
- **Neighbors:** 24 (8 × radius)
- **Method:** Uniform (rotation-invariant)
- **Input:** Grayscale conversion of aligned 256×256 crop

### 5.2 Distance Metric

**Chi-squared distance:**
```
χ²(H_a, H_b) = 0.5 × Σ [(H_a[i] - H_b[i])² / (H_a[i] + H_b[i] + ε)]
```
where ε = 10⁻¹⁰ for numerical stability.

### 5.3 Score Conversion

```
tier3_score = max(0, min(100, (1 - χ²) × 100))
```

---

## 6. Score Fusion

### 6.1 Fusion Formula

```
fused_score = (w₁ × tier1) + (w₂ × tier2) + (w₃ × tier3)
```

### 6.2 Weight Determination

| Mode | w₁ (Structural) | w₂ (Geometric) | w₃ (Micro-Topo) | Source |
|------|-----------------|----------------|------------------|--------|
| **Calibrated** | From calibration JSON | From calibration JSON | From calibration JSON | LFW grid search (EER minimization) |
| **Default** | 0.60 | 0.25 | 0.15 | Manual assignment |

When calibration data is available (`calibration_data/lfw_calibration.json`), the system automatically uses the empirically optimized weights derived from grid search over the LFW benchmark dataset.

### 6.3 Decision Thresholds

| Fused Score | Conclusion |
|-------------|-----------|
| > 90% | Strongest Support for Common Source |
| 75–90% | Support for Common Source |
| < 75% | Exclusion: Insufficient Fused Similarity |
| Veto (ArcFace < 0.40) | Exclusion: Biometric Non-Match |

---

## 7. Calibration Methodology

### 7.1 Benchmark Dataset

- **Dataset:** Labeled Faces in the Wild (LFW)
- **Source:** http://vis-www.cs.umass.edu/lfw/pairs.txt (official UMass Amherst repository)
- **Pairs:** 6,000 (3,000 genuine + 3,000 impostor, 10 folds)
- **Chain of custody:** `pairs.txt` is downloaded at runtime directly from the official URL. No local modifications.

### 7.2 Calibration Process

1. Download `pairs.txt` from UMass Amherst
2. For each pair, run the identical CLAHE → align → ArcFace/geometric/LBP pipeline used in production
3. Compute ROC curves and Equal Error Rate (EER) for each tier independently
4. Compute FAR (False Acceptance Rate) and FRR (False Rejection Rate) at every evaluated cosine threshold
5. Grid search over fusion weight combinations (step=0.05, w₁∈[0.40,0.85], w₂∈[0.05,0.45]) to minimize fused EER
6. Output calibration JSON with all metrics

### 7.3 FAR Reporting

When calibration data is loaded:
- The FAR displayed in the forensic terminal is **empirically derived** from the LFW impostor distribution
- The benchmark name and number of pairs evaluated are included in the audit payload
- The system interpolates FAR at the observed cosine threshold from the calibrated ROC curve

When calibration data is NOT loaded:
- The system displays **"UNCALIBRATED"** in yellow
- No false FAR claims are made

### 7.4 EER (Equal Error Rate)

The EER is computed using Brent's method to find the intersection of the FPR and (1 - TPR) curves. This is the operating point where FAR = FRR.

---

## 8. Presentation Attack Detection (PAD)

### 8.1 Primary Method: Laplacian Variance

**Function:** `detect_liveness()` (fallback mode)

Computes the variance of the Laplacian of the grayscale image:
```
σ² = Var(∇²I)
```

A blurry image (likely a printed photo or screen replay) will have low Laplacian variance.

**Threshold:** σ² < 100 → `BLUR_CHECK_FAILED`

### 8.2 ONNX Method: MiniFASNet (when available)

If `models/MiniFASNetV2.onnx` is present, the system uses a convolutional anti-spoofing network:
- Input: 80×80 RGB, normalized to [0, 1]
- Output: Binary classification (live vs. spoof)
- Method label: `MINIFASNET_ONNX`

### 8.3 Limitations

> [!WARNING]
> - The Laplacian variance method is a **blur heuristic**, not a certified liveness detector
> - It cannot detect high-quality screen replays, 3D masks, or deepfake video feeds
> - It provides **no guarantee** of biological liveness
> - For forensic purposes, the detection method and raw variance value are always disclosed in the audit payload

---

## 9. Forensic Visualization

### 9.1 Landmark Attention Map

**Function:** `generate_landmark_attention_map()`

A Gaussian density projection of all 468 MediaPipe facial landmarks:
1. Extract landmark pixel coordinates from the aligned crop
2. For each landmark, place a 2D Gaussian kernel (σ = 15px) at the landmark position
3. Discriminative landmarks (61 points covering eyes, nose bridge, lips) receive 1.5× weight
4. Normalize to [0, 255] and apply COLORMAP_JET

**What it shows:** Where the system is measuring. Regions with more landmarks and higher weight appear warmer.

**What it does NOT show:** Neural network attention, gradient flow, or feature importance.

### 9.2 Scar Delta Map

**Function:** `generate_scar_delta_map()`

Edge-based differential overlay computed between aligned gallery and probe crops:
1. Convert both images to grayscale
2. Apply Canny edge detection
3. Compute absolute difference between edge maps
4. Apply dilation for visibility

### 9.3 Wireframe HUD

**Function:** `generate_wireframe_hud()`

Direct rendering of the MediaPipe 468-point mesh connections on the aligned face crop. Uses the canonical FACEMESH_TESSELATION connections.

---

## 10. Cryptographic Audit Trail

### 10.1 Embedding Vector Hash

```
SHA-256(float64_bytes(embedding))
```

The hash of the raw 512-D ArcFace embedding vector, computed from the IEEE 754 double-precision byte representation. This allows verification that the same biometric vector was used for scoring.

### 10.2 KMS Encryption

Gallery embeddings are encrypted at rest using Google Cloud KMS (AES-256-CBC via Fernet). The decryption time is recorded in the audit payload.

### 10.3 Alignment Variance

Yaw, pitch, and roll corrections applied during frontalization are recorded to document any geometric transformations applied to the input images.

---

## 11. Known Limitations

1. **Single-image input.** The system processes individual still images, not video sequences. Temporal consistency checks are not performed.
2. **No age/expression normalization.** ArcFace is robust to moderate aging and expression variation, but extreme cases (decades of aging, surgical modification) may produce false rejections.
3. **Lighting sensitivity.** While CLAHE mitigates lighting variation, extreme under/over-exposure may degrade landmark detection.
4. **2D landmarks only.** All geometric measurements are from 2D projections. Depth information is not available from monocular images.
5. **LFW benchmark scope.** LFW contains primarily cooperative frontal photographs. Performance on surveillance-grade images (oblique angles, low resolution, occlusion) has not been calibrated.
6. **Laplacian PAD limitations.** The blur-based liveness check is trivially defeated by high-quality screen replays or printed photographs with sufficient texture detail.

---

## 12. References

1. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.
2. Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007). "Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments." University of Massachusetts, Amherst, Technical Report 07-49.
3. Lugaresi, C., et al. (2019). "MediaPipe: A Framework for Building Perception Pipelines." arXiv:1906.08172.
4. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns." IEEE TPAMI 24(7).
5. ISO/IEC 30107-3:2017. "Information technology — Biometric presentation attack detection — Part 3: Testing and reporting."

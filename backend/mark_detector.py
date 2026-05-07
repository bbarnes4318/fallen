"""Mark Detector v2.1.0 — Multi-channel facial mark detection module.
Pure module: no FastAPI, no DB, no JWT dependencies.
"""
import os
import math
import base64
import cv2
import numpy as np

MARK_DETECTOR_VERSION = "2.1.0"

# MediaPipe landmark index groups for feature exclusion
_LEFT_EYE_IDX = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
_RIGHT_EYE_IDX = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_LEFT_BROW_IDX = [70,63,105,66,107,55,65,52,53,46]
_RIGHT_BROW_IDX = [300,293,334,296,336,285,295,282,283,276]
_NOSE_IDX = [1,2,98,327,168,6,197,195,5,4,45,220,115,48,64,102,49,131,134,236,196,3,51,281,275,440,344,278,294,331,279,360,363,456,420,399,412,351]
_LIPS_IDX = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,80,191,78]
_FACE_OVAL_IDX = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
_BORDER_MARGIN = 10
_EXCLUDE_GROUPS = [_LEFT_EYE_IDX, _RIGHT_EYE_IDX, _LEFT_BROW_IDX, _RIGHT_BROW_IDX, _NOSE_IDX, _LIPS_IDX]

# Strict thresholds
_MIN_AREA = 8
_MAX_AREA = 500
_MIN_OVERLAP_RATIO = 0.50
_MIN_CONTRAST = 1.5
# Fallback thresholds
_FB_MIN_AREA = 4
_FB_MIN_OVERLAP = 0.30
_FB_MIN_CONTRAST = 0.5
_FB_MAX_CANDIDATES = 15
_FB_LR_CAP = 3.0
_MAX_FINAL_MARKS = 30
_DEDUP_DIST_PX = 5
_DEDUP_IOU = 0.3


def _build_skin_mask(shape, landmarks, margin=5):
    """Build binary mask covering face skin. Falls back to ellipse if landmarks insufficient."""
    h, w = shape[:2]
    skin_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mode = "LANDMARK"

    oval_pts = []
    if landmarks is not None:
        for idx in _FACE_OVAL_IDX:
            if idx < len(landmarks):
                lm = landmarks[idx]
                oval_pts.append([int(lm.x * w), int(lm.y * h)])

    if len(oval_pts) >= 3:
        cv2.fillPoly(skin_mask, [np.array(oval_pts, dtype=np.int32)], 255)
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.erode(skin_mask, ek, iterations=1)
    else:
        roi_mode = "LANDMARK_FALLBACK_ROI"
        cx, cy = w // 2, h // 2
        ax, ay = int(w * 0.40), int(h * 0.45)
        cv2.ellipse(skin_mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    if landmarks is not None:
        for idx_group in _EXCLUDE_GROUPS:
            pts = []
            for idx in idx_group:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    pts.append([int(lm.x * w), int(lm.y * h)])
            if len(pts) >= 3:
                hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                M = cv2.moments(hull)
                if M["m00"] > 0:
                    cxh = int(M["m10"] / M["m00"])
                    cyh = int(M["m01"] / M["m00"])
                    inflated = ((hull - [cxh, cyh]) * 1.15 + [cxh, cyh]).astype(np.int32)
                    cv2.fillConvexPoly(skin_mask, inflated, 0)

    skin_mask[:_BORDER_MARGIN, :] = 0
    skin_mask[-_BORDER_MARGIN:, :] = 0
    skin_mask[:, :_BORDER_MARGIN] = 0
    skin_mask[:, -_BORDER_MARGIN:] = 0
    return skin_mask, roi_mode


def _region_label(cx, cy, w, h, landmarks, nearest_idx):
    """Assign a face region label based on nearest landmark and position."""
    if nearest_idx in set(_LEFT_EYE_IDX) | set(_LEFT_BROW_IDX):
        return "left_periocular"
    if nearest_idx in set(_RIGHT_EYE_IDX) | set(_RIGHT_BROW_IDX):
        return "right_periocular"
    if nearest_idx in set(_NOSE_IDX):
        return "nose"
    if nearest_idx in set(_LIPS_IDX):
        return "mouth"
    ny = cy / h
    nx = cx / w
    if ny < 0.33:
        return "forehead"
    if ny > 0.66:
        return "chin_jaw"
    return "left_cheek" if nx < 0.5 else "right_cheek"


def _find_nearest_landmark(cx, cy, w, h, landmarks):
    min_d = float('inf')
    best = -1
    if landmarks is None:
        return best
    for i, lm in enumerate(landmarks):
        d = (lm.x * w - cx)**2 + (lm.y * h - cy)**2
        if d < min_d:
            min_d = d
            best = i
    return best


def _contour_to_descriptor(cnt, gray, skin_mask, h, w, channel, landmarks, mark_index,
                           is_fallback=False):
    """Build a candidate descriptor from a contour. Returns None if invalid."""
    area = cv2.contourArea(cnt)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    ix, iy = int(cx), int(cy)
    if iy >= h or ix >= w:
        return None

    perimeter = cv2.arcLength(cnt, True)
    circ = (4 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
    x, y, bw, bh = cv2.boundingRect(cnt)
    ar = float(bw) / bh if bh > 0 else 0

    orientation = None
    ecc = 0.0
    if len(cnt) >= 5:
        try:
            ell = cv2.fitEllipse(cnt)
            orientation = ell[2]
            maj, mi = max(ell[1]), min(ell[1])
            ecc = math.sqrt(1.0 - (mi / maj)**2) if maj > 0 else 0.0
        except cv2.error:
            pass

    mm = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mm, [cnt], -1, 255, -1)
    mean_int = float(cv2.mean(gray, mask=mm)[0])

    lr = 20
    ly0, ly1 = max(0, iy - lr), min(h, iy + lr)
    lx0, lx1 = max(0, ix - lr), min(w, ix + lr)
    lp = gray[ly0:ly1, lx0:lx1]
    local_mean = float(np.mean(lp)) if lp.size > 0 else mean_int
    contrast = abs(mean_int - local_mean)

    nearest = _find_nearest_landmark(cx, cy, w, h, landmarks)
    region = _region_label(cx, cy, w, h, landmarks, nearest)

    # Salience = contrast * sqrt(area) * (1 if inside mask else 0.5)
    inside = skin_mask[iy, ix] > 0 if iy < h and ix < w else False
    salience = contrast * math.sqrt(area) * (1.0 if inside else 0.5)

    # Classify mark type
    if channel in ("dark", "dark_lesion"):
        mt = "dark_mole" if circ >= 0.6 else "dark_spot"
    elif channel in ("light", "bright_scar"):
        mt = "light_scar"
    elif channel in ("linear_scar", "linear_scar_v2"):
        mt = "linear_scar"
    elif channel in ("texture_cluster", "texture_anomaly"):
        mt = "texture_cluster"
    elif channel == "structural_depression":
        mt = "structural_crater" if circ >= 0.5 else "depression_scar"
    else:
        mt = "unknown_mark"

    # Channel-aware salience adjustment
    if channel == "structural_depression":
        # Reward area, calculate local rim gradient to ensure structural validity
        kx0, kx1 = max(0, x-5), min(w, x+bw+5)
        ky0, ky1 = max(0, y-5), min(h, y+bh+5)
        local_gray = gray[ky0:ky1, kx0:kx1]
        sx = cv2.Scharr(local_gray, cv2.CV_64F, 1, 0)
        sy = cv2.Scharr(local_gray, cv2.CV_64F, 0, 1)
        local_grad = np.sqrt(sx**2 + sy**2)
        
        local_mm = mm[ky0:ky1, kx0:kx1]
        rim_mask = cv2.dilate(local_mm, np.ones((5,5), np.uint8)) - local_mm
        mean_rim_grad = float(cv2.mean(local_grad, mask=rim_mask)[0])
        
        # Boost salience based heavily on area and rim_grad, ignoring low absolute contrast
        structural_score = mean_rim_grad / 255.0
        salience = area * structural_score * (1.0 if inside else 0.5)
        
        conf = min(1.0, salience / 15.0) if not is_fallback else min(0.5, salience / 30.0)
    else:
        conf = min(1.0, salience / 30.0) if not is_fallback else min(0.5, salience / 60.0)

    return {
        "index": mark_index,
        "centroid": (cx / w, cy / h),
        "centroid_px": (float(cx), float(cy)),
        "canonical_position": (cx / w, cy / h),
        "area": float(area),
        "contour_area": float(area),
        "equivalent_radius": float(math.sqrt(area / math.pi)) if area > 0 else 0.0,
        "bbox": (x, y, bw, bh),
        "channel": channel,
        "mark_type": mt,
        "contrast_score": float(contrast),
        "salience_score": float(salience),
        "region_label": region,
        "confidence": float(conf),
        "face_region": region,
        "intensity": float(mean_int),
        "circularity": float(circ),
        "eccentricity": float(ecc),
        "aspect_ratio": float(ar),
        "orientation": float(orientation) if orientation is not None else None,
        "nearest_landmark_index": nearest,
        "contour": cnt,
        "low_confidence": is_fallback,
        "fallback_generated": is_fallback,
        "contrast": float(contrast),
    }


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def _dedup_candidates(marks):
    """Remove overlapping candidates across channels. Keep highest confidence."""
    if len(marks) <= 1:
        return marks, 0
    
    marks_sorted = sorted(marks, key=lambda m: (m.get("confidence", 0), m.get("salience_score", 0)), reverse=True)
    keep = []
    removed = 0
    for m in marks_sorted:
        cx, cy = m["centroid_px"]
        m_area = m.get("area", 1)
        dominated = False
        for k in keep:
            kx, ky = k["centroid_px"]
            k_area = k.get("area", 1)
            dist = math.sqrt((cx - kx)**2 + (cy - ky)**2)
            
            # Check if one is entirely within another or very similar
            iou = _bbox_iou(m["bbox"], k["bbox"])
            
            if iou > _DEDUP_IOU:
                dominated = True
                break
                
            if dist < _DEDUP_DIST_PX:
                # If they share a centroid but one is vastly larger than the other,
                # they are distinct features (e.g. a pore inside a huge crater).
                # Only dedup if they are relatively similar in size.
                area_ratio = min(m_area, k_area) / max(m_area, k_area)
                if area_ratio > 0.15:
                    dominated = True
                    break
                    
        if not dominated:
            keep.append(m)
        else:
            removed += 1
    return keep, removed


def _generate_contours(binary_mask, valid_mask, kernel):
    masked = cv2.bitwise_and(binary_mask, valid_mask)
    cleaned = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def _run_channels(aligned_crop, gray, valid_mask, kernel, h, w, input_is_preprocessed=False):
    """Run all detection channels. Returns dict of channel_name -> contour_list.

    Args:
        input_is_preprocessed: If True, the input has already been CLAHE-enhanced
            by image_preprocessor.py. Internal CLAHE is skipped to avoid double-CLAHE.
            If False (default), internal CLAHE is applied for backward compatibility.
    """
    channels = {}

    # Legacy dark
    dt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=15, C=3)
    channels["dark"] = _generate_contours(dt, valid_mask, kernel)

    # Legacy light
    lt = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=15, C=3)
    channels["light"] = _generate_contours(lt, valid_mask, kernel)

    # Legacy linear scar
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    em = cv2.bitwise_and(edges, valid_mask)
    ec = cv2.morphologyEx(em, cv2.MORPH_CLOSE, kernel)
    lc, _ = cv2.findContours(ec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    channels["linear_scar"] = lc

    # Legacy texture cluster
    sm = cv2.bilateralFilter(gray, 9, 75, 75)
    td = cv2.absdiff(gray, sm)
    _, tt = cv2.threshold(td, 8, 255, cv2.THRESH_BINARY)
    channels["texture_cluster"] = _generate_contours(tt, valid_mask, kernel)

    # --- NEW CHANNELS ---

    # LAB dark lesion
    lab = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    if input_is_preprocessed:
        # Input already CLAHE-enhanced by image_preprocessor — use L directly
        L_enhanced = L
    else:
        # Legacy path: apply internal CLAHE (production routes)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L_enhanced = clahe.apply(L)
    bg = cv2.medianBlur(L_enhanced, 31)
    dark_delta = cv2.subtract(bg, L_enhanced)
    _, dd_thresh = cv2.threshold(dark_delta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    channels["dark_lesion"] = _generate_contours(dd_thresh, valid_mask, kernel)

    # LAB bright scar
    bright_delta = cv2.subtract(L_enhanced, bg)
    p95 = np.percentile(bright_delta[valid_mask > 0], 95) if np.any(valid_mask > 0) else 10
    thr = max(int(p95 * 0.7), 5)
    _, bd_thresh = cv2.threshold(bright_delta, thr, 255, cv2.THRESH_BINARY)
    channels["bright_scar"] = _generate_contours(bd_thresh, valid_mask, kernel)

    # Gradient linear scar v2
    if input_is_preprocessed:
        # Input already CLAHE-enhanced — use gray directly
        gray_enhanced = gray
    else:
        # Legacy path: apply internal CLAHE
        if 'clahe' not in locals():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
    sx = cv2.Scharr(gray_enhanced, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(gray_enhanced, cv2.CV_64F, 0, 1)
    grad = np.sqrt(sx**2 + sy**2).astype(np.uint8)
    _, gm = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined_lines = np.zeros_like(gm)
    for angle in [0, 45, 90, 135]:
        klen = 11
        lk = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        M_rot = cv2.getRotationMatrix2D((klen // 2, 0), angle, 1)
        lk = cv2.warpAffine(lk, M_rot, (klen, klen))
        opened = cv2.morphologyEx(gm, cv2.MORPH_OPEN, lk)
        combined_lines = cv2.bitwise_or(combined_lines, opened)
    channels["linear_scar_v2"] = _generate_contours(combined_lines, valid_mask, kernel)

    # Structural depression (broad low-contrast shadows)
    if input_is_preprocessed:
        # V2 only: Use LAB L channel
        L_blur = cv2.medianBlur(L_enhanced, 7)
        # Use a large median blur to establish local background for broad features
        bg_sd = cv2.medianBlur(L_enhanced, 61)
        dark_delta_sd = cv2.subtract(bg_sd, L_blur)
        
        # Broad structural depressions have low contrast. Threshold > 4 units depth.
        _, sd_thresh = cv2.threshold(dark_delta_sd, 4, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sd_cleaned = cv2.morphologyEx(sd_thresh, cv2.MORPH_OPEN, k_clean)
        channels["structural_depression"] = _generate_contours(sd_cleaned, valid_mask, kernel)
    else:
        channels["structural_depression"] = []

    # Texture anomaly (local variance)
    g32 = gray.astype(np.float32)
    mu = cv2.blur(g32, (7, 7))
    mu2 = cv2.blur(g32 * g32, (7, 7))
    var = mu2 - mu * mu
    var = np.clip(var, 0, None)
    gmu = cv2.blur(var, (31, 31))
    gstd = np.sqrt(cv2.blur((var - gmu)**2, (31, 31)))
    gstd[gstd < 1] = 1
    zscore = ((var - gmu) / gstd)
    anom = (zscore > 2.5).astype(np.uint8) * 255
    channels["texture_anomaly"] = _generate_contours(anom, valid_mask, kernel)

    return channels, {"dark_thresh": dt, "light_thresh": lt, "dd_thresh": dd_thresh,
                      "bd_thresh": bd_thresh, "line_mask": combined_lines, "anom_mask": anom}


def _filter_contours(channels, gray, skin_mask, valid_mask, h, w, landmarks,
                     min_area, max_area, min_overlap, min_contrast, is_fallback=False):
    """Filter contours from all channels through area/shape/overlap/contrast pipeline."""
    marks = []
    rejected = []
    used_mask = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    t = {"area_pass": 0, "shape_pass": 0, "region_pass": 0, "contrast_pass": 0}

    for ch_name, cnts in channels.items():
        ch_min_contrast = min_contrast
        ch_min_area = min_area
        ch_max_area = max_area
        if ch_name == "structural_depression":
            ch_min_contrast = max(0.4, min_contrast * 0.3)
            ch_min_area = max(10, min_area * 2.0)
            ch_max_area = max_area * 10.0
            
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < ch_min_area or area > ch_max_area:
                continue
            t["area_pass"] += 1

            desc = _contour_to_descriptor(cnt, gray, skin_mask, h, w, ch_name, landmarks, idx,
                                          is_fallback=is_fallback)
            if desc is None:
                continue
            t["shape_pass"] += 1

            ix, iy = int(desc["centroid_px"][0]), int(desc["centroid_px"][1])
            if iy >= h or ix >= w:
                continue
            if ch_name != "structural_depression" and used_mask[iy, ix] > 0:
                continue

            rej = None
            if skin_mask[iy, ix] == 0:
                rej = "outside_face_mask"
            if rej is None and (ix < _BORDER_MARGIN or ix >= w - _BORDER_MARGIN or
                                iy < _BORDER_MARGIN or iy >= h - _BORDER_MARGIN):
                rej = "border_artifact"
            if rej is None:
                mm = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mm, [cnt], -1, 255, -1)
                cp = np.count_nonzero(mm)
                if cp > 0:
                    op = np.count_nonzero(cv2.bitwise_and(mm, skin_mask))
                    if op / cp < min_overlap:
                        rej = f"insufficient_face_overlap ({op/cp:.2f})"
            if rej is None:
                t["region_pass"] += 1
            if rej is None and desc["contrast_score"] < ch_min_contrast:
                if desc["contrast_score"] >= ch_min_contrast * 0.5:
                    if desc["salience_score"] < 5.0 and ch_name != "structural_depression":
                        rej = "low_contrast"
                    else:
                        desc["mark_type"] = "blemish"
                else:
                    rej = "low_contrast"
            if rej is not None:
                desc["rejection_reason"] = rej
                rd = {k: v for k, v in desc.items() if k != "contour"}
                rejected.append(rd)
                continue
            t["contrast_pass"] += 1
            cv2.drawContours(used_mask, [cnt], -1, 255, -1)
            desc["index"] = idx
            idx += 1
            marks.append(desc)

    return marks, rejected, t


def detect_facial_marks(aligned_crop: np.ndarray, landmarks,
                        input_is_preprocessed: bool = False) -> tuple:
    """Multi-channel facial mark detector v2.1.0.

    Args:
        aligned_crop: BGR uint8 aligned face crop.
        landmarks: MediaPipe face mesh landmarks or None.
        input_is_preprocessed: If True, input has already been CLAHE-enhanced
            by image_preprocessor.py and internal CLAHE will be skipped.
            Defaults to False for backward compatibility with production routes.

    Returns: (marks, rejected_marks, occ_mask, trace, overlays)
    """
    h, w = aligned_crop.shape[:2]
    gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
    skin_mask, roi_mode = _build_skin_mask(aligned_crop.shape, landmarks)

    occ_mask = np.zeros((h, w), dtype=np.uint8)
    if landmarks is not None:
        for lm in landmarks:
            if getattr(lm, "visibility", 1.0) < 0.85:
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(occ_mask, (px, py), int(min(h, w) * 0.05), 255, -1)

    valid_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(occ_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    internal_clahe_applied = not input_is_preprocessed
    channels, raw_masks = _run_channels(aligned_crop, gray, valid_mask, kernel, h, w,
                                        input_is_preprocessed=input_is_preprocessed)

    trace = {
        "input_is_preprocessed": input_is_preprocessed,
        "internal_clahe_applied": internal_clahe_applied,
        "initial_candidates": sum(len(c) for c in channels.values()),
        "after_skin_mask": 0, "after_area_filter": 0, "after_shape_filter": 0,
        "after_region_exclusion": 0, "after_contrast_filter": 0,
        "dark_lesion_initial_candidates": len(channels.get("dark_lesion", [])),
        "bright_scar_initial_candidates": len(channels.get("bright_scar", [])),
        "linear_scar_initial_candidates": len(channels.get("linear_scar_v2", [])),
        "texture_anomaly_initial_candidates": len(channels.get("texture_anomaly", [])),
        "structural_depression_initial_candidates": len(channels.get("structural_depression", [])),
        "strict_final_valid_marks": 0, "fallback_used": False, "fallback_candidates": 0,
        "fallback_lr_cap": None, "fallback_penalty_applied": False,
        "dedup_removed": 0, "json_serialized_marks_count": 0,
        "detector_status": "OK", "final_valid_marks": 0, "roi_mode": roi_mode,
        "structural_depression_area_pass": 0,
        "structural_depression_contrast_relaxed_pass": 0,
        "structural_depression_final_candidates": 0,
        "structural_depression_thresholds": {
            "min_contrast": max(0.4, _MIN_CONTRAST * 0.3),
            "min_area": max(10, _MIN_AREA * 2.0)
        }
    }

    # Strict pass
    marks, rejected, ft = _filter_contours(
        channels, gray, skin_mask, valid_mask, h, w, landmarks,
        _MIN_AREA, _MAX_AREA, _MIN_OVERLAP_RATIO, _MIN_CONTRAST)
    trace["after_area_filter"] = ft["area_pass"]
    trace["after_shape_filter"] = ft["shape_pass"]
    trace["after_region_exclusion"] = ft["region_pass"]
    trace["after_contrast_filter"] = ft["contrast_pass"]
    trace["strict_final_valid_marks"] = len(marks)

    # Fallback if strict yields 0
    if len(marks) == 0:
        trace["fallback_used"] = True
        fb_marks, fb_rejected, _ = _filter_contours(
            channels, gray, skin_mask, valid_mask, h, w, landmarks,
            _FB_MIN_AREA, _MAX_AREA, _FB_MIN_OVERLAP, _FB_MIN_CONTRAST, is_fallback=True)
        fb_marks.sort(key=lambda m: m["salience_score"], reverse=True)
        fb_marks = fb_marks[:_FB_MAX_CANDIDATES]
        trace["fallback_candidates"] = len(fb_marks)
        trace["fallback_lr_cap"] = float(_FB_LR_CAP)
        trace["fallback_penalty_applied"] = len(fb_marks) > 0
        if len(fb_marks) > 0:
            trace["detector_status"] = "LOW_CONFIDENCE_CANDIDATES"
        else:
            trace["detector_status"] = "NO_CANDIDATES"
        marks = fb_marks
        rejected.extend(fb_rejected)

    # Dedup across channels
    marks, dedup_n = _dedup_candidates(marks)
    trace["dedup_removed"] = dedup_n

    # Cap to top N
    marks.sort(key=lambda m: (m["confidence"], m["salience_score"]), reverse=True)
    marks = marks[:_MAX_FINAL_MARKS]
    
    for m in marks:
        if m.get("channel") == "structural_depression":
            trace["structural_depression_final_candidates"] += 1

    # Re-index
    for i, m in enumerate(marks):
        m["index"] = i
    trace["final_valid_marks"] = len(marks)
    trace["json_serialized_marks_count"] = len(marks)
    trace["after_skin_mask"] = trace["initial_candidates"]

    # Format trace dict for forensic json
    trace_obj = {k: v for k, v in trace.items() if k not in ["structural_depression_area_pass", "structural_depression_contrast_relaxed_pass"]}

    if roi_mode == "LANDMARK_FALLBACK_ROI" and trace["detector_status"] == "OK":
        trace["detector_status"] = "LANDMARK_FALLBACK_ROI"

    # Debug overlays
    overlays = {}
    if os.getenv("DEBUG_FORENSIC") == "true":
        def _enc(img):
            _, buf = cv2.imencode('.png', img)
            return f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"

        overlays["face_roi_mask_b64"] = _enc(skin_mask)

        for key, mask_key in [("dark_candidate_mask_b64", "dark_thresh"),
                              ("bright_candidate_mask_b64", "bd_thresh"),
                              ("linear_candidate_mask_b64", "line_mask"),
                              ("texture_candidate_mask_b64", "anom_mask")]:
            if mask_key in raw_masks:
                m = cv2.bitwise_and(raw_masks[mask_key], valid_mask)
                overlays[key] = _enc(m)

        rej_ov = aligned_crop.copy()
        for rm in rejected:
            b = rm.get("bbox", (0, 0, 0, 0))
            cv2.rectangle(rej_ov, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 0, 255), 1)
        overlays["rejected_overlay_b64"] = _enc(rej_ov)

        fin_ov = aligned_crop.copy()
        for m in marks:
            b = m["bbox"]
            color = (0, 255, 0) if not m.get("low_confidence") else (0, 255, 255)
            cv2.rectangle(fin_ov, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), color, 1)
        overlays["final_marks_overlay_b64"] = _enc(fin_ov)

    return marks, rejected, occ_mask, trace, overlays


def serialize_mark_descriptor(mark: dict) -> dict:
    """Convert a mark descriptor to a JSON-safe dict. Strips contour and numpy types."""
    out = {}
    for k, v in mark.items():
        if k == "contour":
            continue
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = serialize_mark_descriptor(v)
        else:
            out[k] = v
    return out


def get_thresholds() -> dict:
    """Return all detector thresholds as a JSON-safe dict.

    This is the canonical source of truth for threshold reporting.
    Any diagnostic endpoint should call this instead of hardcoding values.
    """
    return {
        # Strict pass
        "min_contour_area": _MIN_AREA,
        "max_contour_area": _MAX_AREA,
        "min_overlap_ratio": _MIN_OVERLAP_RATIO,
        "min_contrast": _MIN_CONTRAST,
        "structural_depression_min_contrast": max(0.4, _MIN_CONTRAST * 0.3),
        "structural_depression_min_area": max(10, _MIN_AREA * 2.0),
        # Fallback pass
        "fallback_min_area": _FB_MIN_AREA,
        "fallback_min_overlap": _FB_MIN_OVERLAP,
        "fallback_min_contrast": _FB_MIN_CONTRAST,
        "fallback_max_candidates": _FB_MAX_CANDIDATES,
        "fallback_lr_cap": _FB_LR_CAP,
        # Post-processing
        "max_final_marks": _MAX_FINAL_MARKS,
        "dedup_distance_px": _DEDUP_DIST_PX,
        "dedup_iou": _DEDUP_IOU,
        "border_margin": _BORDER_MARGIN,
        # Version
        "detector_version": MARK_DETECTOR_VERSION,
    }


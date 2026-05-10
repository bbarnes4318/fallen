"""Phase 3A Analysis — Signal purification, region-normalized UV, availability diagnostics, hard impostor mining.

TELEMETRY ONLY — does not affect scoring, Bayesian fusion, or any production logic.
All functions operate on post-processed validation results.

normalization_reference = "lfw_50_50_sample_phase3a"
normalization_status = "experimental_small_sample"
not_for_scoring = True
"""

import math
from collections import defaultdict


def safe_avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else 0.0


def safe_median(lst):
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def safe_stddev(lst):
    if len(lst) < 2:
        return 0.0
    avg = sum(lst) / len(lst)
    variance = sum((x - avg) ** 2 for x in lst) / (len(lst) - 1)
    return math.sqrt(variance)


def safe_percentile_rank(value, sorted_list):
    """Return percentile rank (0-1) of value within sorted_list."""
    if not sorted_list:
        return 0.0
    count_below = sum(1 for x in sorted_list if x < value)
    return round(count_below / len(sorted_list), 4)


# ── High-value mark types from Phase 2B findings ──
_HIGH_VALUE_TYPES = frozenset({"dark_spot", "depression_scar", "linear_scar", "dark_mole"})
_USEFUL_REGIONS = frozenset({"right_periocular", "right_cheek", "forehead"})
_DISTINCTIVE_MARK_TYPES = frozenset({
    "dark_mole", "mole", "light_scar", "scar",
    "linear_scar", "structural_crater", "depression_scar",
})


def _passes_filter(c, filter_name):
    """Check if a correspondence passes a given filter."""
    if filter_name == "include_all_current":
        return True
    mtype = c.get("mark_type") or "unknown"
    region = c.get("regional_canonical_region_gallery") or "unknown"
    mclass = c.get("mark_class_gallery") or ""
    if filter_name == "suppress_light_scar":
        return mtype != "light_scar"
    if filter_name == "distinctive_only":
        return mtype in _DISTINCTIVE_MARK_TYPES
    if filter_name == "high_value_types_only":
        return mtype in _HIGH_VALUE_TYPES
    if filter_name == "remove_unknown_region":
        return region != "unknown"
    if filter_name == "useful_regions_only":
        return region in _USEFUL_REGIONS
    return True


def compute_signal_purification_filters(results):
    """Compute signal purification filter metrics across all results.

    TELEMETRY ONLY — does not affect scoring.

    Returns dict mapping filter_name -> metrics.
    """
    filter_names = [
        "include_all_current",
        "suppress_light_scar",
        "distinctive_only",
        "high_value_types_only",
        "remove_unknown_region",
        "useful_regions_only",
    ]

    # Collect per-pair stats for FP risk proxy
    pair_filter_data = defaultdict(lambda: {f: {"count": 0} for f in filter_names})

    # Per-filter accumulators
    filter_data = {}
    for fname in filter_names:
        filter_data[fname] = {
            "same_uv": [], "diff_uv": [],
            "same_mq": [], "diff_mq": [],
            "same_count": 0, "diff_count": 0,
        }

    for r in results:
        if r.get("error"):
            continue
        label = r.get("label_same_person", False)
        pair_id = r.get("pair_id", "unknown")
        lr_marks = r.get("lr_marks", 1.0)
        try:
            lr_marks = float(lr_marks)
        except (TypeError, ValueError):
            lr_marks = 1.0

        for c in r.get("accepted_correspondences_detail", []):
            uv = c.get("regional_uv_distance")
            mq = c.get("existing_match_quality")
            for fname in filter_names:
                if _passes_filter(c, fname):
                    fd = filter_data[fname]
                    if label:
                        fd["same_count"] += 1
                        if uv is not None:
                            fd["same_uv"].append(uv)
                        if mq is not None:
                            fd["same_mq"].append(mq)
                    else:
                        fd["diff_count"] += 1
                        if uv is not None:
                            fd["diff_uv"].append(uv)
                        if mq is not None:
                            fd["diff_mq"].append(mq)
                        pair_filter_data[pair_id][fname]["count"] += 1

        # Store per-pair lr_marks for FP risk
        if not label and not r.get("error"):
            for fname in filter_names:
                pair_filter_data[pair_id][fname]["lr_marks"] = lr_marks

    # Build output
    output = {}
    for fname in filter_names:
        fd = filter_data[fname]
        same_avg_uv = safe_avg(fd["same_uv"])
        diff_avg_uv = safe_avg(fd["diff_uv"])
        # FP risk proxy: diff-person pairs with lr_marks > 10 AND >= 3 surviving correspondences
        fp_risk = 0
        for pid, pdata in pair_filter_data.items():
            pf = pdata.get(fname, {})
            if pf.get("count", 0) >= 3 and pf.get("lr_marks", 0) > 10:
                fp_risk += 1

        output[fname] = {
            "correspondences_retained": fd["same_count"] + fd["diff_count"],
            "same_person_correspondences": fd["same_count"],
            "different_person_correspondences": fd["diff_count"],
            "same_avg_uv_distance": same_avg_uv,
            "diff_avg_uv_distance": diff_avg_uv,
            "uv_separation_delta": round(diff_avg_uv - same_avg_uv, 4),
            "same_avg_existing_match_quality": safe_avg(fd["same_mq"]),
            "diff_avg_existing_match_quality": safe_avg(fd["diff_mq"]),
            "fp_risk_proxy": fp_risk,
        }

    return output


def compute_region_normalized_uv(results):
    """Compute region-normalized UV distance metrics.

    TELEMETRY ONLY — does not affect scoring.
    normalization_reference = "lfw_50_50_sample_phase3a"
    normalization_status = "experimental_small_sample"
    not_for_scoring = True

    Two-pass approach:
      Pass 1: Collect per-region UV distributions.
      Pass 2: Compute z-score, percentile, quality-weighted, calibrated per correspondence.
    """
    # Pass 1: collect per-region UV values
    region_uv_all = defaultdict(list)  # region -> [uv_distance]
    region_uv_diff = defaultdict(list)  # region -> [uv_distance] (different-person only)

    # Also collect raw correspondences for pass 2
    all_correspondences = []

    for r in results:
        if r.get("error"):
            continue
        label = r.get("label_same_person", False)
        for c in r.get("accepted_correspondences_detail", []):
            region = c.get("regional_canonical_region_gallery") or "unknown"
            uv = c.get("regional_uv_distance")
            qw = c.get("region_uv_distance_quality_weighted")
            if uv is not None and region != "unknown":
                region_uv_all[region].append(uv)
                if not label:
                    region_uv_diff[region].append(uv)
                all_correspondences.append({
                    "region": region,
                    "uv": uv,
                    "qw": qw,
                    "label": label,
                })

    # Compute per-region reference stats
    reference_stats = {}
    for region in region_uv_all:
        vals = region_uv_all[region]
        diff_vals = region_uv_diff.get(region, [])
        reference_stats[region] = {
            "mean": safe_avg(vals),
            "stddev": round(safe_stddev(vals), 4),
            "diff_median": round(safe_median(diff_vals), 4),
            "n": len(vals),
        }

    # Pass 2: compute normalized values
    same_zscore, diff_zscore = [], []
    same_percentile, diff_percentile = [], []
    same_qw, diff_qw = [], []
    same_calibrated, diff_calibrated = [], []

    for entry in all_correspondences:
        region = entry["region"]
        uv = entry["uv"]
        label = entry["label"]
        stats = reference_stats.get(region, {})

        # z-score
        mean = stats.get("mean", 0)
        stddev = stats.get("stddev", 0)
        if stddev > 0.001:
            zscore = round((uv - mean) / stddev, 4)
        else:
            zscore = 0.0
        if label:
            same_zscore.append(zscore)
        else:
            diff_zscore.append(zscore)

        # percentile
        sorted_region = sorted(region_uv_all.get(region, []))
        pct = safe_percentile_rank(uv, sorted_region)
        if label:
            same_percentile.append(pct)
        else:
            diff_percentile.append(pct)

        # quality-weighted
        qw = entry.get("qw")
        if qw is not None:
            if label:
                same_qw.append(qw)
            else:
                diff_qw.append(qw)

        # region-calibrated (raw / diff_median, so 1.0 = median impostor)
        diff_med = stats.get("diff_median", 0)
        if diff_med > 0.001:
            calibrated = round(uv / diff_med, 4)
        else:
            calibrated = 0.0
        if label:
            same_calibrated.append(calibrated)
        else:
            diff_calibrated.append(calibrated)

    raw_same = safe_avg([e["uv"] for e in all_correspondences if e["label"]])
    raw_diff = safe_avg([e["uv"] for e in all_correspondences if not e["label"]])

    def _metric(same_list, diff_list):
        sa = safe_avg(same_list)
        da = safe_avg(diff_list)
        return {"same_avg": sa, "diff_avg": da, "delta": round(da - sa, 4)}

    return {
        "normalization_reference": "lfw_50_50_sample_phase3a",
        "normalization_status": "experimental_small_sample",
        "not_for_scoring": True,
        "reference_stats": reference_stats,
        "aggregated": {
            "raw": {"same_avg": raw_same, "diff_avg": raw_diff, "delta": round(raw_diff - raw_same, 4)},
            "zscore_by_region": _metric(same_zscore, diff_zscore),
            "percentile_by_region": _metric(same_percentile, diff_percentile),
            "quality_weighted": _metric(same_qw, diff_qw),
            "region_calibrated": _metric(same_calibrated, diff_calibrated),
        },
    }


def compute_availability_diagnostics(results):
    """Diagnose regional UV availability and whether cross-region rescues preserve separation.

    TELEMETRY ONLY — does not affect scoring.
    """
    total_corr = 0
    same_region_count = 0

    face_compat_same_uv, face_compat_diff_uv = [], []
    related_compat_same_uv, related_compat_diff_uv = [], []
    unavailable_reasons = defaultdict(int)
    face_compat_count = 0
    related_compat_count = 0
    unavailable_no_fallback = 0

    for r in results:
        if r.get("error"):
            continue
        label = r.get("label_same_person", False)
        for c in r.get("accepted_correspondences_detail", []):
            total_corr += 1
            mode = c.get("regional_comparison_mode", "unavailable")

            if mode == "same_canonical_region":
                same_region_count += 1
            elif mode == "face_region_compatible_cross_canonical":
                face_compat_count += 1
                cross_dist = c.get("regional_cross_region_distance")
                if cross_dist is not None:
                    if label:
                        face_compat_same_uv.append(cross_dist)
                    else:
                        face_compat_diff_uv.append(cross_dist)
            elif mode == "related_region_compatible":
                related_compat_count += 1
                cross_dist = c.get("regional_cross_region_distance")
                if cross_dist is not None:
                    if label:
                        related_compat_same_uv.append(cross_dist)
                    else:
                        related_compat_diff_uv.append(cross_dist)
            elif mode == "unavailable":
                unavailable_no_fallback += 1
                reason = c.get("regional_unavailable_reason") or "unknown"
                unavailable_reasons[reason] += 1

    def _sep_info(same_list, diff_list):
        sa = safe_avg(same_list)
        da = safe_avg(diff_list)
        delta = round(da - sa, 4)
        return {
            "same_avg_cross_region_distance": sa,
            "diff_avg_cross_region_distance": da,
            "separation_delta": delta,
            "preserves_separation": delta > 0.02,
            "same_count": len(same_list),
            "diff_count": len(diff_list),
        }

    potential_rescue = face_compat_count + related_compat_count
    face_info = _sep_info(face_compat_same_uv, face_compat_diff_uv)
    related_info = _sep_info(related_compat_same_uv, related_compat_diff_uv)
    rescue_preserves = face_info["preserves_separation"]

    noise_risk = "low"
    if not rescue_preserves:
        noise_risk = "high"
    elif face_info.get("separation_delta", 0) < 0.05:
        noise_risk = "medium"

    return {
        "total_correspondences": total_corr,
        "fully_available_same_region": same_region_count,
        "face_region_compatible": {"count": face_compat_count, **face_info},
        "related_region_compatible": {"count": related_compat_count, **related_info},
        "unavailable_no_fallback": {
            "count": unavailable_no_fallback,
            "reasons": dict(unavailable_reasons),
        },
        "potential_rescue_count": potential_rescue,
        "rescue_preserves_separation": rescue_preserves,
        "noise_risk": noise_risk,
    }


def compute_hard_impostor_mining(results):
    """Mine the hardest different-person pairs by impostor danger score.

    TELEMETRY ONLY — does not affect scoring.

    Returns top 25 plus aggregate summary across all different-person pairs.
    """
    impostor_pairs = []

    for r in results:
        if r.get("error"):
            continue
        label = r.get("label_same_person", False)
        if label:
            continue  # Only analyze different-person pairs

        pair_id = r.get("pair_id", "unknown")
        strict_data = r.get("strict_mark_data", {})
        scoring_corr = strict_data.get("scoring_correspondences_count", 0)
        lr_marks = r.get("lr_marks", 1.0)
        try:
            lr_marks = float(lr_marks)
            if not math.isfinite(lr_marks):
                lr_marks = 1.0
        except (TypeError, ValueError):
            lr_marks = 1.0
        distinctive = strict_data.get("distinctive_preserved", 0)
        total_marks_pair = scoring_corr  # approximate

        # Collect mark types and regions from correspondences
        mark_types = defaultdict(int)
        regions = defaultdict(int)
        match_qualities = []
        for c in r.get("accepted_correspondences_detail", []):
            mt = c.get("mark_type") or "unknown"
            rg = c.get("regional_canonical_region_gallery") or "unknown"
            mark_types[mt] += 1
            regions[rg] += 1
            mq = c.get("existing_match_quality")
            if mq is not None:
                match_qualities.append(mq)

        avg_mq = safe_avg(match_qualities)
        total_corr = sum(mark_types.values())
        light_scar_count = mark_types.get("light_scar", 0)
        unknown_region_count = regions.get("unknown", 0)
        light_scar_frac = round(light_scar_count / max(1, total_corr), 4)
        unknown_region_frac = round(unknown_region_count / max(1, total_corr), 4)

        # Danger score
        dist_frac = distinctive / max(1, total_marks_pair)
        danger = (scoring_corr * 0.4) + (lr_marks * 0.3) + (avg_mq * 0.2) + (dist_frac * 0.1)

        top_mark = max(mark_types, key=mark_types.get) if mark_types else "unknown"
        top_region = max(regions, key=regions.get) if regions else "unknown"

        impostor_pairs.append({
            "pair_id": pair_id,
            "danger_score": round(danger, 4),
            "scoring_correspondences": scoring_corr,
            "lr_marks": round(lr_marks, 4),
            "avg_match_quality": avg_mq,
            "mark_types_present": dict(mark_types),
            "regions_present": dict(regions),
            "light_scar_fraction": light_scar_frac,
            "unknown_region_fraction": unknown_region_frac,
            "top_mark_type": top_mark,
            "top_region": top_region,
        })

    # Sort by danger score
    impostor_pairs.sort(key=lambda x: -x["danger_score"])
    top_25 = impostor_pairs[:25]

    # Aggregate across all different-person pairs
    all_mark_types = defaultdict(int)
    all_regions = defaultdict(int)
    total_light_scar = 0
    total_unknown_region = 0
    total_corr_all = 0
    for p in impostor_pairs:
        for mt, cnt in p["mark_types_present"].items():
            all_mark_types[mt] += cnt
            total_corr_all += cnt
            if mt == "light_scar":
                total_light_scar += cnt
        for rg, cnt in p["regions_present"].items():
            all_regions[rg] += cnt
            if rg == "unknown":
                total_unknown_region += cnt

    light_scar_drove = total_light_scar > 0.6 * total_corr_all if total_corr_all > 0 else False
    unknown_drove = total_unknown_region > 0.4 * total_corr_all if total_corr_all > 0 else False

    # Recommended suppressions
    suppressions = []
    if light_scar_drove:
        suppressions.append("suppress_light_scar")
    if unknown_drove:
        suppressions.append("remove_unknown_region")
    # Add any mark type that appears in >30% of top-25 correspondences
    top25_corr = sum(sum(p["mark_types_present"].values()) for p in top_25)
    for mt in sorted(all_mark_types, key=all_mark_types.get, reverse=True):
        top25_mt = sum(p["mark_types_present"].get(mt, 0) for p in top_25)
        if top25_corr > 0 and top25_mt / top25_corr > 0.3 and mt not in ("light_scar",):
            suppressions.append(f"suppress_{mt}")

    return {
        "total_different_person_pairs": len(impostor_pairs),
        "top_25_hardest": top_25,
        "aggregate_all_different_person": {
            "mark_type_counts": dict(sorted(all_mark_types.items(), key=lambda x: -x[1])),
            "region_counts": dict(sorted(all_regions.items(), key=lambda x: -x[1])),
            "total_correspondences": total_corr_all,
            "light_scar_fraction": round(total_light_scar / max(1, total_corr_all), 4),
            "unknown_region_fraction": round(total_unknown_region / max(1, total_corr_all), 4),
        },
        "light_scar_drove_signal": light_scar_drove,
        "unknown_region_drove_signal": unknown_drove,
        "recommended_suppressions": suppressions,
    }

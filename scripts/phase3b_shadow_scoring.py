"""Phase 3B Shadow Scoring Simulation — UV Distance Review Signal Candidates.

TELEMETRY ONLY — does not affect scoring, Bayesian fusion, or any production logic.
Shadow scores never influence fused_score, conclusion, lr_marks, or bayesian_fused_score.

This module:
  1. Applies 6 candidate filters to each pair's correspondences.
  2. Computes a shadow_uv_support_score (0-100) per pair per candidate.
  3. Sweeps thresholds to find zero-false-support operating points.
  4. Analyzes face interaction (strong/weak face x strong/weak shadow).
  5. Tests shadow scores against Phase 3A's top-25 hard impostors.
  6. Reports dataset limitations.
"""

import math
from collections import defaultdict


# ── Constants ──
_HIGH_VALUE_TYPES = frozenset({"dark_spot", "depression_scar", "linear_scar", "dark_mole"})
_USEFUL_REGIONS = frozenset({"right_periocular", "right_cheek", "forehead"})
_THRESHOLDS = [25, 40, 50, 60, 75]
_FACE_THRESHOLDS = [60, 70, 80]
_PRIMARY_FACE_THRESHOLD = 70


def _safe_avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else 0.0


# ──────────────────────────────────────────────────────────────────────
# 1. Candidate filter logic
# ──────────────────────────────────────────────────────────────────────

def _apply_candidate_filter(correspondences, candidate_name):
    """Return filtered list of correspondences for a given candidate.

    Each correspondence is a dict from accepted_correspondences_detail.
    """
    surviving = []
    for c in correspondences:
        mt = c.get("mark_type") or "unknown"
        region = c.get("regional_canonical_region_gallery") or "unknown"
        mode = c.get("regional_comparison_mode") or "unavailable"

        if candidate_name == "suppress_light_scar_uv":
            if mt == "light_scar":
                continue
            surviving.append(c)

        elif candidate_name == "useful_regions_uv":
            if region not in _USEFUL_REGIONS:
                continue
            surviving.append(c)

        elif candidate_name == "combined_best_uv":
            if mt == "light_scar":
                continue
            if region not in _USEFUL_REGIONS:
                continue
            surviving.append(c)

        elif candidate_name == "high_value_types_uv":
            if mt not in _HIGH_VALUE_TYPES:
                continue
            surviving.append(c)

        elif candidate_name == "conservative_constellation_uv":
            if mt == "light_scar":
                continue
            if region == "unknown":
                continue
            if mode == "face_region_compatible_cross_canonical":
                continue
            surviving.append(c)

        elif candidate_name == "conservative_constellation_uv_strict":
            if mt == "light_scar":
                continue
            if region == "unknown":
                continue
            if mode == "face_region_compatible_cross_canonical":
                continue
            surviving.append(c)

        else:
            surviving.append(c)

    # Post-filter minimum requirements for constellation candidates
    if candidate_name == "conservative_constellation_uv":
        if len(surviving) < 2:
            return []
        distinct_regions = len(set(c.get("regional_canonical_region_gallery") or "unknown" for c in surviving))
        hv_count = sum(1 for c in surviving if (c.get("mark_type") or "unknown") in _HIGH_VALUE_TYPES)
        if distinct_regions < 2 and hv_count < 2:
            return []

    elif candidate_name == "conservative_constellation_uv_strict":
        if len(surviving) < 3:
            return []
        distinct_regions = len(set(c.get("regional_canonical_region_gallery") or "unknown" for c in surviving))
        hv_count = sum(1 for c in surviving if (c.get("mark_type") or "unknown") in _HIGH_VALUE_TYPES)
        if distinct_regions < 2 or hv_count < 2:
            return []

    return surviving


# ──────────────────────────────────────────────────────────────────────
# 2. Shadow score computation
# ──────────────────────────────────────────────────────────────────────

def _compute_shadow_score(surviving):
    """Compute shadow_uv_support_score (0-100) from filtered correspondences.

    TELEMETRY ONLY — does not affect scoring.

    Formula:
      distance_score (0-50):  lower UV = stronger support
      count_score (0-25):     more correspondences = more evidence
      quality_score (0-25):   high-value types + useful regions + diversity

    Hard safety gates:
      - If avg_uv_distance > 1.0 or missing → cap total at 49
      - Risk flag for impostor indicators
    """
    n = len(surviving)
    if n == 0:
        return {
            "shadow_uv_support_score": 0,
            "shadow_uv_risk_flag": False,
            "shadow_review_label": "NONE",
            "distance_score": 0.0,
            "count_score": 0.0,
            "quality_score": 0.0,
            "avg_uv_distance": None,
            "surviving_correspondence_count": 0,
            "high_value_count": 0,
            "useful_region_count": 0,
            "distinct_region_count": 0,
        }

    # Collect UV distances
    uv_distances = []
    for c in surviving:
        uv = c.get("regional_uv_distance")
        if uv is not None:
            try:
                uv = float(uv)
                if math.isfinite(uv):
                    uv_distances.append(uv)
            except (TypeError, ValueError):
                pass

    avg_uv = _safe_avg(uv_distances) if uv_distances else None

    # Quality components
    high_value_count = sum(1 for c in surviving if (c.get("mark_type") or "unknown") in _HIGH_VALUE_TYPES)
    useful_region_count = sum(1 for c in surviving if (c.get("regional_canonical_region_gallery") or "unknown") in _USEFUL_REGIONS)
    distinct_regions = set()
    for c in surviving:
        rg = c.get("regional_canonical_region_gallery") or "unknown"
        if rg != "unknown":
            distinct_regions.add(rg)
    distinct_region_count = len(distinct_regions)

    # ── Distance component (0-50) ──
    if avg_uv is not None and avg_uv >= 0:
        distance_score = max(0.0, min(50.0, (1.2 - avg_uv) / 1.2 * 50.0))
    else:
        distance_score = 0.0

    # ── Count component (0-25) ──
    count_score = max(0.0, min(25.0, n * 5.0))

    # ── Quality component (0-25) ──
    hv_score = min(10.0, high_value_count * 5.0)
    region_score = min(9.0, useful_region_count * 3.0)
    diversity_score = min(6.0, distinct_region_count * 3.0)
    quality_score = hv_score + region_score + diversity_score

    raw_total = distance_score + count_score + quality_score

    # ── Hard safety gate: cap at 49 if UV is poor or missing ──
    if avg_uv is None or avg_uv > 1.0:
        raw_total = min(raw_total, 49.0)

    shadow_uv_support_score = int(round(raw_total))
    shadow_uv_support_score = max(0, min(100, shadow_uv_support_score))

    # ── Risk flag ──
    light_scar_frac = sum(1 for c in surviving if (c.get("mark_type") or "unknown") == "light_scar") / max(1, n)
    unknown_region_frac = sum(1 for c in surviving if (c.get("regional_canonical_region_gallery") or "unknown") == "unknown") / max(1, n)

    shadow_uv_risk_flag = False
    if avg_uv is not None and avg_uv > 1.0 and n >= 3:
        shadow_uv_risk_flag = True
    if light_scar_frac > 0.6:
        shadow_uv_risk_flag = True
    if unknown_region_frac > 0.4:
        shadow_uv_risk_flag = True

    # ── Review label ──
    if shadow_uv_risk_flag:
        shadow_review_label = "IMPOSTOR_RISK"
    elif shadow_uv_support_score == 0:
        shadow_review_label = "NONE"
    elif shadow_uv_support_score < 25:
        shadow_review_label = "WEAK_REVIEW_SUPPORT"
    elif shadow_uv_support_score < 50:
        shadow_review_label = "MODERATE_REVIEW_SUPPORT"
    else:
        shadow_review_label = "STRONG_REVIEW_SUPPORT"

    return {
        "shadow_uv_support_score": shadow_uv_support_score,
        "shadow_uv_risk_flag": shadow_uv_risk_flag,
        "shadow_review_label": shadow_review_label,
        "distance_score": round(distance_score, 2),
        "count_score": round(count_score, 2),
        "quality_score": round(quality_score, 2),
        "avg_uv_distance": avg_uv,
        "surviving_correspondence_count": n,
        "high_value_count": high_value_count,
        "useful_region_count": useful_region_count,
        "distinct_region_count": distinct_region_count,
    }


# ──────────────────────────────────────────────────────────────────────
# 3. Main simulation: per-pair × per-candidate
# ──────────────────────────────────────────────────────────────────────

_CANDIDATES = [
    "suppress_light_scar_uv",
    "useful_regions_uv",
    "combined_best_uv",
    "high_value_types_uv",
    "conservative_constellation_uv",
    "conservative_constellation_uv_strict",
]


def compute_shadow_scoring_simulation(results):
    """Compute shadow scores for all pairs across all 6 candidates.

    TELEMETRY ONLY — does not affect scoring.

    Returns:
      {
        "candidates": { candidate_name: aggregate_stats },
        "per_pair_scores": [ { pair_id, label, fused_score, candidate_scores: {} } ]
      }
    """
    per_pair_scores = []

    # Per-candidate accumulators
    candidate_same_scores = {c: [] for c in _CANDIDATES}
    candidate_diff_scores = {c: [] for c in _CANDIDATES}
    candidate_surviving_counts = {c: [] for c in _CANDIDATES}
    candidate_retained_pairs = {c: 0 for c in _CANDIDATES}

    for r in results:
        if r.get("error"):
            continue

        pair_id = r.get("pair_id", "unknown")
        label = r.get("label_same_person", False)
        fused_score = r.get("fused_score", 0.0)
        try:
            fused_score = float(fused_score)
        except (TypeError, ValueError):
            fused_score = 0.0

        correspondences = r.get("accepted_correspondences_detail", [])
        pair_entry = {
            "pair_id": pair_id,
            "label_same_person": label,
            "fused_score": fused_score,
            "candidate_scores": {},
        }

        for cand in _CANDIDATES:
            surviving = _apply_candidate_filter(correspondences, cand)
            score_result = _compute_shadow_score(surviving)
            pair_entry["candidate_scores"][cand] = score_result

            sc = score_result["shadow_uv_support_score"]
            n_surviving = score_result["surviving_correspondence_count"]

            if n_surviving > 0:
                candidate_retained_pairs[cand] += 1

            candidate_surviving_counts[cand].append(n_surviving)
            if label:
                candidate_same_scores[cand].append(sc)
            else:
                candidate_diff_scores[cand].append(sc)

        per_pair_scores.append(pair_entry)

    # Build candidate aggregate stats
    candidates_out = {}
    for cand in _CANDIDATES:
        candidates_out[cand] = {
            "retained_pairs_with_evidence": candidate_retained_pairs[cand],
            "avg_surviving_correspondences": _safe_avg(candidate_surviving_counts[cand]),
            "avg_score_same_person": _safe_avg(candidate_same_scores[cand]),
            "avg_score_different_person": _safe_avg(candidate_diff_scores[cand]),
            "same_person_score_distribution": _score_distribution(candidate_same_scores[cand]),
            "different_person_score_distribution": _score_distribution(candidate_diff_scores[cand]),
        }

    return {
        "candidates": candidates_out,
        "per_pair_scores": per_pair_scores,
    }


def _score_distribution(scores):
    """Bin scores into review label categories."""
    bins = {"NONE": 0, "WEAK": 0, "MODERATE": 0, "STRONG": 0}
    for s in scores:
        if s == 0:
            bins["NONE"] += 1
        elif s < 25:
            bins["WEAK"] += 1
        elif s < 50:
            bins["MODERATE"] += 1
        else:
            bins["STRONG"] += 1
    return bins


# ──────────────────────────────────────────────────────────────────────
# 4. Threshold sweep
# ──────────────────────────────────────────────────────────────────────

def compute_threshold_sweep(per_pair_scores):
    """Sweep thresholds at 25/40/50/60/75 for each candidate.

    TELEMETRY ONLY — does not affect scoring.
    """
    sweep = {}
    for cand in _CANDIDATES:
        cand_thresholds = {}
        best_zfs_threshold = None

        for threshold in _THRESHOLDS:
            same_flagged = 0
            diff_flagged = 0

            for pair in per_pair_scores:
                sc = pair["candidate_scores"].get(cand, {}).get("shadow_uv_support_score", 0)
                risk = pair["candidate_scores"].get(cand, {}).get("shadow_uv_risk_flag", False)
                # If risk flag is set, this pair should not be counted as support
                effective_support = sc >= threshold and not risk

                if pair["label_same_person"]:
                    if effective_support:
                        same_flagged += 1
                else:
                    if effective_support:
                        diff_flagged += 1

            total_same = sum(1 for p in per_pair_scores if p["label_same_person"])
            precision = round(same_flagged / max(1, same_flagged + diff_flagged), 4) if (same_flagged + diff_flagged) > 0 else 0.0
            recall = round(same_flagged / max(1, total_same), 4) if total_same > 0 else 0.0

            is_zero_fs = diff_flagged == 0

            cand_thresholds[str(threshold)] = {
                "same_person_flagged_support": same_flagged,
                "different_person_flagged_support": diff_flagged,
                "true_support_count": same_flagged,
                "false_support_count": diff_flagged,
                "support_precision": precision,
                "support_recall": recall,
                "hard_impostor_exposure": diff_flagged,
                "zero_false_support": is_zero_fs,
            }

            if is_zero_fs and best_zfs_threshold is None:
                best_zfs_threshold = threshold

        sweep[cand] = {
            "thresholds": cand_thresholds,
            "best_zero_false_support_threshold": best_zfs_threshold,
        }

    return sweep


# ──────────────────────────────────────────────────────────────────────
# 5. Face interaction analysis
# ──────────────────────────────────────────────────────────────────────

def compute_face_interaction(results, per_pair_scores):
    """Analyze shadow score behavior across face strength buckets.

    TELEMETRY ONLY — does not affect scoring.

    Buckets (per candidate, per face threshold):
      1. Strong face + Strong shadow
      2. Strong face + Weak shadow
      3. Weak face + Strong shadow → Human Review Needed ONLY
      4. Weak face + Weak shadow
    """
    shadow_strong_threshold = 50  # shadow_uv_support_score >= 50 = strong

    # Build pair lookup
    pair_lookup = {}
    for pair in per_pair_scores:
        pair_lookup[pair["pair_id"]] = pair

    output = {
        "shadow_strong_threshold": shadow_strong_threshold,
        "primary_face_threshold": _PRIMARY_FACE_THRESHOLD,
        "sensitivity_face_thresholds": _FACE_THRESHOLDS,
    }

    for face_threshold in _FACE_THRESHOLDS:
        ft_key = f"face_ge_{face_threshold}"
        ft_results = {}

        for cand in _CANDIDATES:
            buckets = {
                "bucket_1_strong_face_strong_shadow": {"same": 0, "diff": 0, "pairs": []},
                "bucket_2_strong_face_weak_shadow": {"same": 0, "diff": 0, "pairs": []},
                "bucket_3_weak_face_strong_shadow": {"same": 0, "diff": 0, "pairs": []},
                "bucket_4_weak_face_weak_shadow": {"same": 0, "diff": 0, "pairs": []},
            }

            for r in results:
                if r.get("error"):
                    continue
                pair_id = r.get("pair_id", "unknown")
                label = r.get("label_same_person", False)
                fused = r.get("fused_score", 0.0)
                try:
                    fused = float(fused)
                except (TypeError, ValueError):
                    fused = 0.0

                pair_data = pair_lookup.get(pair_id)
                if not pair_data:
                    continue
                cand_score = pair_data["candidate_scores"].get(cand, {})
                shadow_sc = cand_score.get("shadow_uv_support_score", 0)
                risk = cand_score.get("shadow_uv_risk_flag", False)

                strong_face = fused >= face_threshold
                strong_shadow = shadow_sc >= shadow_strong_threshold and not risk

                if strong_face and strong_shadow:
                    bucket_key = "bucket_1_strong_face_strong_shadow"
                elif strong_face and not strong_shadow:
                    bucket_key = "bucket_2_strong_face_weak_shadow"
                elif not strong_face and strong_shadow:
                    bucket_key = "bucket_3_weak_face_strong_shadow"
                else:
                    bucket_key = "bucket_4_weak_face_weak_shadow"

                label_key = "same" if label else "diff"
                buckets[bucket_key][label_key] += 1
                # Track dangerous cases: different-person in bucket 1 or 3
                if not label and bucket_key in ("bucket_1_strong_face_strong_shadow", "bucket_3_weak_face_strong_shadow"):
                    buckets[bucket_key]["pairs"].append(pair_id)

            # Safety checks
            diff_in_strong_shadow = (
                buckets["bucket_1_strong_face_strong_shadow"]["diff"] +
                buckets["bucket_3_weak_face_strong_shadow"]["diff"]
            )

            ft_results[cand] = {
                "buckets": {k: {"same": v["same"], "diff": v["diff"]} for k, v in buckets.items()},
                "weak_face_strong_marks_same_count": buckets["bucket_3_weak_face_strong_shadow"]["same"],
                "weak_face_strong_marks_diff_count": buckets["bucket_3_weak_face_strong_shadow"]["diff"],
                "different_person_in_strong_shadow_bucket": diff_in_strong_shadow,
                "dangerous_diff_pairs": (
                    buckets["bucket_1_strong_face_strong_shadow"]["pairs"] +
                    buckets["bucket_3_weak_face_strong_shadow"]["pairs"]
                ),
                "any_automatic_conclusion_changed": False,  # Shadow scores NEVER change production
                "any_production_decision_changed": False,
                "any_score_changed": False,
            }

        output[ft_key] = ft_results

    return output


# ──────────────────────────────────────────────────────────────────────
# 6. Hard impostor test
# ──────────────────────────────────────────────────────────────────────

def compute_hard_impostor_test(results, per_pair_scores, top_25_impostors):
    """Test shadow scores against the top-25 hard impostors from Phase 3A.

    TELEMETRY ONLY — does not affect scoring.
    """
    impostor_ids = set(p.get("pair_id") for p in top_25_impostors)

    # Build pair lookup
    pair_lookup = {}
    for pair in per_pair_scores:
        pair_lookup[pair["pair_id"]] = pair

    per_impostor = []
    candidate_false_support_25 = {c: 0 for c in _CANDIDATES}
    candidate_false_support_50 = {c: 0 for c in _CANDIDATES}
    candidate_risk_flag_fired = {c: 0 for c in _CANDIDATES}

    for imp in top_25_impostors:
        pair_id = imp.get("pair_id")
        pair_data = pair_lookup.get(pair_id)
        if not pair_data:
            continue

        entry = {
            "pair_id": pair_id,
            "danger_score": imp.get("danger_score"),
            "lr_marks": imp.get("lr_marks"),
            "top_mark_type": imp.get("top_mark_type"),
            "top_region": imp.get("top_region"),
            "candidate_scores": {},
        }

        for cand in _CANDIDATES:
            cand_score = pair_data["candidate_scores"].get(cand, {})
            sc = cand_score.get("shadow_uv_support_score", 0)
            risk = cand_score.get("shadow_uv_risk_flag", False)
            label = cand_score.get("shadow_review_label", "NONE")

            entry["candidate_scores"][cand] = {
                "score": sc,
                "risk_flag": risk,
                "review_label": label,
                "surviving_count": cand_score.get("surviving_correspondence_count", 0),
                "avg_uv": cand_score.get("avg_uv_distance"),
                "high_value_count": cand_score.get("high_value_count", 0),
                "useful_region_count": cand_score.get("useful_region_count", 0),
            }

            if sc >= 25 and not risk:
                candidate_false_support_25[cand] += 1
            if sc >= 50 and not risk:
                candidate_false_support_50[cand] += 1
            if risk:
                candidate_risk_flag_fired[cand] += 1

        per_impostor.append(entry)

    # Find safest candidate
    safest = min(_CANDIDATES, key=lambda c: (candidate_false_support_25[c], candidate_false_support_50[c]))

    return {
        "top_25_tested": len(per_impostor),
        "per_impostor_scores": per_impostor,
        "candidate_false_support_ge_25": candidate_false_support_25,
        "candidate_false_support_ge_50": candidate_false_support_50,
        "candidate_risk_flag_fired": candidate_risk_flag_fired,
        "safest_candidate": safest,
    }


# ──────────────────────────────────────────────────────────────────────
# 7. Dataset limitations
# ──────────────────────────────────────────────────────────────────────

def build_dataset_limitations():
    """Return static dataset limitations dict.

    TELEMETRY ONLY — documents what this simulation can and cannot decide.
    """
    return {
        "dataset": "LFW 50/50 sample",
        "total_pairs": 100,
        "evaluated_pairs": 84,
        "is_sufficient_for_production_scoring": False,
        "is_sufficient_for_shadow_experiment": True,
        "required_for_scoring_approval": "mark_heavy_dataset (>=120 pairs, see MARK_HEAVY_DATASET_STRATEGY.md)",
        "limitations": [
            "LFW has few distinctive marks per subject (dominated by light_scar)",
            "No twin/lookalike pairs - impostor hardness is underestimated",
            "No age-gap pairs - temporal stability unknown",
            "250x250 resolution limits patch descriptor utility",
            "Only 84 evaluated pairs - statistical power for per-region analysis is low",
        ],
        "this_simulation_decides": "which candidate deserves expanded testing on mark-heavy dataset",
        "this_simulation_does_not_decide": "whether to deploy mark UV scoring to production",
    }

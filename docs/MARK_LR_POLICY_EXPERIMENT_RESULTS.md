# Mark LR Policy Simulation Results

This is an offline simulation using the 50/50 LFW validation artifacts. Production scoring was not changed.

## Simulation Results

| Policy | TP | FP | TN | FN | FAR | FRR | Prec | Recall | Diff > 100 LR | Diff > 1k LR | Human Review | FN Recovered | FP Introduced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1_baseline | 40 | 0 | 42 | 2 | 0.00% | 4.76% | 1.00 | 0.95 | 42 | 42 | 0 | 0 | 0 |
| 2_log_cap_only | 40 | 0 | 42 | 2 | 0.00% | 4.76% | 1.00 | 0.95 | 42 | 42 | 0 | 0 | 0 |
| 3_log_cap_plus_floor | 40 | 0 | 42 | 2 | 0.00% | 4.76% | 1.00 | 0.95 | 42 | 42 | 0 | 0 | 0 |
| 4_region_plus_log_cap | 40 | 0 | 42 | 2 | 0.00% | 4.76% | 1.00 | 0.95 | 42 | 42 | 0 | 0 | 0 |
| 5_review_signal_only | 40 | 0 | 42 | 2 | 0.00% | 4.76% | 1.00 | 0.95 | 42 | 42 | 44 | 0 | 0 |
| 6_strict_acceptance_gate | 40 | 0 | 42 | 2 | 0.00% | 4.76% | 1.00 | 0.95 | 0 | 0 | 0 | 0 | 0 |

## Investigation of Different-Person Mark Correspondences

- **Avg Accepted Correspondences (Same Person):** 14.76
- **Avg Accepted Correspondences (Different Person):** 15.86

### Missing Evidence Details
The artifact `validation_results.jsonl` does not contain the detailed mark correspondence array (e.g., spatial distance, specific mark type channels, region diversity). The pipeline currently summarizes `accepted_correspondences_count` and `lr_marks` but drops the granular `details` object from the JSONL output.

**To answer:**
1. Average spatial distance distribution
2. Face region diversity
3. Mark type/channel distribution (freckle vs mole vs pore)
4. Whether LFW image normalization creates random pore/freckle matches
5. Whether repeated generic marks are being treated as rare

We must update `backend/pipeline_core.py` to persist the full `evidence` array into the JSONL artifact in a future run. However, the presence of similar accepted correspondences for *different* people suggests that generic features like pores and freckles are being repeatedly matched at close spatial proximity, driving the astronomical LRs despite the individuals being different.

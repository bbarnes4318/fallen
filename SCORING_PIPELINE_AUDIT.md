# Scoring Pipeline Audit — Mark LR Policy for Exact Self-Match

## Exact Byte-Identical Image Self-Match Policy

**Version:** v2.0.0  
**Effective:** v1.138.0

### Policy

For exact byte-identical image comparisons (where `probe_file_hash == gallery_file_hash`):

- `exact_image_match = true`
- `mark_match_status = "EXACT_SELF_MATCH"`
- `mark_correspondence_score = 100.0`
- `marks_matched = min(len(valid_probe_marks), len(valid_gallery_marks))`
- Correspondences are identity-mapped by index (mark `i` ↔ mark `i`)
- `lr_marks = 1.0` (neutral)
- `mark_lrs = []` (no individual LRs computed)

### Rationale

For exact byte-identical image comparisons, mark LR is **not used as independent evidence** because the probe and gallery are the same source image. The forensic mark correspondence is trivially true (every mark corresponds to itself), so computing Bayesian Likelihood Ratios would be mathematically meaningless — it would always yield perfect correspondence regardless of actual forensic value.

The UI displays **exact self-correspondence** with an explanatory note:

> "Exact image self-match. Mark evidence is self-corresponding by identity."

Identity confidence for self-matches is handled by the **exact-image sanity path** (hash comparison), not by the mark evidence system.

### Frontend Display

- Status banner: `"Exact image self-match. Mark evidence is self-corresponding by identity."`
- Transparency note shown: explains neutral LR_marks = 1.0
- `INSUFFICIENT_MARKS` and `NO_MATCHES` are **never** returned for exact self-matches

### Insufficient Marks Logic

Both `/verify/fuse` and `/vault/search` use **OR** logic:

```python
if len(valid_probe_marks) < 2 or len(valid_gallery_marks) < 2:
    mark_match_status = "INSUFFICIENT_MARKS"
```

If **either** side has fewer than 2 reliable marks, the system cannot make a meaningful mark comparison.

# Mark Evidence Decision Record

## 1. Background
- We added V2 mark detection and matching.
- Dashboard now shows marks.
- Validation framework exists.
- 50/50 LFW sharded validation was run.
- Corrected trace run fixed `mark_type` and `face_region` null bugs.

## 2. Validation Results
- **total evaluated pairs:** 84
- **FACE_NOT_DETECTED:** 16
- **TP / FP / TN / FN under current rule:** TP=40, FP=0, TN=42, FN=2
- **FAR:** 0
- **FRR:** 4.76%
- **same-person avg accepted correspondences:** 14.76
- **different-person avg accepted correspondences:** 15.86
- **same-person avg distinctive marks:** 10.36
- **different-person avg distinctive marks:** 10.48
- **same-person avg generic marks:** 4.40
- **different-person avg generic marks:** 5.38
- **different-person Gate C pass count:** 42/42
- **Gate C alone still passes impostors:** Yes
- **mark LR explosion observed:** Yes

## 3. Decision
- Mark evidence is **NOT** approved for automatic identity confirmation.
- Mark evidence is **NOT** approved to override weak face similarity.
- Gate C is **NOT** approved as a production scoring gate.
- The hard face-similarity safety rule remains required.
- Mark evidence may be shown as explanatory/diagnostic evidence only.
- Mark evidence may trigger Human Review language, but not an automatic match, unless future validation proves otherwise.

## 4. Current Safe Product Behavior
- Face similarity remains the primary automatic identity signal.
- Mark evidence is displayed separately.
- If face evidence and mark evidence conflict, UI should say: "Conflicting Evidence — Human Review Needed."
- Mark evidence should not independently convert a weak face match into a positive same-person result.

## 5. Future Work
Before mark evidence can influence final scoring, we need:
- mark-heavy same-person dataset
- mark-heavy different-person dataset
- lookalike impostor dataset
- age-gap dataset
- stricter mark distinctiveness model
- region/spatial distribution modeling
- generic mark suppression
- evidence of lower false positive risk

## 6. Explicit Non-Changes
- production scoring unchanged
- thresholds unchanged
- Bayesian fusion unchanged
- `/vault/search` untouched
- frontend unchanged unless separate UI copy task is approved

## 7. Recommended Next Engineering Task
Prepare a separate proposal for "Mark Evidence as Review Signal Only":
- no scoring change
- only UI/report wording
- strong marks + weak face = Human Review
- strong marks + strong face = additional supporting explanation
- weak marks + strong face = face result still stands

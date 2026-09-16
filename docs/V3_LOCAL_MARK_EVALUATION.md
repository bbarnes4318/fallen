# Fallen V3 Local-Mark Evaluation

## Purpose

This research workflow measures whether high-resolution local facial features and facial-mark correspondence add identity information beyond the existing global face representation.

It does **not** change the production verification verdict path.

## Pipeline

```text
source image
   |
   +--> existing face alignment / global embeddings (256px baseline)
   |
   +--> high-resolution alignment (1024px)
           |
           +--> existing mark_detector.py
           |
           +--> local_mark_features_v3.py
                  |
                  +--> multi-scale 32/64/128 local/context patches
                  +--> SIFT pooled local descriptor
                  +--> LAB/color + texture descriptor
                  +--> fixed-length local + contextual vectors
           |
           +--> mark matcher
           +--> mark constellation
```

The detector remains unchanged. The local feature layer is an enrichment layer so detector candidates can be evaluated with source-resolution appearance rather than only normalized position/type/shape.

## Dataset construction

Create an image manifest in JSONL with at least:

```json
{"identity_id":"person-001","image_id":"img-001","path":"/data/person-001/img-001.jpg","split":"calibration"}
```

Optional fields include `quality`, `camera_domain`, `age`, `mark_count`, `twin_group`, `sibling_group`, `occlusion`, and `pose`.

Build pair records:

```bash
python scripts/build_v3_pair_dataset.py images.jsonl pairs.jsonl \
  --max-genuine-per-identity 100 \
  --impostor-multiplier 1 \
  --seed 20260916
```

The builder keeps identities within their declared split and labels useful subsets such as twins, siblings, cross-age, cross-camera, poor-quality, and mark-rich comparisons.

## Feature extraction

Run the research pipeline over the pair manifest:

```bash
python scripts/extract_v3_feature_cache.py pairs.jsonl pair_features.jsonl
```

Images are cached in-process so repeated pair comparisons do not repeatedly run the detector/embedding models.

## Ablation

Run identity-disjoint logistic calibration/evaluation:

```bash
python scripts/run_v3_ablation.py pair_features.jsonl v3_ablation_results.json
```

The predefined variants are:

1. `global_only`
2. `global_geometry`
3. `global_legacy_marks`
4. `global_local`
5. `global_local_marks`
6. `global_local_constellation`
7. `full_v3`

Primary reported metrics are EER, TAR @ FAR 1e-3, TAR @ FAR 1e-4, and Brier score.

## Interpretation rule

No component is promoted into production because it looks better on one aggregate metric. The next gate is held-out performance across the difficult subsets, especially siblings, identical twins, cross-age, poor-quality, mark-rich impostors, and cross-camera pairs.

A local feature or constellation layer is useful only if its improvement survives identity-disjoint evaluation and does not materially worsen false-match behavior.

## Learned descriptor roadmap

`local_mark_features_v3.py` uses an explicit backend interface and currently provides a deterministic SIFT + color/texture baseline. A learned local descriptor can be added behind the same interface after its model/weight licensing and Fallen-specific performance are evaluated. The ablation framework is intentionally independent of the descriptor implementation.

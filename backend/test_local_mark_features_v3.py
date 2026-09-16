import numpy as np

from local_mark_features_v3 import (
    augment_detected_marks,
    extract_local_mark_feature,
    local_feature_similarity,
)


def _image():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:] = 120
    # Synthetic localized structures give SIFT/texture extraction something to inspect.
    image[100:120, 110:130] = 30
    image[145:170, 90:115] = 210
    return image


def test_local_feature_is_fixed_length_and_finite():
    mark = {"canonical_position": [0.47, 0.43], "confidence": 0.9, "mark_type": "dark_mole"}
    feature = extract_local_mark_feature(_image(), mark)
    assert len(feature.local_embedding) > 0
    assert len(feature.contextual_embedding) > 0
    assert np.all(np.isfinite(feature.local_embedding))
    assert np.all(np.isfinite(feature.contextual_embedding))


def test_augmentation_preserves_detector_fields():
    marks = [{"canonical_position": [0.47, 0.43], "confidence": 0.9, "mark_type": "dark_mole", "area": 20}]
    enriched = augment_detected_marks(_image(), marks)
    assert len(enriched) == 1
    assert enriched[0]["area"] == 20
    assert "local_embedding" in enriched[0]
    assert "contextual_embedding" in enriched[0]
    assert enriched[0]["local_feature_backend"] == "sift+color_texture"


def test_same_mark_similarity_is_high():
    image = _image()
    mark = {"canonical_position": [0.47, 0.43], "confidence": 0.9, "mark_type": "dark_mole"}
    a = augment_detected_marks(image, [mark])[0]
    b = augment_detected_marks(image, [mark])[0]
    score = local_feature_similarity(a, b)
    assert score is not None
    assert score > 0.99

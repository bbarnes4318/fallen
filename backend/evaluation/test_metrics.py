from metrics import brier_score, cluster_metrics, far_at_tar, tar_at_far


def test_pairwise_operating_metrics():
    y = [1, 1, 0, 0]
    s = [0.95, 0.80, 0.20, 0.10]
    assert tar_at_far(y, s, 0.5)["tar"] >= 0.0
    assert 0.0 <= far_at_tar(y, s, 0.5)["far"] <= 1.0


def test_brier_is_bounded():
    assert 0.0 <= brier_score([1, 0], [0.9, 0.1]) <= 1.0


def test_cluster_metrics_expose_false_merges_and_splits():
    assignments = {"a": "c1", "b": "c1", "c": "c2"}
    truth = {"a": "p1", "b": "p2", "c": "p2"}
    result = cluster_metrics(assignments, truth)
    assert "false_merge_rate" in result
    assert "false_split_rate" in result

import pytest

from tools.path_annotator import detect


def entry(xp_values, first=25, last=47, cls="straight"):
    """first..last を有効行にして xp_values を敷き詰めたエントリを作る。"""
    xp = [0.0] * 48
    h = [0] * 48
    rows = list(range(first, last + 1))
    for k, i in enumerate(rows):
        xp[i] = xp_values[k % len(xp_values)]
        h[i] = 1
    return {"class": cls, "xp": xp, "h_vector": h}


def straight_entry(value=0.5, first=25, last=47):
    return entry([value], first, last)


def test_parse_frame_name_splits_prefix_and_number():
    assert detect.parse_frame_name("630_2ms_nav_000042") == ("630_2ms_nav", 42)


def test_parse_frame_name_handles_dashed_bag_names():
    assert detect.parse_frame_name("rosbag2_2026_08_24-13_21_39_004808") == (
        "rosbag2_2026_08_24-13_21_39",
        4808,
    )


def test_parse_frame_name_rejects_missing_number():
    with pytest.raises(ValueError):
        detect.parse_frame_name("no_number_here")


def test_straight_path_has_no_bend():
    assert detect.frame_metrics([straight_entry()])["bend"] == pytest.approx(0.0)


def test_zigzag_path_has_large_bend():
    assert detect.frame_metrics([entry([0.3, 0.7])])["bend"] > 0.5


def test_centred_path_has_no_edge_contact():
    assert detect.frame_metrics([straight_entry(0.5)])["edge"] == pytest.approx(0.0)


def test_path_pinned_to_left_edge_has_full_edge_score():
    assert detect.frame_metrics([straight_entry(0.005)])["edge"] == pytest.approx(1.0)


def test_path_pinned_to_right_edge_has_full_edge_score():
    assert detect.frame_metrics([straight_entry(0.995)])["edge"] == pytest.approx(1.0)


def test_missing_counts_invalid_rows():
    assert detect.frame_metrics([straight_entry(first=25, last=47)])["missing"] == 25


def test_short_path_has_larger_missing():
    a = detect.frame_metrics([straight_entry(first=40, last=47)])["missing"]
    b = detect.frame_metrics([straight_entry(first=25, last=47)])["missing"]
    assert a > b


def test_contiguous_valid_rows_have_no_gap():
    assert detect.frame_metrics([straight_entry()])["gap"] == 0


def test_holes_inside_valid_span_are_counted_as_gap():
    e = straight_entry()
    e["h_vector"][30] = 0
    e["h_vector"][31] = 0
    assert detect.frame_metrics([e])["gap"] == 2


def test_metrics_take_max_over_entries():
    calm = straight_entry()
    wild = entry([0.3, 0.7], cls="left")
    assert detect.frame_metrics([calm, wild])["bend"] == detect.frame_metrics([wild])["bend"]


def test_empty_valid_rows_do_not_crash():
    e = {"class": "straight", "xp": [0.0] * 48, "h_vector": [0] * 48}
    m = detect.frame_metrics([e])
    assert m["bend"] == 0.0 and m["missing"] == 48


import json

from tools.path_annotator import store as store_mod


def test_identical_neighbour_has_no_jump():
    a = [straight_entry(0.5)]
    assert detect.temporal_metric(a, [a]) == pytest.approx(0.0)


def test_shifted_neighbour_jump_equals_shift():
    a = [straight_entry(0.5)]
    b = [straight_entry(0.6)]
    assert detect.temporal_metric(a, [b]) == pytest.approx(0.1, abs=1e-9)


def test_jump_takes_worst_neighbour():
    a = [straight_entry(0.5)]
    near = [straight_entry(0.52)]
    far = [straight_entry(0.9)]
    assert detect.temporal_metric(a, [near, far]) == pytest.approx(0.4, abs=1e-9)


def test_no_neighbours_means_zero_jump():
    assert detect.temporal_metric([straight_entry(0.5)], []) == pytest.approx(0.0)


def test_jump_ignores_rows_invalid_in_either_frame():
    a = [straight_entry(0.5, first=25, last=47)]
    b = [straight_entry(0.5, first=40, last=47)]
    # b の行 25..39 は無効だが、値としては大きく違うものを入れておく。
    # 無効行を無視できていれば jump は 0 になる。
    for i in range(25, 40):
        b[0]["xp"][i] = 0.99
    assert detect.temporal_metric(a, [b]) == pytest.approx(0.0)


def test_score_frames_ranks_outlier_first():
    records = [
        {"split": "train", "name": f"bag_a_{i:06d}", "metrics": {"bend": 0.0, "edge": 0.0, "missing": 25.0, "gap": 0.0, "jump": 0.0}}
        for i in range(10)
    ]
    records.append(
        {"split": "train", "name": "bag_a_000099", "metrics": {"bend": 0.9, "edge": 0.0, "missing": 25.0, "gap": 0.0, "jump": 0.0}}
    )
    ranked = detect.score_frames(records)
    assert ranked[0]["name"] == "bag_a_000099"


def test_score_frames_names_the_outlier_metric_in_reasons():
    records = [
        {"split": "train", "name": f"bag_a_{i:06d}", "metrics": {"bend": 0.0, "edge": 0.0, "missing": 25.0, "gap": 0.0, "jump": 0.0}}
        for i in range(10)
    ]
    records.append(
        {"split": "train", "name": "bag_a_000099", "metrics": {"bend": 0.9, "edge": 0.0, "missing": 25.0, "gap": 0.0, "jump": 0.0}}
    )
    ranked = detect.score_frames(records)
    assert ranked[0]["reasons"] == ["bend"]


def test_score_frames_handles_zero_variance_without_nan():
    records = [
        {"split": "train", "name": f"bag_a_{i:06d}", "metrics": {k: 1.0 for k in detect.ALL_KEYS}}
        for i in range(5)
    ]
    ranked = detect.score_frames(records)
    assert all(r["score"] == pytest.approx(0.0) for r in ranked)
    assert all(r["reasons"] == [] for r in ranked)


def test_score_dataset_reads_labels_and_ranks(tmp_path):
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    for i in range(6):
        name = f"bag_a_{i * 2:06d}"
        label = [entry([0.3, 0.7])] if i == 3 else [straight_entry(0.5)]
        (tmp_path / "labels" / "train" / f"{name}.txt").write_text(json.dumps(label))
    ranked = detect.score_dataset(store_mod.LabelStore(tmp_path))
    assert ranked[0]["name"] == "bag_a_000006"
    assert len(ranked) == 6


def test_score_dataset_only_compares_within_the_same_bag(tmp_path):
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "bag_a_000000.txt").write_text(
        json.dumps([straight_entry(0.2)])
    )
    (tmp_path / "labels" / "train" / "bag_b_000002.txt").write_text(
        json.dumps([straight_entry(0.9)])
    )
    ranked = detect.score_dataset(store_mod.LabelStore(tmp_path))
    assert all(r["metrics"]["jump"] == pytest.approx(0.0) for r in ranked)


def test_z_scores_are_clipped():
    """ほぼ定数の指標の外れ値 1 件が合成スコアを支配しないこと。"""
    records = [
        {"split": "train", "name": f"bag_a_{i:06d}",
         "metrics": {k: 0.0 for k in detect.ALL_KEYS}}
        for i in range(200)
    ]
    records[0]["metrics"]["gap"] = 2.0
    ranked = detect.score_frames(records)
    assert ranked[0]["name"] == "bag_a_000000"
    assert ranked[0]["score"] <= detect.Z_CLIP * len(detect.ALL_KEYS)


def test_default_weights_exist_and_are_all_one():
    assert detect.DEFAULT_WEIGHTS == {k: 1.0 for k in detect.ALL_KEYS}


def test_score_dataset_ignores_neighbours_further_than_max_gap(tmp_path):
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "bag_a_000000.txt").write_text(
        json.dumps([straight_entry(0.2)])
    )
    (tmp_path / "labels" / "train" / "bag_a_000100.txt").write_text(
        json.dumps([straight_entry(0.9)])
    )
    ranked = detect.score_dataset(store_mod.LabelStore(tmp_path))
    assert all(r["metrics"]["jump"] == pytest.approx(0.0) for r in ranked)

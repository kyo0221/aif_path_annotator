import numpy as np

from tools.path_annotator.vectormap_match import base_stem, bev_match, replace_straight


def test_base_stem():
    assert base_stem("630_2ms_nav_000042__aug_noon_rain") == "630_2ms_nav_000042"
    assert base_stem("630_2ms_nav_000042") == "630_2ms_nav_000042"


def test_bev_match_recovers_lateral_and_yaw_offset():
    xs = np.arange(-5, 40, 0.2)
    lines = np.concatenate([np.stack([xs, np.full_like(xs, y)], 1) for y in (-1.65, 1.65, 4.95)])
    true_pos, true_yaw = np.array([10.0, 0.4]), np.radians(3.0)
    c, s = np.cos(true_yaw), np.sin(true_yaw)
    rel = lines - true_pos
    local = np.stack([rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c], 1)
    bev = local[(local[:, 0] > 1.6) & (local[:, 0] < 7) & (np.abs(local[:, 1]) < 6)]
    r = bev_match(bev, lines, np.array([10.0, 0.0]), 0.0)
    assert abs(r["position"][1] - 0.4) < 0.08 and abs(np.degrees(r["angle"]) - 3.0) < 0.6
    assert r["cost_after"] < r["cost_before"]


def test_replace_straight_keeps_other_entries():
    old = [{"class": "straight", "xp": [0.1] * 48, "h_vector": [1] * 48},
           {"class": "left", "xp": [0.2] * 48, "h_vector": [1] * 48}]
    new = [{"class": "straight", "xp": [0.5] * 48, "h_vector": [1] * 48},
           {"class": "right", "xp": [0.9] * 48, "h_vector": [1] * 48}]
    out = replace_straight(old, new)
    assert [e["class"] for e in out] == ["straight", "left"]
    assert out[0]["xp"][0] == 0.5 and out[1]["xp"][0] == 0.2

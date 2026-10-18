import numpy as np
from scipy.spatial import cKDTree

from tools.path_annotator.matching import NARROW, WIDE, DistanceField, candidates, densify, match
from tools.path_annotator.pose import make_pose, shift_pose

LINE_YS = (-1.65, 1.65, 4.95)


def _polylines():
    xs = np.arange(-5, 40, 0.2)
    return [np.stack([xs, np.full_like(xs, y)], 1) for y in LINE_YS]


def _vehicle_points(pose, polylines):
    """地図の線のうち、姿勢から見て前方 1.6〜7 m・横 ±6 m の点を車両座標で返す。"""
    c, s = np.cos(pose["yaw"]), np.sin(pose["yaw"])
    rel = np.concatenate(polylines) - np.array([pose["x"], pose["y"]])
    local = np.stack([rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c], 1)
    return local[(local[:, 0] > 1.6) & (local[:, 0] < 7) & (np.abs(local[:, 1]) < 6)]


def test_distance_matches_kdtree_within_3cm():
    # 軸に平行な線に加え、斜めの線と、格子点からセルの端数だけずれた線も入れる
    t = np.linspace(0, 30, 300)
    polylines = _polylines() + [np.stack([t, 0.4 * t - 3.0], 1), np.stack([t, np.full_like(t, 3.2 + 0.0173)], 1)]
    field = DistanceField(polylines)
    tree = cKDTree(np.concatenate([densify(p, 0.002) for p in polylines]))
    rng = np.random.default_rng(0)
    xy = np.stack([rng.uniform(0, 30, 3000), rng.uniform(-3, 9, 3000)], 1)
    expected = np.minimum(tree.query(xy)[0], 0.5)
    assert np.max(np.abs(field.distance(xy) - expected)) <= 0.03


def test_distance_outside_grid_is_trunc():
    field = DistanceField(_polylines())
    assert field.distance(np.array([1000.0, 1000.0])) == 0.5
    assert np.all(field.distance(np.array([[1000.0, 1000.0], [-1e6, 5.0]])) == 0.5)


def test_match_wide_recovers_offset():
    polylines = _polylines()
    field = DistanceField(polylines)
    truth = make_pose(10.0, 0.4, np.radians(3.0))
    points = _vehicle_points(truth, polylines)
    r = match(field, points, make_pose(10.0, 0.0, 0.0), WIDE)
    assert abs(r["pose"]["y"] - 0.4) < 0.08
    assert abs(np.degrees(r["pose"]["yaw"]) - 3.0) < 0.6
    assert r["cost"] < r["cost_before"]


def test_match_narrow_recovers_small_offset():
    polylines = _polylines()
    field = DistanceField(polylines)
    truth = make_pose(10.0, 0.3, np.radians(2.0))
    points = _vehicle_points(truth, polylines)
    r = match(field, points, make_pose(10.0, 0.0, 0.0, pitch=1.5), NARROW)
    assert abs(r["pose"]["y"] - 0.3) < 0.06
    assert abs(np.degrees(r["pose"]["yaw"]) - 2.0) < 0.5
    assert abs(r["lateral_m"] - 0.3) < 0.06 and abs(r["yaw_deg"] - 2.0) < 0.5
    assert r["pose"]["pitch"] == 1.5


def _parallel_lines(cross=False):
    xs = np.arange(-5, 60, 0.2)
    lines = [np.stack([xs, np.full_like(xs, y)], 1) for y in (-1.65, 1.65)]
    if cross:
        ys = np.arange(-3, 3.01, 0.1)
        lines.append(np.stack([np.full_like(ys, 15.0), ys], 1))
    return lines


def test_match_keeps_longitudinal_on_parallel_lines():
    """平行線だけでは前後のコストが平らなので、既定の罰則で前後位置が初期値に留まる。"""
    polylines = _parallel_lines()
    field = DistanceField(polylines)
    pose = make_pose(10.0, 0.0, 0.0)
    points = _vehicle_points(pose, polylines)
    r = match(field, points, pose, NARROW)
    assert abs(r["longitudinal_m"]) <= 0.1


def test_match_still_moves_longitudinally_with_a_cross_line():
    """前後の手がかり（直交線）があれば、罰則があっても本当のずれは補正される。"""
    polylines = _parallel_lines(cross=True)
    field = DistanceField(polylines)
    truth = make_pose(10.0, 0.0, 0.0)
    points = _vehicle_points(truth, polylines)
    r = match(field, points, shift_pose(truth, longitudinal=-0.4), NARROW)
    assert abs(r["longitudinal_m"] - 0.4) <= 0.1


def test_candidates_dedupe_and_limit():
    polylines = _polylines()
    field = DistanceField(polylines)
    points = _vehicle_points(make_pose(10.0, 0.0, 0.0), polylines)
    base = make_pose(10.0, 0.0, 0.0)
    same = candidates(field, points, [(base, f"i{i}") for i in range(6)], NARROW)
    assert len(same) == 1 and same[0]["init"] == "i0"

    inits = [(base, "gnss"), (shift_pose(base, lateral=3.3), "lane_left"),
             (shift_pose(base, lateral=-3.3), "lane_right"), (shift_pose(base, longitudinal=5.0), "ahead")]
    out = candidates(field, points, inits, NARROW)
    assert 1 <= len(out) <= 4
    costs = [c["cost"] for c in out]
    assert costs == sorted(costs)
    for c in out:
        assert {"pose", "cost", "lateral_m", "longitudinal_m", "yaw_deg", "init"} <= set(c)
        assert "overlay" not in c


def test_empty_inputs():
    import pytest

    with pytest.raises(ValueError):
        DistanceField([])
    field = DistanceField(_polylines())
    assert match(field, np.empty((0, 2)), make_pose(10.0, 0.0, 0.0), NARROW)["cost"] == 0.5

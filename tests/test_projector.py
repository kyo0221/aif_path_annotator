import numpy as np
import pytest

from tools.path_annotator.pose import make_pose
from tools.path_annotator.store import ROW_START, validate_label

POSE = make_pose(5, 0, 0)


def test_overlay_lines_inside_image(projector):
    lines = projector.overlay(POSE)["lines"]
    assert lines
    pts = np.array([p for line in lines for p in line])
    assert ((pts[:, 0] >= 0) & (pts[:, 0] <= 639) & (pts[:, 1] >= 0) & (pts[:, 1] <= 383)).all()


def _straight(projector, pose):
    entries = [e for e in projector.label(pose) if e["class"] == "straight"]
    assert entries
    xp, valid = np.array(entries[0]["xp"]), np.array(entries[0]["h_vector"], bool)
    return xp[valid]


def test_label_straight_on_centerline(projector):
    xp = _straight(projector, POSE)
    assert len(xp) >= 15
    assert ((xp > 0.4) & (xp < 0.6)).all()
    entry = projector.label(POSE)[0]
    assert projector.overlay(POSE)["rows"]["straight"] == {"xp": entry["xp"], "h_vector": entry["h_vector"]}


def test_label_shifts_with_lateral_pose(projector):
    assert _straight(projector, make_pose(5, 0.5, 0)).mean() > _straight(projector, POSE).mean()


def test_tilt_moves_horizon_up(learning, projector):
    point = np.array([[100.0, 0.0, 0.0]])
    v0 = projector.camera_for(0.0).project(point)[0][0, 1]
    v1 = projector.camera_for(1.0).project(point)[0][0, 1]
    assert 4.0 <= v0 - v1 <= 5.0


def test_bev_points_synthetic(projector):
    mask = np.zeros((384, 640), np.uint8)
    mask[264, 168] = 1
    mask[264, 528] = 255
    pts = projector.bev_points(mask, x_range=(0.5, 10))
    assert len(pts) == 2
    left, right = pts[np.argsort(-pts[:, 1])]
    assert abs(left[0] - 2.0) < 0.1 and abs(right[0] - 2.0) < 0.1
    assert abs(left[1] - 1.3) < 0.1 and abs(right[1] + 1.5) < 0.1


def test_bev_clips_window(projector):
    out = projector.bev(POSE, [[2.0, 0.1]])
    pts = np.array([p for line in out["lines"] for p in line])
    assert len(pts)
    assert ((pts[:, 0] >= -4) & (pts[:, 0] <= 28) & (np.abs(pts[:, 1]) <= 13)).all()
    assert out["points"] == [[2.0, 0.1]]


def test_label_with_pitch_validates(projector):
    label = projector.label(make_pose(5, 0, 0, pitch=3.0))
    assert label
    for entry in label:
        validate_label([entry])
        assert not any(entry["h_vector"][:ROW_START])


def test_label_three_classes_48_rows(projector):
    label = projector.label(make_pose(15, 0, 0))
    assert [e["class"] for e in label] == ["straight", "left", "right"]
    for entry in label:
        assert len(entry["xp"]) == 48 and len(entry["h_vector"]) == 48
    validate_label(label)


# straight.osm のレーン ID: A (x 0→30 の本線) = 9001、E (x=15 で道路を横切る) = 9008
LANE_A, LANE_E = 9001, 9008


def test_lane_candidates_prefers_aligned(projector):
    lanes = projector.lane_candidates(make_pose(14.5, 0.3, 0))
    assert lanes[0]["id"] == LANE_A and lanes[0]["aligned"] and lanes[0]["distance_m"] < 0.5
    e = next(lane for lane in lanes if lane["id"] == LANE_E)
    assert not e["aligned"] and 85 <= e["angle_deg"] <= 95


def test_auto_lanelet_ignores_crossing_lane(projector):
    assert projector.auto_lanelet(make_pose(15, 0, 0)) == LANE_A


def test_lane_candidates_includes_requested_far_lane(projector):
    assert LANE_E in [lane["id"] for lane in projector.lane_candidates(make_pose(5, 0, 0), LANE_E)]


def test_lane_candidates_unknown_id(projector):
    with pytest.raises(ValueError):
        projector.lane_candidates(POSE, 999999)


def test_paths_from_requested_lane(projector):
    pts = projector.paths(make_pose(14.5, 0, 0), LANE_E)["straight"]
    assert (np.abs(pts[:, 0] - 15) < 0.01).all() and (np.diff(pts[:, 1]) > 0).all()


def test_label_with_lanelet_validates(projector):
    pose = make_pose(15, 0, 0)
    for entry in projector.label(pose, LANE_E):  # E は横向きなので有効行 0 で [] のこともある
        validate_label([entry])
    assert projector.resolve_lane(pose, LANE_E)["angle_deg"] >= 60
    assert projector.label(pose, LANE_E) != projector.label(pose)


@pytest.mark.parametrize("call", ["resolve_lane", "paths", "label", "overlay"])
def test_unknown_lanelet_raises_value_error(projector, call):
    with pytest.raises(ValueError):
        getattr(projector, call)(POSE, 999999)


def test_unknown_lanelet_raises_value_error_bev(projector):
    with pytest.raises(ValueError):
        projector.bev(POSE, [], 999999)


def test_bev_has_lanes(projector):
    pose = make_pose(14.5, 0, 0)
    lanes = projector.bev(pose, [[3.0, 0.0]], LANE_E)["lanes"]
    assert {LANE_A, LANE_E} <= {lane["id"] for lane in lanes}
    pts = np.array([p for lane in lanes for p in lane["points"]])
    assert ((pts[:, 0] >= -4) & (pts[:, 0] <= 28) & (np.abs(pts[:, 1]) <= 13)).all()

import json
import math

import pytest

from tools.path_annotator.pose import PoseStore, make_pose, pose_delta, shift_pose, wrap_angle


def test_shift_pose_lateral_is_left():
    p = shift_pose(make_pose(0, 0, 0.0), lateral=1.0)
    assert p["y"] == pytest.approx(1.0) and p["x"] == pytest.approx(0.0)


def test_pose_delta_roundtrip():
    ref = make_pose(3.0, -2.0, 0.7, 1.0)
    p = shift_pose(ref, lateral=0.3, longitudinal=-0.5, dyaw=math.radians(2))
    d = pose_delta(p, ref)
    assert d["lateral_m"] == pytest.approx(0.3, abs=1e-6)
    assert d["longitudinal_m"] == pytest.approx(-0.5, abs=1e-6)
    assert d["yaw_deg"] == pytest.approx(2.0, abs=1e-6)
    assert d["pitch_deg"] == pytest.approx(0.0, abs=1e-6)


def test_wrap_angle():
    assert wrap_angle(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap_angle(math.pi) == pytest.approx(math.pi)


def test_pose_store_roundtrip(tmp_path):
    pose = make_pose(1.0, 2.0, 0.3, 4.0)
    PoseStore(tmp_path).set("train", "a", pose, "auto", 1.5, ["f"], {"k": 1})
    e = PoseStore(tmp_path).get("train", "a")
    assert {k: e[k] for k in pose} == pose
    assert e["source"] == "auto" and e["cost"] == 1.5 and e["flags"] == ["f"]
    assert e["init"] == {"k": 1} and "ts" in e
    assert PoseStore(tmp_path).get("train", "zz") is None
    assert "train/a" in PoseStore(tmp_path).entries()


def test_pose_store_corrupt_file(tmp_path):
    (tmp_path / "poses.json").write_text("not json")
    s = PoseStore(tmp_path)
    assert s.get("train", "a") is None
    s.set("train", "a", make_pose(0, 0, 0), "manual", None, [], None)
    assert "train/a" in json.loads((tmp_path / "poses.json").read_text())["frames"]

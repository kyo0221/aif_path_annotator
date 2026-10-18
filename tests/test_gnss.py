import json
import math

from tools.path_annotator.gnss import GnssIndex


def frame(index, x, y, stamp, yaw_imu=0.0, yaw_motion=0.0, bag="bag"):
    return {
        "bag": bag,
        "image_index": index,
        "stamp": stamp,
        "map": {"x": x, "y": y, "yaw_imu_rad": yaw_imu, "yaw_motion_rad": yaw_motion},
    }


def write_gnss(tmp_path, frames):
    path = tmp_path / "gnss.json"
    path.write_text(json.dumps({"note": "", "bags": {}, "frames": frames}))
    return path


def test_bias_from_moving_frames(tmp_path):
    frames = {
        f"train/f{i}": frame(i, i * 1.0, 0.0, i * 0.1, yaw_imu=0.1, yaw_motion=0.3)
        for i in range(5)
    }
    g = GnssIndex(write_gnss(tmp_path, frames))
    assert math.isclose(g.bias["bag"], 0.2, abs_tol=1e-9)


def test_bias_zero_when_stationary(tmp_path):
    frames = {
        f"train/f{i}": frame(i, 5.0, 5.0, i * 0.1, yaw_imu=0.1, yaw_motion=0.9)
        for i in range(5)
    }
    g = GnssIndex(write_gnss(tmp_path, frames))
    assert g.bias["bag"] == 0.0


def test_initial_pose_applies_bias(tmp_path):
    frames = {
        f"train/f{i}": frame(i, i * 1.0, 0.0, i * 0.1, yaw_imu=0.1, yaw_motion=0.3)
        for i in range(5)
    }
    g = GnssIndex(write_gnss(tmp_path, frames))
    p = g.initial_pose("train/f2")
    assert math.isclose(p["yaw"], 0.3, abs_tol=1e-9)
    assert (p["x"], p["y"], p["pitch"]) == (2.0, 0.0, 0.0)


def test_propagate_adds_deltas(tmp_path):
    frames = {
        "train/a": frame(0, 100.0, 50.0, 0.0, yaw_imu=0.1),
        "train/b": frame(1, 101.0, 50.1, 1.0, yaw_imu=0.15),
    }
    g = GnssIndex(write_gnss(tmp_path, frames))
    prev = {"x": 10.0, "y": 0.0, "yaw": 0.5, "pitch": 3.0}
    p = g.propagate("train/a", prev, "train/b")
    assert math.isclose(p["x"], 11.0)
    assert math.isclose(p["y"], 0.1)
    assert math.isclose(p["yaw"], 0.55)
    assert p["pitch"] == 3.0


def test_unknown_key_raises(tmp_path):
    g = GnssIndex(write_gnss(tmp_path, {"train/f0": frame(0, 0, 0, 0)}))
    assert g.has("train/f0") and not g.has("nope")
    for call in (lambda: g.initial_pose("nope"), lambda: g.propagate("nope", {}, "train/f0")):
        try:
            call()
        except KeyError:
            continue
        raise AssertionError("KeyError expected")


def test_meta_has_bag_and_index(tmp_path):
    g = GnssIndex(write_gnss(tmp_path, {"train/f0": frame(0, 0, 0, 0)}))
    assert g.meta()["train/f0"] == {"bag": "bag", "image_index": 0}

import os
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def learning():
    """学習リポジトリのパス。projection.py が無ければ skip する。"""
    repo = os.environ.get("AIF_LEARNING_REPO", "/home/kyo/formula_ws/src/learning")
    if not (Path(repo) / "Models/data_utils/vision_planner/projection.py").is_file():
        pytest.skip(f"学習リポジトリがありません: {repo}")
    return str(repo)


@pytest.fixture(scope="session")
def projector(learning):
    from tools.path_annotator.projector import load_projector

    return load_projector(learning, DATA / "main_params.yaml", DATA / "straight.osm")


# --- Annotator テスト用の差し替え部品 (torch も学習リポジトリも要らない) ---

import json

import numpy as np


class StubProjector:
    """Projector の最小代替。経路は straight 1 本、y が大きいほど xp が左へ寄る。"""

    def __init__(self, valid_rows=23, n_points=None):
        self.valid_rows = valid_rows
        self.n_points = n_points

    def _entry(self, pose):
        from tools.path_annotator.store import N_ROWS, ROW_START

        xp = float(np.clip(0.5 - 0.1 * pose["y"], 0.0, 1.0))
        h = [0] * N_ROWS
        for row in range(ROW_START, min(N_ROWS, ROW_START + self.valid_rows)):
            h[row] = 1
        return {"class": "straight", "xp": [xp if v else 0.0 for v in h], "h_vector": h}

    def overlay(self, pose):
        return {
            "lines": [[[0.1, 0.9], [0.4, 0.5]]],
            "paths": {"straight": [[0.5, 0.9], [0.5, 0.5]]},
            "rows": self._entry(pose),
        }

    def label(self, pose):
        return [self._entry(pose)]

    def bev_points(self, mask, pitch=0.0, x_range=None):
        xs = np.arange(1.6, 7.0 + 1e-9, 0.2)
        pts = np.array([[x, y] for y in (-1.65, 1.65) for x in xs])
        return pts if self.n_points is None else pts[: self.n_points]

    def bev(self, pose, points):
        return {"lines": [], "paths": {}, "points": np.asarray(points).tolist()}


class FakeDetector:
    def mask(self, image_path, key):
        m = np.zeros((384, 640), np.uint8)
        m[100, 100] = m[101, 101] = 1
        return m

    def cached(self, key):
        return True


@pytest.fixture
def mini_root(tmp_path):
    import cv2

    for split, name in (("train", "f000000"), ("train", "f000002"), ("val", "f000004")):
        d = tmp_path / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{name}.png"), np.zeros((384, 640, 3), np.uint8))
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    h = [0] * 25 + [1] * 23
    (tmp_path / "labels" / "train" / "f000000.txt").write_text(
        json.dumps([{"class": "straight", "xp": [0.5] * 48, "h_vector": h}])
    )
    frames = {}
    for i, (split, name) in enumerate((("train", "f000000"), ("train", "f000002"), ("val", "f000004"))):
        frames[f"{split}/{name}"] = {
            "bag": "bag", "image_index": i * 2, "stamp": i * 0.1,
            "map": {"x": 5.0 + i, "y": 0.3, "yaw_imu_rad": 0.0, "yaw_motion_rad": 0.0},
        }
    (tmp_path / "gnss.json").write_text(json.dumps({"note": "", "bags": {}, "frames": frames}))
    return tmp_path


@pytest.fixture
def make_annotator(mini_root):
    from tools.path_annotator.app import COST_THRESHOLD, Annotator
    from tools.path_annotator.frames import FrameIndex
    from tools.path_annotator.gnss import GnssIndex
    from tools.path_annotator.matching import DistanceField
    from tools.path_annotator.pose import PoseStore
    from tools.path_annotator.store import LabelStore

    def make(projector=None, detector=None, cost_threshold=COST_THRESHOLD):
        store = LabelStore(mini_root)
        gnss = GnssIndex(mini_root / "gnss.json")
        keys = [(s, n) for s in store.splits() for n in store.frames(s)]
        xs = np.linspace(-5, 80, 400)
        field = DistanceField([np.column_stack([xs, np.full_like(xs, y)]) for y in (-1.65, 1.65)])
        return Annotator(
            store, PoseStore(mini_root), FrameIndex(keys, gnss.meta()), gnss,
            projector or StubProjector(), field, detector or FakeDetector(),
            cost_threshold=cost_threshold,
        )

    return make


@pytest.fixture
def annotator(make_annotator):
    return make_annotator()


@pytest.fixture
def annotator_few_rows(make_annotator):
    return make_annotator(projector=StubProjector(valid_rows=5))


@pytest.fixture
def annotator_no_points(make_annotator):
    return make_annotator(projector=StubProjector(n_points=10))

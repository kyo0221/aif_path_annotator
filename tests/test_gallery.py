import json

from tools.path_annotator.frames import FrameIndex
from tools.path_annotator.gallery import Gallery
from tools.path_annotator.gnss import GnssIndex
from tools.path_annotator.pose import PoseStore
from tools.path_annotator.store import LabelStore
from conftest import StubProjector


def make_gallery(root):
    store = LabelStore(root)
    gnss = GnssIndex(root / "gnss.json")
    keys = [(s, n) for s in store.splits() for n in store.frames(s)]
    return Gallery(store, PoseStore(root), FrameIndex(keys, gnss.meta()), gnss, StubProjector()), store


def test_table_lists_all_frames_in_time_order(mini_root):
    g, _ = make_gallery(mini_root)
    t = g.table()
    assert [r["key"] for r in t["frames"]] == ["train/f000000", "train/f000002", "val/f000004"]
    assert t["bags"] == ["bag"] and t["has_manifest"] is False and t["has_relabeled"] is False
    r0 = t["frames"][0]
    assert r0["classes"] == ["straight"] and r0["rows"] == 23 and r0["status"] == "unseen" and r0["edited"] is False
    assert r0["pose"] == "gnss" and r0["flags"] == [] and r0["cost"] is None
    assert t["frames"][1]["classes"] == [] and t["frames"][1]["rows"] == 0
    assert all(r["has_manual_label"] is None and r["relabeled"] is False for r in t["frames"])


def test_table_reads_manifest_and_relabeled(mini_root):
    (mini_root / "manifest.csv").write_text("key,has_manual_label\ntrain/f000000,1\ntrain/f000002,0\n")
    (mini_root / "relabeled_frames.json").write_text(json.dumps({"changed": [{"key": "val/f000004"}]}))
    g, _ = make_gallery(mini_root)
    t = g.table()
    by = {r["key"]: r for r in t["frames"]}
    assert t["has_manifest"] and t["has_relabeled"]
    assert by["train/f000000"]["has_manual_label"] is True and by["train/f000002"]["has_manual_label"] is False
    assert by["val/f000004"]["has_manual_label"] is None and by["val/f000004"]["relabeled"] is True


def test_label_info_follows_file_changes(mini_root):
    g, store = make_gallery(mini_root)
    assert g.label_info("train", "f000002") == {"classes": [], "rows": 0}
    h = [0] * 25 + [1] * 10 + [0] * 13
    store.save("train", "f000002", [{"class": "straight", "xp": [0.5] * 48, "h_vector": h},
                                    {"class": "left", "xp": [0.4] * 48, "h_vector": h}])
    assert g.label_info("train", "f000002") == {"classes": ["straight", "left"], "rows": 10}
    assert g.table()["frames"][1]["status"] == "unseen" and g.table()["frames"][1]["edited"] is True
    assert g.thumb_version("train", "f000002") != "0-0-0"


def test_annotator_has_gallery(annotator):
    assert annotator.gallery.table()["frames"]


import os
import time

import numpy as np


def test_thumb_renders_and_caches_without_touching_poses(mini_root):
    g, store = make_gallery(mini_root)
    before = (mini_root / "poses.json").read_text() if (mini_root / "poses.json").exists() else None
    data = g.thumb("train", "f000002")  # 姿勢エントリ無し → GNSS 初期姿勢で描く
    assert data[:2] == b"\xff\xd8"
    cached = mini_root / "preview" / "train" / "f000002.jpg"
    assert cached.is_file() and cached.read_bytes() == data
    after = (mini_root / "poses.json").read_text() if (mini_root / "poses.json").exists() else None
    assert after == before
    assert g.table()["frames"][1]["pose"] == "gnss"


def test_thumb_rerenders_when_label_is_newer(mini_root):
    g, store = make_gallery(mini_root)
    g.thumb("train", "f000000")
    cached = mini_root / "preview" / "train" / "f000000.jpg"
    old = os.stat(cached).st_mtime_ns
    time.sleep(0.01)
    store.save("train", "f000000", [{"class": "straight", "xp": [0.2] * 48, "h_vector": [0] * 25 + [1] * 23}])
    g.thumb("train", "f000000")
    assert os.stat(cached).st_mtime_ns > old


def test_thumb_without_gnss_or_pose_is_raw_image_with_dots(mini_root):
    gnss = json.loads((mini_root / "gnss.json").read_text())
    del gnss["frames"]["val/f000004"]
    (mini_root / "gnss.json").write_text(json.dumps(gnss))
    g, _ = make_gallery(mini_root)
    assert g.table()["frames"][2]["pose"] == "none"
    data = g.thumb("val", "f000004")
    assert data[:2] == b"\xff\xd8"


def test_draw_overlay_puts_dots_at_label_rows():
    from tools.path_annotator.gallery import draw_overlay

    img = np.zeros((384, 640, 3), np.uint8)
    label = [{"class": "straight", "xp": [0.5] * 48, "h_vector": [0] * 47 + [1]}]
    draw_overlay(img, {"lines": [], "paths": {}, "rows": {}}, label)
    assert tuple(img[379, 319]) == (0, 230, 255)  # 行 47: v = 47*8 + 3.5 → 379.5、u = 0.5*639 → 319.5

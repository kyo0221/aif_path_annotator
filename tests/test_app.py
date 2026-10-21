import cv2
import numpy as np
import pytest

from tools.path_annotator.pose import shift_pose
from tools.path_annotator.store import validate_label


def test_open_first_frame_uses_gnss_and_wide(annotator):
    r = annotator.open_frame("train", "f000000")
    assert r["init"]["source"] == "gnss"
    assert r["source"] == "auto"
    assert abs(r["pose"]["y"]) < 0.1
    assert len(r["candidates"]) <= 4
    assert set(r["cache"]) == {"done", "total"}
    assert r["total"] == 3 and r["index"] == 0


def test_open_next_after_confirm_propagates(annotator):
    r = annotator.open_frame("train", "f000000")
    annotator.confirm("train", "f000000", r["pose"], "auto")
    r2 = annotator.open_frame("train", "f000002")
    assert r2["init"]["source"] == "propagated"
    assert r2["source"] == "propagated"


def test_confirm_writes_label_and_next(annotator, mini_root):
    r = annotator.open_frame("train", "f000000")
    out = annotator.confirm("train", "f000000", r["pose"], "auto")
    validate_label(annotator.store.load("train", "f000000"))
    assert out == {"ok": True, "next": {"split": "train", "name": "f000002"}}
    assert annotator.store.get_status("train", "f000000") == "done"


def test_confirm_rejects_few_rows(annotator_few_rows, mini_root):
    path = mini_root / "labels" / "train" / "f000000.txt"
    before = path.read_text()
    r = annotator_few_rows.open_frame("train", "f000000")
    with pytest.raises(ValueError, match="可視行"):
        annotator_few_rows.confirm("train", "f000000", r["pose"], "auto")
    assert path.read_text() == before


def test_flags_cost_and_jump(annotator, make_annotator):
    r = annotator.open_frame("train", "f000000")
    shifted = shift_pose(r["init"]["pose"], lateral=0.5)
    assert "jump" in annotator.preview("train", "f000000", shifted)["flags"]
    strict = make_annotator(cost_threshold=0.0)
    r = strict.open_frame("train", "f000000")
    shifted = shift_pose(r["init"]["pose"], lateral=0.5)
    assert "cost" in strict.preview("train", "f000000", shifted)["flags"]


def test_queue_flag_filter(annotator):
    r = annotator.open_frame("train", "f000000")
    shifted = shift_pose(r["init"]["pose"], lateral=0.5)
    annotator.set_pose("train", "f000000", shifted, "manual")
    assert len(annotator.queue(flag="any")) == 1
    assert len(annotator.queue(flag="jump")) == 1
    assert len(annotator.queue(status="done")) == 0


def test_missing_gnss_is_value_error(annotator, mini_root):
    cv2.imwrite(str(mini_root / "images" / "val" / "f000099.png"), np.zeros((384, 640, 3), np.uint8))
    a = annotator
    a.frames.__init__(
        [(s, n) for s in a.store.splits() for n in a.store.frames(s)], a.gnss.meta()
    )
    with pytest.raises(ValueError, match="GNSS がありません"):
        a.open_frame("val", "f000099")
    assert a.open_frame("train", "f000000")["split"] == "train"


def test_few_points_skips_matching(annotator_no_points):
    r = annotator_no_points.open_frame("train", "f000000")
    assert r["source"] == "gnss"
    assert "few_points" in r["flags"]
    assert r["candidates"] == []


def test_set_pose_on_done_frame_resets_status(annotator):
    r = annotator.open_frame("train", "f000000")
    annotator.confirm("train", "f000000", r["pose"], "auto")
    annotator.set_pose("train", "f000000", r["pose"], "manual")
    assert annotator.store.get_status("train", "f000000") == "done"
    annotator.set_pose("train", "f000000", shift_pose(r["pose"], lateral=0.2), "manual")
    assert annotator.store.get_status("train", "f000000") == "unseen"


def test_candidate_cache_is_bounded(annotator):
    from tools.path_annotator.app import CANDIDATE_CACHE_SIZE

    assert CANDIDATE_CACHE_SIZE == 32
    for i in range(40):
        annotator._remember_candidates(f"train/f{i:06d}", [{"i": i}])
    assert len(annotator._candidates) == 32
    assert annotator._cached_candidates("train/f000000") is None
    assert annotator._cached_candidates("train/f000007") is None
    assert annotator._cached_candidates("train/f000008") == [{"i": 8}]
    # 参照したキーは新しい扱いになり、次の追加で追い出されない
    annotator._remember_candidates("train/f000040", [])
    assert annotator._cached_candidates("train/f000008") == [{"i": 8}]
    assert annotator._cached_candidates("train/f000009") is None


def test_default_prefers_propagated_within_margin():
    from tools.path_annotator.app import select_default

    found = [{"init": "gnss", "cost": 0.19}, {"init": "propagated", "cost": 0.21}]
    assert select_default(found, "propagated")["init"] == "propagated"
    assert select_default(found, "gnss")["init"] == "gnss"


def test_default_takes_best_beyond_margin():
    from tools.path_annotator.app import select_default

    found = [{"init": "gnss", "cost": 0.19}, {"init": "propagated", "cost": 0.24}]
    assert select_default(found, "propagated")["init"] == "gnss"


def test_open_frame_has_lanes(annotator):
    r = annotator.open_frame("train", "f000000")
    assert r["lanelet"] is None
    assert r["lanes"][0]["auto"] and r["lanes"][0]["selected"]
    assert r["lanes"][1]["aligned"] is False and not r["lanes"][1]["selected"]


def test_preview_with_lanelet(annotator):
    r = annotator.open_frame("train", "f000000")
    p = annotator.preview("train", "f000000", r["pose"], lanelet=2)
    assert "lane" in p["flags"] and "no_path" in p["flags"]
    assert p["label"] == []
    assert p["lanelet"] == 2
    assert p["lanes"][1]["selected"] and not p["lanes"][0]["selected"]


def test_set_pose_saves_lanelet(annotator):
    r = annotator.open_frame("train", "f000000")
    annotator.set_pose("train", "f000000", r["pose"], "manual", lanelet=2)
    assert annotator.poses.get("train", "f000000")["lanelet"] == 2
    assert annotator.open_frame("train", "f000000")["lanelet"] == 2


def test_set_pose_lanelet_change_unconfirms(annotator):
    r = annotator.open_frame("train", "f000000")
    annotator.confirm("train", "f000000", r["pose"], "auto")
    annotator.set_pose("train", "f000000", r["pose"], "manual", lanelet=1)
    assert annotator.store.get_status("train", "f000000") == "unseen"


def test_confirm_with_crossing_lane_is_rejected(annotator):
    r = annotator.open_frame("train", "f000000")
    with pytest.raises(ValueError, match="レーンを選び直す"):
        annotator.confirm("train", "f000000", r["pose"], "auto", lanelet=2)


def test_confirm_with_lanelet_saves_it(annotator):
    r = annotator.open_frame("train", "f000000")
    assert annotator.confirm("train", "f000000", r["pose"], "auto", lanelet=1)["ok"]
    assert annotator.poses.get("train", "f000000")["lanelet"] == 1


def test_lane_flag_when_auto_falls_back(make_annotator):
    from conftest import StubProjector
    a = make_annotator(projector=StubProjector(lanes=[]))
    assert "lane" in a.open_frame("train", "f000000")["flags"]


def test_unknown_lanelet_is_value_error(annotator):
    r = annotator.open_frame("train", "f000000")
    with pytest.raises(ValueError):
        annotator.preview("train", "f000000", r["pose"], lanelet=999)


def test_preview_reads_mask_once(make_annotator):
    from conftest import FakeDetector

    class Counting(FakeDetector):
        calls = 0

        def mask(self, image_path, key):
            Counting.calls += 1
            return super().mask(image_path, key)

    a = make_annotator(detector=Counting())
    r = a.open_frame("train", "f000000")
    Counting.calls = 0
    a.preview("train", "f000000", r["pose"])
    assert Counting.calls <= 1


def test_open_frame_tolerates_stale_lanelet(annotator):
    r = annotator.open_frame("train", "f000000")
    annotator.poses.set("train", "f000000", r["pose"], r["source"], r["cost"], r["flags"], r["init"], lanelet=999)
    r2 = annotator.open_frame("train", "f000000")
    assert r2["lanelet"] is None
    assert "999" in r2["lane_note"]


def test_save_writes_label_and_marks_done(annotator, mini_root):
    out = annotator.save("train", "f000002", {"x": 6.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    assert out["saved"] is True and out["status"] == "done" and out["reasons"] == []
    assert out["classes"] == ["straight"] and out["edited"] is True and out["thumb_version"] != "0-0-0"
    assert (mini_root / "labels" / "train" / "f000002.txt").is_file()
    assert annotator.poses.get("train", "f000002")["x"] == 6.0
    again = annotator.save("train", "f000002", {"x": 6.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    assert again["saved"] is True and again["status"] == "done"


def test_save_same_label_keeps_edited(annotator, mini_root):
    first = annotator.save("train", "f000002", {"x": 6.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    annotator.store._update_frame_state("train", "f000002", edited=False)
    same = annotator.save("train", "f000002", {"x": 6.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    assert first["saved"] and same["saved"] and same["edited"] is False


def test_save_rejected_keeps_label_and_unconfirms(annotator_few_rows, mini_root):
    a = annotator_few_rows
    a.store.set_status("train", "f000000", "done")
    before = (mini_root / "labels" / "train" / "f000000.txt").read_text()
    out = a.save("train", "f000000", {"x": 5.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    assert out["saved"] is False and out["status"] == "unseen" and out["reasons"]
    assert (mini_root / "labels" / "train" / "f000000.txt").read_text() == before
    assert a.poses.get("train", "f000000")["x"] == 5.0


def test_save_without_straight_reports_reason(annotator):
    out = annotator.save("train", "f000000", {"x": 5.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual", lanelet=2)
    assert out["saved"] is False and "straight" in out["reasons"][0]


def test_rejected_save_changes_thumb_version(annotator_few_rows):
    """棄却された保存でも姿勢は変わるので、一覧のサムネイルは描き直されなければならない。"""
    a = annotator_few_rows
    v0 = a.gallery.thumb_version("train", "f000002")
    out = a.save("train", "f000002", {"x": 6.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    assert out["saved"] is False
    assert a.gallery.thumb_version("train", "f000002") != v0


def test_open_frame_includes_thumb_version(annotator):
    """開いただけで姿勢が初期化されるので、一覧がカードを更新できるよう版を返す。"""
    f = annotator.open_frame("train", "f000002")
    assert f["thumb_version"] == annotator.gallery.thumb_version("train", "f000002")
    assert f["thumb_version"] != "0-0-0"

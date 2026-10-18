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

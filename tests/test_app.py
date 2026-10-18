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

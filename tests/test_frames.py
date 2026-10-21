import pytest

from tools.path_annotator.frames import MAX_INDEX_GAP, FrameIndex, parse_frame_name


def _meta(*items):
    return {f"{s}/{n}": {"bag": b, "image_index": i} for s, n, b, i in items}


def _index():
    meta = _meta(("train", "a", "bag1", 0), ("val", "b", "bag1", 2), ("train", "c", "bag1", 4))
    return FrameIndex([("train", "c"), ("val", "b"), ("train", "a")], meta)


def test_order_mixes_splits():
    assert _index().order == [("train", "a"), ("val", "b"), ("train", "c")]


def test_prev_next_cross_split():
    idx = _index()
    assert idx.next("train", "a") == ("val", "b")
    assert idx.prev("val", "b") == ("train", "a")
    assert idx.prev("train", "a") is None
    assert idx.next("train", "c") is None
    assert idx.position("val", "b") == 1
    assert len(idx) == 3


def test_predecessor_gap():
    meta = _meta(("train", "a", "bag1", 0), ("train", "b", "bag1", 2), ("train", "c", "bag1", 10))
    idx = FrameIndex([("train", "a"), ("train", "b"), ("train", "c")], meta)
    assert MAX_INDEX_GAP == 4
    assert idx.predecessor("train", "b") == ("train", "a")
    assert idx.predecessor("train", "c") is None
    assert idx.predecessor("train", "a") is None


def test_predecessor_other_bag_is_none():
    meta = _meta(("train", "a", "bag1", 0), ("train", "b", "bag2", 1))
    idx = FrameIndex([("train", "a"), ("train", "b")], meta)
    assert idx.predecessor("train", "b") is None


def test_fallback_without_meta():
    keys = [("train", "x_000005"), ("val", "x_000001"), ("train", "weird")]
    idx = FrameIndex(keys, {})
    assert idx.order == [("train", "weird"), ("val", "x_000001"), ("train", "x_000005")]
    assert parse_frame_name("x_000005") == ("x", 5)


def test_unknown_frame_raises_keyerror():
    import pytest

    idx = _index()
    for fn in (idx.position, idx.prev, idx.next, idx.predecessor):
        with pytest.raises(KeyError):
            fn("train", "nope")


def test_bag_index_is_public():
    index = FrameIndex([("train", "bagA_000010"), ("val", "bagA_000012")], {})
    assert index.bag_index("train", "bagA_000010") == ("bagA", 10)
    with pytest.raises(KeyError):
        index.bag_index("train", "nope_000000")

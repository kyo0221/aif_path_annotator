from tools.path_annotator.store import LabelStore, N_ROWS, ROW_START, rejection_reasons


def _entry(cls="straight", visible=23, xp=0.5):
    h = [0] * N_ROWS
    for i in range(visible):
        h[N_ROWS - 1 - i] = 1
    return {"class": cls, "xp": [xp] * N_ROWS, "h_vector": h}


def test_constants():
    assert (N_ROWS, ROW_START) == (48, 25)


def test_rejection_reasons_min_rows():
    assert rejection_reasons([_entry(visible=9)]) == ["可視行 9 < 10"]


def test_rejection_reasons_edge():
    e = _entry(visible=20)
    visible_rows = [i for i, f in enumerate(e["h_vector"]) if f]
    for i in visible_rows[:12]:
        e["xp"][i] = 0.0
    reasons = rejection_reasons([e])
    assert len(reasons) == 1 and "画像端" in reasons[0]


def test_rejection_reasons_ok():
    assert rejection_reasons([_entry(visible=23)]) == []


def test_rejection_reasons_empty():
    assert rejection_reasons([]) == ["経路なし"]


def test_frames_listed_from_images(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    for n in ("a", "b"):
        (tmp_path / "images" / "train" / f"{n}.png").write_bytes(b"")
    (tmp_path / "labels" / "train" / "a.txt").write_text("[]")
    s = LabelStore(tmp_path)
    assert s.splits() == ["train"]
    assert s.frames("train") == ["a", "b"]
    assert s.has_label("train", "a") is True
    assert s.has_label("train", "b") is False

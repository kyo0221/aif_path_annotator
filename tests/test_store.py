import json
import pytest

from tools.path_annotator import store


def make_entry(cls="straight", first=25, last=47, value=0.5):
    xp = [0.0] * 48
    h = [0] * 48
    for i in range(first, last + 1):
        xp[i] = value
        h[i] = 1
    return {"class": cls, "xp": xp, "h_vector": h}


@pytest.fixture
def dataset(tmp_path):
    """labels/{train,val} と images/{train,val} を持つ最小データセット。"""
    for split in ("train", "val"):
        (tmp_path / "labels" / split).mkdir(parents=True)
        (tmp_path / "images" / split).mkdir(parents=True)
    (tmp_path / "labels" / "train" / "bag_a_000042.txt").write_text(
        json.dumps([make_entry()])
    )
    (tmp_path / "labels" / "train" / "bag_a_000044.txt").write_text(
        json.dumps([make_entry(value=0.6)])
    )
    (tmp_path / "labels" / "val" / "bag_b_000010.txt").write_text(
        json.dumps([make_entry("left")])
    )
    (tmp_path / "images" / "train" / "bag_a_000042.png").write_bytes(b"fakepng")
    return tmp_path


def test_splits_are_discovered(dataset):
    assert store.LabelStore(dataset).splits() == ["train", "val"]


def test_frames_are_sorted_stems(dataset):
    s = store.LabelStore(dataset)
    assert s.frames("train") == ["bag_a_000042", "bag_a_000044"]


def test_load_returns_parsed_label(dataset):
    label = store.LabelStore(dataset).load("train", "bag_a_000042")
    assert label[0]["class"] == "straight"
    assert len(label[0]["xp"]) == 48


def test_save_load_round_trip(dataset):
    s = store.LabelStore(dataset)
    label = s.load("train", "bag_a_000042")
    label[0]["xp"][30] = 0.123
    s.save("train", "bag_a_000042", label)
    assert s.load("train", "bag_a_000042")[0]["xp"][30] == 0.123


def test_save_leaves_no_tmp_file(dataset):
    s = store.LabelStore(dataset)
    s.save("train", "bag_a_000042", s.load("train", "bag_a_000042"))
    assert list((dataset / "labels" / "train").glob("*.tmp")) == []


def test_save_rejects_wrong_length_xp(dataset):
    s = store.LabelStore(dataset)
    bad = [make_entry()]
    bad[0]["xp"] = [0.5] * 47
    with pytest.raises(ValueError, match="48"):
        s.save("train", "bag_a_000042", bad)


def test_save_rejects_unknown_class(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="class"):
        s.save("train", "bag_a_000042", [make_entry("diagonal")])


def test_save_rejects_out_of_range_xp(dataset):
    s = store.LabelStore(dataset)
    bad = [make_entry()]
    bad[0]["xp"][30] = 1.7
    with pytest.raises(ValueError, match="range"):
        s.save("train", "bag_a_000042", bad)


def test_save_rejects_non_binary_h_vector(dataset):
    s = store.LabelStore(dataset)
    bad = [make_entry()]
    bad[0]["h_vector"][30] = 2
    with pytest.raises(ValueError, match="h_vector"):
        s.save("train", "bag_a_000042", bad)


def test_rejected_save_leaves_original_intact(dataset):
    s = store.LabelStore(dataset)
    before = (dataset / "labels" / "train" / "bag_a_000042.txt").read_text()
    with pytest.raises(ValueError):
        s.save("train", "bag_a_000042", [make_entry("diagonal")])
    assert (dataset / "labels" / "train" / "bag_a_000042.txt").read_text() == before


def test_save_rejects_label_without_straight(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="straight"):
        s.save("train", "bag_a_000042", [make_entry("left")])


def test_save_rejects_duplicate_classes(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="duplicate"):
        s.save("train", "bag_a_000042", [make_entry("straight"), make_entry("straight")])


def test_save_accepts_straight_plus_one_branch(dataset):
    s = store.LabelStore(dataset)
    s.save("train", "bag_a_000042", [make_entry("straight"), make_entry("left")])
    assert len(s.load("train", "bag_a_000042")) == 2


def test_image_path_points_at_png(dataset):
    s = store.LabelStore(dataset)
    assert s.image_path("train", "bag_a_000042").name == "bag_a_000042.png"


def test_save_rejects_path_traversal(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="name"):
        s.save("train", "../../etc/passwd", [make_entry()])


def test_save_rejects_boolean_h_vector(dataset):
    s = store.LabelStore(dataset)
    bad = [make_entry()]
    bad[0]["h_vector"][30] = True
    with pytest.raises(ValueError, match="h_vector"):
        s.save("train", "bag_a_000042", bad)


def test_save_rejects_boolean_xp(dataset):
    s = store.LabelStore(dataset)
    bad = [make_entry()]
    bad[0]["xp"][30] = True
    with pytest.raises(ValueError, match="xp"):
        s.save("train", "bag_a_000042", bad)


def test_save_rejects_non_dict_entry(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="entry"):
        s.save("train", "bag_a_000042", ["straight"])


def test_save_rejects_traversal_in_split(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="name"):
        s.save("../images", "bag_a_000042", [make_entry()])


def test_name_check_rejects_trailing_newline(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="name"):
        s.save("train", "bag_a_000042\n", [make_entry()])


def test_concurrent_saves_never_corrupt_the_label(dataset):
    """同一フレームへ複数スレッドから同時に保存しても、常に読める JSON が残る。"""
    import json as _json
    import threading

    s = store.LabelStore(dataset)
    path = dataset / "labels" / "train" / "bag_a_000042.txt"
    errors = []

    def writer(value):
        try:
            for _ in range(20):
                label = [make_entry(value=value)]
                s.save("train", "bag_a_000042", label)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(v,)) for v in (0.1, 0.2, 0.3, 0.4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    loaded = _json.loads(path.read_text())          # 壊れていれば JSONDecodeError
    assert len(loaded[0]["xp"]) == 48
    assert loaded[0]["xp"][30] in (0.1, 0.2, 0.3, 0.4)   # どれか1つの書き手の結果が丸ごと残る
    assert list((dataset / "labels" / "train").glob("*.tmp")) == []


def test_save_rejects_dotdot_split(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="name"):
        s.save("..", "bag_a_000042", [make_entry()])


def test_frames_rejects_dotdot_split(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="name"):
        s.frames("..")


def test_image_path_rejects_dotdot_split(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="name"):
        s.image_path("..", "bag_a_000042")


def test_save_rejects_single_dot_name(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="name"):
        s.save("train", ".", [make_entry()])


def test_saved_label_uses_the_umask_derived_mode(dataset):
    """mkstemp の 0600 が保存後のラベルに残らず、umask 由来のモードになること。"""
    s = store.LabelStore(dataset)
    s.save("train", "bag_a_000042", s.load("train", "bag_a_000042"))
    mode = (dataset / "labels" / "train" / "bag_a_000042.txt").stat().st_mode & 0o777
    assert mode == store._FILE_MODE


def test_saved_label_is_never_world_writable(dataset):
    import stat

    s = store.LabelStore(dataset)
    s.save("train", "bag_a_000042", s.load("train", "bag_a_000042"))
    mode = (dataset / "labels" / "train" / "bag_a_000042.txt").stat().st_mode
    assert not (mode & stat.S_IWOTH)


def test_ensure_backup_creates_copy_on_first_call(dataset):
    s = store.LabelStore(dataset)
    assert s.ensure_backup() is True
    assert (dataset / "labels_backup" / "train" / "bag_a_000042.txt").exists()


def test_ensure_backup_is_noop_on_second_call(dataset):
    s = store.LabelStore(dataset)
    s.ensure_backup()
    marker = dataset / "labels_backup" / "train" / "bag_a_000042.txt"
    marker.write_text("ORIGINAL")
    assert s.ensure_backup() is False
    assert marker.read_text() == "ORIGINAL"


def test_backup_survives_label_edit(dataset):
    s = store.LabelStore(dataset)
    s.ensure_backup()
    label = s.load("train", "bag_a_000042")
    label[0]["xp"][30] = 0.999
    s.save("train", "bag_a_000042", label)
    backed_up = json.loads(
        (dataset / "labels_backup" / "train" / "bag_a_000042.txt").read_text()
    )
    assert backed_up[0]["xp"][30] == 0.5


def test_state_defaults_to_unseen(dataset):
    s = store.LabelStore(dataset)
    assert s.get_status("train", "bag_a_000042") == "unseen"


def test_set_status_persists_across_instances(dataset):
    store.LabelStore(dataset).set_status("train", "bag_a_000042", "done")
    assert store.LabelStore(dataset).get_status("train", "bag_a_000042") == "done"


def test_set_status_rejects_unknown_status(dataset):
    s = store.LabelStore(dataset)
    with pytest.raises(ValueError, match="status"):
        s.set_status("train", "bag_a_000042", "maybe")


def test_state_file_lives_at_dataset_root(dataset):
    store.LabelStore(dataset).set_status("train", "bag_a_000042", "done")
    assert (dataset / "annotation_state.json").exists()


def test_save_records_edited_flag(dataset):
    s = store.LabelStore(dataset)
    s.save("train", "bag_a_000042", s.load("train", "bag_a_000042"))
    state = s.load_state()
    assert state["frames"]["train/bag_a_000042"]["edited"] is True


def test_concurrent_saves_never_corrupt_the_state_file(dataset):
    """複数スレッドが同時に保存しても annotation_state.json が壊れないこと。"""
    import json as _json
    import threading

    s = store.LabelStore(dataset)
    errors = []

    def writer(name):
        try:
            for _ in range(20):
                s.save("train", name, s.load("train", name))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    names = ["bag_a_000042", "bag_a_000044"] * 2
    threads = [threading.Thread(target=writer, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    state = _json.loads((dataset / "annotation_state.json").read_text())  # 壊れていれば JSONDecodeError
    assert state["frames"]["train/bag_a_000042"]["edited"] is True
    assert list(dataset.glob("*.tmp")) == []


def test_concurrent_saves_of_distinct_frames_do_not_raise(dataset):
    """異なるフレームを並行保存しても save() が例外を返さないこと。

    状態辞書を共有したまま json.dump に渡すと、別スレッドの新規キー挿入で
    RuntimeError: dictionary changed size during iteration になる。
    """
    import json as _json
    import threading

    src = _json.loads((dataset / "labels" / "train" / "bag_a_000042.txt").read_text())
    for i in range(40):
        (dataset / "labels" / "train" / f"gen_{i:06d}.txt").write_text(_json.dumps(src))

    s = store.LabelStore(dataset)
    errors = []

    def writer(lo, hi):
        try:
            for i in range(lo, hi):
                n = f"gen_{i:06d}"
                s.save("train", n, s.load("train", n))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(lo, lo + 10)) for lo in (0, 10, 20, 30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    state = _json.loads((dataset / "annotation_state.json").read_text())
    assert len(state["frames"]) == 40


def test_corrupt_state_file_does_not_break_startup(dataset):
    """壊れた annotation_state.json があっても起動でき、空として扱われること。"""
    (dataset / "annotation_state.json").write_text("{ this is not json")
    s = store.LabelStore(dataset)
    assert s.get_status("train", "bag_a_000042") == "unseen"
    s.set_status("train", "bag_a_000042", "done")
    assert store.LabelStore(dataset).get_status("train", "bag_a_000042") == "done"


def test_state_file_with_wrong_shape_is_normalised(dataset):
    (dataset / "annotation_state.json").write_text('["not", "a", "dict"]')
    s = store.LabelStore(dataset)
    assert s.load_state()["frames"] == {}


def test_ensure_backup_leaves_nothing_behind_when_copy_fails(dataset, monkeypatch):
    """コピーが途中で失敗したら、中途半端な labels_backup/ も一時ディレクトリも残さない。"""
    import shutil as _shutil

    s = store.LabelStore(dataset)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_shutil, "copytree", boom)
    with pytest.raises(OSError):
        s.ensure_backup()

    assert not (dataset / "labels_backup").exists()
    assert list(dataset.glob(".labels_backup-*")) == []

    monkeypatch.undo()
    assert s.ensure_backup() is True
    assert (dataset / "labels_backup" / "train" / "bag_a_000042.txt").exists()


def test_save_rejects_valid_rows_above_row_start(dataset):
    """行 0〜24 に点があるラベルを拒否すること（学習に使われない領域）。"""
    s = store.LabelStore(dataset)
    bad = [make_entry()]
    bad[0]["h_vector"][10] = 1
    bad[0]["xp"][10] = 0.5
    with pytest.raises(ValueError, match="row"):
        s.save("train", "bag_a_000042", bad)


def test_save_accepts_labels_with_no_points_above_row_start(dataset):
    s = store.LabelStore(dataset)
    s.save("train", "bag_a_000042", [make_entry()])   # 行25-47のみ有効
    assert len(s.load("train", "bag_a_000042")) == 1

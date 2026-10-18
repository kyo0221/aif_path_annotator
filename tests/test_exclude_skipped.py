"""exclude_skipped.py のテスト。"""
import json

from tools.path_annotator import exclude_skipped


def _make_root(tmp_path):
    for split, name in (("train", "a"), ("train", "b")):
        (tmp_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / split / f"{name}.png").write_bytes(b"img")
        (tmp_path / "labels" / split / f"{name}.txt").write_text("lbl")
    state = {"frames": {"train/a": {"status": "skip"},
                        "train/b": {"status": "done"}}}
    (tmp_path / "annotation_state.json").write_text(json.dumps(state))


def test_moves_skipped_frames(tmp_path):
    _make_root(tmp_path)
    assert exclude_skipped.main(["--root", str(tmp_path)]) == 0
    assert (tmp_path / "excluded/train/a.png").read_bytes() == b"img"
    assert (tmp_path / "excluded/train/a.txt").read_text() == "lbl"
    assert not (tmp_path / "images/train/a.png").exists()
    assert not (tmp_path / "labels/train/a.txt").exists()
    assert (tmp_path / "images/train/b.png").exists()
    assert (tmp_path / "labels/train/b.txt").exists()
    assert not (tmp_path / "excluded/train/b.png").exists()


def test_dry_run_moves_nothing(tmp_path, capsys):
    _make_root(tmp_path)
    assert exclude_skipped.main(["--root", str(tmp_path), "--dry-run"]) == 0
    assert "train/a" in capsys.readouterr().out
    assert (tmp_path / "images/train/a.png").exists()
    assert (tmp_path / "labels/train/a.txt").exists()
    assert not (tmp_path / "excluded").exists()


def test_missing_state_is_noop(tmp_path):
    assert exclude_skipped.main(["--root", str(tmp_path)]) == 0
    assert not (tmp_path / "excluded").exists()

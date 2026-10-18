from pathlib import Path

import numpy as np
import pytest

from tools.path_annotator import annotate

DATA = Path(__file__).parent / "data"


def test_main_requires_vectormap_args(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        annotate.main(["--root", str(tmp_path)])
    assert e.value.code == 2
    assert "--vectormap" in capsys.readouterr().err


def test_build_app_wires_frames_in_time_order(mini_root, monkeypatch):
    from conftest import StubProjector

    monkeypatch.setattr(annotate, "load_projector", lambda *a, **k: StubProjector())
    monkeypatch.setattr(
        annotate, "load_yolop", lambda weights: (lambda image: np.zeros((384, 640), np.float32))
    )
    annotator, prefetcher = annotate.build_app(
        mini_root, DATA / "straight.osm", "yolop.pt", DATA / "main_params.yaml", "repo"
    )
    expected = [("train", "f000000"), ("train", "f000002"), ("val", "f000004")]
    assert annotator.frames.order == expected
    assert annotator.prefetcher is prefetcher
    assert [k for k, _ in prefetcher.items] == [f"{s}/{n}" for s, n in expected]
    assert prefetcher.items[0][1] == str(mini_root / "images" / "train" / "f000000.png")
    assert prefetcher.detector.cache_dir == mini_root / "cache" / "yolop"


def test_build_app_requires_gnss(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    with pytest.raises(ValueError, match="gnss.json"):
        annotate.build_app(tmp_path, "a.osm", "y.pt", "p.yaml", "repo")

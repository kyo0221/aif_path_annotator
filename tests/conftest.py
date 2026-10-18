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

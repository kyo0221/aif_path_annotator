"""yolop（マスクのディスクキャッシュと先読み）のテスト。torch は使わない。"""
import threading
import time

import cv2
import numpy as np

from tools.path_annotator.yolop import LaneDetector, Prefetcher


def fake_model(calls):
    """呼び出しごとに calls へ画像を記録し、左右 2 本の縦線を 1.0 にした (384,640) を返す。"""
    def model(image):
        calls.append(image)
        prob = np.zeros((384, 640), np.float32)
        prob[:, 100:104] = 1.0
        prob[:, 500:504] = 1.0
        return prob
    return model


def write_image(path, value=0):
    img = np.full((384, 640, 3), value, np.uint8)
    cv2.imwrite(str(path), img)
    return path


def test_mask_cached_on_disk(tmp_path):
    calls = []
    det = LaneDetector(fake_model(calls), tmp_path / "cache")
    img = write_image(tmp_path / "a.png")
    assert not det.cached("train/frame_000010")
    m1 = det.mask(img, "train/frame_000010")
    m2 = det.mask(img, "train/frame_000010")
    assert len(calls) == 1
    assert (tmp_path / "cache" / "train" / "frame_000010.png").exists()
    assert det.cached("train/frame_000010")
    assert m1.dtype == np.uint8 and set(np.unique(m1)) == {0, 1}
    assert m1.shape == (384, 640)
    assert np.array_equal(m1, m2)


def test_corrupt_cache_is_recomputed(tmp_path):
    calls = []
    det = LaneDetector(fake_model(calls), tmp_path / "cache")
    img = write_image(tmp_path / "a.png")
    png = tmp_path / "cache" / "k.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"garbage")
    m = det.mask(img, "k")
    assert len(calls) == 1 and m.sum() > 0
    assert cv2.imread(str(png), cv2.IMREAD_GRAYSCALE) is not None


def test_concurrent_same_key_computes_once(tmp_path):
    calls = []
    base = fake_model(calls)

    def slow(image):
        time.sleep(0.2)
        return base(image)

    det = LaneDetector(slow, tmp_path / "cache")
    img = write_image(tmp_path / "a.png")
    results = []
    threads = [threading.Thread(target=lambda: results.append(det.mask(img, "k"))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(results) == 2
    assert np.array_equal(results[0], results[1])


def test_prefetcher_from_cursor(tmp_path):
    # 画像の画素値を k の番号にして、モデルに渡った画像から計算順を復元する
    order = []

    def model(image):
        order.append(int(image[0, 0, 0]))
        return np.zeros(image.shape[:2], np.float32)

    det = LaneDetector(model, tmp_path / "cache")
    items = [(f"k{i}", str(write_image(tmp_path / f"{i}.png", i))) for i in range(5)]
    pf = Prefetcher(det, items)
    pf.set_cursor("k3")
    pf.set_cursor("nope")  # 無視される
    pf.start()
    deadline = time.time() + 10
    while pf.progress() != (5, 5) and time.time() < deadline:
        time.sleep(0.01)
    pf.stop()
    assert pf.progress() == (5, 5)
    assert order[:2] == [3, 4]
    assert sorted(order) == [0, 1, 2, 3, 4]

"""YOLOPv2 の車線マスクをディスクにキャッシュし、時系列順に先読みする。

モデル呼び出し（約 2.5 秒/枚）は Lock で直列化し、各フレームは一度しか推論しない。
torch は load_yolop の中でだけ import する（テストや他モジュールが torch なしで使えるように）。
"""
import os
import threading
from pathlib import Path

import cv2
import numpy as np


class LaneDetector:
    """model: (image_bgr[H,W,3] uint8) -> 確率マップ[H,W]（0〜1）。"""

    def __init__(self, model, cache_dir, threshold=0.5):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.threshold = threshold
        self._model_lock = threading.Lock()
        self._cond = threading.Condition()
        self._in_progress = set()

    def _png(self, key):
        return self.cache_dir / f"{key}.png"

    def cached(self, key):
        """キャッシュ PNG が存在するか（デコードはしない）。"""
        return self._png(key).exists()

    def _read_cache(self, key, shape):
        img = cv2.imread(str(self._png(key)), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != shape:
            return None
        return (img > 127).astype(np.uint8)

    def mask(self, image_path, key):
        """車線マスク（H,W uint8 {0,1}）を返す。キャッシュがあれば読むだけ。"""
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"画像を読めません: {image_path}")
        shape = image.shape[:2]
        with self._cond:
            # 同じ key を別スレッドが計算中なら、終わるまで待ってからキャッシュを読む
            while key in self._in_progress:
                self._cond.wait()
            if self.cached(key):
                cached = self._read_cache(key, shape)
                if cached is not None:
                    return cached
            self._in_progress.add(key)
        try:
            with self._model_lock:
                prob = np.asarray(self.model(image))
            result = (prob > self.threshold).astype(np.uint8)
            png = self._png(key)
            png.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(png), result * 255):
                raise OSError(f"キャッシュを書けません: {png}")
            return result
        finally:
            with self._cond:
                self._in_progress.discard(key)
                self._cond.notify_all()


def load_yolop(weights, threads=None):
    """TorchScript の YOLOPv2 を CPU で読み込み、BGR 画像 -> 車線確率マップ float32 (H,W) の関数を返す。"""
    import torch

    torch.set_num_threads(threads or max(1, (os.cpu_count() or 2) // 2))
    net = torch.jit.load(str(weights), map_location="cpu").eval()

    def model(image_bgr):
        x = torch.from_numpy(image_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)[None]
        with torch.no_grad():
            out = net(x)[2][0, 0]
        return out.cpu().numpy().astype(np.float32)

    return model


class Prefetcher:
    """items: [(key, image_path)] の時系列順。カーソル位置から未キャッシュを順に処理し、末尾で先頭へ戻る。"""

    def __init__(self, detector, items):
        self.detector = detector
        self.items = list(items)
        self._index = {key: i for i, (key, _) in enumerate(self.items)}
        self._cursor = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._failed = set()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_cursor(self, key):
        """次に処理する位置を key に移す（items にない key は無視）。"""
        i = self._index.get(key)
        if i is None:
            return
        with self._lock:
            self._cursor = i
        self._wake.set()

    def progress(self):
        done = sum(1 for key, _ in self.items if self.detector.cached(key))
        return done, len(self.items)

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _next(self):
        n = len(self.items)
        with self._lock:
            for k in range(n):
                i = (self._cursor + k) % n
                key = self.items[i][0]
                if key not in self._failed and not self.detector.cached(key):
                    self._cursor = (i + 1) % n
                    return self.items[i]
        return None

    def _run(self):
        while not self._stop.is_set():
            item = self._next()
            if item is None:
                # 全部キャッシュ済み: カーソル移動か停止まで待つ
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            try:
                self.detector.mask(item[1], item[0])
            except Exception:
                self._failed.add(item[0])  # 読めない画像で空回りしない

"""一覧ページ用の表（全フレームの状態・ラベル情報）。

一覧を眺めるだけで状態が変わらないよう、ここでは姿勢を初期化せず、poses.json にも書かない。
"""
import csv
import json
import os
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

H, W = 384, 640
LINE_BGR = (255, 200, 0)  # #00c8ff
PATH_BGR = {"straight": (0, 230, 255), "left": (255, 120, 255), "right": (0, 140, 255)}
JPEG_QUALITY = 90


def _polyline(img, pts, color, thickness):
    pts = np.round(np.asarray(pts, float)).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def _dots(img, cls, xp, h_vector):
    row_h = H // len(xp)  # ラベル規約: xp = u / (W - 1)、行 i の v = i*row_h + (row_h-1)/2
    for i, (xp_i, ok) in enumerate(zip(xp, h_vector)):
        if ok:
            u, v = xp_i * (W - 1), i * row_h + (row_h - 1) / 2
            cv2.circle(img, (int(round(u)), int(round(v))), 3, PATH_BGR.get(cls, (255, 255, 255)), -1)


def draw_overlay(img, overlay, label=None):
    """生画像 img に地図線（シアン）・経路（クラス色）・48 行の点を重ねて返す。

    点は保存済みラベル label があればそれから、無ければ overlay["rows"] から描く。
    """
    for piece in overlay.get("lines", []):
        _polyline(img, piece, LINE_BGR, 2)
    for cls, piece in overlay.get("paths", {}).items():
        _polyline(img, piece, PATH_BGR.get(cls, (255, 255, 255)), 3)
    if label:
        entries = [e for e in label if isinstance(e, dict)]
    else:
        rows = overlay.get("rows") or {}
        entries = [rows] if "xp" in rows else [{"class": cls, **r} for cls, r in rows.items()]
    for e in entries:
        _dots(img, e.get("class", "straight"), e["xp"], e["h_vector"])
    return img


class Gallery:
    def __init__(self, store, poses, frames, gnss, projector):
        self.store = store
        self.poses = poses
        self.frames = frames
        self.gnss = gnss
        self.projector = projector
        self.root = Path(store.root)
        self._labels = {}  # "split/name" -> (mtime_ns, {"classes", "rows"})
        self._render_lock = threading.Lock()  # projector を共有するので描画は直列
        self.manual = self._read_manifest()      # key -> bool。manifest.csv が無ければ None
        self.relabeled = self._read_relabeled()  # 再投影で変わったキーの集合。ファイルが無ければ None

    # --- データセット固有の付加情報（無ければ一覧に出さない） ---

    def _read_manifest(self):
        path = self.root / "manifest.csv"
        if not path.is_file():
            return None
        with path.open(newline="", encoding="utf-8") as f:
            return {r["key"]: r.get("has_manual_label") == "1" for r in csv.DictReader(f) if r.get("key")}

    def _read_relabeled(self):
        path = self.root / "relabeled_frames.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return set()
        return {c["key"] for c in data.get("changed", []) if isinstance(c, dict) and "key" in c}

    # --- ラベル情報（ファイルの mtime で更新を検知する） ---

    def label_info(self, split, name):
        """{"classes": [...], "rows": straight の可視行数}。ラベルが無ければ classes [] / rows 0。"""
        key = f"{split}/{name}"
        path = self.store.label_path(split, name)
        try:
            mtime = os.stat(path).st_mtime_ns
        except FileNotFoundError:
            self._labels.pop(key, None)
            return {"classes": [], "rows": 0}
        cached = self._labels.get(key)
        if cached is not None and cached[0] == mtime:
            return dict(cached[1])
        try:
            label = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            label = []
        entries = [e for e in label if isinstance(e, dict)] if isinstance(label, list) else []
        straight = next((e for e in entries if e.get("class") == "straight"), None)
        info = {"classes": [e.get("class") for e in entries],
                "rows": int(sum(1 for v in straight.get("h_vector", []) if v)) if straight else 0}
        self._labels[key] = (mtime, info)
        return dict(info)

    def _label_mtime_ms(self, split, name):
        try:
            return os.stat(self.store.label_path(split, name)).st_mtime_ns // 1_000_000
        except FileNotFoundError:
            return 0

    def _state(self, key):
        return self.store.load_state()["frames"].get(key, {})

    def thumb_version(self, split, name):
        """サムネイルのキャッシュ破り用。ラベルの mtime・状態の ts・姿勢エントリの ts（ms）。

        棄却された保存はラベルも状態も変えずに姿勢だけ変えるので、姿勢の ts が無いと一覧のサムネイルが古いままになる。
        """
        st = self._state(f"{split}/{name}")
        entry = self.poses.get(split, name)
        pose_ms = int(float(entry.get("ts", 0)) * 1000) if entry else 0
        return f"{self._label_mtime_ms(split, name)}-{int(float(st.get('ts', 0)) * 1000)}-{pose_ms}"

    # --- 表 ---

    def table(self):
        frames, bags = [], set()
        for i, (split, name) in enumerate(self.frames.order):
            key = f"{split}/{name}"
            bag, image_index = self.frames.bag_index(split, name)
            entry = self.poses.get(split, name)
            st = self._state(key)
            info = self.label_info(split, name)
            bags.add(bag)
            frames.append({
                "split": split, "name": name, "key": key, "index": i, "bag": bag, "image_index": image_index,
                "status": self.store.get_status(split, name), "edited": bool(st.get("edited", False)),
                "flags": list(entry["flags"]) if entry else [],
                "cost": entry["cost"] if entry else None,
                "source": entry["source"] if entry else None,
                "classes": info["classes"], "rows": info["rows"],
                "has_manual_label": None if self.manual is None else self.manual.get(key),
                "relabeled": False if self.relabeled is None else key in self.relabeled,
                "pose": "entry" if entry else ("gnss" if self.gnss.has(key) else "none"),
                "thumb_version": self.thumb_version(split, name),
            })
        return {"frames": frames, "bags": sorted(bags),
                "has_manifest": self.manual is not None, "has_relabeled": self.relabeled is not None}

    # --- サムネイル（preview/<split>/<name>.jpg にキャッシュ。姿勢は初期化しない） ---

    def _cache_path(self, split, name):
        return self.root / "preview" / split / f"{name}.jpg"

    def _fresh(self, path, split, name):
        """キャッシュがラベル・姿勢エントリ・状態のどれよりも新しければ True。"""
        try:
            cached = os.stat(path).st_mtime_ns
        except FileNotFoundError:
            return False
        try:
            label = os.stat(self.store.label_path(split, name)).st_mtime_ns
        except FileNotFoundError:
            label = 0
        entry = self.poses.get(split, name)
        entry_ts = int(float(entry.get("ts", 0)) * 1e9) if entry else 0
        state_ts = int(float(self._state(f"{split}/{name}").get("ts", 0)) * 1e9)
        return cached >= max(label, entry_ts, state_ts)

    def _render(self, split, name):
        key = f"{split}/{name}"
        img = cv2.imread(str(self.store.image_path(split, name)))
        if img is None:
            raise FileNotFoundError(key)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        label = self.store.load(split, name) if self.store.has_label(split, name) else None
        if not isinstance(label, list):
            label = None
        entry = self.poses.get(split, name)
        if entry is not None:
            pose, lanelet = {k: entry[k] for k in ("x", "y", "yaw", "pitch")}, entry.get("lanelet")
        elif self.gnss.has(key):
            pose, lanelet = self.gnss.initial_pose(key), None
        else:
            pose, lanelet = None, None
        overlay = {"lines": [], "paths": {}, "rows": {}}
        if pose is not None:
            try:
                overlay = self.projector.overlay(pose, lanelet)
            except ValueError:  # 保存済みのレーンが地図に無い
                overlay = self.projector.overlay(pose, None)
        draw_overlay(img, overlay, label)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def thumb(self, split, name):
        """投影画像の JPEG。キャッシュが新しければそれを返し、無い・古ければ描いて保存する。"""
        path = self._cache_path(split, name)
        with self._render_lock:
            if self._fresh(path, split, name):
                return path.read_bytes()
            data = self._render(split, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
            return data

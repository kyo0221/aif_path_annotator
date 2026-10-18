"""フレームごとの車両姿勢と、その保存 (poses.json)。

姿勢は {"x", "y", "yaw", "pitch"} の dict。x, y は地図座標 [m]、yaw は [rad]、pitch は [deg]。
"""
import json
import math
import threading
import time
from pathlib import Path

from tools.path_annotator.store import _atomic_write_json


def make_pose(x, y, yaw, pitch=0.0):
    return {"x": float(x), "y": float(y), "yaw": float(yaw), "pitch": float(pitch)}


def wrap_angle(a):
    """角度を (-π, π] に畳む。"""
    a = math.fmod(a, 2 * math.pi)
    if a > math.pi:
        a -= 2 * math.pi
    elif a <= -math.pi:
        a += 2 * math.pi
    return a


def shift_pose(pose, lateral=0.0, longitudinal=0.0, dyaw=0.0, dpitch=0.0):
    """車両座標でずらした姿勢を返す。lateral は左が +、longitudinal は前が +。"""
    yaw = pose["yaw"]
    left = (-math.sin(yaw), math.cos(yaw))
    fwd = (math.cos(yaw), math.sin(yaw))
    return make_pose(
        pose["x"] + lateral * left[0] + longitudinal * fwd[0],
        pose["y"] + lateral * left[1] + longitudinal * fwd[1],
        wrap_angle(yaw + dyaw),
        pose["pitch"] + dpitch,
    )


def pose_delta(pose, ref):
    """pose を ref の車両座標で表した差分を返す。"""
    dx = pose["x"] - ref["x"]
    dy = pose["y"] - ref["y"]
    yaw = ref["yaw"]
    return {
        "lateral_m": -math.sin(yaw) * dx + math.cos(yaw) * dy,
        "longitudinal_m": math.cos(yaw) * dx + math.sin(yaw) * dy,
        "yaw_deg": math.degrees(wrap_angle(pose["yaw"] - yaw)),
        "pitch_deg": pose["pitch"] - ref["pitch"],
    }


class PoseStore:
    def __init__(self, root):
        self.root = Path(root)
        self._data = None  # poses.json のメモリキャッシュ
        self._lock = threading.Lock()

    def path(self):
        return self.root / "poses.json"

    def _load(self):
        if self._data is None:
            path = self.path()
            try:
                data = json.loads(path.read_text()) if path.exists() else {}
            except json.JSONDecodeError:
                # 壊れていても起動できる方がよい。空として扱う (LabelStore.load_state と同じ方針)。
                data = {}
            if not isinstance(data, dict):
                data = {}
            if not isinstance(data.get("frames"), dict):
                data["frames"] = {}
            self._data = data
        return self._data

    def get(self, split, name):
        with self._lock:
            entry = self._load()["frames"].get(f"{split}/{name}")
            if entry is not None:
                entry.setdefault("lanelet", None)  # レーン選択の導入前に保存したエントリ
            return entry

    def set(self, split, name, pose, source, cost, flags, init, lanelet=None):
        entry = {
            "x": pose["x"], "y": pose["y"], "yaw": pose["yaw"], "pitch": pose["pitch"],
            "source": source,
            "cost": None if cost is None else float(cost),
            "flags": list(flags),
            "init": init,
            "lanelet": lanelet,  # 使用レーンの ID。None は自動
            "ts": time.time(),
        }
        # キャッシュの変異とシリアライズを一括で排他する (LabelStore._update_frame_state と同じ)。
        with self._lock:
            data = self._load()
            data["frames"][f"{split}/{name}"] = entry
            _atomic_write_json(self.path(), data)
        return entry

    def entries(self):
        with self._lock:
            return dict(self._load()["frames"])

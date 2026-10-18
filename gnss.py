"""gnss.json から初期姿勢と、前フレームからの引き継ぎ姿勢を作る。

gnss.json の IMU yaw は進行方向に対して一定のずれ (bag ごと) を持つので、
移動中フレームの yaw_motion - yaw_imu の中央値を bias として初期 yaw に足す。
"""
import json
import math

import numpy as np

from tools.path_annotator.pose import make_pose, wrap_angle

MOVING_SPEED = 0.5  # [m/s] これを超えたら移動中とみなす


class GnssIndex:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.frames = json.load(f)["frames"]
        self.bias = self._estimate_bias()

    def _estimate_bias(self):
        by_bag = {}
        for fr in self.frames.values():
            by_bag.setdefault(fr["bag"], []).append(fr)
        bias = {}
        for bag, items in by_bag.items():
            items.sort(key=lambda fr: fr["image_index"])
            diffs = []
            for prev, cur in zip(items, items[1:]):
                dt = cur["stamp"] - prev["stamp"]
                if dt <= 0:
                    continue
                dist = math.hypot(cur["map"]["x"] - prev["map"]["x"],
                                  cur["map"]["y"] - prev["map"]["y"])
                if dist / dt > MOVING_SPEED:
                    diffs.append(wrap_angle(cur["map"]["yaw_motion_rad"] - cur["map"]["yaw_imu_rad"]))
            bias[bag] = float(np.median(diffs)) if diffs else 0.0
        return bias

    def has(self, key):
        return key in self.frames

    def meta(self):
        return {k: {"bag": fr["bag"], "image_index": fr["image_index"]}
                for k, fr in self.frames.items()}

    def initial_pose(self, key):
        fr = self.frames[key]
        m = fr["map"]
        return make_pose(m["x"], m["y"], wrap_angle(m["yaw_imu_rad"] + self.bias[fr["bag"]]), 0.0)

    def propagate(self, prev_key, prev_pose, key):
        prev = self.frames[prev_key]["map"]
        now = self.frames[key]["map"]
        return make_pose(
            prev_pose["x"] + (now["x"] - prev["x"]),
            prev_pose["y"] + (now["y"] - prev["y"]),
            wrap_angle(prev_pose["yaw"] + wrap_angle(now["yaw_imu_rad"] - prev["yaw_imu_rad"])),
            prev_pose["pitch"],
        )

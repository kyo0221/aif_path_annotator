"""GNSS + 俯瞰(BEV)照合 + vectormap による straight 経路ラベルの生成（/api/vectormap 用）。

1. データセットの gnss.json（tools/label_audit/add_gnss.py が作る）から、このフレームの地図座標系の
   位置と IMU ヨーを初期姿勢として引く（拡張画像は stem の '__' より前の元フレームで引く）。
2. YOLOP の白線マスクを、カメラ行列と平坦路面の仮定で車両座標（前方 X, 左 Y）[m] の点群にする。
   遠方は画像 1 行が数十 cm に相当し横縞状の偽点になるので、前方 1.6〜7m だけ使う。
3. vectormap の観測可能な線（line_thin, is_observable=true）の点群へ、横・前後・ヨーの格子探索で照合し
   姿勢を補正する（評価値: 各 BEV 点から最寄り地図線点までの距離の平均、0.5m で打ち切り）。
4. 補正後の姿勢で自車の lanelet の中心線を前方へたどり、48 行の経路点にする。

torch / cv2 / scipy と学習リポジトリ（Models.data_utils.vision_planner.*）に依存するため、
--vectormap を指定したときだけ annotate.py から読み込む。
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

H, W = 384, 640
NUM_ROWS = H // 8


def base_stem(name):
    return name.split("__", 1)[0]


def replace_straight(old, new):
    """old ラベルの straight エントリだけを new の straight で置き換える（他のエントリは残す）。"""
    straight = [e for e in new if e["class"] == "straight"]
    rest = [e for e in old if e["class"] != "straight"]
    return straight + rest


def bev_match(bev, lines, position, angle, lat=np.arange(-1.5, 1.51, 0.1), lon=np.arange(-1.5, 1.51, 0.25),
              dyaw=np.radians(np.arange(-8, 8.1, 1.0)), trunc=0.5):
    """車両座標の白線点群 bev を地図座標の白線点 lines へ照合し、姿勢（横・前後・ヨー）を補正する。"""
    from scipy.spatial import cKDTree

    tree = cKDTree(lines)
    fwd = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([-np.sin(angle), np.cos(angle)])

    def cost(dl, dx, dy):
        a = angle + dy
        c, s = np.cos(a), np.sin(a)
        p = position + dl * left + dx * fwd
        world = np.stack([p[0] + bev[:, 0] * c - bev[:, 1] * s, p[1] + bev[:, 0] * s + bev[:, 1] * c], 1)
        d, _ = tree.query(world)
        return float(np.mean(np.minimum(d, trunc)))

    base = cost(0.0, 0.0, 0.0)
    best = min((cost(dl, dx, dy), dl, dx, dy) for dy in dyaw for dl in lat for dx in lon)
    _, dl, dx, dy = best
    v, dl, dx, dy = min((cost(dl + a, dx + b, dy + c), dl + a, dx + b, dy + c)
                        for a in np.arange(-0.08, 0.081, 0.02) for b in np.arange(-0.2, 0.21, 0.1)
                        for c in np.radians(np.arange(-0.8, 0.81, 0.2)))
    p = position + dl * left + dx * fwd
    return {"cost_before": base, "cost_after": v, "lateral_m": float(dl), "longitudinal_m": float(dx),
            "yaw_deg": float(np.degrees(dy)), "position": p.tolist(), "angle": float(angle + dy)}


def observable_lines(osm, step=0.2):
    root = ET.parse(osm).getroot()
    nodes = {n.get("id"): (float(n.get("x")), float(n.get("y"))) for n in root.findall("node")}
    pts = []
    for w in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        if tags.get("type") != "line_thin" or tags.get("is_observable") != "true":
            continue
        line = np.array([nodes[nd.get("ref")] for nd in w.findall("nd")], float)
        pts.append(line[0:1])
        for a, b in zip(line[:-1], line[1:]):
            n = max(1, int(np.linalg.norm(b - a) / step))
            pts.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.concatenate(pts)


class VectormapMatcher:
    def __init__(self, root, osm, yolop, params, learning_repo, max_cost=0.25):
        import torch

        sys.path.insert(0, str(learning_repo))
        from Models.data_utils.vision_planner.projection import load_camera, row_anchor_columns
        from Models.data_utils.vision_planner.vectormap import load_vector_map, trace_paths

        self._torch, self._row_anchor_columns, self._trace_paths = torch, row_anchor_columns, trace_paths
        gnss = Path(root) / "gnss.json"
        if not gnss.is_file():
            raise ValueError(f"{gnss} がありません（tools/label_audit/add_gnss.py で作成）")
        self.frames = json.load(open(gnss))["frames"]
        self.vector_map = load_vector_map(str(osm))
        self.lines = observable_lines(osm)
        self.camera = load_camera(str(params), "nHD", H, W)
        self.max_cost = max_cost
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolop = torch.jit.load(str(yolop), map_location=self.device).eval()
        xs = np.geomspace(0.3, 120.0, 6000)
        z = np.zeros_like(xs)
        center, _ = self.camera.project(np.stack([xs, z, z], 1))
        side, _ = self.camera.project(np.stack([xs, z + 0.5, z], 1))
        other, _ = self.camera.project(np.stack([xs, z - 0.5, z], 1))
        order = np.argsort(center[:, 1])
        self._v, self._x, self._u0 = center[order, 1], xs[order], center[order, 0]
        self._ppm = np.abs(side[:, 0] - other[:, 0])[order]

    def lane_lines(self, image_path):
        import cv2

        image = cv2.imread(str(image_path))
        x = self._torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(self.device)
        with self._torch.no_grad():
            return (self.yolop(x)[2][0, 0] > 0.5).cpu().numpy().astype(np.uint8)

    def bev_points(self, ll, x_range=(1.6, 7.0), max_lat=8.0, max_points=800):
        ys, us = np.nonzero(ll[200:H - 12])
        ys = ys + 200
        x = np.interp(ys, self._v, self._x)
        y = (np.interp(ys, self._v, self._u0) - us) / np.interp(ys, self._v, self._ppm)
        m = (x >= x_range[0]) & (x <= x_range[1]) & (np.abs(y) <= max_lat)
        pts = np.stack([x[m], y[m]], 1)
        return pts[:: max(1, len(pts) // max_points)]

    def label_from_pose(self, position, angle, horizon=20.0):
        paths = self._trace_paths(self.vector_map, np.asarray(position), angle, horizon)
        if "straight" not in paths:
            return []
        poly = np.asarray(paths["straight"], float)
        length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
        s = np.r_[np.arange(0.0, length[-1], 0.2), length[-1]]
        poly = np.stack([np.interp(s, length, poly[:, 0]), np.interp(s, length, poly[:, 1])], 1)
        rel = poly - np.asarray(position)
        c, sn = np.cos(-angle), np.sin(-angle)
        base = np.stack([rel[:, 0] * c - rel[:, 1] * sn, rel[:, 0] * sn + rel[:, 1] * c, np.zeros(len(rel))], 1)
        pixels, visible = self.camera.project(base)
        columns, valid = self._row_anchor_columns(pixels, visible, NUM_ROWS, H, W)
        if not valid.any():
            return []
        return [{"class": "straight", "xp": [float(v) for v in columns], "h_vector": [int(v) for v in valid]}]

    def match(self, split, name, image_path):
        key = f"{split}/{base_stem(name)}"
        if key not in self.frames:
            raise ValueError(f"{key} の GNSS がありません")
        g = self.frames[key]["map"]
        bev = self.bev_points(self.lane_lines(image_path))
        if len(bev) < 30:
            raise ValueError(f"白線点が少なすぎます（{len(bev)} 点）")
        r = bev_match(bev, self.lines, np.array([g["x"], g["y"]]), g["yaw_imu_rad"])
        label = self.label_from_pose(r["position"], r["angle"])
        if not label:
            raise ValueError("補正後の位置から straight 経路を投影できません")
        info = {k: round(r[k], 3) for k in ("cost_before", "cost_after", "lateral_m", "longitudinal_m", "yaw_deg")}
        info["reliable"] = r["cost_after"] <= self.max_cost
        return {"label": label, "info": info}

"""姿勢から画像オーバーレイ・48 行ラベル・俯瞰(BEV)・ピッチ補正を作る Projector。

投影の計算は学習リポジトリ（projection.CameraModel / row_anchor_columns, vectormap.trace_paths）に任せ、
ここでは再実装しない。姿勢は pose.py の dict（x, y [m]、yaw [rad]、pitch [deg]）。
"""
import math
import sys

import numpy as np

from tools.path_annotator.matching import densify, observable_polylines
from tools.path_annotator.store import ROW_START

H, W = 384, 640
NUM_ROWS = 48
HORIZON = 20.0
IMAGE_X_RANGE = (0.8, 45.0)
IMAGE_MAX_LAT = 25.0
BEV_ROW_START = 200
BEV_X_RANGE = (1.6, 7.0)
BEV_MAX_LAT = 8.0
BEV_MAX_POINTS = 800
BEV_X_WINDOW = (-4.0, 28.0)
BEV_Y_WINDOW = 13.0
PITCH_QUANTUM = 0.1
STEP = 0.2
CLASSES = ("straight", "left", "right")


def to_vehicle(points_map, pose):
    """地図座標の点 (N,2) を、姿勢の車両座標（前方 x, 左 y）に直す。"""
    rel = np.asarray(points_map, float).reshape(-1, 2) - np.array([pose["x"], pose["y"]])
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    return np.stack([rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c], 1)


def tilted_camera(camera, pitch_deg):
    """車両の左右軸（base Y）まわりに pitch_deg [deg] 傾けたカメラ。+ で水平線が画像の上へ動く。

    yaml の pitch は光軸まわりの回転になるため、yaml の角度は触らず rotation に左右軸まわりの回転を合成する。
    """
    a = math.radians(pitch_deg)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return type(camera)(camera.fx, camera.fy, camera.cx, camera.cy, ry @ camera.rotation,
                        camera.translation, camera.pad_top, camera.pad_left)


def _resample(poly, step=STEP):
    """折れ線を step 間隔で等間隔に打ち直す（終点を含む）。"""
    poly = np.asarray(poly, float)
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    if length[-1] == 0:
        return poly[:1]
    s = np.r_[np.arange(0.0, length[-1], step), length[-1]]
    return np.stack([np.interp(s, length, poly[:, 0]), np.interp(s, length, poly[:, 1])], 1)


def _runs(mask):
    """True が連続する区間の (開始, 終了+1) を返す。"""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))


def _pieces(points, mask):
    return [points[a:b].tolist() for a, b in _runs(mask) if b - a >= 2]


class Projector:
    def __init__(self, camera, vector_map, polylines, trace_paths, row_anchor_columns):
        self.camera = camera
        self.vector_map = vector_map
        self.polylines = polylines
        self._trace_paths = trace_paths
        self._row_anchor_columns = row_anchor_columns
        self._cameras = {}
        self._tables = {}

    def camera_for(self, pitch):
        key = round(pitch / PITCH_QUANTUM)
        if key not in self._cameras:
            self._cameras[key] = self.camera if key == 0 else tilted_camera(self.camera, key * PITCH_QUANTUM)
        return self._cameras[key]

    def paths(self, pose):
        paths = self._trace_paths(self.vector_map, (pose["x"], pose["y"]), pose["yaw"], HORIZON)
        return {cls: np.asarray(paths[cls], float) for cls in CLASSES if cls in paths}

    def project_polyline(self, poly_map, pose):
        """地図座標の折れ線を、画像に写る連続区間ごとの画素列 [[u,v],...] に分ける。"""
        local = to_vehicle(densify(poly_map, STEP), pose)
        pixels, visible = self.camera_for(pose["pitch"]).project(np.c_[local, np.zeros(len(local))])
        ok = (visible & (local[:, 0] >= IMAGE_X_RANGE[0]) & (local[:, 0] <= IMAGE_X_RANGE[1])
              & (np.abs(local[:, 1]) <= IMAGE_MAX_LAT)
              & (pixels[:, 0] >= 0) & (pixels[:, 0] <= W - 1) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= H - 1))
        return _pieces(pixels, ok)

    def _rows(self, path, pose):
        local = to_vehicle(_resample(path), pose)
        pixels, visible = self.camera_for(pose["pitch"]).project(np.c_[local, np.zeros(len(local))])
        columns, valid = self._row_anchor_columns(pixels, visible, NUM_ROWS, H, W)
        # ROW_START より上の行は学習に使わない（store.validate_label が弾く）。ピッチが + だと
        # 20 m 先がここまで上がるので 0 にしておく。ピッチ 0 では元々 0 なので converter.py と一致する。
        xp = [0.0 if row < ROW_START else float(v) for row, v in enumerate(columns)]
        h = [0 if row < ROW_START else int(v) for row, v in enumerate(valid)]
        if not any(h):
            return None
        return {"xp": xp, "h_vector": h}

    def _label_rows(self, paths, pose):
        rows = {}
        for cls, path in paths.items():
            entry = self._rows(path, pose)
            if entry is not None:
                rows[cls] = entry
        return rows

    def label(self, pose):
        rows = self._label_rows(self.paths(pose), pose)
        return [{"class": cls, **rows[cls]} for cls in CLASSES if cls in rows]

    def overlay(self, pose):
        paths = self.paths(pose)
        lines = [piece for poly in self.polylines for piece in self.project_polyline(poly, pose)]
        return {"lines": lines, "paths": self._longest(paths, lambda p: self.project_polyline(p, pose)),
                "rows": self._label_rows(paths, pose)}

    @staticmethod
    def _longest(paths, split):
        """class ごとに、split で切った区間のうち最長の 1 本を返す（区間が無い class は出さない）。"""
        out = {}
        for cls, path in paths.items():
            pieces = split(path)
            if pieces:
                out[cls] = max(pieces, key=len)
        return out

    def _table(self, pitch):
        key = round(pitch / PITCH_QUANTUM)
        if key not in self._tables:
            camera = self.camera_for(pitch)
            xs = np.geomspace(0.3, 120.0, 6000)
            z = np.zeros_like(xs)
            center, _ = camera.project(np.stack([xs, z, z], 1))
            side, _ = camera.project(np.stack([xs, z + 0.5, z], 1))
            other, _ = camera.project(np.stack([xs, z - 0.5, z], 1))
            order = np.argsort(center[:, 1])
            self._tables[key] = (center[order, 1], xs[order], center[order, 0], np.abs(side[:, 0] - other[:, 0])[order])
        return self._tables[key]

    def bev_points(self, mask, pitch=0.0, x_range=BEV_X_RANGE):
        """白線マスク (H,W) の画素を、平坦路面の仮定で車両座標 (N,2) の点群にする。"""
        v_tab, x_tab, u0_tab, ppm_tab = self._table(pitch)
        ys, us = np.nonzero(np.asarray(mask)[BEV_ROW_START:H - 12])
        ys = ys + BEV_ROW_START
        x = np.interp(ys, v_tab, x_tab)
        y = (np.interp(ys, v_tab, u0_tab) - us) / np.interp(ys, v_tab, ppm_tab)
        keep = (x >= x_range[0]) & (x <= x_range[1]) & (np.abs(y) <= BEV_MAX_LAT)
        pts = np.stack([x[keep], y[keep]], 1)
        return pts[:: max(1, len(pts) // BEV_MAX_POINTS)]

    def _bev_pieces(self, poly, pose):
        local = to_vehicle(densify(poly, STEP), pose)
        ok = ((local[:, 0] >= BEV_X_WINDOW[0]) & (local[:, 0] <= BEV_X_WINDOW[1])
              & (np.abs(local[:, 1]) <= BEV_Y_WINDOW))
        return _pieces(local, ok)

    def bev(self, pose, points):
        paths = self.paths(pose)
        return {
            "lines": [piece for poly in self.polylines for piece in self._bev_pieces(poly, pose)],
            "paths": self._longest(paths, lambda p: self._bev_pieces(p, pose)),
            "points": np.asarray(points, float).reshape(-1, 2).tolist(),
        }


def load_projector(learning_repo, params_path, osm_path):
    sys.path.insert(0, str(learning_repo))
    from Models.data_utils.vision_planner.projection import load_camera, row_anchor_columns
    from Models.data_utils.vision_planner.vectormap import load_vector_map, trace_paths

    return Projector(load_camera(str(params_path), "nHD", H, W), load_vector_map(str(osm_path)),
                     observable_polylines(osm_path), trace_paths, row_anchor_columns)

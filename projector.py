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
LANE_SEARCH_RADIUS = 8.0   # レーン候補に挙げる中心線までの距離 [m]
LANE_ALIGN_DEG = 60.0      # 進行方向とレーン接線がこの角度以内なら「向きが合う」
LANE_MAX = 6               # 候補の最大数
LANE_AUTO_DISTANCE = 5.0   # 自動選択で許す中心線までの距離 [m]


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
    def __init__(self, camera, vector_map, polylines, trace_paths, row_anchor_columns,
                 vectormap_module=None):
        self._vm = vectormap_module
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

    def _lane_info(self, lane_id, pose):
        """レーン lane_id の中心線までの距離・接線と進行方向のなす角。中心線が使えなければ None。"""
        hit = self._vm._nearest_point_on_polyline(self.vector_map.centerlines[lane_id], (pose["x"], pose["y"]))
        if hit is None:
            return None
        distance, _, _, tangent = hit
        heading = np.array([math.cos(pose["yaw"]), math.sin(pose["yaw"])])
        cos = float(np.clip(np.dot(tangent, heading), -1.0, 1.0))
        angle = math.degrees(math.acos(cos))
        return {"id": int(lane_id), "distance_m": float(distance), "angle_deg": angle,
                "aligned": bool(angle <= LANE_ALIGN_DEG)}

    def lane_candidates(self, pose, lanelet=None):
        """姿勢の周り（8 m 以内）のレーン候補。向きが合うものを距離順 → 残りを距離順、先頭 6 件。

        lanelet が指定され一覧に無ければ（半径外でも）末尾に足す。存在しない ID は ValueError。
        """
        if self._vm is None:
            return []
        if lanelet is not None and lanelet not in self.vector_map.centerlines:
            raise ValueError(f"lanelet {lanelet} がありません")
        infos = [self._lane_info(lane_id, pose) for lane_id in self.vector_map.centerlines]
        near = sorted((i for i in infos if i is not None and i["distance_m"] <= LANE_SEARCH_RADIUS),
                      key=lambda i: i["distance_m"])
        lanes = ([i for i in near if i["aligned"]] + [i for i in near if not i["aligned"]])[:LANE_MAX]
        if lanelet is not None and all(lane["id"] != lanelet for lane in lanes):
            extra = self._lane_info(lanelet, pose)
            if extra is not None:
                lanes.append(extra)
        return lanes

    def auto_lanelet(self, pose):
        """向きが合い（60° 以内）5 m 以内で最も近いレーンの ID。無ければ None（従来の trace_paths 任せ）。"""
        for lane in self.lane_candidates(pose):
            if lane["aligned"] and lane["distance_m"] <= LANE_AUTO_DISTANCE:
                return lane["id"]
        return None

    def resolve_lane(self, pose, lanelet=None):
        """実際に経路の起点にするレーン {id, auto, distance_m, angle_deg}。決まらなければ id=None。"""
        lane_id = self.auto_lanelet(pose) if lanelet is None else lanelet
        info = None if lane_id is None else self._lane_info(lane_id, pose)
        return {"id": None if info is None else info["id"], "auto": lanelet is None,
                "distance_m": None if info is None else info["distance_m"],
                "angle_deg": None if info is None else info["angle_deg"]}

    def _trace_from(self, lanelet_id, pose):
        """指定レーンの最近傍点から経路を辿る。vectormap.trace_paths の locate_lanelet 以降と同じ手順。"""
        vm, vector_map, position = self._vm, self.vector_map, (pose["x"], pose["y"])
        polyline = vector_map.centerlines[lanelet_id]
        _, _, index, _ = vm._nearest_point_on_polyline(polyline, position)
        start_point, next_index = vm._project_onto_polyline(polyline, index, position)
        trunk, remaining, terminal, visited = vm._walk(vector_map, lanelet_id, start_point, next_index, HORIZON)
        successors = [c for c in vector_map.connections.get(terminal, []) if c[0] not in visited]
        if remaining <= 0.0 or len(successors) < 2:
            return {"straight": trunk}
        paths = {}
        for to_id, turn in successors:
            if turn not in vm.SLOT_NAMES:
                continue
            branch_polyline = vector_map.centerlines[to_id]
            branch, _, _, _ = vm._walk(vector_map, to_id, branch_polyline[0], 1, remaining)
            merged = vm._drop_consecutive_duplicates(np.vstack([trunk, branch]))
            if turn not in paths or len(merged) > len(paths[turn]):
                paths[turn] = merged
        return paths or {"straight": trunk}

    def paths(self, pose, lanelet=None):
        lane_id = self.resolve_lane(pose, lanelet)["id"] if self._vm is not None else None
        if lane_id is not None:
            paths = self._trace_from(lane_id, pose)
        else:
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

    def label(self, pose, lanelet=None):
        rows = self._label_rows(self.paths(pose, lanelet), pose)
        return [{"class": cls, **rows[cls]} for cls in CLASSES if cls in rows]

    def overlay(self, pose, lanelet=None):
        paths = self.paths(pose, lanelet)
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

    def bev(self, pose, points, lanelet=None):
        paths = self.paths(pose, lanelet)
        lanes = [{"id": lane["id"], "points": piece}
                 for lane in self.lane_candidates(pose, lanelet)
                 for piece in self._bev_pieces(self.vector_map.centerlines[lane["id"]], pose)]
        return {
            "lines": [piece for poly in self.polylines for piece in self._bev_pieces(poly, pose)],
            "paths": self._longest(paths, lambda p: self._bev_pieces(p, pose)),
            "points": np.asarray(points, float).reshape(-1, 2).tolist(),
            "lanes": lanes,
        }


def load_projector(learning_repo, params_path, osm_path):
    sys.path.insert(0, str(learning_repo))
    from Models.data_utils.vision_planner.projection import load_camera, row_anchor_columns
    from Models.data_utils.vision_planner import vectormap
    from Models.data_utils.vision_planner.vectormap import load_vector_map, trace_paths

    return Projector(load_camera(str(params_path), "nHD", H, W), load_vector_map(str(osm_path)),
                     observable_polylines(osm_path), trace_paths, row_anchor_columns, vectormap)

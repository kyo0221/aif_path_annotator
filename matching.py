"""距離場による俯瞰(BEV)照合と、複数の初期姿勢からの候補生成。

車両座標（前方 x, 左 y）[m] の白線点群を、地図の観測可能な白線（line_thin, is_observable=true）へ
照合して姿勢（横・前後・ヨー）を補正する。評価値は各点から最寄りの地図線までの距離の平均（TRUNC で打ち切り）。
最寄り距離は地図線を格子に立てた距離場から双一次補間で引くので、仮説をまとめて numpy で評価できる。
"""
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree

from tools.path_annotator.pose import pose_delta, shift_pose

TRUNC = 0.5
CELL = 0.05
LANE_WIDTH = 3.3
DEDUPE_POSITION = 0.4
DEDUPE_YAW_DEG = 2.0
MAX_CANDIDATES = 4

# 細密探索の幅（現行 bev_match と同じ）
FINE_LATERAL, FINE_LATERAL_STEP = 0.08, 0.02
FINE_LONGITUDINAL, FINE_LONGITUDINAL_STEP = 0.2, 0.1
FINE_YAW_DEG, FINE_YAW_STEP_DEG = 0.8, 0.2

# 1 回に評価する点の数の上限（メモリを抑える）
_CHUNK_POINTS = 4_000_000


@dataclass(frozen=True)
class Window:
    """探索窓。lateral / longitudinal は [m]、yaw_deg は [deg]（いずれも ± の幅と刻み）。"""
    lateral: float
    lateral_step: float
    longitudinal: float
    longitudinal_step: float
    yaw_deg: float
    yaw_step_deg: float


NARROW = Window(0.5, 0.05, 0.5, 0.1, 3.0, 0.5)
WIDE = Window(1.5, 0.1, 1.5, 0.25, 8.0, 1.0)


def observable_polylines(osm_path):
    """観測可能な白線（type=line_thin かつ is_observable=true）を (n,2) の折れ線のリストで返す。"""
    root = ET.parse(osm_path).getroot()
    nodes = {n.get("id"): (float(n.get("x")), float(n.get("y"))) for n in root.findall("node")}
    polylines = []
    for way in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if tags.get("type") != "line_thin" or tags.get("is_observable") != "true":
            continue
        polylines.append(np.array([nodes[nd.get("ref")] for nd in way.findall("nd")], float))
    return polylines


def densify(polyline, step):
    """折れ線を、各線分を step 以下の間隔で区切った点列にする（端点を含む）。"""
    polyline = np.asarray(polyline, float)
    out = [polyline[:1]]
    for a, b in zip(polyline[:-1], polyline[1:]):
        n = max(1, math.ceil(np.linalg.norm(b - a) / step))
        out.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.concatenate(out)


class DistanceField:
    """地図線までの距離（TRUNC で打ち切り）を格子に持つ。格子外は TRUNC。"""

    def __init__(self, polylines, cell=CELL, trunc=TRUNC, margin=5.0):
        self.cell, self.trunc = cell, trunc
        if len(polylines) == 0:
            raise ValueError("polylines が空です")
        dense = np.concatenate([densify(p, cell / 2) for p in polylines])
        lo, hi = dense.min(axis=0) - margin, dense.max(axis=0) + margin
        self.origin = (float(lo[0]), float(lo[1]))
        shape = (int(math.ceil((hi[1] - lo[1]) / cell)) + 1, int(math.ceil((hi[0] - lo[0]) / cell)) + 1)
        index = np.rint((dense - lo) / cell).astype(int)
        mask = np.ones(shape, bool)
        mask[index[:, 1], index[:, 0]] = False  # 線のある格子点だけ 0（距離 0 の元）
        # 線を格子点へ丸めた距離は最大 cell*0.71 ずれるので、近傍の格子点だけ厳密な距離で上書きする
        rough = distance_transform_edt(mask) * cell
        self.grid = np.full(shape, trunc, np.float32)
        rows, cols = np.nonzero(rough < trunc + 2 * cell)
        nodes = np.stack([lo[0] + cols * cell, lo[1] + rows * cell], 1)
        exact, _ = cKDTree(np.concatenate([densify(p, cell / 4) for p in polylines])).query(nodes)
        self.grid[rows, cols] = np.minimum(exact, trunc)

    def distance(self, xy):
        """地図座標 xy[..., 2] の最寄り線までの距離を双一次補間で返す。格子外は trunc。"""
        xy = np.asarray(xy)
        col = (xy[..., 0] - self.origin[0]) / self.cell
        row = (xy[..., 1] - self.origin[1]) / self.cell
        shape = col.shape
        out = map_coordinates(self.grid, [row.ravel(), col.ravel()], order=1, mode="constant", cval=self.trunc)
        return out.reshape(shape)


def _costs(field, points, pose, lateral, longitudinal, dyaw):
    """基準姿勢に対する仮説（横・前後・ヨー差。同じ長さの配列）それぞれの評価値を返す。"""
    lateral, longitudinal, dyaw = (np.asarray(a, float) for a in (lateral, longitudinal, dyaw))
    yaw = pose["yaw"]
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    base = np.array([pose["x"], pose["y"]])
    points = np.asarray(points, float)
    if len(points) == 0:
        return np.full(len(lateral), field.trunc)
    out = np.empty(len(lateral))
    chunk = max(1, _CHUNK_POINTS // max(1, len(points)))
    for start in range(0, len(lateral), chunk):
        sl = slice(start, start + chunk)
        centre = base + lateral[sl, None] * left + longitudinal[sl, None] * fwd  # (n,2)
        a = yaw + dyaw[sl]
        c, s = np.cos(a)[:, None], np.sin(a)[:, None]
        world = np.empty((len(centre), len(points), 2), np.float32)
        world[..., 0] = centre[:, 0, None] + points[:, 0] * c - points[:, 1] * s
        world[..., 1] = centre[:, 1, None] + points[:, 0] * s + points[:, 1] * c
        out[sl] = field.distance(world).mean(axis=1)
    return out


def cost(field, points, pose):
    """車両座標の点群 points (N,2) を pose で地図座標に置いたときの評価値。"""
    return float(_costs(field, points, pose, [0.0], [0.0], [0.0])[0])


def _axis(width, step):
    return np.arange(-width, width + step / 2, step)


def match(field, points, pose, window):
    """pose を初期値に、窓の範囲で格子探索 → 最良の周りを細密探索して姿勢を補正する。ピッチは変えない。"""
    points = np.asarray(points, float)
    before = cost(field, points, pose)
    lat = _axis(window.lateral, window.lateral_step)
    lon = _axis(window.longitudinal, window.longitudinal_step)
    dl_grid, dx_grid = (a.ravel() for a in np.meshgrid(lat, lon, indexing="ij"))
    best = (math.inf, 0.0, 0.0, 0.0)
    for dyaw in np.radians(_axis(window.yaw_deg, window.yaw_step_deg)):
        costs = _costs(field, points, pose, dl_grid, dx_grid, np.full(len(dl_grid), dyaw))
        i = int(np.argmin(costs))
        if costs[i] < best[0]:
            best = (float(costs[i]), float(dl_grid[i]), float(dx_grid[i]), float(dyaw))
    _, dl, dx, dy = best

    fl, fx, fy = np.meshgrid(_axis(FINE_LATERAL, FINE_LATERAL_STEP), _axis(FINE_LONGITUDINAL, FINE_LONGITUDINAL_STEP),
                             np.radians(_axis(FINE_YAW_DEG, FINE_YAW_STEP_DEG)), indexing="ij")
    fl, fx, fy = dl + fl.ravel(), dx + fx.ravel(), dy + fy.ravel()
    costs = _costs(field, points, pose, fl, fx, fy)
    i = int(np.argmin(costs))
    result = shift_pose(pose, lateral=float(fl[i]), longitudinal=float(fx[i]), dyaw=float(fy[i]))
    delta = pose_delta(result, pose)
    return {"pose": result, "cost": float(costs[i]), "cost_before": before,
            "lateral_m": delta["lateral_m"], "longitudinal_m": delta["longitudinal_m"], "yaw_deg": delta["yaw_deg"]}


def _is_duplicate(a, b):
    d = pose_delta(a, b)
    return math.hypot(d["lateral_m"], d["longitudinal_m"]) <= DEDUPE_POSITION and abs(d["yaw_deg"]) <= DEDUPE_YAW_DEG


def candidates(field, points, inits, window):
    """各 (pose, init 名) を照合し、近い結果を（先のものを残して）間引き、コスト昇順で最大 MAX_CANDIDATES 件返す。"""
    kept = []
    for pose, name in inits:
        result = match(field, points, pose, window)
        if any(_is_duplicate(result["pose"], k["pose"]) for k in kept):
            continue
        kept.append({**result, "init": name})
    kept.sort(key=lambda r: r["cost"])
    return kept[:MAX_CANDIDATES]

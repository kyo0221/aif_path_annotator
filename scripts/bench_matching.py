"""照合の所要時間を実 OSM で計測する（目標: WIDE 1 回 < 0.3 s、候補 4 起点の合計 < 1 s）。"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from tools.path_annotator.matching import NARROW, WIDE, DistanceField, candidates, match, observable_polylines  # noqa: E402
from tools.path_annotator.pose import make_pose, shift_pose  # noqa: E402


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm", required=True)
    parser.add_argument("--points", type=int, default=800)
    args = parser.parse_args()

    polylines, t_parse = timed(lambda: observable_polylines(args.osm))
    field, t_field = timed(lambda: DistanceField(polylines))
    # 最初の線の途中を走る車両から見た白線点を、横 0.3 m・ヨー 2° ずらした初期姿勢で照合する
    line = polylines[0]
    mid = len(line) // 2
    yaw = float(np.arctan2(*(line[mid + 1] - line[mid])[::-1]))
    truth = make_pose(*line[mid], yaw)
    rng = np.random.default_rng(0)
    points = np.stack([rng.uniform(1.6, 7.0, args.points), rng.normal(0.0, 0.03, args.points)], 1)
    init = shift_pose(truth, lateral=0.3, dyaw=np.radians(2.0))

    _, t_wide = timed(lambda: match(field, points, init, WIDE))
    _, t_narrow = timed(lambda: match(field, points, init, NARROW))
    inits = [(init, "gnss"), (shift_pose(init, lateral=3.3), "lane_left"),
             (shift_pose(init, lateral=-3.3), "lane_right"), (shift_pose(init, longitudinal=5.0), "ahead")]
    out, t_cand = timed(lambda: candidates(field, points, inits, WIDE))
    print(f"polylines={len(polylines)} grid={field.grid.shape} points={args.points}")
    print(f"parse        {t_parse:.3f} s")
    print(f"DistanceField {t_field:.3f} s")
    print(f"WIDE match   {t_wide:.3f} s  (target < 0.3)")
    print(f"NARROW match {t_narrow:.3f} s")
    print(f"candidates(4 inits, WIDE) {t_cand:.3f} s  (target < 1)  -> {len(out)} kept")


if __name__ == "__main__":
    main()

"""制御点による経路曲線の変形。

xp は 48 行ぶんの正規化横位置 [0,1]、h_vector はその行が有効かの 0/1。
制御点を動かすと、その変位がガウシアン重みで有効行へ配分される。
重みは有効行ごとに和が 1 になるよう正規化してあるので、
全制御点を同量動かすと曲線全体がちょうどその量だけ平行移動する。
"""
import numpy as np

N_ROWS = 48

# static/geom.js の ROW_START と一致させること。これより上の行は学習に使われない。
# server.py（API のガード）と store.py（保存時の検証）の双方がここから import する。
ROW_START = 25


def _valid_rows(h_vector):
    return np.array([i for i, v in enumerate(h_vector) if v], dtype=int)


def control_rows(h_vector, k):
    """有効行の範囲に K 個の制御点行を等間隔に置く。"""
    valid = _valid_rows(h_vector)
    if len(valid) == 0:
        return []
    if k == 1:
        return [int(valid[len(valid) // 2])]
    lo, hi = int(valid[0]), int(valid[-1])
    return [int(round(lo + (hi - lo) * j / (k - 1))) for j in range(k)]


def weights(h_vector, k):
    """形状 (k, N_ROWS) の重み行列。有効行では列和が 1、無効行は 0。"""
    w = np.zeros((k, N_ROWS), dtype=float)
    valid = _valid_rows(h_vector)
    if len(valid) == 0:
        return w
    rows = control_rows(h_vector, k)
    span = int(valid[-1] - valid[0]) + 1
    sigma = max(span / (2.0 * max(k - 1, 1)), 1e-6)
    for j, r in enumerate(rows):
        w[j, valid] = np.exp(-((valid - r) ** 2) / (2.0 * sigma ** 2))
    col = w[:, valid].sum(axis=0)
    col[col == 0.0] = 1.0
    w[:, valid] /= col
    return w


def apply_control_delta(xp, h_vector, deltas):
    """制御点の変位 deltas を有効行へ配分して xp に加える。"""
    base = np.asarray(xp, dtype=float)
    valid = _valid_rows(h_vector)
    if len(valid) == 0:
        return base.tolist()
    w = weights(h_vector, len(deltas))
    moved = base + np.asarray(deltas, dtype=float) @ w
    out = base.copy()
    out[valid] = np.clip(moved[valid], 0.0, 1.0)
    return out.tolist()


def apply_shift(xp, h_vector, delta):
    """曲線全体を delta だけ平行移動する。"""
    return apply_control_delta(xp, h_vector, [delta])


def apply_tilt(xp, h_vector, delta):
    """下端（行番号が最大の有効行）を固定して、上端を delta だけ動かす。"""
    base = np.asarray(xp, dtype=float)
    valid = _valid_rows(h_vector)
    if len(valid) < 2:
        return base.tolist()
    i_min, i_max = int(valid[0]), int(valid[-1])
    t = (valid - i_max) / float(i_min - i_max)
    out = base.copy()
    out[valid] = np.clip(base[valid] + t * delta, 0.0, 1.0)
    return out.tolist()


def _catmull_rom(ts, xs, query):
    """制御点 (ts, xs) を通る Catmull-Rom スプラインを query の位置で評価する。"""
    ts = np.asarray(ts, dtype=float)
    xs = np.asarray(xs, dtype=float)
    query = np.asarray(query, dtype=float)
    if len(ts) == 1:
        return np.full(len(query), xs[0])
    if len(ts) == 2:
        return np.interp(query, ts, xs)
    p = np.concatenate([[xs[0]], xs, [xs[-1]]])
    out = np.empty(len(query), dtype=float)
    for qi, q in enumerate(query):
        j = int(np.clip(np.searchsorted(ts, q) - 1, 0, len(ts) - 2))
        t0, t1 = ts[j], ts[j + 1]
        u = 0.0 if t1 == t0 else (q - t0) / (t1 - t0)
        p0, p1, p2, p3 = p[j], p[j + 1], p[j + 2], p[j + 3]
        out[qi] = 0.5 * (
            2 * p1
            + (-p0 + p2) * u
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3
        )
    return out


def regenerate(h_vector, control_xs):
    """制御点だけから有効行の xp を作り直す。元の形状は破棄される。"""
    out = np.zeros(N_ROWS, dtype=float)
    valid = _valid_rows(h_vector)
    if len(valid) == 0:
        return out.tolist()
    rows = control_rows(h_vector, len(control_xs))
    out[valid] = np.clip(_catmull_rom(rows, control_xs, valid), 0.0, 1.0)
    return out.tolist()


def interpolate_rows(xp, h_vector, row_a, xp_a, row_b, xp_b):
    """行 a と行 b を直線で結び、その間（両端含む）の行に点を打つ。

    欠けている区間をまとめて埋めるための操作。補間した行は有効行になる。
    範囲外の行と、範囲内でない行の値は変えない。

    戻り値は (新しい xp, 新しい h_vector) のタプル。
    """
    new_xp = list(xp)
    new_h = list(h_vector)
    a, b, va, vb = int(row_a), int(row_b), float(xp_a), float(xp_b)
    if a > b:
        a, b, va, vb = b, a, vb, va
    if a == b:
        new_xp[a] = float(np.clip(va, 0.0, 1.0))
        new_h[a] = 1
        return new_xp, new_h
    span = b - a
    for row in range(a, b + 1):
        t = (row - a) / span
        new_xp[row] = float(np.clip(va + (vb - va) * t, 0.0, 1.0))
        new_h[row] = 1
    return new_xp, new_h


def clear_rows(xp, h_vector, row_a, row_b):
    """行 a〜b（両端含む）の点を消す。

    範囲内の h_vector を 0 にし、xp を 0.0 にする。
    無効行の xp は converter.py と同じくプレースホルダの 0.0 で揃える。
    範囲外の行は一切変更しない。

    戻り値は (新しい xp, 新しい h_vector) のタプル。
    """
    new_xp = list(xp)
    new_h = list(h_vector)
    a, b = int(row_a), int(row_b)
    if a > b:
        a, b = b, a
    for row in range(a, b + 1):
        new_xp[row] = 0.0
        new_h[row] = 0
    return new_xp, new_h

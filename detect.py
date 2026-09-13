"""壊れていそうなフレームを見つけるためのスコアリング。

指標はすべて「大きいほど異常」に向きを揃えてある。
"""
import re

import numpy as np

from tools.path_annotator.curve import N_ROWS

NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<num>\d{6})$")
EDGE_LO, EDGE_HI = 0.02, 0.98
METRIC_KEYS = ("bend", "edge", "missing", "gap")


def parse_frame_name(name):
    """フレーム名を (bag接頭辞, 連番) に分解する。"""
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"frame name must end with a 6-digit number: {name!r}")
    return m.group("prefix"), int(m.group("num"))


def _entry_metrics(entry):
    h = np.asarray(entry["h_vector"], dtype=bool)
    xp = np.asarray(entry["xp"], dtype=float)
    valid = np.flatnonzero(h)
    if len(valid) == 0:
        return {"bend": 0.0, "edge": 0.0, "missing": float(N_ROWS), "gap": 0.0}

    bend = 0.0
    for a, b, c in zip(valid, valid[1:], valid[2:]):
        if b - a == 1 and c - b == 1:
            bend = max(bend, abs(xp[a] - 2.0 * xp[b] + xp[c]))

    v = xp[valid]
    edge = float(np.mean((v < EDGE_LO) | (v > EDGE_HI)))
    missing = float(N_ROWS - len(valid))
    gap = float((valid[-1] - valid[0] + 1) - len(valid))
    return {"bend": float(bend), "edge": edge, "missing": missing, "gap": gap}


def frame_metrics(label):
    """ラベル全エントリの指標を、キーごとに最大値で集約する。"""
    per_entry = [_entry_metrics(e) for e in label]
    if not per_entry:
        return {k: 0.0 for k in METRIC_KEYS}
    return {k: max(m[k] for m in per_entry) for k in METRIC_KEYS}


TEMPORAL_KEY = "jump"
ALL_KEYS = METRIC_KEYS + (TEMPORAL_KEY,)
MAX_NUM_GAP = 4
Z_REASON_THRESHOLD = 1.5
UNREADABLE_REASON = "unreadable"


def _unreadable_metrics():
    """指標を計算できなかったフレームに与える値。

    キューの先頭に出したいので、各指標に取りうる最大値を入れる。
    ツールの目的は壊れたラベルを直すことなので、読めないラベルこそ
    最優先で人に見せるべきで、キュー全体を落とす理由にはならない。
    """
    return {"bend": 1.0, "edge": 1.0, "missing": float(N_ROWS), "gap": float(N_ROWS), TEMPORAL_KEY: 1.0}


def _first_entry_arrays(label):
    e = label[0]
    return np.asarray(e["xp"], dtype=float), np.asarray(e["h_vector"], dtype=bool)


def temporal_metric(label, neighbor_labels):
    """隣接フレームとの xp 平均差。両方で有効な行だけを見る。"""
    if not label or not neighbor_labels:
        return 0.0
    xp_a, h_a = _first_entry_arrays(label)
    worst = 0.0
    for other in neighbor_labels:
        if not other:
            continue
        xp_b, h_b = _first_entry_arrays(other)
        both = h_a & h_b
        if not both.any():
            continue
        worst = max(worst, float(np.mean(np.abs(xp_a[both] - xp_b[both]))))
    return worst


DEFAULT_WEIGHTS = {"bend": 1.0, "edge": 1.0, "missing": 1.0, "gap": 1.0, TEMPORAL_KEY: 1.0}
Z_CLIP = 5.0


def score_frames(records, weights=None):
    """指標を全体分布で z 化し、重み付き和でスコアにして降順に返す。

    z はほぼ定数の指標（実データでは gap が該当）で外れ値 1 件が合成スコアを
    支配しないよう ±Z_CLIP にクリップする。
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    if not records:
        return []
    z = {}
    for key in ALL_KEYS:
        col = np.array([r["metrics"].get(key, 0.0) for r in records], dtype=float)
        sd = col.std()
        z[key] = np.zeros_like(col) if sd == 0 else np.clip((col - col.mean()) / sd, -Z_CLIP, Z_CLIP)
    out = []
    for i, r in enumerate(records):
        score = sum(w[k] * z[k][i] for k in ALL_KEYS)
        reasons = [k for k in ALL_KEYS if z[k][i] > Z_REASON_THRESHOLD]
        if r.get("unreadable"):
            reasons.insert(0, UNREADABLE_REASON)
        out.append(
            {
                "split": r["split"],
                "name": r["name"],
                "score": float(score),
                "reasons": reasons,
                "metrics": r["metrics"],
            }
        )
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def group_by_bag(names):
    """フレーム名を bag ごとにまとめ、連番順に並べて返す。"""
    parsed = {}
    for n in names:
        try:
            parsed[n] = parse_frame_name(n)
        except ValueError:
            parsed[n] = (n, 0)
    bags = {}
    for n in names:
        bags.setdefault(parsed[n][0], []).append(n)
    for group in bags.values():
        group.sort(key=lambda m: parsed[m][1])
    return bags, parsed


def score_dataset(store, weights=None):
    """データセット全フレームを読み、スコア降順のリストを返す。

    隣接判定は bag ごとに連番順へ並べ替えた列の前後 2 件だけを見る。
    全フレーム総当たりにすると 2 万件で 4 億回の比較になり実用にならない。
    """
    records = []
    for split in store.splits():
        names = store.frames(split)

        # フレームごとに読み込みと frame_metrics を先に 1 回だけ試す。ここで
        # 失敗したフレームは readable=False にして憶えておき、後段で他フレームの
        # 隣接（temporal_metric）計算に混ぜない。混ぜてしまうと、壊れたラベルの
        # 短い xp 配列が正常な隣接フレームの h_vector マスクで参照されて
        # IndexError になり、隣の正常なフレームまで unreadable に道連れになる
        # （壊れたラベル 1 件で複数フレームが巻き添えになった、というのが
        # 実測で見つかったバグ）。
        labels = {}
        base_metrics = {}
        readable = {}
        for n in names:
            try:
                label = store.load(split, n)
                base_metrics[n] = frame_metrics(label)
            except Exception:  # noqa: BLE001
                labels[n] = None
                readable[n] = False
            else:
                labels[n] = label
                readable[n] = True

        bags, parsed = group_by_bag(names)
        for group in bags.values():
            for i, n in enumerate(group):
                num = parsed[n][1]
                neighbors = []
                for j in (i - 2, i - 1, i + 1, i + 2):
                    if 0 <= j < len(group) and abs(parsed[group[j]][1] - num) <= MAX_NUM_GAP:
                        m = group[j]
                        if readable[m]:
                            neighbors.append(labels[m])
                if not readable[n]:
                    # このフレームは _unreadable_metrics() 行きにして残りは処理を
                    # 続ける。ツールの目的は壊れたラベルを直すことなので、
                    # 読めないラベルこそ最優先で人に見せるべきで、キュー全体を
                    # 落とす理由にはならない。
                    metrics = _unreadable_metrics()
                    unreadable = True
                else:
                    try:
                        metrics = base_metrics[n]
                        metrics[TEMPORAL_KEY] = temporal_metric(labels[n], neighbors)
                        unreadable = False
                    except Exception:  # noqa: BLE001
                        # 万一 temporal_metric 自体が想定外の理由で落ちても、
                        # このフレームだけ unreadable にしてキュー全体は落とさない。
                        metrics = _unreadable_metrics()
                        unreadable = True
                records.append(
                    {"split": split, "name": n, "metrics": metrics, "unreadable": unreadable}
                )
    return score_frames(records, weights)

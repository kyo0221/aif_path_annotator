"""姿勢の初期化・照合・候補・確定をまとめる Annotator。HTTP 層はここへ転送するだけ。"""
import threading
from collections import OrderedDict

from tools.path_annotator import matching
from tools.path_annotator.pose import pose_delta, shift_pose
from tools.path_annotator.projector import LANE_ALIGN_DEG
from tools.path_annotator.store import rejection_reasons, validate_label

COST_THRESHOLD = 0.25
JUMP_LATERAL = 0.3
JUMP_YAW_DEG = 2.0
PREFER_PROPAGATED_MARGIN = 0.03
MIN_POINTS = 30
CANDIDATE_CACHE_SIZE = 32  # 候補のオーバーレイは 1 フレーム数百 KB になるので件数を抑える
POSE_TOL = 1e-9


def _ref(item):
    return None if item is None else {"split": item[0], "name": item[1]}


def select_default(candidates, init_source):
    """既定で採る候補。引き継ぎ姿勢から始めた候補が最良と僅差なら、跳ばないようそちらを選ぶ。"""
    if init_source == "propagated":
        for c in candidates:
            if c["init"] == "propagated" and c["cost"] <= candidates[0]["cost"] + PREFER_PROPAGATED_MARGIN:
                return c
    return candidates[0]


class Annotator:
    def __init__(self, store, poses, frames, gnss, projector, field, detector,
                 cost_threshold=COST_THRESHOLD, prefetcher=None):
        self.store = store
        self.poses = poses
        self.frames = frames
        self.gnss = gnss
        self.projector = projector
        self.field = field
        self.detector = detector
        self.cost_threshold = cost_threshold
        self.prefetcher = prefetcher
        # _initialize は _lock を持ったまま候補キャッシュに触るので再入可能にする
        self._lock = threading.RLock()
        self._candidates = OrderedDict()  # "split/name" -> 表示用の候補 (open_frame の LRU キャッシュ)

    # --- 内部 ---

    @staticmethod
    def _key(split, name):
        return f"{split}/{name}"

    @staticmethod
    def _pose_of(entry):
        return {k: entry[k] for k in ("x", "y", "yaw", "pitch")}

    def _remember_candidates(self, key, shown):
        with self._lock:
            self._candidates[key] = shown
            self._candidates.move_to_end(key)
            while len(self._candidates) > CANDIDATE_CACHE_SIZE:
                self._candidates.popitem(last=False)

    def _cached_candidates(self, key):
        """キャッシュ済みの候補。無ければ None。"""
        with self._lock:
            shown = self._candidates.get(key)
            if shown is not None:
                self._candidates.move_to_end(key)
            return shown

    def _points(self, split, name, pitch):
        mask = self.detector.mask(self.store.image_path(split, name), self._key(split, name))
        return self.projector.bev_points(mask, pitch)

    @staticmethod
    def _has_straight(label):
        return any(e["class"] == "straight" for e in label)

    def _flags(self, cost, pose, init_pose, label, n_points, lane):
        flags = []
        if cost is not None and cost > self.cost_threshold:
            flags.append("cost")
        d = pose_delta(pose, init_pose)
        if abs(d["lateral_m"]) > JUMP_LATERAL or abs(d["yaw_deg"]) > JUMP_YAW_DEG:
            flags.append("jump")
        if not self._has_straight(label):
            flags.append("no_path")
        if n_points < MIN_POINTS:
            flags.append("few_points")
        if pose["pitch"] != 0:
            flags.append("pitch")
        if lane["id"] is None or (lane["angle_deg"] is not None and lane["angle_deg"] >= LANE_ALIGN_DEG):
            flags.append("lane")  # 自動選択が学習側の経路辿りに落ちた、または向きの合わないレーン
        return flags

    def _search(self, key, points, start, start_source, window, init_pose):
        """start を起点に候補を探す。(照合結果, 表示用の候補) を返す。"""
        inits = [(start, start_source)]
        if start_source == "propagated":
            inits.append((self.gnss.initial_pose(key), "gnss"))
        inits += [(shift_pose(start, lateral=matching.LANE_WIDTH), "lane_left"),
                  (shift_pose(start, lateral=-matching.LANE_WIDTH), "lane_right")]
        found = matching.candidates(self.field, points, inits, window)
        shown = []
        for c in found:
            d = pose_delta(c["pose"], init_pose)
            overlay = self.projector.overlay(c["pose"])
            shown.append({
                "pose": c["pose"], "cost": c["cost"], "init": c["init"],
                "lateral_m": d["lateral_m"], "longitudinal_m": d["longitudinal_m"], "yaw_deg": d["yaw_deg"],
                "overlay": {"lines": overlay["lines"], "paths": overlay["paths"]},
            })
        return found, shown

    @staticmethod
    def _window(init_source):
        return matching.NARROW if init_source == "propagated" else matching.WIDE

    def _initial_pose(self, split, name):
        """(init, init_source)。確定済みの直前フレームがあれば引き継ぐ。"""
        key = self._key(split, name)
        if not self.gnss.has(key):
            raise ValueError(f"{key} の GNSS がありません")
        pred = self.frames.predecessor(split, name)
        if pred is not None and self.store.get_status(*pred) == "done":
            pred_entry = self.poses.get(*pred)
            if pred_entry is not None:
                init = self.gnss.propagate(self._key(*pred), self._pose_of(pred_entry), key)
                return init, "propagated"
        return self.gnss.initial_pose(key), "gnss"

    def _initialize(self, split, name):
        """初期姿勢 → 照合 → 保存。保存した entry を返す。"""
        key = self._key(split, name)
        init, init_source = self._initial_pose(split, name)
        points = self._points(split, name, init["pitch"])
        enough = len(points) >= MIN_POINTS
        cost_at_init = matching.cost(self.field, points, init) if enough else None
        if enough:
            found, shown = self._search(key, points, init, init_source, self._window(init_source), init)
            pick = select_default(found, init_source)
            pose, cost = pick["pose"], pick["cost"]
            source = "propagated" if init_source == "propagated" else "auto"
        else:
            shown, pose, cost = [], init, None
            source = "propagated" if init_source == "propagated" else "gnss"
        flags = self._flags(cost, pose, init, self.projector.label(pose), len(points),
                            self.projector.resolve_lane(pose, None))
        entry = self.poses.set(split, name, pose, source, cost, flags,
                               init={"pose": init, "source": init_source, "cost": cost_at_init}, lanelet=None)
        self._remember_candidates(key, shown)
        return entry

    def _entry(self, split, name):
        with self._lock:
            entry = self.poses.get(split, name)
            if entry is None:
                entry = self._initialize(split, name)
            return entry

    def _evaluate(self, split, name, pose, init_pose, lanelet=None, points=None):
        """(cost, flags, label, lane) を返す。lane は resolve_lane の結果。"""
        if points is None:
            points = self._points(split, name, pose["pitch"])
        cost = matching.cost(self.field, points, pose) if len(points) >= MIN_POINTS else None
        label = self.projector.label(pose, lanelet)
        lane = self.projector.resolve_lane(pose, lanelet)
        return cost, self._flags(cost, pose, init_pose, label, len(points), lane), label, lane

    def _lanes(self, pose, lanelet):
        """レーン候補に auto / selected を付けたもの。"""
        auto_id = self.projector.resolve_lane(pose, None)["id"]
        lanes = []
        for item in self.projector.lane_candidates(pose, lanelet):
            auto = item["id"] == auto_id
            lanes.append({**item, "auto": auto, "selected": (item["id"] == lanelet) if lanelet is not None else auto})
        return lanes

    # --- API ---

    def queue(self, order="time", status=None, flag=None):
        items = []
        for i, (split, name) in enumerate(self.frames.order):
            st = self.store.get_status(split, name)
            if status is not None and st != status:
                continue
            e = self.poses.get(split, name)
            flags = list(e["flags"]) if e else []
            if flag == "any" and not flags:
                continue
            if flag not in (None, "any") and flag not in flags:
                continue
            items.append({"split": split, "name": name, "index": i, "status": st, "flags": flags,
                          "cost": e["cost"] if e else None, "source": e["source"] if e else None})
        return items

    def open_frame(self, split, name):
        key = self._key(split, name)
        entry = self._entry(split, name)
        pose, init = self._pose_of(entry), entry["init"]
        if self.prefetcher is not None:
            self.prefetcher.set_cursor(key)
        points = self._points(split, name, pose["pitch"])
        shown = self._cached_candidates(key)
        if shown is None:
            init_points = self._points(split, name, init["pose"]["pitch"])
            shown = []
            if len(init_points) >= MIN_POINTS:
                _, shown = self._search(key, init_points, init["pose"], init["source"],
                                        self._window(init["source"]), init["pose"])
            self._remember_candidates(key, shown)
        cache = {"done": 0, "total": 0}
        if self.prefetcher is not None:
            cache = dict(zip(("done", "total"), self.prefetcher.progress()))
        lanelet, lane_note = entry["lanelet"], None
        if lanelet is not None:
            try:
                self.projector.resolve_lane(pose, lanelet)
            except ValueError:  # 保存済みのレーンが地図に無い: この回は自動で開く
                lane_note = f"保存されたレーン #{lanelet} は使えないので自動に戻しました"
                lanelet = None
        return {
            "split": split, "name": name, "index": self.frames.position(split, name),
            "total": len(self.frames), "status": self.store.get_status(split, name),
            "pose": pose, "source": entry["source"], "cost": entry["cost"], "flags": entry["flags"],
            "init": init, "overlay": self.projector.overlay(pose, lanelet),
            "bev": self.projector.bev(pose, points, lanelet),
            "label": self.projector.label(pose, lanelet), "candidates": shown,
            "lanelet": lanelet, "lanes": self._lanes(pose, lanelet),
            "prev": _ref(self.frames.prev(split, name)), "next": _ref(self.frames.next(split, name)),
            "cache": cache, "lane_note": lane_note,
        }

    def preview(self, split, name, pose, lanelet=None):
        init_pose = self._entry(split, name)["init"]["pose"]
        points = self._points(split, name, pose["pitch"])
        cost, flags, label, lane = self._evaluate(split, name, pose, init_pose, lanelet, points)
        return {"overlay": self.projector.overlay(pose, lanelet), "bev": self.projector.bev(pose, points, lanelet),
                "cost": cost, "label": label, "flags": flags,
                "lanelet": lane["id"], "lanes": self._lanes(pose, lanelet)}

    def match(self, split, name, pose, window, lanelet=None):
        init = self._entry(split, name)["init"]
        points = self._points(split, name, pose["pitch"])
        if len(points) < MIN_POINTS:
            return {"pose": pose, "cost": None, "candidates": []}
        win = matching.NARROW if window == "narrow" else matching.WIDE
        found, shown = self._search(self._key(split, name), points, pose, init["source"], win, init["pose"])
        return {"pose": found[0]["pose"], "cost": found[0]["cost"], "candidates": shown}

    def set_pose(self, split, name, pose, source, lanelet=None):
        entry = self._entry(split, name)
        init = entry["init"]
        cost, flags, _, _ = self._evaluate(split, name, pose, init["pose"], lanelet)
        # 確定済みのフレームの姿勢かレーンを変えたら、ラベルと食い違うので未確認へ戻す
        old = self._pose_of(entry)
        if (self.store.get_status(split, name) == "done"
                and (any(abs(pose[k] - old[k]) > POSE_TOL for k in old) or lanelet != entry["lanelet"])):
            self.store.set_status(split, name, "unseen")
        return self.poses.set(split, name, pose, source, cost, flags, init, lanelet=lanelet)

    def confirm(self, split, name, pose, source, lanelet=None):
        label = self.projector.label(pose, lanelet)
        if not self._has_straight(label):
            raise ValueError("straight 経路を投影できません（レーンを選び直すか Esc で初期値へ）")
        validate_label(label)
        reasons = rejection_reasons(label)
        if reasons:
            raise ValueError("; ".join(reasons))
        self.set_pose(split, name, pose, source, lanelet)
        self.store.save(split, name, label)
        self.store.set_status(split, name, "done")
        nxt = self.frames.next(split, name)
        if nxt is not None and self.prefetcher is not None:
            self.prefetcher.set_cursor(self._key(*nxt))
        return {"ok": True, "next": _ref(nxt)}

    def set_status(self, split, name, status):
        self.store.set_status(split, name, status)

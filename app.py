"""姿勢の初期化・照合・候補・確定をまとめる Annotator。HTTP 層はここへ転送するだけ。"""
import threading

from tools.path_annotator import matching
from tools.path_annotator.pose import pose_delta, shift_pose
from tools.path_annotator.store import rejection_reasons, validate_label

COST_THRESHOLD = 0.25
JUMP_LATERAL = 0.3
JUMP_YAW_DEG = 2.0
PREFER_PROPAGATED_MARGIN = 0.03
MIN_POINTS = 30


def _ref(item):
    return None if item is None else {"split": item[0], "name": item[1]}


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
        self._lock = threading.Lock()
        self._candidates = {}  # "split/name" -> 表示用の候補 (open_frame のキャッシュ)

    # --- 内部 ---

    @staticmethod
    def _key(split, name):
        return f"{split}/{name}"

    @staticmethod
    def _pose_of(entry):
        return {k: entry[k] for k in ("x", "y", "yaw", "pitch")}

    def _points(self, split, name, pitch):
        mask = self.detector.mask(self.store.image_path(split, name), self._key(split, name))
        return self.projector.bev_points(mask, pitch)

    @staticmethod
    def _has_straight(label):
        return any(e["class"] == "straight" for e in label)

    def _flags(self, cost, pose, init_pose, label, n_points):
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
            pick = found[0]
            if init_source == "propagated":
                for c in found:
                    if c["init"] == "propagated" and c["cost"] <= found[0]["cost"] + PREFER_PROPAGATED_MARGIN:
                        pick = c
                        break
            pose, cost = pick["pose"], pick["cost"]
            source = "propagated" if init_source == "propagated" else "auto"
        else:
            shown, pose, cost = [], init, None
            source = "propagated" if init_source == "propagated" else "gnss"
        flags = self._flags(cost, pose, init, self.projector.label(pose), len(points))
        entry = self.poses.set(split, name, pose, source, cost, flags,
                               init={"pose": init, "source": init_source, "cost": cost_at_init})
        self._candidates[key] = shown
        return entry

    def _entry(self, split, name):
        with self._lock:
            entry = self.poses.get(split, name)
            if entry is None:
                entry = self._initialize(split, name)
            return entry

    def _evaluate(self, split, name, pose, init_pose):
        points = self._points(split, name, pose["pitch"])
        cost = matching.cost(self.field, points, pose) if len(points) >= MIN_POINTS else None
        label = self.projector.label(pose)
        return points, cost, label, self._flags(cost, pose, init_pose, label, len(points))

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
        if key not in self._candidates:
            init_points = self._points(split, name, init["pose"]["pitch"])
            shown = []
            if len(init_points) >= MIN_POINTS:
                _, shown = self._search(key, init_points, init["pose"], init["source"],
                                        self._window(init["source"]), init["pose"])
            self._candidates[key] = shown
        cache = {"done": 0, "total": 0}
        if self.prefetcher is not None:
            cache = dict(zip(("done", "total"), self.prefetcher.progress()))
        return {
            "split": split, "name": name, "index": self.frames.position(split, name),
            "total": len(self.frames), "status": self.store.get_status(split, name),
            "pose": pose, "source": entry["source"], "cost": entry["cost"], "flags": entry["flags"],
            "init": init, "overlay": self.projector.overlay(pose), "bev": self.projector.bev(pose, points),
            "label": self.projector.label(pose), "candidates": self._candidates[key],
            "prev": _ref(self.frames.prev(split, name)), "next": _ref(self.frames.next(split, name)),
            "cache": cache,
        }

    def preview(self, split, name, pose):
        init_pose = self._entry(split, name)["init"]["pose"]
        points, cost, label, flags = self._evaluate(split, name, pose, init_pose)
        return {"overlay": self.projector.overlay(pose), "bev": self.projector.bev(pose, points),
                "cost": cost, "label": label, "flags": flags}

    def match(self, split, name, pose, window):
        init = self._entry(split, name)["init"]
        points = self._points(split, name, pose["pitch"])
        if len(points) < MIN_POINTS:
            return {"pose": pose, "cost": None, "candidates": []}
        win = matching.NARROW if window == "narrow" else matching.WIDE
        found, shown = self._search(self._key(split, name), points, pose, init["source"], win, init["pose"])
        return {"pose": found[0]["pose"], "cost": found[0]["cost"], "candidates": shown}

    def set_pose(self, split, name, pose, source):
        init = self._entry(split, name)["init"]
        _, cost, _, flags = self._evaluate(split, name, pose, init["pose"])
        return self.poses.set(split, name, pose, source, cost, flags, init)

    def confirm(self, split, name, pose, source):
        label = self.projector.label(pose)
        if not self._has_straight(label):
            raise ValueError("straight 経路を投影できません")
        validate_label(label)
        reasons = rejection_reasons(label)
        if reasons:
            raise ValueError("; ".join(reasons))
        self.set_pose(split, name, pose, source)
        self.store.save(split, name, label)
        self.store.set_status(split, name, "done")
        nxt = self.frames.next(split, name)
        if nxt is not None and self.prefetcher is not None:
            self.prefetcher.set_cursor(self._key(*nxt))
        return {"ok": True, "next": _ref(nxt)}

    def set_status(self, split, name, status):
        self.store.set_status(split, name, status)

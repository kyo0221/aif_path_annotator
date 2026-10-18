"""split をまたぐ時系列フレームインデックス。"""
import re

MAX_INDEX_GAP = 4  # predecessor とみなす image_index 差の上限

NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<num>\d{6})$")


def parse_frame_name(name):
    """フレーム名を (bag接頭辞, 連番) に分解する。"""
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"frame name must end with a 6-digit number: {name!r}")
    return m.group("prefix"), int(m.group("num"))


class FrameIndex:
    """(bag, image_index) 昇順に split を混ぜて並べたフレーム索引。"""

    def __init__(self, keys, meta):
        self._time = {}
        for split, name in keys:
            self._time[(split, name)] = self._bag_index(split, name, meta)
        self.order = sorted(
            self._time, key=lambda k: (self._time[k], k[0], k[1])
        )
        self._pos = {k: i for i, k in enumerate(self.order)}

    @staticmethod
    def _bag_index(split, name, meta):
        m = meta.get(f"{split}/{name}")
        if m is not None:
            return m["bag"], int(m["image_index"])
        try:
            return parse_frame_name(name)
        except ValueError:
            return name, 0

    def __len__(self):
        return len(self.order)

    def position(self, split, name):
        """order 上の位置。未知のフレームは KeyError。"""
        return self._pos[(split, name)]

    def prev(self, split, name):
        i = self.position(split, name)
        return self.order[i - 1] if i > 0 else None

    def next(self, split, name):
        i = self.position(split, name)
        return self.order[i + 1] if i + 1 < len(self.order) else None

    def predecessor(self, split, name):
        """直前フレームが同じ bag かつ image_index 差が MAX_INDEX_GAP 以下なら返す。"""
        p = self.prev(split, name)
        if p is None:
            return None
        bag, idx = self._time[(split, name)]
        pbag, pidx = self._time[p]
        if pbag == bag and idx - pidx <= MAX_INDEX_GAP:
            return p
        return None

"""ラベルの読み書き。書き込みは必ず原子的に行う。"""
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

from tools.path_annotator.curve import N_ROWS, ROW_START

CLASSES = ("straight", "left", "right")
STATUSES = ("unseen", "done", "skip")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_label(label):
    """ラベルが保存してよい形かを検査する。駄目なら ValueError。"""
    if not isinstance(label, list) or not label:
        raise ValueError("label must be a non-empty list of entries")
    for entry in label:
        if not isinstance(entry, dict):
            raise ValueError(f"each entry must be an object, got {entry!r}")
        if entry.get("class") not in CLASSES:
            raise ValueError(f"class must be one of {CLASSES}, got {entry.get('class')!r}")
        xp = entry.get("xp")
        h = entry.get("h_vector")
        if not isinstance(xp, list) or len(xp) != N_ROWS:
            raise ValueError(f"xp must have {N_ROWS} elements")
        if not isinstance(h, list) or len(h) != N_ROWS:
            raise ValueError(f"h_vector must have {N_ROWS} elements")
        for v in xp:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"xp values must be numbers, got {v!r}")
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"xp values must be in range [0,1], got {v!r}")
        for v in h:
            if isinstance(v, bool) or v not in (0, 1):
                raise ValueError(f"h_vector values must be 0 or 1, got {v!r}")
        for row, flag in enumerate(h):
            if row < ROW_START and flag:
                raise ValueError(
                    f"h_vector must be 0 above row {ROW_START} (row {row} is not used for training)"
                )

    classes = [entry["class"] for entry in label]
    if "straight" not in classes:
        raise ValueError("label must contain a 'straight' entry (the trainer requires it)")
    if len(classes) != len(set(classes)):
        raise ValueError(f"duplicate class in label: {classes} (the trainer keeps only the last one)")


def _check_name(name):
    if not NAME_RE.fullmatch(name) or name.strip(".") == "":
        raise ValueError(f"invalid frame name {name!r}")


def _default_file_mode():
    """保存するラベルのパーミッション。import 時に一度だけ umask を読む。

    os.umask は set-and-return でプロセス全体の状態を触るため、save() から
    呼ぶとスレッド間でレースになる。ここで一度だけ評価して定数にしておく。
    """
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


_FILE_MODE = _default_file_mode()


def _atomic_write_json(path, obj):
    """JSON を原子的に書く。

    書き手ごとに一意な tmp を使うので、同じファイルへの同時書き込みでも内容が混ざらない。
    失敗した場合は tmp を残さず、元のファイルにも触れない。
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), _FILE_MODE)
            json.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


class LabelStore:
    def __init__(self, root):
        self.root = Path(root)
        self._state = None  # annotation_state.json のメモリキャッシュ
        self._state_lock = threading.Lock()

    def splits(self):
        return sorted(p.name for p in (self.root / "labels").iterdir() if p.is_dir())

    def frames(self, split):
        _check_name(split)
        return sorted(p.stem for p in (self.root / "labels" / split).glob("*.txt"))

    def label_path(self, split, name):
        _check_name(split)
        _check_name(name)
        return self.root / "labels" / split / f"{name}.txt"

    def image_path(self, split, name):
        _check_name(split)
        _check_name(name)
        return self.root / "images" / split / f"{name}.png"

    def load(self, split, name):
        return json.loads(self.label_path(split, name).read_text())

    def save(self, split, name, label):
        validate_label(label)
        _atomic_write_json(self.label_path(split, name), label)
        self._update_frame_state(split, name, edited=True)

    # --- バックアップ ---

    def backup_dir(self):
        return self.root / "labels_backup"

    def ensure_backup(self):
        """labels/ の複製が無ければ作る。作ったら True、既にあれば False。

        コピー途中で落ちても中途半端な labels_backup/ を残さないよう、
        一時ディレクトリへコピーしてから rename で確定する。
        """
        dst = self.backup_dir()
        if dst.exists():
            return False
        tmp = tempfile.mkdtemp(dir=str(self.root), prefix=".labels_backup-")
        staging = Path(tmp) / "labels"
        try:
            shutil.copytree(self.root / "labels", staging)
            os.rename(staging, dst)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        else:
            shutil.rmtree(tmp, ignore_errors=True)
        return True

    # --- 進捗状態 ---

    def state_path(self):
        return self.root / "annotation_state.json"

    def load_state(self):
        # キューは 1 リクエストで 2 万件ぶん get_status を呼ぶので、
        # 毎回ファイルを読まずメモリに載せておく。
        if self._state is None:
            path = self.state_path()
            try:
                data = json.loads(path.read_text()) if path.exists() else {}
            except json.JSONDecodeError:
                # 壊れた annotation_state.json で起動不能になるくらいなら、
                # 進捗を失っても起動できる方がよい。空として扱ってフォールバックする。
                data = {}
            if not isinstance(data, dict):
                data = {}
            data.setdefault("frames", {})
            if not isinstance(data["frames"], dict):
                data["frames"] = {}
            self._state = data
        return self._state

    def save_state(self, state):
        _atomic_write_json(self.state_path(), state)
        self._state = state

    def _update_frame_state(self, split, name, **fields):
        # 状態辞書はキャッシュを共有しているので、変異とシリアライズを一括で排他する。
        # 分けると json.dump のイテレート中に別スレッドがキーを足して RuntimeError になる。
        with self._state_lock:
            state = self.load_state()
            entry = state["frames"].setdefault(f"{split}/{name}", {})
            entry.update(fields)
            entry["ts"] = time.time()
            self.save_state(state)

    def set_status(self, split, name, status):
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        self._update_frame_state(split, name, status=status)

    def get_status(self, split, name):
        entry = self.load_state()["frames"].get(f"{split}/{name}", {})
        return entry.get("status", "unseen")

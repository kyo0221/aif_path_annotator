# 一覧（ギャラリー）からのアノテーション 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 投影画像の一覧（`/`）からフレームをクリックして既存のアノテーション画面で直し、変更した時点でラベルが保存され、`q` で一覧の同じ場所へ戻れるようにする。

**Architecture:** 一覧ページは既存サーバの新しい静的ページ + `GET /api/gallery`（全フレームの表）+ `GET /api/thumb`（投影 JPEG、`preview/` にキャッシュ）。アノテーション画面は一覧の中に iframe（`/annotate?mode=gallery&frame=…`）で重ね、`postMessage` でキューと保存結果をやり取りする。ギャラリーモードの画面は姿勢が変わるたびに `POST /api/save` を呼び、サーバが姿勢・ラベル・状態（done）を一度に保存する。

**Tech Stack:** Python 3.12 標準ライブラリ + numpy + cv2（既存どおり）、素の HTML / CSS / JS、pytest。

**Spec:** `docs/superpowers/specs/2026-10-21-gallery-annotation-design.md`

**コミット:** 利用者から指示があるまで git commit はしない（作業ツリーに残す）。各タスクの最後にテストを通すところまでを完了条件にする。

## Global Constraints

- サーバは `http.server`（`ThreadingHTTPServer`）のまま。フレームワーク・npm は使わない。ページは `static/` の素の HTML / CSS / JS。
- ラベルの書き込みは `LabelStore.save`（`validate_label` + 原子的書き込み）だけを通す。保存の可否は `validate_label` と `rejection_reasons`。
- データセット構造は今のまま。投影画像のキャッシュは `<root>/preview/<split>/<name>.jpg`（640×384、JPEG 品質 90）。
- サムネイルは姿勢を初期化せず、`poses.json` に書かない。
- 色: 地図線 BGR `(255, 200, 0)`（#00c8ff、太さ 2）、straight `(0, 230, 255)`、left `(255, 120, 255)`、right `(0, 140, 255)`（太さ 3）、48 行の点は半径 3、行 i の v = `i * (384 // 48) + (384 // 48 - 1) / 2`、u = `xp * 639`。
- テストは `cd /home/kyo/aif_dataset && env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/path_annotator/tests -q -p no:cacheprovider` で走らせる（ROS の環境変数を外す）。
- `Esc` は「初期値へ戻す」のまま。戻るキーは `q`。Enter は「そのまま保存して次へ」。

## Review Focus

1. キーリピート中の自動保存: 1 秒に数十回の姿勢変更で `/api/save` を溜めない（送信中は最新だけ控える）。→ Task 5 の自己テスト 10 で、連打後の `/api/save` 呼び出し回数が変更回数より十分少ないことを見る。
2. 保存の応答が返る前に別フレームへ移った場合、応答を新しいフレームに適用しない。→ Task 5 で応答の `frame` と `S.frame` を比べる実装にし、自己テスト 10 でフレーム移動直後の表示を確認する。
3. `gnss.json` にも `poses.json` にも無いフレームのサムネイル: 生画像 + ラベルの点だけを返し、500 にしない。→ Task 3 のテスト。
4. 親のいない `/annotate?mode=gallery&frame=…`（URL 直打ち）: 500 ms 後に `/api/queue` に落ちて普通に使える。→ Task 5 の自己テスト 10。
5. 絞り込みの結果が 0 件、またはラベルの無いフレーム（classes が空）: 一覧が壊れず「該当なし」/空のチップで出る。→ Task 2 のテスト（classes 空）と Task 6 のスクリーンショット確認。

---

## ファイル構成

- 変更 `store.py`: `LabelStore.save(split, name, label, edited=True)`。
- 変更 `frames.py`: `FrameIndex.bag_index(split, name) -> (bag, image_index)` を公開。
- 新規 `gallery.py`: `Gallery`（表・ラベル情報・サムネイル描画とキャッシュ）、描画関数 `draw_overlay`。
- 変更 `scripts/render_overlay.py`: 描画を `gallery.draw_overlay` に置き換える。
- 変更 `app.py`: `Annotator.gallery`、`Annotator.save(...)`。
- 変更 `server.py`: `/`、`/annotate`、`/api/gallery`、`/api/thumb/...`、`/api/save`。
- 変更 `static/index.html`、`static/style.css`、`static/app.js`: ギャラリーモード。
- 新規 `static/gallery.html`、`static/gallery.css`、`static/gallery.js`: 一覧ページ。
- 変更 `README.md`。
- テスト: `tests/test_store.py`、`tests/test_frames.py`、新規 `tests/test_gallery.py`、`tests/test_app.py`、`tests/test_server.py`。

---

### Task 1: `store.save` の `edited` 引数と `FrameIndex.bag_index`

**Files:**
- Modify: `store.py`（`LabelStore.save`）
- Modify: `frames.py`（`FrameIndex`）
- Test: `tests/test_store.py`、`tests/test_frames.py`

**Interfaces:**
- Produces: `LabelStore.save(split, name, label, edited=True)`。`edited=None` なら `annotation_state.json` の `edited` を触らない（`ts` は更新）。`True` / `False` はその値を書く。
- Produces: `FrameIndex.bag_index(split, name) -> tuple[str, int]`（`_time` の値。未知のフレームは `KeyError`）。

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_store.py` の末尾、既存の import を使う）

```python
def test_save_edited_none_keeps_previous_value(tmp_path):
    store = LabelStore(tmp_path)
    (tmp_path / "images" / "train").mkdir(parents=True)
    label = [{"class": "straight", "xp": [0.5] * 48, "h_vector": [0] * 25 + [1] * 23}]
    store.save("train", "f000000", label, edited=False)
    assert store.load_state()["frames"]["train/f000000"]["edited"] is False
    store.save("train", "f000000", label, edited=None)
    assert store.load_state()["frames"]["train/f000000"]["edited"] is False
    store.save("train", "f000000", label)
    assert store.load_state()["frames"]["train/f000000"]["edited"] is True
```

（`tests/test_frames.py` の末尾）

```python
def test_bag_index_is_public():
    index = FrameIndex([("train", "bagA_000010"), ("val", "bagA_000012")], {})
    assert index.bag_index("train", "bagA_000010") == ("bagA", 10)
    with pytest.raises(KeyError):
        index.bag_index("train", "nope_000000")
```

- [ ] **Step 2: 失敗を確認** Run: `... -m pytest tools/path_annotator/tests/test_store.py tools/path_annotator/tests/test_frames.py -q -k "edited_none or bag_index"` Expected: FAIL（`save() got an unexpected keyword argument 'edited'` / `AttributeError: bag_index`）
- [ ] **Step 3: 実装** `store.py` の `save` に `edited=True` を足し、`fields = {} if edited is None else {"edited": edited}` で `_update_frame_state(split, name, **fields)` を呼ぶ。`frames.py` に `bag_index` を足す（`return self._time[(split, name)]`）。
- [ ] **Step 4: 通ることを確認** Run: 同上 Expected: PASS。続けて全体 `... -m pytest tools/path_annotator/tests -q` Expected: 全件 PASS。

---

### Task 2: `gallery.py` の表（`Gallery.table`）と `Annotator.gallery`

**Files:**
- Create: `gallery.py`
- Modify: `app.py`（`Annotator.__init__` で `self.gallery = Gallery(store, poses, frames, gnss, projector)`）
- Test: `tests/test_gallery.py`（新規）

**Interfaces:**
- Consumes: `LabelStore`（`root`, `splits()`, `frames()`, `label_path()`, `has_label()`, `load()`, `load_state()`, `get_status()`）、`PoseStore.get()`、`FrameIndex.order` / `position()` / `bag_index()`（Task 1）、`GnssIndex.has()` / `initial_pose()`、`Projector.overlay()`。
- Produces:
  - `class Gallery(store, poses, frames, gnss, projector)`。
  - `Gallery.table() -> dict`: `{"frames": [row...], "bags": sorted(set(bag)), "has_manifest": bool, "has_relabeled": bool}`。`row` は時系列順（`frames.order`）で
    `{"split","name","key","index","bag","image_index","status","edited","flags","cost","source","classes","rows","has_manual_label","relabeled","pose","thumb_version"}`。
    `pose` は `poses.get` があれば `"entry"`、無くて `gnss.has(key)` なら `"gnss"`、どちらも無ければ `"none"`。`flags`/`cost`/`source` は姿勢エントリが無ければ `[]`/`None`/`None`。
    `edited` は状態の `edited`（無ければ False）。`has_manual_label` は `<root>/manifest.csv` の `has_manual_label == "1"`、ファイルが無ければ `None`。`relabeled` は `<root>/relabeled_frames.json` の `changed[].key` に含まれるか（無ければ False）。
  - `Gallery.label_info(split, name) -> dict`: `{"classes": [...], "rows": int}`（straight の `h_vector` の和。ラベルが無ければ `{"classes": [], "rows": 0}`）。ファイルの mtime をキーにメモリへキャッシュし、mtime が変わったら読み直す。
  - `Gallery.thumb_version(split, name) -> str`: `f"{label_mtime_ms}-{state_ts_ms}"`（ラベル無し・状態無しは 0）。

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_gallery.py`。`mini_root` と `StubProjector` を使う。`mini_root` には `train/f000000` だけラベルがあり、`gnss.json` の frames は 3 件）

```python
import json

from tools.path_annotator.frames import FrameIndex
from tools.path_annotator.gallery import Gallery
from tools.path_annotator.gnss import GnssIndex
from tools.path_annotator.pose import PoseStore
from tools.path_annotator.store import LabelStore
from tools.path_annotator.tests.conftest import StubProjector


def make_gallery(root):
    store = LabelStore(root)
    gnss = GnssIndex(root / "gnss.json")
    keys = [(s, n) for s in store.splits() for n in store.frames(s)]
    return Gallery(store, PoseStore(root), FrameIndex(keys, gnss.meta()), gnss, StubProjector()), store


def test_table_lists_all_frames_in_time_order(mini_root):
    g, _ = make_gallery(mini_root)
    t = g.table()
    assert [r["key"] for r in t["frames"]] == ["train/f000000", "train/f000002", "val/f000004"]
    assert t["bags"] == ["bag"] and t["has_manifest"] is False and t["has_relabeled"] is False
    r0 = t["frames"][0]
    assert r0["classes"] == ["straight"] and r0["rows"] == 23 and r0["status"] == "unseen" and r0["edited"] is False
    assert r0["pose"] == "gnss" and r0["flags"] == [] and r0["cost"] is None
    assert t["frames"][1]["classes"] == [] and t["frames"][1]["rows"] == 0
    assert all(r["has_manual_label"] is None and r["relabeled"] is False for r in t["frames"])


def test_table_reads_manifest_and_relabeled(mini_root):
    (mini_root / "manifest.csv").write_text("key,has_manual_label\ntrain/f000000,1\ntrain/f000002,0\n")
    (mini_root / "relabeled_frames.json").write_text(json.dumps({"changed": [{"key": "val/f000004"}]}))
    g, _ = make_gallery(mini_root)
    t = g.table()
    by = {r["key"]: r for r in t["frames"]}
    assert t["has_manifest"] and t["has_relabeled"]
    assert by["train/f000000"]["has_manual_label"] is True and by["train/f000002"]["has_manual_label"] is False
    assert by["val/f000004"]["has_manual_label"] is None and by["val/f000004"]["relabeled"] is True


def test_label_info_follows_file_changes(mini_root):
    g, store = make_gallery(mini_root)
    assert g.label_info("train", "f000002") == {"classes": [], "rows": 0}
    h = [0] * 25 + [1] * 10 + [0] * 13
    store.save("train", "f000002", [{"class": "straight", "xp": [0.5] * 48, "h_vector": h},
                                    {"class": "left", "xp": [0.4] * 48, "h_vector": h}])
    assert g.label_info("train", "f000002") == {"classes": ["straight", "left"], "rows": 10}
    assert g.table()["frames"][1]["status"] == "unseen" and g.table()["frames"][1]["edited"] is True
    assert g.thumb_version("train", "f000002") != "0-0"


def test_annotator_has_gallery(annotator):
    assert annotator.gallery.table()["frames"]
```

- [ ] **Step 2: 失敗を確認** Run: `... -m pytest tools/path_annotator/tests/test_gallery.py -q` Expected: FAIL（`ModuleNotFoundError: tools.path_annotator.gallery`）
- [ ] **Step 3: 実装** `gallery.py` に `Gallery` を書く。`__init__` で `manifest.csv`（`csv.DictReader`、`key` 列）と `relabeled_frames.json` を 1 回だけ読む（無ければ空）。`label_info` は `{key: (mtime, info)}` の dict でキャッシュ。`table()` は `frames.order` を回して行を作る（`frames.position` は order の添字そのもの）。`app.py` の `Annotator.__init__` の末尾で `self.gallery = Gallery(store, poses, frames, gnss, projector)`（import は `from tools.path_annotator.gallery import Gallery`）。
- [ ] **Step 4: 通ることを確認** Run: 同上 Expected: PASS。全体も PASS。

---

### Task 3: サムネイルの描画とキャッシュ（`Gallery.thumb`）

**Files:**
- Modify: `gallery.py`
- Modify: `scripts/render_overlay.py`（`_polyline` / `render` を消し、`from tools.path_annotator.gallery import draw_overlay` を使う）
- Test: `tests/test_gallery.py`

**Interfaces:**
- Produces:
  - `draw_overlay(img, overlay, label=None) -> img`: `overlay["lines"]`（シアン・2）、`overlay["paths"]`（クラス色・3）、点は `label`（保存済みラベルの list）があればそれから、無ければ `overlay["rows"]` から。Global Constraints の色・位置の式。`img` を書き換えて返す。
  - `Gallery.thumb(split, name) -> bytes`: JPEG。`<root>/preview/<split>/<name>.jpg` があり、その mtime が `max(ラベル mtime, 姿勢エントリ ts, 状態 ts)` 以上ならそれを返す。無い・古いなら描いて書いてから返す。描画は `threading.Lock` で直列。画像が無ければ `FileNotFoundError`。
  - 描く姿勢: `poses.get` があれば `(pose, lanelet)`、無くて `gnss.has(key)` なら `(gnss.initial_pose(key), None)`、どちらも無ければ overlay 無しで生画像 + ラベルの点だけ。`poses.json` には一切書かない。
  - 画像サイズが 640×384 でなければ `cv2.resize`。

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_gallery.py` に追記）

```python
import os
import time

import cv2
import numpy as np


def test_thumb_renders_and_caches_without_touching_poses(mini_root):
    g, store = make_gallery(mini_root)
    before = (mini_root / "poses.json").read_text() if (mini_root / "poses.json").exists() else None
    data = g.thumb("train", "f000002")  # 姿勢エントリ無し → GNSS 初期姿勢で描く
    assert data[:2] == b"\xff\xd8"
    cached = mini_root / "preview" / "train" / "f000002.jpg"
    assert cached.is_file() and cached.read_bytes() == data
    after = (mini_root / "poses.json").read_text() if (mini_root / "poses.json").exists() else None
    assert after == before
    assert g.table()["frames"][1]["pose"] == "gnss"


def test_thumb_rerenders_when_label_is_newer(mini_root):
    g, store = make_gallery(mini_root)
    g.thumb("train", "f000000")
    cached = mini_root / "preview" / "train" / "f000000.jpg"
    old = os.stat(cached).st_mtime_ns
    time.sleep(0.01)
    store.save("train", "f000000", [{"class": "straight", "xp": [0.2] * 48, "h_vector": [0] * 25 + [1] * 23}])
    g.thumb("train", "f000000")
    assert os.stat(cached).st_mtime_ns > old


def test_thumb_without_gnss_or_pose_is_raw_image_with_dots(mini_root):
    gnss = json.loads((mini_root / "gnss.json").read_text())
    del gnss["frames"]["val/f000004"]
    (mini_root / "gnss.json").write_text(json.dumps(gnss))
    g, _ = make_gallery(mini_root)
    assert g.table()["frames"][2]["pose"] == "none"
    data = g.thumb("val", "f000004")
    assert data[:2] == b"\xff\xd8"


def test_draw_overlay_puts_dots_at_label_rows():
    from tools.path_annotator.gallery import draw_overlay

    img = np.zeros((384, 640, 3), np.uint8)
    label = [{"class": "straight", "xp": [0.5] * 48, "h_vector": [0] * 47 + [1]}]
    draw_overlay(img, {"lines": [], "paths": {}, "rows": {}}, label)
    assert tuple(img[379, 319]) == (0, 230, 255)  # 行 47: v = 47*8 + 3.5 → 379.5、u = 0.5*639 → 319.5
```

- [ ] **Step 2: 失敗を確認** Run: `... -m pytest tools/path_annotator/tests/test_gallery.py -q -k "thumb or draw_overlay"` Expected: FAIL（`AttributeError: thumb` / `ImportError: draw_overlay`）
- [ ] **Step 3: 実装** `gallery.py` に `draw_overlay` と `Gallery.thumb` を書く（`scripts/render_overlay.py` の `_polyline` / `render` を移す。JPEG は `cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])`、書き込みは同じディレクトリの一時ファイルへ書いて `os.replace`）。`scripts/render_overlay.py` を `draw_overlay(img, projector.overlay(pose), label)` を呼ぶ形に直す（`label` は `labels/<split>/<name>.txt` があれば読む）。
- [ ] **Step 4: 通ることを確認** Run: 同上 Expected: PASS。全体も PASS。
- [ ] **Step 5: スクリプトの動作確認** Run: `cd /home/kyo/aif_dataset && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/path_annotator/scripts/render_overlay.py --root /home/kyo/Downloads/base4_annotation --learning-repo /home/kyo/Downloads/annotator_kit/annotator_kit/learning --vectormap /home/kyo/Downloads/annotator_kit/annotator_kit/assets/aiformula_course.osm --params /home/kyo/Downloads/annotator_kit/annotator_kit/assets/main_params.yaml --out /tmp/claude-1000/-home-kyo-Downloads/71d25e46-f786-4d1b-a2c4-597c5fe0c8ca/scratchpad/ro_check train/630_3ms_nav_002780` Expected: 1 行出力、PNG ができる。

---

### Task 4: `Annotator.save` と API ルート

**Files:**
- Modify: `app.py`（`save`）
- Modify: `server.py`（`/`、`/annotate`、`/api/gallery`、`/api/thumb/<split>/<name>.jpg`、`/api/save`）
- Test: `tests/test_app.py`、`tests/test_server.py`

**Interfaces:**
- Consumes: Task 2/3 の `Annotator.gallery`（`table()`, `thumb()`, `thumb_version()`, `label_info()`）、Task 1 の `store.save(..., edited=)`。
- Produces:
  - `Annotator.save(split, name, pose, source, lanelet=None) -> dict`:
    `{"saved": bool, "status": str, "reasons": list[str], "flags", "cost", "label", "lanelet", "lanes", "classes", "rows", "edited": bool, "thumb_version": str}`。
    手順: `entry = self._entry(...)`; `cost, flags, label, lane = self._evaluate(split, name, pose, entry["init"]["pose"], lanelet)`; `self.poses.set(split, name, pose, source, cost, flags, entry["init"], lanelet=lanelet)`;
    `reasons`: straight が無ければ `["straight 経路を投影できません（レーンを選び直すか Esc で初期値へ）"]`、`validate_label` の `ValueError` ならその文言、それ以外は `rejection_reasons(label)`。
    `reasons` が空なら `changed = not self.store.has_label(...) or self.store.load(...) != label`、`self.store.save(split, name, label, edited=True if changed else None)`、`set_status("done")`。
    空でなければラベルはそのまま、`get_status == "done"` のとき `set_status("unseen")`。
    `classes`/`rows` は保存後の `gallery.label_info`、`edited` は状態の値、`lanelet` は `lane["id"]`。
  - ルート: `GET /` → `gallery.html`、`GET /annotate` → `index.html`、`GET /api/gallery` → `app.gallery.table()`、
    `GET /api/thumb/<split>/<name>.jpg` → `image/jpeg` + `Cache-Control: no-cache`（`.jpg` を外して `app.gallery.thumb`。画像が無ければ 404）、
    `POST /api/save` → `app.save(split, name, pose, source, lanelet)`。

- [ ] **Step 1: 失敗するテストを書く**（`tests/test_app.py` 末尾）

```python
def test_save_writes_label_and_marks_done(annotator, mini_root):
    out = annotator.save("train", "f000002", {"x": 6.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    assert out["saved"] is True and out["status"] == "done" and out["reasons"] == []
    assert out["classes"] == ["straight"] and out["edited"] is True and out["thumb_version"] != "0-0"
    assert (mini_root / "labels" / "train" / "f000002.txt").is_file()
    assert annotator.poses.get("train", "f000002")["x"] == 6.0
    again = annotator.save("train", "f000002", {"x": 6.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    assert again["saved"] is True and again["status"] == "done"


def test_save_same_label_keeps_edited(annotator, mini_root):
    first = annotator.save("train", "f000002", {"x": 6.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    annotator.store._update_frame_state("train", "f000002", edited=False)
    same = annotator.save("train", "f000002", {"x": 6.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    assert first["saved"] and same["saved"] and same["edited"] is False


def test_save_rejected_keeps_label_and_unconfirms(annotator_few_rows, mini_root):
    a = annotator_few_rows
    a.store.set_status("train", "f000000", "done")
    before = (mini_root / "labels" / "train" / "f000000.txt").read_text()
    out = a.save("train", "f000000", {"x": 5.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual")
    assert out["saved"] is False and out["status"] == "unseen" and out["reasons"]
    assert (mini_root / "labels" / "train" / "f000000.txt").read_text() == before
    assert a.poses.get("train", "f000000")["x"] == 5.0


def test_save_without_straight_reports_reason(annotator):
    out = annotator.save("train", "f000000", {"x": 5.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}, "manual", lanelet=2)
    assert out["saved"] is False and "straight" in out["reasons"][0]
```

（`tests/test_server.py` 末尾）

```python
def test_gallery_pages_and_api(server, mini_root):
    status, html = call(server, "GET", "/")
    assert status == 200 and b"gallery.js" in html
    status, html = call(server, "GET", "/annotate")
    assert status == 200 and b"app.js" in html
    status, t = call(server, "GET", "/api/gallery")
    assert status == 200 and len(t["frames"]) == 3 and t["frames"][0]["key"] == "train/f000000"
    status, jpg = call(server, "GET", "/api/thumb/train/f000000.jpg?v=1")
    assert status == 200 and jpg[:2] == b"\xff\xd8"
    status, _ = call(server, "GET", "/api/thumb/train/nope.jpg")
    assert status == 404
    status, s = call(server, "POST", "/api/save", {"split": "train", "name": "f000002", "pose": POSE, "source": "manual", "lanelet": None})
    assert status == 200 and s["saved"] is True and s["status"] == "done"
    assert (mini_root / "labels" / "train" / "f000002.txt").is_file()
```

- [ ] **Step 2: 失敗を確認** Run: `... -m pytest tools/path_annotator/tests/test_app.py tools/path_annotator/tests/test_server.py -q -k "save or gallery"` Expected: FAIL（`AttributeError: save`、`/` が index.html を返す）
- [ ] **Step 3: 実装** `app.py` に `save` を足す。`server.py` の `do_GET` に `/`（`gallery.html`）、`/annotate`（`index.html`）、`/api/gallery`、`/api/thumb`（`parts == ["api","thumb",split,file]` で `file.endswith(".jpg")`）を足し、`do_POST` の許可パスに `/api/save` を足す。サムネイル応答は `_send` の前に `Cache-Control: no-cache` を付けるため、`_send(code, body, content_type, extra_headers=None)` に拡張する。`static/gallery.html` はこのタスクでは最小の雛形（`<script src="/static/gallery.js">` を含むだけ）を置き、Task 6 で中身を書く。
- [ ] **Step 4: 通ることを確認** Run: 同上 Expected: PASS。全体も PASS。

---

### Task 5: アノテーション画面のギャラリーモード

**Files:**
- Modify: `static/index.html`（上部バーに `<button id="tb-back" hidden>一覧へ戻る (q)</button>` と `<span id="tb-save" class="badge" hidden></span>`、`#keys` の中身を JS で差し替えられるようにする）
- Modify: `static/style.css`（`#tb-back` の見た目。`.badge.saved { background: var(--good) }`、`.badge.unsaved { background: var(--warn) }`）
- Modify: `static/app.js`
- Test: `static/app.js` の自己テスト `?selftest=10`（手動・ヘッドレス）。pytest の対象外。

**Interfaces:**
- Consumes: `POST /api/save`（Task 4 の応答）、`GET /api/queue`（従来）。
- Produces（親との約束。Task 6 が使う）:
  - 子 → 親 `{"type":"ready"}`（起動時、親がいるとき）。
  - 親 → 子 `{"type":"queue","items":[{"split","name","status","flags"}],"current":{"split","name"}}`。
  - 子 → 親 `{"type":"saved","key","saved","status","flags","cost","classes","rows","edited","thumb_version","reasons"}`、`{"type":"status","key","status"}`、`{"type":"current","key"}`、`{"type":"close"}`。
  - URL: `/annotate?mode=gallery&frame=<split>/<name>`。

- [ ] **Step 1: 自己テスト 10 を先に書く**（`selftest` に追加。親無しで `/annotate?mode=gallery&frame=train/<2 番目のフレーム>&selftest=10` を開いたときの期待を `#msg` に出す）

```js
if (mode === 10) {
  // (a) 親がいないので /api/queue に落ちている（S.items が時系列全件）、(b) 開いたのは URL の frame、
  // (c) ← → は S.items の順、(d) 連打 20 回で /api/save は 20 回未満、(e) 移動直後に古い応答で badge が変わらない
  const n0 = S.items.length, name0 = S.frame.name;
  for (let i = 0; i < 20; i++) key('keydown', 'l');
  await wait(50); await idle(); while (S.saving) await wait(30);
  const saves = S.saveCount;
  key('keydown', 'ArrowRight'); await wait(300); await idle(); while (S.saving) await wait(30);
  return msg(`SELFTEST 10: mode=${S.mode} items=${n0} start=${name0} saves=${saves} now=${S.frame.name} badge=${$('tb-save').textContent}`);
}
```

期待: `mode=gallery`、`items=` 全フレーム数、`start=` URL のフレーム、`saves` < 20 かつ ≥ 1、`now=` その次のフレーム、`badge=` 空または「保存 done」。

- [ ] **Step 2: 実装（`app.js`）**
  - `const PARAMS = new URLSearchParams(location.search); S.mode = PARAMS.get('mode') === 'gallery' ? 'gallery' : 'classic'; const inParent = window.parent !== window; S.saving = false; S.saveLatest = null; S.saveCount = 0; S.lastSave = null;`
  - `function notifyParent(m) { if (inParent) window.parent.postMessage(m, location.origin); }`
  - `function queueFromParent(timeoutMs) -> Promise<items|null>`: 親無しなら `null`。`message` を待ちつつ `ready` を送る。`timeoutMs` で `null`。
  - `start()`: ギャラリーモードなら `document.body.classList.add('gallery-mode')`、`$('tb-back').hidden = false`、`$('tb-save').hidden = false`、`#keys` の文字列を `'Enter そのまま保存して次へ ｜ q 一覧へ戻る ｜ ← → 前/次（一覧の順） ｜ h l 横 ｜ j k 前後 ｜ u i ヨー ｜ , . ピッチ（Shift で 5 倍） ｜ Esc 初期値へ ｜ z / Shift+z 戻す/やり直す ｜ g マスク ｜ m 地図線 ｜ [ ] 投影の不透明度 ｜ o 押す間は素の画像 ｜ s skip ｜ t 使用レーンを巡回 ｜ 1〜4 候補を選ぶ ｜ r 広く再探索 ｜ + - 俯瞰の縮尺'` に差し替え。`S.items = (await queueFromParent(500)) || (await loadQueue(), S.items)`。`frame` パラメータを `{split,name}` にして `openFrame`、無ければ `S.items[0]`。
  - `renderTopbar`: ギャラリーモードは `tb-order` を `'一覧の順'`、`tb-index` を `${S.idx + 1} / ${S.items.length}`。
  - `renderSaveBadge()`: `S.lastSave` が無ければ空、`saved` なら `'保存 done'` + class `badge saved`、そうでなければ `'未保存: ' + reasons.join('; ')` + class `badge unsaved`。
  - `autosave()`: ギャラリーモード以外は何もしない。`S.saveLatest = {frame: S.frame, pose: S.pose, source: S.source, lanelet: S.lanelet}`; 送信中なら戻る。送信ループ: `S.saveLatest` を取り出して `/api/save`、`S.saveCount++`。応答の `frame` が `S.frame` と同じときだけ `S.lastSave` を更新して `renderSaveBadge()`、`#msg` に `saved` が偽なら `未保存: …`、真で直前の表示が `未保存` なら消す。`S.items` の該当要素の `status`/`flags` を更新して `renderQueue()`。`notifyParent({type:'saved', key, ...body})`。失敗は `msg('保存に失敗: …')`。
  - `saveSettled()`: `while (S.saving) await wait(30)`。`saveNow()`: `autosave(); await saveSettled(); return S.lastSave`。
  - `applyPose` と `setLane` の末尾で `autosave()`。
  - `neighbor(dir)`: `if (S.mode === 'gallery' || S.flagOnly)` で一覧順。
  - `leaveTo`: ギャラリーモードは `savePose()` の代わりに `await saveSettled()`。`openFrame` の後に `notifyParent({type:'current', key})`。`reloadQueue`: ギャラリーモードは `renderQueue()` だけ。
  - `confirm`（Enter）: ギャラリーモードは `const r = await saveNow(); if (!r || !r.saved) return msg('未保存: ' + (r ? r.reasons.join('; ') : '保存に失敗'));` のあと `neighbor(+1)` へ。
  - `skip`: 先頭で `await saveSettled()`。成功後 `S.items` の status を `skip` にして `notifyParent({type:'status', key, status:'skip'})`。
  - キー: `q` → `closeToGallery()`（`await saveSettled(); notifyParent({type:'close'})`。親が無ければ `msg('一覧から開いたときだけ戻れます')`）。`f` → ギャラリーモードは `msg('一覧側で絞り込んでください')`。`once` の配列に `'q'` を足す。`#tb-back` の click も `closeToGallery`。
  - 自己テスト 10 を Step 1 のとおり足す。
- [ ] **Step 3: 動作確認（ヘッドレス）** 実データを汚さないよう、まず `cp -al /home/kyo/Downloads/base4_annotation /tmp/claude-1000/-home-kyo-Downloads/71d25e46-f786-4d1b-a2c4-597c5fe0c8ca/scratchpad/base4_scratch`（ハードリンクの複製。ツールの書き込みは一時ファイル + `os.replace` なので元の inode は変わらない）を作る。サーバを `annotate.py --root <base4_scratch> --learning-repo /home/kyo/Downloads/annotator_kit/annotator_kit/learning --vectormap /home/kyo/Downloads/annotator_kit/annotator_kit/assets/aiformula_course.osm --yolop /home/kyo/Downloads/annotator_kit/annotator_kit/assets/yolopv2.pt --params /home/kyo/Downloads/annotator_kit/annotator_kit/assets/main_params.yaml --no-browser --port 8799` で起動し、`google-chrome --headless=new --disable-gpu --window-size=1400,900 --virtual-time-budget=15000 --screenshot=... "http://127.0.0.1:8799/annotate?mode=gallery&frame=train/630_3ms_nav_002780&selftest=10"` を撮る。Expected: 画面下の `#msg` に `SELFTEST 10: mode=gallery items=8296 start=630_3ms_nav_002780 saves=<1〜19> now=630_3ms_nav_002782 ...`、上部バーに「一覧へ戻る (q)」と「保存 done」。終わったら `/home/kyo/Downloads/base4_annotation` の `labels/`・`poses.json`・`annotation_state.json` が変わっていないこと（`stat -c %Y` が起動前と同じ）を確認する。
- [ ] **Step 4: 従来モードが変わっていないこと** `/annotate?selftest=2` のスクリーンショットで Enter による確定と次フレームが今までどおり動く。pytest 全体 PASS。

---

### Task 6: 一覧ページ

**Files:**
- Create: `static/gallery.html`（Task 4 の雛形を置き換え）、`static/gallery.css`、`static/gallery.js`
- Test: ヘッドレス Chrome のスクリーンショット（pytest の対象外）

**Interfaces:**
- Consumes: `GET /api/gallery`（Task 2 の表）、`GET /api/thumb/<split>/<name>.jpg?v=`、Task 5 のメッセージ。
- Produces: 一覧ページ。`/home/kyo/Downloads/base4_annotation/preview/index.html`（`scratchpad/scripts/build_gallery.py` が作ったもの）のレイアウト・配色・絞り込み UI を踏襲する。

- [ ] **Step 1: `gallery.html` / `gallery.css`** ヘッダ（タイトル + 件数 + 「順番に回る（従来の画面）」リンク `/annotate`）、絞り込み（split / bag / 状態 / フラグ / 姿勢 / 経路 / 手で直した / 手動ラベル / 再投影 / 並び / 検索 / 幅）、凡例（色と「クリックで開く、q で戻る」）、`#grid`、`#sentinel`、`#empty`、`#annot`（`position:fixed; inset:0; display:none` の層に `<iframe id="annot-frame">` を 100% で置く）。手動ラベル・再投影の `<label>` は `has_manifest` / `has_relabeled` が偽なら `hidden`。配色トークンとカード・チップの CSS は `preview/index.html` のものを使う（`.chip.status-unseen` 灰、`.status-done` 緑、`.status-skip` 橙、`.chip.pose-none` 灰枠「姿勢なし」、`.card.last` に `outline: 2px solid var(--accent)`）。
- [ ] **Step 2: `gallery.js`**
  - 起動: `fetch('/api/gallery')` → `DATA`（各行に `key` を持つ）。`bags` でセレクトを埋める。ハッシュ（`#split=…&bag=…&status=…&flag=…&src=…&cls=…&edited=…&manual=…&relabel=…&sort=…&q=…&w=…`）から絞り込みを復元し、変更のたびに `history.replaceState` で書く。並びの既定は `time`（`index` 順）。
  - `applyFilter()`: 条件で `VIEW` を作り、120 件ずつ `IntersectionObserver` で追加（`preview/index.html` と同じ）。0 件なら `#empty` を表示。
  - `card(row)`: `data-key`、`<img loading="lazy" src="/api/thumb/${split}/${name}.jpg?v=${thumb_version}">`、名前、チップ（状態、bag、source、cost、flags、left/right、手で直した、手動、再投影、姿勢なし）。クリックで `openAnnotator(row)`。
  - `openAnnotator(row)`: `QUEUE = VIEW.map(r => ({split, name, status, flags}))`、`#annot` を表示、`iframe.src = '/annotate?mode=gallery&frame=' + key`、`iframe.focus()`、`document.body.style.overflow = 'hidden'`（背景のスクロールを止めるが `scrollY` は保持する: 表示前に `SAVED_SCROLL = window.scrollY` を取り、閉じたら `window.scrollTo(0, SAVED_SCROLL)`）。
  - `message` 受信（`event.origin === location.origin` のみ）: `ready` → `iframe.contentWindow.postMessage({type:'queue', items: QUEUE, current:{split,name}}, location.origin)`。`saved` → `DATA` の行を更新（`status, flags, cost, classes, rows, edited, thumb_version`）し `CHANGED.add(key)`、カードのチップを描き直す。`status` → 同様に `status` だけ。`current` → `LAST = key`。`close` → `closeAnnotator()`。
  - `closeAnnotator()`: `iframe.src = 'about:blank'`、`#annot` を隠す、`overflow` を戻す、`scrollTo(0, SAVED_SCROLL)`、`CHANGED` のカードの `img.src` を新しい `v` で差し替え、`.card.last` を `LAST` のカードに付け替え、`CHANGED.clear()`。
  - 一覧側のキー: `#annot` 表示中は何もしない。
- [ ] **Step 3: 動作確認（ヘッドレス）** Task 5 のサーバで `http://127.0.0.1:8799/` と `http://127.0.0.1:8799/#status=done&sort=cost` をスクリーンショット。Expected: カードに投影画像と状態チップが出る。`#bag=630_3ms_nav&q=00278` で 2780/2782/2784 だけが出る。`#status=skip` で「該当なし」。
- [ ] **Step 4: 一覧 → iframe → 戻る の往復** ヘッドレスではクリックできないので、`gallery.js` に `?selftest=1` を足す: 起動後に 1 枚目を `openAnnotator`、`ready` に `queue` を返し、`saved`/`current` を受けたら `closeAnnotator()` して `#msg` 相当（`#count`）に `SELFTEST 1: opened=<key> queue=<n> closed=1 changed=<n>` を出す。iframe 側は Task 5 の自己テスト 10 を使わず、`&selftest=11`（起動直後に `l` を 1 回押して `saved` を送り、`q` を押す）を足す。スクリーンショットで `closed=1 changed=1` を確認（サーバは Task 5 と同じ scratch 複製）。

---

### Task 7: README と最終確認

**Files:**
- Modify: `README.md`（「画面」「操作」「進め方」「保存されるファイル」に一覧の流れを追記。起動 URL が一覧になること、`/annotate` が従来画面であること、`q` で戻ること、変更時点で保存・done になること、保存できない姿勢の扱い、`preview/` キャッシュ）
- Modify: `annotate.py` の docstring（起動後に開くのが一覧であること）

- [ ] **Step 1: README を更新**
- [ ] **Step 2: 全テスト** Run: Global Constraints のコマンド Expected: 全件 PASS。
- [ ] **Step 3: 最終確認** scratch 複製のサーバで一覧を開き、`#bag=630_3ms_nav&q=00278` の 3 枚が新しいラベル（straight + left）で描かれていることを確認。実データ `/home/kyo/Downloads/base4_annotation` の `labels/`・`poses.json`・`annotation_state.json`・`relabeled_frames.txt` が変わっていないこと（mtime）を確認。`git status` の変更がツール側のファイルだけであることを確認。

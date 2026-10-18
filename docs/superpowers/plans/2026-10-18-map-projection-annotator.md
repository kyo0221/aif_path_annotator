# 地図投影アノテータ 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 経路ラベルを「フレームごとの車両姿勢から vectormap を投影して生成」する方式に作り替え、人は俯瞰パネルと候補選択で姿勢を直すだけにする。

**Architecture:** Python 側（標準ライブラリ HTTP サーバ）が投影・照合・ラベル生成の正本を持ち、ブラウザは幾何を受け取って canvas に描くだけ。
照合は地図線の距離場（5 cm 格子）で全仮説をベクトル化し、YOLOP マスクはディスクにキャッシュして裏で先読みする。
既存の `store.py`（原子的書き込み・検証・状態）と `bev_match` の探索ロジックは土台として残し、制御点編集・推論ラベル・異常スコアは削除する。

**Tech Stack:** Python 3.12（`/home/kyo/aif_dataset/.venv`: numpy 2.x、scipy、opencv-python-headless、pyyaml、torch CPU、pytest）、素の JS + canvas。学習リポジトリ `/home/kyo/formula_ws/src/learning` の `Models.data_utils.vision_planner.{projection,vectormap}` を投影の正本として import する。

**Spec:** `docs/superpowers/specs/2026-10-18-map-projection-annotator-design.md`

## Global Constraints

- 実行は `cd /home/kyo/aif_dataset && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python ...`。テストは `cd /home/kyo/aif_dataset && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/path_annotator/tests -q`（以下「pytest」と書いたらこのコマンド。`-k` で絞る）。
- import 名は `tools.path_annotator.<module>`（リポジトリは `/home/kyo/aif_dataset/tools/path_annotator` にあり、`annotate.py` が `parents[2]` を `sys.path` に入れる）。
- torch と学習リポジトリはモジュール冒頭で import しない（遅延 import）。学習リポジトリが要るテストは `tests/conftest.py` の `learning` フィクスチャを使い、無ければ skip。
- HTTP は `http.server` のみ、バインドは `127.0.0.1`。フロントはフレームワーク無し。
- ラベル形式: 48 行、`xp` ∈ [0,1]、`h_vector` ∈ {0,1}、行 0〜24 は無効、straight 必須、class 重複なし（`store.validate_label` を通す）。
- 確定時の棄却基準（converter.py と同値、コピー）: `MIN_VISIBLE_ROWS = 10`、`EDGE_RATIO = 0.5`、`EDGE_MARGIN = 0.01`。
- 照合: 狭い窓 横 ±0.5 m / 0.05、前後 ±0.5 m / 0.1、ヨー ±3° / 0.5°。広い窓 横 ±1.5 m / 0.1、前後 ±1.5 m / 0.25、ヨー ±8° / 1°。打ち切り 0.5 m。距離場 5 cm 格子、量子化誤差 ≤ 3 cm。コストしきい値既定 0.25 m。
- 候補: 最大 4 件、間引き 位置 0.4 m・ヨー 2°、レーン幅 3.3 m、引き継ぎ優先のコスト差 0.03 m。要確認: `jump` は横 0.3 m またはヨー 2°。白線点が 30 点未満は照合しない。引き継ぎは同じ bag で `image_index` 差 4 以内。
- 速度予算: 広い窓 0.3 秒以内、候補 4 件合計 1 秒以内（assert しない。計測スクリプトを残す）。
- 色: 地図線 `#00c8ff`、マスク `#00dc00`、straight `#ffe600`、left `#ff78ff`、right `#ff8c00`。線幅は地図線 2 px、経路 3 px、拡大してもスケールしない。
- ファイル配置: `poses.json`・`annotation_state.json`・`labels_backup/`・`cache/yolop/<split>/<name>.png`・`excluded/<split>/` はすべてデータセットのルート直下。
- コミットは小さく、メッセージはこのリポジトリの流儀（絵文字 + 種別、日本語。例 `✨feat:`、`🧪test:`、`♻️refactor:`、`📝docs:`）で、末尾に `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` を付ける。コミット前に `git status` に `__pycache__` が出ていないこと。

## Review Focus

- `gnss.json` にそのフレームのキーが無い: `/api/frame` は 400 で「GNSS がありません」を返し、キューや他フレームは動き続ける（Task 7 のテスト）。
- 姿勢が地図の外（距離場の格子外）: コストは打ち切り値 0.5 で返り、例外にならない（Task 4 のテスト）。
- `poses.json` が壊れた JSON: 空として起動し、次の保存で上書きされる（Task 1 のテスト）。
- マスクキャッシュの PNG が壊れている（読めない）: 再計算して上書きする（Task 6 のテスト）。
- 先読みスレッドと画面操作が同時に YOLOP を呼ぶ: モデル呼び出しはロックで直列化し、同じフレームを二重に計算しない（Task 6 のテスト）。

---

## ファイル構成

| ファイル | 責務 |
|---|---|
| `store.py`（変更） | 定数 `N_ROWS` / `ROW_START`、`validate_label`、`rejection_reasons`、`LabelStore`（フレーム一覧は画像から）、原子的書き込み、状態 |
| `pose.py`（新規） | 姿勢 dict のヘルパ（`make_pose` / `shift_pose` / `pose_delta` / `wrap_angle`）と `PoseStore`（poses.json） |
| `frames.py`（新規） | `parse_frame_name`、`FrameIndex`（split をまたぐ時系列順、prev / next、引き継ぎ元） |
| `gnss.py`（新規） | `GnssIndex`（gnss.json、bag ごとのヨー偏り、初期姿勢、引き継ぎ） |
| `matching.py`（新規、`vectormap_match.py` を置き換え） | `observable_polylines`、`DistanceField`、`Window`、`match`、`candidates` |
| `projector.py`（新規） | 学習リポジトリのカメラ・地図を包む `Projector`（overlay / label / bev_points / bev / ピッチ）、`load_projector` |
| `yolop.py`（新規） | `LaneDetector`（マスク + ディスクキャッシュ + ロック）、`load_yolop`、`Prefetcher` |
| `app.py`（新規） | `Annotator`（フレームを開く・プレビュー・照合・確定・キュー・flags） |
| `server.py`（書き直し） | HTTP ルーティングのみ |
| `annotate.py`（変更） | CLI。`build_app` で組み立て、先読み開始、サーバ起動 |
| `exclude_skipped.py`（新規） | skip フレームを `excluded/` へ移すスクリプト |
| `scripts/bench_matching.py`（新規） | 実 OSM での照合時間の計測 |
| `static/index.html` / `style.css` / `app.js` / `overlay.js` / `bev.js`（書き直し・新規） | レイアウト A の画面 |
| 削除 | `curve.py`、`detect.py`、`infer.py`、`vectormap_match.py`、`static/geom.js`、`tests/test_vectormap_match.py` |
| `tests/conftest.py`（新規） | `learning` フィクスチャ、`mini_root`（合成データセット）、`StubProjector`、`FakeDetector`、合成 OSM |
| `tests/data/straight.osm`、`tests/data/main_params.yaml`（新規） | 合成の直線コースとカメラパラメータ |

共通のデータ形:

- **pose**: `{"x": float, "y": float, "yaw": float, "pitch": float}`（x, y は地図座標 [m]、yaw [rad]、pitch [deg]）。
- **poses.json の 1 エントリ**: pose の 4 項目 + `source`、`cost`（float | null）、`flags`（list[str]）、`init`（`{"pose": pose, "source": "propagated"|"gnss"}`）、`ts`。
- **overlay**: `{"lines": [[[u,v],...],...], "paths": {cls: [[u,v],...]}, "rows": {cls: {"xp": [48], "h_vector": [48]}}}`。
- **bev**: `{"lines": [[[x,y],...],...], "paths": {cls: [[x,y],...]}, "points": [[x,y],...]}`（車両座標、x 前方・y 左 [m]）。
- **candidate**: `{"pose", "cost", "init": "propagated"|"gnss"|"lane_left"|"lane_right", "lateral_m", "longitudinal_m", "yaw_deg", "overlay": {"lines", "paths"}}`（補正量は init 姿勢に対する値）。

---

### Task 1: store.py の整理と pose.py

**Files:**
- Modify: `store.py`
- Create: `pose.py`
- Test: `tests/test_store.py`、`tests/test_pose.py`

**Interfaces:**
- Produces: `store.N_ROWS = 48`、`store.ROW_START = 25`、`store.rejection_reasons(label: list[dict]) -> list[str]`、`store.LabelStore.frames(split) -> list[str]`（`images/<split>/*.png` の stem）、`store.LabelStore.splits()`（`images/` 配下）、`store.LabelStore.has_label(split, name) -> bool`。
- Produces: `pose.make_pose(x, y, yaw, pitch=0.0) -> dict`、`pose.wrap_angle(a) -> float`（(-π, π]）、`pose.shift_pose(pose, lateral=0.0, longitudinal=0.0, dyaw=0.0, dpitch=0.0) -> dict`（車両座標の横 + は左）、`pose.pose_delta(pose, ref) -> {"lateral_m", "longitudinal_m", "yaw_deg", "pitch_deg"}`（ref の車両座標で表す）、`pose.PoseStore(root)` の `get(split, name) -> dict | None`、`set(split, name, pose, source, cost, flags, init) -> dict`、`entries() -> dict`。

- [ ] **Step 1: 失敗するテストを書く（store）**

```python
# tests/test_store.py
def test_rejection_reasons_min_rows(): 可視 9 行の straight → ['可視行 9 < 10']
def test_rejection_reasons_edge(): 可視 20 行のうち 12 行が xp=0.0 → 1 件、'画像端' を含む
def test_rejection_reasons_ok(): 可視 23 行・中央 → []
def test_rejection_reasons_empty(): [] → ['経路なし']
def test_frames_listed_from_images(tmp_path): images/train に a.png, b.png、labels/train に a.txt だけ → frames('train') == ['a','b']、has_label('train','b') is False
```

- [ ] **Step 2: 失敗を確認** — pytest `-k "rejection or frames_listed"`。Expected: ImportError / AttributeError。

- [ ] **Step 3: 実装** — `store.py` に `N_ROWS`、`ROW_START`、`MIN_VISIBLE_ROWS`、`EDGE_RATIO`、`EDGE_MARGIN`（出典コメント: `learning/Models/data_parsing/rosbag/converter.py` の `rejection_reasons`）、`rejection_reasons`（converter と同じ判定順: 空 → `経路なし`、最大可視行 < 10、straight の端率 ≥ 0.5）。`LabelStore.splits` / `frames` を画像基準に変え、`has_label` を追加。`from tools.path_annotator.curve import ...` を消す（`curve.py` は Task 12 まで残るが、store からは参照しない）。

- [ ] **Step 4: 通過を確認** — pytest `-k "rejection or frames_listed"`。Expected: PASS。既存 `tests/test_vectormap_match.py` も壊れていないこと。

- [ ] **Step 5: 失敗するテストを書く（pose）**

```python
# tests/test_pose.py
def test_shift_pose_lateral_is_left(): yaw=0 で lateral=1 → y が +1
def test_pose_delta_roundtrip(): ref から shift_pose(lateral=0.3, longitudinal=-0.5, dyaw=radians(2)) した pose の pose_delta(pose, ref) ≈ (0.3, -0.5, 2.0, 0.0)（1e-6）
def test_wrap_angle(): wrap_angle(pi + 0.1) ≈ -pi + 0.1
def test_pose_store_roundtrip(tmp_path): set → 新しい PoseStore で get が同じ内容、'ts' を持つ
def test_pose_store_corrupt_file(tmp_path): poses.json に 'not json' → get は None、set 後に正しい JSON になる
```

- [ ] **Step 6: 失敗を確認** — pytest `-k pose`。Expected: ModuleNotFoundError。

- [ ] **Step 7: 実装** — `pose.py`。`PoseStore` は `store._atomic_write_json` を使い、`threading.Lock` でキャッシュ + 書き込みを排他（`LabelStore._update_frame_state` と同じ構造）。壊れた JSON は `{"frames": {}}` として扱う。

- [ ] **Step 8: 通過を確認** — pytest `-k pose`。Expected: PASS。

- [ ] **Step 9: Commit** — `♻️refactor: store に棄却基準と画像基準のフレーム一覧を追加し、姿勢の保存を pose.py に分離`

---

### Task 2: frames.py（時系列インデックス）

**Files:**
- Create: `frames.py`
- Test: `tests/test_frames.py`

**Interfaces:**
- Produces: `frames.MAX_INDEX_GAP = 4`、`frames.parse_frame_name(name) -> (prefix: str, num: int)`（`detect.py` から移植）、`frames.FrameIndex(keys: list[tuple[str, str]], meta: dict[str, dict])`（`meta` のキーは `"<split>/<name>"`、値は `{"bag": str, "image_index": int}`。meta に無いフレームは `parse_frame_name` の (prefix, num) で代用）。属性 `order: list[tuple[str, str]]`（(bag, image_index) 昇順、split を混ぜる）。メソッド `position(split, name) -> int`、`prev(split, name) -> tuple | None`、`next(split, name) -> tuple | None`、`predecessor(split, name) -> tuple | None`（order 上の直前で、同じ bag かつ image_index 差 ≤ 4）、`__len__`。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_order_mixes_splits(): train a(idx 0), val b(idx 2), train c(idx 4) → order == [(train,a),(val,b),(train,c)]
def test_prev_next_cross_split(): next(train,a) == (val,b); prev(val,b) == (train,a); prev(train,a) is None
def test_predecessor_gap(): idx 0 → 2 は predecessor あり、idx 2 → 10 は None（差 8）
def test_predecessor_other_bag_is_none(): 別 bag が直前でも None
def test_fallback_without_meta(): meta 無しは名前末尾の 6 桁で並ぶ
```

- [ ] **Step 2: 失敗を確認** — pytest `-k frames`。Expected: ModuleNotFoundError。
- [ ] **Step 3: 実装** — `frames.py`。
- [ ] **Step 4: 通過を確認** — pytest `-k frames`。Expected: PASS。
- [ ] **Step 5: Commit** — `✨feat: split をまたぐ時系列フレームインデックスを追加`

---

### Task 3: gnss.py（初期姿勢と引き継ぎ）

**Files:**
- Create: `gnss.py`
- Test: `tests/test_gnss.py`

**Interfaces:**
- Consumes: `pose.make_pose`、`pose.wrap_angle`。
- Produces: `gnss.MOVING_SPEED = 0.5`、`gnss.GnssIndex(path)`。属性 `frames: dict[str, dict]`（gnss.json の `frames`）、`bias: dict[str, float]`（bag → `median(yaw_motion_rad - yaw_imu_rad)` を wrap して移動中フレームで取る。移動中が無ければ 0.0）。メソッド `has(key) -> bool`、`meta() -> dict[str, {"bag", "image_index"}]`、`initial_pose(key) -> pose`（`map.x`, `map.y`, `wrap(yaw_imu_rad + bias[bag])`, pitch 0）、`propagate(prev_key, prev_pose, key) -> pose`（`x,y += map 位置の差`、`yaw += wrap(yaw_imu[key] - yaw_imu[prev])`、pitch はコピー）。
- 移動中の判定: 同じ bag を `image_index` 順に並べ、隣接フレームとの map 位置差 / `stamp` 差 > 0.5 m/s。

- [ ] **Step 1: 失敗するテストを書く** — 合成 gnss.json を `tmp_path` に書くヘルパ `write_gnss(tmp_path, frames)` を test 内に置く。

```python
def test_bias_from_moving_frames(): 5 フレーム、x を 1 m/0.1 s で進め、yaw_motion = yaw_imu + 0.2 → bias['bag'] ≈ 0.2
def test_bias_zero_when_stationary(): 位置が同じ → 0.0
def test_initial_pose_applies_bias(): yaw_imu=0.1, bias=0.2 → yaw ≈ 0.3
def test_propagate_adds_deltas(): prev_pose (10, 0, 0.5), gnss 位置差 (1, 0.1), yaw_imu 差 0.05 → (11, 0.1, 0.55)、pitch 引き継ぎ
def test_meta_has_bag_and_index(): meta()['train/f0'] == {'bag': 'bag', 'image_index': 0}
```

- [ ] **Step 2: 失敗を確認** — pytest `-k gnss`。Expected: ModuleNotFoundError。
- [ ] **Step 3: 実装** — `gnss.py`。
- [ ] **Step 4: 通過を確認** — pytest `-k gnss`。Expected: PASS。
- [ ] **Step 5: Commit** — `✨feat: gnss.json から偏り補正済みの初期姿勢と引き継ぎ姿勢を作る`

---

### Task 4: matching.py（距離場による照合と候補）

**Files:**
- Create: `matching.py`、`scripts/bench_matching.py`
- Test: `tests/test_matching.py`

**Interfaces:**
- Consumes: `pose.make_pose` / `shift_pose` / `pose_delta`。
- Produces:
  - `matching.TRUNC = 0.5`、`matching.CELL = 0.05`、`matching.LANE_WIDTH = 3.3`、`matching.DEDUPE_POSITION = 0.4`、`matching.DEDUPE_YAW_DEG = 2.0`、`matching.MAX_CANDIDATES = 4`。
  - `matching.Window(lateral, lateral_step, longitudinal, longitudinal_step, yaw_deg, yaw_step_deg)`（frozen dataclass）、`matching.NARROW`、`matching.WIDE`（値は Global Constraints）。
  - `matching.observable_polylines(osm_path) -> list[np.ndarray]`（`type=line_thin` かつ `is_observable=true` の way を (n,2) で返す。`vectormap_match.observable_lines` の折れ線版）。
  - `matching.densify(polyline, step) -> np.ndarray`。
  - `matching.DistanceField(polylines, cell=CELL, trunc=TRUNC, margin=5.0)`。属性 `origin: (x0, y0)`、`grid: np.ndarray float32`。メソッド `distance(xy: np.ndarray[..., 2]) -> np.ndarray[...]`（双一次補間、格子外は trunc。実装は折れ線を `cell/2` で densify して格子に立て、`scipy.ndimage.distance_transform_edt` × cell を trunc で clip、照会は `scipy.ndimage.map_coordinates(order=1, mode='constant', cval=trunc)`）。
  - `matching.cost(field, points, pose) -> float`（points は車両座標 (N,2)、地図座標へ変換して平均）。
  - `matching.match(field, points, pose, window) -> dict`：`{"pose", "cost", "cost_before", "lateral_m", "longitudinal_m", "yaw_deg"}`（補正量は入力 pose に対する `pose_delta`）。探索は格子（ヨーごとにループし、横×前後の全オフセットをまとめて評価）→ 最良の周りを細密探索（横 ±0.08 / 0.02、前後 ±0.2 / 0.1、ヨー ±0.8° / 0.2°。現行 `bev_match` と同じ）。ピッチは変えない。
  - `matching.candidates(field, points, inits: list[tuple[dict, str]], window) -> list[dict]`：各 (pose, init 名) を `match` し、結果に `"init"` を付け、位置 0.4 m 以内かつヨー 2° 以内の重複を（先に来た方を残して）間引き、コスト昇順で最大 4 件。

- [ ] **Step 1: 失敗するテストを書く** — 合成の線: `xs = arange(-5, 40, 0.2)` で y ∈ {-1.65, 1.65, 4.95}（`tests/test_vectormap_match.py::test_bev_match_recovers_lateral_and_yaw_offset` の作り方を流用）。

```python
def test_distance_matches_kdtree_within_3cm(): 乱数点 500 個について |field.distance - cKDTree 距離（0.5 で clip）| の最大 ≤ 0.03
def test_distance_outside_grid_is_trunc(): (1000, 1000) → 0.5、例外なし
def test_match_wide_recovers_offset(): 真値 横 +0.4 m・ヨー +3°、初期 (10,0,0) → WIDE で |y-0.4| < 0.08、|yaw-3°| < 0.6、cost < cost_before
def test_match_narrow_recovers_small_offset(): 真値 横 +0.3・ヨー +2° → NARROW で |y-0.3| < 0.06、|yaw-2°| < 0.5
def test_candidates_dedupe_and_limit(): 同じ初期姿勢を 6 回 → 1 件。±3.3 m の初期値を含む 4 起点 → 最大 4 件、コスト昇順、各要素に 'init' と 'overlay' 以外のキー（pose/cost/lateral_m/longitudinal_m/yaw_deg）がある
```

- [ ] **Step 2: 失敗を確認** — pytest `-k matching`。Expected: ModuleNotFoundError。
- [ ] **Step 3: 実装** — `matching.py`。格子評価は float32 で、ヨー 1 値あたり (n_lat×n_lon, N, 2) を作る。
- [ ] **Step 4: 通過を確認** — pytest `-k matching`。Expected: PASS。
- [ ] **Step 5: 計測スクリプト** — `scripts/bench_matching.py --osm <path> [--points 800]`：実 OSM で `DistanceField` 構築時間、WIDE 1 回、NARROW 1 回、`candidates` 4 起点の合計時間を print。実行: `.venv/bin/python tools/path_annotator/scripts/bench_matching.py --osm /home/kyo/formula_ws/src/map/vectormap_server/config/aiformula_course.osm`。Expected: WIDE < 0.3 s、候補合計 < 1 s。超える場合は格子評価のチャンク分割や float32 化を見直す（予算は spec の要件）。
- [ ] **Step 6: Commit** — `✨feat: 距離場による高速な俯瞰照合と候補生成を追加`

---

### Task 5: projector.py（投影・ラベル・俯瞰・ピッチ）

**Files:**
- Create: `projector.py`、`tests/data/straight.osm`、`tests/data/main_params.yaml`、`tests/conftest.py`
- Test: `tests/test_projector.py`

**Interfaces:**
- Consumes: 学習リポジトリ `projection.load_camera(params_path, "nHD", 384, 640)`、`projection.CameraModel`（`project(points) -> (pixels, visible)`、`rotation`、`translation`、`fx, fy, cx, cy, pad_top, pad_left`）、`projection.row_anchor_columns(pixels, visible, num_rows, height, width)`、`vectormap.load_vector_map(osm)`、`vectormap.trace_paths(vector_map, position, heading, horizon)`。`matching.observable_polylines`、`matching.densify`。
- Produces:
  - 定数 `H = 384`、`W = 640`、`NUM_ROWS = 48`、`HORIZON = 20.0`、`IMAGE_X_RANGE = (0.8, 45.0)`、`IMAGE_MAX_LAT = 25.0`、`BEV_ROW_START = 200`、`BEV_X_RANGE = (1.6, 7.0)`、`BEV_MAX_LAT = 8.0`、`BEV_MAX_POINTS = 800`、`BEV_X_WINDOW = (-4.0, 28.0)`、`BEV_Y_WINDOW = 13.0`、`PITCH_QUANTUM = 0.1`。
  - `projector.to_vehicle(points_map, pose) -> np.ndarray (N,2)`。
  - `projector.tilted_camera(camera, pitch_deg)`：同じクラスの新しい CameraModel。rotation に車両左右軸まわりの回転を合成する。符号はテスト（+1° で水平線が上へ約 4.4 px）で決める。
  - `projector.Projector(camera, vector_map, polylines, trace_paths, row_anchor_columns)`。メソッド:
    - `camera_for(pitch) -> CameraModel`（`round(pitch / 0.1) * 0.1` で丸めてキャッシュ）。
    - `paths(pose) -> dict[str, np.ndarray]`（`trace_paths(vector_map, (x,y), yaw, HORIZON)`）。
    - `project_polyline(poly_map, pose) -> list[list[list[float]]]`（0.2 m で densify → 車両座標 → 前方 0.8〜45 m・横 ±25 m かつ visible の連続区間ごとに画素列 [[u,v],...] に分ける）。
    - `overlay(pose) -> overlay`（`lines` は全観測線、`paths` は class ごと、`rows` は class ごとの 48 行。`rows` は `label(pose)` と同じ値）。
    - `label(pose) -> list[dict]`（class ごとに 0.2 m 再サンプリング → 車両座標 → 投影 → `row_anchor_columns`。有効行が無い class は出さない。straight を先頭に）。
    - `bev_points(mask, pitch=0.0, x_range=BEV_X_RANGE) -> np.ndarray (N,2)`（現行 `VectormapMatcher.bev_points` と同じ手順。行→距離・中心列・px/m の表は `camera_for(pitch)` で作り、ピッチごとにキャッシュ）。
    - `bev(pose, points) -> bev`（観測線と経路を車両座標に変換し、前方 -4〜28 m・横 ±13 m の区間に切る。`points` はそのまま list に）。
  - `projector.load_projector(learning_repo, params_path, osm_path) -> Projector`（`sys.path.insert(0, learning_repo)` して import。`matching.observable_polylines(osm_path)` も渡す）。
- `tests/conftest.py` に `learning` フィクスチャ（env `AIF_LEARNING_REPO`、既定 `/home/kyo/formula_ws/src/learning`。`Models/data_utils/vision_planner/projection.py` が無ければ `pytest.skip`。`sys.path` に入れてモジュール 2 つを返す）。
- `tests/data/straight.osm`: x 軸に沿う直線コース。lanelet A（centerline y=0、x 0→30）と B（x 30→60）、`lane_connection` A→B（`turn_direction=straight`）、観測線 `line_thin` `is_observable=true` を y=±1.65 と y=4.95 に x 0→60 で置く。ノードは `x` / `y` 属性（実 OSM と同じ）。
- `tests/data/main_params.yaml`: `/**: ros__parameters: camera: {position: {x: -0.008, y: 0.0325, z: 0.5547}, orientation: {roll: -89.801314, pitch: 0.497098, yaw: -89.640661}}`（実機の値。`load_camera` が読む形）。

- [ ] **Step 1: 失敗するテストを書く**（全テストが `learning` フィクスチャを使う）

```python
def test_overlay_lines_inside_image(projector): pose (5,0,0) → overlay['lines'] が空でなく、全点が 0<=u<=639, 0<=v<=383
def test_label_straight_on_centerline(projector): pose (5,0,0) → label に straight があり、有効行 ≥ 15、有効行の xp が 0.4〜0.6
def test_label_shifts_with_lateral_pose(projector): pose (5, +0.5, 0)（左へ）→ straight の有効行の xp 平均が pose (5,0,0) より大きい（中心線が右へ写る）
def test_tilt_moves_horizon_up(learning): 点 (100,0,0) の v が camera_for(1.0) で camera_for(0.0) より 4.0〜5.0 px 小さい
def test_bev_points_synthetic(projector): mask の (v=264,u=168) と (264,528) の 2 画素 → x ≈ 2.0±0.1、y ≈ +1.3±0.1 と -1.5±0.1（BEV_X_RANGE を (0.5,10) に広げて呼ぶ引数 x_range を持たせる）
def test_bev_clips_window(projector): bev(pose,(5,0,0), points) の lines 全点が x ∈ [-4,28]、|y| ≤ 13
```

- [ ] **Step 2: 失敗を確認** — pytest `-k projector`。Expected: ModuleNotFoundError（学習リポジトリが無い環境では skip）。
- [ ] **Step 3: 実装** — `projector.py` と conftest、データ 2 ファイル。
- [ ] **Step 4: 通過を確認** — pytest `-k projector`。Expected: PASS（6 件）。
- [ ] **Step 5: Commit** — `✨feat: 姿勢から画像・俯瞰・48 行ラベルを作る Projector を追加`

---

### Task 6: yolop.py（マスク、キャッシュ、先読み）

**Files:**
- Create: `yolop.py`
- Test: `tests/test_yolop.py`

**Interfaces:**
- Produces:
  - `yolop.LaneDetector(model, cache_dir, threshold=0.5)`：`model` は `callable(image_bgr: np.ndarray[H,W,3]) -> np.ndarray[H,W]`（0〜1 の確率）。メソッド `mask(image_path, key) -> np.ndarray[H,W] uint8 {0,1}`（`cache_dir/<key>.png` に 0/255 で保存。あれば読んで返す。読めない PNG は再計算して上書き）、`cached(key) -> bool`。モデル呼び出しは `threading.Lock` で直列化し、同じ key の計算が重ならないよう key ごとの進行中集合を持つ（待ってから読む）。
  - `yolop.load_yolop(weights, threads=None) -> callable`（遅延 import。`torch.jit.load(map_location="cpu")`、`torch.set_num_threads(threads or max(1, os.cpu_count() // 2))`。入力は BGR→RGB、/255、`[None]`、出力 `model(x)[2][0, 0]` を sigmoid 済みとして numpy に）。
  - `yolop.Prefetcher(detector, items: list[tuple[str, str]])`（(key, image_path) の時系列順）。`start()`（daemon スレッド）、`set_cursor(key)`（次に処理する位置をその key に移す。未キャッシュの key を順に処理）、`progress() -> (done, total)`（done はキャッシュ済み数）、`stop()`。

- [ ] **Step 1: 失敗するテストを書く** — `fake_model(calls)` は呼ばれた回数を数え、(384,640) で左右 2 本の縦線を 1.0 にした配列を返す。

```python
def test_mask_cached_on_disk(tmp_path): 2 回 mask → model 呼び出し 1 回、cache png が存在、値は {0,1}
def test_corrupt_cache_is_recomputed(tmp_path): cache png に b'garbage' → mask は正常に返り、model 呼び出し 1 回、png が上書きされる
def test_concurrent_same_key_computes_once(tmp_path): 2 スレッドで同じ key → model 呼び出し 1 回
def test_prefetcher_from_cursor(tmp_path): items k0..k4、set_cursor('k3') → 完了後 progress()==(5,5) で、計算順の先頭 2 件が k3, k4
```

- [ ] **Step 2: 失敗を確認** — pytest `-k yolop`。Expected: ModuleNotFoundError。
- [ ] **Step 3: 実装** — `yolop.py`。
- [ ] **Step 4: 通過を確認** — pytest `-k yolop`。Expected: PASS。
- [ ] **Step 5: Commit** — `✨feat: YOLOP マスクのディスクキャッシュと時系列先読みを追加`

---

### Task 7: app.py（Annotator）

**Files:**
- Create: `app.py`
- Modify: `tests/conftest.py`（`StubProjector`、`FakeDetector`、`mini_root`、`annotator` フィクスチャ）
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `store.LabelStore` / `validate_label` / `rejection_reasons`、`pose.*`、`frames.FrameIndex`、`gnss.GnssIndex`、`matching.DistanceField` / `match` / `candidates` / `NARROW` / `WIDE`、`projector.Projector`（`overlay` / `label` / `bev_points` / `bev`）、`yolop.LaneDetector`（`mask`）、`yolop.Prefetcher`（`set_cursor` / `progress`、省略可）。
- Produces: 定数 `COST_THRESHOLD = 0.25`、`JUMP_LATERAL = 0.3`、`JUMP_YAW_DEG = 2.0`、`PREFER_PROPAGATED_MARGIN = 0.03`、`MIN_POINTS = 30`。`app.Annotator(store, poses, frames, gnss, projector, field, detector, cost_threshold=COST_THRESHOLD, prefetcher=None)` のメソッド:
  - `queue(order="time", status=None, flag=None) -> list[dict]`：`{"split","name","index","status","flags","cost","source"}`。`flag="any"` は flags が空でないもの、それ以外の文字列はその flag を含むもの。
  - `open_frame(split, name) -> dict`：`{"split","name","index","total","status","pose","source","cost","flags","init","overlay","bev","label","candidates","prev","next","cache"}`。poses.json に無ければ初期化: `frames.predecessor` が確定済み（status done）なら `gnss.propagate` + NARROW、無ければ `gnss.initial_pose` + WIDE。白線点 < 30 なら照合せず `few_points`。候補は `[(init, "propagated"|"gnss"), (gnss 初期姿勢, "gnss"), (init を横 +3.3, "lane_left"), (init を横 -3.3, "lane_right")]`（引き継ぎが無いときは gnss 起点のみ 3 つ）。既定は先頭候補、ただし引き継ぎ由来の候補が存在しコスト最小との差 < 0.03 ならそれ。保存する `source` は引き継ぎなら `"propagated"`、GNSS なら `"auto"`、照合しなかったときは `"gnss"`。`gnss.has(key)` が偽なら `ValueError("<key> の GNSS がありません")`。開いたら `prefetcher.set_cursor(key)`。
  - `preview(split, name, pose) -> dict`：`{"overlay","bev","cost","label","flags"}`。保存しない。
  - `match(split, name, pose, window: str) -> dict`：`{"pose","cost","candidates"}`（`"narrow"` / `"wide"`。候補起点は open_frame と同じ構成で init の代わりに与えた pose）。
  - `set_pose(split, name, pose, source) -> dict`（flags と cost を計算して保存）。
  - `confirm(split, name, pose, source) -> dict`：`label = projector.label(pose)` → `validate_label` → `rejection_reasons` が空でなければ `ValueError("; ".join(reasons))` → `store.save` → `poses.set` → status `done` → `{"ok": True, "next": {"split","name"} | None}`。
  - `set_status(split, name, status)`。
  - 内部: `_flags(cost, pose, init_pose, label, n_points) -> list[str]`（`cost` / `jump` / `no_path` / `few_points` / `pitch`）。
- conftest:
  - `StubProjector(valid_rows=23, n_points=None)`：`overlay(pose)` は固定の線 1 本と straight の経路、`rows`；`label(pose)` は straight 1 本、行 25 から `valid_rows` 行が有効、`xp = clip(0.5 - 0.1 * pose["y"], 0, 1)`；`bev_points(mask, pitch=0.0, x_range=None)` は mask を無視し x ∈ [1.6,7] 0.2 刻み × y ∈ {-1.65, +1.65} の点（`n_points` が整数なら先頭 n 点だけ）；`bev(pose, points)` は `{"lines": [], "paths": {}, "points": points.tolist()}`。
  - `FakeDetector`：`mask(image_path, key)` は (384,640) のゼロ配列に 2 画素、`cached` は True。
  - `mini_root(tmp_path)`：`images/train/f000000.png, f000002.png`、`images/val/f000004.png`（640×384 の黒 PNG）、`labels/train/f000000.txt`（既存ラベル 1 つ）、`gnss.json`（1 bag、x = 5, 6, 7、y = 0.3（横ずれ）、yaw_imu = 0、yaw_motion = 0、stamp 0.1 刻み）。
  - `make_annotator(mini_root)`：`make(projector=None, detector=None, cost_threshold=0.25) -> Annotator` を返すファクトリ。合成線 y ∈ {-1.65, 1.65}（x -5〜80、2 本だけ。3 本目を置くとレーン 1 本ぶんずれた解もコスト 0 になりテストが不安定になる）の `DistanceField`、`StubProjector()`、`FakeDetector()` が既定。
  - フィクスチャ `annotator = make()`、`annotator_few_rows = make(projector=StubProjector(valid_rows=5))`、`annotator_no_points = make(projector=StubProjector(n_points=10))`。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_open_first_frame_uses_gnss_and_wide(annotator): open_frame('train','f000000') → init.source=='gnss'、source=='auto'、|pose.y| < 0.1（0.3 が補正される）、候補 ≤ 4、cache キーあり
def test_open_next_after_confirm_propagates(annotator): confirm f000000 → open_frame('train','f000002') の init.source=='propagated'、source=='propagated'
def test_confirm_writes_label_and_next(annotator, mini_root): confirm('train','f000000', pose, 'auto') → labels/train/f000000.txt が validate_label を通る、next == {'split':'train','name':'f000002'}、status 'done'
def test_confirm_rejects_few_rows(annotator_few_rows): StubProjector(valid_rows=5) → ValueError に '可視行' を含む、ラベル未変更
def test_flags_cost_and_jump(annotator, make_annotator): preview に init から横 0.5 m ずらした pose → flags に 'jump'；make_annotator(cost_threshold=0.0) では preview の flags に 'cost'
def test_queue_flag_filter(annotator): set_pose で flags 付き 1 件 → queue(flag='any') が 1 件、queue(flag='jump') も 1 件
def test_missing_gnss_is_value_error(annotator, mini_root): images/val/f000099.png を追加（gnss 無し）→ open_frame で ValueError 'GNSS がありません'、その後 open_frame('train','f000000') は成功
def test_few_points_skips_matching(annotator_no_points): bev_points が 10 点 → source=='gnss'、flags に 'few_points'、候補 []
```

- [ ] **Step 2: 失敗を確認** — pytest `-k app`。Expected: ModuleNotFoundError。
- [ ] **Step 3: 実装** — `app.py` と conftest。候補の `overlay` は `projector.overlay(cand_pose)` から `rows` を除いたもの。
- [ ] **Step 4: 通過を確認** — pytest `-k app`。Expected: PASS（8 件）。
- [ ] **Step 5: Commit** — `✨feat: 姿勢の初期化・照合・候補・確定をまとめる Annotator を追加`

---

### Task 8: server.py（HTTP ルーティング）

**Files:**
- Modify: `server.py`（全面書き直し。`make_handler` / `make_server` の構造と `_send` / `_json` / `_error` / `_body` / `_drain` ヘルパは現行から流用）
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `app.Annotator`（メソッドと、属性 `store`（`image_path`）・`detector`（`mask`。`/api/mask` 用）・`frames`）。
- Produces: `server.make_server(annotator, port=8765) -> ThreadingHTTPServer`（`127.0.0.1`）。ルート:
  - `GET /` → `static/index.html`、`GET /static/<file>`。
  - `GET /api/queue?order=time&status=&flag=&offset=&limit=` → `{"items": [...], "total": n}`（offset/limit 省略時は全件）。
  - `GET /api/frame/<split>/<name>` → `open_frame`。
  - `GET /api/image/<split>/<name>` → PNG。`GET /api/mask/<split>/<name>` → マスク PNG（0/255）。
  - `POST /api/preview` `{split,name,pose}`、`POST /api/match` `{split,name,pose,window}`、`POST /api/pose` `{split,name,pose,source}`、`POST /api/confirm` `{split,name,pose,source}`、`POST /api/status` `{split,name,status}`。
  - エラー: `ValueError` → 400 `{"error": str}`、`FileNotFoundError` → 404、`KeyError`/`TypeError` → 400 `bad request`、他 → 500。

- [ ] **Step 1: 失敗するテストを書く** — `make_server(annotator, 0)` をスレッドで `serve_forever` し、`urllib.request` で叩くヘルパ `call(method, path, body=None) -> (status, json)`。

```python
def test_queue_and_frame(server): GET /api/queue → total 3、items[0].name=='f000000'；GET /api/frame/train/f000000 → 200、'overlay' と 'candidates' を含む
def test_preview_and_confirm(server, mini_root): POST /api/preview → 200 with cost；POST /api/confirm → 200 {ok, next}、ラベルファイルあり
def test_confirm_rejection_is_400(server_few_rows): （`annotator_few_rows` で立てたサーバ）POST /api/confirm → 400、error に '可視行'
def test_mask_png(server): GET /api/mask/train/f000000 → 200、Content-Type image/png、先頭 8 バイトが PNG シグネチャ
def test_unknown_frame_is_400(server): GET /api/frame/val/f000099（gnss 無し）→ 400
```

- [ ] **Step 2: 失敗を確認** — pytest `-k server`。Expected: 旧 `make_server(root, ...)` のシグネチャ不一致で TypeError。
- [ ] **Step 3: 実装** — `server.py` を書き直す。
- [ ] **Step 4: 通過を確認** — pytest `-k server`。Expected: PASS（5 件）。
- [ ] **Step 5: Commit** — `♻️refactor: HTTP 層を姿勢ベースの API に書き直す`

---

### Task 9: annotate.py（CLI と組み立て）

**Files:**
- Modify: `annotate.py`
- Test: `tests/test_annotate.py`

**Interfaces:**
- Consumes: `projector.load_projector`、`matching.DistanceField` / `observable_polylines`、`yolop.load_yolop` / `LaneDetector` / `Prefetcher`、`gnss.GnssIndex`、`frames.FrameIndex`、`pose.PoseStore`、`store.LabelStore`、`app.Annotator`、`server.make_server`。
- Produces: `annotate.build_app(root, vectormap, yolop, params, learning_repo, cost_threshold=0.25, cache_dir=None) -> tuple[Annotator, Prefetcher]`（`cache_dir` 既定 `root/cache/yolop`。`LabelStore.ensure_backup()` を呼ぶ。`FrameIndex` は `store.splits()` × `store.frames()` の全フレームと `gnss.meta()` から作る。Prefetcher の items は `frames.order` の順）。`annotate.main(argv=None) -> int`：引数 `--root`（必須）、`--vectormap` / `--yolop` / `--params` / `--learning-repo`（4 つとも必須。無ければ `ap.error`）、`--port`（8765）、`--no-browser`、`--cost-threshold`（0.25）。`--weights` / `--branch-threshold` は削除。起動時に `prefetcher.start()`、`srv.serve_forever()`、終了時 `prefetcher.stop()`。ポート使用中の案内は現行どおり。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_main_requires_vectormap_args(tmp_path, capsys): main(['--root', str(tmp_path)]) → SystemExit(2)、stderr に '--vectormap'
def test_build_app_wires_frames_in_time_order(mini_root, monkeypatch): load_projector と load_yolop を monkeypatch でスタブ化 → build_app の Annotator.frames.order が [(train,f000000),(train,f000002),(val,f000004)]、Prefetcher の items が同じ順
```

- [ ] **Step 2: 失敗を確認** — pytest `-k annotate`。Expected: AttributeError（build_app 無し）。
- [ ] **Step 3: 実装** — `annotate.py`。
- [ ] **Step 4: 通過を確認** — pytest `-k annotate`。Expected: PASS。
- [ ] **Step 5: 実機で起動確認** — スクラッチのミニデータ（`/tmp/claude-1000/-home-kyo-aif-dataset/2f8bfe29-a02b-4b06-9c5b-b0ab850d33f4/scratchpad/mini`、無ければ `20261012_data` から画像 5 枚と `gnss.json` をコピーして作る）で

      cd /home/kyo/aif_dataset && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/path_annotator/annotate.py --root <mini> --port 8798 --no-browser \
        --learning-repo /home/kyo/formula_ws/src/learning \
        --vectormap /home/kyo/formula_ws/src/map/vectormap_server/config/aiformula_course.osm \
        --yolop /home/kyo/formula_ws/src/perception/road_detector/weights/yolopv2.pt \
        --params /home/kyo/formula_ws/src/main_executor/config/main_params.yaml

  別シェルで `curl -s http://127.0.0.1:8798/api/frame/train/perception_planning_20261012_164813_000000 | head -c 300`。Expected: `"pose"` と `"candidates"` を含む JSON。`<mini>/cache/yolop/train/*.png` ができる。
- [ ] **Step 6: Commit** — `✨feat: CLI を地図投影アノテータの構成に合わせて書き直す`

---

### Task 10: 画面（画像・重ね表示・スライダー・流れ）

**Files:**
- Modify: `static/index.html`、`static/style.css`、`static/app.js`（全面書き直し）
- Create: `static/overlay.js`
- Delete: `static/geom.js`

**Interfaces:**
- Consumes: Task 8 の API。
- Produces:
  - `index.html` の要素 id: `topbar`（子 `tb-name`、`tb-index`、`tb-order`、`tb-cost`、`tb-flags`、`tb-cache`）、`canvas`（画像、backing サイズは表示サイズに合わせる）、`bev`（canvas 300×360、Task 11 で使う）、`hud-correction`、`hud-cost`、`hud-rows`、`hud-source`、`opacity-map`（`<input type=range min=0 max=100>` 既定 50）、`opacity-mask`（既定 35）、`candidates`、`keys`、`queue-list`（右列の下、スクロール）、`msg`。スクリプト読み込み順 `overlay.js`、`bev.js`、`app.js`。
  - `overlay.js`：`window.Overlay = { draw(ctx, {img, maskImg, overlay, scale, opts}) }`。`opts = {showMask, showLines, mapAlpha, maskAlpha, hideAll}`。描画順: 画像 → マスク（`maskImg` をオフスクリーン canvas で `#00dc00` に着色し `maskAlpha` で合成）→ `mapAlpha` で地図線（`#00c8ff`、2 px）→ 経路（class 色、3 px）→ 行の点（半径 3）。`hideAll` なら画像のみ。座標は `scale` 倍、線幅は固定。
  - `app.js` の状態 `S = {order:'time', flagOnly:false, items:[], idx:-1, frame:null, pose:null, undo:[], redo:[], opts:{...}, pending:false, latest:null}`。API ラッパは現行 `api.get/send`（`r.ok` 判定付き）を流用。
  - ふるまい:
    - 起動: `GET /api/queue?order=time` → 最初の `status!=='done'` の項目を開く（無ければ先頭）。
    - フレーム表示: `GET /api/frame` → 画像と `/api/mask` を読み込んで描画、HUD・上部バー・候補（Task 11）・キュー更新。`S.pose = frame.pose`、undo/redo を空に。
    - プレビュー: `S.pose` が変わるたび `POST /api/preview`。応答待ち中は `S.latest` に最新の pose を保持し、応答後に差があれば再送（現行 mousemove の `pending` 方式）。応答の overlay / bev / cost / flags で描画と HUD を更新。
    - キー: `Enter`（`POST /api/confirm` → 成功なら `next` を開く。400 なら `msg` に理由）、`←`/`→`（`POST /api/pose` してから prev/next）、`h`/`l`（横 ∓0.05、Shift ×5）、`j`/`k`（前後 ∓0.10）、`u`/`i`（ヨー ∓0.5°）、`,`/`.`（ピッチ ∓0.2°）、`Esc`（`frame.init.pose` に戻す）、`z`/`Shift+z`（undo/redo）、`g`/`m`（表示切替）、`[`/`]`（地図不透明度 ∓10%）、`o`（keydown で hideAll、keyup で戻す）、`s`（`POST /api/status skip` → next）、`f`（flagOnly 切替 → `GET /api/queue?flag=any` で一覧を入れ替え）。修飾キー付きは無視（現行と同じ理由）。横・前後・ヨーの加算は車両座標で行うため、JS 側に `shiftPose(pose, lateral, longitudinal, dyawDeg, dpitch)` を持つ（pose.py の `shift_pose` と同じ式）。
    - スライダー: `input` イベントで `opts.mapAlpha` / `opts.maskAlpha` を更新して再描画。値は `localStorage`（`try/catch`）に保存・復元。
    - 上部バー: コストはしきい値（`frame.flags` に `cost` が無ければ良好）で緑 / 橙。`tb-cache` は `frame.cache.done/total`。
    - HUD: `hud-correction` は `frame.init.pose` に対する差（JS で `poseDelta` を計算: pose.py の `pose_delta` と同じ式）、`hud-cost` は `init.cost → cost`、`hud-rows` は `label` の class ごとの有効行数、`hud-source`。
    - キュー一覧: 項目をクリックで開く。`done` は薄く、flags ありは橙の印。

- [ ] **Step 1: index.html / style.css を書く**（レイアウト A: 上部バー、左に画像、右列に BEV・HUD・スライダー・キュー、下に候補と `keys`）。
- [ ] **Step 2: overlay.js を書く**。
- [ ] **Step 3: app.js を書く**（候補と BEV の描画呼び出しは Task 11 の `Bev` / 候補描画を呼ぶ形にし、未定義なら何もしない）。
- [ ] **Step 4: 手動確認** — Task 9 Step 5 と同じ起動で `http://127.0.0.1:8798/` を開く。確認: 画像に半透明の地図線・経路・行の点が出る；スライダーで薄くなる；`g` `m` `o` が効く；`h` を押すと経路が動き HUD の横が -0.05 になる；`Enter` で次のフレームへ進み `labels/` にファイルが書かれる；`Esc` で戻る；`z` で戻る；リロードで不透明度が保たれる。
- [ ] **Step 5: Commit** — `✨feat: 地図投影を半透明で重ねる画面と姿勢操作の流れを追加`

---

### Task 11: 俯瞰パネルと候補

**Files:**
- Create: `static/bev.js`
- Modify: `static/app.js`（候補描画、BEV 連携、`r` / `1`〜`4` / `+` `-`）

**Interfaces:**
- Produces: `window.Bev = { create(canvas, {onDrag(dLateral, dLongitudinal), onWheel(dYawDeg)}) -> {draw(bev, paths), setPpm(ppm), ppm} }`。車両は `(w/2, h-40)` の白三角、前方が上、`ppm` 既定 12（8〜24）。地図線 `#00c8ff` 2 px、経路 class 色 2 px、点群 `#00dc00` 半径 2、5 m 格子の薄い線。
  - ドラッグ: `mousemove` の差分 `(dx, dy)` [px] を `onDrag(dx / ppm, -dy / ppm)` で渡す（右へ d px → 横 +d/ppm、上へ d px → 前後 -d/ppm）。`mouseup` で最後の差分を確定。
  - ホイール: `deltaY < 0` → `onWheel(+0.5)`、`> 0` → `onWheel(-0.5)`。`preventDefault`。
- `app.js`:
  - 候補: `frame.candidates`（または `/api/match` の結果）を `#candidates` に最大 4 枚。各枚は 200×120 の canvas に画像を縮小して描き、`candidate.overlay` の線と経路を `Overlay.draw`（`rows` 無し、`scale = 200/640`）で重ねる。角に `"{i}: {cost} m"` と横・ヨー。クリックまたは `1`〜`4` で `S.pose = candidate.pose`（undo に積む）→ プレビュー。選択中は枠を強調。
  - `r`: `POST /api/match {pose: S.pose, window: 'wide'}` → `S.pose = result.pose`、候補を差し替え。
  - BEV: `frame.bev` / プレビューの `bev` を `Bev.draw`。`onDrag` は `shiftPose` で `S.pose` を更新しプレビュー（ドラッグ開始時に undo に積む）。`onWheel` も同様。`+`/`-` で `setPpm`。

- [ ] **Step 1: bev.js を書く**。
- [ ] **Step 2: app.js に候補・BEV・`r`・`1`〜`4`・`+`/`-` を足す**。
- [ ] **Step 3: 手動確認** — 起動して確認: 俯瞰に地図線・経路・緑の点群・車両が出る；ドラッグで地図が動き画像の重ねも動く；ホイールで回る；候補 4 枚にコストが出て、クリックで姿勢が切り替わる；`r` で候補が作り直される；`+`/`-` で縮尺が変わる。
- [ ] **Step 4: Commit** — `✨feat: 俯瞰パネルのドラッグ操作と候補選択を追加`

---

### Task 12: exclude_skipped.py

**Files:**
- Create: `exclude_skipped.py`
- Test: `tests/test_exclude_skipped.py`

**Interfaces:**
- Produces: `exclude_skipped.main(argv=None) -> int`：`--root`（必須）、`--dry-run`。`annotation_state.json` の `frames` で `status == "skip"` の `<split>/<name>` について `images/<split>/<name>.png` と `labels/<split>/<name>.txt`（あれば）を `excluded/<split>/` へ `os.replace` で移し、移した件数を print。`--dry-run` は一覧だけ print。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_moves_skipped_frames(tmp_path): skip 1 件・done 1 件 → skip の画像とラベルが excluded/train/ にあり、done は残る、戻り値 0
def test_dry_run_moves_nothing(tmp_path, capsys): --dry-run → ファイル不変、stdout に名前
```

- [ ] **Step 2: 失敗を確認** — pytest `-k exclude`。Expected: ModuleNotFoundError。
- [ ] **Step 3: 実装**。
- [ ] **Step 4: 通過を確認** — pytest `-k exclude`。Expected: PASS。
- [ ] **Step 5: Commit** — `✨feat: skip したフレームを excluded/ へ移すスクリプトを追加`

---

### Task 13: 旧コードの削除と README

**Files:**
- Delete: `curve.py`、`detect.py`、`infer.py`、`vectormap_match.py`、`static/geom.js`、`tests/test_vectormap_match.py`
- Modify: `README.md`（全面書き直し）、`.gitignore`（新規: `__pycache__/`、`*.pyc`）。コミット済みの `__pycache__/*.pyc` は `git rm --cached` で追跡から外す。

- [ ] **Step 1: 参照が残っていないことを確認** — `grep -rn "curve\|detect\.\|infer\|vectormap_match\|geom\.js\|Geom\." --include=*.py --include=*.js --include=*.html /home/kyo/aif_dataset/tools/path_annotator | grep -v docs/`。Expected: 出力なし（docs 以外）。残っていれば直す。
- [ ] **Step 2: 削除と .gitignore** — `git rm` で 6 ファイルを消し、`git rm -r --cached __pycache__`、`.gitignore` を追加。
- [ ] **Step 3: README を書く** — 目的（姿勢を直す方式）、起動コマンド（Global Constraints の形。重みの実体パスを明記）、画面の説明（レイアウト A、色、スライダー）、キー一覧（Task 10 / 11 の表）、進め方 W2、要確認フラグの意味、`poses.json` / `cache/` / `excluded/` の説明、`exclude_skipped.py`、テストの実行方法（`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`）。
- [ ] **Step 4: 全テスト** — pytest。Expected: 全件 PASS（学習リポジトリ無し環境では projector が skip）。
- [ ] **Step 5: Commit** — `🔥chore: 制御点編集・推論ラベル・旧照合コードを削除し README を書き直す`

---

### Task 14: 実データでの通し確認

**Files:** なし（確認のみ。問題があれば該当タスクのファイルを直して追加コミット）。

- [ ] **Step 1: 照合時間の計測** — Task 4 Step 5 のコマンド。Expected: WIDE < 0.3 s、候補合計 < 1 s。
- [ ] **Step 2: 50 枚の通し** — `20261012_data` から時系列で連続する 50 枚（`gnss.json` の `image_index` 順で val 000632 付近を含む範囲）の画像と、`gnss.json` をスクラッチの新しいディレクトリにコピーして起動（ラベルは無くてよい）。ブラウザで先頭から `Enter` を 20 回押す。Expected: 2 枚目以降の `init.source` が `propagated`、Enter 後の表示が 1 秒程度（先読みが追いつけば）、`labels/` に 20 ファイル、`poses.json` に 20 エントリ、`labels_backup/` と `cache/yolop/` ができる。
- [ ] **Step 3: 重ね画像での確認** — 確定した 3 枚について `projector.overlay(pose)` の線を画像に描いた PNG を作り（`scripts/` に小さなスクリプトを置いてよい）、目視で白線と地図線が重なっていること。
- [ ] **Step 4: 要確認の巡回** — `f` で flags 付きだけの一覧になり、`f` で戻ること。
- [ ] **Step 5: 結果を README の末尾「動作確認」に 3 行で記録して Commit** — `📝docs: 実データでの通し確認を記録`

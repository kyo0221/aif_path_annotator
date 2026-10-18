# ヨー調整ダイヤルと使用レーン選択 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ハンドル型ダイヤルでヨーを連続調整できるようにし、経路に使うレーンを「進行方向 ±60° 以内の最近傍」で自動選択しつつ GUI で選び直せるようにする。

**Architecture:** サーバ側は `projector.py` にレーン列挙・自動選択・指定レーンからの経路辿りを足し、`app.py` / `pose.py` / `server.py` が `lanelet`（null = 自動）を姿勢と一緒に受け渡す。フロントは `static/dial.js` を新設し、`app.js` にレーン一覧と `t` キー、`bev.js` に候補レーンの破線描画を足す。学習リポジトリは変更しない。

**Tech Stack:** 既存どおり（Python 3.12 venv、numpy、stdlib HTTP、素の JS）。学習リポジトリ `vectormap.py` の `_nearest_point_on_polyline` / `_project_onto_polyline` / `_walk` / `_drop_consecutive_duplicates` / `SLOT_NAMES` を import して使う。

**Spec:** `docs/superpowers/specs/2026-10-18-yaw-dial-and-lane-selection-design.md`（前提: `docs/superpowers/specs/2026-10-18-map-projection-annotator-design.md`）

## Global Constraints

- 実行・テストは前計画と同じ: `cd /home/kyo/aif_dataset && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/path_annotator/tests -q`（以下「pytest」。現在 71 件）。import 名は `tools.path_annotator.<module>`。
- レーン列挙: 探索半径 `LANE_SEARCH_RADIUS = 8.0` m、`aligned` は `angle_deg < 60`（`LANE_ALIGN_DEG = 60.0`）、一覧は最大 `LANE_MAX = 6` 件（指定レーンは件数に関わらず含める）、自動選択は `aligned` かつ `distance_m <= 5.0`（`LANE_AUTO_DISTANCE = 5.0`）の先頭。該当が無ければ `None`（学習リポジトリの `trace_paths` に任せる）。
- `angle_deg` は最近傍点での中心線接線と進行方向 `(cos yaw, sin yaw)` のなす角の絶対値（0〜180°）。
- フラグ `lane`: 使ったレーンの `angle_deg >= 60`、または自動選択が `None` で `trace_paths` に落ちたとき。要確認バッジの対象。
- `lanelet` は `poses.json` のエントリ項目（int | null）。次フレームへ引き継がない。`Esc` で null に戻る。
- ダイヤル: 直径 120 px、`DEG_PER_DIAL_DEG = 0.1`（ダイヤル 1° = ヨー 0.1°）、Shift で 5 倍、時計回りでヨーが減る、ダブルクリックでヨーだけ初期値へ、1 ドラッグ = undo 1 回、出どころ `manual`。
- JSON の数値は Python の float / int。色: 候補レーンの破線は `rgba(0,200,255,0.45)`、選択中の経路は従来色。
- コミットは小さく、絵文字 + 種別 + 日本語の件名、末尾に `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。`git status` に `__pycache__` が出ないこと。

## Review Focus

- 指定した `lanelet` が存在しない ID: `ValueError`（400）になり、サーバは落ちない（Task 1 / 3 のテスト）。
- 候補レーンが 1 つも無い場所（半径 8 m に中心線なし）: `lanes` は空、自動は `trace_paths` に落ち、`lane` フラグが立つ。経路が無ければ従来どおり `no_path`（Task 2 のテスト）。
- 指定レーンから straight 経路が取れない（レーンが短く後続が無い）: 行が 0 なら `label` は straight を含まず、確定は「レーンを選び直すか Esc」の文で 400（Task 2 のテスト）。
- `lanelet` を省略した旧クライアントの POST: `None` として扱い自動になる（Task 3 のテスト）。
- ダイヤルを ±180° をまたいで回す: 角度差は ±180° に正規化され、ヨーが飛ばない（Task 4 の selftest で確認）。

---

## ファイル構成

| ファイル | 変更 |
|---|---|
| `projector.py` | `lane_candidates` / `auto_lanelet` / `resolve_lane` / `_trace_from` を追加。`paths` / `overlay` / `label` / `bev` に `lanelet=None` を追加。`bev` に `lanes` を追加 |
| `pose.py` | `PoseStore.set(..., lanelet=None)`、エントリに `lanelet` |
| `app.py` | `_flags` に `lane`、`open_frame` / `preview` / `set_pose` / `confirm` が `lanelet` を扱う |
| `server.py` | POST body の `lanelet` を渡す |
| `static/dial.js`（新規） | ダイヤルの描画と入力 |
| `static/app.js` | `S.lanelet` / `S.lanes`、ダイヤル連携、レーン一覧、`t`、Esc、undo に lanelet |
| `static/bev.js` | `bev.lanes` の破線描画 |
| `static/index.html` / `style.css` | `#dial`、`#lanes`、キー表 |
| `tests/data/straight.osm` | 横切りレーン E（centerline のみ、x = 15、y −10→+10） |
| `tests/conftest.py` | `StubProjector` に `lane_candidates` / `auto_lanelet` / `resolve_lane` / `lanelet` 引数 |
| `README.md` | ダイヤル・レーン一覧・`t`・`lane` フラグ・動作確認 |

データ形:

- **lane**: `{"id": int, "distance_m": float, "angle_deg": float, "aligned": bool}`。API ではこれに `"auto": bool`、`"selected": bool` が付く。
- **resolve_lane の結果**: `{"id": int | None, "auto": bool, "angle_deg": float | None, "distance_m": float | None}`（`id` が None = `trace_paths` 任せ）。
- **bev**: 既存の `lines` / `paths` / `points` に `lanes: [{"id": int, "points": [[x, y], ...]}]`（車両座標、前方 −4〜28 m・横 ±13 m に切った区間ごと。1 レーンが複数区間なら `id` を繰り返す）。

---

### Task 1: projector のレーン列挙・自動選択・指定レーンからの経路

**Files:**
- Modify: `projector.py`（`Projector.__init__` 69、`paths` 84、`label` 117、`overlay` 121、`bev` 167、`load_projector` 176）
- Modify: `tests/data/straight.osm`、`tests/conftest.py`（`projector` フィクスチャは変更不要）
- Test: `tests/test_projector.py`

**Interfaces:**
- Consumes: 学習リポジトリ `vectormap` モジュールの `_nearest_point_on_polyline(polyline, position) -> (distance, point, index, tangent) | None`、`_project_onto_polyline(polyline, index, position) -> (point, next_index)`、`_walk(vector_map, lanelet_id, start_point, next_index, remaining) -> (points, remaining, terminal, visited)`、`_drop_consecutive_duplicates(points)`、`SLOT_NAMES`、`trace_paths`。
- Produces:
  - 定数 `LANE_SEARCH_RADIUS = 8.0`、`LANE_ALIGN_DEG = 60.0`、`LANE_MAX = 6`、`LANE_AUTO_DISTANCE = 5.0`。
  - `Projector.__init__(camera, vector_map, polylines, trace_paths, row_anchor_columns, vectormap_module=None)`：`vectormap_module` は上記の私有関数を持つモジュール。`None` のときレーン機能は `trace_paths` 任せ（`lane_candidates` は `[]`、`auto_lanelet` は `None`）。
  - `lane_candidates(pose, lanelet=None) -> list[lane]`：全 `vector_map.centerlines` を走査し距離 ≤ 8 m を集め、`aligned` 真を距離順 → 残りを距離順、先頭 6 件。`lanelet` が与えられ一覧に無ければ末尾に足す（半径外でも）。存在しない ID は `ValueError(f"lanelet {lanelet} がありません")`。
  - `auto_lanelet(pose) -> int | None`。
  - `resolve_lane(pose, lanelet=None) -> dict`（上の形。`lanelet` 指定なら `auto=False` でそのレーンの距離・角度、`None` なら `auto_lanelet` の結果、それも None なら `id=None, angle_deg=None`）。
  - `paths(pose, lanelet=None)`：`resolve_lane` の `id` が None なら従来どおり `trace_paths`、あれば `_trace_from(id, pose)`。
  - `_trace_from(lanelet_id, pose) -> dict[str, np.ndarray]`：`trace_paths` の `locate_lanelet` 以降と同じ手順（最近傍点の index → `_project_onto_polyline` → `_walk(horizon=HORIZON)` → 後続が 2 つ以上なら turn ごとに `_walk` して `_drop_consecutive_duplicates(vstack)`、長い方を採用、無ければ `{"straight": trunk}`）。出典（`vectormap.trace_paths`）をコメントに書く。
  - `label(pose, lanelet=None)`、`overlay(pose, lanelet=None)`、`bev(pose, points, lanelet=None)`。`bev` の戻りに `lanes` を追加（`lane_candidates(pose, lanelet)` の各レーンの中心線を `_bev_pieces` で切る）。
  - `load_projector` は `vectormap` モジュールを `vectormap_module` として渡す。
- `tests/data/straight.osm`：ノードを 3 つ追加（(15, −10)、(15, 0)、(15, 10)）、way E（centerline、タグなし）、lanelet relation E（`type=lanelet`、`centerline` メンバー）。接続は足さない。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_lane_candidates_prefers_aligned(projector): pose (14.5, 0.3, 0) → 先頭が A（aligned, distance < 0.5）、E が含まれ aligned=False かつ 85 <= angle_deg <= 95
def test_auto_lanelet_ignores_crossing_lane(projector): pose (15, 0, 0)（E の真上）→ auto_lanelet == A の id
def test_lane_candidates_includes_requested_far_lane(projector): pose (5, 0, 0) で lanelet=E の id → 一覧に E がある（距離 10 m）
def test_lane_candidates_unknown_id(projector): lanelet=999999 → ValueError
def test_paths_from_requested_lane(projector): pose (14.5, 0, 0), lanelet=E → paths["straight"] の点が x ≈ 15（全点 |x−15| < 0.01）で y が増加
def test_label_with_lanelet_validates(projector): pose (15, 0, 0), lanelet=E → label の straight が validate_label を通り有効行 ≥ 1（E は x=15 を横切るので車から見て横向き。有効行が 0 なら label は []。どちらでも ValueError にならないことを確認し、resolve_lane の angle_deg >= 60 を assert）
def test_bev_has_lanes(projector): bev(pose (14.5, 0, 0), points) に lanes があり、id に A と E を含み、各 points が窓内
```

- [ ] **Step 2: 失敗を確認** — pytest `-k "lane or lanelet"`。Expected: AttributeError / TypeError。
- [ ] **Step 3: 実装** — `projector.py`、`straight.osm`。
- [ ] **Step 4: 通過を確認** — pytest `-k projector`。Expected: PASS（既存 10 件 + 新 7 件）。
- [ ] **Step 5: Commit** — `✨feat: レーン候補の列挙と ±60° 以内の自動選択、指定レーンからの経路辿りを追加`

---

### Task 2: pose.py と app.py の `lanelet` と `lane` フラグ

**Files:**
- Modify: `pose.py`（`PoseStore.set` 82）、`app.py`（`_flags` 80、`_evaluate` 159、`open_frame` 183、`preview` 211、`set_pose` 226、`confirm` 237）
- Modify: `tests/conftest.py`（`StubProjector`）
- Test: `tests/test_app.py`、`tests/test_pose.py`

**Interfaces:**
- Consumes: Task 1 の `lane_candidates` / `resolve_lane` / `label(pose, lanelet)` / `overlay(pose, lanelet)` / `bev(pose, points, lanelet)`。
- Produces:
  - `PoseStore.set(split, name, pose, source, cost, flags, init, lanelet=None)`；エントリに `"lanelet"`。既存エントリに無ければ `get` は `None` を補う。
  - `Annotator._flags(cost, pose, init_pose, label, n_points, lane) -> list[str]`：`lane` は `resolve_lane` の結果。`cost, jump, no_path, few_points, pitch` の後に `lane`（`lane["id"] is None` または `lane["angle_deg"] >= 60`）。
  - `_evaluate(split, name, pose, init_pose, lanelet) -> (cost, flags, label, lane)`。
  - `open_frame` の応答に `"lanelet"`（保存値、None = 自動）と `"lanes"`（`lane_candidates(pose, lanelet)` の各要素に `"auto": id == resolve_lane(pose, None)["id"]`、`"selected": id == lanelet or (lanelet is None and auto)` を付けたもの）。`overlay` / `bev` / `label` / `flags` は保存済みの `lanelet` で作る。
  - `preview(split, name, pose, lanelet=None) -> {"overlay","bev","cost","label","flags","lanelet": resolve_lane の id, "lanes": [...]}`。
  - `set_pose(split, name, pose, source, lanelet=None)`、`confirm(split, name, pose, source, lanelet=None)`（straight 無しのエラー文は `straight 経路を投影できません（レーンを選び直すか Esc で初期値へ）`）。`match` は `lanelet` を受けても候補は自動レーンで作る（引数は無視してよい）。
  - 引き継ぎ初期化（`_initialize`）では `lanelet=None` で保存する。
- `StubProjector(valid_rows=23, n_points=None, lanes=None)`：`lanes` 属性（既定 `[{"id": 1, "distance_m": 0.2, "angle_deg": 3.0, "aligned": True}, {"id": 2, "distance_m": 0.6, "angle_deg": 85.0, "aligned": False}]`、`lanes=[]` を渡すと候補なし）、`lane_candidates(pose, lanelet=None)`（`lanes` を返す。`lanelet` が無い ID なら `ValueError`）、`auto_lanelet` → 1、`resolve_lane(pose, lanelet=None)`、`paths` 相当として `label(pose, lanelet=None)` / `overlay(pose, lanelet=None)` / `bev(pose, points, lanelet=None)` は `lanelet == 2` のとき straight の有効行を 0 にする（横向きレーン相当。`label` は `[]`）。`bev` に `lanes: []` を足す。

- [ ] **Step 1: 失敗するテストを書く**

```python
# test_pose.py
def test_pose_store_lanelet_roundtrip(tmp_path): set(..., lanelet=5) → get()["lanelet"] == 5；set(...) 省略 → None
# test_app.py
def test_open_frame_has_lanes(annotator): open_frame → "lanelet" is None、lanes[0]["auto"] and lanes[0]["selected"]、lanes[1]["aligned"] is False
def test_preview_with_lanelet(annotator): preview(..., lanelet=2) → "lane" in flags、label == []、"no_path" in flags、lanes[1]["selected"]
def test_set_pose_saves_lanelet(annotator): set_pose(..., lanelet=2) → poses.get()["lanelet"] == 2；open_frame の "lanelet" == 2
def test_confirm_with_crossing_lane_is_rejected(annotator): confirm(..., lanelet=2) → ValueError に "レーンを選び直す"
def test_confirm_with_lanelet_saves_it(annotator): confirm(..., lanelet=1) → ok、poses.get()["lanelet"] == 1
def test_lane_flag_when_auto_falls_back(make_annotator): StubProjector(lanes=[]) で auto_lanelet None → open_frame の flags に "lane"
def test_unknown_lanelet_is_value_error(annotator): preview(..., lanelet=999) → ValueError
```

- [ ] **Step 2: 失敗を確認** — pytest `-k "lanelet or lanes or lane_flag"`。Expected: TypeError / AssertionError。
- [ ] **Step 3: 実装**。
- [ ] **Step 4: 通過を確認** — pytest（全件）。Expected: PASS。
- [ ] **Step 5: Commit** — `✨feat: 姿勢に使用レーンを持たせ、レーン候補と lane フラグを返す`

---

### Task 3: server.py の `lanelet` 受け渡し

**Files:**
- Modify: `server.py`（POST ルーティング 97〜115）
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: Task 2 の `preview(..., lanelet)` / `set_pose(..., lanelet)` / `confirm(..., lanelet)`。
- Produces: `/api/preview`、`/api/pose`、`/api/confirm` が `body.get("lanelet")` を渡す（省略 = None = 自動）。`/api/match` は従来どおり。`/api/frame` と `/api/preview` の応答に `lanelet` / `lanes` が含まれる（app 側で済む）。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_frame_has_lanes(server): GET /api/frame → "lanes" と "lanelet" を含む
def test_preview_passes_lanelet(server): POST /api/preview {.., "lanelet": 2} → 200、body["lanelet"] == 2、"lane" in flags
def test_preview_without_lanelet_is_auto(server): POST /api/preview（lanelet 無し）→ 200、body["lanelet"] == 1
def test_unknown_lanelet_is_400(server): POST /api/preview {"lanelet": 999} → 400
def test_confirm_passes_lanelet(server, mini_root): POST /api/confirm {.., "lanelet": 1} → 200、poses.json の lanelet == 1
```

- [ ] **Step 2: 失敗を確認** — pytest `-k server`。Expected: KeyError / AssertionError。
- [ ] **Step 3: 実装**。
- [ ] **Step 4: 通過を確認** — pytest `-k server`。Expected: PASS。
- [ ] **Step 5: Commit** — `✨feat: API で使用レーンを受け渡す`

---

### Task 4: ヨー調整ダイヤル

**Files:**
- Create: `static/dial.js`
- Modify: `static/index.html`（`#bev` の直下に `<canvas id="dial" width="120" height="120">`、スクリプト読み込み順 `overlay.js`、`bev.js`、`dial.js`、`app.js`、キー表にダイヤル）、`static/style.css`、`static/app.js`

**Interfaces:**
- Produces: `window.Dial.create(canvas, {onTurnStart(), onTurn(dYawDeg), onReset()}) -> {draw(yawDeg)}`、`window.Dial.DEG_PER_DIAL_DEG = 0.1`。
  - 描画: 外輪（半径 54、線幅 6、`#3a3f47`、ハンドルの握り部分を 3 本のスポークで示す）、30° ごとの目盛り、上端の赤いマーク、中心のテキスト `yawDeg.toFixed(1) + "°"`（符号付き）。リム・スポーク・マークは `-yawDeg / DEG_PER_DIAL_DEG` 度（ヨー +1° で反時計回り 10°）回して描く。
  - 入力: `mousedown` で中心からの角度 `a0 = atan2(y − cy, x − cx)` を記録し `onTurnStart()`；`mousemove`（window）で角度 `a` を取り `Δ = wrap(a − a0)`（±π に正規化）、`onTurn(-degrees(Δ) · 0.1 · (ev.shiftKey ? 5 : 1))`、`a0 = a`；`mouseup`（window）で終了；`dblclick` で `onReset()`。
- `app.js`：`dialPanel = window.Dial.create($('dial'), {onTurnStart: pushUndo, onTurn: d => applyPose(shiftPose(S.pose, 0, 0, d, 0), 'manual'), onReset: () => setPose({...S.pose, yaw: S.frame.init.pose.yaw}, 'manual')})`；`renderHud` で `dialPanel.draw(poseDelta(S.pose, S.frame.init.pose).yaw_deg)`。

- [ ] **Step 1: dial.js を書く**。
- [ ] **Step 2: app.js / index.html / style.css を変える**。
- [ ] **Step 3: 手動確認** — 前計画 Task 10 と同じ起動（`mini2_t10`、ポート 8798）と `?selftest=`：`selftest` にダイヤルの `mousedown`/`mousemove`/`mouseup` を合成するケースを足し、(a) 時計回り 90° のドラッグでヨーが −9.0°（HUD）、source `manual`、undo 1 回、(b) ±180° をまたぐドラッグでヨーが飛ばない、(c) ダブルクリックでヨーが初期値に戻る、を `#msg` に出してスクリーンショット（`task-4-dial-*.png` を `.superpowers/sdd/<plan>/` に保存）。見た目の確認: 中心の数字、マークの回転。
- [ ] **Step 4: Commit** — `✨feat: ハンドル型のヨー調整ダイヤルを追加`

---

### Task 5: レーン一覧・俯瞰の候補レーン・`t` キー

**Files:**
- Modify: `static/app.js`、`static/bev.js`（`draw(bev, paths)` 57）、`static/index.html`（ダイヤルの下に `<h2>使用レーン</h2><ol id="lanes"></ol>`、キー表に `t`）、`static/style.css`、`README.md`

**Interfaces:**
- Consumes: Task 2/3 の `lanelet` / `lanes`（frame と preview）。
- Produces:
  - `app.js`：`S.lanelet`（null = 自動）、`S.lanes`。`openFrame` で `f.lanelet` / `f.lanes` を取り込む。`preview()` の body に `lanelet: S.lanelet`、応答の `lanes` で `S.lanes` を更新し一覧と俯瞰を再描画。`/api/pose` と `/api/confirm` の body に `lanelet: S.lanelet`。undo エントリに `lanelet` を含める（`pushUndo` / `history` / `applyPose` の引数に `lanelet`）。`setLane(id)`（null 可）= `pushUndo` → `S.lanelet = id` → `preview()`。`t` は `null → lanes[0].id → … → null` を巡回。`Esc` は `S.lanelet = null` も行う。
  - レーン一覧 `#lanes`：先頭行「自動（#<auto id>）」、以下 `#id  0.6 m  85°`。`selected` を強調、`aligned` でない行は `.warn`（薄い橙）。クリックで `setLane`。
  - `bev.js`：`draw(bev, paths)` で `bev.lanes` を `rgba(0,200,255,0.45)` の破線（`setLineDash([4,4])`、線幅 1）で描き、各区間の先頭付近に `#id` を小さく描く。
  - README：操作表に `t` とダイヤル（ドラッグ・ダブルクリック）、画面説明にレーン一覧、フラグ表に `lane`、仕組みの 5 にレーン自動選択の規則。

- [ ] **Step 1: bev.js / app.js / index.html / style.css を変える**。
- [ ] **Step 2: 手動確認** — 起動して `?selftest=` に (a) `t` を 1 回押すと `S.lanelet` が `lanes[0].id` になり一覧の強調が移る、(b) もう一度ずつ押して最後に null へ戻る、(c) 横切りレーンを選ぶと HUD の行数が変わりバッジに `lane`、を足す。スクリーンショット（`task-5-lanes-*.png`）で俯瞰の破線と番号、一覧の表示を見る。
- [ ] **Step 3: README を直す**。
- [ ] **Step 4: Commit** — `✨feat: 使用レーンの一覧と選択、俯瞰への候補レーン表示を追加`

---

### Task 6: 実データ（bag 先頭）での確認

**Files:** `README.md`（動作確認に 2 行追加）

- [ ] **Step 1: スクラッチのデータセット** — `20261012_data` から `train/…_000000`〜`…_000024`（存在する 13 枚）の画像と `gnss.json` を新しいディレクトリへコピーして起動。
- [ ] **Step 2: 自動レーン** — `GET /api/frame/train/perception_planning_20261012_164813_000000` の `lanes[0]`（auto）が進行方向に沿うレーン（`angle_deg < 60`）で、`label` に straight があり、`flags` に `no_path` が無いこと。続けて 000010・000012 も同様。`POST /api/confirm` が 3 枚とも 200。
- [ ] **Step 3: 手動レーン** — 000000 で `lanes` の `aligned=False` のレーンを `lanelet` に指定して `/api/preview` → `lane` フラグと行数の変化を確認。
- [ ] **Step 4: 画面** — スクリーンショット 1 枚（第 1 フレーム: 正面レーンの経路、ダイヤル、レーン一覧）。
- [ ] **Step 5: README に記録して Commit** — `📝docs: bag 先頭フレームでのレーン自動選択の確認を記録`

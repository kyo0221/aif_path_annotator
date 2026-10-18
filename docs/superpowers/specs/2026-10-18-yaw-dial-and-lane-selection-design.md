# ヨー調整ダイヤルと使用レーンの選択 設計

日付: 2026-10-18
対象: `tools/path_annotator`（ブランチ `feat/map-projection-annotator` の続き）
前提: `2026-10-18-map-projection-annotator-design.md` の実装が完了していること。

## 1. 意図

1. **ヨー調整ダイヤル**: ヨーの微調整はキー（`u` `i`）とホイールしかなく、連続量を手で合わせにくい。
   車のハンドルを模したダイヤルをドラッグで回し、投影のヨーを連続的に合わせられるようにする。
2. **使用レーンの選択**: 経路は学習リポジトリの `trace_paths` が「5 m 以内で最も近く、進行方向との角度が 90° 未満の中心線」を自車レーンとして選ぶ。
   交差点内（bag の先頭フレームなど）では、進行方向と 85° ずれた横切りのレーンが選ばれ、straight 経路が画像の外へ出て確定できない。
   自動選択を進行方向に沿うレーン優先に改め、さらに人が GUI でレーンを選べるようにする。

## 2. 利用者の決定

- ダイヤルは俯瞰図の直下、直径 120 px。中心に初期値からのヨー補正量を表示。ドラッグで中心まわりの角度変化ぶん回す。ダイヤルの外へ出ても離すまで追従。1 回のドラッグが undo 1 回ぶん。
- 向きはハンドルと同じ: 時計回りに回すと車は右を向く（ヨーが減る）。
- 感度: ダイヤル 1° = ヨー 0.1°（1 周 36°）。Shift で 5 倍。ダブルクリックでヨーだけ初期値へ戻す。出どころは `manual`。
- 自車の向きから ±60° 以上ずれた方向に経路が出る場合は、別のレーンを候補として出す。
- 使用レーンを GUI でも設定できる。

## 3. 設計

### 3.1 レーンの列挙と自動選択（サーバ）

- `projector.py` に `lane_candidates(pose, lanelet=None) -> list[dict]` を足す。全 lanelet の中心線について最近傍点を求め（学習リポジトリの `_nearest_point_on_polyline` を使う）、
  距離 ≤ `LANE_SEARCH_RADIUS = 8.0` m のものを集める。各要素は
  `{"id": int, "distance_m": float, "angle_deg": float（最近傍点での接線と進行方向の角度の絶対値、0〜180）, "aligned": angle_deg < 60}`。
  並び順は `aligned` が真のものを距離順、その後に残りを距離順。最大 6 件。
  `lanelet` に ID を渡した場合、そのレーンが半径外や 7 件目以降でも必ず一覧に含める（末尾に足す）。存在しない ID は `ValueError`。
  これにより、指定レーンの距離と角度は常に一覧から引ける（`lane` フラグと画面表示に使う）。
- 自動選択 `auto_lanelet(pose) -> int | None`: `aligned` かつ `distance_m ≤ 5.0`（`trace_paths` の `max_distance` と同じ）の先頭。
  無ければ `None`（= 学習リポジトリの `trace_paths` にそのまま任せる。従来どおり 90° 未満の最近傍）。
- `paths(pose, lanelet=None)`: `lanelet` が指定されていればそのレーンから、`None` なら `auto_lanelet(pose)` の結果から辿る。
  指定レーンからの辿りは学習リポジトリの `_project_onto_polyline` と `_walk`、`SLOT_NAMES` を使い、`trace_paths` の分岐処理（trunk → 複数の後続があれば turn ごとに合成）と同じ手順で作る
  （`trace_paths` 本体は `locate_lanelet` を内部で呼ぶため、開始レーンを与えられない。分岐処理は `trace_paths` から写し、出典をコメントに書く）。
  `auto_lanelet` が `None` のときは `trace_paths` をそのまま呼ぶ。
- `overlay(pose, lanelet=None)`、`label(pose, lanelet=None)`、`bev(pose, points, lanelet=None)` は `paths(pose, lanelet)` を使う。
  `bev` には候補レーンの中心線（車両座標、前方 −4〜28 m・横 ±13 m に切る）を `lanes: [{"id", "points": [[x,y],...]}]` として加える。

### 3.2 姿勢エントリと API

- `poses.json` のエントリに `lanelet: int | null` を足す（null = 自動）。引き継ぎでは次のフレームへ持ち越さない（毎フレーム自動から始める）。
- `flags` に `lane` を足す: 使っているレーン（自動または指定）の `angle_deg ≥ 60`、または `auto_lanelet` が `None` で `trace_paths` に落ちたとき。
- API:
  - `GET /api/frame` の応答に `lanelet`（現在の選択、null = 自動）、`lanes`（`lane_candidates` に `selected: bool`、`auto: bool` を足したもの）を加える。
  - `POST /api/preview`、`/api/pose`、`/api/confirm`、`/api/match` の body に任意項目 `lanelet`（int | null）を加える。省略時は null と同じ = 自動（ページは常に送る。サーバは省略と null を区別しない）。
  - `/api/preview` の応答にも `lanelet`（実際に使ったレーン ID。自動のときは自動で選ばれた ID）と `lanes` を含める。
    一覧の距離・角度・自動レーンは姿勢で変わるので、姿勢を動かすたびにプレビューで更新する（全 lanelet の最近傍点計算は数十レーン分の小さなループで、プレビューの予算内に収まる）。
    `/api/pose` は `{"ok": true}` のまま、`/api/match` は従来の形のまま（直後のプレビューで一覧が更新される）。
  - 候補（`candidates`）の overlay は自動レーンで作る（候補ごとのレーン指定はしない）。
- `Annotator.confirm` は `label(pose, lanelet)` で検証・保存する。straight が無い場合のエラー文は
  「straight 経路を投影できません（レーンを選び直すか Esc で初期値へ）」にする。

### 3.3 画面

- **ダイヤル** `static/dial.js`: `window.Dial.create(canvas, {onTurn(dYawDeg), onReset()}) -> {draw(yawDeg)}`。
  描画は外輪（リム）、上端のマーク、30° ごとの目盛り、中心の補正量テキスト（`+2.4°`）。リムとマークは `yawDeg / 0.1` 度だけ回して描く（ヨー +1° でダイヤルは反時計回りに 10°）。
  `mousedown` で中心からの角度を記録し、`mousemove`（window）で角度差 Δ を求め `onTurn(-Δ · 0.1 · (Shift ? 5 : 1))` を呼ぶ（時計回り Δ > 0 でヨーが減る）。角度差は ±180° に正規化する。`mouseup`（window）で終了。`dblclick` で `onReset()`。
- **レーン一覧**: ダイヤルの下に「使用レーン」の小さな一覧。先頭は「自動（#302）」、続いて候補を `#id 0.6 m 85°` の形で表示。`aligned` でない行は薄い橙で示す。クリックで選択。キー `t` で自動 → 候補 1 → 候補 2 → … → 自動 と巡回。
  選択するとプレビュー（`/api/preview` に `lanelet` を付ける）で画像・俯瞰・コスト・行数が更新され、`/api/pose` と `/api/confirm` に `lanelet` が付く。
- **俯瞰図**: 候補レーンの中心線を細い破線（薄いシアン）で描き、番号を添える。選択中のレーンの経路は従来どおり太線。
- **上部バー**: `lane` フラグは他のフラグと同じく「要確認」の対象。
- `app.js` の状態に `lanelet`（null = 自動）と `lanes` を足す。フレームを開くとサーバの値で初期化。`Esc` はヨー・位置とともにレーンも自動に戻す。
- キー表（`#keys`、README）に `t` とダイヤル、ダブルクリックを足す。

### 3.4 テスト

- `tests/data/straight.osm` に、A の途中（x = 15）を横切るレーン E（y = −10 → +10、x = 15、centerline のみ）を足す。
- projector: `lane_candidates` が pose (14.5, 0.3, 0) で A（aligned, 0.3 m）と E（0.5 m, 90°, aligned=False）を返し、A が先頭であること；`auto_lanelet` が A であること；
  pose を E の真上（15, 0, 0）に置いても `auto_lanelet` が A（aligned）であること；`paths(pose, lanelet=E の id)` が E に沿った経路を返すこと；`label(pose, lanelet)` が 48 行で検証を通ること。
- app: `open_frame` の応答に `lanelet` と `lanes`（`selected` / `auto` 付き）があること；`set_pose` / `preview` / `confirm` が `lanelet` を受け取り poses.json に保存すること；
  指定レーンが `angle_deg ≥ 60` なら `lane` フラグが立つこと。`StubProjector` に `lane_candidates` / `auto_lanelet` と `lanelet` 引数を足す。
- server: `/api/frame` の `lanes`、`/api/preview` と `/api/confirm` の `lanelet` 受け渡し。
- ダイヤルと一覧は手動確認（ヘッドレス Chrome のスクリーンショットと `?selftest`）。

## 4. 設計者の判断

- 自動選択を「±60° 以内で最も近いレーン」に変える（従来は 90° 未満）。converter.py とは選び方が変わるが、交差点内で横切りレーンを拾う問題への直接の対策であり、ずれた場合は `lane` フラグで分かる。
- レーン選択は引き継がない（交差点を抜ければ自動で正しいレーンに戻るため）。
- 候補レーンの探索半径 8 m、一覧は最大 6 件。指定したレーンは半径外でも一覧に残し、選択は維持する（自動に戻さない）。
- レーン一覧と自動レーンはプレビューのたびに再計算して返す（姿勢に依存するため）。
- ダイヤルの感度 0.1°/°、Shift 5 倍。
- 候補（pose の候補）は自動レーンで描く。

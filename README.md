# 地図投影アノテータ

ベクターマップを車両姿勢から画像へ投影して、48 行の経路ラベル（`labels/<split>/<name>.txt`）を作るローカル Web ツール。
人が点を打つのではなく、フレームごとの車両姿勢（x, y, ヨー, ピッチ）を俯瞰図・候補・キーで直し、地図の線が画像の白線に合ったところで確定する。

## 必要なもの

- Python 3.12 の venv（CPU 版 torch 入り）。例では `/home/kyo/aif_dataset/.venv`。
- 学習リポジトリ（`--learning-repo`。投影コードの取得元）。
- vectormap（`aiformula_course.osm`）、YOLOPv2 の重み（`yolopv2.pt`）、カメラ外部パラメータの `main_params.yaml`。
- データセットのルート: `images/<split>/*.png` と `gnss.json`。`labels/` は無くてよい。
  `gnss.json` には各フレームの `map.x` / `map.y` / `map.yaw_imu_rad` / `map.yaw_motion_rad`、`bag`、`image_index`、`stamp` が要る。

## 起動コマンド

    cd /home/kyo/aif_dataset
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/path_annotator/annotate.py \
      --root /home/kyo/aif_dataset/20261012_data \
      --learning-repo /home/kyo/formula_ws/src/learning \
      --vectormap /home/kyo/formula_ws/src/map/vectormap_server/config/aiformula_course.osm \
      --yolop /home/kyo/formula_ws/src/perception/road_detector/weights/yolopv2.pt \
      --params /home/kyo/formula_ws/src/main_executor/config/main_params.yaml

引数:

- `--root`（必須）: データセットのルート。
- `--learning-repo` / `--vectormap` / `--yolop` / `--params`（4 つとも必須）: 上記のとおり。
- `--port`: 待受ポート（既定 8765）。
- `--no-browser`: ブラウザを自動で開かない。
- `--cost-threshold`: 候補を採用する照合コストの上限、m（既定 0.25）。

## 画面

- 上部バー: フレーム名、位置（n / 全体）、順序（時系列）、照合コスト、要確認フラグ、YOLOP キャッシュの状況。
- 左: 画像。地図の線（シアン `#00c8ff`）と YOLOP の白線マスク（緑 `#00dc00`）を半透明で重ねる。
  経路ラベルは straight が黄 `#ffe600`、left が桃 `#ff78ff`、right が橙 `#ff8c00`。
- 右列: 俯瞰図（BEV）、数値（補正量・コスト・行数・姿勢の出どころ）、不透明度スライダー（投影・検出）、キュー。
- 下: 照合の候補（最大 4 枚）。クリックまたは `1`〜`4` で姿勢をその候補にする。

## 操作

| キー | 動作 |
|---|---|
| Enter | 確定して次へ |
| ← → | 前 / 次のフレーム（姿勢は保存される） |
| h l | 横へ動かす |
| j k | 前後へ動かす |
| u i | ヨー |
| , . | ピッチ |
| Shift + 上の移動キー | 5 倍の刻み |
| Esc | 初期値へ戻す |
| z / Shift+z | 戻す / やり直す |
| g | 白線マスクの表示切替 |
| m | 地図線の表示切替 |
| [ ] | 投影の不透明度を下げる / 上げる |
| o | 押している間だけ素の画像 |
| s | skip |
| f | 要確認のフレームだけ巡回（切替） |
| 1〜4 | 候補を選ぶ |
| r | 広い窓で再探索 |
| + - | 俯瞰の縮尺 |

俯瞰図ではドラッグで横・前後、ホイールでヨーを動かす。スライダーで投影・検出の不透明度を変える。

## 進め方

時系列に並べたフレームを先頭から見る。各フレームは前フレームの姿勢を引き継いだ初期値で開く
（最初のフレームは GNSS の初期姿勢）。必要なら候補・キーで直し、Enter で確定する。
確定するとその姿勢でラベルを保存して次のフレームへ進み、次の初期値には確定した姿勢が使われる。
`f` で「要確認」フラグ付きのフレームだけに絞れる。

## 要確認フラグ

- `cost`: 照合コストがしきい値（`--cost-threshold`）を超える。
- `jump`: 引き継いだ初期値から横 0.3 m、またはヨー 2° を超えて動いた。
- `no_path`: straight 経路が取れない。
- `few_points`: 白線の点が 30 未満。
- `pitch`: ピッチが 0 ではない。

## 確定時の棄却基準

`converter.py` と同じ基準で、次の場合は確定できない（400 を返し、メッセージを出す）。

- 可視行が 10 行未満。
- straight の可視行の 50% 以上が画像端にある。

## 保存されるファイル（`--root` 以下）

- `labels/<split>/<name>.txt`: 48 行形式（従来どおり）。
- `poses.json`: フレームごとの確定姿勢。
- `annotation_state.json`: フレームごとの状態（done / skip など）。
- `labels_backup/`: 上書き前の既存ラベルの退避。
- `cache/yolop/`: YOLOP マスクのキャッシュ。消しても再計算される。
- `excluded/`: `exclude_skipped.py` が移した skip フレーム。

## skip したフレームを除外する

    .venv/bin/python tools/path_annotator/exclude_skipped.py --root <dataset> [--dry-run]

`annotation_state.json` で skip のフレームの画像とラベルを `excluded/<split>/` へ移す。
戻すには同名ファイルを `images/`・`labels/` へ戻す。`--dry-run` は移さずに一覧だけ出す。

## 照合の仕組み

1. GNSS の位置と IMU ヨーから初期姿勢を作る。IMU ヨーには bag ごとの偏りがあるので、補正してから使う。
2. YOLOP の白線マスクを、カメラ行列と平坦路面の仮定で車両前方の俯瞰座標の点へ変換する。
3. vectormap の観測可能な線から距離場を作り、白線点の平均距離（コスト）が小さくなる姿勢を探す。
4. 探索は初期値まわりの狭い窓と、`r` キーによる広い窓の二段。コストの低い順に候補を最大 4 つ出す。
5. 選んだ姿勢で自車の経路中心線を投影して 48 行のラベルにする。

## テスト

    cd /home/kyo/aif_dataset
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/path_annotator/tests -q

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` は、ROS の PYTHONPATH が持つ pytest プラグイン（`lark` を import する）を読ませないため。
`PYTHONDONTWRITEBYTECODE=1` は `__pycache__` を作らせないため。
`projector` のテストは学習リポジトリが無いと skip される。

照合の速度は次で測る。

    .venv/bin/python tools/path_annotator/scripts/bench_matching.py --osm <aiformula_course.osm> [--points 800]

## 動作確認

- 2026-10-18、20261012_data の val 連続 50 枚（000632〜000734）で通し確認。
- 照合: WIDE 0.18 s、候補 4 起点 0.71 s、距離場の最大誤差 0.025 m。
- Enter 20 回で全 20 枚を確定（400 なし）。2 枚目以降は `propagated`（image_index 差 6 の 000674 だけ GNSS から再初期化）、Enter 後の表示は先読み済みで 0.5 s 前後。横方向は白線と地図線が重なるが、引き継ぎ中に縦方向が 1 枚 0.3〜0.5 m ずつ遅れ、18 枚で約 8 m ずれた（要対応）。

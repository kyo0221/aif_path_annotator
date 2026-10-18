# 経路ラベル アノテーションツール

`labels/<split>/<frame>.txt` の経路ラベル（`xp` / `h_vector` / `class`）を手で直すためのローカル Web ツール。

## 実行コマンド

    python3 tools/path_annotator/annotate.py --root <データセットのルート>

例:

    python3 tools/path_annotator/annotate.py --root dataset/sim/dataset_v2

引数:

- `--root`（必須）: データセットのルート。`images/<split>/*.png` と `labels/<split>/*.txt` を含むディレクトリを指定する。
- `--port`: 待受ポート（既定 8765）。使用中なら変える。
- `--no-browser`: 起動時にブラウザを自動で開かない。
- `--weights`: 推論に使う重みファイル。指定するとキューを「既存ラベルと推論ラベルの乖離」の大きい順にする（無指定なら従来どおり異常スコア順）。
- `--learning-repo`: ネットワーク実装のあるリポジトリ（`--weights` と併用必須）。
- `--branch-threshold`: 自動アノテーションで left/right を出す最小の横差、px 単位（既定 60.0）。
- `--vectormap`: vectormap（`aiformula_course.osm`）。指定すると、GNSS と白線の俯瞰照合で straight 経路を作る機能（`v` キー / 地図ボタン）が有効になる。
  `--yolop`（YOLOP の TorchScript 重み）、`--params`（カメラ外部パラメータの `main_params.yaml`）、`--learning-repo` と併用する。
  データセットのルートに `gnss.json`（`tools/label_audit/add_gnss.py` で作成。フレームごとの GNSS 位置・ヨー）が必要。

vectormap 照合の例:

    python3 tools/path_annotator/annotate.py --root ~/aiformula_dataset/real/manual_annotation/base \
      --learning-repo ~/formula_ws/src/learning \
      --vectormap ~/formula_ws/src/map/vectormap_server/config/aiformula_course.osm \
      --yolop ~/formula_ws/src/perception/road_detector/data/weights/yolopv2.pt \
      --params ~/formula_ws/src/main_executor/config/main_params.yaml

照合の手順（`vectormap_match.py`）: `gnss.json` の地図座標の位置と IMU ヨーを初期値に、YOLOP の白線を車両前方 1.6〜7m の俯瞰座標
（カメラ行列・平坦路面）へ変換し、vectormap の観測可能な線へ横 ±1.5m・前後 ±1.5m・ヨー ±8° の格子探索で照合して姿勢を補正する。
補正後の位置から自車の lanelet の中心線を前方 20m たどってカメラへ投影し、straight だけを置き換える（left/right は残す、undo 可）。
HUD に補正量と照合コスト（白線点と地図線の平均距離）を出し、0.25m を超えると「照合不良: 要確認」と表示する。

## 操作方法

画面右上のバッジで add / delete モードを切り替える（`d` キー、既定 add）。
クリックの意味はモードで決まる: add は点を置く（既存の点があれば位置を更新するだけで消さない）、
delete は点を消す。どちらのモードでも、クリックした行が次の範囲操作の始点になり、
続けて `Shift+クリック` で終点を指定すると、始点〜終点（両端含む）をまとめて
add なら直線補間、delete なら削除できる。始点マーカーはバッジと同じ色（add: 緑 / delete: 赤）
かつ色だけに頼らない形（add: 輪 / delete: ×印）で描かれ、バッジを見なくても区別できる。
始点は次でクリア（解除）される: フレーム移動、エントリ切替（`Tab`）、モード切替（`d`）、
エントリ追加・削除（`n` / `x`。undo/redo も含めていずれも内部の epoch が進むため自動的に無効になる）、
Shift+クリックによる範囲操作の実行後。行 0〜24（`ROW_START` 未満）はどちらの操作も受け付けない。

`--weights` を指定して起動した場合、モードバッジの左隣の「自動」ボタン（または `a` キー）で、
このフレームを推論したラベルに丸ごと置き換えられる（保存前と同様 `u` で undo 可）。
重みを指定していないときは押すと HUD にエラーが出る。

## キーバインド

| キー／操作 | 動作 |
|---|---|
| ドラッグ | 制御点を動かす（近傍行が滑らかに追従） |
| `←` `→` | 前／次フレーム |
| `Enter` | 確認済みにして次へ |
| `h` `l` / `Shift+h` `Shift+l` | 全体を左右に 1px / 5px |
| `j` `k` | 傾き変更（下端固定で上端を左右） |
| `1` `2` `3` | class を straight / left / right |
| `d` | add / delete モードの切替（既定 add、右上のバッジに表示） |
| `a` / 自動ボタン | このフレームを推論し、返ったラベルで丸ごと置き換える（`--weights` 指定時のみ。無指定なら HUD にエラー） |
| `v` / 地図ボタン | GNSS + 俯瞰照合 + vectormap のレーン中央で straight だけを置き換える（`--vectormap` 指定時のみ、undo 可） |
| 行をクリック | add: 点を置く（既存でも位置更新・消さない） ／ delete: 点を消す（無ければ何もしない）。行 0〜24 は編集不可 |
| `Shift+クリック` | 始点〜その行を add: 直線補間 ／ delete: まとめて削除（両端含む）。始点未選択なら HUD に警告 |
| 始点のクリア（解除）条件 | フレーム移動 / `Tab` / `d` / `n` `x`（エントリ増減。undo・redo も epoch が進むため自動的に無効）/ Shift+クリックの実行後 |
| `n` / `x` / `Tab` | 未使用の class でエントリ追加 / 削除（`straight` は削除不可）/ 切替 |
| `p` | 前後 ±N フレームへ伝播 |
| `,` `.` | 伝播幅 N を 1〜50 で増減（既定 5） |
| `[` `]` | 制御点数 K を 1〜5 で増減 |
| `R` | 制御点から曲線を再生成 |
| `u` / `Ctrl+u` | undo / redo |

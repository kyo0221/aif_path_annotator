# 一覧（ギャラリー）からのアノテーション 設計

日付: 2026-10-21
対象: `tools/path_annotator`
前提: `2026-10-18-map-projection-annotator-design.md`、`2026-10-18-yaw-dial-and-lane-selection-design.md` の実装が完了していること。
`projector.py` の分岐クラス修正（2026-10-21、README「経路のクラス」）が入っていること。

## 1. 意図

今のツールは全フレームを時系列に 1 枚ずつ回る前提で作られている。しかし自動投影でラベルが一通り付いたあとは、
「おかしいフレームだけを見つけて直す」作業が中心になる。`~/Downloads/base4_annotation/preview/index.html`
（投影画像を並べた静的な一覧ページ）でスクロールしながら問題のあるフレームを探す方法が実際に役立ったので、
これをツールに取り込み、一覧 → クリックでアノテーション → 一覧の同じ場所へ戻る、という流れを一つの画面で行えるようにする。

## 2. 利用者の決定

- 投影画像（地図線 + 経路 + 48 行の点）がグリッドで並ぶ一覧を、スクロールしながら確認する。
- 問題のある画像をクリックすると、今のツールと同じアノテーション画面（画像・俯瞰・ダイヤル・候補・レーン一覧）に入り、同じ操作で直せる。
- 地図をずらすなどの変更をした時点で保存する。**Enter は不要**で、変更した時点でラベル（`labels/<split>/<name>.txt`）まで書き換わり、状態は done になる。
- キー一つで一覧に戻る。戻り先は、もともと見ていた地点（絞り込み・並び・スクロール位置がそのまま）。
- アノテーション画面での `←` `→`（前後のフレーム）は、**一覧で絞り込んで並んでいた順**で隣に移る。

## 3. 目標・非目標・避けること

目標

- 一覧はツールのサーバが配信し、状態（unseen / done / skip、要確認フラグ、手で直したか）と投影画像が常に最新になる。
- 一覧で絞り込み・並べ替え・検索ができ、気になるフレームだけを開ける。
- 直した結果が一覧に戻った時点でサムネイルと状態に反映される。

非目標

- 従来の「時系列に順番に回る」画面を無くすこと（残す。`/annotate` で開く）。
- 配布キット（`annotator_kit`）の zip を作り直すこと（別作業）。
- 照合・投影・レーン選択の仕組みを変えること。
- 複数人の同時編集。

避けること（これが起きたら失敗）

- 壊れたラベル（straight が無い、可視行が少ない、画像端に張り付く）を黙って書くこと。
- 一覧に戻ったとき、絞り込みやスクロール位置が失われて最初からやり直しになること。
- 一覧を開くたびに全フレームの投影を描き直して、表示が数十秒止まること。

## 4. 制約

- サーバは今の `http.server`（`ThreadingHTTPServer`）のまま。新しいフレームワークや npm は入れない。
  ページは今の `static/` と同じ素の HTML / CSS / JS。
- ラベルの書き込みは `LabelStore.save`（原子的書き込み + `validate_label`）を通す。保存の可否は `validate_label` と `rejection_reasons` で決める（今の確定と同じ基準）。
- 既存のテスト（`tests/`、pytest）を壊さない。新しい API と保存の挙動にはテストを足す。
- データセットの構造は今のまま（`images/`、`labels/`、`poses.json`、`annotation_state.json`、`cache/yolop/`）。
  `preview/` を投影画像のキャッシュ置き場として使う（`~/Downloads/base4_annotation/preview/` に既にある jpg はそのまま使える）。

## 5. 設計

### 5.1 画面構成と遷移

- `GET /` → 一覧ページ（`static/gallery.html`、新規）。`GET /annotate` → 今のアノテーション画面（`static/index.html`）。
  `annotate.py` がブラウザで開く URL は `/`（一覧）にする。
- 一覧ページの中に、アノテーション画面を**画面いっぱいの iframe**（`/annotate?mode=gallery&frame=<split>/<name>`）として重ねて開く。
  一覧の DOM はそのまま残るので、閉じれば絞り込み・並び・スクロール位置が厳密に元のままになる。
  （1 ページに統合して `app.js` を作り直す案もあるが、キー操作の衝突を避けつつ変更を小さくするため iframe にする。これは実装側の判断。）
- 戻る操作: アノテーション画面の上部バーに「一覧へ戻る（q）」ボタンを置き、キー `q` でも戻る。
  `Esc` は今のツールで「初期値へ戻す」に使っているので変えない。
- 一覧と iframe の間は同一オリジンの `postMessage` でやり取りする（5.6）。

### 5.2 一覧ページ

**データ** `GET /api/gallery` → 全フレームの表:

```
{"frames": [{"split", "name", "index"(時系列での位置), "bag", "image_index",
             "status"(unseen|done|skip), "edited"(bool), "flags"([...]), "cost", "source",
             "classes"(["straight","left"] など。ラベルが無ければ []), "rows"(straight の可視行数),
             "has_manual_label"(bool|null), "relabeled"(bool), "pose"(entry|gnss|none), "thumb_version"(文字列)}],
 "bags": [...], "has_manifest": bool, "has_relabeled": bool}
```

- `bag` / `image_index` は `FrameIndex` と同じ出どころ（`gnss.json` の meta、無ければフレーム名）。
- `classes` / `rows` はラベルファイルから。初回の `/api/gallery` で全ラベルを読んでメモリに持ち、保存のたびにそのフレームぶんだけ更新する
  （8 千枚で 1〜2 秒。起動時ではなく初回要求時に作る）。
- `has_manual_label` は `<root>/manifest.csv` があればその列から、無ければ null。`relabeled` は `<root>/relabeled_frames.json` があれば
  その `changed[].key` に含まれるか、無ければ false。この 2 つはデータセット固有の付加情報なので、無いときは一覧の絞り込みに出さない。
- `thumb_version` はサムネイルのキャッシュ破りに使う文字列（姿勢エントリの `ts` とラベルの mtime から作る。実装側で決める）。

**絞り込み・並び**（すべてページ側で行う。今の `index.html` と同じ作り）

- split、bag、状態（all / unseen / done / skip）、フラグ（all / あり / なし / jump / cost / no_path / few_points / pitch / lane）、
  姿勢の出どころ（propagated / auto / gnss / manual / candidate）、経路（all / left あり / right あり / straight のみ）、
  手で直した（edited）、手動ラベル（`has_manifest` のときだけ）、再投影で変更（`has_relabeled` のときだけ）、フレーム名の検索、サムネイル幅。
- 並び: 時系列（既定。ツールのキューと同じ順）、名前、コスト降順。
- 絞り込み・並び・幅は URL のハッシュにも書き、再読み込みしても保たれる。スクロール位置はページを閉じない限り DOM に残る。

**カード**: サムネイル（`loading="lazy"`）、`split/name`、チップ: 状態（unseen / done / skip を色分け）、bag、出どころ、cost、フラグ（橙）、
left / right（経路の色）、手で直した、再投影で変更。クリックでそのフレームを開く（5.5）。
一覧の凡例に色（地図線シアン、straight 黄、left 桃、right 橙、点 = 48 行ラベル）とキー（クリックで開く、q で戻る）を書く。
上部に「順番に回る（従来の画面）」へのリンク（`/annotate`）を置く。

### 5.3 サムネイル

- `GET /api/thumb/<split>/<name>.jpg?v=<thumb_version>` → 投影画像の JPEG（640×384、品質 90）。
- 描画は `scripts/render_overlay.py` と同じ規約: 生画像に `projector.overlay(pose, lanelet)` の地図線（シアン `#00c8ff`、太さ 2）と経路（straight 黄、left 桃、right 橙、太さ 3）を重ね、
  48 行の点は**保存済みラベル**から描く（ラベルが無いフレームは overlay の rows から）。姿勢・レーンは `poses.json` のエントリから。
- **サムネイルは何も初期化せず、何も書かない。** `poses.json` にエントリが無いフレームは、`gnss.json` の初期姿勢（`GnssIndex.initial_pose`。照合も YOLOP も使わない）で
  地図線・経路を描き、`/api/gallery` の `pose`（`entry` / `gnss` / `none`）でそれを示す。カードには「姿勢なし」のチップを出す。GNSS も無ければ生画像 + ラベルの点だけ。
  初期化（照合・`poses.json` への書き込み）は、クリックして開いたとき（`open_frame`）に今までどおり行う。一覧を眺めるだけで状態が変わってはいけない。
- キャッシュ: `<root>/preview/<split>/<name>.jpg`。あればそれを返し、無ければ描いて保存してから返す。
  保存（5.4）や状態変更でフレームの見た目が変わったら、そのフレームのキャッシュファイルを消す（次の要求で描き直す）。
  変更のたびに描き直さないのは、キー連打で 1 秒に何十回も保存が走るため。
- 描画は `projector` を共有するので、1 つのロックで直列にする（1 枚 50 ms 程度。初回の一覧表示で見える分だけ描く）。
- 応答ヘッダは `Cache-Control: no-cache`。ページ側は `v` を変えて取り直す。

### 5.4 変更時の保存（自動保存）

`POST /api/save` body `{split, name, pose, source, lanelet}` → `Annotator.save(...)`:

1. 姿勢を保存する（`/api/pose` と同じく `poses.json` へ。cost・flags も更新）。ただし `/api/pose` の「done のフレームを動かしたら unseen に戻す」は、ここでは次の判定に置き換える。
2. `label = projector.label(pose, lanelet)` を作り、straight があり、`validate_label` を通り、`rejection_reasons` が空なら
   ラベルを書く → 状態を `done` にする → サムネイルのキャッシュを消す。
   `edited`（「手で直した」）は、**書くラベルが既存のラベルファイルと異なるときだけ** 真にする（同じなら前の値のまま）。
   Enter で変更無しのフレームを確認しただけのものや、自動投影と同じ結果に戻したものを「手で直した」と数えないため。
   `store.save` に `edited` を渡せるようにする（省略時は今までどおり真）。
3. 通らなければラベルは**前のまま**残し、状態は `done` なら `unseen` に戻す（ラベルと姿勢が食い違うため。skip はそのまま）。
4. 応答 `{"saved": bool, "status": 保存後の状態, "reasons": [...], "flags", "cost", "label", "lanelet", "lanes"}`。
   `reasons` は確定時と同じ文言（「straight 経路を投影できません（レーンを選び直すか Esc で初期値へ）」、「可視行 N < 10」、「straight の可視行の N% が画像端」）。

ページ側（ギャラリーモードのときだけ）:

- 姿勢やレーンが変わる操作のすべて（h l j k u i , .、候補 1〜4、r、t・レーンのクリック、ダイヤル、俯瞰のドラッグとホイール、z / Shift+z、Esc）で保存する。
  表示だけの操作（g m o [ ] + -、不透明度）は保存しない。
- 保存はプレビューと同じ方式でまとめる: 送信中は最新の姿勢を控えておき、応答が返ったら差があれば送り直す（キーリピートで要求を溜めない）。
- 応答の `saved` が偽なら `reasons` を `#msg` に出す。上部バーに保存状態の印を出す（例: 「保存 done」/「未保存: 可視行 6 < 10」）。
- フレームを離れる前（← → Enter s q）に、控えている保存を送り終えるのを待つ。
- `Enter` は「今の姿勢をそのまま保存（done）して一覧の順で次へ」。変更していないフレームを「確認済み」にするために残す。
  `s` は今までどおり skip にして次へ。
- 従来モード（`/annotate` を直接開いたとき）は今の動き（姿勢は離れるときに保存、ラベルは Enter）のまま。

### 5.5 アノテーション画面のギャラリーモード

`/annotate?mode=gallery&frame=<split>/<name>` で開いたとき:

- 起動時に `/api/queue` を読まず、親（一覧）から `queue` メッセージで受け取った一覧（絞り込み・並び済み）を `S.items` にする。
  親がいない（URL を直接開いた）ときは 500 ms 待っても来なければ `/api/queue`（時系列）に落ちる。`frame` のフレームを開く。
- `←` `→`、Enter 後の次、s 後の次は、常に `S.items` の順で決める（今の「要確認のみ」のときの動きと同じ。一覧に無いフレームなら時系列の隣）。
- `f`（要確認のみ）は何もせず「一覧側で絞り込んでください」と出す。
- 右列の「キュー」には親から受け取った一覧を表示する（状態つき。今の表示と同じ）。上部バーの順序表示は「一覧の順」、位置は「n / N」。
- 上部バーに「一覧へ戻る（q）」ボタン。`q` キーでも戻る。画面下のキー説明をギャラリーモード用に書き換える（Enter: 保存して次へ、q: 一覧へ戻る）。
- 保存結果・状態変更・表示中フレームの変化を親に知らせる（5.6）。

### 5.6 一覧 ⇄ iframe のやり取り

同一オリジンの `postMessage`。

- 子 → 親 `{"type": "ready"}`: 読み込み完了。親はこれを受けて `queue` を送る。
- 親 → 子 `{"type": "queue", "items": [{"split","name","status"}...], "current": {"split","name"}}`。
- 子 → 親 `{"type": "saved", "key", "saved", "status", "flags", "cost", "classes", "rows", "thumb_version"}`: 保存のたび。
- 子 → 親 `{"type": "status", "key", "status"}`: skip など状態だけの変更。
- 子 → 親 `{"type": "current", "key"}`: 表示フレームが変わった。
- 子 → 親 `{"type": "close"}`: q / ボタン。親は iframe を隠して一覧にフォーカスを戻す。

親は `saved` / `status` を受けたカードの状態チップを書き換え、閉じたときにそのカードのサムネイルを新しい `v` で取り直す。
最後に表示していたフレームのカードを強調する（スクロールはしない。「もともと見ていた地点」を優先する）。
iframe を開いている間、一覧側のキー操作は無効（キーは iframe に向く。開いたとき iframe にフォーカスを与える）。

### 5.7 データ

読む: `images/`、`labels/`、`poses.json`、`annotation_state.json`、`gnss.json`、`cache/yolop/`、あれば `manifest.csv`、`relabeled_frames.json`。
書く: `labels/<split>/<name>.txt`（`store.save`）、`annotation_state.json`（status / edited / ts）、`poses.json`、`preview/<split>/<name>.jpg`（キャッシュ。消しても作り直される）。

### 5.8 起動

`annotate.py` の引数は変えない。ブラウザで開く URL を `/` にする。`start.sh`（キット）はそのままでよい。
README の「画面」「操作」「進め方」に一覧からの流れを足す。

## 6. テスト

pytest（`tests/conftest.py` の `StubProjector` / `FakeDetector` / `mini_root` を使う）:

- `GET /` が `gallery.html`、`GET /annotate` が `index.html` を返す。
- `GET /api/gallery`: 全フレーム、各項目、ラベルからの classes / rows、状態、`manifest.csv` と `relabeled_frames.json` の有無による `has_manifest` / `has_relabeled`。
- `POST /api/save`: 有効な姿勢 → ラベルが書かれ、状態 done、`edited` 真、`saved` 真、`poses.json` も更新。
  同じ内容のラベルをもう一度保存（Enter 相当）→ 状態 done のまま、`edited` は変わらない。
  無効な姿勢（`StubProjector` のレーン 2 = straight が引けない、または `valid_rows=5`）→ ラベルは前のまま、done なら unseen、`reasons` が非空、姿勢は保存される。
- `GET /api/thumb`: `image/jpeg` を返し、キャッシュファイルができる。`/api/save` 後にキャッシュが消え、次の要求で作り直される。
  `poses.json` にエントリの無いフレームのサムネイルを取っても `poses.json` は変わらず、照合（`_initialize`）が呼ばれない。
- ページ: 既存の `?selftest=N` の仕組みでギャラリーモードの自己テストを足す（`queue` メッセージの受け取り、← → が一覧の順で動く、q で `close` を送る、変更で `/api/save` が呼ばれる）。
  一覧ページはヘッドレス Chrome のスクリーンショットで確認する（README「動作確認」と同じやり方）。

## 7. 私（設計者）の判断

利用者の言葉から決めていない点は次のとおり決めた。読んで違えば変える。

- `/` を一覧、`/annotate` を従来の画面にする。
- 戻るキーは `q`（Esc は既存の「初期値へ戻す」）。ボタンも置く。
- Enter は「そのまま保存（done）して次へ」として残す。
- 保存できない姿勢（straight 無し・可視行不足・画像端）のときはラベルを書かず、理由を出して状態を unseen にする。
- 一覧の既定の並びは時系列。
- 一覧へ戻ったときはスクロール位置を動かさず、最後に見ていたカードを強調するだけ。
- ギャラリーモードでは `f` を無効にする。
- サムネイルは変更時に消して次の要求で描き直す（変更のたびには描かない）。描画は 1 つのロックで直列。
- サムネイルは姿勢を初期化しない（姿勢の無いフレームは GNSS 初期姿勢で描き「姿勢なし」と示す）。一覧を見るだけで `poses.json` が変わらないようにするため。
- `edited` は書くラベルが既存ファイルと異なるときだけ真にする。
- iframe + postMessage で合体する（1 ページへの統合はしない）。

## 8. 実装者に委ねること

- CSS、チップの色（今の `preview/index.html` の配色を流用してよい）、カードの細部。
- `thumb_version` の作り方、保存のまとめ方（プレビューと同じ pending / latest 方式でよい）。
- 親の有無の判定（`window.parent !== window`）、`ready` の待ち方。
- `/api/gallery` のメモリキャッシュの持ち方と保存時の更新。
- メッセージ文言。

## 9. 範囲外・今後

- キット（`annotator_kit`）への反映と zip の作り直し。
- 一覧での複数選択・一括 skip などの操作。
- 携帯画面向けのレイアウト。

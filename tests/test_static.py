"""静的ファイルの配信と、app.js がサーバ／geom.js と食い違っていないかの検証。

ブラウザ操作を伴う目視確認は自動化できないので、配線が合っているかだけを機械的に確かめる。
"""
import json
import re
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

from tools.path_annotator import server as server_mod

STATIC = Path(__file__).resolve().parents[1] / "static"


@pytest.fixture
def base_url(tmp_path):
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    xp = [0.0] * 48
    h = [0] * 48
    for i in range(25, 48):
        xp[i] = 0.5
        h[i] = 1
    (tmp_path / "labels" / "train" / "bag_a_000000.txt").write_text(
        json.dumps([{"class": "straight", "xp": xp, "h_vector": h}])
    )
    srv = server_mod.make_server(tmp_path, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return r.headers["Content-Type"], r.read()


def test_index_html_is_served_at_root(base_url):
    ctype, body = fetch(base_url + "/")
    assert ctype == "text/html"
    assert b"<canvas" in body


def test_static_assets_are_served_with_correct_types(base_url):
    for name, expected in [
        ("app.js", "text/javascript"),
        ("geom.js", "text/javascript"),
        ("style.css", "text/css"),
    ]:
        ctype, body = fetch(f"{base_url}/static/{name}")
        assert ctype == expected, name
        assert body, name


def test_app_js_has_no_syntax_errors():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    r = subprocess.run(
        ["node", "--check", str(STATIC / "app.js")], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


def test_index_html_references_only_existing_static_files():
    html = (STATIC / "index.html").read_text()
    for ref in re.findall(r'(?:src|href)="/static/([^"]+)"', html):
        assert (STATIC / ref).is_file(), ref


def test_app_js_calls_only_routes_the_server_implements():
    app = (STATIC / "app.js").read_text()
    used = set(re.findall(r"/api/[a-z_]+", app))
    server_src = (Path(__file__).resolve().parents[1] / "server.py").read_text()
    routed = set(re.findall(r'"api", "([a-z_]+)"', server_src))
    routed |= {m for m in re.findall(r'path == "/api/([a-z_]+)"', server_src)}
    for path in used:
        assert path.split("/")[-1] in routed, f"{path} is not routed by server.py"


def test_app_js_uses_only_functions_geom_exports():
    app = (STATIC / "app.js").read_text()
    geom = (STATIC / "geom.js").read_text()
    exported = set(re.findall(r"const api = \{([^}]*)\}", geom)[0].replace(" ", "").split(","))
    exported = {e for e in exported if e}
    for name in set(re.findall(r"\bG\.([A-Za-z_]\w*)", app)):
        assert name in exported, f"G.{name} is not exported by geom.js"


def test_every_entry_changing_handler_resyncs_control_rows():
    """エントリや h_vector を変える操作の後、制御点行をサーバから取り直していること。

    取り直しを忘れると、クライアントの deltas の添字とサーバが計算する制御点行がずれ、
    ドラッグが別の点を動かす。

    提案された検証は「キーごとに最初の分岐だけ」を見る形だったが、
    'u' は undo (!ev.ctrlKey) と redo (ev.ctrlKey) の 2 分岐があり、
    最初の分岐だけ見ると片方の取り直し漏れを見逃す。そのため、"if (k === " を
    区切りとして全分岐をブロック化し、キーに一致する分岐を「すべて」検査する形に
    強化した（redo 側だけ syncControlRows を外しても FAIL することをタスク実装時に
    手元で確認済み。詳細はタスクの実装報告を参照）。
    """
    app = (STATIC / "app.js").read_text()
    # 'n' 'x' 'u' の分岐は、CapsLock/修飾キー対応 (修正3・修正11) で
    # "k === " ではなく "lower === "（lower = ev.key.toLowerCase()）という形に
    # 変わったため、どの前置きでも分岐の境界として拾えるようにしてある。
    starts = [
        m.start() for m in re.finditer(
            r"if \((?:k|lower|ev\.key\.toLowerCase\(\)) === ", app
        )
    ] + [len(app)]
    blocks = [app[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    for key in ["'Tab'", "'n'", "'x'", "'u'"]:
        matches = [b for b in blocks if f"=== {key}" in b]
        assert matches, f"handler for {key} not found"
        for b in matches:
            assert "syncControlRows" in b, f"a handler for {key} does not resync control rows"


def test_regenerate_does_not_send_raw_xp_of_invalid_rows():
    """R（再生成）が cur().xp[r] を直接送らず、controlXs() 経由であること。

    修正11（CapsLock/修飾キー対応）で 'R' の判定が k === 'R' から
    lower === 'r'（lower = ev.key.toLowerCase()）に変わったため、
    検索文字列を実装に合わせて更新した。
    """
    app = (STATIC / "app.js").read_text()
    assert "function controlXs()" in app
    idx = app.find("=== 'r'")
    assert idx != -1
    nxt = app.find("if (k === ", idx + 10)
    block = app[idx: nxt if nxt != -1 else len(app)]
    assert "controlXs()" in block
    assert "cur().xp[r]" not in block


def test_propagate_uses_the_frame_load_baseline_not_the_saved_label():
    """伝播が「保存済みラベルとの差分」を送っていないこと。

    編集のたびに save() しているので、保存済みラベルとの差分は常に 0 になり、
    伝播が成功メッセージを出しながら何もしない no-op になる。

    基準は S.baseXp（S.label と同じ添字で並ぶ、エントリごとの xp の配列）であること。
    エントリ全体をコピーする S.baseLabel だと、x（エントリ削除）で添字がずれたときに
    無関係なエントリ同士の差分を近傍フレームへ書き込んでしまう。

    修正11（CapsLock/修飾キー対応）で 'p' 自身も、その次の分岐（'u'）も
    "if (k === " ではなく "if (lower === "（lower = ev.key.toLowerCase()）という
    形に変わったため、検索・境界検出をその形も拾えるようにしてある（これをしないと
    block が undo/redo を越えてファイル末尾の起動処理まで飲み込み、
    そこにある api.get( を誤検出する）。
    """
    app = (STATIC / "app.js").read_text()
    assert "S.baseXp" in app
    assert "S.baseLabel" not in app, "baseLabel (whole-entry copy) was replaced by baseXp"
    idx = app.find("=== 'p'")
    assert idx != -1
    nxt_candidates = [
        p for p in (
            app.find("if (k === ", idx + 10),
            app.find("if (lower === ", idx + 10),
            app.find("if (ev.key.toLowerCase() === ", idx + 10),
        ) if p != -1
    ]
    nxt = min(nxt_candidates) if nxt_candidates else -1
    block = app[idx: nxt if nxt != -1 else len(app)]
    assert "baseXp" in block, "p handler does not use the frame-load baseline"
    assert "api.get(" not in block, "p handler still re-fetches the saved label"


def test_redo_prevents_the_browser_view_source_shortcut():
    """Ctrl+u が preventDefault されていること（されないと view-source が開く）。

    修正3（CapsLock 対応）で比較が "k === 'u'" から "ev.key.toLowerCase() === 'u'"
    に変わったため、検索文字列も先頭の "k"/"ev.key.toLowerCase()" を問わない形にしてある。
    """
    app = (STATIC / "app.js").read_text()
    idx = app.find("=== 'u' && ev.ctrlKey")
    assert idx != -1
    nxt = app.find("if (k === ", idx + 10)
    block = app[idx: nxt if nxt != -1 else len(app)]
    assert "preventDefault" in block


def test_mouseup_settles_the_final_drag_position():
    """離した位置を確定させてから保存していること（表示とディスクの食い違いを防ぐ）。

    以前は固定 900 文字の窓を取っていたため、mouseup ハンドラの後に続く
    applyPreview() の本体まで拾ってしまい、"preview(" の検査が実質 mouseup に
    結びついていなかった。他のテストと同様、次の addEventListener 呼び出しか
    "// ---" 区切りコメントのどちらか近い方までを窓にする。
    """
    app = (STATIC / "app.js").read_text()
    idx = app.find("'mouseup'")
    assert idx != -1
    end_candidates = [
        e for e in (
            app.find("addEventListener(", idx + 10),
            app.find("// ---", idx + 10),
        ) if e != -1
    ]
    end = min(end_candidates) if end_candidates else len(app)
    block = app[idx:end]
    assert "lastX" in block
    assert "preview(" in block


def test_node_is_required_for_the_syntax_check():
    """node が無い環境では構文チェックを skip する（error にしない）。

    提示された検証は「このファイル全体に "shutil.which" という文字列があるか」を見る
    ものだったが、この assert 文自身の文字列リテラルの中に "shutil.which" が含まれる
    ため、ガードを消してもこのファイル自身の中に同じ文字列が残り続けてしまい、
    常に通ってしまうトートロジーだった（実際に試して確認: ガードを外しても
    1 passed のままだった）。そこで test_app_js_has_no_syntax_errors の関数本体
    「だけ」を切り出して検査する形に直した。
    """
    src = (Path(__file__).resolve().parent / "test_static.py").read_text()
    start = src.index("def test_app_js_has_no_syntax_errors")
    end = src.index("\ndef ", start + 1)
    block = src[start:end]
    assert "shutil.which" in block or "which(" in block


def test_entry_add_and_delete_keep_the_baseline_array_aligned():
    """S.baseXp が S.label と同じ配列操作を受けること。

    ずれると p が無関係なエントリとの差分を近傍フレームへ書き込む。

    修正11（CapsLock/修飾キー対応）で 'n'/'x' の判定が k === から
    lower ===（lower = ev.key.toLowerCase()）に変わったため、検索文字列を
    実装に合わせて更新した。
    """
    app = (STATIC / "app.js").read_text()
    for key, op in [("'n'", "S.baseXp.push"), ("'x'", "S.baseXp.splice")]:
        idx = app.find(f"=== {key}")
        assert idx != -1, key
        nxt_candidates = [
            p for p in (
                app.find("if (k === ", idx + 10),
                app.find("if (lower === ", idx + 10),
            ) if p != -1
        ]
        nxt = min(nxt_candidates) if nxt_candidates else -1
        block = app[idx: nxt if nxt != -1 else len(app)]
        assert op in block, f"{key} handler does not keep S.baseXp aligned ({op})"


def test_undo_snapshots_include_the_baseline():
    """undo/redo が label と baseXp をまとめて復元すること。"""
    app = (STATIC / "app.js").read_text()
    idx = app.find("function snapshot()")
    assert idx != -1
    block = app[idx: app.find("\n  }", idx)]
    assert "baseXp" in block


def test_propagate_advances_the_baseline_on_success():
    """伝播に成功したら基準を現在値に進めること（連打で二重加算しない）。

    epoch ガード（下の test_propagate_guards_the_baseline_advance_against_concurrent_changes）
    を導入した際に、代入先が S.baseXp[S.entry] から S.baseXp[entry]（await 前に
    固定した捕捉変数）へ変わったため、検索文字列を実装に合わせて更新した。

    修正11（CapsLock/修飾キー対応）で 'p' の判定が k === から lower ===
    （lower = ev.key.toLowerCase()）に変わったため、検索文字列も更新した。
    """
    app = (STATIC / "app.js").read_text()
    idx = app.find("=== 'p'")
    assert idx != -1
    nxt = app.find("if (k === ", idx + 10)
    block = app[idx: nxt if nxt != -1 else len(app)]
    assert "S.baseXp[entry] = xpNow" in block
    # 失敗時に進めていないこと: 代入が catch より前にあること
    assign = block.find("S.baseXp[entry] = ")
    catch = block.find("catch")
    assert assign != -1 and catch != -1 and assign < catch


def test_undo_and_redo_restore_the_baseline():
    """undo/redo が label だけでなく baseXp も復元すること。

    復元し忘れると、エントリの増減を含む操作を戻したときに baseXp と label の
    添字がずれ、p が無関係なエントリとの差分を近傍へ書き込む。

    修正3（CapsLock 対応）で比較が "k === 'u'" から "ev.key.toLowerCase() === 'u'"
    に変わったため、検索文字列も先頭の "k"/"ev.key.toLowerCase()" を問わない形にしてある。
    """
    app = (STATIC / "app.js").read_text()
    for key in ["=== 'u' && !ev.ctrlKey", "=== 'u' && ev.ctrlKey"]:
        idx = app.find(key)
        assert idx != -1, key
        nxt = app.find("if (k === ", idx + 10)
        block = app[idx: nxt if nxt != -1 else len(app)]
        assert "S.baseXp = snap.baseXp" in block, f"{key} does not restore baseXp"


def test_propagate_guards_the_baseline_advance_against_concurrent_changes():
    """基準前進が await 後の状態を読まず、構造が変わっていないときだけ書くこと。

    修正11（CapsLock/修飾キー対応）で 'p' の判定が k === から lower ===
    （lower = ev.key.toLowerCase()）に変わったため、検索文字列も更新した。
    """
    app = (STATIC / "app.js").read_text()
    idx = app.find("=== 'p'")
    assert idx != -1
    nxt = app.find("if (k === ", idx + 10)
    block = app[idx: nxt if nxt != -1 else len(app)]
    assert "const entry = S.entry" in block
    assert "const epoch = S.epoch" in block
    assert "S.epoch === epoch" in block
    assert "S.baseXp[entry] = xpNow" in block
    assert "S.baseXp[S.entry]" not in block, "still reads S.entry after the await"


def test_modifier_keys_do_not_trigger_edits():
    """Ctrl/Alt/Meta 付きのキーが編集操作を起こさないこと（Ctrl+P で伝播が走らない）。"""
    app = (STATIC / "app.js").read_text()
    idx = app.find("addEventListener('keydown'")
    assert idx != -1
    head = app[idx: idx + 600]
    assert "ev.metaKey" in head and "ev.altKey" in head
    assert "isRedo" in head or "toLowerCase()" in head


def test_api_get_checks_response_status():
    """api.get が r.ok を見ていること（見ないと失敗が無言のフリーズになる）。

    block を隣の "send:" プロパティの手前までに絞ってある。固定 400 文字の窓のままだと
    send 側の r.ok チェックまで拾ってしまい、get 自身のチェックを外す変異を見逃す
    （mutation 確認で実際に確認済み）。
    """
    app = (STATIC / "app.js").read_text()
    idx = app.find("get:")
    assert idx != -1
    end = app.find("send:", idx)
    block = app[idx: end if end != -1 else idx + 400]
    assert "r.ok" in block


def test_hud_uses_the_dataset_total_not_the_queue_length():
    app = (STATIC / "app.js").read_text()
    assert "S.total" in app


def test_redo_is_case_insensitive():
    """CapsLock ON や Ctrl+Shift+U でも redo が効くこと。"""
    app = (STATIC / "app.js").read_text()
    idx = app.find("ev.ctrlKey")
    assert idx != -1
    block = app[max(0, idx - 200): idx + 200]
    assert "toLowerCase()" in block


def test_preview_operations_are_serialized():
    """サーバ往復を伴う操作が直列化され、await 後に状態を照合していること。

    直列化しないと、キーリピートで 2 回目が 1 回目と同じ基準から計算し、
    後の書き込みが前の結果を打ち消して押下が失われる。
    """
    app = (STATIC / "app.js").read_text()
    assert "function enqueue(" in app
    for fn in ["function applyPreview(", "function syncControlRows("]:
        idx = app.find(fn)
        assert idx != -1, fn
        block = app[idx: idx + 700]
        assert "enqueue(" in block, f"{fn} does not go through the serialization chain"
    idx = app.find("function applyPreview(")
    block = app[idx: idx + 700]
    assert "S.epoch !== epoch" in block, "applyPreview does not re-check state after the await"


def test_rows_above_row_start_are_not_editable():
    app = (STATIC / "app.js").read_text()
    idx = app.find("'mousedown'")
    block = app[idx: idx + 900]
    assert "G.ROW_START" in block


def test_new_entry_picks_an_unused_class():
    app = (STATIC / "app.js").read_text()
    idx = app.find("lower === 'n'")
    assert idx != -1
    nxt = app.find("if (lower === ", idx + 10)
    block = app[idx: nxt if nxt != -1 else len(app)]
    assert "used" in block and "free" in block


def test_straight_cannot_be_deleted():
    app = (STATIC / "app.js").read_text()
    idx = app.find("lower === 'x'")
    assert idx != -1
    nxt = app.find("if (lower === ", idx + 10)
    block = app[idx: nxt if nxt != -1 else len(app)]
    assert "straight" in block


def test_pixel_convention_matches_the_trainer():
    """ラベル規約 u/(width-1) と行アンカー i*8+3.5 に合っていること。

    640 で割ると実データの preview と 1px ずれる（IoU 0.58 → 1.00）。
    """
    geom = (STATIC / "geom.js").read_text()
    assert "(IMG_W - 1)" in geom
    assert "(ROW_H - 1) / 2" in geom


def test_rendering_matches_the_dataset_preview():
    """geom.js の規約で描いた点が、データセットの preview 画像と完全一致すること。

    preview は converter.py の draw_preview が生成したもので、これが規約の権威。
    実データは読み取りのみで、一切書き換えない。
    """
    import json as _json

    import numpy as np
    import cv2

    root = Path(__file__).resolve().parents[3] / "dataset/real/dataset_real_20260909"
    name = "630_2ms_nav_000042"
    img_p = root / "images" / "train" / f"{name}.png"
    pv_p = root / "preview" / f"{name}.png"
    if not (img_p.is_file() and pv_p.is_file()):
        pytest.skip("実データが無い")

    img = cv2.imread(str(img_p))
    pv = cv2.imread(str(pv_p))
    label = _json.loads((root / "labels" / "train" / f"{name}.txt").read_text())

    pv_mask = cv2.absdiff(img, pv).max(axis=2) > 0
    drawn = img.copy()
    for entry in label:
        for row, (xp, ok) in enumerate(zip(entry["xp"], entry["h_vector"])):
            if not ok:
                continue
            u = int(xp * (640 - 1))              # geom.js の xpToX と同じ規約
            v = int(row * 8 + (8 - 1) / 2)       # geom.js の rowToY と同じ規約
            cv2.circle(drawn, (u, v), 3, (255, 255, 255), -1)
    mask = cv2.absdiff(img, drawn).max(axis=2) > 0

    assert (mask & pv_mask).sum() == (mask | pv_mask).sum(), "preview と描画位置が一致しない"


def test_shift_click_selects_the_interpolation_endpoint():
    app = (STATIC / "app.js").read_text()
    idx = app.find("'mousedown'")
    assert idx != -1
    nxt = app.find("addEventListener('mousemove'", idx)
    block = app[idx: nxt if nxt != -1 else idx + 1500]
    assert "shiftKey" in block
    assert "S.anchor" in block
    assert "S.mode" in block  # 終点の意味（補間／削除）はモードで決まる
    assert "'interp'" in block or '"interp"' in block
    assert "'erase'" in block or '"erase"' in block


def test_interpolation_math_is_not_duplicated_in_js():
    """補間の計算が JS に複製されていないこと（数学は Python 側のみ）。"""
    app = (STATIC / "app.js").read_text()
    idx = app.find("'mousedown'")
    nxt = app.find("addEventListener('mousemove'", idx)
    block = app[idx: nxt if nxt != -1 else idx + 1500]
    # 行の差で割る＝線形補間をクライアントで計算している兆候
    assert "row_b - row_a" not in block
    assert "/ (row" not in block


def test_mode_defaults_to_add_and_toggles_with_d():
    app = (STATIC / "app.js").read_text()
    assert "mode: 'add'" in app or 'mode: "add"' in app
    idx = app.find("lower === 'd'")
    assert idx != -1, "d キーのハンドラが無い"
    nxt = app.find("if (lower === ", idx + 10)
    block = app[idx: nxt if nxt != -1 else len(app)]
    assert "S.mode" in block
    assert "S.anchor" in block, "モード切替で anchor をクリアしていない"


def test_add_mode_never_erases_and_delete_mode_never_adds():
    """クリックの意味がモードで決まること（サーバのモード名で判定）。"""
    app = (STATIC / "app.js").read_text()
    idx = app.find("'mousedown'")
    nxt = app.find("addEventListener('mousemove'", idx)
    block = app[idx: nxt if nxt != -1 else idx + 2000]
    assert "'interp'" in block or '"interp"' in block
    assert "'erase'" in block or '"erase"' in block
    assert "S.mode" in block


def test_mode_badge_exists_and_is_updated():
    html = (STATIC / "index.html").read_text()
    css = (STATIC / "style.css").read_text()
    app = (STATIC / "app.js").read_text()
    assert 'id="mode-badge"' in html
    assert "#mode-badge" in css
    assert "mode-badge" in app


def test_row_toggle_goes_through_the_server():
    """行の追加削除をクライアント側で直接いじっていないこと（数学は Python 側）。"""
    app = (STATIC / "app.js").read_text()
    idx = app.find("'mousedown'")
    nxt = app.find("addEventListener('mousemove'", idx)
    block = app[idx: nxt if nxt != -1 else idx + 2000]
    assert "cur().h_vector[row] =" not in block, "クライアントで h_vector を直接書き換えている"


def test_anchor_is_tied_to_the_epoch():
    """始点が epoch に紐付き、構造が変わったら自動的に無効になること。

    個別にクリアして回ると n / x / undo / redo のどれかで漏れる。
    """
    app = (STATIC / "app.js").read_text()
    assert "function anchor()" in app
    idx = app.find("function anchor()")
    block = app[idx: idx + 300]
    assert "S.epoch" in block
    # 始点を読む側が S.anchor を直接見ていないこと
    md = app.find("'mousedown'")
    nxt = app.find("addEventListener('mousemove'", md)
    mdblock = app[md: nxt if nxt != -1 else md + 2000]
    assert "anchor()" in mdblock


def test_anchor_marker_uses_the_mode_colour():
    """始点マーカーがモードで色と形を変えること（バッジを見なくても分かるように）。"""
    app = (STATIC / "app.js").read_text()
    assert "MODE_COLORS" in app or "modeColor" in app
    idx = app.find("function draw()")
    assert idx != -1
    nxt = app.find("function refreshHud()")
    block = app[idx: nxt if nxt != -1 else idx + 2500]
    assert "anchor()" in block
    assert "S.mode" in block, "draw() が始点の描き分けにモードを見ていない"


def test_help_documents_when_the_anchor_is_cleared():
    html = (STATIC / "index.html").read_text()
    assert "始点" in html
    assert "クリア" in html or "解除" in html


def test_delete_mode_reports_when_the_row_has_no_point():
    app = (STATIC / "app.js").read_text()
    idx = app.find("'mousedown'")
    nxt = app.find("addEventListener('mousemove'", idx)
    block = app[idx: nxt if nxt != -1 else idx + 2000]
    assert "点はありません" in block


def test_propagate_skips_rows_whose_point_presence_changed():
    """点の有無が変わった行を伝播しないこと。

    配ると近傍フレームの同じ行が左端に張り付く、あるいは大きく飛ぶ。
    範囲補間・範囲削除で 23 行を一撃で作れるので被害が大きい。
    """
    app = (STATIC / "app.js").read_text()
    assert "S.baseH" in app
    idx = app.find("lower === 'p'")
    assert idx != -1
    nxt = app.find("if (lower === ", idx + 10)
    block = app[idx: nxt if nxt != -1 else len(app)]
    assert "baseH" in block, "p が h_vector の変化を見ていない"
    assert "=== baseH[i]" in block or "=== baseH [i]" in block


def test_base_h_is_kept_aligned_like_base_xp():
    """S.baseH が S.baseXp と同じ配列操作を受けること。"""
    app = (STATIC / "app.js").read_text()
    for key, ops in [("'n'", ["S.baseXp.push", "S.baseH.push"]),
                     ("'x'", ["S.baseXp.splice", "S.baseH.splice"])]:
        idx = app.find(f"lower === {key}")
        assert idx != -1, key
        nxt = app.find("if (lower === ", idx + 10)
        block = app[idx: nxt if nxt != -1 else len(app)]
        for op in ops:
            assert op in block, f"{key} handler is missing {op}"
    idx = app.find("function snapshot()")
    assert idx != -1
    block = app[idx: app.find("\n  }", idx)]
    assert "baseH" in block, "snapshot が baseH を含んでいない"


def test_annotate_requires_root():
    """--root を省略したら usage を出して終了コード 2 になること。"""
    import subprocess, sys as _sys
    from pathlib import Path as _Path
    script = _Path(__file__).resolve().parents[1] / "annotate.py"
    r = subprocess.run([_sys.executable, str(script), "--no-browser"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 2
    assert "--root" in r.stderr


def test_no_absolute_paths_outside_tests():
    """tools/path_annotator/ 配下（tests を除く）に環境依存の絶対パスが無いこと。"""
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parents[1]
    bad = []
    for p in root.rglob("*"):
        if not p.is_file() or "tests" in p.parts or "__pycache__" in p.parts:
            continue
        if p.suffix not in (".py", ".md", ".js", ".html", ".css"):
            continue
        if "/home/" in p.read_text(errors="ignore"):
            bad.append(str(p.relative_to(root)))
    assert bad == [], f"絶対パスが残っている: {bad}"

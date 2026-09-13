import json
import threading
import urllib.request
import urllib.error

import pytest

from tools.path_annotator import server as server_mod


def make_entry(cls="straight", first=25, last=47, value=0.5):
    xp = [0.0] * 48
    h = [0] * 48
    for i in range(first, last + 1):
        xp[i] = value
        h[i] = 1
    return {"class": cls, "xp": xp, "h_vector": h}


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    for i in range(4):
        name = f"bag_a_{i * 2:06d}"
        (tmp_path / "labels" / "train" / f"{name}.txt").write_text(
            json.dumps([make_entry(value=0.5)])
        )
        (tmp_path / "images" / "train" / f"{name}.png").write_bytes(b"\x89PNG-fake")
    return tmp_path


@pytest.fixture
def base_url(dataset):
    srv = server_mod.make_server(dataset, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def send_json(url, payload, method):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_queue_returns_all_frames(base_url):
    body = get_json(base_url + "/api/queue")
    assert body["total"] == 4
    assert len(body["items"]) == 4


def test_queue_items_are_sorted_by_score_desc(base_url):
    items = get_json(base_url + "/api/queue")["items"]
    scores = [i["score"] for i in items]
    assert scores == sorted(scores, reverse=True)


def test_queue_respects_limit(base_url):
    body = get_json(base_url + "/api/queue?limit=2")
    assert len(body["items"]) == 2
    assert body["total"] == 4


def test_frame_returns_label_and_neighbours(base_url):
    body = get_json(base_url + "/api/frame/train/bag_a_000002")
    assert len(body["label"][0]["xp"]) == 48
    assert body["prev"] == "bag_a_000000"
    assert body["next"] == "bag_a_000004"


def test_first_frame_has_no_prev(base_url):
    assert get_json(base_url + "/api/frame/train/bag_a_000000")["prev"] is None


def test_image_is_served_as_png(base_url):
    with urllib.request.urlopen(base_url + "/api/image/train/bag_a_000000") as r:
        assert r.headers["Content-Type"] == "image/png"
        assert r.read() == b"\x89PNG-fake"


def test_put_frame_persists_label(base_url, dataset):
    label = get_json(base_url + "/api/frame/train/bag_a_000000")["label"]
    label[0]["xp"][30] = 0.321
    send_json(base_url + "/api/frame/train/bag_a_000000", {"label": label}, "PUT")
    saved = json.loads((dataset / "labels" / "train" / "bag_a_000000.txt").read_text())
    assert saved[0]["xp"][30] == 0.321


def test_put_frame_rejects_invalid_label(base_url):
    bad = [make_entry("diagonal")]
    with pytest.raises(urllib.error.HTTPError) as e:
        send_json(base_url + "/api/frame/train/bag_a_000000", {"label": bad}, "PUT")
    assert e.value.code == 400


def test_put_frame_rejects_missing_label_key(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        send_json(base_url + "/api/frame/train/bag_a_000000", {}, "PUT")
    assert e.value.code == 400


def test_preview_control_mode_translates_curve(base_url):
    e = make_entry(value=0.5)
    body = send_json(
        base_url + "/api/preview",
        {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "control", "deltas": [0.1, 0.1, 0.1]},
        "POST",
    )
    assert body["xp"][30] == pytest.approx(0.6, abs=1e-9)


def test_preview_tilt_mode_keeps_bottom_fixed(base_url):
    e = make_entry(value=0.5)
    body = send_json(
        base_url + "/api/preview",
        {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "tilt", "deltas": [0.2]},
        "POST",
    )
    assert body["xp"][47] == pytest.approx(0.5, abs=1e-9)


def test_preview_returns_control_rows(base_url):
    e = make_entry(value=0.5)
    body = send_json(
        base_url + "/api/preview",
        {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "control", "deltas": [0.0, 0.0, 0.0]},
        "POST",
    )
    assert body["control_rows"] == [25, 36, 47]


def test_preview_rejects_unknown_mode(base_url):
    e = make_entry()
    with pytest.raises(urllib.error.HTTPError) as exc:
        send_json(
            base_url + "/api/preview",
            {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "wobble", "deltas": [0.1]},
            "POST",
        )
    assert exc.value.code == 400


def test_preview_rejects_empty_deltas(base_url):
    e = make_entry()
    with pytest.raises(urllib.error.HTTPError) as exc:
        send_json(
            base_url + "/api/preview",
            {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "control", "deltas": []},
            "POST",
        )
    assert exc.value.code == 400


def test_propagate_applies_delta_to_neighbours(base_url, dataset):
    delta = [0.0] * 48
    for i in range(25, 48):
        delta[i] = 0.05
    body = send_json(
        base_url + "/api/propagate",
        {"split": "train", "name": "bag_a_000002", "entry_index": 0, "delta": delta, "range_n": 1},
        "POST",
    )
    assert sorted(body["applied"]) == ["bag_a_000000", "bag_a_000004"]
    saved = json.loads((dataset / "labels" / "train" / "bag_a_000000.txt").read_text())
    assert saved[0]["xp"][30] == pytest.approx(0.55, abs=1e-9)


def test_propagate_does_not_cross_bags(tmp_path):
    # bag をまたがないことを見たいので、2 つの bag を持つデータセットを
    # サーバ起動前に作っておく（Annotator は起動時に bag 構成をキャッシュする）。
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    for name in ("bag_a_000000", "bag_a_000002", "bag_z_000000"):
        (tmp_path / "labels" / "train" / f"{name}.txt").write_text(
            json.dumps([make_entry(value=0.5)])
        )
    srv = server_mod.make_server(tmp_path, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        body = send_json(
            url + "/api/propagate",
            {"split": "train", "name": "bag_a_000000", "entry_index": 0,
             "delta": [0.05] * 48, "range_n": 5},
            "POST",
        )
    finally:
        srv.shutdown()
        srv.server_close()
    assert body["applied"] == ["bag_a_000002"]
    assert "bag_z_000000" not in body["applied"]


def test_neighbors_respect_the_frame_number_gap(tmp_path):
    """連番が飛んでいる箇所では伝播しないこと。"""
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    for num in (0, 2, 100):          # 2 と 100 の間は 98 空いている
        (tmp_path / "labels" / "train" / f"bag_a_{num:06d}.txt").write_text(
            json.dumps([make_entry(value=0.5)])
        )
    srv = server_mod.make_server(tmp_path, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        body = send_json(
            url + "/api/propagate",
            {"split": "train", "name": "bag_a_000002", "entry_index": 0,
             "delta": [0.05] * 48, "range_n": 1},
            "POST",
        )
    finally:
        srv.shutdown()
        srv.server_close()
    assert body["applied"] == ["bag_a_000000"]
    assert "bag_a_000100" not in body["applied"]


def test_status_is_persisted(base_url, dataset):
    send_json(
        base_url + "/api/status",
        {"split": "train", "name": "bag_a_000000", "status": "done"},
        "POST",
    )
    state = json.loads((dataset / "annotation_state.json").read_text())
    assert state["frames"]["train/bag_a_000000"]["status"] == "done"


def test_backup_is_created_on_startup(base_url, dataset):
    assert (dataset / "labels_backup" / "train" / "bag_a_000000.txt").exists()


def test_unknown_path_returns_404(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        get_json(base_url + "/api/nope")
    assert e.value.code == 404


# --- レビュー Important 指摘への回帰テスト ---
# 生ソケットでの実測により、以下はすべて「レスポンス無しで切断」
# （do_POST/do_PUT が TypeError/IndexError を捕捉していないため）に
# なることを確認済み。ハンドラに受け皿を足したことで HTTPError になる。


def test_propagate_rejects_wrong_length_delta(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        send_json(
            base_url + "/api/propagate",
            {"split": "train", "name": "bag_a_000002", "entry_index": 0,
             "delta": [0.05] * 10, "range_n": 1},
            "POST",
        )
    assert e.value.code in (400, 500)


def test_propagate_rejects_non_integer_entry_index(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        send_json(
            base_url + "/api/propagate",
            {"split": "train", "name": "bag_a_000002", "entry_index": None,
             "delta": [0.0] * 48, "range_n": 1},
            "POST",
        )
    assert e.value.code in (400, 500)


def test_preview_rejects_array_body(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        send_json(base_url + "/api/preview", ["not", "an", "object"], "POST")
    assert e.value.code in (400, 500)


def test_put_frame_rejects_array_body(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        send_json(base_url + "/api/frame/train/bag_a_000000", ["nope"], "PUT")
    assert e.value.code in (400, 500)


def test_queue_survives_a_structurally_broken_label(tmp_path):
    """壊れたラベルが 1 件あってもキューは返り、その1件が先頭に出ること。"""
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    for i in range(4):
        (tmp_path / "labels" / "train" / f"bag_a_{i * 2:06d}.txt").write_text(
            json.dumps([make_entry(value=0.5)])
        )
    broken = make_entry()
    broken["xp"] = [0.5] * 10  # 48 要素でない
    (tmp_path / "labels" / "train" / "bag_a_000008.txt").write_text(json.dumps([broken]))

    srv = server_mod.make_server(tmp_path, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        body = get_json(url + "/api/queue")
    finally:
        srv.shutdown()
        srv.server_close()

    assert body["total"] == 5
    assert body["items"][0]["name"] == "bag_a_000008"
    assert "unreadable" in body["items"][0]["reasons"]


def test_propagate_skips_neighbours_with_no_valid_rows(tmp_path):
    """有効行の無い近傍はスキップし、edited を立てないこと。"""
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "bag_a_000000.txt").write_text(
        json.dumps([make_entry(value=0.5)])
    )
    empty = make_entry()
    empty["h_vector"] = [0] * 48
    empty["xp"] = [0.0] * 48
    (tmp_path / "labels" / "train" / "bag_a_000002.txt").write_text(json.dumps([empty]))

    srv = server_mod.make_server(tmp_path, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        body = send_json(
            url + "/api/propagate",
            {"split": "train", "name": "bag_a_000000", "entry_index": 0,
             "delta": [0.05] * 48, "range_n": 1},
            "POST",
        )
    finally:
        srv.shutdown()
        srv.server_close()

    assert body["applied"] == []
    assert [s["name"] for s in body["skipped"]] == ["bag_a_000002"]
    state_path = tmp_path / "annotation_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        assert "train/bag_a_000002" not in state["frames"]


def test_queue_is_rescored_after_a_put(base_url, dataset):
    """保存するとそのフレームのスコアが再計算されること。"""
    before = get_json(base_url + "/api/queue")["items"]
    target = next(i for i in before if i["name"] == "bag_a_000000")
    label = get_json(base_url + "/api/frame/train/bag_a_000000")["label"]
    for i in range(25, 48):                 # 極端にギザギザにする
        label[0]["xp"][i] = 0.2 if i % 2 else 0.8
    send_json(base_url + "/api/frame/train/bag_a_000000", {"label": label}, "PUT")
    after = get_json(base_url + "/api/queue")["items"]
    updated = next(i for i in after if i["name"] == "bag_a_000000")
    assert updated["score"] != target["score"]
    assert "bend" in updated["reasons"]


def test_frame_of_a_broken_label_does_not_drop_the_connection(tmp_path):
    """壊れたラベルのフレームを開いてもコネクションが切れず、エラー応答が返ること。"""
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train").mkdir(parents=True)
    broken = make_entry()
    broken["xp"] = [0.5] * 10
    (tmp_path / "labels" / "train" / "bag_a_000000.txt").write_text(json.dumps([broken]))

    srv = server_mod.make_server(tmp_path, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            get_json(url + "/api/frame/train/bag_a_000000")
    finally:
        srv.shutdown()
        srv.server_close()
    assert e.value.code in (400, 500)  # 切断ではなく HTTP エラーが返ること


def test_preview_interp_fills_the_gap(base_url):
    e = make_entry(value=0.5)
    for i in range(30, 36):
        e["xp"][i] = 0.0
        e["h_vector"][i] = 0
    body = send_json(
        base_url + "/api/preview",
        {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "interp",
         "points": [[29, 0.2], [36, 0.9]]},
        "POST",
    )
    assert all(body["h_vector"][i] == 1 for i in range(29, 37))
    assert body["xp"][32] == pytest.approx(0.2 + 0.7 * 3 / 7, abs=1e-9)
    assert "control_rows" in body


def test_preview_interp_rejects_rows_above_row_start(base_url):
    e = make_entry()
    with pytest.raises(urllib.error.HTTPError) as exc:
        send_json(
            base_url + "/api/preview",
            {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "interp",
             "points": [[10, 0.2], [36, 0.9]]},
            "POST",
        )
    assert exc.value.code == 400


def test_preview_interp_rejects_malformed_points(base_url):
    e = make_entry()
    with pytest.raises(urllib.error.HTTPError) as exc:
        send_json(
            base_url + "/api/preview",
            {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "interp", "points": [[29, 0.2]]},
            "POST",
        )
    assert exc.value.code == 400


def test_preview_other_modes_still_return_h_vector(base_url):
    e = make_entry(value=0.5)
    body = send_json(
        base_url + "/api/preview",
        {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "control", "deltas": [0.1, 0.1, 0.1]},
        "POST",
    )
    assert body["h_vector"] == e["h_vector"]


def test_preview_erase_clears_the_range(base_url):
    e = make_entry(value=0.5)
    body = send_json(
        base_url + "/api/preview",
        {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "erase",
         "points": [[30, 0.0], [35, 0.0]]},
        "POST",
    )
    assert all(body["h_vector"][i] == 0 for i in range(30, 36))
    assert all(body["xp"][i] == 0.0 for i in range(30, 36))
    assert body["h_vector"][29] == 1


def test_preview_erase_rejects_rows_above_row_start(base_url):
    e = make_entry()
    with pytest.raises(urllib.error.HTTPError) as exc:
        send_json(
            base_url + "/api/preview",
            {"xp": e["xp"], "h_vector": e["h_vector"], "mode": "erase",
             "points": [[10, 0.0], [35, 0.0]]},
            "POST",
        )
    assert exc.value.code == 400


def test_preview_rejects_rows_beyond_the_last(base_url):
    e = make_entry()
    for mode in ("interp", "erase"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            send_json(
                base_url + "/api/preview",
                {"xp": e["xp"], "h_vector": e["h_vector"], "mode": mode,
                 "points": [[30, 0.5], [99, 0.5]]},
                "POST",
            )
        assert exc.value.code == 400, mode

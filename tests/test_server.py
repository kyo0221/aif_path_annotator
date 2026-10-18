import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from tools.path_annotator.server import make_server

POSE = {"x": 5.0, "y": 0.3, "yaw": 0.0, "pitch": 0.0}


def _serve(annotator):
    srv = make_server(annotator, 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t


def _stop(srv, t):
    srv.shutdown()
    srv.server_close()
    t.join(timeout=5)


@pytest.fixture
def server(annotator):
    srv, t = _serve(annotator)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    _stop(srv, t)


@pytest.fixture
def server_few_rows(annotator_few_rows):
    srv, t = _serve(annotator_few_rows)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    _stop(srv, t)


def call(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            status, raw, ctype = r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        status, raw, ctype = e.code, e.read(), e.headers.get("Content-Type")
    if ctype and ctype.startswith("application/json"):
        return status, json.loads(raw)
    return status, raw


def test_queue_and_frame(server):
    status, q = call(server, "GET", "/api/queue")
    assert status == 200 and q["total"] == 3
    assert q["items"][0]["name"] == "f000000"
    status, page = call(server, "GET", "/api/queue?offset=1&limit=1")
    assert page["total"] == 3 and len(page["items"]) == 1
    status, f = call(server, "GET", "/api/frame/train/f000000")
    assert status == 200 and "overlay" in f and "candidates" in f


def test_preview_and_confirm(server, mini_root):
    body = {"split": "train", "name": "f000002", "pose": POSE}
    status, p = call(server, "POST", "/api/preview", body)
    assert status == 200 and "cost" in p
    status, c = call(server, "POST", "/api/confirm", dict(body, source="manual"))
    assert status == 200 and c["ok"] is True and "next" in c
    assert (mini_root / "labels" / "train" / "f000002.txt").is_file()


def test_match_pose_status(server):
    base = {"split": "train", "name": "f000000", "pose": POSE}
    status, m = call(server, "POST", "/api/match", dict(base, window=1.0))
    assert status == 200 and "cost" in m
    status, r = call(server, "POST", "/api/pose", dict(base, source="manual"))
    assert (status, r) == (200, {"ok": True})
    status, r = call(server, "POST", "/api/status",
                     {"split": "train", "name": "f000000", "status": "done"})
    assert (status, r) == (200, {"ok": True})


def test_confirm_rejection_is_400(server_few_rows):
    body = {"split": "train", "name": "f000002", "pose": POSE, "source": "manual"}
    status, r = call(server_few_rows, "POST", "/api/confirm", body)
    assert status == 400 and "可視行" in r["error"]


def test_mask_and_image_png(server):
    sig = b"\x89PNG\r\n\x1a\n"
    for kind in ("mask", "image"):
        with urllib.request.urlopen(f"{server}/api/{kind}/train/f000000") as r:
            assert r.status == 200 and r.headers["Content-Type"] == "image/png"
            data = r.read()
        assert data[:8] == sig
    status, raw = call(server, "GET", "/api/mask/train/f000000")
    m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(m)) == {0, 255}


def test_not_found_and_bad_request(server):
    assert call(server, "GET", "/api/image/train/nope")[0] == 404
    assert call(server, "GET", "/api/nothing") == (404, {"error": "not found"})
    assert call(server, "POST", "/api/nothing", {"a": 1}) == (404, {"error": "not found"})
    status, r = call(server, "POST", "/api/preview", {"split": "train"})
    assert status == 400 and r["error"].startswith("bad request")
    assert call(server, "GET", "/")[0] == 200


def test_unknown_frame_is_400(mini_root, make_annotator):
    cv2.imwrite(str(mini_root / "images" / "val" / "f000099.png"),
                np.zeros((384, 640, 3), np.uint8))
    srv, t = _serve(make_annotator())
    try:
        status, r = call(f"http://127.0.0.1:{srv.server_address[1]}",
                         "GET", "/api/frame/val/f000099")
    finally:
        _stop(srv, t)
    assert status == 400 and "error" in r

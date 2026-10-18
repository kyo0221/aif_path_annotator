"""アノテーションツールの HTTP API。標準ライブラリのみで動く（マスクの PNG 化に cv2 を使う）。

ここはルーティングだけを担い、処理は app.Annotator に委ねる。
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        # --- 送信ヘルパ ---

        def _send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _error(self, code, message):
            self._json({"error": message}, code)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def _drain(self):
            """ボディを読み捨てる。

            早期 404 の前に呼ぶ。読まずに応答を返すと、HTTP/1.1 keep-alive の
            同一コネクション上でボディの残骸が次のリクエストの先頭行として
            誤解釈される（ブラウザ側で実測済み。urllib は毎回 Connection: close
            を付けるため露見しない）。
            """
            n = int(self.headers.get("Content-Length", 0))
            if n:
                self.rfile.read(n)

        def _guard(self, fn):
            """fn() の例外を HTTP エラーへ写像する。fn は応答済みなら True、未処理なら False を返す。"""
            try:
                if fn():
                    return
            except ValueError as e:
                return self._error(400, str(e))
            except (KeyError, TypeError) as e:
                return self._error(400, f"bad request: {e}")
            except FileNotFoundError:
                return self._error(404, "not found")
            except Exception:  # noqa: BLE001
                # 未知の例外でコネクションを無言で切らない。socketserver の既定は
                # traceback を stderr に出して切断するだけで、クライアントには
                # 何も届かない。
                return self._error(500, "internal error")
            return self._error(404, "not found")

        # --- ルーティング ---

        def do_GET(self):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]

            def route():
                if u.path == "/":
                    return self._static("index.html")
                if parts[:1] == ["static"] and len(parts) == 2:
                    return self._static(parts[1])
                if parts[:2] == ["api", "queue"] and len(parts) == 2:
                    return self._queue(parse_qs(u.query))
                if len(parts) == 4 and parts[:2] == ["api", "frame"]:
                    return self._json(app.open_frame(parts[2], parts[3])) or True
                if len(parts) == 4 and parts[:2] == ["api", "image"]:
                    return self._image(parts[2], parts[3])
                if len(parts) == 4 and parts[:2] == ["api", "mask"]:
                    return self._mask(parts[2], parts[3])
                return False

            self._guard(route)

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/api/preview", "/api/match", "/api/pose",
                            "/api/confirm", "/api/status"):
                self._drain()
                return self._error(404, "not found")

            def route():
                p = self._body()
                split, name = p["split"], p["name"]
                if path == "/api/preview":
                    out = app.preview(split, name, p["pose"])
                elif path == "/api/match":
                    out = app.match(split, name, p["pose"], p["window"])
                elif path == "/api/pose":
                    app.set_pose(split, name, p["pose"], p["source"])
                    out = {"ok": True}
                elif path == "/api/confirm":
                    out = app.confirm(split, name, p["pose"], p["source"])
                else:
                    app.set_status(split, name, p["status"])
                    out = {"ok": True}
                self._json(out)
                return True

            self._guard(route)

        # --- 各ハンドラ ---

        def _static(self, filename):
            path = STATIC_DIR / filename
            if not path.is_file() or path.parent != STATIC_DIR:
                return False
            ctype = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
            self._send(200, path.read_bytes(), ctype)
            return True

        def _queue(self, q):
            def one(key):
                return q.get(key, [""])[0] or None

            items = app.queue(order=one("order") or "time",
                              status=one("status"), flag=one("flag"))
            total = len(items)
            offset, limit = one("offset"), one("limit")
            if offset is not None:
                items = items[int(offset):]
            if limit is not None:
                items = items[:int(limit)]
            self._json({"items": items, "total": total})
            return True

        def _image(self, split, name):
            path = app.store.image_path(split, name)
            if not path.is_file():
                return False
            self._send(200, path.read_bytes(), "image/png")
            return True

        def _mask(self, split, name):
            path = app.store.image_path(split, name)
            if not path.is_file():
                return False
            mask = app.detector.mask(path, f"{split}/{name}")
            ok, buf = cv2.imencode(".png", mask * 255)
            if not ok:
                raise RuntimeError("PNG encode failed")
            self._send(200, buf.tobytes(), "image/png")
            return True

    return Handler


def make_server(annotator, port=8765):
    """127.0.0.1 にだけバインドしたサーバを返す（serve_forever は呼び出し側で）。"""
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(annotator))

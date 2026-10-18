"""アノテーションツールの HTTP API。標準ライブラリのみで動く。

推論（tools.path_annotator.infer）は torch と別リポジトリの学習コードに依存するため、
ここでは無条件 import しない。inferencer を使う経路（キューの乖離順並べ替え・
/api/autolabel）でだけ、呼び出し側から渡された Inferencer インスタンスを使う。
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from tools.path_annotator import curve, detect
from tools.path_annotator.store import LabelStore, validate_label

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}

# curve.ROW_START が正本（store.py の保存時検証とも共有する）。ここでは別名で束ねるだけ。
ROW_START = curve.ROW_START


class Annotator:
    """ストアとキュー（スコア順フレーム一覧）を束ねる。"""

    def __init__(self, root, inferencer=None):
        self.store = LabelStore(root)
        self.store.ensure_backup()
        self.inferencer = inferencer
        self._queue = None
        self._bags = {}  # split -> (bag名 -> 連番順のフレーム名リスト)

    def queue(self, refresh=False):
        """フレームの一覧をキュー順（悪い順）で返す。

        inferencer が無ければ従来どおり detect.score_dataset（異常スコア）。
        inferencer があれば全フレームを推論し、既存ラベルとの乖離（px）降順で
        並べる。異常スコアは自己位置推定が正しい前提の指標で、ドリフト箇所では
        「滑らかで自信を持って間違っているラベル」を見逃すため、重みがある場合は
        使わない。
        """
        if self._queue is None or refresh:
            if self.inferencer is not None:
                self._queue = self._queue_by_divergence()
            else:
                self._queue = detect.score_dataset(self.store)
        return self._queue

    def _queue_by_divergence(self):
        """全フレームを推論し、既存ラベルとの乖離（px）降順で返す。"""
        items = []  # (split, name)
        for split in self.store.splits():
            for name in self.store.frames(split):
                items.append((split, name))
        if not items:
            return []
        image_paths = [self.store.image_path(split, name) for split, name in items]
        predictions = self.inferencer.predict_batch(image_paths)
        out = []
        for (split, name), predicted in zip(items, predictions):
            try:
                existing = self.store.load(split, name)
                score = detect.label_divergence(existing, predicted)
                reasons = ["divergence"]
            except Exception:  # noqa: BLE001
                # 読めないラベルは乖離を計算できないが、queue の目的は壊れた
                # ラベルを人に見せることなので、キュー全体を落とさず最上位に出す。
                score = detect.MAX_DIVERGENCE_PX
                reasons = [detect.UNREADABLE_REASON, "divergence"]
            out.append(
                {
                    "split": split,
                    "name": name,
                    "score": score,
                    "reasons": reasons,
                    "metrics": {"divergence": score},
                }
            )
        out.sort(key=lambda r: r["score"], reverse=True)
        return out

    def autolabel(self, split, name):
        """このフレームを推論し、保存はせず推論ラベルを返す。"""
        if self.inferencer is None:
            raise ValueError("重みファイルが指定されていません")
        return self.inferencer.predict(self.store.image_path(split, name))

    def bags(self, split):
        """split の bag グルーピング。フレーム表示のたび 2 万件を走査しないようキャッシュする。"""
        if split not in self._bags:
            groups, _ = detect.group_by_bag(self.store.frames(split))
            self._bags[split] = groups
        return self._bags[split]

    def neighbors(self, split, name, n):
        """同一 bag で、フレーム一覧上の距離が n 以内かつ連番差が MAX_NUM_GAP 以内の名前を返す。

        連番差を見ないと、bag 内で撮り逃しがある箇所（実データに 142 箇所、最大 22）で
        時間的に断絶したフレームまで伝播してしまう。`MAX_NUM_GAP * n` としているのは、
        n フレームぶんの窓なら連番差も n 倍まで許すという意味（実データは 2 刻みなので
        n=5 なら連番差 20 まで）。
        """
        try:
            prefix, num = detect.parse_frame_name(name)
        except ValueError:
            return []
        same_bag = self.bags(split).get(prefix, [])
        if name not in same_bag:
            return []
        i = same_bag.index(name)
        lo, hi = max(0, i - n), min(len(same_bag), i + n + 1)
        out = []
        for m in same_bag[lo:hi]:
            if m == name:
                continue
            if abs(detect.parse_frame_name(m)[1] - num) <= detect.MAX_NUM_GAP * n:
                out.append(m)
        return out

    def adjacent(self, split, name):
        """フレーム一覧上のひとつ前・ひとつ後（bag をまたがない）。"""
        near = self.neighbors(split, name, 1)
        prev = next((m for m in near if m < name), None)
        nxt = next((m for m in near if m > name), None)
        return prev, nxt

    def propagate(self, split, name, entry_index, delta, range_n):
        """delta を同一 bag の前後 range_n フレームへ適用する。

        近傍が壊れていたり、そのエントリに有効行が無い場合はスキップする。
        壊れたラベルはキューの先頭に出す設計なので、伝播の巻き添えで
        さらに壊すより、飛ばして結果を返す方がよい。
        """
        applied = []
        skipped = []
        d = np.asarray(delta, dtype=float)
        for m in self.neighbors(split, name, range_n):
            try:
                label = self.store.load(split, m)
                validate_label(label)
            except Exception:  # noqa: BLE001
                skipped.append({"name": m, "reason": "unreadable"})
                continue
            if entry_index >= len(label):
                skipped.append({"name": m, "reason": "no such entry"})
                continue
            entry = label[entry_index]
            valid = np.asarray(entry["h_vector"], dtype=bool)
            if not valid.any():
                skipped.append({"name": m, "reason": "no valid rows"})
                continue
            xp = np.asarray(entry["xp"], dtype=float)
            xp[valid] = np.clip(xp[valid] + d[valid], 0.0, 1.0)
            entry["xp"] = xp.tolist()
            try:
                self.store.save(split, m, label)
            except Exception as e:  # noqa: BLE001
                skipped.append({"name": m, "reason": str(e)})
                continue
            applied.append(m)
        return applied, skipped

    def rescore(self, split, names):
        """指定フレームとその近傍のスコアを計算し直してキャッシュを更新する。

        1 件直すと近傍の jump が変わるので、近傍も巻き込んで計算し直す。
        全件（22,460 件・1.5 秒）を毎回走らせる必要はない。
        """
        if self._queue is None:
            return
        targets = set()
        for n in names:
            targets.add(n)
            targets.update(self.neighbors(split, n, detect.MAX_NUM_GAP))
        if not targets:
            return
        fresh = {}
        for n in targets:
            try:
                label = self.store.load(split, n)
                metrics = detect.frame_metrics(label)
                neighbors = []
                for m in self.neighbors(split, n, 2):
                    try:
                        neighbors.append(self.store.load(split, m))
                    except Exception:  # noqa: BLE001
                        pass
                metrics[detect.TEMPORAL_KEY] = detect.temporal_metric(label, neighbors)
                fresh[n] = {"metrics": metrics, "unreadable": False}
            except Exception:  # noqa: BLE001
                fresh[n] = {"metrics": detect._unreadable_metrics(), "unreadable": True}
        records = []
        for item in self._queue:
            if item["split"] == split and item["name"] in fresh:
                f = fresh[item["name"]]
                records.append({"split": split, "name": item["name"],
                                "metrics": f["metrics"], "unreadable": f["unreadable"]})
            else:
                records.append({"split": item["split"], "name": item["name"],
                                "metrics": item["metrics"],
                                "unreadable": "unreadable" in item.get("reasons", [])})
        self._queue = detect.score_frames(records)


def _preview(payload):
    """(xp, h_vector, control_rows) を返す。interp/erase 以外は入力の h_vector をそのまま返す。"""
    xp = payload["xp"]
    h = payload["h_vector"]
    mode = payload.get("mode")
    deltas = payload.get("deltas", [])
    if mode == "interp":
        points = payload.get("points")
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError("interp mode needs points = [[row_a, xp_a], [row_b, xp_b]]")
        (row_a, xp_a), (row_b, xp_b) = points
        row_a, row_b = int(row_a), int(row_b)
        if not (ROW_START <= min(row_a, row_b) and max(row_a, row_b) < curve.N_ROWS):
            raise ValueError(f"行は {ROW_START}〜{curve.N_ROWS - 1} の範囲で指定してください")
        new_xp, new_h = curve.interpolate_rows(xp, h, row_a, float(xp_a), row_b, float(xp_b))
        return new_xp, new_h, curve.control_rows(new_h, len(deltas) if deltas else 3)
    if mode == "erase":
        # points の形は interp と共有しているが、erase は行番号しか使わない（xp は無視）。
        points = payload.get("points")
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError("erase mode needs points = [[row_a, xp_a], [row_b, xp_b]]")
        row_a, row_b = int(points[0][0]), int(points[1][0])
        if not (ROW_START <= min(row_a, row_b) and max(row_a, row_b) < curve.N_ROWS):
            raise ValueError(f"行は {ROW_START}〜{curve.N_ROWS - 1} の範囲で指定してください")
        new_xp, new_h = curve.clear_rows(xp, h, row_a, row_b)
        return new_xp, new_h, curve.control_rows(new_h, len(deltas) if deltas else 3)
    if not deltas:
        raise ValueError("deltas must not be empty")
    rows = curve.control_rows(h, len(deltas))
    if mode == "control":
        return curve.apply_control_delta(xp, h, deltas), h, rows
    if mode == "shift":
        return curve.apply_shift(xp, h, deltas[0]), h, rows
    if mode == "tilt":
        return curve.apply_tilt(xp, h, deltas[0]), h, rows
    if mode == "regen":
        return curve.regenerate(h, deltas), h, rows
    raise ValueError(f"unknown mode {mode!r}")


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

        # --- ルーティング ---

        def do_GET(self):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            try:
                if u.path == "/":
                    return self._static("index.html")
                if parts[:1] == ["static"] and len(parts) == 2:
                    return self._static(parts[1])
                if parts[:2] == ["api", "queue"]:
                    return self._queue(parse_qs(u.query))
                if parts[:2] == ["api", "frame"] and len(parts) == 4:
                    return self._frame(parts[2], parts[3])
                if parts[:2] == ["api", "image"] and len(parts) == 4:
                    return self._image(parts[2], parts[3])
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

        def do_PUT(self):
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            if parts[:2] != ["api", "frame"] or len(parts) != 4:
                self._drain()
                return self._error(404, "not found")
            try:
                app.store.save(parts[2], parts[3], self._body()["label"])
            except ValueError as e:
                return self._error(400, str(e))
            except (KeyError, TypeError) as e:
                return self._error(400, f"bad request: {e}")
            except FileNotFoundError:
                return self._error(404, "not found")
            except Exception:  # noqa: BLE001
                return self._error(500, "internal error")
            app.rescore(parts[2], [parts[3]])
            return self._json({"ok": True})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                payload = self._body()
                if path == "/api/preview":
                    xp, hv, rows = _preview(payload)
                    return self._json({"xp": xp, "h_vector": hv, "control_rows": rows})
                if path == "/api/propagate":
                    applied, skipped = app.propagate(
                        payload["split"],
                        payload["name"],
                        int(payload.get("entry_index", 0)),
                        payload["delta"],
                        int(payload.get("range_n", 5)),
                    )
                    app.rescore(payload["split"], applied + [payload["name"]])
                    return self._json({"applied": applied, "skipped": skipped})
                if path == "/api/status":
                    app.store.set_status(payload["split"], payload["name"], payload["status"])
                    return self._json({"ok": True})
                if path == "/api/autolabel":
                    label = app.autolabel(payload["split"], payload["name"])
                    return self._json({"label": label})
            except ValueError as e:
                return self._error(400, str(e))
            except (KeyError, TypeError, FileNotFoundError) as e:
                # FileNotFoundError をここでは 400 のまま据え置く
                # （do_GET/do_PUT の 404 との非対称は既知・別途対応）。
                return self._error(400, f"bad request: {e}")
            except Exception:  # noqa: BLE001
                return self._error(500, "internal error")
            return self._error(404, "not found")

        # --- 各ハンドラ ---

        def _static(self, filename):
            path = STATIC_DIR / filename
            if not path.is_file() or path.parent != STATIC_DIR:
                return self._error(404, "not found")
            ctype = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
            return self._send(200, path.read_bytes(), ctype)

        def _queue(self, q):
            items = app.queue()
            status_filter = q.get("status", [None])[0]
            if status_filter:
                items = [
                    i for i in items
                    if app.store.get_status(i["split"], i["name"]) == status_filter
                ]
            total = len(items)
            offset = int(q.get("offset", [0])[0])
            limit = int(q.get("limit", [200])[0])
            page = items[offset : offset + limit]
            out = [
                dict(i, status=app.store.get_status(i["split"], i["name"]))
                for i in page
            ]
            return self._json({"items": out, "total": total})

        def _frame(self, split, name):
            label = app.store.load(split, name)
            prev, nxt = app.adjacent(split, name)
            return self._json(
                {
                    "split": split,
                    "name": name,
                    "label": label,
                    "metrics": detect.frame_metrics(label),
                    "prev": prev,
                    "next": nxt,
                }
            )

        def _image(self, split, name):
            path = app.store.image_path(split, name)
            if not path.is_file():
                return self._error(404, "not found")
            return self._send(200, path.read_bytes(), "image/png")

    return Handler


def make_server(root, port=8765, inferencer=None):
    """127.0.0.1 にだけバインドしたサーバを返す（serve_forever は呼び出し側で）。"""
    app = Annotator(root, inferencer=inferencer)
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))

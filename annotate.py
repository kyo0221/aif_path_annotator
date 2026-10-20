"""経路ラベル アノテーションツール（地図投影版）の起動スクリプト。

    python3 tools/path_annotator/annotate.py --root <dataset> \
        --vectormap <aiformula_course.osm> --yolop <yolopv2.pt> \
        --params <main_params.yaml> --learning-repo <learning>
    # --port 9000 / --no-browser / --cost-threshold 0.25
"""
import argparse
import errno
import socket
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.path_annotator.app import Annotator  # noqa: E402
from tools.path_annotator.frames import FrameIndex  # noqa: E402
from tools.path_annotator.gnss import GnssIndex  # noqa: E402
from tools.path_annotator.matching import DistanceField, observable_polylines  # noqa: E402
from tools.path_annotator.pose import PoseStore  # noqa: E402
from tools.path_annotator.projector import load_projector  # noqa: E402
from tools.path_annotator.server import make_server  # noqa: E402
from tools.path_annotator.store import LabelStore  # noqa: E402
from tools.path_annotator.yolop import LaneDetector, Prefetcher, load_yolop  # noqa: E402


def _report_port_in_use(port):
    print(f"ポート {port} は既に使われています。", file=sys.stderr)
    print(f"  別のポートで起動する: --port {port + 1}", file=sys.stderr)
    print(f"  使用中のプロセスを調べる: ss -lptn 'sport = :{port}'", file=sys.stderr)


def _port_available(port):
    """モデルロード（遅い）の前にポートの空きを軽く先読みする。

    ここでの成功は確保を保証しない（実際の確保は make_server 側）。
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            return False
        raise
    finally:
        probe.close()
    return True


def build_app(root, vectormap, yolop, params, learning_repo, cost_threshold=0.25, cache_dir=None):
    """部品を組み立てて (Annotator, Prefetcher) を返す。Prefetcher は未開始。"""
    root = Path(root)
    gnss_path = root / "gnss.json"
    if not gnss_path.is_file():
        raise ValueError(f"{root} に gnss.json がありません")
    store = LabelStore(root)
    store.ensure_backup()
    gnss = GnssIndex(gnss_path)
    keys = [(split, name) for split in store.splits() for name in store.frames(split)]
    frames = FrameIndex(keys, gnss.meta())
    field = DistanceField(observable_polylines(vectormap))
    projector = load_projector(learning_repo, params, vectormap)
    detector = LaneDetector(load_yolop(yolop), cache_dir or root / "cache" / "yolop")
    items = [(f"{s}/{n}", str(store.image_path(s, n))) for (s, n) in frames.order]
    prefetcher = Prefetcher(detector, items)
    annotator = Annotator(
        store, PoseStore(root), frames, gnss, projector, field, detector,
        cost_threshold=cost_threshold, prefetcher=prefetcher,
    )
    return annotator, prefetcher


def main(argv=None):
    ap = argparse.ArgumentParser(description="経路ラベル アノテーションツール（地図投影版）")
    ap.add_argument("--root", required=True, help="データセットのルート（images/ と gnss.json を含むディレクトリ）")
    ap.add_argument("--vectormap", help="vectormap(.osm)")
    ap.add_argument("--yolop", help="YOLOP の TorchScript 重み")
    ap.add_argument("--params", help="カメラ外部パラメータの main_params.yaml")
    ap.add_argument("--learning-repo", help="学習リポジトリ（投影コードの取得元）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    ap.add_argument("--cost-threshold", type=float, default=0.25, help="要確認フラグ cost を立てる照合コストのしきい値 [m]（候補の採否には使わない）")
    args = ap.parse_args(argv)

    if not (args.vectormap and args.yolop and args.params and args.learning_repo):
        ap.error("--vectormap / --yolop / --params / --learning-repo はすべて必須です")

    root = Path(args.root).resolve()
    if not (root / "images").is_dir():
        ap.error(f"{root} に images/ がありません")

    # モデルロードは遅いので、先にポートの空きを確認する。
    if not _port_available(args.port):
        _report_port_in_use(args.port)
        return 1

    print(f"データセット: {root}")
    print("モデルを読み込んでいます（数秒〜数十秒かかります）...")
    try:
        annotator, prefetcher = build_app(
            root, args.vectormap, args.yolop, args.params, args.learning_repo,
            cost_threshold=args.cost_threshold,
        )
    except ValueError as e:
        ap.error(str(e))
    try:
        srv = make_server(annotator, args.port)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        _report_port_in_use(args.port)
        return 1
    prefetcher.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print(f"起動しました: {url}  （Ctrl+C で終了）")
    timer = None
    if not args.no_browser:
        timer = threading.Timer(1.0, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n終了します")
    finally:
        if timer is not None:
            timer.cancel()
        prefetcher.stop()
        srv.shutdown()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""経路ラベル アノテーションツールの起動スクリプト。

    python3 tools/path_annotator/annotate.py --root dataset/sim/dataset_v2
    python3 tools/path_annotator/annotate.py --root dataset/sim/dataset_v2 --port 9000
    python3 tools/path_annotator/annotate.py --root dataset/sim/dataset_v2 --no-browser  # ブラウザを自動で開かない
"""
import argparse
import errno
import socket
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.path_annotator.server import make_server  # noqa: E402


def _report_port_in_use(port):
    print(f"ポート {port} は既に使われています。", file=sys.stderr)
    print(f"  別のポートで起動する: --port {port + 1}", file=sys.stderr)
    print(f"  使用中のプロセスを調べる: ss -lptn 'sport = :{port}'", file=sys.stderr)


def _port_available(port):
    """--weights 使用時、重みロード（数十秒〜）の前にポートの空きを軽く先読みする。

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


def main(argv=None):
    ap = argparse.ArgumentParser(description="経路ラベル アノテーションツール")
    ap.add_argument("--root", required=True, help="データセットのルート（images/ と labels/ を含むディレクトリ）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    ap.add_argument("--weights", help="推論に使う重みファイル。指定するとキューを既存ラベルとの乖離順にする")
    ap.add_argument("--learning-repo", help="ネットワーク実装のあるリポジトリ（--weights と併用）")
    ap.add_argument("--branch-threshold", type=float, default=60.0, help="分岐を出す最小の横差(px)")
    ap.add_argument("--vectormap", help="vectormap(.osm)。指定すると GNSS+俯瞰照合で straight 経路を作る機能(v キー/地図ボタン)を有効にする")
    ap.add_argument("--yolop", help="YOLOP の TorchScript 重み（--vectormap と併用）")
    ap.add_argument("--params", help="カメラ外部パラメータの main_params.yaml（--vectormap と併用）")
    args = ap.parse_args(argv)

    if args.weights and not args.learning_repo:
        ap.error("--weights を指定する場合は --learning-repo も指定してください")
    if args.vectormap and not (args.yolop and args.params and args.learning_repo):
        ap.error("--vectormap を指定する場合は --yolop / --params / --learning-repo も指定してください")

    root = Path(args.root).resolve()
    if not (root / "labels").is_dir():
        ap.error(f"{root} に labels/ がありません")

    inferencer = None
    if args.weights:
        # 重みロードに数十秒かかるため、ポートが使えるかを先に軽く確認しておく
        # （実際の確保は make_server 側。ここで通ってもレースで失敗しうる）。
        if not _port_available(args.port):
            _report_port_in_use(args.port)
            return 1
        # 実測基準: 22,460 件で約 95 秒（GPU）。この比から概算時間を出す。
        SECONDS_PER_FRAME = 95.0 / 22460
        n = sum(1 for _ in root.glob("labels/*/*.txt"))
        print(f"乖離順のキューを作るため全フレームを推論します（約 {n * SECONDS_PER_FRAME:.0f} 秒）。")
        # torch / 学習リポジトリが無い環境でも --weights を使わない起動が
        # 壊れないよう、import はここまで遅延させてある。
        from tools.path_annotator.infer import Inferencer  # noqa: E402

        inferencer = Inferencer(
            args.weights, args.learning_repo, branch_threshold=args.branch_threshold
        )

    matcher = None
    if args.vectormap:
        from tools.path_annotator.vectormap_match import VectormapMatcher  # noqa: E402

        matcher = VectormapMatcher(root, args.vectormap, args.yolop, args.params, args.learning_repo)

    print(f"データセット: {root}")
    try:
        srv = make_server(root, args.port, inferencer=inferencer, matcher=matcher)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        _report_port_in_use(args.port)
        return 1
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print(f"起動しました: {url}  （Ctrl+C で終了）")
    print("最初にブラウザで開いたとき、全フレームのスコアリングに数秒かかります。")
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
        srv.shutdown()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""経路ラベル アノテーションツールの起動スクリプト。

    python3 tools/path_annotator/annotate.py --root dataset/sim/dataset_v2
    python3 tools/path_annotator/annotate.py --root dataset/sim/dataset_v2 --port 9000
    python3 tools/path_annotator/annotate.py --root dataset/sim/dataset_v2 --no-browser  # ブラウザを自動で開かない
"""
import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.path_annotator.server import make_server  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="経路ラベル アノテーションツール")
    ap.add_argument("--root", required=True, help="データセットのルート（images/ と labels/ を含むディレクトリ）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not (root / "labels").is_dir():
        ap.error(f"{root} に labels/ がありません")

    print(f"データセット: {root}")
    srv = make_server(root, args.port)
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


if __name__ == "__main__":
    main()

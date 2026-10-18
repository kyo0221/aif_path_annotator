"""skip としたフレームを学習から除外するスクリプト。

    python3 tools/path_annotator/exclude_skipped.py --root <dataset> [--dry-run]

annotation_state.json で status == "skip" のフレームの画像とラベルを
excluded/<split>/ へ移す（元に戻すには同名ファイルを images/・labels/ へ戻す）。
"""
import argparse
import json
import os
from pathlib import Path


def _skipped_keys(root: Path) -> list[str] | None:
    """state から skip のキー（"<split>/<name>"）を返す。読めなければ None。"""
    try:
        state = json.loads((root / "annotation_state.json").read_text())
        frames = state["frames"]
        return sorted(k for k, v in frames.items()
                      if isinstance(v, dict) and v.get("status") == "skip")
    except (OSError, ValueError, KeyError, AttributeError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="skip フレームを excluded/ へ移す")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    root: Path = args.root

    keys = _skipped_keys(root)
    if keys is None:
        print("annotation_state.json が無いか読めないため、何もしません")
        return 0

    count = 0
    for key in keys:
        split, _, name = key.partition("/")
        pairs = [(root / "images" / split / f"{name}.png",
                  root / "excluded" / split / f"{name}.png"),
                 (root / "labels" / split / f"{name}.txt",
                  root / "excluded" / split / f"{name}.txt")]
        pairs = [(s, d) for s, d in pairs if s.exists()]
        if not pairs:
            continue
        if args.dry_run:
            print(f"[dry-run] {key}")
        else:
            for src, dst in pairs:
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src, dst)
            print(f"moved {key}")
        count += 1
    verb = "移動予定" if args.dry_run else "移動"
    print(f"{verb}: {count} フレーム")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""poses.json の確定姿勢で地図線・パスを画像に重ねた PNG を書き出す（目視確認用）。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2  # noqa: E402

from tools.path_annotator.gallery import draw_overlay  # noqa: E402
from tools.path_annotator.projector import H, W, load_projector  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--learning-repo", required=True)
    parser.add_argument("--vectormap", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("keys", nargs="+", help="split/name")
    args = parser.parse_args()

    root = Path(args.root)
    frames = json.loads((root / "poses.json").read_text())["frames"]
    projector = load_projector(args.learning_repo, args.params, args.vectormap)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in args.keys:
        split, name = key.split("/", 1)
        entry = frames[key]
        pose = {k: entry[k] for k in ("x", "y", "yaw", "pitch")}
        img = cv2.imread(str(root / "images" / split / f"{name}.png"))
        if img is None:
            raise SystemExit(f"画像が読めません: {key}")
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        label_path = root / "labels" / split / f"{name}.txt"
        label = json.loads(label_path.read_text()) if label_path.is_file() else None
        path = out / f"{name}.png"
        cv2.imwrite(str(path), draw_overlay(img, projector.overlay(pose, entry.get("lanelet")), label))
        print(path, entry["source"], entry["cost"], entry["flags"])


if __name__ == "__main__":
    main()

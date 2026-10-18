"""poses.json の確定姿勢で地図線・パスを画像に重ねた PNG を書き出す（目視確認用）。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.path_annotator.projector import H, W, load_projector  # noqa: E402

LINE_BGR = (255, 200, 0)  # #00c8ff
PATH_BGR = {"straight": (0, 230, 255), "left": (255, 120, 255), "right": (0, 140, 255)}


def _polyline(img, pts, color, thickness):
    pts = np.round(np.asarray(pts, float)).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def render(img, ov):
    for piece in ov["lines"]:
        _polyline(img, piece, LINE_BGR, 2)
    for cls, piece in ov["paths"].items():
        _polyline(img, piece, PATH_BGR.get(cls, (255, 255, 255)), 3)
    for cls, rows in ov["rows"].items():
        row_h = H // len(rows["xp"])  # ラベル規約: xp = u / (W - 1)、行 i の v = i*row_h + (row_h-1)/2
        for i, (xp, ok) in enumerate(zip(rows["xp"], rows["h_vector"])):
            if ok:
                u, v = xp * (W - 1), i * row_h + (row_h - 1) / 2
                cv2.circle(img, (int(round(u)), int(round(v))), 3, PATH_BGR.get(cls, (255, 255, 255)), -1)
    return img


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
        path = out / f"{name}.png"
        cv2.imwrite(str(path), render(img, projector.overlay(pose)))
        print(path, entry["source"], entry["cost"], entry["flags"])


if __name__ == "__main__":
    main()

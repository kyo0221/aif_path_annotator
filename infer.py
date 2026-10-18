"""学習済み重みで経路ラベルを推論する。

ネットワーク実装は別リポジトリ（呼び出し側が `learning_repo` として渡すパス）に
あるため、このモジュール自身はそこへの絶対パスを一切持たない。

torch や学習リポジトリが無い環境でも tools/path_annotator の既存テストが
壊れないよう、torch/cv2/numpy およびネットワーク実装の import はすべて
Inferencer.__init__ / predict_batch まで遅延させてある（モジュール冒頭では
何も import しない）。
"""
import sys

CLASSES = ("straight", "left", "right")  # ネットワークのスロット順と一致
N_ROWS = 48
IMG_W = 640
DEFAULT_BRANCH_THRESHOLD = 60.0  # px
BATCH_SIZE = 32


class Inferencer:
    """学習済み重みで経路ラベルを推論する。

    ネットワーク実装は別リポジトリにあるので、その場所は呼び出し側が渡す。
    """

    def __init__(self, weights, learning_repo, version="n", device=None,
                 branch_threshold=DEFAULT_BRANCH_THRESHOLD):
        import torch  # 遅延 import。torch が無い環境でも本モジュール自体は import できる。

        sys.path.insert(0, str(learning_repo))
        from Models.model_components.vision_planner.vision_planner_network import Network
        from Models.model_components.vision_planner.vision_planner_head import ROW_START

        self._torch = torch
        self._row_start = ROW_START
        self.branch_threshold = branch_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        model = Network(version)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model.to(self.device)

    def predict(self, image_path):
        """1 枚の画像からラベル（{class, xp, h_vector} のリスト）を作る。"""
        return self.predict_batch([image_path])[0]

    def predict_batch(self, image_paths):
        """複数枚をまとめて推論する。

        22,460 件クラスの規模でも全画像を一度にメモリへ載せないよう、
        BATCH_SIZE 枚ずつ読み込んでは推論する（読み込みと forward の両方）。
        """
        import cv2
        import numpy as np
        torch = self._torch

        labels = []
        for start in range(0, len(image_paths), BATCH_SIZE):
            chunk = image_paths[start:start + BATCH_SIZE]
            images = []
            for path in chunk:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"failed to read image: {path}")
                images.append(img.transpose(2, 0, 1))
            batch = np.ascontiguousarray(np.stack(images))
            sample = torch.from_numpy(batch).to(self.device).float() / 255.0
            with torch.no_grad():
                valid_logit, position = self.model(sample)
            valid = (valid_logit.sigmoid() > 0.5).cpu().numpy()
            position = position.cpu().numpy()
            for i in range(len(chunk)):
                labels.append(self._to_label(valid[i], position[i]))
        return labels

    def _to_label(self, valid, position):
        """(valid, position)（形状 (3, 23)）から 1 フレームぶんのラベルを作る。

        straight は必ず出す（行が全て無効なら全行を有効にして出す）。
        left/right は straight との最大分離が branch_threshold(px) を超えたときだけ出す。
        """
        import numpy as np

        n_local_rows = position.shape[1]
        px_scale = IMG_W - 1

        def rows_from_local(mask_local, values_local):
            xp = np.zeros(N_ROWS, dtype=float)
            h = np.zeros(N_ROWS, dtype=int)
            rows = self._row_start + np.flatnonzero(mask_local)
            xp[rows] = values_local[mask_local]
            h[rows] = 1
            return xp.tolist(), h.tolist()

        straight_valid = valid[0]
        if not straight_valid.any():
            straight_valid = np.ones(n_local_rows, dtype=bool)
        straight_xp, straight_h = rows_from_local(straight_valid, position[0])
        out = [{"class": "straight", "xp": straight_xp, "h_vector": straight_h}]

        for slot, cls in ((1, "left"), (2, "right")):
            slot_valid = valid[slot]
            if not slot_valid.any():
                continue
            separation_px = np.abs(position[slot] - position[0]) * px_scale
            if separation_px[slot_valid].max() <= self.branch_threshold:
                continue
            xp, h = rows_from_local(slot_valid, position[slot])
            out.append({"class": cls, "xp": xp, "h_vector": h})
        return out

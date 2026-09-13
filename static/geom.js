// キャンバス座標とラベル座標の相互変換。
// 描画倍率は SCALE にだけ持たせ、xp は常に正規化値で扱う。
(function (root) {
  const IMG_W = 640;
  const IMG_H = 384;
  const N_ROWS = 48;
  const ROW_H = IMG_H / N_ROWS;      // 8
  const SCALE = 2;
  const ROW_START = 25;              // これより上の行は学習に使われない（head の ROW_START）

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  // ラベル規約は u/(width-1)（projection.py: columns = u / (width - 1)）。
  // 640 で割ると実データの preview と 1px ずれる。
  function xpToX(xp) {
    return xp * (IMG_W - 1) * SCALE;
  }

  function xToXp(x) {
    return clamp(x / SCALE / (IMG_W - 1), 0, 1);
  }

  // 行アンカーは row_anchor_targets(48, 384) = i*8 + (8-1)/2 = i*8 + 3.5
  function rowToY(row) {
    return (row * ROW_H + (ROW_H - 1) / 2) * SCALE;
  }

  function yToRow(y) {
    const row = Math.floor(y / SCALE / ROW_H);
    return clamp(row, 0, N_ROWS - 1);
  }

  function hitTest(x, y, controlRows, xp, radius) {
    let best = -1;
    let bestD = radius;
    controlRows.forEach((row, i) => {
      const dx = x - xpToX(xp[row]);
      const dy = y - rowToY(row);
      const d = Math.hypot(dx, dy);
      if (d <= bestD) {
        bestD = d;
        best = i;
      }
    });
    return best;
  }

  const api = { IMG_W, IMG_H, N_ROWS, ROW_H, SCALE, ROW_START, rowToY, yToRow, xpToX, xToXp, hitTest };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.Geom = api;
})(typeof window !== 'undefined' ? window : globalThis);

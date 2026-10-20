// 画像 canvas への描画。幾何（画素座標）はサーバが作るので、ここは倍率を掛けて描くだけ。
// 線幅は倍率に合わせてスケールしない（拡大しても白線を覆わない）。
(function () {
  const IMG_W = 640;
  const ROW_H = 8; // 384 / 48
  const COLORS = { straight: '#ffe600', left: '#ff78ff', right: '#ff8c00' };
  const LINE_COLOR = '#00c8ff';
  const MASK_COLOR = '#00dc00';

  // マスクの着色は画像 1 枚ごとに 1 回で足りる。描画のたびに作り直さない。
  const tinted = new WeakMap();

  function tint(maskImg) {
    let c = tinted.get(maskImg);
    if (c) return c;
    c = document.createElement('canvas');
    c.width = maskImg.naturalWidth;
    c.height = maskImg.naturalHeight;
    const g = c.getContext('2d');
    // マスクは白 = 白線・黒 = 背景の不透明 PNG。輝度をアルファにしないと
    // source-in で背景まで緑に塗られるので、黒を透明にしてから着色する。
    g.drawImage(maskImg, 0, 0);
    const px = g.getImageData(0, 0, c.width, c.height);
    const d = px.data;
    for (let i = 0; i < d.length; i += 4) d[i + 3] = d[i];
    g.putImageData(px, 0, 0);
    g.globalCompositeOperation = 'source-in';
    g.fillStyle = MASK_COLOR;
    g.fillRect(0, 0, c.width, c.height);
    tinted.set(maskImg, c);
    return c;
  }

  function polyline(ctx, pts, scale) {
    if (!pts || pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0][0] * scale, pts[0][1] * scale);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * scale, pts[i][1] * scale);
    ctx.stroke();
  }

  // ラベル規約は xp = u / (width - 1)、行アンカーは i*8 + (8-1)/2（旧 geom.js と同じ）。
  function rowDots(ctx, row, scale) {
    for (let i = 0; i < row.h_vector.length; i++) {
      if (!row.h_vector[i]) continue;
      ctx.beginPath();
      ctx.arc(row.xp[i] * (IMG_W - 1) * scale, (i * ROW_H + (ROW_H - 1) / 2) * scale, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function draw(ctx, { img, maskImg, overlay, scale, opts }) {
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    if (img && img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, w, h);
    if (opts.hideAll) { ctx.restore(); return; }
    if (opts.showMask && maskImg && maskImg.complete && maskImg.naturalWidth) {
      ctx.globalAlpha = opts.maskAlpha;
      ctx.drawImage(tint(maskImg), 0, 0, w, h);
    }
    if (overlay) {
      ctx.globalAlpha = opts.mapAlpha;
      ctx.lineJoin = 'round';
      if (opts.showLines) {
        ctx.strokeStyle = LINE_COLOR;
        ctx.lineWidth = 2;
        (overlay.lines || []).forEach((pts) => polyline(ctx, pts, scale));
      }
      ctx.lineWidth = 3;
      Object.entries(overlay.paths || {}).forEach(([cls, pts]) => {
        ctx.strokeStyle = COLORS[cls] || '#ffffff';
        polyline(ctx, pts, scale);
      });
      Object.entries(overlay.rows || {}).forEach(([cls, row]) => {
        ctx.fillStyle = COLORS[cls] || '#ffffff';
        rowDots(ctx, row, scale);
      });
    }
    ctx.restore();
  }

  window.Overlay = { draw, COLORS };
})();

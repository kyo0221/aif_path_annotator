// 俯瞰パネル。車両座標（x 前・y 左 [m]）の地図線・経路・白線点を描き、
// ドラッグとホイールを姿勢の差分としてコールバックに渡すだけ（状態と API は app.js が持つ）。
(function () {
  const PPM_MIN = 8;
  const PPM_MAX = 24;
  const GRID_M = 5;
  const LINE_COLOR = '#00c8ff';
  const POINT_COLOR = '#00dc00';
  const GRID_COLOR = 'rgba(255, 255, 255, 0.08)';
  const YAW_STEP = 0.5; // ホイール 1 刻みのヨー [deg]

  // handlers: {onDragStart(), onDrag(dLateral, dLongitudinal), onWheel(dYawDeg)}。どれも省略可。
  function create(canvas, handlers) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h - 40; // 車両の位置
    const panel = { ppm: 12, draw, setPpm };
    let last = null; // {bev, paths}（縮尺変更で描き直す用）

    // 車両座標 → 画素。前方が上、左が左。
    const toPx = (x, y) => [cx - y * panel.ppm, cy - x * panel.ppm];

    function polyline(pts) {
      if (!pts || pts.length < 2) return;
      ctx.beginPath();
      pts.forEach(([x, y], i) => {
        const [px, py] = toPx(x, y);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    function grid() {
      ctx.strokeStyle = GRID_COLOR;
      ctx.lineWidth = 1;
      const step = GRID_M * panel.ppm;
      ctx.beginPath();
      // 車両を通る線が格子に乗るように、車両位置から両側へ引く。
      for (let px = cx % step; px <= w; px += step) { ctx.moveTo(px + 0.5, 0); ctx.lineTo(px + 0.5, h); }
      for (let py = cy % step; py <= h; py += step) { ctx.moveTo(0, py + 0.5); ctx.lineTo(w, py + 0.5); }
      ctx.stroke();
    }

    function vehicle() {
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.moveTo(cx, cy - 12);
      ctx.lineTo(cx - 7, cy + 8);
      ctx.lineTo(cx + 7, cy + 8);
      ctx.closePath();
      ctx.fill();
    }

    // bev: {lines, paths, points}。paths を別に渡したときはそちらを描く（省略時は bev.paths）。
    function draw(bev, paths) {
      last = { bev, paths };
      ctx.save();
      ctx.clearRect(0, 0, w, h);
      grid();
      if (bev) {
        ctx.lineJoin = 'round';
        ctx.lineWidth = 2;
        ctx.strokeStyle = LINE_COLOR;
        (bev.lines || []).forEach(polyline);
        const colors = (window.Overlay && window.Overlay.COLORS) || {};
        Object.entries(paths || bev.paths || {}).forEach(([cls, pts]) => {
          ctx.strokeStyle = colors[cls] || '#ffffff';
          polyline(pts);
        });
        ctx.fillStyle = POINT_COLOR;
        (bev.points || []).forEach(([x, y]) => {
          const [px, py] = toPx(x, y);
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fill();
        });
      }
      vehicle();
      ctx.restore();
    }

    function setPpm(ppm) {
      panel.ppm = Math.min(PPM_MAX, Math.max(PPM_MIN, ppm));
      if (last) draw(last.bev, last.paths);
    }

    // --- マウス ---
    // 地図を右へ d px 動かす = 車が左へ動いた → 横 +d/ppm。上へ d px → 前後 -d/ppm。
    // mousemove ごとに前回からの差分を渡す（mouseup で最後の差分も渡して確定）。
    let drag = null; // {x, y}（前回の位置）

    function emit(ev) {
      const dx = ev.clientX - drag.x;
      const dy = ev.clientY - drag.y;
      drag = { x: ev.clientX, y: ev.clientY };
      // 画面の y は下向きが + なので、上へ d px（dy = -d）で前後 -d/ppm になるのは dy / ppm。
      if ((dx || dy) && handlers.onDrag) handlers.onDrag(dx / panel.ppm, dy / panel.ppm);
    }

    canvas.addEventListener('mousedown', (ev) => {
      if (ev.button !== 0) return;
      ev.preventDefault();
      drag = { x: ev.clientX, y: ev.clientY };
      if (handlers.onDragStart) handlers.onDragStart();
    });
    // canvas の外へ出ても追えるように window で受ける。
    window.addEventListener('mousemove', (ev) => { if (drag) emit(ev); });
    window.addEventListener('mouseup', (ev) => {
      if (!drag) return;
      emit(ev);
      drag = null;
    });

    canvas.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      if (!ev.deltaY || !handlers.onWheel) return;
      handlers.onWheel(ev.deltaY < 0 ? YAW_STEP : -YAW_STEP);
    }, { passive: false });

    return panel;
  }

  window.Bev = { create, PPM_MIN, PPM_MAX };
})();

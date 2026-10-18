// ヨー調整ダイヤル（ハンドル型）。ドラッグで回した角度をヨーの差分としてコールバックに渡し、
// 現在のヨー補正に合わせて回った姿で描くだけ（状態と API は app.js が持つ）。
// ハンドルと同じく時計回りに回すと右へ曲がる = ヨーが減る。
(function () {
  const DEG_PER_DIAL_DEG = 0.1; // ダイヤル 1° でヨー 0.1°（1 周で 36°）。Shift で 5 倍
  const SHIFT_GAIN = 5;
  const RIM_R = 54;
  const RIM_W = 6;
  const HUB_R = 14;
  const RIM_COLOR = '#3a3f47';
  const TICK_COLOR = '#6b717a';
  const MARK_COLOR = '#ff5252';
  const TEXT_COLOR = '#e8eaed';
  const SPOKES_DEG = [90, 210, 330]; // canvas の角度（右が 0°・時計回り）。下と左上・右上
  const DEG = Math.PI / 180;

  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a)); // (−π, π]

  // handlers: {onTurnStart(), onTurn(dYawDeg), onReset()}。どれも省略可。
  function create(canvas, handlers) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;

    // yawDeg: 初期値からのヨー補正 [deg]。
    function draw(yawDeg) {
      ctx.save();
      ctx.clearRect(0, 0, w, h);
      ctx.translate(cx, cy);
      ctx.save();
      // ヨー +1° でダイヤルは反時計回りに 10°（canvas の回転は時計回りが +）。
      ctx.rotate((-yawDeg / DEG_PER_DIAL_DEG) * DEG);
      ctx.lineCap = 'round';
      // 外輪
      ctx.strokeStyle = RIM_COLOR;
      ctx.lineWidth = RIM_W;
      ctx.beginPath();
      ctx.arc(0, 0, RIM_R, 0, Math.PI * 2);
      ctx.stroke();
      // スポーク
      ctx.lineWidth = 5;
      SPOKES_DEG.forEach((d) => {
        const c = Math.cos(d * DEG);
        const s = Math.sin(d * DEG);
        ctx.beginPath();
        ctx.moveTo(c * HUB_R, s * HUB_R);
        ctx.lineTo(c * RIM_R, s * RIM_R);
        ctx.stroke();
      });
      // 30° ごとの目盛り（外輪の上に重ねる）
      ctx.strokeStyle = TICK_COLOR;
      ctx.lineWidth = 1.5;
      for (let d = 0; d < 360; d += 30) {
        const c = Math.cos(d * DEG);
        const s = Math.sin(d * DEG);
        ctx.beginPath();
        ctx.moveTo(c * (RIM_R - RIM_W / 2), s * (RIM_R - RIM_W / 2));
        ctx.lineTo(c * (RIM_R + RIM_W / 2), s * (RIM_R + RIM_W / 2));
        ctx.stroke();
      }
      // 上端の赤いマーク
      ctx.strokeStyle = MARK_COLOR;
      ctx.lineWidth = RIM_W;
      ctx.lineCap = 'butt';
      ctx.beginPath();
      ctx.arc(0, 0, RIM_R, -90 * DEG - 0.12, -90 * DEG + 0.12);
      ctx.stroke();
      ctx.restore();
      // 中心の数字は回さない
      ctx.fillStyle = TEXT_COLOR;
      ctx.font = '14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${yawDeg >= 0 ? '+' : ''}${yawDeg.toFixed(1)}°`, 0, 0);
      ctx.restore();
    }

    // --- マウス ---
    // 画面上の位置 → 中心からの角度（canvas 座標なので時計回りが +）。表示サイズと backing サイズの差は補正する。
    function angleOf(ev) {
      const r = canvas.getBoundingClientRect();
      const x = ((ev.clientX - r.left) * w) / (r.width || w);
      const y = ((ev.clientY - r.top) * h) / (r.height || h);
      return Math.atan2(y - cy, x - cx);
    }

    // {a0, started}。undo はクリックだけ（ダブルクリックの 1 回目など）で積まないよう、最初に回ったときに積む。
    let drag = null;

    function emit(ev) {
      const a = angleOf(ev);
      const delta = wrap(a - drag.a0); // ±180° をまたいでも短い方の弧
      drag.a0 = a;
      if (!delta) return;
      if (!drag.started) {
        drag.started = true;
        if (handlers.onTurnStart) handlers.onTurnStart();
      }
      if (handlers.onTurn) handlers.onTurn((-delta / DEG) * DEG_PER_DIAL_DEG * (ev.shiftKey ? SHIFT_GAIN : 1));
    }

    canvas.addEventListener('mousedown', (ev) => {
      if (ev.button !== 0) return;
      ev.preventDefault();
      drag = { a0: angleOf(ev), started: false };
    });
    // canvas の外へ出ても追えるように window で受ける。
    window.addEventListener('mousemove', (ev) => {
      if (!drag) return;
      if (!(ev.buttons & 1)) { drag = null; return; } // ボタンが離れていたらドラッグ終了
      emit(ev);
    });
    window.addEventListener('mouseup', (ev) => {
      if (!drag) return;
      emit(ev);
      drag = null;
    });
    canvas.addEventListener('dblclick', (ev) => {
      ev.preventDefault();
      if (handlers.onReset) handlers.onReset();
    });

    return { draw };
  }

  window.Dial = { create, DEG_PER_DIAL_DEG };
})();

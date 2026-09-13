// 画面の状態と操作。曲線の数学はすべてサーバ側 (/api/preview) に置いてある。
// 制御点の行番号 (S.controlRows) もサーバの curve.control_rows と一致させる必要が
// あるため、JS 側では計算せずサーバから受け取った値をそのまま使う。
(function () {
  const G = window.Geom;
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const COLORS = { straight: '#ff4d4d', left: '#4dff88', right: '#4da3ff' };
  // style.css の #mode-badge（ADD）/ #mode-badge.delete（DELETE）と同じ値。
  // バッジと始点マーカーの色をここ一箇所にまとめる（CSS とは値を手で合わせるだけで、
  // 共有の仕組みまでは作らない。二重管理だが 2 値だけなので許容する）。
  const MODE_COLORS = { add: '#1a9c6e', delete: '#d14343' };

  const S = {
    queue: [], total: 0, idx: -1, split: null, name: null,
    label: null, baseXp: null, baseH: null, entry: 0, k: 3, rangeN: 5, controlRows: [],
    img: new Image(), undo: [], redo: [], drag: null, pending: false,
    epoch: 0, // S.label の構造（フレーム・エントリの増減）が変わるたびに進む
    anchor: null, // 範囲操作（補間／削除）の始点 { row, epoch } | null。add/delete のクリックで覚える。
    // xp は持たない: h/l/j/k/R・ドラッグは epoch を進めないまま cur().xp を書き換えるため、
    // 値をここに複製すると古くなる。読む側は必ず cur().xp[row] を都度引く。
    mode: 'add', // 'add' | 'delete'。クリック／Shift+クリックの意味を決める（既定 add）
  };

  const api = {
    // r.ok を見ないと、/api/frame などが 400/500 を返したときに body が
    // {error: ...} のまま呼び出し側へ渡り、後続の body.label 参照が
    // 未捕捉の TypeError になる（async リスナ内なのでコンソール以外どこにも出ず、
    // ユーザーには「固まった」としか見えない）。
    get: async (u) => {
      const r = await fetch(u);
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
      return body;
    },
    send: (u, body, method) =>
      fetch(u, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        .then(async (r) => { if (!r.ok) throw new Error((await r.json()).error); return r.json(); }),
  };

  const msg = (t) => { document.getElementById('hud-msg').textContent = t || ''; };
  const cur = () => S.label[S.entry];

  // 始点は「今の S.label の構成」に対してのみ意味がある。フレーム移動・エントリ増減・
  // undo/redo はすべて S.epoch を進めるので、個別に S.anchor = null を足して回らなくても
  // epoch が食い違った時点で自動的に無効になる（S.baseXp のインデックスずれと同じ
  // クラスの問題を epoch 一本で防ぐ）。読む側は必ずこの関数を通し、S.anchor を直接見ない。
  function anchor() {
    return S.anchor && S.anchor.epoch === S.epoch ? S.anchor : null;
  }

  // --- 描画 ---

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (S.img.complete && S.img.naturalWidth) ctx.drawImage(S.img, 0, 0, canvas.width, canvas.height);
    // 行 0〜24 は学習に使われない（head の ROW_START）。触っても無駄なので覚える。
    // 点列・制御点より前に描き、後で重ねる点までは隠さないようにする
    // （帯そのものが編集対象外だと伝わればよく、既存データを隠す必要はない）。
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, canvas.width, G.ROW_START * G.ROW_H * G.SCALE);
    if (!S.label) return;
    S.label.forEach((e, ei) => {
      const color = COLORS[e.class] || '#ffffff';
      ctx.globalAlpha = ei === S.entry ? 1 : 0.35;
      ctx.fillStyle = color;
      e.h_vector.forEach((v, i) => {
        if (!v) return;
        ctx.beginPath();
        ctx.arc(G.xpToX(e.xp[i]), G.rowToY(i), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    S.controlRows.forEach((row) => {
      if (!cur().h_vector[row]) return; // 無効行にはハンドルを出さない
      ctx.beginPath();
      ctx.arc(G.xpToX(cur().xp[row]), G.rowToY(row), 8, 0, Math.PI * 2);
      ctx.stroke();
    });
    const a = anchor();
    if (a) {
      // 範囲操作の始点。バッジと同じ色でモードを示す（ADD: 緑 / DELETE: 赤）。
      // 色だけに頼らないよう形も変える: add は輪（補間で線がつながるイメージ）、
      // delete は×印（消す操作であることを形でも示す）。
      ctx.strokeStyle = MODE_COLORS[S.mode] || MODE_COLORS.add;
      ctx.lineWidth = 2;
      const cx = G.xpToX(cur().xp[a.row]); // 値は複製せず都度読む（古い位置に飛ばないため）
      const cy = G.rowToY(a.row);
      ctx.beginPath();
      if (S.mode === 'delete') {
        const r = 9;
        ctx.moveTo(cx - r, cy - r); ctx.lineTo(cx + r, cy + r);
        ctx.moveTo(cx + r, cy - r); ctx.lineTo(cx - r, cy + r);
      } else {
        ctx.arc(cx, cy, 11, 0, Math.PI * 2);
      }
      ctx.stroke();
    }
  }

  function refreshHud() {
    const done = S.queue.filter((q) => q.status === 'done').length;
    document.getElementById('hud-name').textContent = S.name || '—';
    document.getElementById('hud-class').textContent = `class: ${S.label ? cur().class : '—'}`;
    document.getElementById('hud-entry').textContent = `entry: ${S.entry + 1}/${S.label ? S.label.length : 0}`;
    document.getElementById('hud-k').textContent = `K: ${S.k}`;
    document.getElementById('hud-n').textContent = `伝播N: ${S.rangeN}`;
    const a = anchor();
    document.getElementById('hud-anchor').textContent = a ? `始点: 行 ${a.row}` : '始点: —';
    const badge = document.getElementById('mode-badge');
    badge.textContent = S.mode === 'delete' ? 'DELETE' : 'ADD';
    badge.classList.toggle('delete', S.mode === 'delete');
    document.getElementById('hud-progress').textContent =
      `${done}/${S.queue.length} 確認済み（全 ${S.total} 件中の上位 ${S.queue.length} 件を表示）`;
  }

  function renderQueue() {
    const ol = document.getElementById('queue-list');
    ol.innerHTML = '';
    S.queue.forEach((q, i) => {
      const li = document.createElement('li');
      li.className = (i === S.idx ? 'active ' : '') + (q.status === 'done' ? 'done' : '');
      li.innerHTML = `<span class="score">${q.score.toFixed(1)}</span>${q.name}`;
      li.title = q.reasons.join(', ');
      li.onclick = () => loadIndex(i);
      ol.appendChild(li);
    });
  }

  // --- サーバとの往復 ---

  async function preview(mode, deltas, baseXp) {
    // {xp, control_rows} を両方返す。呼び出し側は必要な方だけ使う。
    return api.send('/api/preview',
      { xp: baseXp || cur().xp, h_vector: cur().h_vector, mode, deltas }, 'POST');
  }

  // サーバ往復を伴う操作を直列に流す。
  // preview は呼び出し時点の cur().xp を読むので、応答前に次を投げると
  // 両方が同じ基準から計算し、後の書き込みが前の結果を打ち消す（キーリピートで押下が失われる）。
  let opChain = Promise.resolve();

  function enqueue(fn) {
    opChain = opChain.then(fn).catch((e) => msg(e.message));
    return opChain;
  }

  // 制御点行はサーバ（curve.control_rows）が権威。曲線を動かさない
  // (deltas は全部 0) まま、現在の h_vector に対応する制御点行だけを
  // サーバに問い合わせる（返ってくる xp は元と同じなので捨ててよい）。
  // 有効行が変わる操作（エントリ切替・h_vector 変更・undo/redo など）の
  // 後は必ず呼ぶこと。取り直しを忘れると、クライアントが送る deltas の
  // 添字とサーバが計算する制御点行がずれ、ドラッグが別の点を動かす。
  //
  // enqueue しない版。enqueue 済みの関数（Shift+クリック・単一クリックのハンドラ）の
  // 内側から await する専用で、これ自身は opChain に触らない。
  // すでに opChain の中で実行されている関数が、ここで再び enqueue すると
  // 「自分の完了を待つ Promise を自分で待つ」循環になりデッドロックする
  // （旧実装で実際に発生していた。詳細は下の syncControlRows のコメント参照）。
  async function fetchControlRows() {
    try {
      const body = await preview('control', new Array(S.k).fill(0));
      S.controlRows = body.control_rows;
    } catch (e) {
      msg(`制御点の取得に失敗: ${e.message}`);
    }
  }

  // リスナ（keydown/loadIndex）の直下など、まだ opChain の外から呼ぶ専用。
  // enqueue 済みの関数の内側からはこちらを呼ばず、必ず fetchControlRows() を直接 await すること。
  // ここから呼ぶと、opChain = opChain.then(...) が「今まさに実行中の自分自身」を
  // 待つ形になり、二度と解決しない。
  function syncControlRows() {
    return enqueue(fetchControlRows);
  }

  function controlXs() {
    // 制御点行が無効行に落ちている場合、その行の xp は 0.0 のダミーなので
    // 使えない（左端に引っ張られる）。最も近い有効行の値で代用する。
    const valid = [];
    cur().h_vector.forEach((v, i) => { if (v) valid.push(i); });
    if (!valid.length) return S.controlRows.map(() => 0.5);
    return S.controlRows.map((row) => {
      if (cur().h_vector[row]) return cur().xp[row];
      let best = valid[0];
      for (const i of valid) {
        if (Math.abs(i - row) < Math.abs(best - row)) best = i;
      }
      return cur().xp[best];
    });
  }

  // --- 読み込みと保存 ---

  async function loadIndex(i) {
    if (i < 0 || i >= S.queue.length) return;
    S.idx = i;
    const q = S.queue[i];
    let body;
    try {
      body = await api.get(`/api/frame/${q.split}/${q.name}`);
    } catch (e) {
      // 前のフレームの表示・状態はそのまま保ち、失敗だけを HUD に出す
      // （catch しないと async リスナ内の未捕捉例外になり、無言でフリーズしたように見える）。
      msg(`フレームの読み込みに失敗: ${e.message}`);
      return;
    }
    S.split = q.split; S.name = q.name; S.label = body.label;
    S.anchor = null; // フレームが変わったら補間の始点は無効
    // 伝播は「このフレームを開いてから加えた変更」を近傍へ配る。
    // 編集のたびに save() しているので、サーバの保存済みラベルとの差分は常に 0 になってしまう。
    // S.label と同じ添字で並ぶ配列にして、エントリの増減 (n/x/undo/redo) に追従させる
    // （ラベル全体をコピーする方式だと、x でインデックスがずれたときに
    // 無関係なエントリ同士の差分を伝播してしまう）。
    S.baseXp = body.label.map((e) => e.xp.slice());
    S.baseH = body.label.map((e) => e.h_vector.slice());
    S.entry = 0; S.undo = []; S.redo = [];
    S.epoch += 1; // ラベルを丸ごと差し替えた
    // syncControlRows が失敗した場合に出す HUD メッセージを、この直後の msg('') が
    // 同じマイクロタスク列で消してしまわないよう、msg('') は await の前に呼んでおく。
    msg('');
    await syncControlRows();
    S.img.onload = draw;
    S.img.src = `/api/image/${q.split}/${q.name}`;
    renderQueue(); refreshHud(); draw();
  }

  function snapshot() {
    S.undo.push(JSON.stringify({ label: S.label, baseXp: S.baseXp, baseH: S.baseH }));
    if (S.undo.length > 100) S.undo.shift();
    S.redo = [];
  }

  async function save() {
    try {
      await api.send(`/api/frame/${S.split}/${S.name}`, { label: S.label }, 'PUT');
    } catch (e) { msg(`保存失敗: ${e.message}`); }
  }

  // --- マウス操作 ---

  function canvasPos(ev) {
    const r = canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }

  canvas.addEventListener('mousedown', async (ev) => {
    if (!S.label) return; // キュー読み込み前・空のときにクリックされても落ちないようにする
    const { x, y } = canvasPos(ev);
    if (ev.shiftKey) {
      // Shift+クリックは範囲操作（始点〜終点）の指定。制御点の掴み判定より優先する
      // （そうしないと制御点の真上を終点にできない）。add では直線補間、delete ではまとめて削除。
      return enqueue(async () => {
        const row = G.yToRow(y);
        const start = anchor();
        if (!start) return msg('先に始点をクリックしてください');
        if (row < G.ROW_START || start.row < G.ROW_START) {
          return msg('行 0〜24 は学習に使われません');
        }
        const entry = S.entry;
        const epoch = S.epoch;
        const deleting = S.mode === 'delete';
        snapshot();
        try {
          const body = await api.send('/api/preview', {
            xp: cur().xp, h_vector: cur().h_vector,
            mode: deleting ? 'erase' : 'interp',
            points: [[start.row, cur().xp[start.row]], [row, G.xToXp(x)]],
            deltas: new Array(S.k).fill(0), // これが無いと server 側が K=3 決め打ちで control_rows を返す
          }, 'POST');
          if (S.epoch !== epoch || S.entry !== entry) return; // 応答を待つ間に構成が変わった
          cur().xp = body.xp;
          cur().h_vector = body.h_vector;
        } catch (e) {
          return msg(`${deleting ? '削除' : '補間'}に失敗: ${e.message}`);
        }
        S.anchor = null;
        S.epoch += 1; // 有効行が変わった
        await fetchControlRows(); // 有効行が変わったので制御点行を取り直す（enqueue 済みなので syncControlRows は使わない）
        refreshHud(); draw(); save();
      });
    }
    // ドラッグでの制御点掴みは add モードのときだけ。delete 中に白い輪へ触れても
    // 「クリック＝点を消す」以外の意味を持たせない（誤ってドラッグで曲線を動かさない）。
    if (S.mode !== 'delete') {
      const hit = G.hitTest(x, y, S.controlRows, cur().xp, 14);
      if (hit >= 0 && cur().h_vector[S.controlRows[hit]]) {
        snapshot();
        S.drag = { hit, baseXp: cur().xp.slice(), startX: x };
        return;
      }
    }
    // 制御点を掴んでいなければ、クリックした 1 行に対して現在のモードの操作を行う。
    // add/delete とも単一行の操作はサーバの interp/erase に同じ行を 2 回渡して実現する
    // （行を直接いじるコードを持たない。数学も状態遷移もサーバ側に一本化する）。
    return enqueue(async () => {
      const row = G.yToRow(y);
      if (row < G.ROW_START) return msg('行 0〜24 は学習に使われません');
      const deleting = S.mode === 'delete';
      if (deleting && !cur().h_vector[row]) return msg('この行に点はありません');
      const entry = S.entry;
      const epoch = S.epoch;
      const xp = G.xToXp(x);
      snapshot();
      try {
        const body = await api.send('/api/preview', {
          xp: cur().xp, h_vector: cur().h_vector,
          mode: deleting ? 'erase' : 'interp',
          points: [[row, xp], [row, xp]],
          deltas: new Array(S.k).fill(0), // これが無いと server 側が K=3 決め打ちで control_rows を返す
        }, 'POST');
        if (S.epoch !== epoch || S.entry !== entry) return;
        cur().xp = body.xp;
        cur().h_vector = body.h_vector;
      } catch (e) {
        return msg(`更新に失敗: ${e.message}`);
      }
      // 置いた／消した行を次の範囲操作の始点にする。epoch を紐付けておくことで、
      // この後 n/x/undo/redo で構成が変わればここでの始点が自動的に無効になる。
      S.anchor = { row, epoch: S.epoch };
      await fetchControlRows(); // 有効行が変わったので制御点行を取り直す（enqueue 済みなので syncControlRows は使わない）
      refreshHud(); draw(); save();
    });
  });

  canvas.addEventListener('mousemove', async (ev) => {
    if (!S.drag) return;
    const { x } = canvasPos(ev);
    S.drag.lastX = x;              // 間引かれてもここは必ず更新する
    if (S.pending) return;
    S.pending = true;
    const deltas = S.controlRows.map((row, i) => (i === S.drag.hit ? G.xToXp(x) - S.drag.baseXp[row] : 0));
    try {
      const body = await preview('control', deltas, S.drag.baseXp);
      cur().xp = body.xp;
    } catch (e) { msg(e.message); }
    requestAnimationFrame(draw);
    S.pending = false;
  });

  window.addEventListener('mouseup', async () => {
    if (!S.drag) return;
    const drag = S.drag;
    S.drag = null;
    if (drag.lastX !== undefined) {
      // 間引きで捨てられた最後の移動を、離した位置で 1 回だけ確定させる。
      // これをしないと画面の表示とディスクの内容が食い違う。
      const deltas = S.controlRows.map(
        (row, i) => (i === drag.hit ? G.xToXp(drag.lastX) - drag.baseXp[row] : 0));
      try {
        const body = await preview('control', deltas, drag.baseXp);
        cur().xp = body.xp;
        draw();
      } catch (e) { msg(e.message); }
    }
    save();
  });

  // --- キーボード操作 ---

  const PX = 1 / (G.IMG_W - 1);   // 1 正規化単位 = 画像 1px

  function applyPreview(mode, deltas) {
    return enqueue(async () => {
      // entry/epoch は await の前に固定する。応答待ちの間に Tab やフレーム切替が
      // 割り込んで cur() の指す先が変わっても、無関係なエントリに結果を書かない。
      const entry = S.entry;
      const epoch = S.epoch;
      snapshot();
      try {
        const body = await preview(mode, deltas);
        if (S.epoch !== epoch || S.entry !== entry) return; // 応答を待つ間に構成が変わった
        cur().xp = body.xp;
      } catch (e) {
        msg(`プレビューに失敗: ${e.message}`);
        return;
      }
      draw(); save();
    });
  }

  window.addEventListener('keydown', async (ev) => {
    if (!S.label) return;
    // 修飾キー付きはブラウザのショートカット。redo(Ctrl+u) だけが例外。
    // これが無いと Ctrl+P が印刷ダイアログと同時に伝播を実行して近傍ラベルを書き換えたり、
    // Ctrl+L（アドレスバー）でシフト＋保存、Ctrl+N でエントリ複製＋保存が走ったりする。
    const isRedo = ev.ctrlKey && !ev.altKey && !ev.metaKey && ev.key.toLowerCase() === 'u';
    if ((ev.ctrlKey || ev.altKey || ev.metaKey) && !isRedo) return;
    const k = ev.key;
    // CapsLock で大文字になっても効くよう、大文字小文字を区別しないキー
    // (j k n x p u R) はここで正規化する。h/l/H/L だけは意図的に 1px/5px を
    // 区別しているので、lower では判定せず ev.shiftKey を直接見る。
    const lower = k.toLowerCase();
    if (k === 'ArrowRight') return loadIndex(S.idx + 1);
    if (k === 'ArrowLeft') return loadIndex(S.idx - 1);
    if (k === 'Enter') {
      await api.send('/api/status', { split: S.split, name: S.name, status: 'done' }, 'POST');
      S.queue[S.idx].status = 'done';
      renderQueue(); refreshHud();
      return loadIndex(S.idx + 1);
    }
    if (lower === 'h') return applyPreview('shift', [ev.shiftKey ? -5 * PX : -PX]);
    if (lower === 'l') return applyPreview('shift', [ev.shiftKey ? 5 * PX : PX]);
    if (lower === 'j') return applyPreview('tilt', [-5 * PX]);
    if (lower === 'k') return applyPreview('tilt', [5 * PX]);
    if (lower === 'r') return applyPreview('regen', controlXs());
    if (lower === 'd') {
      S.mode = S.mode === 'delete' ? 'add' : 'delete';
      S.anchor = null; // 別モードの始点が残っていると混乱するのでクリアする
      refreshHud();
      return;
    }
    if (k === '1' || k === '2' || k === '3') {
      const next = { 1: 'straight', 2: 'left', 3: 'right' }[k];
      if (cur().class === next) return;
      if (S.label.some((e) => e.class === next)) {
        return msg(`${next} は既に別のエントリで使われています`);
      }
      const straightElsewhere = S.label.some((e, i) => i !== S.entry && e.class === 'straight');
      if (cur().class === 'straight' && !straightElsewhere) {
        return msg('straight を別の class に変えると straight が無くなります');
      }
      snapshot();
      cur().class = next;
      refreshHud(); draw(); return save();
    }
    if (k === 'Tab') {
      ev.preventDefault();
      S.entry = (S.entry + 1) % S.label.length;
      S.anchor = null; // 別エントリでは行の対応が変わるので始点は無効
      await syncControlRows(); // 別エントリは h_vector が違いうる
      refreshHud(); return draw();
    }
    if (lower === 'n') {
      const used = new Set(S.label.map((e) => e.class));
      const free = ['straight', 'left', 'right'].find((c) => !used.has(c));
      if (!free) return msg('straight / left / right は全て使用済みです');
      snapshot();
      const copy = JSON.parse(JSON.stringify(cur()));
      copy.class = free;
      S.label.push(copy);
      S.baseXp.push(null); // 追加したエントリには伝播の基準が無い
      S.baseH.push(null);
      S.entry = S.label.length - 1;
      S.epoch += 1; // エントリ数が変わった
      await syncControlRows(); // 追加直後に S.entry が移る
      refreshHud(); draw(); return save();
    }
    if (lower === 'x') {
      if (S.label.length <= 1) return msg('最後のエントリは消せません');
      if (cur().class === 'straight') return msg('straight は削除できません（学習に必須）');
      snapshot();
      S.label.splice(S.entry, 1);
      S.baseXp.splice(S.entry, 1); // S.label と同じ位置を削り、添字のずれを防ぐ
      S.baseH.splice(S.entry, 1);
      S.entry = 0;
      S.epoch += 1; // エントリ数が変わった
      await syncControlRows(); // S.entry が 0 に戻る
      refreshHud(); draw(); return save();
    }
    if (k === '[') { S.k = Math.max(1, S.k - 1); await syncControlRows(); refreshHud(); return draw(); }
    if (k === ']') { S.k = Math.min(5, S.k + 1); await syncControlRows(); refreshHud(); return draw(); }
    if (k === ',') { S.rangeN = Math.max(1, S.rangeN - 1); return refreshHud(); }
    if (k === '.') { S.rangeN = Math.min(50, S.rangeN + 1); return refreshHud(); }
    if (lower === 'p') {
      // entry / epoch / xpNow はすべて await の前に固定する。応答待ちの間に
      // Tab やフレーム切替が割り込んでも、後述の epoch チェックで基準の
      // 書き込み先を誤らないようにするため。
      const entry = S.entry;
      const epoch = S.epoch;
      const baseXp = S.baseXp && S.baseXp[entry];
      const baseH = S.baseH && S.baseH[entry];
      if (!baseXp || !baseH) return msg('このエントリは読み込み後に追加されたので、伝播の基準がありません');
      const xpNow = cur().xp.slice();
      const hNow = cur().h_vector;
      // 点の有無が変わった行は「横に動いた」わけではないので配らない。
      // 配ると近傍フレームの同じ行が左端に張り付く、あるいは大きく飛ぶ。
      const delta = xpNow.map((v, i) => (hNow[i] === baseH[i] ? v - baseXp[i] : 0));
      if (delta.every((d) => d === 0)) return msg('このフレームに変更がないので伝播しません');
      try {
        const body = await api.send('/api/propagate',
          { split: S.split, name: S.name, entry_index: entry, delta, range_n: S.rangeN }, 'POST');
        // 応答を待つ間にフレームやエントリの構成が変わっていたら基準は進めない。
        // 進めると、無関係なエントリの未伝播の編集が「配り済み」になって永久に失われる。
        if (S.epoch === epoch) { S.baseXp[entry] = xpNow; S.baseH[entry] = cur().h_vector.slice(); }
        const n = body.applied.length;
        const s = body.skipped.length;
        return msg(s ? `${n} フレームに伝播（${s} 件スキップ）` : `${n} フレームに伝播しました`);
      } catch (e) {
        return msg(`伝播に失敗: ${e.message}`);
      }
    }
    if (lower === 'u' && !ev.ctrlKey) {
      if (!S.undo.length) return msg('これ以上戻せません');
      S.redo.push(JSON.stringify({ label: S.label, baseXp: S.baseXp, baseH: S.baseH }));
      const snap = JSON.parse(S.undo.pop());
      S.label = snap.label;
      S.baseXp = snap.baseXp;
      S.baseH = snap.baseH;
      S.entry = Math.min(S.entry, S.label.length - 1);
      S.epoch += 1; // ラベル構造が undo で入れ替わった
      await syncControlRows(); // S.label が丸ごと入れ替わる
      refreshHud(); draw(); return save();
    }
    if (lower === 'u' && ev.ctrlKey) {
      ev.preventDefault(); // Ctrl+U はブラウザの「ページのソースを表示」なので抑止する
      if (!S.redo.length) return msg('やり直せる操作がありません');
      S.undo.push(JSON.stringify({ label: S.label, baseXp: S.baseXp, baseH: S.baseH }));
      const snap = JSON.parse(S.redo.pop());
      S.label = snap.label;
      S.baseXp = snap.baseXp;
      S.baseH = snap.baseH;
      S.entry = Math.min(S.entry, S.label.length - 1);
      S.epoch += 1; // ラベル構造が redo で入れ替わった
      await syncControlRows(); // S.label が丸ごと入れ替わる
      refreshHud(); draw(); return save();
    }
  });

  // --- 起動 ---

  (async function start() {
    try {
      const body = await api.get('/api/queue?limit=500');
      S.queue = body.items;
      S.total = body.total; // HUD の分母。limit=500 で切った件数と混同しないため別で持つ
      renderQueue();
      if (S.queue.length) await loadIndex(0);
    } catch (e) {
      msg(`キューの読み込みに失敗: ${e.message}`);
    }
  })();
})();

// 画面の状態・API・キー操作・フレームの流れ。投影・照合・ラベル生成はすべてサーバ側にあり、
// ここは姿勢 (x, y, yaw [rad], pitch [deg]) を動かして /api/preview の結果を描くだけ。
(function () {
  const $ = (id) => document.getElementById(id);
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');
  const IMG_W = 640;
  const IMG_H = 384;
  const MAX_SCALE = 1.5;
  const CLASSES = ['straight', 'left', 'right'];
  const STEP = { lateral: 0.05, longitudinal: 0.10, yawDeg: 0.5, pitch: 0.2 }; // Shift で 5 倍
  const DEG = Math.PI / 180;

  const S = {
    order: 'time', flagOnly: false, items: [], idx: -1, frame: null, pose: null, undo: [], redo: [],
    opts: { showMask: true, showLines: true, mapAlpha: 0.5, maskAlpha: 0.35, hideAll: false },
    pending: false, latest: null,
    source: null,   // /api/pose・/api/confirm に送る出どころ（手で動かしたら manual）
    view: null,     // 現在の姿勢に対する {overlay, bev, cost, label, flags}（frame か preview の応答）
    img: null, maskImg: null, scale: 1,
    candidates: [], // 候補の列に出している候補（frame か /api/match の応答）
    busy: false,    // 確定・skip・移動の POST からフレーム読み込みまでの間は次の移動を受け付けない（二重確定を防ぐ）
  };

  const api = {
    // r.ok を見ないと、/api/frame などが 400/500 を返したときに body が
    // {error: ...} のまま呼び出し側へ渡り、後続の参照が未捕捉の TypeError になる
    // （async リスナ内なのでコンソール以外どこにも出ず、ユーザーには「固まった」としか見えない）。
    get: async (u) => {
      const r = await fetch(u);
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
      return body;
    },
    send: (u, body, method) =>
      fetch(u, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        .then(async (r) => {
          const out = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(out.error || `${r.status} ${r.statusText}`);
          return out;
        }),
  };

  const msg = (t) => { $('msg').textContent = t || ''; };
  window.onerror = (m) => { msg(`JS エラー: ${m}`); };
  window.addEventListener('unhandledrejection', (ev) => msg(`JS エラー: ${ev.reason && ev.reason.message}`));

  // --- 姿勢の計算（pose.py の shift_pose / pose_delta と同じ式） ---

  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

  // lateral は左が +、longitudinal は前が +（車両座標）。
  function shiftPose(pose, lateral, longitudinal, dyawDeg, dpitch) {
    const c = Math.cos(pose.yaw);
    const s = Math.sin(pose.yaw);
    return {
      x: pose.x + lateral * -s + longitudinal * c,
      y: pose.y + lateral * c + longitudinal * s,
      yaw: wrap(pose.yaw + dyawDeg * DEG),
      pitch: pose.pitch + dpitch,
    };
  }

  // pose を ref の車両座標で表した差分。
  function poseDelta(pose, ref) {
    const dx = pose.x - ref.x;
    const dy = pose.y - ref.y;
    const c = Math.cos(ref.yaw);
    const s = Math.sin(ref.yaw);
    return {
      lateral: -s * dx + c * dy,
      longitudinal: c * dx + s * dy,
      yawDeg: wrap(pose.yaw - ref.yaw) / DEG,
      pitch: pose.pitch - ref.pitch,
    };
  }

  // --- 不透明度（localStorage は使えない環境もあるので必ず try/catch） ---

  function loadAlpha(key, fallback) {
    try {
      const v = Number(localStorage.getItem(key));
      if (localStorage.getItem(key) !== null && Number.isFinite(v)) return Math.min(100, Math.max(0, v)) / 100;
    } catch (e) { /* 保存できない環境では既定値 */ }
    return fallback;
  }

  function saveAlpha(key, alpha) {
    try { localStorage.setItem(key, String(Math.round(alpha * 100))); } catch (e) { /* 無視 */ }
  }

  const SLIDERS = [
    { id: 'opacity-map', opt: 'mapAlpha', key: 'pa.mapAlpha' },
    { id: 'opacity-mask', opt: 'maskAlpha', key: 'pa.maskAlpha' },
  ];

  function setAlpha(sl, alpha) {
    S.opts[sl.opt] = Math.min(1, Math.max(0, alpha));
    const pct = Math.round(S.opts[sl.opt] * 100);
    $(sl.id).value = pct;
    $(`${sl.id}-val`).textContent = `${pct}%`;
    saveAlpha(sl.key, S.opts[sl.opt]);
    draw();
  }

  SLIDERS.forEach((sl) => {
    S.opts[sl.opt] = loadAlpha(sl.key, S.opts[sl.opt]);
    const el = $(sl.id);
    el.value = Math.round(S.opts[sl.opt] * 100);
    $(`${sl.id}-val`).textContent = `${el.value}%`;
    el.addEventListener('input', () => setAlpha(sl, Number(el.value) / 100));
    // フォーカスが残ると ← → がスライダーを動かしてフレーム移動と二重に効くので外す。
    el.addEventListener('change', () => el.blur());
  });

  // --- 描画 ---

  // backing サイズ = 表示サイズ。左列の幅に収まる倍率（最大 1.5 倍）にする。
  function resizeCanvas() {
    const avail = $('stage').clientWidth;
    S.scale = Math.min(MAX_SCALE, Math.max(0.25, avail / IMG_W));
    canvas.width = Math.round(IMG_W * S.scale);
    canvas.height = Math.round(IMG_H * S.scale);
    canvas.style.width = `${canvas.width}px`;
    canvas.style.height = `${canvas.height}px`;
  }

  function draw() {
    window.Overlay.draw(ctx, {
      img: S.img, maskImg: S.maskImg, overlay: S.view && S.view.overlay, scale: S.scale, opts: S.opts,
    });
  }

  // 俯瞰パネル（bev.js）。ドラッグは 1 回で undo 1 つ（押した時点で積み、動かす間は積まない）。
  let bevPanel = null;
  function drawBev() {
    if (!S.view) return;
    if (!bevPanel) {
      bevPanel = window.Bev.create($('bev'), {
        onDragStart: () => { if (S.frame) pushUndo(); },
        onDrag: (dLateral, dLongitudinal) => {
          if (S.frame) applyPose(shiftPose(S.pose, dLateral, dLongitudinal, 0, 0), 'manual');
        },
        onWheel: (dYawDeg) => { if (S.frame) nudge(0, 0, dYawDeg, 0); },
      });
    }
    bevPanel.draw(S.view.bev);
  }

  function setPpm(dir) {
    bevPanel.setPpm(bevPanel.ppm + dir * 2);
  }

  const fmt = (v, d) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}`;
  const fmtCost = (c) => (c === null || c === undefined ? '—' : `${c.toFixed(2)} m`);

  // --- 候補の列 ---
  // 縮小した画像に候補の地図線・経路を重ね、角にコストと補正量を出す。クリック / 1〜4 で選ぶ。

  const THUMB_SCALE = 200 / IMG_W;
  const THUMB_OPTS = { showMask: false, showLines: true, mapAlpha: 1, maskAlpha: 0, hideAll: false };
  const samePose = (a, b) => !!a && !!b && ['x', 'y', 'yaw', 'pitch'].every((k) => Math.abs(a[k] - b[k]) < 1e-9);

  function renderCandidates(candidates) {
    S.candidates = (candidates || []).slice(0, 4);
    const box = $('candidates');
    box.innerHTML = '';
    S.candidates.forEach((c, i) => {
      const card = document.createElement('div');
      card.className = 'cand';
      const cv = document.createElement('canvas');
      cv.width = 200;
      cv.height = Math.round(IMG_H * THUMB_SCALE);
      const cap = document.createElement('div');
      cap.className = 'cap';
      const cost = c.cost === null || c.cost === undefined ? '—' : c.cost.toFixed(2);
      cap.textContent = `${i + 1}: ${cost} m  横${fmt(c.lateral_m, 2)} 前後${fmt(c.longitudinal_m, 2)} ヨー${fmt(c.yaw_deg, 1)}°`;
      card.title = c.init;
      card.append(cv, cap);
      card.onclick = () => chooseCandidate(i);
      box.appendChild(card);
    });
    drawThumbs();
    highlightCandidate();
  }

  // 画像の読み込みが後から終わったときにも呼ぶ。
  function drawThumbs() {
    [...$('candidates').querySelectorAll('canvas')].forEach((cv, i) => {
      window.Overlay.draw(cv.getContext('2d'), {
        img: S.img, maskImg: null, overlay: S.candidates[i].overlay, scale: THUMB_SCALE, opts: THUMB_OPTS,
      });
    });
  }

  function highlightCandidate() {
    [...$('candidates').children].forEach((card, i) => {
      card.classList.toggle('active', samePose(S.candidates[i].pose, S.pose));
    });
  }

  function chooseCandidate(i) {
    const c = S.candidates[i];
    if (!S.frame || !c) return;
    setPose(c.pose, 'candidate');
  }

  function renderTopbar() {
    const f = S.frame;
    const flags = S.view.flags || [];
    $('tb-name').textContent = `${f.split}/${f.name}`;
    $('tb-index').textContent = `${f.index + 1} / ${f.total}`;
    $('tb-order').textContent = S.flagOnly ? '時系列・要確認のみ' : '時系列';
    const cost = $('tb-cost');
    const bad = flags.length > 0; // どの flag でも要確認
    cost.textContent = `照合 ${fmtCost(S.view.cost)} ${bad ? '要確認' : '良好'}`;
    cost.className = `badge ${bad ? 'warn' : 'good'}`;
    $('tb-flags').textContent = flags.join(' ');
    $('tb-cache').textContent = `cache ${f.cache.done} / ${f.cache.total}`;
  }

  function renderHud() {
    const f = S.frame;
    const d = poseDelta(S.pose, f.init.pose);
    $('hud-correction').textContent =
      `横 ${fmt(d.lateral, 2)} m / 前後 ${fmt(d.longitudinal, 2)} m\nヨー ${fmt(d.yawDeg, 1)}° / ピッチ ${fmt(d.pitch, 1)}°`;
    $('hud-cost').textContent = `${fmtCost(f.init.cost)} → ${fmtCost(S.view.cost)}`;
    const rows = {};
    (S.view.label || []).forEach((e) => { rows[e.class] = e.h_vector.reduce((a, b) => a + b, 0); });
    $('hud-rows').textContent = CLASSES.map((c) => `${c} ${rows[c] === undefined ? '—' : rows[c]}`).join(' / ');
    $('hud-source').textContent = `${S.source}（初期値 ${f.init.source}）`;
  }

  function renderQueue() {
    const ol = $('queue-list');
    ol.innerHTML = '';
    S.idx = S.frame ? S.items.findIndex((q) => q.split === S.frame.split && q.name === S.frame.name) : -1;
    S.items.forEach((q, i) => {
      const li = document.createElement('li');
      li.className = (i === S.idx ? 'active ' : '') + (q.status === 'done' ? 'done' : '');
      if (q.flags.length) {
        const mark = document.createElement('span');
        mark.className = 'mark';
        mark.textContent = '●';
        li.appendChild(mark);
      }
      li.appendChild(document.createTextNode(q.name));
      li.title = [q.status, ...q.flags].join(' ');
      li.onclick = () => leaveTo({ split: q.split, name: q.name });
      ol.appendChild(li);
    });
    if (ol.children[S.idx]) ol.children[S.idx].scrollIntoView({ block: 'nearest' });
  }

  function renderAll() {
    draw(); drawBev(); renderTopbar(); renderHud();
  }

  // --- サーバとの往復 ---

  async function loadQueue() {
    const q = `/api/queue?order=${S.order}${S.flagOnly ? '&flag=any' : ''}`;
    const body = await api.get(q);
    S.items = body.items;
    renderQueue();
  }

  async function openFrame(ref) {
    let f;
    try {
      f = await api.get(`/api/frame/${ref.split}/${ref.name}`);
    } catch (e) {
      // 前のフレームの表示・状態はそのまま保ち、失敗だけを出す。
      msg(`フレームの読み込みに失敗: ${e.message}`);
      return;
    }
    msg('');
    S.frame = f;
    S.pose = f.pose;
    S.source = f.source;
    S.undo = []; S.redo = [];
    S.latest = null;
    S.view = { overlay: f.overlay, bev: f.bev, cost: f.cost, label: f.label, flags: f.flags };
    const img = new Image();
    const maskImg = new Image();
    // 応答が前後しても、今のフレームの画像だけを描く。
    img.onload = () => { if (S.img === img) { draw(); drawThumbs(); } };
    maskImg.onload = () => { if (S.maskImg === maskImg) draw(); };
    S.img = img; S.maskImg = maskImg;
    img.src = `/api/image/${f.split}/${f.name}`;
    maskImg.src = `/api/mask/${f.split}/${f.name}`;
    renderAll();
    renderCandidates(f.candidates);
    renderQueue();
  }

  // 応答待ちの間は最新の姿勢を S.latest に置くだけにし、応答後に差があれば送り直す
  // （キーリピートで要求が溜まらない。旧 mousemove の pending 方式と同じ）。
  function preview() {
    S.latest = S.pose;
    if (S.pending) return;
    S.pending = true;
    (async () => {
      while (S.latest) {
        const pose = S.latest;
        const f = S.frame;
        S.latest = null;
        try {
          const body = await api.send('/api/preview', { split: f.split, name: f.name, pose }, 'POST');
          if (S.frame !== f) continue; // 応答を待つ間にフレームが変わった
          S.view = body;
          renderAll();
        } catch (e) {
          msg(`プレビューに失敗: ${e.message}`);
        }
      }
      S.pending = false;
    })();
  }

  function pushUndo() {
    S.undo.push({ pose: S.pose, source: S.source });
    if (S.undo.length > 200) S.undo.shift();
    S.redo = [];
  }

  // undo には積まずに姿勢を変える（俯瞰のドラッグ中・undo/redo 用）。
  function applyPose(pose, source) {
    S.pose = pose;
    S.source = source;
    renderHud();
    highlightCandidate();
    preview();
  }

  function setPose(pose, source) {
    pushUndo();
    applyPose(pose, source);
  }

  function nudge(lateral, longitudinal, dyawDeg, dpitch) {
    setPose(shiftPose(S.pose, lateral, longitudinal, dyawDeg, dpitch), 'manual');
  }

  function history(from, to) {
    if (!from.length) return msg(from === S.undo ? 'これ以上戻せません' : 'やり直せる操作がありません');
    to.push({ pose: S.pose, source: S.source });
    const h = from.pop();
    applyPose(h.pose, h.source);
  }

  const savePose = () =>
    api.send('/api/pose', { split: S.frame.split, name: S.frame.name, pose: S.pose, source: S.source }, 'POST');

  const reloadQueue = () => loadQueue().catch((e) => msg(`キューの読み込みに失敗: ${e.message}`)); // flags・状態が変わりうる

  // 移動系の操作は 1 つずつ。POST からフレーム読み込み・キュー更新までを通して busy にし、
  // 応答待ちの間に押された Enter / s / ← → / キューのクリックは捨てる。
  async function exclusive(fn) {
    if (S.busy) return;
    S.busy = true;
    try {
      await fn();
    } finally {
      S.busy = false;
    }
  }

  // 前後のフレーム（dir = -1 / +1）。要確認のみの間はその一覧の中で前後に動き、
  // 今のフレームが一覧に無いときだけ時系列の前後にする。無ければ null。
  function neighbor(dir) {
    const f = S.frame;
    if (S.flagOnly) {
      const i = S.items.findIndex((q) => q.split === f.split && q.name === f.name);
      if (i >= 0) {
        const q = S.items[i + dir];
        return q ? { split: q.split, name: q.name } : null;
      }
    }
    return dir > 0 ? f.next : f.prev;
  }

  const noNeighbor = () => msg(S.flagOnly ? '要確認のフレームはこれ以上ありません' : 'これ以上フレームがありません');

  // 確定せずに別フレームへ移る。今の姿勢は保存しておく（最初の読み込みに失敗して S.frame が無ければ保存しない）。
  const leaveTo = (ref) => exclusive(async () => {
    if (!ref) return noNeighbor();
    if (S.frame) {
      try {
        await savePose();
      } catch (e) {
        return msg(`姿勢の保存に失敗: ${e.message}`);
      }
    }
    try {
      await openFrame(ref);
    } finally {
      await reloadQueue(); // 確定済みのフレームを動かすと unseen に戻るので、開けなくても一覧を更新する
    }
  });

  const confirm = () => exclusive(async () => {
    const f = S.frame;
    const flagNext = S.flagOnly ? neighbor(+1) : undefined; // 確定前の一覧で決める
    let body;
    try {
      body = await api.send('/api/confirm', { split: f.split, name: f.name, pose: S.pose, source: S.source }, 'POST');
    } catch (e) {
      // 400 は棄却理由（straight が投影できない・可視行が少ない等）。姿勢を直すか s で skip。
      return msg(`確定できません: ${e.message}`);
    }
    const next = flagNext === undefined ? body.next : flagNext;
    if (next) await openFrame(next);
    else msg(S.flagOnly ? '確定しました（要確認のフレームはこれ以上ありません）' : '最後のフレームを確定しました');
    await reloadQueue();
  });

  const skip = () => exclusive(async () => {
    const f = S.frame;
    const next = neighbor(+1);
    try {
      await api.send('/api/status', { split: f.split, name: f.name, status: 'skip' }, 'POST');
    } catch (e) {
      return msg(`skip に失敗: ${e.message}`);
    }
    if (next) await openFrame(next);
    else msg(S.flagOnly ? 'skip しました（要確認のフレームはこれ以上ありません）' : '最後のフレームを skip しました');
    await reloadQueue();
  });

  // 今の姿勢から広い窓で照合し直し、最良の候補へ移して候補の列を作り直す。
  const rematch = () => exclusive(async () => {
    const f = S.frame;
    msg('再探索中…');
    let body;
    try {
      body = await api.send('/api/match', { split: f.split, name: f.name, pose: S.pose, window: 'wide' }, 'POST');
    } catch (e) {
      return msg(`再探索に失敗: ${e.message}`);
    }
    if (S.frame !== f) return;
    if (!body.candidates.length) return msg('白線点が少なく再探索できません');
    setPose(body.pose, 'candidate');
    renderCandidates(body.candidates);
    msg(`再探索: コスト ${fmtCost(body.cost)}`);
  });

  async function toggleFlagOnly() {
    S.flagOnly = !S.flagOnly;
    try {
      await loadQueue();
    } catch (e) {
      S.flagOnly = !S.flagOnly;
      return msg(`キューの読み込みに失敗: ${e.message}`);
    }
    renderTopbar();
    msg(S.flagOnly ? `要確認のみ: ${S.items.length} 件` : '');
  }

  // --- キーボード操作 ---

  window.addEventListener('keydown', (ev) => {
    if (!S.frame) return;
    // 修飾キー付きはブラウザのショートカット。拾うと Ctrl+L（アドレスバー）で横移動、
    // Ctrl+Z で undo などが、ブラウザ側の動作と同時に走ってしまう。
    if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
    const k = ev.key;
    // CapsLock・Shift で大文字になっても同じキーとして扱い、倍率は ev.shiftKey で決める。
    const lower = k.length === 1 ? k.toLowerCase() : k;
    const n = ev.shiftKey ? 5 : 1;
    // 押しっぱなしの自動リピートで、見ていないフレームを確定・skip・通過しないようにする。
    const once = k === 'Enter' || k === 'ArrowLeft' || k === 'ArrowRight' || lower === 's' || lower === 'f' || lower === 'r';
    if (once && ev.repeat) { ev.preventDefault(); return; }
    if (k === 'Enter') { ev.preventDefault(); return confirm(); }
    if (k === 'ArrowLeft') { ev.preventDefault(); return leaveTo(neighbor(-1)); }
    if (k === 'ArrowRight') { ev.preventDefault(); return leaveTo(neighbor(+1)); }
    if (lower === 'h') return nudge(-STEP.lateral * n, 0, 0, 0);
    if (lower === 'l') return nudge(STEP.lateral * n, 0, 0, 0);
    if (lower === 'j') return nudge(0, -STEP.longitudinal * n, 0, 0);
    if (lower === 'k') return nudge(0, STEP.longitudinal * n, 0, 0);
    if (lower === 'u') return nudge(0, 0, -STEP.yawDeg * n, 0);
    if (lower === 'i') return nudge(0, 0, STEP.yawDeg * n, 0);
    // Shift+, / Shift+. は '<' / '>' として届く。
    if (k === ',' || k === '<') return nudge(0, 0, 0, -STEP.pitch * n);
    if (k === '.' || k === '>') return nudge(0, 0, 0, STEP.pitch * n);
    if (k === 'Escape') return setPose(S.frame.init.pose, S.frame.init.source);
    if (lower === 'z') return ev.shiftKey ? history(S.redo, S.undo) : history(S.undo, S.redo);
    if (lower === 'g') { S.opts.showMask = !S.opts.showMask; return draw(); }
    if (lower === 'm') { S.opts.showLines = !S.opts.showLines; return draw(); }
    if (k === '[') return setAlpha(SLIDERS[0], S.opts.mapAlpha - 0.1);
    if (k === ']') return setAlpha(SLIDERS[0], S.opts.mapAlpha + 0.1);
    if (lower === 'o') {
      if (!S.opts.hideAll) { S.opts.hideAll = true; draw(); }
      return;
    }
    if (lower === 's') return skip();
    if (lower === 'f') return toggleFlagOnly();
    if (lower === 'r') return rematch();
    if (k >= '1' && k <= '4' && k.length === 1) return chooseCandidate(Number(k) - 1);
    // '=' は Shift 無しの '+'（US 配列）。
    if (k === '+' || k === '=') return setPpm(+1);
    if (k === '-') return setPpm(-1);
  });

  // o は押している間だけ。離したとき（またはウィンドウを離れたとき）に戻す。
  function showOverlay() {
    if (S.opts.hideAll) { S.opts.hideAll = false; draw(); }
  }
  window.addEventListener('keyup', (ev) => { if (ev.key.toLowerCase() === 'o') showOverlay(); });
  window.addEventListener('blur', showOverlay);

  window.addEventListener('resize', () => { resizeCanvas(); draw(); });

  // --- 起動 ---

  async function start() {
    resizeCanvas();
    await loadQueue();
    // 続きは最初の unseen から。無ければ最初の done 以外（skip）、それも無ければ先頭。
    const first = S.items.find((q) => q.status === 'unseen') || S.items.find((q) => q.status !== 'done') || S.items[0];
    if (!first) return msg('フレームがありません');
    await openFrame(first);
  }

  start()
    .then(() => {
      const m = /[?&]selftest=(\d)/.exec(location.search);
      if (m) selftest(Number(m[1]));
    })
    .catch((e) => msg(`キューの読み込みに失敗: ${e.message}`));

  // --- 自己テスト（?selftest=N のときだけ。ヘッドレスのスクリーンショット確認用） ---
  // 実際の keydown/keyup を window に投げ、結果を #msg に書き出す。
  // 1: 姿勢キー・undo/redo・表示切替、2: Enter で確定して次へ、3: 確定できない姿勢で Enter（400）、
  // 4: 不透明度を変えて保存（再読み込みで保たれるかの確認用）、
  // 5: 俯瞰のドラッグ・ホイール・候補 2 の選択、6: 5 に続けて r で再探索・+ で縮尺を上げる、
  // 7: ← で前のフレームへ移って候補の枚数を見る。

  async function selftest(mode) {
    const key = (type, k, shiftKey = false) => window.dispatchEvent(new KeyboardEvent(type, { key: k, shiftKey }));
    const wait = (r) => new Promise((ok) => setTimeout(ok, r));
    const idle = async () => { while (S.pending || S.busy) await wait(50); };
    if (mode === 4) {
      key('keydown', '['); key('keydown', '[');
      const el = $('opacity-mask');
      el.value = 20;
      el.dispatchEvent(new Event('input'));
      return msg(`SELFTEST 4: map=${$('opacity-map').value} mask=${el.value}`);
    }
    if (mode === 7) {
      key('keydown', 'ArrowLeft'); await wait(300); await idle();
      return msg(`SELFTEST 7: ${S.frame.name} 候補 ${S.candidates.length} 枚 (${S.candidates.map((c) => c.init).join(', ')})`);
    }
    if (mode === 2 || mode === 3) {
      const before = S.frame.name;
      if (mode === 3) setPose({ x: 0, y: 0, yaw: 0, pitch: 0 }, 'manual');
      await idle();
      key('keydown', 'Enter');
      await wait(300); await idle();
      return msg(`SELFTEST ${mode}: ${before} → ${S.frame.name}\n${$('msg').textContent}`);
    }
    const hud = () => $('hud-correction').textContent.replace('\n', ' / ');
    const out = [`start: ${hud()}`];
    if (mode === 5 || mode === 6) {
      const bev = $('bev');
      const r = bev.getBoundingClientRect();
      const mouse = (type, x, y, target = window) =>
        target.dispatchEvent(new MouseEvent(type, { clientX: r.left + x, clientY: r.top + y, button: 0, bubbles: true }));
      const active = () => [...$('candidates').children].findIndex((c) => c.classList.contains('active')) + 1;
      const undo0 = S.undo.length;
      out.push(`cards=${S.candidates.length} active=${active()}`);
      mouse('mousedown', 150, 200, bev);
      mouse('mousemove', 162, 194); mouse('mousemove', 174, 188);
      mouse('mouseup', 174, 188);
      await wait(100); await idle();
      out.push(`drag 右24 上12 (ppm ${bevPanel.ppm}): ${hud()} src=${S.source} undo+${S.undo.length - undo0}`);
      bev.dispatchEvent(new WheelEvent('wheel', { deltaY: -100, bubbles: true, cancelable: true }));
      await idle(); out.push(`wheel 上: ${hud()}`);
      key('keydown', '2'); await idle();
      out.push(`2: ${hud()} src=${S.source} active=${active()}`);
      if (mode === 6) {
        const before = S.candidates.map((c) => c.cost.toFixed(2)).join(',');
        key('keydown', 'r'); await wait(100); await idle();
        out.push(`r: ${$('msg').textContent} 候補 ${before} → ${S.candidates.map((c) => c.cost.toFixed(2)).join(',')} active=${active()} src=${S.source}`);
        key('keydown', '+'); key('keydown', '='); key('keydown', '+'); out.push(`+ = +: ppm=${bevPanel.ppm}`);
        key('keydown', '-'); out.push(`-: ppm=${bevPanel.ppm}`);
      }
      return msg(`SELFTEST ${mode}\n${out.join('\n')}`);
    }
    key('keydown', 'h'); await idle(); out.push(`h: ${hud()} src=${S.source}`);
    key('keydown', 'H', true); await idle(); out.push(`H: ${hud()}`);
    key('keydown', 'i'); key('keydown', '.'); await idle(); out.push(`i .: ${hud()}`);
    key('keydown', 'z'); await idle(); out.push(`z: ${hud()}`);
    key('keydown', 'Z', true); await idle(); out.push(`Z: ${hud()}`);
    key('keydown', 'Escape'); await idle(); out.push(`Esc: ${hud()} src=${S.source}`);
    key('keydown', 'o'); out.push(`o down hideAll=${S.opts.hideAll}`);
    key('keyup', 'o'); out.push(`o up hideAll=${S.opts.hideAll}`);
    key('keydown', 'g'); key('keydown', 'm'); out.push(`g m: mask=${S.opts.showMask} lines=${S.opts.showLines}`);
    key('keydown', 'g'); key('keydown', 'm');
    key('keydown', '['); out.push(`[: map=${$('opacity-map').value}`);
    key('keydown', ']'); out.push(`]: map=${$('opacity-map').value}`);
    msg(`SELFTEST\n${out.join('\n')}`);
  }
})();

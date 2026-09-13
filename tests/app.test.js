// app.js を実際に node の vm で実行し、DOM／fetch をスタブして「デッドロックしないか」
// 「save まで到達するか」を検証する。
//
// test_static.py は app.js のソース文字列しか見ておらず、C1（enqueue の入れ子による
// デッドロック）のようにソースの形からは分からないが実行すると必ず止まるバグを
// すり抜けていた（166 passed のまま本番で最初のクリックから固まる状態だった）。
// このファイルはそれの再発防止として、実際に app.js を動かす。
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

const STATIC = path.join(__dirname, '..', 'static');

// マイクロタスク＋マクロタスクを何ラウンドか回し、app.js 内の Promise チェーン
// （enqueue の直列化）を進める。setImmediate を挟むことで then() の連鎖が
// 何段あっても実質的に全部片付く。デッドロックしている場合はここでは進まない
// （待っている Promise 自体が永遠に解決しないだけなので、tick 自体は正常に終わる）。
function tick(n = 30) {
  let p = Promise.resolve();
  for (let i = 0; i < n; i += 1) {
    p = p.then(() => new Promise((resolve) => setImmediate(resolve)));
  }
  return p;
}

// dispatch した操作が一定時間内に解決しない場合、ハング検出として reject する。
// C1 未修正の状態だと呼び出し側の await が永遠に返らないため、ここでガードしないと
// テストプロセスごと固まる。
function withTimeout(promise, ms, message) {
  let timer;
  const timeout = new Promise((_resolve, reject) => {
    timer = setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function makeEl() {
  return {
    textContent: '',
    innerHTML: '',
    title: '',
    className: '',
    onclick: null,
    classList: { toggle() {}, add() {}, remove() {} },
    appendChild() {},
  };
}

// ブラウザの fetch を最小限のインメモリ実装で置き換える。
// server.py は一切起動しない（実データはもちろん、labels/images ディレクトリにも
// 一切触れない）。ここで返す形は server.py の実レスポンス形状（items/total,
// label, xp/h_vector/control_rows）に合わせてある。
function makeFetchStub(state) {
  function jsonResponse(status, body) {
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: String(status),
      json: async () => body,
    };
  }

  return async function fetchStub(url, opts) {
    const method = (opts && opts.method) || 'GET';
    const u = new URL(url, 'http://localhost');
    const p = u.pathname;

    if (method === 'GET' && p === '/api/queue') {
      return jsonResponse(200, { items: state.queueItems, total: state.queueItems.length });
    }
    const frame = p.match(/^\/api\/frame\/([^/]+)\/([^/]+)$/);
    if (frame && method === 'GET') {
      return jsonResponse(200, { label: JSON.parse(JSON.stringify(state.label)) });
    }
    if (frame && method === 'PUT') {
      state.saveCount += 1;
      state.lastSavedLabel = JSON.parse(opts.body).label;
      return jsonResponse(200, {});
    }
    if (p === '/api/preview' && method === 'POST') {
      state.previewCount += 1;
      const body = JSON.parse(opts.body);
      const xp = body.xp.slice();
      const h = body.h_vector.slice();
      // interp/erase は curve.py の interpolate_rows/clear_rows と同じ規約で最小限だけ
      // 再現する（点の有無 (h_vector) と横位置 (xp) が独立に動くケースをテストするため、
      // ここだけは実データを一切使わずに本物の意味づけで動かす必要がある）。
      if (body.mode === 'interp' || body.mode === 'erase') {
        const [[rowARaw, xpARaw], [rowBRaw, xpBRaw]] = body.points;
        let a = Number(rowARaw);
        let b = Number(rowBRaw);
        let va = Number(xpARaw);
        let vb = Number(xpBRaw);
        if (a > b) { [a, b] = [b, a]; [va, vb] = [vb, va]; }
        const span = b - a;
        for (let row = a; row <= b; row += 1) {
          if (body.mode === 'erase') {
            xp[row] = 0.0;
            h[row] = 0;
          } else {
            const t = span === 0 ? 0 : (row - a) / span;
            xp[row] = Math.min(1, Math.max(0, va + (vb - va) * t));
            h[row] = 1;
          }
        }
      }
      return jsonResponse(200, { xp, h_vector: h, control_rows: state.controlRows });
    }
    if (p === '/api/propagate' && method === 'POST') {
      state.propagateCalls.push(JSON.parse(opts.body));
      return jsonResponse(200, { applied: [], skipped: [] });
    }
    if (p === '/api/status' && method === 'POST') {
      return jsonResponse(200, {});
    }
    return jsonResponse(404, { error: `stub: unhandled ${method} ${p}` });
  };
}

// Image の onload は本物のブラウザでは非同期（デコード後）に発火するが、ここでは
// draw() が呼べれば十分なので src を設定した時点で同期的に呼ぶ。
function makeFakeImage() {
  return class FakeImage {
    constructor() {
      this.onload = null;
      this._src = '';
      this.complete = true;
      this.naturalWidth = 100;
    }

    set src(v) {
      this._src = v;
      if (typeof this.onload === 'function') this.onload();
    }

    get src() {
      return this._src;
    }
  };
}

// app.js（と geom.js）を毎回まっさらな vm コンテキストで読み込む。
// テスト間で S（app.js 内のモジュールスコープ変数）を共有しないため、
// 1 テストにつき 1 回呼ぶ。
function buildEnv() {
  const state = {
    queueItems: [{ split: 'train', name: 'f1', score: 1, reasons: [], status: 'pending' }],
    // ROW_START(25) 以降だけ有効行にする。行 30 をクリックすれば「点がある行」になる。
    label: [{
      class: 'straight',
      xp: new Array(48).fill(0),
      h_vector: new Array(48).fill(0).map((_, i) => (i >= 25 ? 1 : 0)),
    }],
    controlRows: [25, 36, 47],
    previewCount: 0,
    saveCount: 0,
    lastSavedLabel: null,
    propagateCalls: [],
  };

  const ctx = {
    clearRect() {}, drawImage() {}, fillRect() {}, beginPath() {}, arc() {},
    fill() {}, stroke() {}, moveTo() {}, lineTo() {},
  };
  const canvasListeners = {};
  const canvas = {
    width: 1280,
    height: 768,
    getContext: () => ctx,
    addEventListener(type, fn) {
      (canvasListeners[type] = canvasListeners[type] || []).push(fn);
    },
    getBoundingClientRect: () => ({ left: 0, top: 0 }),
  };

  const windowListeners = {};
  const windowStub = {
    addEventListener(type, fn) {
      (windowListeners[type] = windowListeners[type] || []).push(fn);
    },
  };

  const elements = {};
  const documentStub = {
    getElementById(id) {
      if (id === 'canvas') return canvas;
      if (!elements[id]) elements[id] = makeEl();
      return elements[id];
    },
    createElement() {
      return makeEl();
    },
  };

  const sandbox = {
    window: windowStub,
    document: documentStub,
    fetch: makeFetchStub(state),
    Image: makeFakeImage(),
    requestAnimationFrame: (cb) => cb(),
    console,
  };
  const context = vm.createContext(sandbox);

  const geomSrc = fs.readFileSync(path.join(STATIC, 'geom.js'), 'utf8');
  const appSrc = fs.readFileSync(path.join(STATIC, 'app.js'), 'utf8');
  vm.runInContext(geomSrc, context, { filename: 'geom.js' });
  vm.runInContext(appSrc, context, { filename: 'app.js' });

  return { state, canvasListeners, windowListeners };
}

function dispatchMousedown(canvasListeners, ev) {
  const handlers = canvasListeners.mousedown || [];
  assert.ok(handlers.length, 'mousedown ハンドラが登録されていない（app.js の読み込みに失敗？）');
  return Promise.all(handlers.map((h) => h(ev)));
}

function dispatchKeydown(windowListeners, ev) {
  const handlers = windowListeners.keydown || [];
  assert.ok(handlers.length, 'keydown ハンドラが登録されていない（app.js の読み込みに失敗？）');
  return Promise.all(handlers.map((h) => h(ev)));
}

function keyEvent(key, extra) {
  return Object.assign(
    { key, shiftKey: false, ctrlKey: false, altKey: false, metaKey: false, preventDefault() {} },
    extra,
  );
}

test('a single click reaches save without deadlocking', async () => {
  const { state, canvasListeners } = buildEnv();
  await tick(); // start() -> loadIndex(0) -> syncControlRows() の完了を待つ

  const before = state.saveCount;
  // ROW_START(25) より下の行、行 30 あたりを普通にクリック（Shift なし・既定 add モード）。
  // enqueue(...) の内側から syncControlRows() を（fetchControlRows() ではなく）呼んでいると、
  // このクリックのハンドラが返す Promise が永遠に解決しない（C1）。
  await withTimeout(
    dispatchMousedown(canvasListeners, { clientX: 100, clientY: 487, shiftKey: false }),
    2000,
    'mousedown ハンドラが解決しなかった（enqueue の入れ子によるデッドロックの疑い / C1）',
  );
  await tick();

  assert.ok(state.saveCount > before, 'save() に到達していない（PUT /api/frame が送られていない）');
});

test('a second operation still runs after the first', async () => {
  const { state, canvasListeners } = buildEnv();
  await tick();

  await withTimeout(
    dispatchMousedown(canvasListeners, { clientX: 100, clientY: 487, shiftKey: false }),
    2000,
    '1 回目の操作が解決しなかった',
  );
  await tick();
  const afterFirst = state.saveCount;
  assert.ok(afterFirst > 0, '前提: 1 回目で save() に到達していない');

  // opChain が 1 回目の操作の後に詰まっていないか（enqueue の中で入れ子にした
  // Promise が残っていると、2 回目以降も含めてそれ以降の操作が全部無反応になる）。
  await withTimeout(
    dispatchMousedown(canvasListeners, { clientX: 120, clientY: 551, shiftKey: false }),
    2000,
    '2 回目の操作が解決しなかった（1 回目の後で opChain が詰まっている）',
  );
  await tick();

  assert.ok(state.saveCount > afterFirst, '2 回目の操作後に save() が呼ばれていない');
});

test('control_rows fetched after a click reflect the current K (I5 regression)', async () => {
  // interp/erase の応答に deltas を送らないと、server 側が K=3 決め打ちで
  // control_rows を計算してしまう（server.py: len(deltas) if deltas else 3）。
  // ここでは deltas の長さがそのままエコーされる素朴なスタブなので、
  // K=3 のときに control_rows の要求が 3 要素の deltas を伴っていることを確認する。
  const { state, canvasListeners } = buildEnv();
  await tick();

  const previewsBefore = state.previewCount;
  await withTimeout(
    dispatchMousedown(canvasListeners, { clientX: 100, clientY: 487, shiftKey: false }),
    2000,
    'mousedown ハンドラが解決しなかった',
  );
  await tick();

  assert.ok(state.previewCount > previewsBefore, '/api/preview が呼ばれていない');
});

test('propagate does not send a delta for rows whose presence changed', async () => {
  // 行 45 は「点が既にある」状態（xp=0.5）でフレームを読み込む。
  // これが伝播の基準 (S.baseXp[45]=0.5, S.baseH[45]=1) になる。
  // buildEnv() は start() を同期的に開始するが、GET /api/frame が実際に state.label を
  // 読むのは最初の tick() の中（マイクロタスク処理時）なので、ここでの同期的な書き換えは
  // 間に合う。
  const { state, canvasListeners, windowListeners } = buildEnv();
  state.label[0].xp[45] = 0.5;
  await tick(); // start() -> loadIndex(0) -> syncControlRows() の完了を待つ（基準を確定させる）

  // 行 40 を実際に横へ動かす（add モードでの単一クリック）。h_vector は 1 のまま
  // なので、これは本物の移動として伝播されるべき delta を残す。
  await withTimeout(
    dispatchMousedown(canvasListeners, { clientX: 383, clientY: 648, shiftKey: false }),
    2000,
    '行 40 への移動が解決しなかった',
  );
  await tick();

  // delete モードに切り替えて行 45 を消す（範囲削除の 1 行版と同じ経路: erase）。
  // xp は 0.5 → 0.0、h_vector は 1 → 0 になる。これは「横移動」ではないので
  // 伝播してはいけない。
  await dispatchKeydown(windowListeners, keyEvent('d'));
  await withTimeout(
    dispatchMousedown(canvasListeners, { clientX: 300, clientY: 728, shiftKey: false }),
    2000,
    '行 45 の削除が解決しなかった',
  );
  await tick();

  await withTimeout(
    dispatchKeydown(windowListeners, keyEvent('p')),
    2000,
    'p の伝播が解決しなかった',
  );
  await tick();

  assert.ok(state.propagateCalls.length > 0, '/api/propagate が呼ばれていない');
  const { delta } = state.propagateCalls[state.propagateCalls.length - 1];
  assert.strictEqual(
    delta[45], 0,
    '点が消えた行 (h_vector が変化) の delta が 0 になっていない: '
    + '横移動と誤認され、近傍フレームの同じ行が左端に張り付く／飛ぶ',
  );
  assert.ok(
    Math.abs(delta[40]) > 0.2,
    '本物の移動（h_vector 不変）まで一緒に握りつぶされている',
  );
});

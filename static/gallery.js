// 一覧ページ。/api/gallery の表をグリッドに描き、クリックで /annotate（ギャラリーモード）を iframe で開く。
// iframe とは同一オリジンの postMessage でやり取りする（queue / ready / saved / status / current / close）。
(function () {
  const $ = (id) => document.getElementById(id);
  const SELFTEST = /[?&]selftest=(\d+)/.exec(location.search);
  const BATCH = 120;
  const FILTER_IDS = ['f-split', 'f-bag', 'f-status', 'f-flag', 'f-src', 'f-cls', 'f-edited', 'f-manual', 'f-relabel', 'f-sort', 'f-q', 'f-size'];
  const HASH_KEYS = { 'f-split': 'split', 'f-bag': 'bag', 'f-status': 'status', 'f-flag': 'flag', 'f-src': 'src', 'f-cls': 'cls',
    'f-edited': 'edited', 'f-manual': 'manual', 'f-relabel': 'relabel', 'f-sort': 'sort', 'f-q': 'q', 'f-size': 'w' };

  let DATA = [];          // /api/gallery の frames（key で引ける）
  let BY_KEY = new Map();
  let VIEW = [];          // 絞り込み・並べ替え後
  let cursor = 0;         // VIEW のうち DOM に出した数
  let QUEUE = [];         // iframe に渡した一覧（絞り込み済み）
  let SAVED_SCROLL = 0;
  let LAST = null;        // 最後に表示していたフレームの key
  const CHANGED = new Set();
  let savedCount = 0;

  const grid = $('grid');
  const frame = $('annot-frame');
  const isOpen = () => !$('annot').hidden;

  // --- 絞り込みの状態（URL のハッシュに保つ） ---

  function readHash() {
    const h = new URLSearchParams(location.hash.replace(/^#/, ''));
    for (const id of FILTER_IDS) {
      const v = h.get(HASH_KEYS[id]);
      if (v !== null) $(id).value = v;
    }
    if (!$('f-sort').value) $('f-sort').value = 'time';
    applySize();
  }

  function writeHash() {
    const h = new URLSearchParams();
    for (const id of FILTER_IDS) {
      const v = $(id).value;
      const def = id === 'f-sort' ? 'time' : id === 'f-size' ? '320' : '';
      if (v !== def) h.set(HASH_KEYS[id], v);
    }
    history.replaceState(null, '', h.toString() ? `#${h}` : location.pathname + location.search);
  }

  function applySize() {
    document.documentElement.style.setProperty('--thumb', `${$('f-size').value}px`);
  }

  // --- 表示 ---

  function thumbUrl(d) {
    return `/api/thumb/${d.split}/${d.name}.jpg?v=${encodeURIComponent(d.thumb_version)}`;
  }

  function chipsHtml(d) {
    const chips = [`<span class="chip status-${d.status}">${d.status}</span>`, `<span class="chip">${d.bag}</span>`];
    if (d.source) chips.push(`<span class="chip">${d.source}</span>`);
    if (d.cost !== null && d.cost !== undefined) chips.push(`<span class="chip">cost ${d.cost.toFixed(3)}</span>`);
    for (const f of d.flags) chips.push(`<span class="chip warn">${f}</span>`);
    for (const c of d.classes) if (c !== 'straight') chips.push(`<span class="chip cls" style="background:var(--c-${c})">${c}</span>`);
    if (d.edited) chips.push('<span class="chip acc">手で直した</span>');
    if (d.has_manual_label) chips.push('<span class="chip">手動</span>');
    if (d.relabeled) chips.push('<span class="chip">再投影で変更</span>');
    if (d.pose !== 'entry') chips.push('<span class="chip pose-none">姿勢なし</span>');
    return chips.join('');
  }

  function card(d) {
    const el = document.createElement('div');
    el.className = 'card' + (d.key === LAST ? ' last' : '');
    el.dataset.key = d.key;
    el.innerHTML = `<img loading="lazy" decoding="async" src="${thumbUrl(d)}" alt="${d.name}">`
      + `<div class="cap"><div class="name">${d.key}</div><div class="sub">${chipsHtml(d)}</div></div>`;
    el.addEventListener('click', () => openAnnotator(d));
    return el;
  }

  function refreshCard(key, withImage) {
    const d = BY_KEY.get(key);
    const el = grid.querySelector(`.card[data-key="${CSS.escape(key)}"]`);
    if (!d || !el) return;
    el.querySelector('.sub').innerHTML = chipsHtml(d);
    if (withImage) el.querySelector('img').src = thumbUrl(d);
  }

  function matches(d) {
    const v = (id) => $(id).value;
    const flag = v('f-flag'), cls = v('f-cls'), q = v('f-q').trim().toLowerCase();
    const tri = (id, actual) => v(id) === '' || actual === (v(id) === '1');
    return (!v('f-split') || d.split === v('f-split'))
      && (!v('f-bag') || d.bag === v('f-bag'))
      && (!v('f-status') || d.status === v('f-status'))
      && (!flag || (flag === 'any' ? d.flags.length > 0 : flag === 'none' ? d.flags.length === 0 : d.flags.includes(flag)))
      && (!v('f-src') || d.source === v('f-src'))
      && (!cls || (cls === 'only' ? d.classes.length === 1 : d.classes.includes(cls)))
      && tri('f-edited', !!d.edited) && tri('f-manual', !!d.has_manual_label) && tri('f-relabel', !!d.relabeled)
      && (!q || d.name.toLowerCase().includes(q));
  }

  function applyFilter() {
    VIEW = DATA.filter(matches);
    const sort = $('f-sort').value;
    if (sort === 'name') VIEW.sort((a, b) => a.key.localeCompare(b.key));
    else if (sort === 'cost') VIEW.sort((a, b) => (b.cost ?? -1) - (a.cost ?? -1) || a.index - b.index);
    else VIEW.sort((a, b) => a.index - b.index);
    grid.innerHTML = '';
    cursor = 0;
    $('count').textContent = `${VIEW.length} / ${DATA.length} 枚`;
    $('empty').hidden = VIEW.length > 0;
    renderMore();
    writeHash();
  }

  function renderMore() {
    const frag = document.createDocumentFragment();
    const end = Math.min(cursor + BATCH, VIEW.length);
    for (let i = cursor; i < end; i++) frag.appendChild(card(VIEW[i]));
    cursor = end;
    grid.appendChild(frag);
  }

  new IntersectionObserver((es) => { if (es[0].isIntersecting && cursor < VIEW.length) renderMore(); }, { rootMargin: '1200px' }).observe($('sentinel'));

  // --- アノテーション画面（iframe） ---

  function openAnnotator(d) {
    if (isOpen()) return;
    QUEUE = VIEW.map((r) => ({ split: r.split, name: r.name, status: r.status, flags: r.flags }));
    SAVED_SCROLL = window.scrollY;
    LAST = d.key;
    $('annot').hidden = false;
    document.body.style.overflow = 'hidden';
    frame.src = `/annotate?mode=gallery&frame=${encodeURIComponent(d.split)}/${encodeURIComponent(d.name)}${SELFTEST ? '&selftest=11' : ''}`;
    frame.addEventListener('load', () => { try { frame.contentWindow.focus(); } catch (e) { /* 無視 */ } }, { once: true });
  }

  function closeAnnotator() {
    frame.src = 'about:blank';
    $('annot').hidden = true;
    document.body.style.overflow = '';
    window.scrollTo(0, SAVED_SCROLL);
    for (const key of CHANGED) refreshCard(key, true);
    for (const el of grid.querySelectorAll('.card.last')) el.classList.remove('last');
    if (LAST) { const el = grid.querySelector(`.card[data-key="${CSS.escape(LAST)}"]`); if (el) el.classList.add('last'); }
    const changed = CHANGED.size;
    CHANGED.clear();
    if (SELFTEST) $('count').textContent = `SELFTEST 1: opened=${LAST} queue=${QUEUE.length} saved=${savedCount} closed=1 changed=${changed}`;
  }

  window.addEventListener('message', (ev) => {
    if (ev.origin !== location.origin || !ev.data || ev.source !== frame.contentWindow) return;
    const m = ev.data;
    if (m.type === 'ready') {
      const cur = LAST ? BY_KEY.get(LAST) : null;
      frame.contentWindow.postMessage({ type: 'queue', items: QUEUE, current: cur ? { split: cur.split, name: cur.name } : null }, location.origin);
    } else if (m.type === 'saved') {
      const d = BY_KEY.get(m.key);
      if (d) {
        Object.assign(d, { status: m.status, flags: m.flags || [], cost: m.cost, classes: m.classes || [], rows: m.rows,
          edited: m.edited, thumb_version: m.thumb_version, pose: 'entry' });
        CHANGED.add(m.key);
        refreshCard(m.key, false);
      }
      savedCount += 1;
    } else if (m.type === 'status') {
      const d = BY_KEY.get(m.key);
      if (d) { d.status = m.status; CHANGED.add(m.key); refreshCard(m.key, false); }
    } else if (m.type === 'current') {
      LAST = m.key;
      // 開いただけで姿勢が初期化される（姿勢なし → あり）ので、版が変わっていればカードも更新する
      const d = BY_KEY.get(m.key);
      if (d && m.thumb_version && (d.thumb_version !== m.thumb_version || d.pose !== 'entry')) {
        d.thumb_version = m.thumb_version; d.pose = 'entry';
        if (m.status) d.status = m.status;
        CHANGED.add(m.key); refreshCard(m.key, false);
      }
    } else if (m.type === 'close') {
      closeAnnotator();
    }
  });

  // iframe を開いている間は一覧側のキー操作を受けない（キーは iframe に向く）。
  window.addEventListener('keydown', (ev) => { if (isOpen()) ev.stopPropagation(); }, true);

  // --- 起動 ---

  async function main() {
    const r = await fetch('/api/gallery');
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const t = await r.json();
    DATA = t.frames;
    BY_KEY = new Map(DATA.map((d) => [d.key, d]));
    const fill = (id, values) => {
      for (const v of values) { const o = document.createElement('option'); o.value = v; o.textContent = v; $(id).appendChild(o); }
    };
    fill('f-split', [...new Set(DATA.map((d) => d.split))].sort());
    fill('f-bag', t.bags);
    $('l-manual').hidden = !t.has_manifest;
    $('l-relabel').hidden = !t.has_relabeled;
    readHash();
    for (const id of FILTER_IDS) {
      if (id === 'f-size') $(id).addEventListener('input', () => { applySize(); writeHash(); });
      else $(id).addEventListener(id === 'f-q' ? 'input' : 'change', applyFilter);
    }
    applyFilter();
    if (SELFTEST) {
      const first = grid.querySelector('.card');
      if (!first) { $('count').textContent = 'SELFTEST 1: opened=none'; return; }
      first.click();
    }
  }

  main().catch((e) => { $('count').textContent = `読み込みに失敗: ${e.message}`; });
})();

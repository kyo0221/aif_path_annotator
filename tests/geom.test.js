const test = require('node:test');
const assert = require('node:assert');
const geom = require('../static/geom.js');

test('canvas spans the image at the configured scale', () => {
  assert.strictEqual(geom.IMG_W * geom.SCALE, 1280);
  assert.strictEqual(geom.IMG_H * geom.SCALE, 768);
});

test('rowToY puts the row centre at row*8+3.5 scaled (row_anchor_targets(48, 384))', () => {
  assert.strictEqual(geom.rowToY(0), 3.5 * geom.SCALE);
  assert.strictEqual(geom.rowToY(47), (47 * 8 + 3.5) * geom.SCALE);
});

test('yToRow inverts rowToY for every one of the 48 rows', () => {
  for (let row = 0; row < geom.N_ROWS; row += 1) {
    assert.strictEqual(geom.yToRow(geom.rowToY(row)), row);
  }
});

test('yToRow clamps outside the image', () => {
  assert.strictEqual(geom.yToRow(-500), 0);
  assert.strictEqual(geom.yToRow(99999), 47);
});

test('xpToX maps the normalised range onto (IMG_W - 1) px (label convention u/(width-1))', () => {
  assert.strictEqual(geom.xpToX(0), 0);
  assert.strictEqual(geom.xpToX(1), (geom.IMG_W - 1) * geom.SCALE);
  assert.strictEqual(geom.xpToX(0.5), ((geom.IMG_W - 1) * geom.SCALE) / 2);
});

test('xToXp inverts xpToX', () => {
  assert.ok(Math.abs(geom.xToXp(geom.xpToX(0.37)) - 0.37) < 1e-12);
});

test('xToXp clamps to the unit range', () => {
  assert.strictEqual(geom.xToXp(-100), 0);
  assert.strictEqual(geom.xToXp(999999), 1);
});

test('hitTest finds the control point under the cursor', () => {
  const xp = new Array(48).fill(0.5);
  const rows = [25, 36, 47];
  const x = geom.xpToX(0.5);
  const y = geom.rowToY(36);
  assert.strictEqual(geom.hitTest(x, y, rows, xp, 12), 1);
});

test('hitTest returns -1 when nothing is close enough', () => {
  const xp = new Array(48).fill(0.5);
  assert.strictEqual(geom.hitTest(0, 0, [25, 36, 47], xp, 12), -1);
});

test('hitTest prefers the nearest control point', () => {
  const xp = new Array(48).fill(0.5);
  const rows = [25, 26, 47];
  const y = geom.rowToY(25) + 1;
  assert.strictEqual(geom.hitTest(geom.xpToX(0.5), y, rows, xp, 40), 0);
});

test('ROW_START is exported so the app can grey out and lock rows above the head input', () => {
  assert.strictEqual(geom.ROW_START, 25);
});

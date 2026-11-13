// Tile builder. Parses one 1 km tile out of a chunk file and turns footprints and
// road centre-lines into renderable buffers. Runs off the main thread.
// Grid coords: x east, y north (metres). Scene coords: (x, up, -y), tile-local.
import { ShapeUtils, Vector2 } from "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";

let CFG = null, HG = null, sky = null;
const skyWaiters = [], chunkCache = new Map();
const CHUNK_MAGIC = 1144209997, SKY_MAGIC = 1263747661;

function ground(x, y) {
  const g = CFG.hg;
  const fx = Math.min(g.w - 1.001, Math.max(0, (x - g.x0) / g.res)), fy = Math.min(g.h - 1.001, Math.max(0, (g.y1 - y) / g.res));
  const ix = fx | 0, iy = fy | 0, tx = fx - ix, ty = fy - iy, i = iy * g.w + ix;
  return (HG[i] * (1 - tx) + HG[i + 1] * tx) * (1 - ty) + (HG[i + g.w] * (1 - tx) + HG[i + g.w + 1] * tx) * ty;
}

async function chunk(cx, cy) {
  const k = cx + "_" + cy;
  if (!chunkCache.has(k)) {
    chunkCache.set(k, fetch(`${CFG.base}c_${cx}_${cy}.bin`).then(r => { if (!r.ok) throw new Error("chunk " + k + " HTTP " + r.status); return r.arrayBuffer(); }));
    if (chunkCache.size > 12) chunkCache.delete(chunkCache.keys().next().value);
  }
  return chunkCache.get(k);
}

function setSky(buf) {
  sky = new Map();
  const dv = new DataView(buf);
  if (buf.byteLength > 8 && dv.getUint32(0, true) === SKY_MAGIC)
    for (let i = 0, n = dv.getUint16(6, true); i < n; i++) {
      const e = 8 + i * 12, off = dv.getUint32(e + 4, true);
      sky.set(dv.getUint16(e, true) + "_" + dv.getUint16(e + 2, true), buf.slice(off, off + dv.getUint32(e + 8, true)));
    }
  skyWaiters.splice(0).forEach(f => f());
}

// ---- geometry accumulator -------------------------------------------------
class Mesh {
  constructor() { this.pos = []; this.nor = []; this.col = []; this.inf = []; this.idx = []; this.n = 0; }
  v(x, y, up, nx, ny, nup, c, a, b, part, seed) {
    this.pos.push(x, up, -y); this.nor.push(nx * 127, nup * 127, -ny * 127); this.col.push(c[0], c[1], c[2]); this.inf.push(a, b, part, seed);
    return this.n++;
  }
  quad(a, b, c, d) { this.idx.push(a, b, c, a, c, d); }
  out() {
    return { pos: new Float32Array(this.pos), nor: new Int8Array(this.nor), col: new Uint8Array(this.col), inf: new Float32Array(this.inf),
      idx: this.n > 65535 ? new Uint32Array(this.idx) : new Uint16Array(this.idx) };
  }
}

const rnd = (seed, k) => { let n = (seed * 374761393 + k * 668265263) | 0; n = Math.imul(n ^ (n >>> 13), 1274126177); return ((n ^ (n >>> 16)) >>> 0) / 4294967295; };
const lin = v => Math.round(255 * Math.pow(Math.min(255, Math.max(0, v)) / 255, 2.2));   // sRGB byte -> linear byte

// Wall palettes by building type (pipeline: 0 industrial, 1 low residential, 2 mid residential, 3 high-rise,
// 4 commercial, 5 civic, 6 religious, 7 transport, 9 stadium). Noida reads cream, grey and pale ochre.
const WALLS = {
  0: [[170, 172, 176], [150, 158, 168], [186, 178, 164]],
  1: [[226, 218, 200], [214, 204, 186], [232, 226, 214], [206, 190, 170], [198, 204, 206], [222, 200, 176]],
  2: [[230, 224, 210], [216, 208, 196], [204, 198, 190], [224, 206, 184], [196, 204, 212]],
  3: [[232, 228, 220], [208, 210, 212], [220, 208, 192], [186, 196, 206], [236, 224, 204]],
  4: [[122, 148, 170], [104, 128, 146], [150, 166, 178], [176, 180, 184]],
  5: [[214, 196, 172], [200, 188, 176], [222, 214, 200]],
  6: [[238, 232, 220], [226, 204, 170]],
  7: [[176, 180, 186]],
  9: [[200, 204, 208]],
};

function ringArea(r) { let a = 0; for (let i = 0, n = r.length; i < n; i += 2) { const j = (i + 2) % n; a += r[i] * r[j + 1] - r[j] * r[i + 1]; } return a / 2; }
function inRing(r, x, y) {
  let c = false;
  for (let i = 0, n = r.length, j = n - 2; i < n; j = i, i += 2)
    if ((r[i + 1] > y) !== (r[j + 1] > y) && x < (r[j] - r[i]) * (y - r[i + 1]) / (r[j + 1] - r[i + 1]) + r[i]) c = !c;
  return c;
}

// Far tiles: drop vertices that sit within `tol` metres of the line through their neighbours.
function thin(r, tol) {
  const n = r.length / 2; if (n <= 4) return r;
  const out = [r[0], r[1]];
  for (let i = 1; i < n; i++) {
    const ax = out[out.length - 2], ay = out[out.length - 1], px = r[i * 2], py = r[i * 2 + 1], j = (i + 1) % n, dx = r[j * 2] - ax, dy = r[j * 2 + 1] - ay;
    if (Math.abs((px - ax) * dy - (py - ay) * dx) / (Math.hypot(dx, dy) || 1) >= tol) out.push(px, py);
  }
  return out.length >= 6 ? new Float32Array(out) : r;
}

function box(M, cx, cy, z0, z1, hx, hy, ang, c, part, seed) {
  const ca = Math.cos(ang), sa = Math.sin(ang);
  const P = [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].map(([a, b]) => [cx + a * ca - b * sa, cy + a * sa + b * ca]);
  for (let i = 0; i < 4; i++) {
    const a = P[i], b = P[(i + 1) % 4], dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy) || 1, nx = dy / l, ny = -dx / l;
    M.quad(M.v(a[0], a[1], z0, nx, ny, 0, c, 0, 0, part, seed), M.v(b[0], b[1], z0, nx, ny, 0, c, l, 0, part, seed),
      M.v(b[0], b[1], z1, nx, ny, 0, c, l, z1 - z0, part, seed), M.v(a[0], a[1], z1, nx, ny, 0, c, 0, z1 - z0, part, seed));
  }
  const t = P.map(p => M.v(p[0], p[1], z1, 0, 0, 1, c, 0, 0, part, seed));
  M.quad(t[0], t[1], t[2], t[3]);
}

function building(M, rings, base, h, type, r565, seed, lod) {
  const top = base + h, pal = WALLS[type] || WALLS[1], w = pal[(rnd(seed, 1) * pal.length) | 0], k = 0.92 + rnd(seed, 2) * 0.14;
  const wall = [lin(w[0] * k), lin(w[1] * k), lin(w[2] * k)];
  // Roof: satellite colour pulled toward concrete grey, since 10 m pixels bleed street and tree colour in.
  const rr = (r565 >> 11 & 31) * 8.2, rg = (r565 >> 5 & 63) * 4.05, rb = (r565 & 31) * 8.2, m = 0.35;
  const roof = [lin(rr * (1 - m) + 168 * m), lin(rg * (1 - m) + 164 * m), lin(rb * (1 - m) + 158 * m)];
  rings.forEach((r, i) => { if ((ringArea(r) < 0) === (i === 0)) { for (let a = 0, b = r.length - 2; a < b; a += 2, b -= 2) { [r[a], r[b]] = [r[b], r[a]];[r[a + 1], r[b + 1]] = [r[b + 1], r[a + 1]]; } } });
  const z0 = base - 4;                                    // sink walls so sloping ground never shows a gap
  for (const r of rings) {
    let u = 0;
    for (let i = 0, n = r.length; i < n; i += 2) {
      const j = (i + 2) % n, ax = r[i], ay = r[i + 1], bx = r[j], by = r[j + 1], dx = bx - ax, dy = by - ay, l = Math.hypot(dx, dy);
      if (l < 0.05) continue;
      const nx = dy / l, ny = -dx / l;
      M.quad(M.v(ax, ay, z0, nx, ny, 0, wall, u, -4, 0, seed), M.v(bx, by, z0, nx, ny, 0, wall, u + l, -4, 0, seed),
        M.v(bx, by, top, nx, ny, 0, wall, u + l, h, 0, seed), M.v(ax, ay, top, nx, ny, 0, wall, u, h, 0, seed));
      u += l;
    }
  }
  const first = M.n, toV = r => { const o = []; for (let i = 0; i < r.length; i += 2) o.push(new Vector2(r[i], r[i + 1])); return o; };
  const outer = toV(rings[0]), holes = rings.slice(1).map(toV);
  let tris = [];
  try { tris = ShapeUtils.triangulateShape(outer, holes); } catch { /* degenerate footprint: walls only */ }
  const all = outer.concat(...holes);
  for (const p of all) M.v(p.x, p.y, top, 0, 0, 1, roof, 0, h, 1, seed);
  for (const t of tris) {
    const a = all[t[0]], b = all[t[1]], c = all[t[2]], ccw = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
    M.idx.push(first + t[0], first + (ccw ? t[1] : t[2]), first + (ccw ? t[2] : t[1]));
  }
  if (lod === 0 && h > 9 && type !== 0) {                 // rooftop clutter: stair head and water tanks
    const r = rings[0], area = Math.abs(ringArea(r));
    if (area > 90) {
      let mx = 0, my = 0; const n = r.length / 2;
      for (let i = 0; i < r.length; i += 2) { mx += r[i] / n; my += r[i + 1] / n; }
      const ang = Math.atan2(r[3] - r[1], r[2] - r[0]), s = Math.min(4, Math.sqrt(area) * 0.16);
      for (let q = 0, made = 0; q < 6 && made < (area > 600 ? 3 : 1); q++) {
        const px = mx + (rnd(seed, 10 + q) - 0.5) * Math.sqrt(area) * 0.5, py = my + (rnd(seed, 20 + q) - 0.5) * Math.sqrt(area) * 0.5;
        if (!inRing(r, px + s, py + s) || !inRing(r, px - s, py - s) || !inRing(r, px + s, py - s) || !inRing(r, px - s, py + s)) continue;
        const tank = made > 0;
        box(M, px, py, top, top + (tank ? 1.6 : 2.8), tank ? 1.1 : s, tank ? 1.1 : s * 0.7, ang, tank ? [8, 8, 10] : wall, 2, seed);
        made++;
      }
    }
  }
}

// Road classes from the pipeline: 0 motorway .. 5 residential, 6 service, 7 foot, 8 rail, 9 metro.
const ROAD_COL = [[66, 68, 74], [70, 72, 78], [76, 78, 82], [82, 84, 88], [88, 90, 92], [96, 97, 98], [104, 104, 102], [150, 140, 122], [92, 84, 78], [150, 152, 156]].map(c => c.map(lin));
function road(M, pts, cls, flags, deck, width, ox, oy, lod) {
  const n = pts.length / 2, col = ROAD_COL[cls] || ROAD_COL[5], lift = deck + 0.35 + (9 - cls) * 0.04;
  let u = 0, prev = -1;
  for (let i = 0; i < n; i++) {
    const x = pts[i * 2], y = pts[i * 2 + 1], a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    let dx = pts[b * 2] - pts[a * 2], dy = pts[b * 2 + 1] - pts[a * 2 + 1]; const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
    if (i) u += Math.hypot(x - pts[i * 2 - 2], y - pts[i * 2 - 1]);
    const hw = width / 2, z = ground(ox + x, oy + y) + lift;
    const R = M.v(x + dy * hw, y - dx * hw, z, 0, 0, 1, col, u, 1, cls, width), L = M.v(x - dy * hw, y + dx * hw, z, 0, 0, 1, col, u, -1, cls, width);
    if (prev >= 0) M.idx.push(prev, R, L, prev, L, prev + 1);
    prev = R;
    if (deck > 0 && lod <= 1 && i && Math.floor(u / 32) !== Math.floor((u - Math.hypot(x - pts[i * 2 - 2], y - pts[i * 2 - 1])) / 32))
      box(M, x, y, z - lift - 3, z - 0.6, 0.9, 0.9, Math.atan2(dy, dx), [60, 60, 62], 20, 0);   // viaduct pier
  }
}

function tile(dv, tx, ty, lod, B, R) {
  if (dv.getUint32(0, true) !== CHUNK_MAGIC) throw new Error("bad chunk data");
  const e = 8 + ((ty % 4) * 4 + tx % 4) * 24, offB = dv.getUint32(e, true), nB = dv.getUint32(e + 4, true), nSig = dv.getUint32(e + 8, true),
    nMid = dv.getUint32(e + 12, true), offR = dv.getUint32(e + 16, true), nR = dv.getUint32(e + 20, true);
  const ox = CFG.extent[0] + tx * CFG.tile, oy = CFG.extent[1] + ty * CFG.tile;
  const count = lod >= 2 ? nSig : lod === 1 ? nSig + nMid : nB;
  let o = offB, maxH = 0;
  for (let i = 0; i < count; i++) {
    const h = dv.getUint16(o, true) / 10, base = dv.getInt16(o + 4, true) / 10, r565 = dv.getUint16(o + 6, true), type = dv.getUint8(o + 8), nr = dv.getUint8(o + 10);
    o += 11;
    const rings = [];
    for (let r = 0; r < nr; r++) {
      const n = dv.getUint8(o); o += 1;
      const a = new Float32Array(n * 2);
      for (let k = 0; k < n * 2; k++, o += 2) a[k] = dv.getInt16(o, true) / 16;
      rings.push(a);
    }
    const shape = lod === 0 ? rings : lod === 1 ? rings.map(r => thin(r, 1.2)) : [thin(rings[0], 3)];   // skyline: no courtyards
    if (rings.length && rings[0].length >= 6) building(B, shape, base, h, type, r565, (tx * 7919 + ty * 104729 + i * 31) & 65535, lod);
    if (base + h > maxH) maxH = base + h;
  }
  o = offR;
  for (let i = 0; i < nR; i++) {
    const cls = dv.getUint8(o), flags = dv.getUint8(o + 1), deck = dv.getUint8(o + 2), w = dv.getUint8(o + 5) / 10, np = dv.getUint16(o + 6, true);
    o += 8;
    if (lod === 0 || (lod === 1 && cls !== 6 && cls !== 7) || (lod >= 2 && (cls <= 2 || cls >= 8))) {
      const pts = new Float32Array(np * 2);
      for (let k = 0; k < np * 2; k++) pts[k] = dv.getInt16(o + k * 2, true) / 8;
      road(R, pts, cls, flags, deck, w, ox, oy, lod);
    }
    o += np * 4;
  }
  return maxH;
}

async function build(job) {
  const { tx, ty, lod } = job, cx = tx >> 2, cy = ty >> 2;
  let buf;
  if (lod >= 2) {
    if (!sky) await new Promise(r => skyWaiters.push(r));
    buf = sky.get(cx + "_" + cy);
  } else buf = await chunk(cx, cy);
  const B = new Mesh(), R = new Mesh(), maxH = buf ? tile(new DataView(buf), tx, ty, lod, B, R) : 0;
  const b = B.out(), r = R.out();
  return [{ id: job.id, tx, ty, lod, maxH, b, r }, [b.pos, b.nor, b.col, b.inf, b.idx, r.pos, r.nor, r.col, r.inf, r.idx].map(a => a.buffer)];
}

self.onmessage = async ev => {
  const m = ev.data;
  if (m.type === "init") { CFG = m.cfg; HG = m.hg; return; }
  if (m.type === "sky") return setSky(m.buf);
  try { const [out, tr] = await build(m); self.postMessage(out, tr); }
  catch (err) { self.postMessage({ id: m.id, tx: m.tx, ty: m.ty, lod: m.lod, error: String(err && err.message || err) }); }
};

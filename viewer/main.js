// Noida 3D viewer. Scene coords: x east, y up, z = -north (metres on the pipeline grid).
import * as THREE from "three";
import { MapControls } from "three/addons/controls/MapControls.js";
import { Sky } from "three/addons/objects/Sky.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";

const DATA = window.NOIDA_DATA, $ = id => document.getElementById(id);
// full-detail radius, mid radius, max pixel ratio, bloom, multisamples, shadow map size (0 = no shadows)
const QUALITY = { lite: [700, 1800, 1, 0, 0, 0], low: [1100, 2800, 1, 0, 2, 1024], medium: [1800, 4500, 1.5, 1, 4, 2048], high: [2800, 7000, 2, 1, 4, 2048], ultra: [4200, 11000, 2, 1, 4, 4096] };
// Final colour pass: a little more colour and contrast, a warm cast, soft corners, and a fixed dither against sky banding.
const GRADE = {
  uniforms: { tDiffuse: { value: null } },
  vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: `uniform sampler2D tDiffuse; varying vec2 vUv;
    void main() {
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      c = mix(vec3(dot(c, vec3(0.299, 0.587, 0.114))), c, 1.12);
      c = mix(c, smoothstep(0.0, 1.0, c), 0.18) * vec3(1.02, 1.0, 0.97);
      c *= mix(0.85, 1.0, smoothstep(0.82, 0.3, length((vUv - 0.5) * vec2(1.0, 0.9))));
      float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
      gl_FragColor = vec4(c + (n - 0.5) * 0.014, 1.0);
    }`,
};
const DAY_OF_YEAR = 319;                                    // 15 November, matching the imagery
const U = { uNight: { value: 0 }, uTime: { value: 0 } };
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

const boot = (text, frac) => { $("bootText").textContent = text; $("bootBar").style.width = Math.round(frac * 100) + "%"; };
const fail = e => { console.error(e); $("bootErr").hidden = false; $("bootErr").textContent = "Could not load the model: " + (e.message || e); };

async function pixels(url) {
  const bmp = await createImageBitmap(await (await fetch(url)).blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const c = new OffscreenCanvas(bmp.width, bmp.height), g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(bmp, 0, 0);
  return g.getImageData(0, 0, bmp.width, bmp.height);
}

// height PNG (u16 in R,G) -> node grid sampled every `step` pixels, row 0 = north
function nodeGrid(img, meta, ext, step) {
  const res = meta.res * step, w = Math.round((ext[2] - ext[0]) / res) + 1, h = Math.round((ext[3] - ext[1]) / res) + 1, a = new Float32Array(w * h);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const p = (Math.min(j * step, img.height - 1) * img.width + Math.min(i * step, img.width - 1)) * 4;
    a[j * w + i] = (img.data[p] * 256 + img.data[p + 1]) / meta.scale - meta.offset;
  }
  return { a, w, h, res, x0: ext[0], y1: ext[3] };
}
function sampleGrid(g, x, y) {
  const fx = Math.min(g.w - 1.001, Math.max(0, (x - g.x0) / g.res)), fy = Math.min(g.h - 1.001, Math.max(0, (g.y1 - y) / g.res));
  const ix = fx | 0, iy = fy | 0, tx = fx - ix, ty = fy - iy, i = iy * g.w + ix;
  return (g.a[i] * (1 - tx) + g.a[i + 1] * tx) * (1 - ty) + (g.a[i + g.w] * (1 - tx) + g.a[i + g.w + 1] * tx) * ty;
}
function terrainGeometry(g, i0, j0, ni, nj, uv, drop = 0) {
  const pos = new Float32Array((ni + 1) * (nj + 1) * 3), tex = new Float32Array((ni + 1) * (nj + 1) * 2), idx = new Uint32Array(ni * nj * 6);
  for (let j = 0, k = 0; j <= nj; j++) for (let i = 0; i <= ni; i++, k++) {
    const x = g.x0 + (i0 + i) * g.res, y = g.y1 - (j0 + j) * g.res;
    pos.set([x, g.a[(j0 + j) * g.w + i0 + i] - drop, -y], k * 3);
    tex.set([(x - uv[0]) / uv[2], (y - uv[1]) / uv[3]], k * 2);
  }
  for (let j = 0, q = 0; j < nj; j++) for (let i = 0; i < ni; i++) { const a = j * (ni + 1) + i, b = a + ni + 1; idx.set([a, b, a + 1, a + 1, b, b + 1], q); q += 6; }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setAttribute("uv", new THREE.BufferAttribute(tex, 2)); geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  return geo;
}

// Sun elevation and azimuth (from north, clockwise) for a clock hour in IST.
function sun(hour, lat, lon) {
  const r = Math.PI / 180, decl = -23.44 * r * Math.cos(2 * Math.PI / 365 * (DAY_OF_YEAR + 10)), b = 2 * Math.PI * (DAY_OF_YEAR - 81) / 364;
  const eot = 9.87 * Math.sin(2 * b) - 7.53 * Math.cos(b) - 1.5 * Math.sin(b), ha = 15 * r * (hour + (4 * (lon - 82.5) + eot) / 60 - 12), p = lat * r;
  return { el: Math.asin(Math.sin(p) * Math.sin(decl) + Math.cos(p) * Math.cos(decl) * Math.cos(ha)), az: Math.atan2(-Math.sin(ha), Math.tan(decl) * Math.cos(p) - Math.sin(p) * Math.cos(ha)) };
}

const HASH = "float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }";
function patch(mat, vertDecl, fragDecl, color, emissive) {
  const key = "noida" + patch.n++;
  mat.customProgramCacheKey = () => key;                   // same hook source for every material, so three would share one program
  mat.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, U, mat.userData.uniforms || {});
    sh.vertexShader = sh.vertexShader.replace("#include <common>", "#include <common>\n" + vertDecl.decl).replace("#include <begin_vertex>", "#include <begin_vertex>\n" + vertDecl.main);
    sh.fragmentShader = sh.fragmentShader.replace("#include <common>", "#include <common>\nuniform float uNight; uniform float uTime;\n" + HASH + "\n" + fragDecl)
      .replace("#include <color_fragment>", "#include <color_fragment>\nvec3 glow = vec3(0.);\n" + color)
      .replace("#include <emissivemap_fragment>", "#include <emissivemap_fragment>\ntotalEmissiveRadiance += glow;\n" + (emissive || ""));
  };
  return mat;
}
patch.n = 0;
const INFO_VERT = { decl: "attribute vec4 aInfo; varying vec4 vInfo;", main: "vInfo = aInfo;" };

function buildingMaterial() {
  return patch(new THREE.MeshLambertMaterial({ vertexColors: true }), INFO_VERT, "varying vec4 vInfo;", `
    if (vInfo.z < 0.5 && vInfo.y > 0.0) {                  // walls: a window grid, 2.8 m bays, 3.1 m floors
      vec2 cell = vec2(vInfo.x / 2.8, vInfo.y / 3.1), id = floor(cell), f = fract(cell);
      float fade = clamp(1.6 - fwidth(cell.y) * 3.0, 0.0, 1.0);
      float win = step(0.2, f.x) * step(f.x, 0.8) * step(0.32, f.y) * step(f.y, 0.8) * fade;
      float r = hash(id + vInfo.w * 0.37);
      diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.05, 0.08, 0.11) + 0.05 * r, win * 0.8);
      diffuseColor.rgb *= 1.0 - 0.12 * step(f.y, 0.06) * fade;        // floor slab line
      float on = step(1.0 - 0.42 * uNight, r);
      glow = win * on * mix(vec3(1.0, 0.72, 0.38), vec3(0.75, 0.85, 1.0), step(0.86, hash(id.yx + vInfo.w))) * 1.5 * uNight;
    }`);
}

function roadMaterial() {
  return patch(new THREE.MeshLambertMaterial({ vertexColors: true, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -6 }), INFO_VERT, "varying vec4 vInfo;", `
    if (vInfo.z < 9.5) {
      float cls = vInfo.z, half_w = vInfo.w * 0.5, across = vInfo.y * half_w, along = vInfo.x;
      float near = clamp(1.5 - fwidth(along) * 0.5, 0.0, 1.0);
      if (cls < 4.5) {
        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.7), step(abs(across), 0.16) * step(fract(along / 9.0), 0.45) * near * 0.8);
        float side = sign(across), lane = abs(abs(across) - half_w * 0.5);        // one stream of cars each way, keep-left
        float speed = 9.0 + 6.0 * step(cls, 1.5), s = along - side * uTime * speed, slot = floor(s / 26.0);
        float car = step(0.55, hash(vec2(slot, cls + side))) * step(abs(fract(s / 26.0) - 0.5) * 26.0, 2.1) * step(lane, 0.8) * near;
        vec3 paint = 0.25 + 0.6 * vec3(hash(vec2(slot, 1.0)), hash(vec2(slot, 2.0)), hash(vec2(slot, 3.0)));
        diffuseColor.rgb = mix(diffuseColor.rgb, paint, car * (1.0 - uNight));
        glow += car * uNight * mix(vec3(1.0, 0.1, 0.05), vec3(1.0, 0.95, 0.8), step(0.0, side)) * 1.6;
      }
      if (cls < 5.5) {                                     // street lamps along both kerbs
        vec2 d = vec2(mod(along, 34.0) - 17.0, abs(across) - half_w + 0.6);
        glow += vec3(1.0, 0.62, 0.28) * exp(-dot(d, d) * 0.11) * uNight * 1.1;
      }
    }`);
}

// 10 m satellite imagery is hazy and pale: deepen it so the ground reads as land, not as a faded photograph.
const DEEPEN = "diffuseColor.rgb = mix(vec3(dot(diffuseColor.rgb, vec3(0.3, 0.59, 0.11))), diffuseColor.rgb, 1.35) * 0.8;";
function terrainMaterial(map, mask, ext) {
  if (!mask) return patch(new THREE.MeshLambertMaterial({ map }), { decl: "", main: "" }, "", DEEPEN);
  const m = new THREE.MeshLambertMaterial({ map });
  m.userData.uniforms = { uMask: { value: mask }, uExt: { value: new THREE.Vector4(ext[0], ext[1], ext[2] - ext[0], ext[3] - ext[1]) } };
  return patch(m, { decl: "uniform vec4 uExt; varying vec2 vGrid;", main: "vec4 wp = modelMatrix * vec4(position, 1.0); vGrid = vec2((wp.x - uExt.x) / uExt.z, (-wp.z - uExt.y) / uExt.w);" },
    "uniform sampler2D uMask; varying vec2 vGrid;", `
    float water = texture2D(uMask, vGrid).g;
    float ripple = 0.5 + 0.5 * sin(vGrid.x * 9000.0 + uTime * 0.8) * sin(vGrid.y * 13000.0 - uTime * 0.6) * clamp(1.0 - fwidth(vGrid.y) * 9000.0, 0.0, 1.0);   // fades out before it can alias
    ${DEEPEN}
    diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.05, 0.11, 0.13) + 0.03 * ripple, water * 0.8);`);
}

async function start() {
  boot("Reading the map index", 0.03);
  const M = await (await fetch(DATA + "manifest.json")).json();
  const [X0, Y0, X1, Y1] = M.extent, RE = M.region.extent, TILE = M.tile;
  const total = M.chunks.reduce((s, c) => s + c.n.reduce((a, b) => a + b, 0), 0);
  $("sub").innerHTML = `Noida and Greater Noida · <b>${total.toLocaleString("en-IN")}</b> buildings placed from the map`;

  boot("Reading terrain", 0.08);
  const [hImg, rImg] = await Promise.all([pixels(DATA + M.height.file), pixels(DATA + M.region.height.file)]);
  const G = nodeGrid(hImg, M.height, M.extent, 4), RG = nodeGrid(rImg, M.region.height, RE, 4);
  const groundAt = (x, y) => sampleGrid(G, x, y);

  // ---- renderer, scene, lights -------------------------------------------
  const canvas = $("gl"), renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });   // edges are smoothed by the multisampled composer target
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.shadowMap.autoUpdate = false;   // redrawn only when sun, tiles or view centre change
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 5, 400000);
  scene.fog = new THREE.FogExp2(0xb8c6d4, 0.000014);
  const sky = new Sky(); sky.scale.setScalar(350000); scene.add(sky);
  Object.assign(sky.material.uniforms.turbidity, { value: 4.6 }); sky.material.uniforms.rayleigh.value = 1.7;
  sky.material.uniforms.mieCoefficient.value = 0.006; sky.material.uniforms.mieDirectionalG.value = 0.8;
  const sunLight = new THREE.DirectionalLight(0xffffff, 3), hemi = new THREE.HemisphereLight(0xcfe0ff, 0x8a8070, 1);
  sunLight.castShadow = true; sunLight.shadow.bias = -0.0004;
  scene.add(sunLight, sunLight.target, hemi);
  const under = new THREE.Mesh(new THREE.PlaneGeometry(900000, 900000), new THREE.MeshLambertMaterial({ color: 0x4d4a3c }));   // land colour past the edge of the imagery
  under.rotation.x = -Math.PI / 2; under.position.y = 150; scene.add(under);

  const controls = new MapControls(camera, canvas);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.09, minDistance: 35, maxDistance: 160000, maxPolarAngle: Math.PI / 2 - 0.035, zoomToCursor: true, rotateSpeed: 0.6, zoomSpeed: 1.1 });

  const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: 4 }));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.2, 0.55, 1.1);       // threshold above 1: only lit windows, lamps and sun glints bloom
  composer.addPass(new RenderPass(scene, camera)); composer.addPass(bloom); composer.addPass(new OutputPass()); composer.addPass(new ShaderPass(GRADE));

  const LEVELS = Object.keys(QUALITY);
  let level = "medium", quality = QUALITY[level], auto = true, dirty = true, shDirty = true;
  const touch = () => { dirty = true; };                   // something visible changed: draw one frame
  function resize() {
    dirty = shDirty = true;
    for (const rt of [composer.renderTarget1, composer.renderTarget2]) if (rt.samples !== quality[4]) { rt.samples = quality[4]; rt.dispose(); }
    const sh = sunLight.shadow;
    if (quality[5] && sh.mapSize.x !== quality[5]) { sh.mapSize.set(quality[5], quality[5]); if (sh.map) { sh.map.dispose(); sh.map = null; } }
    const w = canvas.clientWidth, h = canvas.clientHeight, pr = Math.min(devicePixelRatio, quality[2]);
    renderer.setPixelRatio(pr); renderer.setSize(w, h, false); composer.setPixelRatio(pr); composer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  addEventListener("resize", resize);

  // ---- ground -------------------------------------------------------------
  boot("Laying out the ground", 0.15);
  const loader = new THREE.TextureLoader(), tex = (f, srgb = true) => { const t = loader.load(DATA + f, touch); if (srgb) t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t; };
  const mask = tex(M.masks.file, false);
  scene.add(new THREE.Mesh(terrainGeometry(RG, 0, 0, RG.w - 1, RG.h - 1, [RE[0], RE[1], RE[2] - RE[0], RE[3] - RE[1]], 4), terrainMaterial(tex(M.region.sat.file))));
  const satM = M.sat.tile * M.sat.res, per = satM / G.res;
  for (let ty = 0; ty < M.sat.ny; ty++) for (let tx = 0; tx < M.sat.nx; tx++) {
    const geo = terrainGeometry(G, tx * per, (M.sat.ny - 1 - ty) * per, per, per, [X0 + tx * satM, Y0 + ty * satM, satM, satM]);
    const ground = new THREE.Mesh(geo, terrainMaterial(tex(`sat_${tx}_${ty}.jpg`), mask, M.extent));
    ground.receiveShadow = true; scene.add(ground);
  }
  let trees = null;                                         // red channel of the mask, for scattering trees
  pixels(DATA + M.masks.file).then(img => { const a = new Uint8Array(img.width * img.height); for (let i = 0; i < a.length; i++) a[i] = img.data[i * 4]; trees = { a, w: img.width, res: M.masks.res };
    for (const t of tiles) if (t.det && t.detLod === 0) { const tr = plantTrees(t); if (tr) t.det.add(tr); }   // tiles that arrived before the mask
    touch();
  });
  const treeGeo = new THREE.IcosahedronGeometry(1, 0), treeMat = new THREE.MeshLambertMaterial({ color: 0x35552f });
  function plantTrees(t) {
    if (!trees) return null;
    const n = TILE / trees.res, c0 = (t.ox - X0) / trees.res, r0 = (Y1 - t.oy) / trees.res - n, mats = [], m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      if (trees.a[(r0 + j) * trees.w + c0 + i] < 128) continue;
      const h = Math.sin((c0 + i) * 12.9898 + (r0 + j) * 78.233) * 43758.5453, r = h - Math.floor(h), r2 = (r * 7.13) % 1;
      if (r > 0.6) continue;
      const lx = (i + r2) * trees.res, ly = TILE - (j + ((r * 3.7) % 1)) * trees.res, k = 3 + r2 * 3;
      mats.push(m4.compose(p.set(lx, groundAt(t.ox + lx, t.oy + ly) + k * 1.1, -ly), q, s.set(k, k * 1.5, k)).clone());
    }
    if (!mats.length) return null;
    const im = new THREE.InstancedMesh(treeGeo, treeMat, mats.length);
    mats.forEach((m, i) => im.setMatrixAt(i, m));
    im.castShadow = true;
    return im;
  }

  // ---- tiles --------------------------------------------------------------
  const matB = buildingMaterial(), matR = roadMaterial(), tiles = [];
  for (const c of M.chunks) for (let t = 0; t < 16; t++) if (c.n[t] || c.r[t]) {
    const tx = c.cx * 4 + t % 4, ty = c.cy * 4 + (t >> 2);
    tiles.push({ tx, ty, ox: X0 + tx * TILE, oy: Y0 + ty * TILE, far: null, det: null, detLod: 2, busy: false, d: 0 });
  }
  function toMesh(part, mat) {
    if (!part.idx.length) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(part.pos, 3)); g.setAttribute("normal", new THREE.BufferAttribute(part.nor, 3, true));
    g.setAttribute("color", new THREE.BufferAttribute(part.col, 3, true)); g.setAttribute("aInfo", new THREE.BufferAttribute(part.inf, 4));
    g.setIndex(new THREE.BufferAttribute(part.idx, 1)); g.computeBoundingSphere();
    const mesh = new THREE.Mesh(g, mat); mesh.receiveShadow = true; mesh.castShadow = mat === matB;
    return mesh;
  }
  function drop(group) { if (!group) return; scene.remove(group); group.traverse(o => { if (o.geometry && o.geometry !== treeGeo) o.geometry.dispose(); }); }
  // Detail meshes that scroll out of range are parked, not freed, so panning back costs no rebuild.
  const parked = new Map(), PARK_MAX = 48;
  function park(t) {
    scene.remove(t.det); parked.delete(t); parked.set(t, { group: t.det, lod: t.detLod });
    if (parked.size > PARK_MAX) { const [k, v] = parked.entries().next().value; parked.delete(k); drop(v.group); }
    t.det = null; t.detLod = 2; if (t.far) t.far.visible = true;
  }
  function unpark(t) {
    const p = parked.get(t); parked.delete(t);
    drop(t.det); t.det = p.group; t.detLod = p.lod; scene.add(p.group); if (t.far) t.far.visible = false;
  }

  const nWorkers = Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) - 2)), workers = [], jobs = new Map();
  let jobId = 0, farLeft = tiles.length;
  const farQueue = tiles.slice();
  function onDone(w, m) {
    w.load--;
    const t = jobs.get(m.id); jobs.delete(m.id); dirty = shDirty = true;
    if (m.error) { console.warn("tile", m.tx, m.ty, m.error); if (m.lod >= 2) farLeft--; else t.busy = false; return pump(); }
    const group = new THREE.Group(); group.position.set(t.ox, 0, -t.oy);
    for (const mesh of [toMesh(m.b, matB), toMesh(m.r, matR)]) if (mesh) group.add(mesh);
    group.matrixAutoUpdate = false; group.updateMatrix();
    if (m.lod >= 2) {
      farLeft--;
      if (group.children.length) { t.far = group; group.visible = !t.det; scene.add(group); }
      boot("Raising the skyline", 0.25 + 0.75 * (1 - farLeft / tiles.length));
      if (farLeft === 0) { $("boot").classList.add("done"); setTimeout(() => $("tip").classList.add("off"), 9000); }
    } else {
      t.busy = false;
      if (t.want !== m.lod) return pump();                  // camera moved on while this was building
      if (m.lod === 0) { const tr = plantTrees(t); if (tr) group.add(tr); }
      drop(t.det); t.det = group; t.detLod = m.lod; scene.add(group);
      if (t.far) t.far.visible = false;
    }
    pump();
  }
  for (let i = 0; i < nWorkers; i++) {
    const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    w.load = 0; w.onmessage = e => onDone(w, e.data); w.onerror = e => fail(new Error("tile worker failed: " + e.message));
    w.postMessage({ type: "init", cfg: { base: new URL(DATA, location.href).href, extent: M.extent, tile: TILE, hg: { w: G.w, h: G.h, res: G.res, x0: G.x0, y1: G.y1 } }, hg: G.a });
    workers.push(w);
  }
  fetch(DATA + M.sky.file + ".bin").then(r => r.arrayBuffer()).then(buf => workers.forEach(w => w.postMessage({ type: "sky", buf })));
  let nearQueue = [];
  function pump() {
    for (const w of workers) while (w.load < 2) {
      let t = nearQueue.shift(), lod;
      if (t) { if (t.busy || t.want === t.detLod || t.want >= 2) continue; lod = t.want; t.busy = true; }
      else if ((t = farQueue.shift())) lod = 2;
      else break;
      w.load++; jobs.set(++jobId, t);
      w.postMessage({ type: "tile", id: jobId, tx: t.tx, ty: t.ty, lod });
    }
  }
  const frustum = new THREE.Frustum(), pm = new THREE.Matrix4(), ball = new THREE.Sphere();
  function schedule() {
    const cx = controls.target.x, cy = -controls.target.z, dcam = camera.position.distanceTo(controls.target);
    // Detail is invisible from far up: windows are sub-pixel. Ask for it only when the camera is close enough to see it.
    const r0 = dcam < quality[0] * 3 ? quality[0] : 0, r1 = dcam < quality[1] * 3 ? quality[1] : 0;
    frustum.setFromProjectionMatrix(pm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
    for (const t of tiles) {
      t.d = Math.hypot(t.ox + TILE / 2 - cx, t.oy + TILE / 2 - cy);
      ball.center.set(t.ox + TILE / 2, controls.target.y, -t.oy - TILE / 2); ball.radius = TILE;
      t.vis = frustum.intersectsSphere(ball);
      t.want = t.d < r0 ? 0 : t.d < r1 ? 1 : 2;
      if (!t.vis && t.want < 2) t.want = t.det ? t.detLod : 1;        // off-screen: keep what is there, never upgrade
      if (t.det && t.d > Math.max(r1, quality[1] * 0.5) * 1.2) park(t);
      const p = parked.get(t);
      if (p && t.want < 2 && t.want !== t.detLod && p.lod <= t.want) { unpark(t); t.want = t.detLod; dirty = shDirty = true; }
    }
    nearQueue = tiles.filter(t => t.want < 2 && t.want !== t.detLod && !t.busy).sort((a, b) => (b.vis - a.vis) || (a.d - b.d));
    nearLabels(cx, cy, dcam);
    pump();
  }

  // ---- time of day --------------------------------------------------------
  const dir = new THREE.Vector3(), cDay = new THREE.Color(0xb8c6d4), cGold = new THREE.Color(0xe2a878), cNight = new THREE.Color(0x05080d), tmp = new THREE.Color();
  function setHour(h) {
    const s = sun(h, M.origin.lat, M.origin.lon), up = Math.sin(s.el), day = smooth(-0.06, 0.14, up), gold = day * (1 - smooth(0.08, 0.42, up));
    dir.set(Math.cos(s.el) * Math.sin(s.az), up, -Math.cos(s.el) * Math.cos(s.az));
    sky.material.uniforms.sunPosition.value.copy(dir);
    // Strong warm sun against weak cool sky light: that contrast, with cast shadows, is what gives the buildings depth.
    sunLight.intensity = 4.6 * day * (1 - 0.2 * gold); sunLight.color.setRGB(1, 0.95 - 0.33 * gold, 0.87 - 0.5 * gold);
    hemi.intensity = 0.5 + 0.8 * day; hemi.color.setRGB(0.6 + 0.2 * gold, 0.72, 1 - 0.15 * gold);
    scene.fog.color.copy(tmp.copy(cNight).lerp(cDay, day).lerp(cGold, gold * 0.7));
    renderer.toneMappingExposure = 0.85 + 0.3 * (1 - day);
    U.uNight.value = 1 - smooth(-0.1, 0.06, up); dirty = shDirty = true;
    const hh = Math.floor(h) % 24, mm = Math.floor((h % 1) * 60);
    $("hourText").textContent = `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")} IST`;
  }
  $("hour").addEventListener("input", e => setHour(+e.target.value));
  document.querySelectorAll("[data-h]").forEach(b => b.addEventListener("click", () => { $("hour").value = b.dataset.h; setHour(+b.dataset.h); }));

  // ---- camera moves -------------------------------------------------------
  let flight = null;
  const pose = () => { const o = camera.position.clone().sub(controls.target), d = o.length(); return { x: controls.target.x, z: controls.target.z, d, tilt: Math.acos(o.y / d), turn: Math.atan2(o.x, o.z) }; };
  function place(p) {
    const y = groundAt(p.x, -p.z); controls.target.set(p.x, y, p.z);
    camera.position.set(p.x + p.d * Math.sin(p.tilt) * Math.sin(p.turn), y + p.d * Math.cos(p.tilt), p.z + p.d * Math.sin(p.tilt) * Math.cos(p.turn));
  }
  function fly(to, ms = 2600) {
    const from = pose(); to = { ...from, ...to };
    to.turn = from.turn + ((to.turn - from.turn + Math.PI * 3) % (Math.PI * 2) - Math.PI);
    // Long hops rise in an arc and come back down, so the ground never streaks past at street level.
    const arc = Math.min(1.2, Math.hypot(to.x - from.x, to.z - from.z) / (6 * Math.max(from.d, to.d)));
    flight = { from, to, t0: performance.now(), ms, arc };
  }
  // Sun shadows cover a square around the view centre that grows with zoom. The map is redrawn only when something moved.
  let shR = 0; const shC = new THREE.Vector3(1e9, 0, 0);
  function shadows(d) {
    const on = quality[5] > 0 && dir.y > 0.03, c = controls.target, cam = sunLight.shadow.camera;
    if (sunLight.castShadow !== on) { sunLight.castShadow = on; shDirty = true; }
    if (!on) { sunLight.target.position.copy(c); sunLight.position.copy(c).addScaledVector(dir.y > 0 ? dir : tmpV.copy(dir).negate(), 50000); return; }
    const r = Math.round(Math.min(3600, Math.max(280, d * 0.85)) / 40) * 40;
    if (r === shR && !shDirty && c.distanceToSquared(shC) < (r * 0.04) ** 2) return;
    // ponytail: shadow square re-centres in steps, so edges can shift a texel on a long pan; snap to the texel grid if that shows
    cam.left = cam.bottom = -r; cam.right = cam.top = r; cam.near = 10; cam.far = r * 8; cam.updateProjectionMatrix();
    sunLight.shadow.normalBias = 0.5 + r / 650;
    sunLight.target.position.copy(c); sunLight.position.copy(c).addScaledVector(dir, r * 4); sunLight.target.updateMatrixWorld();
    shC.copy(c); shR = r; shDirty = false; renderer.shadowMap.needsUpdate = true;
  }
  const places = (M.places || []).concat([{ n: "Whole city", x: 0, y: 0, d: 62000, tilt: 0.75, turn: 0 }]);
  for (const p of places) {
    const b = document.createElement("button"); b.className = "pill"; b.textContent = p.n; b.setAttribute("role", "listitem");
    b.onclick = () => fly({ x: p.x, z: -p.y, d: p.d, tilt: p.tilt ?? 1.1, turn: (p.turn ?? 0) * Math.PI / 180 });
    $("go").append(b);
  }
  // ---- search: sectors from search.json plus every named label ---------------
  const norm = t => t.toLowerCase().replace(/\bsec(tor)?\b\.?/g, "sector").replace(/[^a-z0-9]+/g, " ").trim();
  const index = M.labels.filter(l => l.k !== "region").map(l => ({ n: l.n, x: l.x, y: l.y, d: l.k === "tower" ? 900 : 2200 }));
  fetch(DATA + "search.json").then(r => r.ok ? r.json() : []).catch(() => []).then(found => {
    const sectors = found.filter(s => !s.k);
    index.unshift(...sectors.map(s => ({ ...s, d: 1800 })), ...(M.places || []).filter(p => !sectors.some(s => s.n === p.n)), ...found.filter(s => s.k).map(s => ({ ...s, d: 900 })));
    $("findList").innerHTML = [...new Set(index.map(e => e.n))].map(n => `<option value="${n.replace(/[<&"]/g, "")}">`).join("");
    labels.push(...sectors.map(s => ({ ...s, k: "sector", pos: new THREE.Vector3(s.x, groundAt(s.x, s.y) + 30, -s.y) })));
    labels.sort((a, b) => RANK[a.k] - RANK[b.k]); named = labels.filter(l => l.k === "sector" || l.k === "town" || l.k === "hood"); touch();
  });
  function find(q, pick) {
    q = norm(q); if (!q) return;
    const hit = index.find(e => norm(e.n) === q) || (pick && (index.find(e => norm(e.n).startsWith(q)) || index.find(e => norm(e.n).includes(q))));
    $("find").classList.toggle("miss", pick && !hit);
    if (hit) { fly({ x: hit.x, z: -hit.y, d: hit.d, tilt: 1.05 }); $("find").blur(); }
    return hit;
  }
  $("find").addEventListener("input", e => find(e.target.value, false));          // datalist pick = exact name
  $("find").addEventListener("keydown", e => { if (e.key === "Enter") find(e.target.value, true); });
  $("north").onclick = () => fly({ turn: 0 }, 700);
  const keys = new Set();
  addEventListener("keydown", e => { if (!e.ctrlKey && !e.metaKey && e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") keys.add(e.code); });
  addEventListener("keyup", e => keys.delete(e.code));
  canvas.addEventListener("pointerdown", () => { flight = null; $("tip").classList.add("off"); });
  canvas.addEventListener("wheel", () => { flight = null; }, { passive: true });

  // ---- labels -------------------------------------------------------------
  let labelsOn = true;
  // The closer the camera, the finer the label tier, like a street map: town, sector, society, building, shop.
  const RANGE = { region: 1e9, town: 60000, sector: 14000, hood: 9000, tower: 8000, soc: 5000, bld: 1800, lm: 1800, poi: 650 };
  const RANK = { region: 0, town: 1, tower: 2, sector: 3, hood: 4 }, pool = [], v = new THREE.Vector3();
  const labels = M.labels.map(l => ({ ...l, pos: new THREE.Vector3(l.x, l.z ?? groundAt(l.x, l.y) + 25, -l.y) })).sort((a, b) => RANK[a.k] - RANK[b.k]);
  // Fine tiers live in one file per 4 km chunk and are fetched only when the camera is low over that chunk.
  const labelChunks = new Map(), LIFT = { soc: 35, lm: 18, poi: 9, bld: 20 }, have = new Set(M.chunks.map(c => c.cx + "_" + c.cy));
  let fine = [];
  function nearLabels(cx, cy, dcam) {
    fine = [];
    if (dcam > 9000) return;
    const lists = [];
    for (let j = Math.floor((cy - 2500 - Y0) / M.chunk); j <= Math.floor((cy + 2500 - Y0) / M.chunk); j++)
      for (let i = Math.floor((cx - 2500 - X0) / M.chunk); i <= Math.floor((cx + 2500 - X0) / M.chunk); i++) {
        const key = i + "_" + j;
        if (!have.has(key)) continue;
        if (!labelChunks.has(key)) {
          const e = { list: null }; labelChunks.set(key, e);
          fetch(`${DATA}l_${key}.json`).then(r => r.ok ? r.json() : []).catch(() => []).then(rows => {
            e.list = rows.map(([n, x, y, k, z]) => ({ n, k, pos: new THREE.Vector3(x, z ?? groundAt(x, y) + LIFT[k], -y) })); touch(); schedule();
          });
          if (labelChunks.size > 16) labelChunks.delete(labelChunks.keys().next().value);
        }
        const e = labelChunks.get(key); if (e.list) lists.push(e.list);
      }
    for (const k of ["soc", "bld", "lm", "poi"]) for (const list of lists) for (const l of list) if (l.k === k) fine.push(l);
  }
  function drawLabels() {
    const w = canvas.clientWidth, h = canvas.clientHeight, boxes = []; let n = 0;
    if (labelsOn) for (const l of labels.concat(fine)) {
      if (n >= 90) break;
      const dist = l.pos.distanceTo(camera.position);
      if (dist > RANGE[l.k] || (l.k === "town" && dist < 6000) || (l.k === "sector" && dist < 1200)) continue;
      v.copy(l.pos).project(camera);
      if (v.z > 1 || Math.abs(v.x) > 1 || Math.abs(v.y) > 1) continue;
      const sx = (v.x * 0.5 + 0.5) * w, sy = (0.5 - v.y * 0.5) * h, hw = l.n.length * 3.4 + 8;   // half width guessed from text length
      if (boxes.some(b => Math.abs(b[0] - sx) < b[2] + hw && Math.abs(b[1] - sy) < 17)) continue;
      boxes.push([sx, sy, hw]);
      const el = pool[n] || (pool[n] = $("tags").appendChild(document.createElement("div")));
      if (el._l !== l) { el._l = l; el.className = "tag " + l.k; el.textContent = l.k === "tower" ? `${l.n} ${Math.round(l.h)} m` : l.n; }
      el.style.display = ""; el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) translate(-50%, -100%)`; n++;
    }
    for (let i = n; i < pool.length; i++) pool[i].style.display = "none";
  }
  let named = labels.filter(l => l.k === "town" || l.k === "hood");
  function readout() {
    const x = controls.target.x, y = -controls.target.z; let best = null, bd = 4000;
    for (const l of named) { const d = Math.hypot(l.x - x, l.y - y) * (l.k === "sector" ? 0.5 : 1); if (d < bd) { bd = d; best = l; } }   // sectors win ties
    $("spotName").textContent = best ? best.n : "Noida";
    $("spotLL").textContent = `${(M.origin.lat + y / 110826).toFixed(4)}° N  ${(M.origin.lon + x / 97904).toFixed(4)}° E`;   // metres per degree at 28.5° N
  }

  // ---- panel wiring -------------------------------------------------------
  $("tLabels").onclick = e => { labelsOn = !labelsOn; e.currentTarget.setAttribute("aria-pressed", labelsOn); touch(); };
  function setLevel(l) { level = l; quality = QUALITY[l]; $("qAuto").textContent = `Auto (${l})`; resize(); schedule(); }
  $("quality").onchange = e => { auto = e.target.value === "auto"; if (!auto) setLevel(e.target.value); };
  $("tInfo").onclick = () => { $("info").hidden = !$("info").hidden; }; $("infoX").onclick = () => { $("info").hidden = true; };
  const names = { measured: "Mapped or published height", floors: "Mapped floor count", satellite: "Satellite estimate", raster: "Outlined from satellite height", planned: "Planned, from published project details", default: "Footprint-based guess" }, st = M.stats;
  $("srcBars").innerHTML = Object.keys(names).filter(k => st[k]).map(k => `<span>${names[k]}</span><div><i style="width:${Math.max(1, 100 * st[k] / st.total)}%"></i></div><span>${st[k].toLocaleString("en-IN")}</span>`).join("");
  $("srcNote").textContent = `${(100 * (st.measured + st.floors) / st.total).toFixed(1)}% of buildings carry a mapped height or floor count. The rest are estimated.`;
  $("srcList").innerHTML = M.sources.map(s => `<li>${s.replace(/[<&]/g, "")}</li>`).join("");

  // ---- go -----------------------------------------------------------------
  const first = places[0];
  place({ x: first.x, z: -first.y, d: 2800, tilt: 1.2, turn: -0.5 });
  setHour(+$("hour").value); resize(); schedule();
  let last = performance.now(), frame = 0, lastDraw = 0, wasActive = false, ema = 16, slow = 0, fast = 0, hold = 3;
  renderer.setAnimationLoop(now => {
    const dt = Math.min(0.1, (now - last) / 1000); last = now; U.uTime.value = now / 1000;
    const flying = !!flight || keys.size > 0;
    if (flight) {
      const k = Math.min(1, (now - flight.t0) / flight.ms), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(2 - 2 * k, 3) / 2, a = flight.from, b = flight.to, mix = key => a[key] + (b[key] - a[key]) * e;
      place({ x: mix("x"), z: mix("z"), d: Math.exp(Math.log(a.d) + (Math.log(b.d) - Math.log(a.d)) * e) * (1 + flight.arc * Math.sin(Math.PI * k)), tilt: mix("tilt"), turn: mix("turn") });
      if (k === 1) flight = null;
    } else if (keys.size) {
      const p = pose(), step = p.d * 0.9 * dt, f = (keys.has("KeyW") || keys.has("ArrowUp") ? 1 : 0) - (keys.has("KeyS") || keys.has("ArrowDown") ? 1 : 0), r = (keys.has("KeyD") || keys.has("ArrowRight") ? 1 : 0) - (keys.has("KeyA") || keys.has("ArrowLeft") ? 1 : 0);
      if (f || r) { p.x += (-Math.sin(p.turn) * f + Math.cos(p.turn) * r) * step; p.z += (-Math.cos(p.turn) * f - Math.sin(p.turn) * r) * step; place(p); }
    }
    controls.target.x = Math.min(RE[2], Math.max(RE[0], controls.target.x)); controls.target.z = Math.min(-RE[1], Math.max(-RE[3], controls.target.z));
    const gy = groundAt(controls.target.x, -controls.target.z), ty = controls.target.y;
    controls.target.y = Math.abs(gy - ty) < 0.05 ? gy : ty + (gy - ty) * Math.min(1, dt * 6);   // the pivot rides the terrain, eased
    const active = controls.update() || flying || controls.target.y !== gy;   // update() is true while the camera is still easing
    const d = camera.position.distanceTo(controls.target);
    // Idle: nothing moved. Keep traffic and water moving at full frame rate when close enough to see them; otherwise draw nothing.
    const ticking = level !== "lite" && d < 4500;
    if (wasActive && !active) schedule();
    wasActive = active;
    if (!active && !dirty && !ticking) return;
    if (active && now - lastDraw < 100) {                    // auto quality: judge only frames drawn back to back while moving
      ema += (now - lastDraw - ema) * 0.08;
      slow = ema > 30 ? slow + 1 : 0; fast = ema < 18 ? fast + 1 : 0;     // vsync hides headroom: 60 fps held is the only "fast" signal
      const i = LEVELS.indexOf(level);
      if (auto && slow > 40 && i > 0) { setLevel(LEVELS[i - 1]); slow = 0; ema = 20; hold = i - 1; }   // a level that proved slow is never retried
      else if (auto && fast > 300 && i < hold) { setLevel(LEVELS[i + 1]); fast = 0; ema = 20; }
    }
    dirty = false; lastDraw = now;
    const floor = groundAt(camera.position.x, -camera.position.z) + 6;
    if (camera.position.y < floor) camera.position.y = floor;
    scene.fog.density = Math.min(0.000014, 0.22 / d);       // light haze near the ground, almost none from high up
    const near = Math.min(250, Math.max(1.5, (camera.position.y - floor + 6) * 0.06));       // near plane from height, so walls never clip at street level
    if (Math.abs(camera.near - near) / near > 0.08) camera.near = near;
    camera.far = Math.max(120000, d * 6); camera.updateProjectionMatrix();
    sky.position.copy(camera.position); under.position.set(camera.position.x, 150, camera.position.z);
    shadows(d);
    $("needle").style.transform = `rotate(${Math.atan2(camera.position.x - controls.target.x, camera.position.z - controls.target.z)}rad)`;
    if (frame % 15 === 0) schedule();
    if (frame % 2 === 0) drawLabels();
    if (frame++ % 30 === 0) readout();
    bloom.enabled = !!quality[3]; bloom.strength = 0.2 + 0.5 * U.uNight.value;
    composer.render();
  });
  window.__noida = { find, fly, setHour, pose, tiles, renderer, parked, get level() { return level; }, get fine() { return fine.length; } };    // debugging handle
}
const tmpV = new THREE.Vector3();
start().catch(fail);

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createScene } from './viz/scene.js';
import { TokenRibbon } from './viz/tokens.js';
import { LayerRings, LAYER_COLORS, RING_Y } from './viz/rings.js';
import { Tethers } from './viz/tethers.js';
import { Storm, STORM_Y } from './viz/storm.js';
import { makeTextSprite } from './viz/textsprite.js';
import { PROBE_DIMS } from './engine/model.js';

const $ = (id) => document.getElementById(id);

// ---------- scene ----------
const { renderer, scene, camera, composer, starMat } = createScene($('app'));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 12, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.55;
controls.minDistance = 14;
controls.maxDistance = 90;

const ribbon = new TokenRibbon(scene);
const rings = new LayerRings(scene);
const tethers = new Tethers(scene);
const storm = new Storm(scene);

// the "thought pulse" that rises from the newest word through the layers
const pulse = {
  sprite: null,
  curve: null,
  t: 1,
  dur: 1,
  active: false,
};
{
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(180,240,255,.7)');
  g.addColorStop(1, 'rgba(120,210,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  pulse.sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }),
  );
  pulse.sprite.scale.setScalar(2.4);
  scene.add(pulse.sprite);
}

function firePulse(from, to, dur) {
  pulse.curve = new THREE.CatmullRomCurve3([
    from.clone(),
    new THREE.Vector3(from.x * 0.35, (from.y + to.y) / 2, from.z * 0.35),
    to.clone(),
  ]);
  pulse.t = 0;
  pulse.dur = dur;
  pulse.active = true;
}

// helical streak climbing the column from ring L to ring L+1
function helixCurve(y0, y1, a0, turns = 1.2, r = 4.6) {
  const pts = [];
  for (let k = 0; k <= 10; k++) {
    const u = k / 10;
    const ang = a0 + turns * Math.PI * 2 * u;
    const rr = r + Math.sin(u * Math.PI * 3 + a0) * 0.5;
    pts.push(new THREE.Vector3(Math.cos(ang) * rr, THREE.MathUtils.lerp(y0, y1, u), Math.sin(ang) * rr));
  }
  return new THREE.CatmullRomCurve3(pts);
}

const headColor = (h) => new THREE.Color().setHSL((h / 16 + 0.52) % 1, 0.85, 0.62);

// click a ring -> that layer detonates into its 16 attention heads, each
// striking the exact word it's reading (real per-head argmax attention)
let lastLayers = null;
const emaWork = [null, null, null, null];
// 3D head readout: floating rows of text on the LEFT of the view, one per
// attention head — exact token + attention value, in that head's color.
// Parented to the camera so the stack holds screen position while orbiting.
const readout = new THREE.Group();
readout.position.set(0, 0, -16);
camera.add(readout);
scene.add(camera);
let readoutRows = []; // {sprite, born, life, delay, targetX, y}

function clearReadout() {
  for (const r of readoutRows) {
    readout.remove(r.sprite);
    r.sprite.material.map.dispose();
    r.sprite.material.dispose();
  }
  readoutRows = [];
}

function addReadoutRow(sprite, i, y) {
  sprite.material.opacity = 0;
  const targetX = -9.6 + sprite.scale.x / 2; // left-aligned column
  sprite.position.set(targetX - 1.2, y, 0);
  readout.add(sprite);
  readoutRows.push({ sprite, born: clock.elapsedTime, life: 5, delay: i * 0.05, targetX, y });
}

function showHeadReadout(L, headTop) {
  clearReadout();
  const sorted = [...headTop].sort((a, b) => b.w - a.w).slice(0, 4);
  const title = makeTextSprite(`LAYER ${L + 1} · STRONGEST HEADS`, {
    color: '#a8cce8',
    glow: '#0d2335',
    size: 52,
    scale: 0.011,
    blur: 4,
  });
  addReadoutRow(title, 0, 3.1);
  sorted.forEach(({ h, t, w }, i) => {
    const word = t >= ribbon.tokens.length ? '⟨itself⟩' : ribbon.tokens[t].text.trim() || '⟨space⟩';
    const col = '#' + headColor(h).getHexString();
    const sprite = makeTextSprite(`“${word}”  ${(w * 100).toFixed(0)}%`, {
      color: col,
      glow: col,
      size: 60,
      scale: 0.0125 * (0.8 + 0.45 * w), // stronger heads literally larger
      blur: 5,
    });
    addReadoutRow(sprite, i + 1, 2.2 - i * 0.78);
  });
}

function headBurst(L) {
  rings.pulse(L, 2);
  rings.setActivity(L, 1);
  if (!lastLayers || !lastLayers[L]) return;
  showHeadReadout(L, lastLayers[L].headTop);
  const tmp = new THREE.Vector3();
  lastLayers[L].headTop.forEach(({ h, t, w }) => {
    if (t >= ribbon.tokens.length) return;
    const to = ribbon.worldPos(t);
    const az = Math.atan2(to.x, to.z);
    rings.ringWorldPoint(L, az, tmp);
    tethers.strike(tmp, to, {
      color: headColor(h),
      weight: Math.min(1, 0.45 + w),
      dur: 0.8,
      tail: 0.6,
      delay: h * 0.035,
    });
    ribbon.flash(t, Math.min(1, w * 1.5));
  });
}

window.__headBurst = headBurst; // for tests / directing video shots
window.__readoutCount = () => readoutRows.length;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => (downAt = [e.clientX, e.clientY]));
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6) return;
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(rings.clickTargets, false);
  if (hits.length) headBurst(hits[0].object.userData.layer);
});
renderer.domElement.addEventListener('pointermove', (e) => {
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  renderer.domElement.style.cursor = raycaster.intersectObjects(rings.clickTargets, false).length
    ? 'pointer'
    : 'grab';
});

// ---------- worker ----------
const worker = new Worker(new URL('./engine/worker.js', import.meta.url), { type: 'module' });
let pace = 1.0;
let running = false;
let pendingToken = null;
let animating = false;
let tokensDreamed = 0;
let stopRequested = false;

const wait = (ms) => new Promise((r) => setTimeout(r, ms / pace));

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'progress') {
    $('loader-bar').style.width = `${Math.round(m.frac * 100)}%`;
    $('loader-pct').textContent = `${Math.round(m.frac * 100)}%`;
    $('loader-stage').textContent = m.stage.toUpperCase() + '…';
  } else if (m.type === 'ready') {
    $('loader').classList.add('done');
    setTimeout(() => start(), 900); // auto-dream on load
  } else if (m.type === 'prompt_done') {
    playPrompt(m.tokens);
  } else if (m.type === 'token') {
    pendingToken = m;
    maybeAnimate();
  }
};
worker.postMessage({ type: 'load' });

// ---------- transcript ----------
function addTranscript(text, cls) {
  const span = document.createElement('span');
  span.className = 'tok ' + cls;
  span.textContent = text;
  $('transcript-inner').appendChild(span);
  if (cls === 'fresh') setTimeout(() => span.classList.remove('fresh'), 700);
  const inner = $('transcript-inner');
  while (inner.childNodes.length > 220) inner.removeChild(inner.firstChild);
}

// ---------- generation flow ----------
function start() {
  if (running) {
    stopRequested = true;
    return;
  }
  running = true;
  stopRequested = false;
  tokensDreamed = 0;
  pendingToken = null;
  ribbon.reset();
  $('transcript-inner').innerHTML = '';
  $('tok-count').textContent = '0';
  $('go').textContent = 'WAKE';
  $('go').classList.add('stop');
  worker.postMessage({
    type: 'start',
    prompt: $('prompt').value || 'Once upon a time',
    temperature: parseFloat($('temp').value),
    topk: 40,
  });
}

async function playPrompt(tokens) {
  for (const t of tokens) {
    const i = ribbon.addToken(t.text, { isPrompt: true });
    ribbon.flash(i, 0.7);
    addTranscript(t.text, 'prompt');
    rings.pulse(Math.floor(Math.random() * 4), 0.3);
    await wait(70);
  }
  worker.postMessage({ type: 'next' });
}

function maybeAnimate() {
  if (!animating && pendingToken) {
    const data = pendingToken;
    pendingToken = null;
    animateToken(data);
  }
}

async function animateToken(d) {
  animating = true;
  lastLayers = d.layers;
  // ask for the next token NOW so compute overlaps animation
  if (!d.done && !stopRequested && tokensDreamed < 400) worker.postMessage({ type: 'next' });

  // 1 ── thought pulse rises from the newest word into the stack
  const from = ribbon.tokens.length
    ? ribbon.worldPos(ribbon.tokens.length - 1)
    : new THREE.Vector3(0, 0.5, 12);
  firePulse(from, new THREE.Vector3(0, RING_Y[0], 0), 0.32 / pace);
  await wait(290);

  // 2 ── each layer fires: ring pulses + attention strikes down at the words it reads.
  // Deep layers always have bigger raw norms, so each layer is graded against
  // its OWN running average: light means "unusually active for this token".
  const work = d.layers.map((l) => l.attnNorm + l.mlpNorm);
  rings.setResidual(d.layers.map((l) => l.residSample), PROBE_DIMS);
  const rel = work.map((v, L) => {
    if (emaWork[L] == null) emaWork[L] = v;
    const dev = v / emaWork[L] - 1;
    emaWork[L] += (v - emaWork[L]) * 0.12;
    return THREE.MathUtils.clamp(0.35 + dev * 3.2, 0.02, 1);
  });
  for (let L = 0; L < d.layers.length; L++) {
    const lay = d.layers[L];
    rings.pulse(L, 0.35 + 0.85 * rel[L]);
    rings.setActivity(L, 0.08 + 0.92 * rel[L] * rel[L]);
    const bar = $(`l${L}`);
    if (bar) {
      bar.style.width = `${Math.round(12 + 88 * rel[L])}%`;
      $(`l${L}-val`).textContent = `focus ${(3.2 - Math.min(3.2, lay.meanEntropy)).toFixed(1)}`;
    }
    // only attention that means something gets light: top 3, brightness ∝ weight²
    const tmp = new THREE.Vector3();
    const strikes = lay.top.filter(({ t, w }) => t < ribbon.tokens.length && w >= 0.07).slice(0, 3);
    strikes.forEach(({ t, w }, rank) => {
      const to = ribbon.worldPos(t);
      const az = Math.atan2(to.x, to.z);
      rings.ringWorldPoint(L, az, tmp);
      const wgt = Math.min(1, Math.pow(w * 2.4, 1.6));
      tethers.strike(tmp, to, {
        color: LAYER_COLORS[L],
        weight: rank === 0 ? Math.min(1, wgt * 1.3) : wgt * 0.7,
        dur: 0.7 / pace,
        tail: rank === 0 ? 0.55 : 0.35,
      });
      ribbon.flash(t, rank === 0 ? Math.min(1.4, w * 2.4) : Math.min(0.7, w * 1.2));
    });
    // the single strongest head gathers its word upward into the ring
    const best = [...lay.headTop].sort((a, b) => b.w - a.w)[0];
    if (best && best.t < ribbon.tokens.length && best.w > 0.12) {
      const from = ribbon.worldPos(best.t);
      const az = Math.atan2(from.x, from.z);
      tethers.strike(from, rings.ringWorldPoint(L, az, new THREE.Vector3()), {
        color: headColor(best.h),
        weight: Math.min(0.9, best.w),
        dur: 0.55 / pace,
        tail: 0.4,
        delay: 0.12 / pace,
      });
    }
    if (L < d.layers.length - 1) {
      firePulse(new THREE.Vector3(0, RING_Y[L], 0), new THREE.Vector3(0, RING_Y[L + 1], 0), 0.18 / pace);
      // residual surge: brightness carries the layer's actual output magnitude
      const baseAz = Math.random() * Math.PI * 2;
      const col = new THREE.Color().lerpColors(
        new THREE.Color(LAYER_COLORS[L]),
        new THREE.Color(LAYER_COLORS[L + 1]),
        0.5,
      );
      const n = 2 + Math.round(rel[L] * 2);
      for (let s = 0; s < n; s++) {
        tethers.streak(helixCurve(RING_Y[L], RING_Y[L + 1], baseAz + (s * Math.PI * 2) / n), {
          color: col,
          weight: 0.12 + 0.45 * rel[L],
          dur: 0.5 / pace,
          tail: 0.5,
          delay: s * 0.03,
        });
      }
    }
    await wait(195);
  }

  // 3 ── the probability storm condenses the next word
  firePulse(new THREE.Vector3(0, RING_Y[3], 0), new THREE.Vector3(0, STORM_Y, 0), 0.22 / pace);
  for (let s = 0; s < 6; s++) {
    tethers.streak(helixCurve(RING_Y[3], STORM_Y, (s * Math.PI * 2) / 6, 1.6, 3.2), {
      color: 0xffc46b,
      weight: 0.5,
      dur: 0.55 / pace,
      tail: 0.5,
      delay: s * 0.025,
    });
  }
  storm.setDistribution(d.stormPs, d.labels, d.winnerLabelIndex, d.winnerStormIndex);
  storm.mat.uniforms.uWinner.value = d.winnerStormIndex;
  storm.setEnergy(0.8);
  $('entropy-val').textContent = `${d.entropy.toFixed(2)} bits`;
  $('entropy-val').classList.toggle('hot', d.entropy > 4);
  $('conf-val').textContent = `${(d.p * 100).toFixed(1)}%`;
  await wait(520);
  storm.setCollapse(1);
  await wait(330);

  // 4 ── the chosen word falls from the storm into the sentence
  const drop = makeTextSprite((d.text.trim() || '␣'), { color: '#ffffff', glow: '#ffd84b', size: 54 });
  drop.position.set(0, STORM_Y, 0);
  scene.add(drop);
  const target = ribbon.nextSlotPos();
  const t0 = performance.now();
  const dur = 620 / pace;
  await new Promise((resolve) => {
    (function fall() {
      const t = Math.min(1, (performance.now() - t0) / dur);
      const e = 1 - Math.pow(1 - t, 3);
      drop.position.set(
        Math.sin(e * Math.PI * 1.5) * 2 * (1 - e) + target.x * e,
        STORM_Y + (target.y - STORM_Y) * e,
        target.z * e,
      );
      drop.material.opacity = 1;
      if (t < 1) requestAnimationFrame(fall);
      else resolve();
    })();
  });
  scene.remove(drop);
  drop.material.map.dispose();
  drop.material.dispose();

  const idx = ribbon.addToken(d.text);
  ribbon.flash(idx, 1.4);
  addTranscript(d.text, 'fresh');
  tokensDreamed++;
  $('tok-count').textContent = tokensDreamed;
  storm.setCollapse(0);
  storm.setEnergy(0.22);

  animating = false;
  if (d.done || stopRequested || tokensDreamed >= 400) {
    running = false;
    stopRequested = false;
    $('go').textContent = 'DREAM';
    $('go').classList.remove('stop');
    return;
  }
  maybeAnimate();
}

// ---------- UI ----------
$('go').addEventListener('click', start);
$('prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !running) start();
});
$('temp').addEventListener('input', () => {
  $('temp-lbl').textContent = parseFloat($('temp').value).toFixed(2);
  worker.postMessage({ type: 'set', temperature: parseFloat($('temp').value) });
});
$('speed').addEventListener('input', () => {
  pace = parseFloat($('speed').value);
});

// H = hide controls (clean video capture), shows everything else
addEventListener('keydown', (e) => {
  if (e.key === 'h' && document.activeElement !== $('prompt')) {
    const c = document.querySelector('.controls');
    c.style.display = c.style.display === 'none' ? 'flex' : 'none';
  }
});

// ---------- render loop ----------
const clock = new THREE.Clock();
const stormCenter = new THREE.Vector3(0, STORM_Y, 0);
function tick() {
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, clock.getDelta());
  const time = clock.elapsedTime;

  ribbon.update(dt);
  rings.update(dt, time, camera.position);
  tethers.update(dt);
  storm.update(dt, time, camera.position.distanceTo(stormCenter));
  starMat.uniforms.uTime.value = time;

  // head readout rows: staggered fly-in, hold, fade out
  for (let i = readoutRows.length - 1; i >= 0; i--) {
    const r = readoutRows[i];
    const age = time - r.born - r.delay;
    if (age < 0) continue;
    if (age > r.life) {
      readout.remove(r.sprite);
      r.sprite.material.map.dispose();
      r.sprite.material.dispose();
      readoutRows.splice(i, 1);
      continue;
    }
    const fadeIn = Math.min(1, age * 3.5);
    const fadeOut = Math.min(1, (r.life - age) / 1.4); // smooth ~1.4s dissolve
    r.sprite.material.opacity = Math.min(fadeIn, fadeOut) * 0.95;
    // drift gently upward as it fades, so it reads as "dissolving away"
    r.sprite.position.x = THREE.MathUtils.lerp(r.sprite.position.x, r.targetX, Math.min(1, age * 4));
    r.sprite.position.y = r.y + (1 - fadeOut) * 0.6;
  }

  if (pulse.active) {
    pulse.t += dt / pulse.dur;
    if (pulse.t >= 1) {
      pulse.active = false;
      pulse.sprite.material.opacity = 0;
    } else {
      pulse.sprite.position.copy(pulse.curve.getPoint(pulse.t));
      pulse.sprite.material.opacity = Math.sin(pulse.t * Math.PI);
    }
  }

  controls.update();
  composer.render();
}
tick();

import * as THREE from 'three';

const POOL = 160;
const SEGS = 32;

// Light-streak system: a bright head races along a bezier/custom curve leaving
// a fading tail. Per-vertex "alpha" is faked with vertex colors + additive
// blending (black draws as invisible). Used for attention strikes, per-head
// bursts, and the helical inter-layer surges.
export class Tethers {
  constructor(scene) {
    this.pool = [];
    for (let i = 0; i < POOL; i++) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((SEGS + 1) * 3), 3));
      geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array((SEGS + 1) * 3), 3));
      const mat = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 1,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const line = new THREE.Line(geo, mat);
      line.visible = false;
      line.frustumCulled = false;
      scene.add(line);

      const headMat = new THREE.SpriteMaterial({
        map: headTexture(),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const head = new THREE.Sprite(headMat);
      head.scale.setScalar(1.4);
      head.visible = false;
      scene.add(head);

      this.pool.push({
        line,
        head,
        t: 1,
        dur: 1,
        weight: 0,
        tail: 0.45,
        color: new THREE.Color(),
        curve: null,
        active: false,
      });
    }
  }

  // Fire a streak between two points along an outward-bowed bezier.
  strike(from, to, { color = 0x4be8ff, weight = 1, dur = 0.55, tail = 0.45, delay = 0 } = {}) {
    const mid = from.clone().lerp(to, 0.45);
    const out = mid.clone().setY(0).normalize().multiplyScalar(3.5 + weight * 3);
    mid.add(out).y += 1.2;
    this.streak(new THREE.QuadraticBezierCurve3(from.clone(), mid, to.clone()), { color, weight, dur, tail, delay });
  }

  // Fire a streak along any THREE curve.
  streak(curve, { color = 0x4be8ff, weight = 1, dur = 0.55, tail = 0.45, delay = 0 } = {}) {
    const slot = this.pool.find((s) => !s.active);
    if (!slot) return;
    slot.curve = curve;
    const pos = slot.line.geometry.attributes.position;
    for (let i = 0; i <= SEGS; i++) {
      const p = curve.getPoint(i / SEGS);
      pos.setXYZ(i, p.x, p.y, p.z);
    }
    pos.needsUpdate = true;
    slot.color.set(color);
    slot.head.material.color.set(color);
    slot.t = -delay / dur;
    slot.dur = dur;
    slot.tail = tail;
    slot.weight = weight;
    slot.active = true;
    slot.line.visible = false;
    slot.head.visible = false;
  }

  update(dt) {
    for (const s of this.pool) {
      if (!s.active) continue;
      s.t += dt / s.dur;
      if (s.t < 0) continue; // still delayed
      s.line.visible = true;
      s.head.visible = true;
      if (s.t >= 1.35) {
        s.active = false;
        s.line.visible = false;
        s.head.visible = false;
        continue;
      }
      const headT = Math.min(1, s.t);
      const fadeOut = 1 - THREE.MathUtils.smoothstep(s.t, 1, 1.35);
      const colors = s.line.geometry.attributes.color;
      for (let i = 0; i <= SEGS; i++) {
        const u = i / SEGS;
        let b = 0;
        if (u <= headT) {
          const behind = headT - u;
          b = Math.exp(-behind / (s.tail * 0.55)) * (0.25 + 0.85 * s.weight);
        }
        b *= fadeOut;
        colors.setXYZ(i, s.color.r * b, s.color.g * b, s.color.b * b);
      }
      colors.needsUpdate = true;
      const p = s.curve.getPoint(headT);
      s.head.position.copy(p);
      s.head.material.opacity = (0.35 + 0.65 * s.weight) * (s.t >= 1 ? fadeOut : 1);
      s.head.scale.setScalar(0.7 + s.weight * 1.5);
    }
  }
}

let _headTex = null;
function headTexture() {
  if (_headTex) return _headTex;
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.3, 'rgba(255,255,255,.55)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  _headTex = new THREE.CanvasTexture(c);
  return _headTex;
}

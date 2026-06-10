import * as THREE from 'three';
import { makeTextSprite } from './textsprite.js';

export const RING_Y = [6.5, 11.5, 16.5, 21.5];
export const RING_R = 6.2;
export const LAYER_COLORS = [0x35d2ff, 0x6e8bff, 0xb46bff, 0xff5fd0];

// Four transformer-layer rings + the residual-stream "river" of particles
// spiraling up through them. Layer activity modulates pulse and river brightness.
export class LayerRings {
  constructor(scene) {
    this.scene = scene;
    this.rings = [];
    this.activity = [0, 0, 0, 0];
    for (let L = 0; L < 4; L++) {
      const geo = new THREE.TorusGeometry(RING_R, 0.09, 12, 140);
      const mat = new THREE.MeshBasicMaterial({
        color: LAYER_COLORS[L],
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending,
      });
      const ring = new THREE.Mesh(geo, mat);
      ring.rotation.x = Math.PI / 2;
      ring.position.y = RING_Y[L];
      scene.add(ring);

      const glowGeo = new THREE.TorusGeometry(RING_R, 0.28, 12, 140);
      const glowMat = new THREE.MeshBasicMaterial({
        color: LAYER_COLORS[L],
        transparent: true,
        opacity: 0.1,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const glow = new THREE.Mesh(glowGeo, glowMat);
      glow.rotation.x = Math.PI / 2;
      glow.position.y = RING_Y[L];
      scene.add(glow);

      // fat invisible torus = generous click target
      const hit = new THREE.Mesh(
        new THREE.TorusGeometry(RING_R, 1.1, 8, 48),
        new THREE.MeshBasicMaterial({ visible: false }),
      );
      hit.rotation.x = Math.PI / 2;
      hit.position.y = RING_Y[L];
      hit.userData.layer = L;
      scene.add(hit);

      const label = makeTextSprite(`LAYER ${L + 1}`, {
        color: '#6f93b4',
        glow: '#16334a',
        size: 30,
        scale: 0.012,
      });
      label.position.set(Math.sin(0.65) * (RING_R + 2.6), RING_Y[L], Math.cos(0.65) * (RING_R + 2.6));
      label.material.opacity = 0.4;
      scene.add(label);
      this.rings.push({ ring, glow, mat, glowMat, hit, label, stats: null, statsFade: 0, pulse: 0 });
    }

    // residual river
    const N = 3200;
    const seed = new Float32Array(N * 4); // angle0, radius jitter, speed, phase
    for (let i = 0; i < N; i++) {
      seed[i * 4] = Math.random() * Math.PI * 2;
      seed[i * 4 + 1] = Math.random();
      seed[i * 4 + 2] = 0.5 + Math.random();
      seed[i * 4 + 3] = Math.random();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3)); // unused, computed in shader
    geo.setAttribute('seed', new THREE.BufferAttribute(seed, 4));
    this.riverMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uAct: { value: new Float32Array([0, 0, 0, 0]) },
        uColors: {
          value: LAYER_COLORS.map((c) => new THREE.Color(c)),
        },
      },
      vertexShader: `
        attribute vec4 seed;
        uniform float uTime;
        uniform float uAct[4];
        uniform vec3 uColors[4];
        varying vec3 vColor;
        varying float vA;
        void main() {
          float t = fract(seed.w + uTime * 0.045 * seed.z); // 0..1 along the climb
          float y = mix(1.5, 24.5, t);
          float band = clamp(floor((y - 4.0) / 5.0), 0.0, 3.0);
          float act = uAct[0];
          act = mix(act, uAct[1], step(0.5, band));
          act = mix(act, uAct[2], step(1.5, band));
          act = mix(act, uAct[3], step(2.5, band));
          float swirl = seed.x + uTime * (0.55 + 0.9 * act) * seed.z + t * 9.0;
          float r = mix(2.2, 5.6, seed.y) + sin(t * 31.4 + seed.x * 7.0) * 0.35;
          vec3 p = vec3(cos(swirl) * r, y, sin(swirl) * r);
          vec3 col = uColors[0];
          col = mix(col, uColors[1], step(0.5, band));
          col = mix(col, uColors[2], step(1.5, band));
          col = mix(col, uColors[3], step(2.5, band));
          vColor = col * (0.55 + 1.6 * act);
          float edge = smoothstep(0.0, 0.07, t) * smoothstep(1.0, 0.93, t);
          vA = edge * (0.17 + 0.55 * act * act); // visible matter at rest; activation buys the glow
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_PointSize = (2.0 + 2.6 * act + seed.y * 1.3) * (120.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        varying vec3 vColor;
        varying float vA;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float a = smoothstep(0.5, 0.05, d) * vA;
          gl_FragColor = vec4(vColor, a);
        }`,
    });
    this.river = new THREE.Points(geo, this.riverMat);
    scene.add(this.river);

    // faint vertical core beam
    const beamGeo = new THREE.CylinderGeometry(0.35, 0.55, 24, 24, 1, true);
    const beamMat = new THREE.MeshBasicMaterial({
      color: 0x2a7fb0,
      transparent: true,
      opacity: 0.12,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.beam = new THREE.Mesh(beamGeo, beamMat);
    this.beam.position.y = 13;
    scene.add(this.beam);

    // probe dots: 16 river particles we track on the CPU (same math as the
    // shader) so a value label can ride along with each one when zoomed in
    this.probes = [];
    this.time = 0;
    this.layerSamples = null;
    for (let k = 0; k < 16; k++) {
      // probe k IS river particle k — same seeds, same shader math
      this.probes.push({
        seed: [seed[k * 4], seed[k * 4 + 1], seed[k * 4 + 2], seed[k * 4 + 3]],
        dimIdx: k,
        label: null,
        fade: 0,
      });
    }
  }

  // Each token, store the real sampled residual-stream values per layer and
  // refresh the probe-dot labels (value of one actual dimension of the
  // 768-dim thought vector, at the probe's current layer band).
  setResidual(layerSamples, dims) {
    this.layerSamples = layerSamples;
    for (const pr of this.probes) {
      const t = (pr.seed[3] + this.time * 0.045 * pr.seed[2]) % 1;
      const y = THREE.MathUtils.lerp(1.5, 24.5, t);
      const band = THREE.MathUtils.clamp(Math.floor((y - 4) / 5), 0, 3);
      const v = layerSamples[band] ? layerSamples[band][pr.dimIdx] : 0;
      const dim = dims ? dims[pr.dimIdx] : pr.dimIdx;
      if (pr.label) {
        pr.label.material.map.dispose();
        pr.label.material.dispose();
        this.scene.remove(pr.label);
      }
      pr.label = makeTextSprite(`x${dim}  ${v >= 0 ? '+' : ''}${v.toFixed(2)}`, {
        color: v >= 0 ? '#ffc98a' : '#8ad2ff',
        glow: '#0d2335',
        size: 42,
        scale: 0.0075,
        blur: 3,
      });
      pr.label.material.opacity = 0;
      this.scene.add(pr.label);
    }
  }

  get clickTargets() {
    return this.rings.map((r) => r.hit);
  }

  pulse(L, strength = 1) {
    this.rings[L].pulse = Math.min(2, this.rings[L].pulse + strength);
  }

  setActivity(L, v) {
    this.activity[L] = v;
  }

  ringWorldPoint(L, azimuth, out = new THREE.Vector3()) {
    out.set(Math.sin(azimuth) * RING_R, RING_Y[L], Math.cos(azimuth) * RING_R);
    return out;
  }

  update(dt, time, camPos = null) {
    this.riverMat.uniforms.uTime.value = time;
    for (let L = 0; L < 4; L++) {
      const r = this.rings[L];
      r.pulse = Math.max(0, r.pulse - dt * 2.4);
      this.activity[L] *= Math.exp(-dt * 1.1); // activation is an event, not a state
      const act = THREE.MathUtils.lerp(this.riverMat.uniforms.uAct.value[L], this.activity[L], dt * 6);
      this.riverMat.uniforms.uAct.value[L] = act;
      const p = r.pulse;
      // quiet at rest — ring brightness IS the activation readout
      r.mat.opacity = 0.3 + 0.7 * Math.min(1, p + act);
      r.glowMat.opacity = 0.05 + 0.32 * Math.min(1, p + act);
      const s = 1 + 0.11 * p + 0.05 * act + Math.sin(time * 1.3 + L) * 0.003;
      r.ring.scale.setScalar(s);
      r.glow.scale.setScalar(s * (1 + 0.05 * p));
      r.ring.rotation.z += dt * (0.05 + p * 0.5) * (L % 2 ? -1 : 1);
      r.label.material.opacity = 0.22 + 0.55 * Math.min(1, p + act);
    }
    this.beam.material.opacity = 0.07 + 0.03 * Math.sin(time * 0.8);

    // probe dots: labels ride their river particle; visible only when zoomed close
    this.time = time;
    if (camPos) {
      const acts = this.riverMat.uniforms.uAct.value;
      for (const pr of this.probes) {
        if (!pr.label) continue;
        const [sx, sy, sz, sw] = pr.seed;
        const t = (sw + time * 0.045 * sz) % 1;
        const y = THREE.MathUtils.lerp(1.5, 24.5, t);
        const band = THREE.MathUtils.clamp(Math.floor((y - 4) / 5), 0, 3);
        const act = acts[band];
        const swirl = sx + time * (0.55 + 0.9 * act) * sz + t * 9.0;
        const rr = THREE.MathUtils.lerp(2.2, 5.6, sy) + Math.sin(t * 31.4 + sx * 7.0) * 0.35;
        pr.label.position.set(Math.cos(swirl) * rr + 0.55, y + 0.35, Math.sin(swirl) * rr);
        const d = camPos.distanceTo(pr.label.position);
        const edge = THREE.MathUtils.smoothstep(t, 0, 0.07) * (1 - THREE.MathUtils.smoothstep(t, 0.93, 1));
        const target = d < 17 ? Math.min(1, (17 - d) / 6) * edge : 0;
        pr.fade = THREE.MathUtils.lerp(pr.fade, target, dt * 6);
        pr.label.material.opacity = pr.fade * 0.95;
      }
    }
  }
}

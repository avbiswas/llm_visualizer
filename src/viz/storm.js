import * as THREE from 'three';
import { makeTextSprite } from './textsprite.js';

export const STORM_Y = 27.5;
const N_PARTICLES = 5200;
const K = 384; // candidate buckets (matches worker STORM_K)

// The probability storm: every candidate next-word is a swarm of particles.
// High-probability words swirl tight, bright and gold near the eye; the long
// tail is a cold blue haze. On collapse the winner spirals into the eye and
// everything else is blown outward.
export class Storm {
  constructor(scene) {
    this.group = new THREE.Group();
    this.group.position.y = STORM_Y;
    scene.add(this.group);

    const bucket = new Float32Array(N_PARTICLES);
    const seed = new Float32Array(N_PARTICLES * 4);
    for (let i = 0; i < N_PARTICLES; i++) {
      // more particles for likelier buckets: bias bucket assignment toward low indices
      bucket[i] = Math.floor(Math.pow(Math.random(), 2.6) * K);
      seed[i * 4] = Math.random() * Math.PI * 2;
      seed[i * 4 + 1] = Math.random();
      seed[i * 4 + 2] = 0.6 + Math.random() * 0.9;
      seed[i * 4 + 3] = Math.random();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N_PARTICLES * 3), 3));
    geo.setAttribute('bucket', new THREE.BufferAttribute(bucket, 1));
    geo.setAttribute('seed', new THREE.BufferAttribute(seed, 4));

    this.probTex = new THREE.DataTexture(new Float32Array(K * 4), K, 1, THREE.RGBAFormat, THREE.FloatType);
    this.probTex.minFilter = THREE.NearestFilter;
    this.probTex.magFilter = THREE.NearestFilter;
    this.probTex.needsUpdate = true;

    this.mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uProbs: { value: this.probTex },
        uCollapse: { value: 0 },
        uWinner: { value: -1 },
        uEnergy: { value: 0.4 },
      },
      vertexShader: `
        attribute float bucket;
        attribute vec4 seed;
        uniform sampler2D uProbs;
        uniform float uTime, uCollapse, uWinner, uEnergy;
        varying vec3 vColor;
        varying float vA;
        void main() {
          float p = texture2D(uProbs, vec2((bucket + 0.5) / ${K}.0, 0.5)).r;
          float rank = bucket / ${K}.0;
          // radius: likely words orbit the eye, the tail forms the outer disc
          float baseR = mix(2.6, 10.5, pow(1.0 - p, 3.0) * (0.35 + 0.65 * rank));
          float speed = (1.8 - rank) * seed.z * (0.5 + uEnergy);
          float ang = seed.x + uTime * speed;
          float wob = sin(uTime * 2.0 * seed.z + seed.x * 9.0);
          float y = (seed.w - 0.5) * (1.4 + 3.0 * rank) + wob * 0.3;
          float isWinner = step(abs(bucket - uWinner), 0.4);
          // collapse: winner spirals in, losers blasted outward and dimmed
          float r = baseR;
          r = mix(r, 0.25, uCollapse * isWinner);
          r = mix(r, r * (1.0 + 1.9 * uCollapse), (1.0 - isWinner));
          ang += uCollapse * isWinner * 7.0;
          y = mix(y, 0.0, uCollapse * isWinner);
          vec3 pos = vec3(cos(ang) * r, y, sin(ang) * r);
          float heat = pow(p, 0.32);
          vColor = mix(vec3(0.18, 0.3, 0.9), vec3(1.0, 0.85, 0.35), heat);
          vColor = mix(vColor, vec3(1.0), isWinner * uCollapse);
          vColor *= 0.55 + 0.6 * uEnergy;
          float lose = (1.0 - isWinner) * uCollapse;
          vA = (0.045 + 0.4 * heat * heat) * (1.0 - lose * 0.85); // tail = cold haze, mass = fire
          vec4 mv = modelViewMatrix * vec4(pos, 1.0);
          gl_PointSize = (1.2 + 3.4 * heat + isWinner * uCollapse * 3.0) * (110.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        varying vec3 vColor;
        varying float vA;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float a = smoothstep(0.5, 0.04, d) * vA;
          gl_FragColor = vec4(vColor, a);
        }`,
    });
    this.points = new THREE.Points(geo, this.mat);
    this.group.add(this.points);

    // the eye
    const eyeMat = new THREE.SpriteMaterial({
      map: glowTexture(),
      color: 0xbfe9ff,
      transparent: true,
      opacity: 0.5,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.eye = new THREE.Sprite(eyeMat);
    this.eye.scale.setScalar(3);
    this.group.add(this.eye);

    this.labels = []; // {sprite, p}
    this.collapse = 0;
    this.collapseTarget = 0;
    this.detailFade = 0; // 1 when the camera is zoomed into the storm
  }

  // probs: Float32Array(K) sorted desc. labels: [{text,p}] for top words.
  setDistribution(probs, labels, winnerIndex) {
    const data = this.probTex.image.data;
    for (let i = 0; i < K; i++) data[i * 4] = probs[i];
    this.probTex.needsUpdate = true;
    this.mat.uniforms.uWinner.value = winnerIndex;

    for (const l of this.labels) {
      l.sprite.material.map.dispose();
      l.sprite.material.dispose();
      this.group.remove(l.sprite);
    }
    this.labels = [];
    labels.forEach((lab, i) => {
      const txt = lab.text.trim() || '⟨space⟩';
      const isWin = i === winnerIndex;
      const big = i < 14 || isWin;
      // big: the head candidates, always visible. detail: the long tail with
      // its probabilities — only revealed when the camera flies in close.
      const sprite = big
        ? makeTextSprite(txt, {
            color: isWin ? '#ffffff' : '#e4eeff',
            glow: isWin ? '#ffd84b' : '#5b8fff',
            size: 52,
            scale: 0.017 * (0.55 + Math.sqrt(lab.p) * 1.3),
          })
        : makeTextSprite(`${txt} ${(lab.p * 100).toFixed(1)}%`, {
            color: '#9db8e8',
            glow: '#16294a',
            size: 44,
            scale: 0.0085,
            blur: 3,
          });
      const ang = (i / labels.length) * Math.PI * 2 + Math.random() * 0.3;
      const r = 2.2 + (1 - Math.sqrt(lab.p)) * 7.5;
      sprite.position.set(Math.cos(ang) * r, (Math.random() - 0.5) * 1.6, Math.sin(ang) * r);
      sprite.userData = { ang, r, speed: 0.25 + lab.p * 1.4, isWin, p: lab.p, baseScale: sprite.scale.clone(), big };
      sprite.material.opacity = 0;
      this.group.add(sprite);
      this.labels.push({ sprite, p: lab.p, isWin, big });
    });
  }

  // 0 = swirling, 1 = collapsed onto winner
  setCollapse(v) {
    this.collapseTarget = v;
  }

  setEnergy(v) {
    this.mat.uniforms.uEnergy.value = v;
  }

  winnerWorldPos(out = new THREE.Vector3()) {
    return out.set(0, STORM_Y, 0);
  }

  update(dt, time, camDist = 100) {
    this.mat.uniforms.uTime.value = time;
    this.collapse = THREE.MathUtils.lerp(this.collapse, this.collapseTarget, dt * 7);
    this.mat.uniforms.uCollapse.value = this.collapse;
    this.detailFade = THREE.MathUtils.lerp(this.detailFade, camDist < 26 ? Math.min(1, (26 - camDist) / 8) : 0, dt * 5);
    this.eye.material.opacity = 0.05 + 0.03 * Math.sin(time * 2.2) + this.collapse * 0.13;
    this.eye.scale.setScalar(1.2 + this.collapse * 1.2);
    for (const l of this.labels) {
      const u = l.sprite.userData;
      u.ang += dt * u.speed * (1 + this.collapse * (u.isWin ? 5 : 0.4));
      const r = u.isWin ? THREE.MathUtils.lerp(u.r, 0.1, this.collapse) : u.r * (1 + this.collapse * 1.6);
      l.sprite.position.x = Math.cos(u.ang) * r;
      l.sprite.position.z = Math.sin(u.ang) * r;
      let target;
      if (u.isWin) target = 1;
      else if (u.big) target = (1 - this.collapse) * (0.35 + u.p * 1.6);
      else target = (1 - this.collapse) * this.detailFade * 0.85; // zoom-only detail
      l.sprite.material.opacity = THREE.MathUtils.lerp(l.sprite.material.opacity, Math.min(1, target), dt * 5);
      const s = u.isWin ? 1 + this.collapse * 0.8 : 1;
      l.sprite.scale.copy(u.baseScale).multiplyScalar(s);
    }
  }
}

let _glowTex = null;
function glowTexture() {
  if (_glowTex) return _glowTex;
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.25, 'rgba(200,230,255,.5)');
  g.addColorStop(1, 'rgba(160,200,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  _glowTex = new THREE.CanvasTexture(c);
  return _glowTex;
}

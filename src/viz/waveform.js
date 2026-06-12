import * as THREE from 'three';

// The music, made visible: the live mix drawn as aurora bands on the inside
// of a giant cylinder wrapping the whole scene — part of the sky, not a HUD.
// Orbiting gives true parallax (the bands slide past the stars), and the
// transformer always sits in front. Silence fades it out; the piano carves it.
const SAMPLES = 512;
const RADIUS = 260;
const HEIGHT = 300;

export class WaveformBackdrop {
  constructor(scene) {
    this.data = new Float32Array(SAMPLES);
    this.smoothed = new Float32Array(SAMPLES);
    this.tex = new THREE.DataTexture(this.data, SAMPLES, 1, THREE.RedFormat, THREE.FloatType);
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.magFilter = THREE.LinearFilter;
    this.tex.wrapS = THREE.RepeatWrapping;
    this.level = 0;

    this.mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uWave: { value: this.tex },
        uTime: { value: 0 },
        uLevel: { value: 0 },
        uColorA: { value: new THREE.Color(0x35c9e8) }, // layer-1 cyan
        uColorB: { value: new THREE.Color(0xe85fb0) }, // layer-4 pink
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: `
        uniform sampler2D uWave;
        uniform float uTime;
        uniform float uLevel;
        uniform vec3 uColorA;
        uniform vec3 uColorB;
        varying vec2 vUv;

        // triangle-fold the azimuth so sampling is seamless where the
        // cylinder wraps (x=0 meets x=1 with matching values and slope)
        float fold(float x) { return 1.0 - abs(1.0 - 2.0 * fract(x)); }
        float wave(float x) { return texture2D(uWave, vec2(fold(x), 0.5)).r; }

        // neon trace: crisp core plus a wide soft halo
        float trace(float d, float core, float halo) {
          return core / (abs(d) + core) * 0.5 + halo / (abs(d) + halo) * 0.14;
        }

        void main() {
          vec3 col = vec3(0.0);
          float drift = uTime * 0.01;
          // slow aurora undulation so the bands breathe even on held chords
          float sway = sin(vUv.x * 12.5664 + uTime * 0.22) * 0.02;

          // main band: the mix, wrapped right around the sky
          float w0 = wave(vUv.x + drift) * 0.16;
          col += mix(uColorA, uColorB, fold(vUv.x + 0.15)) *
                 trace(vUv.y - 0.5 - w0 - sway, 0.0022, 0.04);

          // echo bands: same signal, time-offset, higher and lower — a braid
          float w1 = wave(vUv.x * 0.5 + drift + 0.25) * 0.12;
          col += mix(uColorB, uColorA, fold(vUv.x)) * 0.5 *
                 trace(vUv.y - 0.66 - w1 + sway, 0.0018, 0.025);
          float w2 = wave(vUv.x * 0.25 + drift + 0.6) * 0.09;
          col += mix(uColorA, uColorB, vUv.y) * 0.35 *
                 trace(vUv.y - 0.34 - w2 - sway * 0.6, 0.0018, 0.02);

          // fade toward the cylinder rims so the bands live mid-sky
          float rim = smoothstep(0.0, 0.18, vUv.y) * smoothstep(1.0, 0.82, vUv.y);
          // deliberately dim: distant space weather, never competing with the mind
          float a = clamp(uLevel * 0.55, 0.0, 0.4) * rim;
          gl_FragColor = vec4(col * a, a);
        }`,
    });

    this.mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(RADIUS, RADIUS, HEIGHT, 96, 1, true),
      this.mat,
    );
    this.mesh.position.y = 10;
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = -10; // first to draw: everything else paints over it
    scene.add(this.mesh);
  }

  update(music, dt, time) {
    this.mat.uniforms.uTime.value = time;
    // counter-drift: the sky slowly turns against the auto-orbit
    this.mesh.rotation.y = time * 0.004;
    const raw = music.getWave?.();
    if (!raw) {
      this.mat.uniforms.uLevel.value = Math.max(0, this.mat.uniforms.uLevel.value - dt);
      return;
    }
    const stride = raw.length / SAMPLES;
    let rms = 0;
    for (let i = 0; i < SAMPLES; i++) {
      let v = 0;
      const off = Math.floor(i * stride);
      for (let j = 0; j < stride; j++) v += raw[off + j];
      v /= stride;
      // temporal smoothing keeps the lines liquid instead of jittery
      this.smoothed[i] += (v - this.smoothed[i]) * 0.35;
      this.data[i] = this.smoothed[i];
      rms += v * v;
    }
    rms = Math.sqrt(rms / SAMPLES);
    this.level += (Math.min(1, rms * 7) - this.level) * Math.min(1, dt * 5);
    this.mat.uniforms.uLevel.value = music.muted ? 0 : this.level;
    this.tex.needsUpdate = true;
  }
}

import * as THREE from 'three';
import { makeTextSprite } from './textsprite.js';

const STEP = 0.205; // radians between tokens on the carousel
const R0 = 13.5;

// The generated sentence, orbiting the base of the mind-core like a galaxy ring.
// Newest token always faces the camera (azimuth 0 = +z); older tokens wind away.
export class TokenRibbon {
  constructor(scene) {
    this.group = new THREE.Group();
    scene.add(this.group);
    this.tokens = []; // {sprite, isPrompt, flash}
    this.threadGeo = new THREE.BufferGeometry();
    this.threadMax = 1024;
    this.threadGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.threadMax * 3), 3));
    this.threadGeo.setDrawRange(0, 0);
    this.thread = new THREE.Line(
      this.threadGeo,
      new THREE.LineBasicMaterial({ color: 0x2a6f99, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending }),
    );
    scene.add(this.thread);
  }

  slotPos(i, newest, out = new THREE.Vector3()) {
    const k = newest - i; // 0 = newest
    const angle = -k * STEP;
    const r = R0 + k * 0.035;
    const y = 0.4 + Math.sin(k * 0.45) * 0.5 - k * 0.012;
    out.set(Math.sin(angle) * r, y, Math.cos(angle) * r);
    return out;
  }

  // World position where the NEXT token will land.
  nextSlotPos(out = new THREE.Vector3()) {
    return this.slotPos(this.tokens.length, this.tokens.length, out);
  }

  addToken(text, { isPrompt = false } = {}) {
    const display = text.replace(/ /g, ' ');
    const sprite = makeTextSprite(display || '·', {
      color: isPrompt ? '#9dc4e4' : '#ffffff',
      glow: isPrompt ? '#23506f' : '#4be8ff',
      size: 46,
      scale: 0.0165,
    });
    sprite.userData.baseScale = sprite.scale.clone();
    const tok = { sprite, isPrompt, flash: 0, spawn: 0, text };
    this.tokens.push(tok);
    this.slotPos(this.tokens.length - 1, this.tokens.length - 1, sprite.position);
    sprite.material.opacity = 0;
    this.group.add(sprite);
    return this.tokens.length - 1;
  }

  flash(i, amount = 1) {
    if (this.tokens[i]) this.tokens[i].flash = Math.min(1.6, this.tokens[i].flash + amount);
  }

  worldPos(i, out = new THREE.Vector3()) {
    return out.copy(this.tokens[i].sprite.position);
  }

  update(dt) {
    const newest = this.tokens.length - 1;
    const target = new THREE.Vector3();
    const posAttr = this.threadGeo.attributes.position;
    for (let i = 0; i < this.tokens.length; i++) {
      const tk = this.tokens[i];
      this.slotPos(i, newest, target);
      tk.sprite.position.lerp(target, Math.min(1, dt * 6));
      tk.spawn = Math.min(1, tk.spawn + dt * 2.5);
      const k = newest - i;
      const distFade = Math.max(0.12, 1 - k * 0.012);
      tk.flash = Math.max(0, tk.flash - dt * 2.2);
      const f = tk.flash;
      tk.sprite.material.opacity = tk.spawn * distFade * (tk.isPrompt ? 0.75 : 1) + f * 0.3;
      const s = 1 + f * 0.55;
      tk.sprite.scale.copy(tk.sprite.userData.baseScale).multiplyScalar(s * (0.7 + 0.3 * tk.spawn));
      tk.sprite.material.color.setRGB(1, 1 - f * 0.18, 1 - f * 0.5); // flash toward gold
      if (i < this.threadMax) posAttr.setXYZ(i, tk.sprite.position.x, tk.sprite.position.y - 0.55, tk.sprite.position.z);
    }
    this.threadGeo.setDrawRange(0, Math.min(this.tokens.length, this.threadMax));
    posAttr.needsUpdate = true;
  }

  reset() {
    for (const tk of this.tokens) {
      tk.sprite.material.map.dispose();
      tk.sprite.material.dispose();
      this.group.remove(tk.sprite);
    }
    this.tokens = [];
    this.threadGeo.setDrawRange(0, 0);
  }
}

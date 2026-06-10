import * as THREE from 'three';

// Canvas-backed glowing text sprite. Returns THREE.Sprite sized to text.
export function makeTextSprite(text, { font = 600, size = 44, color = '#eaf6ff', glow = '#4be8ff', pad = 18, scale = 0.011, blur = 16 } = {}) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  const fontStr = `${font} ${size}px "SF Mono", ui-monospace, Menlo, monospace`;
  ctx.font = fontStr;
  const w = Math.ceil(ctx.measureText(text).width) + pad * 2;
  const h = size + pad * 2;
  c.width = w * 2;
  c.height = h * 2;
  ctx.scale(2, 2);
  ctx.font = fontStr;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.shadowColor = glow;
  ctx.shadowBlur = blur;
  ctx.fillStyle = color;
  ctx.fillText(text, w / 2, h / 2 + 2);
  ctx.shadowBlur = 0;
  ctx.fillText(text, w / 2, h / 2 + 2);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(w * scale, h * scale, 1);
  sprite.userData.aspect = w / h;
  return sprite;
}

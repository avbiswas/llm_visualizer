// GPT-Neo (TinyStories-33M) inference in pure JS, instrumented for visualization.
// Exposes per-layer attention probabilities, residual-stream stats, and full logits.

const _convU32 = new Uint32Array(1);
const _convF32 = new Float32Array(_convU32.buffer);

function f16ToF32Array(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) << 16;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;
    let bits;
    if (exp === 0) {
      if (frac === 0) {
        bits = sign;
      } else {
        // subnormal
        let e = -1;
        let f = frac;
        do {
          e++;
          f <<= 1;
        } while ((f & 0x400) === 0);
        bits = sign | ((127 - 15 - e) << 23) | ((f & 0x3ff) << 13);
      }
    } else if (exp === 0x1f) {
      bits = sign | 0x7f800000 | (frac << 13);
    } else {
      bits = sign | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    _convU32[0] = bits;
    out[i] = _convF32[0];
  }
  return out;
}

// y[m] = W[m,n] @ x[n] + b   (W row-major; rows are output neurons)
function matvec(W, x, b, out, m, n) {
  for (let i = 0; i < m; i++) {
    let sum = 0;
    const off = i * n;
    for (let j = 0; j < n; j++) sum += W[off + j] * x[j];
    out[i] = sum + (b ? b[i] : 0);
  }
}

function layerNorm(x, g, b, eps, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const d = x[i] - mean;
    varSum += d * d;
  }
  const inv = 1 / Math.sqrt(varSum / n + eps);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];
}

function geluNew(x) {
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    x[i] = 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
  }
}

// fixed sample of residual-stream dimensions shipped to the viz every token
export const PROBE_DIMS = Array.from({ length: 32 }, (_, k) => (k * 47 + 11) % 768);

function norm(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s);
}

export class GPTNeo {
  constructor(manifest, weightsBuffer, config, onProgress = null) {
    this.config = config;
    this.nLayer = config.num_layers;
    this.nHead = config.num_heads;
    this.nEmbd = config.hidden_size;
    this.headDim = this.nEmbd / this.nHead;
    this.nVocab = config.vocab_size;
    this.nInner = config.intermediate_size || 4 * this.nEmbd;
    this.window = config.window_size;
    this.attentionLayers = config.attention_layers; // ["global","local",...]
    this.eps = config.layer_norm_epsilon;

    const u8 = new Uint8Array(weightsBuffer);
    this.w = {};
    const entries = Object.entries(manifest);
    entries.forEach(([name, t], i) => {
      const u16 = new Uint16Array(u8.buffer, u8.byteOffset + t.offset, t.length / 2);
      this.w[name] = f16ToF32Array(u16);
      if (onProgress) onProgress((i + 1) / entries.length);
    });

    this.reset();

    // scratch buffers
    const E = this.nEmbd;
    this.buf = {
      x: new Float32Array(E),
      ln: new Float32Array(E),
      q: new Float32Array(E),
      k: new Float32Array(E),
      v: new Float32Array(E),
      attnOut: new Float32Array(E),
      proj: new Float32Array(E),
      fc: new Float32Array(this.nInner),
      logits: new Float32Array(this.nVocab),
    };
  }

  reset() {
    // KV cache: per layer, growable arrays of Float32Array(nEmbd)
    this.kCache = Array.from({ length: this.nLayer }, () => []);
    this.vCache = Array.from({ length: this.nLayer }, () => []);
    this.pos = 0;
  }

  layer(name) {
    return this.w[name];
  }

  // Runs one token through the model. Returns logits plus instrumentation:
  // attn[layer] = Float32Array(nHead * seqLen) of attention probs for this token,
  // stats[layer] = { residNorm, attnNorm, mlpNorm, headEntropy: Float32Array(nHead) }
  forward(tokenId, { captureAttention = true } = {}) {
    const { nLayer, nHead, nEmbd: E, headDim: D, eps } = this;
    const b = this.buf;
    const pos = this.pos;
    const w = this.w;

    const wte = w['transformer.wte.weight'];
    const wpe = w['transformer.wpe.weight'];
    for (let i = 0; i < E; i++) b.x[i] = wte[tokenId * E + i] + wpe[pos * E + i];

    const attnCapture = [];
    const stats = [];

    for (let L = 0; L < nLayer; L++) {
      const p = `transformer.h.${L}.`;
      layerNorm(b.x, w[p + 'ln_1.weight'], w[p + 'ln_1.bias'], eps, b.ln);

      matvec(w[p + 'attn.attention.q_proj.weight'], b.ln, null, b.q, E, E);
      matvec(w[p + 'attn.attention.k_proj.weight'], b.ln, null, b.k, E, E);
      matvec(w[p + 'attn.attention.v_proj.weight'], b.ln, null, b.v, E, E);

      this.kCache[L].push(Float32Array.from(b.k));
      this.vCache[L].push(Float32Array.from(b.v));
      const ks = this.kCache[L];
      const vs = this.vCache[L];
      const seqLen = ks.length;

      const isLocal = this.attentionLayers[L] === 'local';
      const start = isLocal ? Math.max(0, seqLen - this.window) : 0;
      const span = seqLen - start;

      const probsAll = captureAttention ? new Float32Array(nHead * seqLen) : null;
      const headEntropy = new Float32Array(nHead);
      const scores = new Float32Array(span);

      for (let h = 0; h < nHead; h++) {
        const hOff = h * D;
        // GPT-Neo applies no 1/sqrt(headDim) scaling.
        let maxScore = -Infinity;
        for (let t = 0; t < span; t++) {
          const kt = ks[start + t];
          let s = 0;
          for (let d = 0; d < D; d++) s += b.q[hOff + d] * kt[hOff + d];
          scores[t] = s;
          if (s > maxScore) maxScore = s;
        }
        let sum = 0;
        for (let t = 0; t < span; t++) {
          scores[t] = Math.exp(scores[t] - maxScore);
          sum += scores[t];
        }
        let ent = 0;
        for (let t = 0; t < span; t++) {
          const pr = scores[t] / sum;
          scores[t] = pr;
          if (pr > 1e-12) ent -= pr * Math.log(pr);
          if (probsAll) probsAll[h * seqLen + start + t] = pr;
        }
        headEntropy[h] = ent;
        for (let d = 0; d < D; d++) {
          let acc = 0;
          for (let t = 0; t < span; t++) acc += scores[t] * vs[start + t][hOff + d];
          b.attnOut[hOff + d] = acc;
        }
      }

      matvec(w[p + 'attn.attention.out_proj.weight'], b.attnOut, w[p + 'attn.attention.out_proj.bias'], b.proj, E, E);
      const attnNorm = norm(b.proj);
      for (let i = 0; i < E; i++) b.x[i] += b.proj[i];

      layerNorm(b.x, w[p + 'ln_2.weight'], w[p + 'ln_2.bias'], eps, b.ln);
      matvec(w[p + 'mlp.c_fc.weight'], b.ln, w[p + 'mlp.c_fc.bias'], b.fc, this.nInner, E);
      geluNew(b.fc);
      matvec(w[p + 'mlp.c_proj.weight'], b.fc, w[p + 'mlp.c_proj.bias'], b.proj, E, this.nInner);
      const mlpNorm = norm(b.proj);
      for (let i = 0; i < E; i++) b.x[i] += b.proj[i];

      const residSample = new Float32Array(PROBE_DIMS.length);
      for (let i = 0; i < PROBE_DIMS.length; i++) residSample[i] = b.x[PROBE_DIMS[i]];
      attnCapture.push(probsAll);
      stats.push({ residNorm: norm(b.x), attnNorm, mlpNorm, headEntropy, residSample });
    }

    layerNorm(b.x, w['transformer.ln_f.weight'], w['transformer.ln_f.bias'], eps, b.ln);
    matvec(wte, b.ln, null, b.logits, this.nVocab, E);

    this.pos++;
    return { logits: b.logits, attn: attnCapture, stats, seqLen: this.pos };
  }
}

// Softmax over logits with temperature; returns sorted top-k [{id, p}].
export function topkProbs(logits, k, temperature = 1.0) {
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  const t = Math.max(temperature, 1e-4);
  let sum = 0;
  const exps = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    exps[i] = Math.exp((logits[i] - max) / t);
    sum += exps[i];
  }
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b2) => exps[b2] - exps[a]);
  const top = [];
  for (let i = 0; i < k; i++) top.push({ id: idx[i], p: exps[idx[i]] / sum });
  let entropy = 0;
  for (let i = 0; i < n; i++) {
    const pr = exps[i] / sum;
    if (pr > 1e-12) entropy -= pr * Math.log2(pr);
  }
  return { top, entropy };
}

// Sample from top-k distribution (renormalized).
export function sampleTopK(top, rng = Math.random) {
  let sum = 0;
  for (const t of top) sum += t.p;
  let r = rng() * sum;
  for (const t of top) {
    r -= t.p;
    if (r <= 0) return t.id;
  }
  return top[0].id;
}

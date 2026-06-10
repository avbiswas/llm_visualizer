// Runs TinyStories-33M off the main thread. Protocol:
//   in : {type:'load'} | {type:'start', prompt, temperature, topk} | {type:'next'} | {type:'reset'}
//   out: {type:'progress'} | {type:'ready'} | {type:'prompt_done', tokens:[{id,text}]}
//        {type:'token', ...instrumentation}
import { Tokenizer } from './tokenizer.js';
import { GPTNeo, topkProbs, sampleTopK } from './model.js';

let model = null;
let tokenizer = null;
let temperature = 0.8;
let topk = 40;
let lastLogits = null;
let lastForward = null;
const EOS = 50256;
const STORM_K = 384; // candidates sent for the probability storm

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  const total = +res.headers.get('Content-Length') || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  return buf.buffer;
}

async function load() {
  const base = '/model/';
  const post = (stage, frac) => postMessage({ type: 'progress', stage, frac });
  post('downloading weights', 0);
  const [config, manifest, vocab, merges] = await Promise.all([
    fetch(base + 'config.json').then((r) => r.json()),
    fetch(base + 'manifest.json').then((r) => r.json()),
    fetch(base + 'vocab.json').then((r) => r.json()),
    fetch(base + 'merges.txt').then((r) => r.text()),
  ]);
  const weights = await fetchWithProgress(base + 'weights.bin', (l, t) =>
    post('downloading weights', t ? (l / t) * 0.7 : 0),
  );
  post('decoding fp16 → fp32', 0.72);
  await new Promise((r) => setTimeout(r, 30)); // let the message flush
  tokenizer = new Tokenizer(vocab, merges);
  model = new GPTNeo(manifest, weights, config, (frac) => post('decoding fp16 → fp32', 0.72 + frac * 0.28));
  post('ready', 1);
  postMessage({ type: 'ready' });
}

function instrument(fw, sampledId) {
  // Trim attention to top weights so we don't ship 16*seq floats x 4 layers raw.
  // For each layer: aggregate over heads -> top 8 attended positions; plus per-head top-1.
  const layers = fw.attn.map((probs, L) => {
    const seq = fw.seqLen;
    const nHead = model.nHead;
    const agg = new Float32Array(seq);
    for (let h = 0; h < nHead; h++)
      for (let t = 0; t < seq; t++) agg[t] += probs[h * seq + t];
    for (let t = 0; t < seq; t++) agg[t] /= nHead;
    const idx = Array.from({ length: seq }, (_, i) => i).sort((a, b) => agg[b] - agg[a]);
    const top = idx.slice(0, 8).map((t) => ({ t, w: agg[t] }));
    const headTop = [];
    for (let h = 0; h < nHead; h++) {
      let best = 0;
      for (let t = 1; t < seq; t++) if (probs[h * seq + t] > probs[h * seq + best]) best = t;
      headTop.push({ h, t: best, w: probs[h * seq + best] });
    }
    const s = fw.stats[L];
    let entSum = 0;
    for (let h = 0; h < nHead; h++) entSum += s.headEntropy[h];
    return {
      top,
      headTop,
      residNorm: s.residNorm,
      attnNorm: s.attnNorm,
      mlpNorm: s.mlpNorm,
      meanEntropy: entSum / nHead,
      residSample: s.residSample,
    };
  });
  return layers;
}

function emitToken() {
  const { top, entropy } = topkProbs(lastLogits, Math.max(topk, STORM_K), temperature);
  const sampled = sampleTopK(top.slice(0, topk));
  const fw = model.forward(sampled, { captureAttention: true });
  lastLogits = fw.logits;
  const stormIds = new Int32Array(STORM_K);
  const stormPs = new Float32Array(STORM_K);
  for (let i = 0; i < STORM_K; i++) {
    stormIds[i] = top[i].id;
    stormPs[i] = top[i].p;
  }
  const labels = top.slice(0, 40).map((t) => ({ text: tokenizer.decodeOne(t.id), p: t.p }));
  const winnerStormIndex = top.findIndex((t) => t.id === sampled);
  let winnerLabelIndex = winnerStormIndex < 40 ? winnerStormIndex : -1;
  if (winnerLabelIndex === -1) {
    labels.push({ text: tokenizer.decodeOne(sampled), p: top[winnerStormIndex]?.p ?? 0 });
    winnerLabelIndex = labels.length - 1;
  }
  const sampledP = top.find((t) => t.id === sampled)?.p ?? 0;
  postMessage({
    type: 'token',
    id: sampled,
    text: tokenizer.decodeOne(sampled),
    p: sampledP,
    entropy,
    labels,
    winnerStormIndex,
    winnerLabelIndex,
    stormIds,
    stormPs,
    layers: instrument(fw, sampled),
    seqLen: fw.seqLen,
    done: sampled === EOS,
  });
}

onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'load') {
    await load();
  } else if (msg.type === 'start') {
    temperature = msg.temperature ?? 0.8;
    topk = msg.topk ?? 40;
    model.reset();
    const ids = tokenizer.encode(msg.prompt);
    let fw = null;
    for (const id of ids) fw = model.forward(id, { captureAttention: false });
    lastLogits = fw.logits;
    postMessage({
      type: 'prompt_done',
      tokens: ids.map((id) => ({ id, text: tokenizer.decodeOne(id) })),
    });
  } else if (msg.type === 'next') {
    emitToken();
  } else if (msg.type === 'set') {
    if (msg.temperature !== undefined) temperature = msg.temperature;
  }
};

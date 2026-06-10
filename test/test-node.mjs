import { readFileSync } from 'node:fs';
import { Tokenizer } from '../src/engine/tokenizer.js';
import { GPTNeo, topkProbs, sampleTopK } from '../src/engine/model.js';

const dir = new URL('../public/model/', import.meta.url);
const config = JSON.parse(readFileSync(new URL('config.json', dir), 'utf8'));
const manifest = JSON.parse(readFileSync(new URL('manifest.json', dir), 'utf8'));
const vocab = JSON.parse(readFileSync(new URL('vocab.json', dir), 'utf8'));
const merges = readFileSync(new URL('merges.txt', dir), 'utf8');
const weights = readFileSync(new URL('weights.bin', dir));

const tok = new Tokenizer(vocab, merges);
const ids = tok.encode('Once upon a time, there was a little girl named Lily.');
console.log('encoded:', ids.join(' '));
console.log('roundtrip:', JSON.stringify(tok.decode(ids)));

console.time('load');
const model = new GPTNeo(manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength), config);
console.timeEnd('load');

const prompt = process.argv[2] || 'Once upon a time, there was a little girl named Lily.';
const promptIds = tok.encode(prompt);
let out = null;
console.time('prompt');
for (const id of promptIds) out = model.forward(id, { captureAttention: false });
console.timeEnd('prompt');

const genIds = [];
const N = 60;
console.time('generate');
let last = null;
for (let i = 0; i < N; i++) {
  const { top, entropy } = topkProbs(out.logits, 40, 0.8);
  const id = sampleTopK(top);
  genIds.push(id);
  if (i === 0) {
    console.log('first-token top5:', top.slice(0, 5).map((t) => `${JSON.stringify(tok.decodeOne(t.id))}:${t.p.toFixed(3)}`).join(' '), '| entropy', entropy.toFixed(2));
  }
  out = model.forward(id, { captureAttention: true });
  last = out;
}
console.timeEnd('generate');
console.log('tokens/sec:', (N / 1).toFixed(1), '(see generate time)');
console.log('attention shape: layers', last.attn.length, 'x', last.attn[0].length, '(heads*seq), seq =', last.seqLen);
console.log('layer stats:', last.stats.map((s) => `resid=${s.residNorm.toFixed(1)} attn=${s.attnNorm.toFixed(1)} mlp=${s.mlpNorm.toFixed(1)}`).join(' | '));
console.log('\n--- OUTPUT ---\n' + prompt + tok.decode(genIds));

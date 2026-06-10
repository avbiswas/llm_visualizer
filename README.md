# INSIDE THE MIND

https://github.com/user-attachments/assets/3e36aaca-e60e-4f1c-a864-1da1bf94622d

> _Produced by Claude Fable-5_

**A real language model, thinking, live in your browser.** TinyStories-33M
(GPT-Neo, 68M parameters) runs entirely client-side in a Web Worker — the
inference engine is hand-written JS (`src/engine/model.js`), which exposes what
off-the-shelf runtimes hide: every attention head, the residual stream, and the
full next-token probability distribution. The visualization renders all of it
with three.js + UnrealBloom:

- **The carousel** — the generated story orbits the base; each word is a sprite.
- **The core** — the residual stream spirals up through 4 transformer-layer rings
  (cyan → magenta). Ring pulses and river brightness are driven by real
  per-layer activation norms.
- **Attention strikes** — when a layer attends to earlier words, arcs of that
  layer's color strike down at exactly the tokens being read (top
  head-averaged attention weights), and those words flash gold.
- **The probability storm** — 5,200 particles at the crown, one swarm per
  candidate token, radius ∝ improbability, heat ∝ probability. Real top-384
  softmax values every token. The storm collapses, the sampled word condenses
  in gold and falls into the sentence.

Nothing is faked. Every spark is real math from the actual forward pass.

## Run

```bash
npm install
npm run prepare-model   # one-time: downloads + converts weights (needs python3 + torch)
npm run dev             # open the printed URL
```

The model auto-dreams on load. Controls: prompt box + **DREAM**, HEAT
(temperature), PACE (animation speed). Drag to orbit, scroll to zoom.
Press **H** to hide the control bar for clean video capture.

If `model_src/` is already populated, `prepare-model` only re-runs the
conversion. The converted model lives in `public/model/` (~137MB fp16).

## Files

- `src/engine/tokenizer.js` — GPT-2 byte-level BPE
- `src/engine/model.js` — GPT-Neo forward pass (KV cache, attention capture, top-k)
- `src/engine/worker.js` — off-main-thread inference + instrumentation protocol
- `src/viz/` — scene, token carousel, layer rings + residual river, attention
  tethers, probability storm
- `test/test-node.mjs` — engine smoke test (`node test/test-node.mjs "A prompt"`)
- `test/snap.py` — headless screenshot test (playwright)

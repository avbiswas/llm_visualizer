import assert from 'node:assert/strict';
import { residualChroma } from '../src/engine/model.js';
import { dominantBand, midiToHz, modelNote, surpriseBits } from '../src/audio/music.js';

const residual = new Float32Array(24);
residual[5] = 3;
residual[17] = 4;
residual[2] = 1;
const chroma = residualChroma(residual);

assert.equal(dominantBand(chroma), 5);
assert.ok(Math.abs(chroma.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
assert.equal(residualChroma(new Float32Array(12)).every((value) => value === 0), true);

const quiet = new Float32Array(12);
quiet[3] = 1;
assert.equal(modelNote(2, quiet, 0, 0), 52);
assert.equal(modelNote(2, quiet, 2, 1), 74);
assert.ok(Math.abs(midiToHz(69) - 440) < 1e-9);
assert.equal(surpriseBits(1), 0);
assert.equal(surpriseBits(0.25), 2);
assert.equal(surpriseBits(0), 16);

console.log('audio mapping tests passed');

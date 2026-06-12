const DORIAN = [0, 2, 3, 5, 7, 9, 10];

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, Number.isFinite(v) ? v : lo));

export function dominantBand(chroma) {
  let best = 0;
  for (let i = 1; i < 12; i++) {
    if ((chroma?.[i] ?? 0) > (chroma?.[best] ?? 0)) best = i;
  }
  return best;
}

export function modelNote(tonic, chroma, layer = 0, octave = 0) {
  const degree = dominantBand(chroma) % DORIAN.length;
  return 45 + tonic + DORIAN[degree] + layer * 5 + octave * 12;
}

export function midiToHz(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function surpriseBits(probability) {
  return clamp(-Math.log2(Math.max(1e-6, probability || 0)), 0, 16);
}

// Five modes from radiant to shadowed. The PROMPT picks one: the interval
// between the two strongest chroma bands of its final residual state maps to
// a mode, so different openings genuinely sound like different pieces.
const MODES = [
  { name: 'lydian', scale: [0, 2, 4, 6, 7, 9, 11] }, //     wonder
  { name: 'major', scale: [0, 2, 4, 5, 7, 9, 11] }, //      warmth
  { name: 'mixolydian', scale: [0, 2, 4, 5, 7, 9, 10] }, // adventure
  { name: 'dorian', scale: [0, 2, 3, 5, 7, 9, 10] }, //     bittersweet
  { name: 'aeolian', scale: [0, 2, 3, 5, 7, 8, 10] }, //    melancholy
];
// interval between top two chroma bands -> mode index (bright thirds/sixths
// land radiant, minor intervals land shadowed)
const INTERVAL_MODE = [2, 4, 2, 3, 0, 1, 0, 1, 4, 1, 3, 0];

// Functional harmony grammar: which scale-degree chords may follow each
// degree. The model's hidden state picks WHICH allowed door to walk through,
// so progressions wander with the story but never sound wrong.
const NEXT_DEGREES = [
  [3, 4, 5, 1, 2, 0], // from I
  [4, 3, 0], //          from ii
  [5, 3, 0], //          from iii
  [0, 4, 1, 5], //       from IV
  [0, 5, 3], //          from V
  [3, 1, 4, 0], //       from vi
];

export class ModelMusic {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.musicBus = null;
    this.fxBus = null;
    this.zoomGain = null;
    this.zoomFilter = null;
    this.tonic = 0;
    this.muted = false;
    this.drone = [];

    // continuous engine state
    this.clock = null;
    this.step = 0;
    this.nextTime = 0;
    this.bpm = 108;
    this.energy = 0.4; // how hard the piano plays; bumped by model activity
    this.tension = 0.3; // entropy of the storm; colors the voicings
    this.sparkle = 0; // one-shot flourish budget from token landings
    this.arpDir = 1;
    this.arpPos = 0;

    // model-composed material
    this.mode = MODES[1]; // until a prompt chooses
    this.chordDeg = 0; //    current scale-degree chord
    this.nextChordDeg = 0; // chosen by each token's hidden state
    this.melody = []; //     queue of {deg, dur} distilled from residual streams
    this.melodyHold = 0; //  beats left on the current lead note
    this.tokensHeard = 0; // position -> rhythm masks + sections

    // camera-as-instrument state (fed by updateCamera every frame)
    this.proximity = 0.6; // 1 = nose against the mind, 0 = drifting far away
    this.panSpin = 0; // stereo field rotates with the orbit azimuth
    this.swirl = 0; // accumulated fast manual rotation -> sparkle notes
  }

  async unlock({ muted = false } = {}) {
    if (!this.ctx) this.createGraph();
    this.muted = muted;
    this.master.gain.setValueAtTime(muted ? 0 : 1.0, this.ctx.currentTime);
    if (this.ctx.state === 'suspended') await this.ctx.resume();
  }

  createGraph() {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AudioCtx();
    const ctx = this.ctx;

    this.musicBus = ctx.createGain();
    this.musicBus.gain.value = 0.9;
    this.fxBus = ctx.createGain();
    this.fxBus.gain.value = 0.34;
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    const compressor = ctx.createDynamicsCompressor();
    compressor.threshold.value = -14;
    compressor.knee.value = 20;
    compressor.ratio.value = 5;
    compressor.attack.value = 0.004;
    compressor.release.value = 0.26;

    this.delay = ctx.createDelay(1);
    this.delay.delayTime.value = 60 / this.bpm / 2; // echo locked to tempo (follows zoom pacing)
    const feedback = ctx.createGain();
    feedback.gain.value = 0.3;
    this.delay.connect(feedback).connect(this.delay);
    this.fxBus.connect(this.delay).connect(this.master);

    const convolver = ctx.createConvolver();
    convolver.buffer = this.makeImpulse(3.2, 2.4);
    this.fxBus.connect(convolver).connect(this.master);
    this.musicBus.connect(this.master);
    // proximity filter: the whole mix muffles as you drift away from the mind
    this.proximityFilter = ctx.createBiquadFilter();
    this.proximityFilter.type = 'lowpass';
    this.proximityFilter.frequency.value = 6500;
    this.proximityFilter.Q.value = 0.3;
    this.master.connect(this.proximityFilter).connect(compressor).connect(ctx.destination);

    // shimmer: airy high noise that blooms when you spin the camera fast
    const shimmerNoise = ctx.createBufferSource();
    shimmerNoise.buffer = this.makeNoise(2.3);
    shimmerNoise.loop = true;
    const shimmerFilter = ctx.createBiquadFilter();
    shimmerFilter.type = 'bandpass';
    shimmerFilter.frequency.value = 4800;
    shimmerFilter.Q.value = 0.7;
    this.shimmerGain = ctx.createGain();
    this.shimmerGain.gain.value = 0;
    shimmerNoise.connect(shimmerFilter).connect(this.shimmerGain).connect(this.master);
    shimmerNoise.start();

    const zoomNoise = ctx.createBufferSource();
    zoomNoise.buffer = this.makeNoise(2);
    zoomNoise.loop = true;
    this.zoomFilter = ctx.createBiquadFilter();
    this.zoomFilter.type = 'bandpass';
    this.zoomFilter.frequency.value = 500;
    this.zoomFilter.Q.value = 1.2;
    this.zoomGain = ctx.createGain();
    this.zoomGain.gain.value = 0;
    zoomNoise.connect(this.zoomFilter).connect(this.zoomGain).connect(this.master);
    zoomNoise.start();
  }

  makeNoise(seconds) {
    const length = Math.floor(this.ctx.sampleRate * seconds);
    const buffer = this.ctx.createBuffer(1, length, this.ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i++) data[i] = Math.random() * 2 - 1;
    return buffer;
  }

  makeImpulse(seconds, decay) {
    const length = Math.floor(this.ctx.sampleRate * seconds);
    const buffer = this.ctx.createBuffer(2, length, this.ctx.sampleRate);
    for (let channel = 0; channel < 2; channel++) {
      const data = buffer.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
      }
    }
    return buffer;
  }

  // live time-domain signal of everything being heard (post proximity filter),
  // for the background oscilloscope. Returns null until audio is unlocked.
  getWave() {
    if (!this.ctx) return null;
    if (!this.analyser) {
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.proximityFilter.connect(this.analyser);
      this.waveBuf = new Float32Array(this.analyser.fftSize);
    }
    this.analyser.getFloatTimeDomainData(this.waveBuf);
    return this.waveBuf;
  }

  setMuted(muted) {
    this.muted = muted;
    if (!this.ctx) return;
    this.master.gain.cancelScheduledValues(this.ctx.currentTime);
    this.master.gain.setTargetAtTime(muted ? 0 : 1.0, this.ctx.currentTime, 0.035);
  }

  // ---------- session ----------

  beginSession(chroma) {
    if (!this.ctx) return;
    // the prompt's final hidden state picks both KEY and MODE. The dominant
    // band alone is too stable across prompts (a few residual dims always
    // dominate), so the key mixes the top two bands for real prompt spread.
    const best = dominantBand(chroma);
    let second = best === 0 ? 1 : 0;
    for (let i = 0; i < 12; i++) {
      if (i !== best && (chroma?.[i] ?? 0) > (chroma?.[second] ?? 0)) second = i;
    }
    this.tonic = (best + 7 * second) % 12;
    this.mode = MODES[INTERVAL_MODE[(second - best + 12) % 12]];
    console.log(`[music] key ${this.tonic} · mode ${this.mode.name}`);
    this.stopDrone();
    this.startPad();
    this.chordDeg = 0;
    this.nextChordDeg = 0;
    this.melody = [];
    this.melodyHold = 0;
    this.tokensHeard = 0;
    this.energy = 0.45;
    this.tension = 0.3;
    this.startClock();
  }

  endSession() {
    // let the piano ring out, settle the pad — but keep a heartbeat so the
    // scene never goes dead silent; the next session re-tunes it.
    this.energy = 0.18;
  }

  startPad() {
    const now = this.ctx.currentTime;
    const root = midiToHz(33 + this.tonic);
    // warm pad: detuned triangles on root + fifth + tenth, slow filter breath
    [
      [1, 0, 'triangle', 0.035],
      [1.005, 0, 'sawtooth', 0.012],
      [1.5, 1, 'triangle', 0.022],
      [2.52, 2, 'sine', 0.014], // major tenth — instant optimism
    ].forEach(([ratio, i, type, level]) => {
      const osc = this.ctx.createOscillator();
      const filter = this.ctx.createBiquadFilter();
      const gain = this.ctx.createGain();
      osc.type = type;
      osc.frequency.value = root * ratio;
      filter.type = 'lowpass';
      filter.frequency.value = 320 + i * 160;
      const lfo = this.ctx.createOscillator();
      const lfoGain = this.ctx.createGain();
      lfo.frequency.value = 0.07 + i * 0.03;
      lfoGain.gain.value = 90;
      lfo.connect(lfoGain).connect(filter.frequency);
      lfo.start(now);
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(level, now + 2.5);
      osc.connect(filter).connect(gain).connect(this.musicBus);
      gain.connect(this.fxBus);
      osc.start(now);
      this.drone.push({ osc, gain, lfo });
    });
  }

  stopDrone() {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    for (const voice of this.drone) {
      voice.gain.gain.cancelScheduledValues(now);
      voice.gain.gain.setTargetAtTime(0, now, 0.3);
      voice.osc.stop(now + 1.2);
      if (voice.lfo) voice.lfo.stop(now + 1.2);
    }
    this.drone = [];
  }

  // ---------- the continuous piano ----------

  startClock() {
    if (this.clock) return;
    this.step = 0;
    this.nextTime = this.ctx.currentTime + 0.06;
    this.clock = setInterval(() => this.scheduleAhead(), 50);
  }

  stopClock() {
    if (this.clock) clearInterval(this.clock);
    this.clock = null;
  }

  scheduleAhead() {
    if (!this.ctx) return;
    const horizon = this.ctx.currentTime + 0.16;
    const stepDur = 60 / this.bpm / 2; // continuous eighth notes
    while (this.nextTime < horizon) {
      this.scheduleStep(this.step, this.nextTime, stepDur);
      this.nextTime += stepDur;
      this.step++;
    }
  }

  // a scale-degree note as semitones above the tonic, carrying octaves
  scaleAt(deg) {
    const s = this.mode.scale;
    return s[((deg % 7) + 7) % 7] + 12 * Math.floor(deg / 7);
  }

  // stacked-thirds chord on the current degree, with a 7th and 9th for color
  chord() {
    const d = this.chordDeg;
    return {
      root: 48 + this.tonic,
      intervals: [this.scaleAt(d), this.scaleAt(d + 2), this.scaleAt(d + 4), this.scaleAt(d + 6), this.scaleAt(d + 8)],
    };
  }

  // position-in-sequence rotates the accompaniment rhythm: the piece develops
  // new grooves as the story gets longer instead of looping one pattern
  static RHYTHMS = [
    [1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 1, 1, 0],
  ];

  scheduleStep(step, when, stepDur) {
    // harmony moves at bar boundaries, to wherever the last token pointed it
    if (step % 8 === 0) this.chordDeg = this.nextChordDeg;
    const { root, intervals } = this.chord();
    const e = clamp(this.energy, 0.12, 1);
    this.energy = Math.max(0.3, this.energy * 0.992); // breathe back toward a base groove

    // bass: chord root, its fifth answering on the back half
    if (step % 8 === 0) this.bass(midiToHz(root + intervals[0] - 24), when, stepDur * 7);
    else if (step % 8 === 4) this.bass(midiToHz(root + intervals[2] - 24), when, stepDur * 3, 0.55);

    // every 16 tokens the piece enters a new section: calm -> full -> lifted
    const section = Math.floor(this.tokensHeard / 16) % 3;
    const mask = ModelMusic.RHYTHMS[(Math.floor(this.tokensHeard / 8) + section) % ModelMusic.RHYTHMS.length];
    const lift = (this.proximity > 0.72 ? 12 : 0) + (section === 2 ? 12 : 0);

    // LEAD: the melody the model wrote — residual-stream phrases, one note per
    // beat, held longer for longer words
    if (this.melodyHold > 0) this.melodyHold--;
    else if (step % 2 === 0 && this.melody.length) {
      const note = this.melody.shift();
      this.melodyHold = note.dur * 2 - 1;
      this.piano(midiToHz(60 + this.tonic + this.scaleAt(note.deg) + lift), clamp(0.5 + 0.4 * e, 0, 1) , when, {
        pan: clamp(this.panSpin * 0.4, -1, 1),
        bright: 1.15 + this.proximity * 0.3,
      });
    }

    // ACCOMPANIMENT: chord tones under the lead, gated by the position rhythm;
    // far away it thins out, calm phrases leave air
    const span = intervals.length + Math.round(e * 2);
    this.arpPos += this.arpDir;
    if (this.arpPos >= span - 1) this.arpDir = -1;
    if (this.arpPos <= 0) this.arpDir = 1;
    const pos = clamp(this.arpPos, 0, span - 1);
    const interval = intervals[pos % intervals.length] + 12 * Math.floor(pos / intervals.length);
    const beatAccent = step % 4 === 0 ? 1 : step % 2 === 0 ? 0.78 : 0.6;
    const vel = clamp((0.28 + 0.55 * e) * beatAccent, 0.08, 0.85);
    const playing = mask[step % 8] && !(this.proximity < 0.3 && step % 2 === 1) && !(e < 0.4 && step % 8 === 6);
    if (playing) {
      this.piano(midiToHz(root + interval), vel * (0.85 + this.proximity * 0.3), when, {
        pan: clamp(-0.3 + (pos / Math.max(1, span - 1)) * 0.6 + this.panSpin, -1, 1),
        bright: 0.8 + this.proximity * 0.4,
      });
    }

    // sparkle: token landings buy bright echoes two octaves up on off-beats
    if (this.sparkle > 0 && step % 2 === 1) {
      this.sparkle--;
      const si = intervals[(pos + 2) % intervals.length];
      this.piano(midiToHz(root + 36 + si), 0.5, when + stepDur / 2, {
        pan: clamp(0.4 + this.panSpin, -1, 1),
        bright: 1.4,
      });
    }
  }

  // a piano-ish key: stacked partials, hammer attack, long ringing decay
  piano(frequency, velocity, when, { pan = 0, bright = 1 } = {}) {
    if (!this.ctx || this.muted) return;
    const v = clamp(velocity, 0.05, 1);
    const out = this.ctx.createGain();
    const panner = this.ctx.createStereoPanner();
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = clamp((900 + v * 5200) * bright, 400, 9000);
    filter.Q.value = 0.4;
    out.connect(filter).connect(panner).connect(this.musicBus);
    panner.pan.value = clamp(pan, -1, 1);
    panner.connect(this.fxBus);

    const ring = 1.1 + v * 1.6;
    [
      [1, 1, 0],
      [1.0012, 0.42, 0], // unison detune — the shimmer inside a real string
      [2, 0.28, 0.001],
      [3.01, 0.1, 0.002],
    ].forEach(([ratio, level, lag]) => {
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = ratio < 2 ? 'triangle' : 'sine';
      osc.frequency.value = clamp(frequency * ratio, 30, 9000);
      const t = when + lag;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.16 * v * level, t + 0.006); // hammer
      g.gain.exponentialRampToValueAtTime(0.0001, t + ring * (ratio < 2 ? 1 : 0.45));
      osc.connect(g).connect(out);
      osc.start(t);
      osc.stop(t + ring + 0.1);
    });
  }

  bass(frequency, when, hold, level = 1) {
    if (!this.ctx || this.muted) return;
    const osc = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    const filter = this.ctx.createBiquadFilter();
    osc.type = 'sine';
    osc2.type = 'triangle';
    osc.frequency.value = frequency;
    osc2.frequency.value = frequency * 2.003;
    filter.type = 'lowpass';
    filter.frequency.value = 300;
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(0.14 * level, when + 0.02);
    g.gain.setTargetAtTime(0.0001, when + hold * 0.7, 0.12);
    osc.connect(filter);
    osc2.connect(filter);
    filter.connect(g).connect(this.musicBus);
    osc.start(when);
    osc2.start(when);
    osc.stop(when + hold + 0.5);
    osc2.stop(when + hold + 0.5);
  }

  // ---------- model-driven events ----------

  playPromptToken(id, index) {
    // prompt tokens read in the session's own scale — each token id is a degree
    const midi = 60 + this.tonic + this.scaleAt(Math.abs(id) % 9) + (index % 4 === 3 ? 12 : 0);
    this.piano(midiToHz(midi), 0.55, this.ctx ? this.ctx.currentTime : 0, {
      pan: -0.4 + (index % 5) * 0.2,
      bright: 1.2,
    });
  }

  playLayer(layer, data, activity, seqLen, pace = 1) {
    // layers feed the engine rather than blipping on their own: activity makes
    // the piano play harder, focus brightens a chord-tone accent per layer
    this.energy = clamp(this.energy + activity * 0.16, 0, 1.15);
    const focus = 1 - clamp(data.meanEntropy / 4, 0, 1);
    if (focus > 0.45 && this.ctx) {
      const { root, intervals } = this.chord();
      const interval = intervals[layer % intervals.length];
      this.piano(midiToHz(root + 12 + interval + (layer >= 2 ? 12 : 0)), 0.3 + 0.4 * focus, this.ctx.currentTime, {
        pan: -0.55 + layer * 0.36,
        bright: 0.9 + focus * 0.5,
      });
    }
  }

  playStorm(entropy, probability) {
    if (!this.ctx) return;
    this.tension = clamp(entropy / 7, 0, 1);
    const now = this.ctx.currentTime;
    const { root, intervals } = this.chord();
    // a rolled chord swells out of the storm — the 3rd suspends to a 4th when
    // the model is uncertain, resolving only once it commits
    const tones =
      this.tension > 0.55
        ? [intervals[0], this.scaleAt(this.chordDeg + 3), intervals[2], intervals[0] + 12]
        : intervals;
    tones.forEach((interval, i) => {
      this.piano(midiToHz(root + interval), 0.34 + probability * 0.3, now + i * 0.045, {
        pan: -0.5 + (i / Math.max(1, tones.length - 1)),
        bright: 1 + this.tension * 0.5,
      });
    });
    this.noiseHit(0.02 + this.tension * 0.04, 0.5, 600 + this.tension * 2600);
  }

  // Every generated token composes: its hidden state steers the harmony, its
  // residual stream is distilled into the melody, its position shapes form.
  onToken({ layers, p, text = '', seqLen = 0 }) {
    if (!this.ctx) return;
    const final = layers?.at(-1);
    if (!final) return;
    this.tokensHeard = seqLen;

    // 1 ── melody: fold the token's 32 sampled residual dims into a 4-note
    // phrase. Same thought -> same phrase; kindred words sing kindred lines.
    const rs = final.residSample;
    let phrase = null;
    if (rs?.length >= 32) {
      const dur = text.trim().length > 5 ? 2 : 1;
      phrase = [];
      for (let k = 0; k < 4; k++) {
        let v = 0;
        for (let j = 0; j < 8; j++) v += rs[k * 8 + j];
        phrase.push({ deg: Math.abs(Math.round(v * 2.5)) % 14, dur }); // two octaves of scale
      }
      this.melody.push(...phrase);
      if (this.melody.length > 16) this.melody.splice(0, this.melody.length - 16);
    }

    // 2 ── harmony: the token's melodic fingerprint nominates a scale degree
    // (the raw chroma band is too static to steer with); the grammar walks to
    // the closest allowed chord, preferring to MOVE rather than sit still
    const target = (phrase ? phrase[0].deg : dominantBand(final.residChroma)) % 7;
    const doors = NEXT_DEGREES[this.chordDeg % NEXT_DEGREES.length];
    const moves = doors.filter((d) => d !== this.chordDeg);
    this.nextChordDeg = (moves.length ? moves : doors).reduce(
      (best, d) => (Math.abs(d - target) < Math.abs(best - target) ? d : best),
      (moves.length ? moves : doors)[0],
    );

    // 3 ── the landing itself: a confident chord-tone hit, runs when surprised
    const surprise = surpriseBits(p);
    const now = this.ctx.currentTime;
    const { root, intervals } = this.chord();
    this.piano(midiToHz(root + 24 + intervals[0]), 0.85, now, { bright: 1.3 });
    this.piano(midiToHz(root + 12 + intervals[2 % intervals.length]), 0.6, now + 0.02, { pan: -0.2 });
    if (surprise > 3.5) {
      [0, 1, 2, 3].forEach((k) => {
        const interval = intervals[k % intervals.length] + 12 * Math.floor(k / intervals.length);
        this.piano(midiToHz(root + 24 + interval), 0.5, now + 0.06 + k * 0.06, { pan: 0.3, bright: 1.5 });
      });
    }
    this.energy = clamp(this.energy + 0.25 + clamp(p, 0, 0.5) * 0.3, 0, 1.2);
    this.sparkle = Math.min(6, this.sparkle + (surprise > 5 ? 4 : 2));
  }

  playHeadBurst(layer, heads, tokenCount) {
    if (!this.ctx) return;
    // Snap the burst to the arpeggiator's grid so it lands ON the beat,
    // and roll strictly ascending chord tones — a harp gliss inside the key,
    // not a cluster fighting the groove.
    const { root, intervals } = this.chord();
    const stepDur = 60 / this.bpm / 2;
    const start = Math.max(this.nextTime, this.ctx.currentTime + 0.04);
    this.bass(midiToHz(root - 12), start, stepDur * 3, 0.7); // soft downbeat thump
    const tones = [...heads]
      .sort((a, b) => b.w - a.w)
      .slice(0, 6)
      .filter(({ t, w }) => t < tokenCount && w >= 0.05);
    tones.forEach(({ t, w }, i) => {
      const interval = intervals[i % intervals.length] + 12 * Math.floor(i / intervals.length);
      this.piano(midiToHz(root + 12 + interval), 0.28 + clamp(w, 0, 1) * 0.4, start + i * (stepDur / 2), {
        pan: tokenCount > 1 ? clamp((t / (tokenCount - 1)) * 1.4 - 0.7 + this.panSpin, -1, 1) : 0,
        bright: 1.15,
      });
    });
    this.energy = clamp(this.energy + 0.25, 0, 1.2);
    this.sparkle = Math.min(6, this.sparkle + 2);
  }

  playUi(kind = 'click') {
    if (!this.ctx) return;
    const { root, intervals } = this.chord();
    if (kind === 'stop') this.piano(midiToHz(root - 12), 0.5, this.ctx.currentTime);
    else this.piano(midiToHz(root + 24 + intervals[0]), 0.6, this.ctx.currentTime, { bright: 1.3 });
  }

  // Called every frame: the camera is an instrument.
  //  - proximity (zoom): close = bright, loud, intimate; far = muffled, sparse,
  //    reverb-washed — like hearing the mind through its skull from outside.
  //  - azimuth (orbit): the stereo image rotates with you.
  //  - fast manual spins shake loose shimmer + sparkle notes.
  updateCamera({ proximity = 0.6, zoomVel = 0, azimuth = 0, rotVel = 0 } = {}) {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    this.proximity = clamp(proximity, 0, 1);
    this.panSpin = Math.sin(azimuth) * 0.55;

    // zoom whoosh (velocity), pitched up when pushing in
    const intensity = clamp(Math.abs(zoomVel) / 22, 0, 1);
    const direction = zoomVel < 0 ? 1 : 0;
    this.zoomGain.gain.setTargetAtTime(intensity * 0.075, now, intensity ? 0.025 : 0.09);
    this.zoomFilter.frequency.setTargetAtTime(300 + intensity * 1700 + direction * 900, now, 0.035);

    // proximity: open the mix up close, drown it in reverb far away.
    // Far end sits at 550Hz — genuinely underwater — and fully opens at 11kHz.
    const p = this.proximity;
    this.proximityFilter.frequency.setTargetAtTime(550 * Math.pow(11000 / 550, p), now, 0.12);
    this.fxBus.gain.setTargetAtTime(0.56 - p * 0.3, now, 0.15);
    this.musicBus.gain.setTargetAtTime(0.68 + p * 0.42, now, 0.15);

    // pacing: the mind thinks faster the closer you lean in (96 -> 122 BPM);
    // the echo stays locked to the tempo so the groove never smears
    this.bpm = 96 + p * 26;
    this.delay.delayTime.setTargetAtTime(60 / this.bpm / 2, now, 0.3);

    // pitch: zoom velocity Doppler-bends the pad — diving in pulls it sharp,
    // pulling away lets it sag flat, then it settles back to true
    const cents = clamp(-zoomVel * 1.6, -60, 60);
    for (const v of this.drone) v.osc.detune.setTargetAtTime(cents, now, 0.07);

    // rotation: ignore the slow auto-orbit; reward deliberate spins
    const excess = Math.max(0, Math.abs(rotVel) - 0.25);
    this.shimmerGain.gain.setTargetAtTime(clamp(excess / 3, 0, 1) * 0.05, now, excess ? 0.05 : 0.2);
    this.swirl += excess * 0.016;
    if (this.swirl > 0.6) {
      this.swirl = 0;
      this.sparkle = Math.min(6, this.sparkle + 2); // the orbit flings off glitter
    }
  }

  noiseHit(gainValue, duration, frequency) {
    if (!this.ctx || this.muted) return;
    const now = this.ctx.currentTime;
    const source = this.ctx.createBufferSource();
    const filter = this.ctx.createBiquadFilter();
    const gain = this.ctx.createGain();
    source.buffer = this.makeNoise(Math.max(0.1, duration));
    filter.type = 'bandpass';
    filter.frequency.value = frequency;
    filter.Q.value = 0.8;
    gain.gain.setValueAtTime(gainValue, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
    source.connect(filter).connect(gain).connect(this.musicBus);
    gain.connect(this.fxBus);
    source.start(now);
    source.stop(now + duration);
  }
}

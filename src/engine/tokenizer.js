// GPT-2 byte-level BPE tokenizer (used by GPT-Neo / TinyStories).

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const byteToChar = {};
  const charToByte = {};
  bs.forEach((b, i) => {
    byteToChar[b] = String.fromCharCode(cs[i]);
    charToByte[String.fromCharCode(cs[i])] = b;
  });
  return { byteToChar, charToByte };
}

const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class Tokenizer {
  constructor(vocab, mergesText) {
    this.encoder = vocab; // token string -> id
    this.decoder = {};
    for (const [k, v] of Object.entries(vocab)) this.decoder[v] = k;
    this.bpeRanks = new Map();
    const lines = mergesText.split('\n');
    let rank = 0;
    for (const line of lines) {
      if (!line || line.startsWith('#version')) continue;
      this.bpeRanks.set(line.trim(), rank++);
    }
    const { byteToChar, charToByte } = bytesToUnicode();
    this.byteToChar = byteToChar;
    this.charToByte = charToByte;
    this.cache = new Map();
  }

  bpe(token) {
    if (this.cache.has(token)) return this.cache.get(token);
    let word = token.split('');
    while (word.length > 1) {
      let minRank = Infinity;
      let minPair = null;
      for (let i = 0; i < word.length - 1; i++) {
        const pair = word[i] + ' ' + word[i + 1];
        const r = this.bpeRanks.get(pair);
        if (r !== undefined && r < minRank) {
          minRank = r;
          minPair = i;
        }
      }
      if (minPair === null) break;
      word = [
        ...word.slice(0, minPair),
        word[minPair] + word[minPair + 1],
        ...word.slice(minPair + 2),
      ];
    }
    this.cache.set(token, word);
    return word;
  }

  encode(text) {
    const ids = [];
    const utf8 = new TextEncoder();
    for (const match of text.matchAll(PAT)) {
      const mapped = Array.from(utf8.encode(match[0]), (b) => this.byteToChar[b]).join('');
      for (const piece of this.bpe(mapped)) {
        const id = this.encoder[piece];
        if (id !== undefined) ids.push(id);
      }
    }
    return ids;
  }

  decode(ids) {
    const text = ids.map((id) => this.decoder[id] ?? '').join('');
    const bytes = new Uint8Array(Array.from(text, (c) => this.charToByte[c] ?? 32));
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes);
  }

  // Decode a single token id to display text (for sprites/labels).
  decodeOne(id) {
    return this.decode([id]);
  }
}

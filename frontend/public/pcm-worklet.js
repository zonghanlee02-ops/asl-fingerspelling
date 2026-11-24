// AudioWorklet: microphone float samples -> 16 kHz mono 16-bit PCM chunks
// for the local Vosk speech recogniser. Runs in the browser's audio thread.
class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;   // sampleRate is a worklet global
    this.pos = 0;
    this.buf = new Int16Array(1600);   // 0.1 s per chunk
    this.n = 0;
    this.calls = 0;
    this.peak = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    // Resample to 16 kHz: average the input samples that fall into each
    // output sample (a simple low-pass filter that avoids harsh aliasing).
    while (this.pos < ch.length) {
      const a = Math.floor(this.pos);
      const b = Math.min(ch.length, Math.max(a + 1, Math.floor(this.pos + this.ratio)));
      let s = 0;
      for (let i = a; i < b; i++) s += ch[i];
      s /= b - a;
      this.buf[this.n++] = Math.max(-1, Math.min(1, s)) * 0x7fff;
      if (this.n === this.buf.length) {
        this.port.postMessage(this.buf.slice().buffer, []);
        this.n = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= ch.length;
    // level reading for the recording indicator, ~12 times per second
    for (let i = 0; i < ch.length; i++) this.peak = Math.max(this.peak, Math.abs(ch[i]));
    if (++this.calls % 30 === 0) {
      this.port.postMessage({ level: this.peak });
      this.peak = 0;
    }
    return true;
  }
}
registerProcessor("pcm-worklet", PcmWorklet);

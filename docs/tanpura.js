// Zero-Weight WebAudio Sympathetic Tanpura Drone Layer (docs/tanpura.js)
// Pure mathematical additive Jivari string synthesizer (Pa - Sa - Sa - LowSa) tuned to Sakhiya's chest tonic Sa (~215 Hz).
// Off by default for regular speech; optional subtle -22 dB sympathetic drone for Vedic / Mantra Chant mode.
(function (root) {
  class TanpuraDrone {
    constructor(tonicHz = 215.0, gainDb = -22.0) {
      this.tonicHz = tonicHz;
      this.gainDb = gainDb;
      this.ctx = null;
      this.timer = null;
      this.active = false;
    }

    // Render an offline sympathetic Pa-Sa-Sa-LowSa drone buffer at sample rate `sr` mixed at `gainDb` into `speechWav`
    mixIntoBuffer(speechWav, sr = 24000, tonicHz = this.tonicHz, gainDb = this.gainDb) {
      const n = speechWav.length;
      const out = new Float32Array(n);
      const linGain = Math.pow(10.0, gainDb / 20.0);
      // 4 traditional strings: Pa (3/4 below Sa = 0.75 * Sa), Sa (1.0), Sa (1.0), Low Sa (0.5 * Sa)
      const strFreqs = [tonicHz * 0.75, tonicHz, tonicHz * 1.0006, tonicHz * 0.5];
      const pluckPeriod = 1.35; // seconds between gentle string plucks
      for (let i = 0; i < n; i++) {
        const t = i / sr;
        let drone = 0.0;
        for (let s = 0; s < 4; s++) {
          const dt = (t - s * pluckPeriod + 100 * pluckPeriod) % (4 * pluckPeriod);
          // Smooth raised-cosine attack (80ms) + slow Jivari decay
          const attack = dt < 0.08 ? 0.5 * (1.0 - Math.cos(Math.PI * dt / 0.08)) : 1.0;
          const env = attack * Math.exp(-dt * 0.42);
          const f0 = strFreqs[s];
          // Jivari bridge shimmer: harmonics 1..6 with gentle phase modulation
          const jivari = 1.0 + 0.18 * Math.sin(2.0 * Math.PI * 0.45 * t + s);
          const sig =
            0.52 * Math.sin(2.0 * Math.PI * f0 * t) +
            0.26 * jivari * Math.sin(2.0 * Math.PI * 2.0 * f0 * t + 0.3) +
            0.14 * jivari * Math.sin(2.0 * Math.PI * 3.0 * f0 * t + 0.7) +
            0.06 * Math.sin(2.0 * Math.PI * 4.0 * f0 * t) +
            0.03 * Math.sin(2.0 * Math.PI * 5.0 * f0 * t);
          drone += env * sig * 0.28;
        }
        // Gentle fade-in and fade-out at clip boundaries
        let edge = 1.0;
        if (i < sr * 0.06) edge = i / (sr * 0.06);
        else if (n - i < sr * 0.08) edge = (n - i) / (sr * 0.08);
        out[i] = Math.max(-0.98, Math.min(0.98, speechWav[i] + linGain * edge * drone));
      }
      return out;
    }

    startLive(tonicHz = this.tonicHz, gainDb = this.gainDb) {
      if (typeof window === "undefined") return;
      this.stopLive();
      this.tonicHz = tonicHz;
      this.gainDb = gainDb;
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      this.ctx = new AudioCtx();
      this.active = true;
      const strFreqs = [tonicHz * 0.75, tonicHz, tonicHz * 1.0006, tonicHz * 0.5];
      let step = 0;
      const pluckOne = () => {
        if (!this.active || !this.ctx) return;
        const now = this.ctx.currentTime;
        const f0 = strFreqs[step % 4];
        step++;
        const master = this.ctx.createGain();
        const lin = Math.pow(10.0, this.gainDb / 20.0) * 0.22;
        master.gain.setValueAtTime(0.0001, now);
        master.gain.linearRampToValueAtTime(lin, now + 0.12);
        master.gain.exponentialRampToValueAtTime(0.0001, now + 4.8);
        master.connect(this.ctx.destination);
        [1, 2, 3, 4, 5].forEach((h, idx) => {
          const osc = this.ctx.createOscillator();
          const g = this.ctx.createGain();
          osc.type = "sine";
          osc.frequency.setValueAtTime(f0 * h, now);
          g.gain.setValueAtTime([0.52, 0.26, 0.14, 0.06, 0.03][idx], now);
          osc.connect(g);
          g.connect(master);
          osc.start(now);
          osc.stop(now + 4.9);
        });
      };
      pluckOne();
      this.timer = setInterval(pluckOne, 1350);
    }

    stopLive() {
      this.active = false;
      if (this.timer) { clearInterval(this.timer); this.timer = null; }
      if (this.ctx) { try { this.ctx.close(); } catch (_) {} this.ctx = null; }
    }
  }

  root.TanpuraDrone = TanpuraDrone;
  if (typeof module !== "undefined" && module.exports) module.exports = { TanpuraDrone };
})(typeof window !== "undefined" ? window : globalThis);

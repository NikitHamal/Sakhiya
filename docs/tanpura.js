// Toggleable Zero-Weight WebAudio Sympathetic Tanpura Drone (docs/tanpura.js)
// Tuned to Sakhiya's natural chest tonic Sa (~215.0 Hz), subtle -22 dB default mix in Vedic mode, off for speech.
(function (root) {
  const DEFAULT_TONIC_HZ = 215.0;
  const DEFAULT_DB_VEDIC = -22.0; // subtle -22 dB default mix in Vedic mode

  function dbToGain(db) {
    return Math.pow(10.0, Number(db) / 20.0);
  }

  // Synthesize sympathetic 4-string Tanpura cycle (Pa - Sa - Sa - low Sa) with subtle Jiva (Jawari) overtone shimmer
  function synthesizeTanpuraBuffer(durationSec, sr = 24000, tonicHz = DEFAULT_TONIC_HZ, mixDb = DEFAULT_DB_VEDIC) {
    const n = Math.max(1, Math.floor(Number(durationSec) * sr));
    const out = new Float32Array(n);
    const gain = dbToGain(mixDb);
    const sa = Number(tonicHz) || DEFAULT_TONIC_HZ;
    // 4 strings in Just Intonation: low Pa (3/4 Sa), upper Sa (1/1), upper Sa (1/1), low Sa (1/2 Sa)
    const strings = [
      { f0: sa * 0.75, offsetSec: 0.0, periodSec: 4.8 },
      { f0: sa * 1.00, offsetSec: 1.2, periodSec: 4.8 },
      { f0: sa * 1.0008, offsetSec: 2.4, periodSec: 4.8 }, // slight sympathetic detune
      { f0: sa * 0.50, offsetSec: 3.6, periodSec: 4.8 },
    ];
    const harmonics = [
      [1, 1.00, 0.9],
      [2, 0.62, 1.1],
      [3, 0.40, 1.4],
      [4, 0.25, 1.7],
      [5, 0.16, 2.0],
      [6, 0.10, 2.4],
    ];

    for (let i = 0; i < n; i++) {
      const t = i / sr;
      let sample = 0.0;
      for (const st of strings) {
        const rel = ((t - st.offsetSec) % st.periodSec + st.periodSec) % st.periodSec;
        const attack = Math.min(1.0, rel / 0.045);
        for (const [h, amp, decay] of harmonics) {
          const env = attack * Math.exp(-decay * (rel / st.periodSec));
          // Subtle jawari bridge modulation
          const jawari = 1.0 + 0.08 * Math.sin(2.0 * Math.PI * 3.2 * rel + h);
          sample += amp * env * jawari * Math.sin(2.0 * Math.PI * (st.f0 * h) * t);
        }
      }
      // Smooth fade-in and fade-out
      const fadeIn = Math.min(1.0, t / 0.18);
      const fadeOut = Math.min(1.0, (durationSec - t) / 0.35);
      out[i] = sample * 0.18 * gain * Math.max(0.0, fadeIn * fadeOut);
    }
    return out;
  }

  function mixTanpuraOnPcm(pcmFloat32, sr = 24000, opts = {}) {
    const enabled = opts.enabled !== undefined ? Boolean(opts.enabled) : (opts.mode === "vedic" || opts.style === "vedic_chant");
    if (!enabled || !pcmFloat32 || !pcmFloat32.length) return pcmFloat32;
    const tonicHz = opts.tonicHz || DEFAULT_TONIC_HZ;
    const mixDb = opts.mixDb !== undefined ? Number(opts.mixDb) : DEFAULT_DB_VEDIC;
    const drone = synthesizeTanpuraBuffer(pcmFloat32.length / sr, sr, tonicHz, mixDb);
    const mixed = new Float32Array(pcmFloat32.length);
    for (let i = 0; i < pcmFloat32.length; i++) {
      mixed[i] = Math.max(-0.98, Math.min(0.98, pcmFloat32[i] + drone[i]));
    }
    return mixed;
  }

  class TanpuraDroneController {
    constructor(audioCtx, opts = {}) {
      this.ctx = audioCtx || null;
      this.tonicHz = opts.tonicHz || DEFAULT_TONIC_HZ;
      this.mixDb = opts.mixDb !== undefined ? opts.mixDb : DEFAULT_DB_VEDIC;
      this.enabled = false;
      this.source = null;
      this.gainNode = null;
    }
    setMode(styleOrMode) {
      const shouldEnable = styleOrMode === "vedic" || styleOrMode === "vedic_chant";
      this.setEnabled(shouldEnable);
    }
    setEnabled(flag) {
      this.enabled = Boolean(flag);
      if (!this.enabled) this.stop();
    }
    setMixDb(db) {
      this.mixDb = Number(db);
      if (this.gainNode && this.ctx) {
        this.gainNode.gain.setTargetAtTime(dbToGain(this.mixDb), this.ctx.currentTime, 0.05);
      }
    }
    start(durationSec = 24.0) {
      if (!this.enabled || !this.ctx) return;
      this.stop();
      const sr = this.ctx.sampleRate || 24000;
      const pcm = synthesizeTanpuraBuffer(durationSec, sr, this.tonicHz, 0.0);
      const buf = this.ctx.createBuffer(1, pcm.length, sr);
      buf.getChannelData(0).set(pcm);
      this.source = this.ctx.createBufferSource();
      this.source.buffer = buf;
      this.source.loop = true;
      this.gainNode = this.ctx.createGain();
      this.gainNode.gain.value = dbToGain(this.mixDb);
      this.source.connect(this.gainNode);
      this.gainNode.connect(this.ctx.destination);
      this.source.start();
    }
    stop() {
      if (this.source) {
        try { this.source.stop(); } catch (_) {}
        try { this.source.disconnect(); } catch (_) {}
        this.source = null;
      }
    }
  }

  const exported = {
    DEFAULT_TONIC_HZ,
    DEFAULT_DB_VEDIC,
    dbToGain,
    synthesizeTanpuraBuffer,
    mixTanpuraOnPcm,
    TanpuraDroneController,
  };
  root.TANPURA = exported;
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
})(typeof window !== "undefined" ? window : globalThis);

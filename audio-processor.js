// AudioWorklet: turns microphone audio into 16 kHz mono frames of 512 samples
// (32 ms), the window size the voice-activity detector expects.

const TARGET_RATE = 16000;
const FRAME_SIZE = 512;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Float32Array(FRAME_SIZE);
    this.filled = 0;
    // `sampleRate` is the AudioContext rate. We ask for 16 kHz, but some browsers
    // refuse, so fall back to averaging input samples down to 16 kHz here.
    this.step = sampleRate / TARGET_RATE;
    this.sum = 0;
    this.count = 0;
    this.position = 0;
  }

  push(value) {
    this.frame[this.filled++] = value;
    if (this.filled === FRAME_SIZE) {
      const out = this.frame;
      this.frame = new Float32Array(FRAME_SIZE);
      this.filled = 0;
      this.port.postMessage(out, [out.buffer]);
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channels = input.length;
    const length = input[0].length;

    for (let i = 0; i < length; i++) {
      let sample = 0;
      for (let c = 0; c < channels; c++) sample += input[c][i];
      sample /= channels;

      if (this.step <= 1) {
        this.push(sample);
        continue;
      }
      this.sum += sample;
      this.count++;
      this.position++;
      if (this.position >= this.step) {
        this.position -= this.step;
        this.push(this.sum / this.count);
        this.sum = 0;
        this.count = 0;
      }
    }
    return true;
  }
}

registerProcessor('capture', CaptureProcessor);

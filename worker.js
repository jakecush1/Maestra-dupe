// Speech worker. Everything here runs on the user's device.
//
// The page starts up to two copies of this file:
//
//   "live" worker   microphone frames -> Silero VAD (where does speech start and stop?)
//                   -> Whisper Base re-transcribes the sentence in progress a couple of
//                      times a second and posts it as a `partial` (the grey preview text).
//                   -> when the speaker pauses, the finished audio is posted as a `segment`.
//
//   "final" worker  receives finished segments as `transcribe` jobs, runs a larger Whisper
//                   model over them, and posts the `final` text.
//
// Keeping the two apart means the preview never freezes while the big model is thinking.
// In "Fast" mode there is no second worker: the live worker handles the jobs itself.

import { pipeline, AutoModel, Tensor } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

const SAMPLE_RATE = 16000;
const WINDOW = 512; // samples per VAD window (32 ms)
const VAD_CONTEXT = 64; // Silero also wants the tail of the previous window

// Whisper checkpoints, converted to ONNX by the Hugging Face community. `f16` is used when
// the GPU supports half-precision shaders (smaller and faster), otherwise `f32`.
const SPECS = {
  base: {
    id: 'onnx-community/whisper-base',
    f16: { encoder_model: 'fp16', decoder_model_merged: 'fp16' },
    f32: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  },
  small: {
    id: 'onnx-community/whisper-small',
    f16: { encoder_model: 'fp16', decoder_model_merged: 'q4' },
    f32: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  },
  turbo: {
    id: 'onnx-community/whisper-large-v3-turbo',
    f16: { encoder_model: 'q4f16', decoder_model_merged: 'q4f16' },
    f32: { encoder_model: 'q4', decoder_model_merged: 'q4' },
  },
};
const CPU_DTYPE = 'q8'; // used when WebGPU is unavailable

// Voice-activity detection tuning.
const SPEECH_ON = 0.5; // probability that starts a segment
const SPEECH_OFF = 0.35; // probability below which a window counts as silence
const PRE_ROLL_WINDOWS = 10; // audio kept from just before speech starts (320 ms)
const TAIL_PAD = 0.25 * SAMPLE_RATE; // silence kept after the last word
const MIN_VOICED = 0.2 * SAMPLE_RATE; // shorter blips are ignored
const MAX_SEGMENT = 28 * SAMPLE_RATE; // Whisper sees at most 30 s at a time
const INTERIM_STEP = 0.5 * SAMPLE_RATE; // least new audio needed before the next partial

const post = (message, transfer) => self.postMessage(message, transfer ?? []);
const errorText = (error) => String(error?.message ?? error);

// ---------------------------------------------------------------- models

let vad = null;
let vadState = null;
let vadRate = null;
let vadInput = new Float32Array(VAD_CONTEXT + WINDOW);
let asr = null;
let asrSpec = null;
let device = null;
let encodeMs = 0;
let loadChain = Promise.resolve();

// All inference goes through one queue so the two models never run at once.
let inferenceChain = Promise.resolve();
function exclusive(fn) {
  const run = inferenceChain.then(fn);
  inferenceChain = run.catch(() => {});
  return run;
}

async function pickDevice() {
  try {
    const adapter = navigator.gpu && (await navigator.gpu.requestAdapter());
    if (adapter) return { name: 'webgpu', f16: adapter.features.has('shader-f16') };
  } catch {}
  return { name: 'wasm', f16: false };
}

function resetVad() {
  vadState = new Tensor('float32', new Float32Array(2 * 1 * 128), [2, 1, 128]);
  vadInput = new Float32Array(VAD_CONTEXT + WINDOW);
}

async function load({ spec, withVad }) {
  try {
    device ??= await pickDevice();

    if (withVad && !vad) {
      vad = await AutoModel.from_pretrained('onnx-community/silero-vad', {
        config: { model_type: 'custom' },
        dtype: 'fp32',
        device: 'wasm',
      });
      vadRate = new Tensor('int64', [SAMPLE_RATE], []);
      resetVad();
    }

    if (!asr || asrSpec !== spec) {
      const model = SPECS[spec];
      const dtype = device.name === 'webgpu' ? (device.f16 ? model.f16 : model.f32) : CPU_DTYPE;
      const loaded = await pipeline('automatic-speech-recognition', model.id, {
        device: device.name,
        dtype,
        progress_callback: (info) => {
          if (info.status === 'progress_total') post({ type: 'progress', loaded: info.loaded, total: info.total });
        },
      });

      // The first run compiles GPU shaders; do it now instead of on the first words.
      post({ type: 'warming' });
      const silence = new Float32Array(SAMPLE_RATE);
      const options = { language: 'en', task: 'transcribe', max_new_tokens: 1 };
      await exclusive(() => loaded(silence, options));
      const started = performance.now();
      await exclusive(() => loaded(silence, options));
      encodeMs = performance.now() - started; // how long one pass takes before any words

      asr = loaded;
      asrSpec = spec;
    }

    post({ type: 'ready', spec, device: device.name, encodeMs });
    pump();
  } catch (error) {
    post({ type: 'load-error', spec, message: errorText(error) });
  }
}

// ------------------------------------------------------------- segmenter

let language = 'en';
let epoch = 0; // bumped whenever the transcript is cleared, so stale results are dropped
let patient = false; // wait for longer pauses when the final model is slow
let closingPreview = false; // preview the finished segment while the final model works on it
let accepting = false;
let stopRequested = false;

const inbox = []; // audio windows and control messages, handled strictly in order
let draining = false;
let carry = new Float32Array(WINDOW);
let carryLength = 0;

const segment = new Float32Array(MAX_SEGMENT + WINDOW * (PRE_ROLL_WINDOWS + 2));
let segmentLength = 0;
let segmentStart = 0; // position of segment[0] on the sample clock
let segmentId = 0;
let segmentProbs = [];
let inSpeech = false;
let voiced = 0;
let voicedStart = 0; // where in the segment the first word begins (after the pre-roll)
let lastVoicedEnd = 0;
let silenceRun = 0;
let lastInterimLength = 0;
let preRoll = [];
let clock = 0; // samples heard since the transcript was last cleared

const jobs = []; // finished segments waiting for their final transcription
const closing = []; // finished segments waiting for one last preview
let pumping = false;
let previewSamples = 0; // how much audio goes by while one preview is computed

function enqueueAudio(samples) {
  if (carryLength === 0 && samples.length === WINDOW) {
    inbox.push(samples);
    return;
  }
  let offset = 0;
  while (offset < samples.length) {
    const count = Math.min(WINDOW - carryLength, samples.length - offset);
    carry.set(samples.subarray(offset, offset + count), carryLength);
    carryLength += count;
    offset += count;
    if (carryLength === WINDOW) {
      inbox.push(carry);
      carry = new Float32Array(WINDOW);
      carryLength = 0;
    }
  }
}

async function drain() {
  if (draining) return;
  draining = true;
  try {
    while (inbox.length) {
      const item = inbox.shift();
      if (item instanceof Float32Array) {
        if (accepting && vad) await handleWindow(item);
      } else {
        handleControl(item);
      }
    }
  } catch (error) {
    post({ type: 'error', message: errorText(error) });
  } finally {
    draining = false;
  }
  pump();
  checkFlushed();
}

function handleControl(message) {
  switch (message.type) {
    case 'start':
      epoch = message.epoch;
      language = message.language;
      clock = message.clock;
      patient = message.patient;
      closingPreview = message.closingPreview;
      accepting = true;
      stopRequested = false;
      carryLength = 0;
      preRoll = [];
      if (vad) resetVad();
      break;
    case 'stop':
      if (inSpeech) closeSegment();
      accepting = false;
      stopRequested = true;
      break;
    case 'reset':
      epoch = message.epoch;
      jobs.length = 0;
      closing.length = 0;
      if (inSpeech) {
        post({ type: 'drop', id: segmentId, epoch });
        segmentId++;
        inSpeech = false;
      }
      segmentLength = 0;
      segmentProbs = [];
      preRoll = [];
      clock = 0;
      break;
  }
}

async function speechProbability(window) {
  // Input = last 64 samples of the previous window + this window.
  const next = new Float32Array(VAD_CONTEXT + WINDOW);
  next.set(vadInput.subarray(WINDOW));
  next.set(window, VAD_CONTEXT);
  vadInput = next;
  const input = new Tensor('float32', next, [1, next.length]);
  const { stateN, output } = await vad({ input, sr: vadRate, state: vadState });
  vadState = stateN;
  return output.data[0];
}

// How long a silence (in seconds) ends the current segment. A real pause always does.
// Once a segment has run for a while we settle for less: first the short breath
// between sentences, eventually any gap at all, so that text keeps being finalised
// even when someone talks for minutes without stopping.
function pauseNeeded(seconds) {
  if (patient) {
    // The final model is slow on this device, so give it fewer, longer segments.
    return seconds > 26 ? 0.09 : seconds > 22 ? 0.16 : seconds > 16 ? 0.19 : seconds > 9 ? 0.6 : 1.4;
  }
  return seconds > 20 ? 0.09 : seconds > 14 ? 0.16 : seconds > 8 ? 0.19 : 0.6;
}

async function handleWindow(window) {
  const probability = await exclusive(() => speechProbability(window));
  clock += window.length;

  if (!inSpeech) {
    preRoll.push(window);
    if (preRoll.length > PRE_ROLL_WINDOWS) preRoll.shift();
    if (probability < SPEECH_ON) return;

    // Speech begins. Seed the segment with the moments just before it so the
    // first syllable is not clipped.
    inSpeech = true;
    segmentLength = 0;
    segmentProbs = [];
    for (const earlier of preRoll) {
      segment.set(earlier, segmentLength);
      segmentLength += earlier.length;
      segmentProbs.push(0);
    }
    segmentProbs[segmentProbs.length - 1] = probability;
    segmentStart = clock - segmentLength;
    preRoll = [];
    voiced = window.length;
    voicedStart = segmentLength - window.length;
    lastVoicedEnd = segmentLength;
    silenceRun = 0;
    lastInterimLength = 0;
    return;
  }

  segment.set(window, segmentLength);
  segmentLength += window.length;
  segmentProbs.push(probability);
  if (probability >= SPEECH_OFF) {
    voiced += window.length;
    lastVoicedEnd = segmentLength;
    silenceRun = 0;
  } else {
    silenceRun += window.length;
  }

  if (silenceRun >= pauseNeeded(segmentLength / SAMPLE_RATE) * SAMPLE_RATE) closeSegment();
  else if (segmentLength >= MAX_SEGMENT) splitSegment();
}

// Hand the first `end` samples of the current segment over for final transcription.
function emitSegment(end) {
  // Times describe the words themselves, not the padding around them.
  const info = {
    id: segmentId,
    epoch,
    language,
    start: (segmentStart + voicedStart) / SAMPLE_RATE,
    end: (segmentStart + Math.min(end, lastVoicedEnd)) / SAMPLE_RATE,
  };
  if (closingPreview && end - lastInterimLength > 0.3 * SAMPLE_RATE) {
    closing.push({ ...info, audio: segment.slice(0, end) });
  }
  const audio = segment.slice(0, end);
  post({ type: 'segment', ...info, audio }, [audio.buffer]);
}

function closeSegment() {
  if (voiced >= MIN_VOICED) emitSegment(Math.min(segmentLength, lastVoicedEnd + TAIL_PAD));
  else post({ type: 'drop', id: segmentId, epoch });
  segmentId++;
  inSpeech = false;
  segmentLength = 0;
  segmentProbs = [];
  pump();
}

// No pause at all for 28 s: cut at the quietest moment of the last 6 s and carry on.
function splitSegment() {
  const total = segmentProbs.length;
  const from = Math.max(1, total - Math.round((6 * SAMPLE_RATE) / WINDOW));
  let quietest = total - 1;
  for (let i = from; i < total; i++) {
    if (segmentProbs[i] < segmentProbs[quietest]) quietest = i;
  }
  const cut = (quietest + 1) * WINDOW;
  emitSegment(cut);
  segmentId++;
  segment.copyWithin(0, cut, segmentLength);
  segmentLength -= cut;
  segmentStart += cut;
  segmentProbs = segmentProbs.slice(quietest + 1);
  voiced = segmentLength;
  voicedStart = 0;
  lastVoicedEnd = segmentLength;
  silenceRun = 0;
  lastInterimLength = 0;
  pump();
}

// ----------------------------------------------------------- transcription

function tidy(text) {
  // Whisper marks non-speech as [BLANK_AUDIO], (music), ♪ and so on.
  const cleaned = (text || '')
    .replace(/\[[^\]]*\]|\([^)]*\)|[♪♫]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return /[\p{L}\p{N}]/u.test(cleaned) ? cleaned : '';
}

async function transcribe(audio, spokenLanguage) {
  const seconds = audio.length / SAMPLE_RATE;
  const output = await exclusive(() =>
    asr(audio, {
      language: spokenLanguage,
      task: 'transcribe',
      // Bounds the damage if the model ever gets stuck repeating itself.
      max_new_tokens: Math.min(400, Math.ceil(seconds * 10) + 32),
    }),
  );
  return tidy(output.text);
}

async function pump() {
  if (pumping || !asr) return;
  pumping = true;
  try {
    for (;;) {
      // 1. Final transcriptions come first.
      if (jobs.length) {
        const job = jobs[0];
        let text = '';
        let failed = false;
        try {
          text = await transcribe(job.audio, job.language);
        } catch (error) {
          failed = true; // the page keeps the preview text instead
          post({ type: 'error', message: errorText(error) });
        }
        const index = jobs.indexOf(job);
        if (index >= 0) jobs.splice(index, 1);
        post({ type: 'final', id: job.id, epoch: job.epoch, text, failed, start: job.start, end: job.end });
        continue;
      }

      // 2. One last preview of a segment that just ended.
      if (closing.length) {
        const item = closing.shift();
        const text = await transcribe(item.audio, item.language);
        if (text) post({ type: 'partial', id: item.id, epoch: item.epoch, text, start: item.start, end: item.end });
        continue;
      }

      // 3. Preview of the sentence still being spoken.
      // Leave the GPU some breathing room between previews: the final model shares it.
      const caughtUp = inbox.length < 4;
      const step = Math.max(INTERIM_STEP, previewSamples * 1.4);
      if (inSpeech && caughtUp && segmentLength - lastInterimLength >= step) {
        const id = segmentId;
        const snapshotEpoch = epoch;
        const start = (segmentStart + voicedStart) / SAMPLE_RATE;
        const end = (segmentStart + lastVoicedEnd) / SAMPLE_RATE;
        lastInterimLength = segmentLength;
        const began = performance.now();
        const text = await transcribe(segment.slice(0, segmentLength), language);
        previewSamples = ((performance.now() - began) / 1000) * SAMPLE_RATE;
        if (text) post({ type: 'partial', id, epoch: snapshotEpoch, text, start, end });
        continue;
      }
      break;
    }
  } catch (error) {
    post({ type: 'error', message: errorText(error) });
  } finally {
    pumping = false;
  }
  checkFlushed();
}

function checkFlushed() {
  if (stopRequested && !pumping && !draining && jobs.length === 0 && closing.length === 0) {
    stopRequested = false;
    post({ type: 'flushed' });
  }
}

// --------------------------------------------------------------- messages

self.onmessage = ({ data }) => {
  switch (data.type) {
    case 'load':
      if (data.idBase) segmentId = Math.max(segmentId, data.idBase);
      loadChain = loadChain.then(() => load(data));
      break;
    case 'audio':
      enqueueAudio(data.samples);
      drain();
      break;
    case 'transcribe':
      jobs.push(data);
      pump();
      break;
    case 'language':
      language = data.language;
      break;
    case 'start':
    case 'stop':
    case 'reset':
      inbox.push(data);
      drain();
      break;
  }
};

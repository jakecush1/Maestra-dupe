// Parola — page logic: microphone, transcript, and controls.
// The speech recognition itself lives in worker.js.

const $ = (id) => document.getElementById(id);

const SAMPLE_RATE = 16000;
const STORE_KEY = 'parola:v1';
const PREBUFFER_FRAMES = 320; // ~10 s of audio kept while the models are still loading
const PREBUFFER_MAX_WAIT = 12000; // ms; after a longer wait (first download) start from a clean slate
const PARAGRAPH_GAP = 2; // seconds of silence that start a new paragraph
const PARAGRAPH_CHARS = 600; // long paragraphs break at the next sentence that follows a pause
const PAUSE = 0.45; // seconds of silence that count as a real pause between pieces
const SLOW_MODEL_MS = 1500; // a final model slower than this gets longer segments

// Every on-device mode previews with Whisper Base. `final` is the model whose
// text is kept. Sizes are the one-time downloads, in MB.
const MODES = {
  fast: { final: 'base', name: 'Whisper Base' },
  balanced: { final: 'small', name: 'Whisper Small' },
  best: { final: 'turbo', name: 'Whisper Large v3 Turbo' },
  browser: {},
};
const MODEL_MB = { base: 140, small: 390, turbo: 540 };
const LOCALES = { en: 'en-US', es: 'es-ES', fr: 'fr-FR', it: 'it-IT' };

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let gpuAvailable = 'gpu' in navigator;

const el = {
  stage: $('stage'),
  notice: $('notice'),
  transcript: $('transcript'),
  language: $('language'),
  mode: $('engine'),
  mic: $('mic'),
  status: $('statusText'),
  timer: $('timer'),
  progress: $('progress'),
  progressBar: $('progressBar'),
  stamps: $('stamps'),
  copy: $('copy'),
  download: $('download'),
  downloadMenu: $('downloadMenu'),
  clear: $('clear'),
  privacy: $('privacy'),
  toast: $('toast'),
};

// ------------------------------------------------------------------ state

const state = {
  phase: 'idle', // idle | starting | listening | stopping
  language: 'en',
  mode: gpuAvailable ? 'balanced' : 'fast',
  showTimes: false,
  items: [], // transcript pieces in order: { id, text, start, end, final, para, el }
  clock: 0, // seconds of audio heard since the transcript was last cleared
  shift: 0, // added to times reported by the engine (set when restoring a transcript)
  epoch: 0, // bumped on clear so late results for the old transcript are ignored
};

let activeMode = null; // mode in use for the current session
let runId = 0; // identifies the current start attempt, so cancelled ones can bail out
let errorText = '';
let idleWaiters = [];

const elapsed = () => state.clock + state.shift;
const finals = () => state.items.filter((item) => item.final);

function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
    if (!saved) return;
    if (saved.language in LOCALES) state.language = saved.language;
    if (saved.mode in MODES) state.mode = saved.mode;
    state.showTimes = Boolean(saved.showTimes);
    if (Array.isArray(saved.segments)) {
      state.items = saved.segments
        .filter((s) => s && typeof s.text === 'string' && s.text)
        .map((s) => ({ id: null, text: s.text, start: +s.start || 0, end: +s.end || 0, para: Boolean(s.para), final: true }));
    }
    state.shift = Number(saved.elapsed) || 0;
  } catch {}
}

let saveTimer = 0;
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(
        STORE_KEY,
        JSON.stringify({
          language: state.language,
          mode: state.mode,
          showTimes: state.showTimes,
          segments: finals().map(({ text, start, end, para }) => ({ text, start, end, para })),
          elapsed: elapsed(),
        }),
      );
    } catch {}
  }, 300);
}

const downloadedKey = (spec) => `parola:downloaded:${spec}`;
function isDownloaded(spec) {
  try {
    return localStorage.getItem(downloadedKey(spec)) === '1';
  } catch {
    return false;
  }
}
const specsFor = (mode) => (mode === 'browser' ? [] : [...new Set(['base', MODES[mode].final])]);
const missingSpecs = (mode) => specsFor(mode).filter((spec) => !isDownloaded(spec));

// ------------------------------------------------------------- transcript

const pendingItems = new Map(); // id -> item still waiting for its final text
const doneIds = new Set(); // ids that are finished; late previews for them are ignored

const waitingEl = document.createElement('p');
waitingEl.className = 'waiting';
waitingEl.textContent = 'Start speaking. Your words will appear here.';
waitingEl.hidden = true;
el.transcript.before(waitingEl);

function clockLabel(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function startsParagraph(start) {
  const previous = state.items.at(-1);
  if (!previous) return true;
  const silence = start - previous.end;
  if (silence >= PARAGRAPH_GAP) return true;
  const paragraph = previous.el?.parentElement;
  const long = (paragraph?.textContent.length ?? 0) >= PARAGRAPH_CHARS;
  return long && silence >= PAUSE && /[.!?…]["')\]]?$/.test(previous.text);
}

function keepInView(change) {
  const { stage } = el;
  const pinned = stage.scrollHeight - stage.scrollTop - stage.clientHeight < 160;
  change();
  if (pinned) stage.scrollTop = stage.scrollHeight;
}

function paint(item) {
  item.el.textContent = item.text ? `${item.text} ` : '';
  item.el.classList.toggle('interim', !item.final);
}

// Add an item's element at the end of the transcript.
function attach(item) {
  let paragraph = el.transcript.lastElementChild;
  if (item.para || !paragraph) {
    paragraph = document.createElement('p');
    const time = document.createElement('span');
    time.className = 'time';
    time.textContent = clockLabel(item.start);
    paragraph.append(time);
    el.transcript.append(paragraph);
  }
  item.el = document.createElement('span');
  paint(item);
  paragraph.append(item.el);
}

function renderAll() {
  el.transcript.replaceChildren();
  for (const item of state.items) attach(item);
  el.stage.scrollTop = el.stage.scrollHeight;
  refresh();
}

function createItem(id, start, end) {
  const item = { id, text: '', start, end, final: false, para: startsParagraph(start) };
  state.items.push(item);
  pendingItems.set(id, item);
  attach(item);
  return item;
}

function removeItem(item) {
  const index = state.items.indexOf(item);
  if (index < 0) return;
  state.items.splice(index, 1);
  const next = state.items[index];
  if (item.para && next && !next.para) next.para = true; // next piece inherits the paragraph
  renderAll();
}

// Grey, still-changing text for a piece of speech.
function preview(id, text, start, end) {
  if (doneIds.has(id) || !text) return;
  keepInView(() => {
    const item = pendingItems.get(id) ?? createItem(id, start, end);
    item.text = text;
    item.end = Math.max(item.end, end);
    paint(item);
  });
  refresh();
}

// The finished text for a piece of speech. Empty text means "nothing was said".
function finalize(id, text, start, end) {
  if (doneIds.has(id)) return;
  doneIds.add(id);
  keepInView(() => {
    let item = pendingItems.get(id);
    pendingItems.delete(id);
    if (!text) {
      if (item) removeItem(item);
      return;
    }
    item ??= createItem(id, start, end);
    pendingItems.delete(id);
    Object.assign(item, { text, start, end, final: true });
    paint(item);
    item.el.classList.add('fresh');
    mendJoin(item);
  });
  refresh();
  save();
}

// Long speech is cut into pieces at brief pauses, and Whisper ends every piece with
// a full stop. If the next piece carries straight on in lower case, that full stop
// was really a comma.
function mendJoin(item) {
  const previous = state.items[state.items.indexOf(item) - 1];
  if (!previous?.final || item.para || item.start - previous.end >= PAUSE) return;
  if (/^\p{Ll}/u.test(item.text) && /[^.]\.$/.test(previous.text)) {
    previous.text = `${previous.text.slice(0, -1)},`;
    paint(previous);
  }
}

// Keep whatever preview text is still showing (used if a final never arrives).
function settlePending() {
  for (const [id, item] of [...pendingItems]) finalize(id, item.text, item.start, item.end);
}

function paragraphs() {
  const result = [];
  for (const item of finals()) {
    if (item.para || !result.length) result.push({ start: item.start, parts: [] });
    result.at(-1).parts.push(item.text);
  }
  return result;
}

function asText() {
  return paragraphs()
    .map((p) => (state.showTimes ? `[${clockLabel(p.start)}] ` : '') + p.parts.join(' '))
    .join('\n\n');
}

function srtTime(seconds) {
  const ms = Math.max(0, Math.round(seconds * 1000));
  const pad = (n, width = 2) => String(n).padStart(width, '0');
  return `${pad(Math.floor(ms / 3600000))}:${pad(Math.floor(ms / 60000) % 60)}:${pad(Math.floor(ms / 1000) % 60)},${pad(ms % 1000, 3)}`;
}

function asSrt() {
  const cues = [];
  for (const item of finals()) {
    // Split long stretches into caption-sized lines, sharing the time by length.
    const lines = [];
    for (const word of item.text.split(/\s+/)) {
      if (lines.length && lines.at(-1).length + word.length + 1 <= 84) lines[lines.length - 1] += ` ${word}`;
      else lines.push(word);
    }
    const duration = Math.max(item.end - item.start, 0.6);
    const total = lines.reduce((sum, line) => sum + line.length, 0) || 1;
    let at = item.start;
    for (const line of lines) {
      const span = (duration * line.length) / total;
      cues.push(`${cues.length + 1}\n${srtTime(at)} --> ${srtTime(at + span)}\n${line}\n`);
      at += span;
    }
  }
  return cues.join('\n');
}

// ------------------------------------------------------------------ UI

function setPhase(phase) {
  state.phase = phase;
  if (phase === 'idle') {
    activeMode = null;
    setLevel(0);
    for (const resolve of idleWaiters.splice(0)) resolve();
  }
  refresh();
}

function showError(message) {
  errorText = message;
  refresh();
}

function refresh() {
  const { phase } = state;
  const active = phase !== 'idle';
  const hasText = state.items.some((item) => item.text);
  const hasFinals = state.items.some((item) => item.final);
  const loading = loadingState();

  document.body.classList.toggle('is-active', active);
  document.body.classList.toggle('is-listening', phase === 'listening');
  document.body.classList.toggle('is-busy', phase === 'starting' || phase === 'stopping');
  document.body.classList.toggle('has-text', hasText);
  document.body.classList.toggle('no-meter', activeMode === 'browser');
  waitingEl.hidden = !(active && !hasText);

  let text;
  if (phase === 'listening') text = 'Listening…';
  else if (phase === 'stopping') text = 'Finishing up…';
  else if (phase === 'starting') text = loading?.text || 'Starting…';
  else if (errorText) text = errorText;
  else if (loading) text = loading.text;
  else text = hasText ? 'Click the microphone to continue' : 'Click the microphone to start';
  el.status.textContent = text;
  el.status.parentElement.classList.toggle('is-error', phase === 'idle' && Boolean(errorText));

  el.mic.setAttribute('aria-pressed', String(phase === 'listening'));
  el.mic.setAttribute('aria-label', phase === 'listening' ? 'Stop transcribing' : 'Start transcribing');

  el.timer.hidden = !(active || elapsed() > 0);
  el.timer.textContent = clockLabel(elapsed());

  const fraction = loading?.fraction ?? null;
  el.progress.hidden = fraction === null;
  if (fraction !== null) el.progressBar.style.width = `${fraction * 100}%`;

  el.copy.disabled = !hasFinals;
  el.download.disabled = !hasFinals;
  el.clear.disabled = state.items.length === 0;
  el.stamps.setAttribute('aria-pressed', String(state.showTimes));
  el.transcript.classList.toggle('show-times', state.showTimes);

  if (state.mode === 'browser') {
    el.privacy.textContent =
      'Instant mode uses your browser’s built-in speech service, which may send audio to its provider. Choose another accuracy level to keep audio on your device.';
  } else {
    const megabytes = missingSpecs(state.mode).reduce((sum, spec) => sum + MODEL_MB[spec], 0);
    const download = megabytes ? ` One-time ${megabytes} MB download.` : '';
    const tip = state.language !== 'en' && state.mode !== 'best' ? ' Tip: “Best” is noticeably more accurate outside English.' : '';
    el.privacy.textContent = `${MODES[state.mode].name} runs on your device. Your audio is never uploaded.${download}${tip}`;
  }
}

let level = 0;
function setLevel(target) {
  level = target > level ? target : level * 0.82 + target * 0.18;
  if (level < 0.01) level = 0;
  document.body.style.setProperty('--level', level.toFixed(3));
}

let toastTimer = 0;
function toast(message, action) {
  clearTimeout(toastTimer);
  el.toast.replaceChildren(message);
  if (action) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = action.label;
    button.onclick = () => {
      el.toast.hidden = true;
      action.run();
    };
    el.toast.append(button);
  }
  el.toast.hidden = false;
  toastTimer = setTimeout(() => (el.toast.hidden = true), action ? 7000 : 2600);
}

// ---------------------------------------------------------- Whisper engines

// Two workers: `live` hears the microphone and previews with Whisper Base;
// `finisher` re-transcribes finished sentences with the larger model.
const makeEngine = () => ({ worker: null, spec: null, status: 'none', loaded: 0, total: 0, warming: false, encodeMs: 0, waiters: [] });
const live = makeEngine();
const finisher = makeEngine();
let nextIdBase = 0;
let liveFlushed = true;
let lagWarned = false;
const awaitingFinal = new Set(); // segment ids sent for final transcription

const finalEngine = (mode) => (MODES[mode].final === 'base' ? live : finisher);
const enginesFor = (mode) => (mode === 'browser' ? [] : [...new Set([live, finalEngine(mode)])]);

function settleEngine(engine, error) {
  engine.status = error ? 'error' : 'ready';
  engine.warming = false;
  for (const waiter of engine.waiters.splice(0)) error ? waiter.reject(error) : waiter.resolve();
  refresh();
}

function releaseEngine(engine) {
  engine.worker?.terminate();
  for (const waiter of engine.waiters.splice(0)) waiter.reject(new Error('superseded'));
  Object.assign(engine, { worker: null, spec: null, status: 'none', loaded: 0, total: 0, warming: false });
}

function ensureEngine(engine, spec) {
  if (engine.spec === spec && engine.status === 'ready') return Promise.resolve();
  if (engine.spec !== spec || engine.status !== 'loading') {
    if (engine.spec !== spec) releaseEngine(engine); // a worker holds one model for its lifetime
    if (!engine.worker) {
      engine.worker = new Worker('worker.js', { type: 'module' });
      engine.worker.onmessage = (event) => onWorkerMessage(engine, event.data);
      engine.worker.onerror = (event) => {
        console.error('Speech worker failed:', event.message);
        const failed = engine.worker;
        settleEngine(engine, new Error('model'));
        failed?.terminate();
        if (engine.worker === failed) engine.worker = null;
      };
    }
    Object.assign(engine, { spec, status: 'loading', loaded: 0, total: 0, warming: false });
    nextIdBase += 1_000_000;
    engine.worker.postMessage({ type: 'load', spec, withVad: engine === live, idBase: nextIdBase });
    refresh();
  }
  return new Promise((resolve, reject) => engine.waiters.push({ resolve, reject }));
}

function ensureModels(mode) {
  const spec = MODES[mode].final;
  if (spec === 'base') {
    releaseEngine(finisher); // frees the larger model's memory
    return ensureEngine(live, 'base');
  }
  return Promise.all([ensureEngine(live, 'base'), ensureEngine(finisher, spec)]);
}

// What to tell the user while models download or start up (null when nothing is loading).
function loadingState() {
  const engines = enginesFor(state.mode).filter((engine) => engine.status === 'loading');
  if (!engines.length) return null;
  if (engines.every((engine) => engine.warming)) return { text: 'Almost ready…', fraction: null };
  const all = enginesFor(state.mode);
  const total = all.reduce((sum, engine) => sum + engine.total, 0);
  const loaded = all.reduce((sum, engine) => sum + engine.loaded, 0);
  const fraction = total ? loaded / total : 0;
  if (!missingSpecs(state.mode).length) return { text: 'Loading speech model…', fraction: total ? fraction : null };
  return { text: `Downloading speech model · ${Math.floor(fraction * 100)}% · one-time setup`, fraction };
}

function cpuOnly() {
  if (!gpuAvailable) return;
  gpuAvailable = false;
  applyCapabilities();
  refresh();
}

// `navigator.gpu` can exist and still have nothing to offer; ask before relying on it.
const gpuCheck = (async () => {
  try {
    if (!(await navigator.gpu?.requestAdapter())) cpuOnly();
  } catch {
    cpuOnly();
  }
})();

function onWorkerMessage(engine, data) {
  switch (data.type) {
    case 'progress':
      engine.loaded = data.loaded;
      engine.total = data.total;
      refresh();
      break;
    case 'warming':
      engine.warming = true;
      engine.loaded = engine.total;
      refresh();
      break;
    case 'ready':
      if (data.spec !== engine.spec) break;
      engine.encodeMs = data.encodeMs;
      try {
        localStorage.setItem(downloadedKey(data.spec), '1');
      } catch {}
      settleEngine(engine, null);
      if (data.device !== 'webgpu') cpuOnly();
      break;
    case 'load-error':
      if (data.spec !== engine.spec) break;
      console.error('Model failed to load:', data.message);
      settleEngine(engine, new Error('model'));
      break;

    case 'partial':
      if (data.epoch === state.epoch) preview(data.id, data.text, data.start + state.shift, data.end + state.shift);
      break;
    case 'segment': {
      // A sentence is complete: reserve its place, then get the final text.
      if (data.epoch !== state.epoch) break;
      if (!pendingItems.has(data.id)) keepInView(() => createItem(data.id, data.start + state.shift, data.end + state.shift));
      awaitingFinal.add(data.id);
      const target = finalEngine(activeMode ?? state.mode);
      target.worker?.postMessage({ ...data, type: 'transcribe' }, [data.audio.buffer]);
      if (awaitingFinal.size >= 3 && !lagWarned && state.phase === 'listening') {
        lagWarned = true;
        toast('Your device is falling behind. Try a faster accuracy setting.');
      }
      break;
    }
    case 'final':
      if (data.epoch !== state.epoch) break;
      awaitingFinal.delete(data.id);
      // If the accurate pass failed, keep the preview rather than lose the sentence.
      finalize(data.id, data.failed ? (pendingItems.get(data.id)?.text ?? '') : data.text, data.start + state.shift, data.end + state.shift);
      maybeFinishStop();
      break;
    case 'drop':
      if (data.epoch === state.epoch) finalize(data.id, '', 0, 0);
      break;
    case 'flushed':
      liveFlushed = true;
      maybeFinishStop();
      break;
    case 'error':
      console.error('Speech engine error:', data.message);
      break;
  }
}

// ------------------------------------------------------------- microphone

let mic = null; // { stream, context, node }
let prebuffer = [];

async function openMic() {
  if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('insecure'), { name: 'InsecureContext' });

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: true, autoGainControl: true },
  });

  let context;
  let source;
  try {
    context = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: 'interactive' });
    source = context.createMediaStreamSource(stream);
  } catch {
    // Some browsers cannot resample the microphone for us; the worklet will.
    await context?.close().catch(() => {});
    context = new AudioContext();
    source = context.createMediaStreamSource(stream);
  }

  const handle = { stream, context, node: null };
  try {
    await context.audioWorklet.addModule('audio-processor.js');
    const node = new AudioWorkletNode(context, 'capture', { numberOfInputs: 1, numberOfOutputs: 1 });
    const silent = new GainNode(context, { gain: 0 });
    source.connect(node).connect(silent).connect(context.destination);
    node.port.onmessage = (event) => onFrame(event.data);
    handle.node = node;
    if (context.state === 'suspended') await context.resume();
  } catch (error) {
    releaseMic(handle);
    throw error;
  }

  stream.getAudioTracks()[0]?.addEventListener('ended', () => {
    if (mic === handle && state.phase === 'listening') {
      stop();
      toast('The microphone was disconnected.');
    }
  });
  return handle;
}

function releaseMic(handle = mic) {
  if (!handle) return;
  if (handle.node) handle.node.port.onmessage = null;
  handle.stream.getTracks().forEach((track) => track.stop());
  handle.context.close().catch(() => {});
  if (handle === mic) mic = null;
}

function sendFrame(frame) {
  const before = Math.floor(elapsed());
  state.clock += frame.length / SAMPLE_RATE;
  live.worker.postMessage({ type: 'audio', samples: frame }, [frame.buffer]);
  if (Math.floor(elapsed()) !== before) el.timer.textContent = clockLabel(elapsed());
}

function onFrame(frame) {
  if (state.phase === 'listening') {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / frame.length);
    setLevel(Math.max(0, Math.min(1, (Math.sqrt(rms) - 0.07) * 2.4)));
    sendFrame(frame);
  } else if (state.phase === 'starting') {
    prebuffer.push(frame);
    if (prebuffer.length > PREBUFFER_FRAMES) prebuffer.shift();
  }
}

function describeError(error) {
  switch (error?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone access is blocked. Allow it in the address bar, then try again.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No microphone found. Connect one and try again.';
    case 'NotReadableError':
      return 'Your microphone is busy in another app. Close it and try again.';
    case 'InsecureContext':
      return 'The microphone needs a secure page. Open this app from http://localhost or https.';
  }
  if (error?.message === 'model') return 'Couldn’t load the speech model. Check your connection and try again.';
  return 'Something went wrong starting the microphone. Please try again.';
}

// ----------------------------------------------------- browser-built-in engine

let recognition = null;
let browserTick = 0;
let browserSession = 0;

function startBrowserEngine() {
  const rec = new Recognition();
  recognition = rec;
  rec.continuous = true;
  rec.interimResults = true;
  rec.maxAlternatives = 1;
  rec.lang = LOCALES[state.language];

  let session = `browser-${++browserSession}`;
  let fatal = '';
  const starts = new Map(); // result id -> time its first words were heard

  rec.onstart = () => {
    if (state.phase === 'starting') setPhase('listening');
    clearInterval(browserTick);
    let last = performance.now();
    browserTick = setInterval(() => {
      const now = performance.now();
      if (state.phase === 'listening') state.clock += (now - last) / 1000;
      last = now;
      el.timer.textContent = clockLabel(elapsed());
    }, 250);
  };

  rec.onresult = (event) => {
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      const id = `${session}:${i}`;
      const text = result[0].transcript.trim();
      if (!starts.has(id)) starts.set(id, elapsed());
      if (result.isFinal) finalize(id, text && text[0].toUpperCase() + text.slice(1), starts.get(id), elapsed());
      else preview(id, text, starts.get(id), elapsed());
    }
  };

  rec.onerror = (event) => {
    if (event.error !== 'no-speech' && event.error !== 'aborted') fatal = event.error;
  };

  rec.onend = () => {
    if (recognition !== rec) return;
    settlePending();
    // The browser ends recognition after a stretch of silence; keep it going.
    if (!fatal && state.phase === 'listening') {
      session = `browser-${++browserSession}`;
      starts.clear();
      rec.lang = LOCALES[state.language];
      try {
        rec.start();
        return;
      } catch {}
    }
    recognition = null;
    clearInterval(browserTick);
    setPhase('idle');
    if (fatal === 'not-allowed' || fatal === 'service-not-allowed') showError(describeError({ name: 'NotAllowedError' }));
    else if (fatal === 'audio-capture') showError(describeError({ name: 'NotFoundError' }));
    else if (fatal === 'network') showError('Instant mode needs an internet connection. Try another accuracy level.');
    else if (fatal) showError('Instant mode isn’t available in this browser. Try another accuracy level.');
  };

  rec.start();
}

// --------------------------------------------------------------- start/stop

async function start() {
  if (state.phase !== 'idle') return;
  errorText = '';
  const run = ++runId;
  const began = performance.now();
  prebuffer = [];
  setPhase('starting');

  await gpuCheck; // may switch the mode if this browser has no usable GPU
  if (run !== runId) return;
  const mode = state.mode;
  activeMode = mode;
  refresh();

  try {
    if (mode === 'browser') {
      startBrowserEngine();
      return;
    }

    const modelsReady = ensureModels(mode);
    modelsReady.catch(() => {}); // handled below; avoids a warning if the mic fails first

    const handle = await openMic();
    if (run !== runId) return releaseMic(handle);
    mic = handle;

    await modelsReady;
    if (run !== runId) return;

    const separateFinal = finalEngine(mode) !== live;
    liveFlushed = true;
    lagWarned = false;
    live.worker.postMessage({
      type: 'start',
      language: state.language,
      epoch: state.epoch,
      clock: Math.round(state.clock * SAMPLE_RATE),
      patient: separateFinal && finisher.encodeMs > SLOW_MODEL_MS,
      closingPreview: separateFinal,
    });
    setPhase('listening');
    // Words spoken while a cached model was loading are not lost.
    const early = prebuffer.splice(0);
    if (performance.now() - began < PREBUFFER_MAX_WAIT) early.forEach(sendFrame);
  } catch (error) {
    if (run !== runId) return;
    releaseMic();
    recognition = null;
    setPhase('idle');
    showError(describeError(error));
  }
}

let stopTimer = 0;

function finishStop() {
  clearTimeout(stopTimer);
  if (state.phase !== 'stopping') return;
  awaitingFinal.clear();
  settlePending();
  setPhase('idle');
}

function maybeFinishStop() {
  if (state.phase === 'stopping' && liveFlushed && awaitingFinal.size === 0) finishStop();
}

function stop() {
  if (state.phase === 'starting') {
    // Cancel: the models keep loading in the background for next time.
    runId++;
    releaseMic();
    prebuffer = [];
    if (recognition) {
      const rec = recognition;
      recognition = null;
      rec.abort();
    }
    setPhase('idle');
    return;
  }
  if (state.phase !== 'listening') return;

  setPhase('stopping');
  if (activeMode === 'browser') {
    recognition?.stop(); // its `onend` handler returns us to idle
    return;
  }
  releaseMic();
  liveFlushed = false;
  live.worker.postMessage({ type: 'stop' });
  stopTimer = setTimeout(finishStop, 45000); // never leave the button stuck
}

function whenIdle() {
  if (state.phase === 'idle') return Promise.resolve();
  return new Promise((resolve) => idleWaiters.push(resolve));
}

function toggle() {
  if (state.phase === 'idle') start();
  else if (state.phase !== 'stopping') stop();
}

function preload() {
  const mode = state.mode;
  if (location.protocol === 'file:' || mode === 'browser' || missingSpecs(mode).length) return;
  ensureModels(mode).catch(() => {});
}

// ----------------------------------------------------------------- controls

el.mic.addEventListener('click', toggle);

document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space' || event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
  if (document.activeElement && document.activeElement !== document.body) return;
  event.preventDefault();
  toggle();
});

el.language.addEventListener('change', () => {
  state.language = el.language.value;
  refresh();
  live.worker?.postMessage({ type: 'language', language: state.language });
  if (recognition && state.phase === 'listening') recognition.stop(); // restarts in the new language
  save();
});

el.mode.addEventListener('change', async () => {
  const wasActive = state.phase === 'listening' || state.phase === 'starting';
  if (wasActive) stop();
  await whenIdle();
  state.mode = el.mode.value;
  errorText = '';
  save();
  refresh();
  if (wasActive) start();
  else preload();
});

el.stamps.addEventListener('click', () => {
  state.showTimes = !state.showTimes;
  keepInView(refresh);
  save();
});

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Older route, for browsers that refuse the clipboard API.
    const area = document.createElement('textarea');
    area.value = text;
    area.style.cssText = 'position:fixed;opacity:0';
    document.body.append(area);
    area.select();
    const copied = document.execCommand('copy');
    area.remove();
    return copied;
  }
}

el.copy.addEventListener('click', async () => {
  const copied = await copyText(asText()).catch(() => false);
  toast(copied ? 'Copied to clipboard' : 'Couldn’t copy. Select the text and copy it manually.');
});

function closeMenu() {
  el.downloadMenu.hidden = true;
  el.download.setAttribute('aria-expanded', 'false');
}

el.download.addEventListener('click', (event) => {
  event.stopPropagation();
  const open = el.downloadMenu.hidden;
  el.downloadMenu.hidden = !open;
  el.download.setAttribute('aria-expanded', String(open));
});

el.downloadMenu.addEventListener('click', (event) => {
  const format = event.target.closest('button')?.dataset.format;
  if (!format) return;
  const body = format === 'srt' ? asSrt() : `${asText()}\n`;
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  const stamp = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([body], { type: 'text/plain;charset=utf-8' }));
  link.download = `transcript-${stamp}.${format}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  closeMenu();
});

document.addEventListener('click', closeMenu);
document.addEventListener('keydown', (event) => event.key === 'Escape' && closeMenu());

el.clear.addEventListener('click', () => {
  const backup = { items: finals(), elapsed: elapsed() };
  state.items = [];
  state.clock = 0;
  state.shift = 0;
  state.epoch++;
  pendingItems.clear();
  doneIds.clear();
  awaitingFinal.clear();
  for (const engine of [live, finisher]) engine.worker?.postMessage({ type: 'reset', epoch: state.epoch });
  renderAll();
  save();
  maybeFinishStop();
  if (!backup.items.length) return;

  toast('Transcript cleared', {
    label: 'Undo',
    run: () => {
      // Put the old transcript back in front of anything said since.
      for (const item of state.items) {
        item.start += backup.elapsed;
        item.end += backup.elapsed;
      }
      if (state.items.length) state.items[0].para = true;
      state.items = [...backup.items, ...state.items];
      state.shift += backup.elapsed;
      renderAll();
      save();
    },
  });
});

// --------------------------------------------------------------------- boot

// Hide the options this browser cannot run, and say why.
function applyCapabilities() {
  if (!Recognition) el.mode.querySelector('[value="browser"]')?.remove();
  if (!gpuAvailable) {
    el.mode.querySelector('[value="balanced"]')?.remove();
    el.mode.querySelector('[value="best"]')?.remove();
    if (state.mode === 'balanced' || state.mode === 'best') state.mode = 'fast';
  }
  if (state.mode === 'browser' && !Recognition) state.mode = 'fast';
  el.mode.value = state.mode;

  if (location.protocol === 'file:') {
    el.notice.innerHTML =
      'This page needs to be opened through a local web server. Double-click <code>start.command</code> in this folder, or run <code>python3 -m http.server 8000</code> and open <code>http://localhost:8000</code>.';
    el.notice.hidden = false;
  } else if (!gpuAvailable) {
    el.notice.textContent =
      'This browser can’t use your graphics card for speech recognition, so text will appear slowly. For the best experience use a current version of Chrome, Edge or Safari' +
      (Recognition ? ', or choose “Instant” below.' : '.');
    el.notice.hidden = false;
  }
}

restore();
applyCapabilities();
el.language.value = state.language;
renderAll();
gpuCheck.then(preload); // if these models were downloaded before, warm them up so the first click is instant

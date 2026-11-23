# Parola

Free, open-source, real-time speech-to-text in your browser.

**Try it now: <https://maestra-dupe.vercel.app>**

Press the microphone and talk. Your words appear as you speak, with punctuation and paragraphs. There is no account, no fee, and no server: speech recognition runs on your own computer, so your audio never leaves it.

- **Languages:** English, Spanish, French, Italian
- **Export:** copy to clipboard, save as `.txt`, or save as `.srt` subtitles
- **Private:** the transcript is kept only in your browser, and survives a page reload

## Use it online

Open <https://maestra-dupe.vercel.app> in Chrome and press the microphone. Nothing to install.

## Run it on a Mac

To run your own copy instead, double-click **`start.command`**. It opens Parola in Chrome at `http://localhost:8000`. Leave the Terminal window open while you use it.

The first time you press the microphone, the speech models download (once) and are cached by the browser. After that they load from the cache in a few seconds.

If macOS refuses to open the file, run `chmod +x start.command` in Terminal once, or start it by hand:

```sh
python3 -m http.server 8000
```

then open <http://localhost:8000>. Opening `index.html` directly from Finder does not work; browsers only allow microphones and background workers on `localhost` or `https`.

## Accuracy settings

| Setting | Final text from | One-time download | Best for |
| --- | --- | --- | --- |
| Fast | Whisper Base | 140 MB | Older or slower computers |
| Balanced (default) | Whisper Small | 530 MB | English on most laptops |
| Best | Whisper Large v3 Turbo | 680 MB | Highest accuracy, and Spanish, French or Italian |
| Instant (online) | Your browser's speech service | none | Trying it without a download |

Balanced and Best both show a quick grey preview while you talk, then replace it with the more accurate text when you pause. On an Apple M2 the preview updates about every 0.7 seconds; the final text follows 2 to 3 seconds after you pause on Balanced and 5 to 6 seconds on Best.

Instant uses the speech recognition built into Chrome, Edge and Safari. It needs no download, but the browser may send your audio to its provider (Google, in Chrome). The other three settings never upload anything.

## Browser support

Parola needs WebGPU to run the models on your graphics card. It is developed and tested in Chrome on Apple Silicon. Current versions of Edge and Safari support the same features but have had less testing. In a browser without WebGPU only Fast and Instant are offered, and Fast will be slow.

## How it works

```
microphone ──► audio-processor.js ──► worker.js (live) ──────────► grey preview text
               16 kHz frames          Silero VAD + Whisper Base
                                             │ finished sentence
                                             ▼
                                      worker.js (final) ─────────► final text
                                      Whisper Small / Large v3 Turbo
```

1. `audio-processor.js` (an AudioWorklet) converts the microphone to 16 kHz mono.
2. The **live** worker runs [Silero VAD](https://github.com/snakers4/silero-vad) to find where speech starts and stops, and re-transcribes the sentence in progress with Whisper Base for the preview.
3. When you pause, the finished audio goes to the **final** worker, which runs the larger Whisper model. Two workers are used so the preview never freezes while the large model is busy. On Fast there is only one worker.
4. `app.js` draws the transcript, saves it to `localStorage`, and handles the controls.

During long speech without pauses, text is finalised every 10 to 20 seconds at a breath between phrases.

Models run with [Transformers.js](https://github.com/huggingface/transformers.js) on WebGPU. The library is loaded from the jsDelivr CDN and the models from the Hugging Face Hub (`onnx-community/whisper-base`, `whisper-small`, `whisper-large-v3-turbo`, `silero-vad`).

There is no build step and there are no dependencies to install. The whole app is five files:

| File | What it does |
| --- | --- |
| `index.html`, `styles.css` | The page |
| `app.js` | Microphone, transcript, controls |
| `worker.js` | Voice detection and speech recognition |
| `audio-processor.js` | Microphone capture |

## Put it online for free

Parola is a static site, so any static host works. The live version at <https://maestra-dupe.vercel.app> is hosted on Vercel: import the GitHub repository with the framework preset set to "Other", no build command, and the project root as the output directory. GitHub Pages, Cloudflare Pages and Netlify work the same way. The host must serve it over `https`, which all of these do.

## Customising

- **Add a language:** add an `<option>` to the language menu in `index.html` and a matching entry in `LOCALES` in `app.js`. Whisper supports about 100 languages; use its two-letter code as the value.
- **Change the models:** edit `SPECS` in `worker.js` and `MODES` in `app.js`.
- **Rename it:** the name "Parola" is shown from `index.html`.

## Limits

- Accuracy depends on your microphone and how much background noise there is. A headset helps a lot.
- Whisper is given one language at a time. It does not follow a speaker who switches language mid-sentence.
- It does not label who is speaking.
- Phones can run it in principle, but the larger models may exceed their memory. Use Fast or Instant there.

## Credits and licences

Parola's own code is released under the [MIT licence](LICENSE).

- [Whisper](https://github.com/openai/whisper) by OpenAI, MIT
- [Silero VAD](https://github.com/snakers4/silero-vad), MIT
- [Transformers.js](https://github.com/huggingface/transformers.js) by Hugging Face, Apache 2.0

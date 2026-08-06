# Visual Socratic Tutor

Voice math tutor: a camera watches the problem + the student's handwritten work,
the server diagnoses where the student is stuck, and speaks the **minimum
necessary Socratic hint — never the answer** — through the laptop speaker.
Spec: [CLAUDE.md](CLAUDE.md).

```text
XIAO camera (ws /camera) ─┐
Device (laptop mic · browser) → WebSocket → server.py
  → Silero VAD turn detection (hands-free) → STT
  → Grok VLM recognition → Grok ConceptTagger (problem_type + concepts)
  → Domain KB (EXACT/TEMPLATE/CONCEPT/SEMANTIC/NEW)
  → Grok Solver fallback → Student State Estimator
  → Pedagogical Policy (L0–L4) → Hint Generator (+ answer-leak guard)
  → xAI TTS → laptop speaker (or streamed back to the browser)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,sim]"
cp .env.example .env   # if you don't have one; fill XAI_API_KEY
.venv/bin/python -m tutor.scripts.seed_db      # build data/knowledge.db (sympy-verified)
.venv/bin/python -m tutor.scripts.gen_assets   # render simulator worksheet JPEGs
```

`ffplay` (ffmpeg) is used for TTS playback: `sudo apt install ffmpeg`.

## Run

```bash
.venv/bin/python server.py
```

- **Echo mode** — with no `XAI_API_KEY` in `.env` the server runs fully offline:
  canned recognition/diagnosis, hints printed instead of spoken. The whole
  websocket pipeline works.
- **Live mode** — with `XAI_API_KEY` set: Grok VLM recognition + solver +
  estimator (via `CHAT_MODEL`), STT `POST /v1/stt`, TTS `POST /v1/tts`
  (voice `TTS_VOICE`, Korean). No STT/TTS model names are needed.

Device simulator (no hardware needed) in a second terminal:

```bash
.venv/bin/python -m simulator.device_sim --server ws://localhost:8765 --images simulator/assets/lin_001_wrong_sign.jpg simulator/assets/lin_001_step1_ok.jpg simulator/assets/lin_001_solved.jpg --wav simulator/assets/hint.wav
```

Keys: `h` = hint request, `n` = next image (the student "wrote more"),
`a` = send the WAV as a voice utterance (`--mic` records 3 s instead), `q` = quit.

Demo: press `h` (L1 Socratic question) → `h` again with unchanged work
(escalates to L2) → `n` then `h` (progress detected → back to L1).
Preflight: `.venv/bin/python -m tutor.scripts.live_demo`.

## Hands-free voice

No buttons, no keys: talk, stop talking, the tutor answers, it listens again.
Two front ends share one turn-taking implementation
([tutor/speech/turn.py](tutor/speech/turn.py)) — they differ only in *where*
the VAD runs.

```text
LISTENING --onset--> USER_SPEAKING --800ms silence--> PROCESSING
    ^                                                     |
    +---- tail guard <---- AGENT_SPEAKING <---- TTS starts
```

The VAD only runs in the first two states, so the tutor cannot transcribe its
own voice (barge-in is deliberately not implemented); `tail_guard_ms` covers
room decay after playback. Tuning (`.env`): `VAD_PREFIX_MS` 300,
`VAD_MIN_SPEECH_MS` 250, `VAD_SILENCE_MS` 800, `VAD_THRESHOLD` 0.5,
`VAD_TAIL_GUARD_MS` 250.

### A. Browser client — works over SSH (recommended)

The laptop only captures and plays; the VAD, STT, pipeline and TTS all stay on
the server, which is what makes this work on a headless SSH host.

```text
browser mic → AudioWorklet → 16 kHz mono int16 PCM → AUDIO frames
  → Silero VAD on the server → utterance → STT → RAG/hint pipeline
  → TTS bytes → TTS_AUDIO frame → browser speaker → playback_done
```

On the server:

```bash
.venv/bin/pip install -e ".[dev,vad]"
.venv/bin/python server.py
```

On the Windows laptop — forward the port, then open the page:

```bash
ssh -N -L 8765:localhost:8765 user@ssh-server
```

Open <http://localhost:8765/> in Chrome/Edge, press 시작 once (browsers require
a gesture for mic access), and talk. **Use the tunnel, not the server's IP**:
`getUserMedia` only works in a secure context, and `localhost` counts as one
without any TLS setup. HTTP and WebSocket share the port, so one tunnel is
enough. Optionally attach a worksheet photo with the file picker — it is sent
on `capture_request`; without one the tutor asks to see the problem.

### B. Local laptop mic device (no browser)

```bash
.venv/bin/pip install -e ".[voice]"
.venv/bin/python -m simulator.voice_device --server ws://localhost:8765 --images simulator/assets/lin_001_wrong_sign.jpg
```

Here the mic is just another device on the existing wire protocol: VAD runs
client-side and TTS plays on the machine running the server, so both must be
the same room (the XIAO setup). `--list-devices` lists microphones,
`--input-device` selects one, `--images` replays a worksheet.

## The worksheet photo

By default the picture comes from the browser page: choose a file, drag one in,
or paste a screenshot with `Ctrl+V`. A thumbnail shows what the tutor is
actually looking at, which is the fastest way to catch the usual problem — a
photo that turns out blurred, cropped short, or of the wrong page.

Cropping happens server-side before the model sees the frame, because a vision
model resizes whatever it is given into a fixed budget and a photo of a desk
spends most of that budget on the desk:

```bash
WORKSHEET_ROI=0.18,0.36,0.36,0.55   # exact region, for a fixed camera mount
AUTO_CROP=0                          # or turn the automatic one off
```

The automatic crop refuses rather than guesses: anything that is not
page-shaped, page-sized and brighter than the rest of the frame is passed
through whole. See [tutor/vision/framing.py](tutor/vision/framing.py).

### XIAO camera (opt-in)

```bash
INPUT_MODE=camera python server.py
```

The board is the eyes only — mic and speaker stay on the laptop, where the
sound comes out. It connects to `ws://<laptop>:8765/camera` and answers each
`capture_request` with one JPEG. Flashing, wiring and network notes:
[firmware/README.md](firmware/README.md).

It is off by default for a measured reason: on a desk mount the 2 MP
fixed-focus sensor put an A4 page across ~540 px, about 65 DPI, with
handwriting 25-30 px tall. Handwriting recognition wants 150+ DPI, and no
model reads pixels that were never captured. Upload is what works today.

Running the server on the laptop (the demo setup — no tunnel, sound just works):

```bash
python -m tutor.scripts.live_demo
```

prints what is missing plus the sketch's `SERVER_HOST` / `SERVER_PORT` with
this machine's LAN IP filled in, and the firewall rule the board needs. On
Windows use `.venv\Scripts\python` in place of `.venv/bin/python`, and skip the
`[rag]` extra unless you want semantic retrieval — the tutor runs without it
(that extra pulls torch).

Try the pairing without hardware:

```bash
.venv/bin/python -m simulator.camera_device --server ws://localhost:8765 --images simulator/assets/lin_001_wrong_sign.jpg
```

If the tutor keeps saying it cannot see the worksheet, stop the server and run

```bash
.venv/bin/python -m tutor.scripts.camera_check
```

which asks the board for one photo, saves it, and sends that photo to the VLM —
so you can tell "no frame arrived" from "the frame was unreadable" instead of
guessing. See [firmware/README.md](firmware/README.md#6-카메라에-다시-보여-줄래요-and-nothing-else).

### Reading the worksheet with Gemini

Vision is the one job that can move to another model without touching the
pedagogy — the solver, the diagnosis, the hint ladder and the leak guard all
stay on Grok:

```bash
VISION_PROVIDER=gemini python server.py
```

Needs `GOOGLE_API_KEY` in `.env` and `pip install -e ".[vision-gemini]"`. The
model is `GEMINI_VISION_MODEL` (default `gemini-3.6-flash`). A missing key or
package logs an error and falls back to Grok rather than refusing to start.

## Tests

```bash
.venv/bin/python -m pytest
```

Fully offline: LLM calls are mocked (`EchoLLMClient`), audio is a `NullSpeaker`,
the VAD is a scripted stub. Covers the wire protocol, sympy matching
(EXACT/TEMPLATE/CONCEPT/NEW), the policy rule table, hint-effectiveness
lifecycle, per-purpose tool allowlists, the answer-leak guard, turn taking
(prefix padding, onset debounce, endpointing, the four-state gate), and
end-to-end websocket smoke tests including full hands-free voice turns on both
the local-mic device and the browser client.

## Design highlights

- **Two whitelisted tag layers**: recognition says what is written, a separate
  [ConceptTagger](tutor/knowledge/tagger.py) says what it is —
  `problem_type` (exactly one, from [taxonomy.py](tutor/knowledge/taxonomy.py))
  plus 0–4 `concepts` (from [seeds/concepts.json](tutor/knowledge/seeds/concepts.json)).
  Python re-enforces both whitelists, so an invented id never reaches the KB.
  Solution *strategies* are deliberately not tags. Tagging is one call per
  problem, cached for every later hint on it.
- **Tool-calling, read-only**: the LLM's only tool is kind-scoped
  `search_domain_kb`; per-purpose allowlists (`tutor/tools/registry.py`) keep
  `phrase` away from solutions/answers entirely. Session state and hint history
  are prefetched by the orchestrator into prompts; **all** store writes happen
  in the orchestrator (`tutor/server/session.py`).
- **Hint lifecycle**: hints are stored with `effective=null`, resolved after the
  next estimate by `hint_was_effective` = step progress OR misconception
  resolved OR status improved. Escalation is exactly +1 and only after an
  ineffective hint; progress fades back to L1.
- **Verified knowledge first**: seeds are sympy-verified before insertion;
  template instantiations are recomputed and re-verified; Grok solutions are
  machine-checked but stored unverified (spec rule: never auto-verified).
- **Leak guard** (`tutor/hints/guard.py`): typed answers
  (`SCALAR`/`ROOT_SET`/`EXPRESSION`) checked numerically *and* symbolically —
  a rewritten derivative like `2x + 3x²` for `3x² + 2x` is still caught.
- **`problem_hash`** covers problem text + equations + choices + diagram
  conditions and excludes student work, so a growing worksheet keeps the
  cached problem context.

## Later

- XIAO ESP32S3 Sense firmware (the simulator speaks the exact wire protocol
  the firmware will use: JPEG on `capture_request`, 16 kHz/16-bit mono PCM
  push-to-talk, JSON events).
- Mathpix fallback if Grok recognition proves insufficient.
- Proactive stuck detection ("힌트 필요해요?") and richer fading.

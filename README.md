# Visual Socratic Tutor

Voice math tutor: a camera watches the problem + the student's handwritten work,
the server diagnoses where the student is stuck, and speaks the **minimum
necessary Socratic hint — never the answer** — through the laptop speaker.
Spec: [CLAUDE.md](CLAUDE.md).

```text
Phone camera (wss /camera) ───────────────┐
Device (laptop mic · browser) → WebSocket → server.py
  → Silero VAD turn detection (hands-free) → STT
  → utterance intent (hint request / work check / answer / stay quiet)
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
    ^                    ^                                |
    |                    +--- barge-in ---+               |
    +---- tail guard <---- AGENT_SPEAKING <---- TTS starts
```

`tail_guard_ms` covers room decay after playback. Tuning (`.env`):
`VAD_PREFIX_MS` 300, `VAD_MIN_SPEECH_MS` 250, `VAD_SILENCE_MS` 800,
`VAD_THRESHOLD` 0.5, `VAD_TAIL_GUARD_MS` 250.

### Barge-in

Talk over the tutor and it stops. The audio being played is cut, anything
queued behind it is dropped, the rest of that turn goes unsaid, and the words
you interrupted with become the next question — they are already in the VAD's
prefix buffer when the interruption is noticed, so nothing of "왜 그렇게 해요?"
is lost.

What makes it survivable is echo cancellation: `getUserMedia` subtracts what
the browser is playing from what it hears, so an open mic during playback does
not feed the tutor its own voice. What is left of it is handled by a higher
onset bar — taking the floor needs `BARGE_IN_FRAMES` (8, ~250 ms) of sustained
speech against the 3 frames it takes to start a turn in silence, so a cough or
a leaked syllable does not interrupt.

```bash
BARGE_IN=0            # if the tutor interrupts itself (external speaker, bad AEC)
BARGE_IN_FRAMES=12    # or just make it harder
```

Two places it deliberately does not apply. **PROCESSING** stays deaf: there is
nothing to interrupt while the tutor thinks, and a half-turn must not survive
into the answer. And the **local-mic device** ([simulator/voice_device.py](simulator/voice_device.py))
forces it off whatever the setting says — its speaker feeds straight into its
own microphone with no browser in between, so an open mic there would have the
tutor interrupt itself, every time.

A hint that was half-spoken is not rolled back: it stays in the history, so the
policy does not offer it again as though it had never been heard.

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
the same room. `--list-devices` lists microphones,
`--input-device` selects one, `--images` replays a worksheet.

## What the student says, and what the tutor does about it

The same sentence means different things depending on what is already on the
table, so every utterance is classified first
([tutor/speech/intent.py](tutor/speech/intent.py)) and only then routed:

| 발화 | 의도 | 카메라 |
|---|---|---|
| "이 문제 힌트 줄래?" · "도와줘" · "모르겠어" | `HINT_REQUEST` | 새로 촬영 → VLM |
| "풀이 맞아?" · "풀이 봐줘" · "내가 쓴 거 봐줘" | `WORK_CHECK` | 새로 촬영 → VLM |
| "5예요" · "네, 맞아요" · "그렇게 하면 돼요" | `ANSWER` | 촬영 안 함 |
| 혼잣말·잡담 | `NONE` | 무응답 |

A work check has to **name the written work** — 풀이, 내가 쓴 거, 이렇게 하는 거,
어디가 틀렸 — or ask for it to be looked at (봐 줘). A bare "맞아?" is
deliberately not enough: "네, 맞아요" is how a student *agrees* with the tutor,
and treating that as a work check stopped the lesson to photograph the page
every time they did.

A hint request and a work check otherwise run the **same** turn — checking work
*is* re-reading the worksheet and re-diagnosing. The one difference is what
comes out of it: if the work is **correct**, the tutor says
"맞아요! 이대로 하면 돼요." and stops — the student asked a yes/no question, and
a hint would push them at a step they have not reached and imply something was
wrong. No hint record either, so the L1–L4 ladder does not move. If the work is
**wrong**, the unchanged hint generator runs, behind a short "지금 쓴 줄을 같이
볼까요?".

An answer is graded from the transcript alone: no capture, no VLM, and no
classifier call either — that is what makes it fast enough to feel like a
conversation. `AnswerEvaluator` can still redirect one to a work check if the
keywords misread it.

Two things the rules deliberately keep as they were: "모르겠어요" or "힌트 더
주세요" against a pending question is still an *answer* (the evaluator reads it
as "escalate"), and "이렇게 하는 거 맞아?" before any photo exists is promoted to
a hint request, because the camera has to see the page first either way.
Ambiguous phrasings the keyword rules miss go to one small no-tools LLM call;
`AnswerEvaluator` can also redirect a mis-routed answer to a work check.

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

### Phone camera (opt-in)

```bash
INPUT_MODE=camera python server.py
```

The phone is the eyes only — mic and speaker stay on the laptop, where the sound
comes out. It connects to `wss://<laptop>:8766/camera` and answers each
`capture_request` with one JPEG; the voice session borrows it whenever it has no
camera of its own. It says hello and waits, and never starts a turn, so the
pedagogy needs to know nothing about the camera at all.

Live capture was shelved once for a measured reason, and it was a sensor
problem rather than a plumbing one: on a desk mount the ESP32 board's 2 MP
fixed-focus camera put an A4 page across ~540 px, about 65 DPI, with
handwriting 25-30 px tall. Handwriting recognition wants 150+ DPI, and no model
reads pixels that were never captured. A phone asks for a 2560 px-wide frame
and autofocuses, which is several times that — enough to be worth trying, but
upload stays the default until it has been measured the same way.

Before a demo:

```bash
python -m tutor.scripts.live_demo
```

prints what is missing, this machine's LAN IP, and the firewall rule the phone
needs. On Windows use `.venv\Scripts\python` in place of `.venv/bin/python`, and
skip the `[rag]` extra unless you want semantic retrieval — the tutor runs
without it (that extra pulls torch).

Try the pairing without a phone:

```bash
.venv/bin/python -m simulator.camera_device --server ws://localhost:8765 --images simulator/assets/lin_001_wrong_sign.jpg
```

If the tutor keeps saying it cannot see the worksheet, stop the server and run

```bash
.venv/bin/python -m tutor.scripts.camera_check
```

which asks the camera for one photo, saves it, and sends that photo to the VLM —
so you can tell "no frame arrived" from "the frame was unreadable" instead of
guessing.

`getUserMedia` only exists in a **secure context**, and `http://<lan-ip>:8765` is
not one — `navigator.mediaDevices` is undefined there, so the page can never work
over the plain port. Mint a certificate for this machine's LAN address:

```bash
.venv/bin/python -m tutor.scripts.make_cert
```

Add the two `TLS_CERT` / `TLS_KEY` lines it prints to `.env` and restart. The
plain port is untouched — `localhost` is already a secure context — so this
only *adds* a listener on `TLS_PORT` (default `WS_PORT + 1`):

1. laptop: <http://localhost:8765/> → press 시작 (mic + speaker)
2. phone, same Wi-Fi: `https://<lan-ip>:8766/phone` → accept the certificate
   warning once (self-signed) → allow the camera → press 시작
3. point it at the worksheet and talk **to the laptop**

`certs/` needs `cryptography` (`pip install -e ".[phone]"`) or `openssl` on
PATH. The cert carries the LAN IP in `subjectAltName` — without an `IP:` entry
browsers reject an IP address URL no matter how many warnings you click through.
Allow the TLS port through the firewall too (the rule
`python -m tutor.scripts.live_demo` prints, with the port changed).

### Blurry photos

**The lens is the focus fix.** A four-camera phone exposes all of them and only
some can focus on paper held close; `facingMode: environment` gets you whichever
the browser calls the default rear camera, which on real hardware could not. The
page therefore picks the **lowest-numbered rear camera by label** —
`camera2 0, facing back`, the main lens on Android. Not `videoinput[0]`: that
index is the *selfie* camera on this phone. There is no picker; this is not a
decision to hand a student mid-lesson.

On top of that, continuous autofocus is requested explicitly (advanced
constraints are dropped inside `getUserMedia`, so it has to be
`applyConstraints` on the live track), and **tapping the preview focuses there**
— continuous AF often locks onto the desk rather than the page.

Framing: only a **width** is requested (2560) plus a *soft* `aspectRatio` of
16:9. Naming a height would pin the ratio and make the browser crop the sensor
to reach it, which cut the sides off the page; `ideal` lets the camera use a
native 16:9 mode if it has one and ignore the request if it does not. The
preview is `object-fit: contain`, so what the student frames is what gets sent.

The line under the button reports what the phone actually gave us — lens label,
stream size, and focus mode — so "초점이 안 맞는다" does not have to be a guess.

If a photo is still blurry: hold ~20 cm away (most phones cannot focus nearer),
keep still for a moment, and turn a light on — a dim room means a long exposure,
and the resulting motion blur looks exactly like a focus failure. To see what
the tutor actually received, set `SAVE_CAPTURES_DIR=data/captures` in `.env`.

### Reading the worksheet with Gemini

Vision is the one job that can move to another model without touching the
pedagogy — the solver, the diagnosis, the hint ladder and the leak guard all
stay on Grok:

```bash
VISION_PROVIDER=gemini python server.py
```

Hints can move too — Gemini carries Google's LearnLM tuning, which is about
following pedagogical system instructions rather than styling them:

```bash
HINT_PROVIDER=gemini python server.py     # GEMINI_HINT_MODEL, default gemini-3.6-flash
```

`gemini-3.1-pro-preview` phrases hints better — it reached for an everyday
analogy where flash restated the rule — but at 8-12s per hint against flash's
4-6s it takes a turn from 11s to 18s, so flash is the default.

The policy that picks the hint level and the leak guard that checks the result
are deterministic and do not move with the model, so this changes the wording
and nothing about how much is given away.

**Two doors, and the billing differs.** An AI Studio key spends prepaid credits
and answers `429 ... prepayment credits are depleted` when they run out; a
Cloud project spends its own:

```bash
VERTEX_PROJECT=gen-lang-client-0586206831   # + gcloud auth application-default login
VERTEX_LOCATION=global                       # not us-central1: 3.1-pro-preview 404s there
```

Set `VERTEX_PROJECT` and it wins over `GOOGLE_API_KEY`. Either way, if the
model is unreachable mid-lesson the turn falls back to the chat model for 60
seconds rather than being lost ([tutor/llm/fallback.py](tutor/llm/fallback.py)).

Needs `GOOGLE_API_KEY` in `.env` and `pip install -e ".[gemini]"`. The
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
(prefix padding, onset debounce, endpointing, the four-state gate), utterance
intent routing (including that an answer never costs a classifier call and a
work check always costs a capture), and end-to-end websocket smoke tests
including full hands-free voice turns on both the local-mic device and the
browser client.

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

- Mathpix fallback if Grok recognition proves insufficient.
- Proactive stuck detection ("힌트 필요해요?") and richer fading.

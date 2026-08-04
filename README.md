# Visual Socratic Tutor

Voice math tutor: a camera watches the problem + the student's handwritten work,
the server diagnoses where the student is stuck, and speaks the **minimum
necessary Socratic hint — never the answer** — through the laptop speaker.
Spec: [CLAUDE.md](CLAUDE.md).

```text
Device (camera/mic, or the laptop's own mic) → WebSocket → server.py
  → Silero VAD turn detection (hands-free) → STT
  → Grok VLM recognition → Domain KB (EXACT/TEMPLATE/CONCEPT/NEW)
  → Grok Solver fallback → Student State Estimator
  → Pedagogical Policy (L0–L4) → Hint Generator (+ answer-leak guard)
  → xAI TTS → laptop speaker
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

## Hands-free voice (laptop mic + speaker)

No buttons, no keys: talk, stop talking, the tutor answers, it listens again.

```bash
.venv/bin/pip install -e ".[voice]"
.venv/bin/python -m simulator.voice_device --server ws://localhost:8765 --images simulator/assets/lin_001_wrong_sign.jpg
```

The laptop mic is just another device on the existing wire protocol: 16 kHz mono
PCM → Silero VAD turn detection → the same `AUDIO` frames the XIAO sends → the
unchanged server pipeline (STT → hint → TTS on the laptop speaker). Only the
endpointed utterance is sent, so STT and Grok never see room noise. `--images`
is optional (it replays a worksheet on `capture_request`); `--list-devices`
picks a microphone, `--input-device` selects one.

```text
LISTENING --onset--> USER_SPEAKING --800ms silence--> PROCESSING
    ^                                                     |
    +---- tail guard <---- AGENT_SPEAKING <---- TTS starts (speech_state)
```

The VAD only runs in the first two states, so the tutor cannot transcribe its
own voice; `tail_guard_ms` covers room decay after playback. Tuning (`.env`):
`VAD_PREFIX_MS` 300, `VAD_MIN_SPEECH_MS` 250, `VAD_SILENCE_MS` 800,
`VAD_THRESHOLD` 0.5, `VAD_TAIL_GUARD_MS` 250 — the endpointing lives in
[tutor/speech/turn.py](tutor/speech/turn.py) and is unit-tested with a scripted
VAD (no mic, no torch).

## Tests

```bash
.venv/bin/python -m pytest
```

Fully offline: LLM calls are mocked (`EchoLLMClient`), audio is a `NullSpeaker`,
the VAD is a scripted stub. Covers the wire protocol, sympy matching
(EXACT/TEMPLATE/CONCEPT/NEW), the policy rule table, hint-effectiveness
lifecycle, per-purpose tool allowlists, the answer-leak guard, turn taking
(prefix padding, onset debounce, endpointing, the four-state gate), and
end-to-end websocket smoke tests including a full hands-free voice turn.

## Design highlights

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

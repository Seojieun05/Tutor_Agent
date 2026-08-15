# Visual Socratic Tutor

## Goal

Voice tutor: a phone camera watches the page, the laptop listens and speaks. Observe math problem + student work, detect where the student is stuck, and give the **minimum necessary Socratic hint instead of the answer**.

## Architecture

```text
Phone Camera (browser page)
  → Wi-Fi/WebSocket
  → Laptop Server
    → STT
    → Grok VLM: problem + student-work recognition
    → Domain Knowledge DB
    → Grok Solver fallback
    → Student State Estimator
    → Pedagogical Policy
    → Tutor
    → TTS → Laptop Speaker
```

No Android app.

## Problem Recognition

MVP: Grok VLM only.

Output:

```json
{
  "problem_text": "...",
  "equations": [],
  "student_work": [],
  "uncertain_regions": []
}
```

Add Mathpix only if Grok recognition is insufficient.

## Domain Knowledge DB

Store verified domain knowledge, not just exact problems.

```text
Problem: text, equations, parameters, answer, source, verification
Solution: steps, template, concepts
Pedagogy: misconceptions, hint_templates
```

Matching:

* `EXACT`: same/equivalent problem → verified solution
* `TEMPLATE`: same solution structure, different values → reuse template + recalculate/verify
* `CONCEPT`: same concept only → reuse concepts/misconceptions/hints + Grok Solver
* `NEW`: Grok Solver

Do not rely on embeddings alone; verify numbers, equations, and conditions.

Grok-generated solutions are not automatically marked verified.

## Student State

Compare student work with the reference solution.

```json
{
  "current_step": "...",
  "last_correct_step": 1,
  "status": "CONCEPT_ERROR",
  "misconception": "...",
  "attempt_count": 2,
  "previous_hint_effective": false
}
```

Statuses:
`CORRECT`, `CALCULATION_ERROR`, `CONCEPT_ERROR`, `PROCEDURAL_ERROR`, `MISREAD`, `STUCK`, `UNCERTAIN`.

## Pedagogical Policy

Re-evaluate after every student action. No fixed long-term plan.

Actions:
`WAIT`, `PROBE`, `SOCRATIC_QUESTION`, `CONCEPT_HINT`, `PROCEDURAL_HINT`, `PARTIAL_STEP`, `ASK_RECAPTURE`.

Hint levels:

* L0: wait
* L1: Socratic question
* L2: concept hint
* L3: procedural hint
* L4: partial step

Use the weakest sufficient hint.
Escalate only after repeated failure.
Reduce support after improvement.
Do not reveal the final answer by default.

## Device Communication

Single WebSocket:

```text
IMAGE → JPEG
AUDIO → PCM
EVENT → hint/capture/speech events
```

Do not continuously stream high-FPS video.
Capture a high-quality frame on hint/problem requests; cache recognized problem data.

## MVP

1. Phone camera → server
2. Grok VLM recognition
3. Domain DB retrieval
4. Grok Solver fallback
5. Student State estimation
6. Adaptive hint on `"hint"`
7. Laptop TTS output

Later:

* hint history + escalation/fading
* automatic work-change/stuck detection
* proactive `"Need a hint?"`

## Rules

1. Prefer verified DB knowledge over generated solutions.
2. Use Grok Solver only when needed.
3. Never expose the Solver's full solution directly.
4. Diagnose the student's blocking point before helping.
5. Use the minimum sufficient hint.
6. Re-estimate state after every response.
7. Do not interrupt correct progress.
8. Start with Grok VLM for vision.

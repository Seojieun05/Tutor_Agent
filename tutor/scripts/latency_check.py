"""Where the seconds go, per scenario.

    python -m tutor.scripts.latency_check
    python -m tutor.scripts.latency_check --image data/captures/whatever.jpg
    python -m tutor.scripts.latency_check --repeat 3     # medians, not one roll

Runs the three turns a student can provoke against the REAL models and times
every stage, because the interesting question is not "is it slow" but "which
call". The three differ in what they are allowed to skip:

    HINT_REQUEST  recognize(+tags) → match → phrase  (solve runs in the background)
    WORK_CHECK    recognize → estimate → phrase        (problem already known)
    ANSWER        evaluate → phrase                    (no photo at all)

Capture and STT are excluded: they are measured elsewhere and depend on the
phone and the microphone rather than on any decision made here. TTS is included,
because the student waits through it before hearing a word.

THIS SPENDS REAL API CALLS — roughly a dozen per --repeat.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tutor.config import PROJECT_ROOT, load_settings
from tutor.console import say, soften_stdout
from tutor.hints.generator import HintGenerator
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.matching import Matcher, problem_hash
from tutor.knowledge.models import MatchResult, Tier
from tutor.llm import timing
from tutor.policy.engine import Action, Decision
from tutor.solver.grok_solver import GrokSolver
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator
from tutor.state.models import StudentState
from tutor.tools.registry import ToolRegistry

ANSWERED = "양변에서 5를 빼면 돼요"


class Timer:
    """Stage name → (seconds, model calls), in the order they ran.

    The call count is not decoration. A stage that took no time because a
    verified DB template matched reads exactly like a fast model, and the two
    have opposite implications for what to optimise.
    """

    def __init__(self) -> None:
        self.marks: list[tuple[str, float, int]] = []
        # (seconds the solve took, seconds AFTER the hint it became ready) —
        # background work, deliberately outside `total`: the student never waits on it
        self.background: tuple[float, float] | None = None

    def run(self, label: str, fn, *args, **kwargs):
        started, before = time.perf_counter(), timing.model_calls()
        try:
            return fn(*args, **kwargs)
        finally:
            self.marks.append(
                (label, time.perf_counter() - started, timing.model_calls() - before)
            )

    @property
    def total(self) -> float:
        return sum(seconds for _, seconds, _ in self.marks)


def build(settings):
    # Imported here, not at module scope: build_shared loads the knowledge DB
    # and (optionally) the embedding index, which --help should not pay for.
    from tutor.server.app import build_shared
    from tutor.vision.recognizer import Recognizer

    (db, llm, _transcriber, speaker, semantic,
     vision_llm, hint_llm, eval_llm) = build_shared(settings)
    return {
        "db": db,
        "recognizer": Recognizer(vision_llm, settings),
        "matcher": Matcher(db, semantic=semantic),
        "solver": GrokSolver(llm, db),
        "estimator": StudentStateEstimator(llm, db, settings.recog_conf_threshold),
        "hint_gen": HintGenerator(hint_llm, db, settings.input_mode),
        "evaluator": AnswerEvaluator(eval_llm, db),
        "speaker": speaker,
    }


def hint_request(dep, jpeg: bytes) -> tuple[Timer, object]:
    """A fresh problem, shaped exactly like the server now shapes it: the
    solver runs in the background while the first L1 hint — which needs only
    the concepts — is phrased and spoken. Two numbers matter: when the hint
    lands, and how long after it the reference arrives (the earliest a spoken
    answer could be graded)."""
    t = Timer()
    rec = t.run("recognize", dep["recognizer"].recognize, jpeg)
    match = t.run("match", dep["matcher"].match, rec)
    reference = match.reference
    future = None
    if reference is None:
        solve_started = time.perf_counter()
        future = ThreadPoolExecutor(max_workers=1).submit(
            dep["solver"].solve, rec, problem_hash(rec)
        )
    else:
        # KB hit: the reference already exists, so diagnosis runs as before
        state = t.run(
            "estimate", lambda: dep["estimator"].estimate(
                rec=rec, reference=reference, prev_state=None, prev_work=None, history=[]
            )
        )
    decision = Decision(Action.SOCRATIC_QUESTION, 1, 1, None, "bench")
    text = t.run("phrase", dep["hint_gen"].generate, decision, match, reference, rec, [])
    t.run("speak", dep["speaker"].synthesize, text)
    if future is not None:
        hint_at = time.perf_counter()
        reference = future.result()
        done_at = time.perf_counter()
        t.background = (done_at - solve_started, max(0.0, done_at - hint_at))
    return t, (rec, match, reference)


def work_check(dep, jpeg: bytes, ctx) -> Timer:
    """Same page, already identified: no tagging, no matching, no solving."""
    rec0, match, reference = ctx
    t = Timer()
    rec = t.run("recognize", dep["recognizer"].recognize, jpeg)
    rec.problem_type, rec.concepts = rec0.problem_type, rec0.concepts
    state = t.run(
        "estimate", lambda: dep["estimator"].estimate(
            rec=rec, reference=reference, prev_state=None, prev_work=rec0.student_work,
            history=[], transcript="풀이 맞아요?",
        )
    )
    decision = Decision(Action.SOCRATIC_QUESTION, 1, state.last_correct_step + 1, None, "bench")
    text = t.run(
        "phrase", dep["hint_gen"].generate, decision, match, reference, rec, [],
        "풀이 맞아요?",
    )
    t.run("speak", dep["speaker"].synthesize, text)
    return t


def answer(dep, ctx) -> Timer:
    """Spoken reply: the transcript is the only new evidence, so no photo."""
    rec, match, reference = ctx
    t = Timer()
    verdict = t.run(
        "evaluate", lambda: dep["evaluator"].evaluate(
            problem_text=rec.problem_text, reference=reference,
            question="어떤 항을 반대쪽으로 옮겨야 할까요?", target_step=1, transcript=ANSWERED,
        )
    )
    decision = Decision(Action.SOCRATIC_QUESTION, 1, 2, None, "bench")
    text = t.run(
        "phrase", dep["hint_gen"].generate, decision, match, reference, rec, [], ANSWERED
    )
    t.run("speak", dep["speaker"].synthesize, (verdict.feedback or "") + " " + text)
    return t


def report(name: str, runs: list[Timer]) -> None:
    labels: list[str] = []
    for run in runs:
        for label, _, _ in run.marks:
            if label not in labels:
                labels.append(label)

    totals = [run.total for run in runs]
    say(f"\n{name}  —  {statistics.median(totals):.1f}s"
        + (f"  (runs: {', '.join(f'{x:.1f}' for x in totals)})" if len(runs) > 1 else ""))
    for label in labels:
        values = [s for run in runs for lbl, s, _ in run.marks if lbl == label]
        calls = [c for run in runs for lbl, _, c in run.marks if lbl == label]
        median = statistics.median(values)
        share = median / statistics.median(totals) * 100
        bar = "#" * max(1, round(share / 4))
        made = statistics.median(calls)
        note = f"{made:.0f} call" + ("s" if made != 1 else "") if made else "DB/규칙, 호출 없음"
        say(f"    {label:<10} {median:5.1f}s  {share:4.0f}%  {bar:<25} {note}")

    bg = [run.background for run in runs if run.background is not None]
    if bg:
        total = statistics.median(b[0] for b in bg)
        after = statistics.median(b[1] for b in bg)
        say(f"    {'solve':<10} {total:5.1f}s   백그라운드 — 첫 힌트 이후 +{after:.1f}s에 기준 풀이 준비됨")


def main() -> None:
    soften_stdout()
    # The per-call and per-round lines from tutor.llm.timing are the point of
    # running this, and they are INFO — without this they are silently dropped.
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None,
                        help="worksheet photo (default: newest in data/captures)")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    settings = load_settings()
    if settings.echo_mode:
        sys.exit("XAI_API_KEY가 없어 에코 모드입니다. 실제 지연을 재려면 키가 필요합니다.")

    image = args.image
    if image is None:
        captures = sorted((PROJECT_ROOT / "data" / "captures").glob("*.jpg"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not captures:
            sys.exit("측정할 사진이 없습니다. --image 로 지정하거나 "
                     ".env에 SAVE_CAPTURES_DIR=data/captures 를 넣고 한 장 찍어 주세요.")
        image = captures[0]
    jpeg = image.read_bytes()
    say(f"사진: {image}  ({len(jpeg) // 1024} KB)")
    say(f"모델: chat={settings.chat_model} vision={settings.gemini_vision_model} "
        f"hint={settings.gemini_hint_model}")

    dep = build(settings)
    hints, checks, answers, ctx = [], [], [], None
    for i in range(args.repeat):
        say(f"\n--- run {i + 1}/{args.repeat} ---")
        timer, ctx = hint_request(dep, jpeg)
        hints.append(timer)
        checks.append(work_check(dep, jpeg, ctx))
        answers.append(answer(dep, ctx))

    say("\n" + "=" * 58)
    report("HINT_REQUEST (새 문제: 첫 힌트까지 — solve는 백그라운드)", hints)
    report("WORK_CHECK   (같은 문제: 인식+진단+힌트)", checks)
    report("ANSWER       (음성 답변: 채점+힌트, 사진 없음)", answers)
    say("\n캡처와 STT는 제외했습니다 — 폰과 마이크에 달린 값이라 여기서 바꿀 수 없습니다.")


if __name__ == "__main__":
    main()

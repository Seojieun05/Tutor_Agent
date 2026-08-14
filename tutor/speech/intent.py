"""What the student's utterance is FOR: four turns, only two of them look.

The tutor hears one transcript and has to pick between very different turns,
and the words alone do not decide it — what is already on the table does:

    HINT_REQUEST  "이 문제 힌트 줄래?" · "도와줘"   → capture + VLM, hint the next step
    WORK_CHECK    "풀이 맞아?" · "풀이 봐줘"        → capture + VLM, react to what they wrote
    ANSWER        "5예요" · "마이너스 3이요"        → no camera: the transcript IS the evidence
    NONE          thinking out loud                 → say nothing (spec rule 7)

The split that matters is WORK_CHECK vs ANSWER. Both arrive while a tutor
question is pending, and grading "풀이 맞아?" as an answer is how the tutor
ends up re-explaining its own question at a worksheet it never looked at. The
opposite mistake costs more in practice: hearing "아 이렇게 하면 되는구나" as a
work check stops the lesson for a photo nobody asked for — so the camera door
only opens when the student names their written work (see refers_to_work).

Rules decide the confident cases at zero cost — a bare "5예요" must never pay
for an LLM round trip, because that is the turn the student feels as a
conversation. Only genuinely ambiguous transcripts reach the one small
no-tools call, and its answer is re-checked against the same invariants the
rules enforce: you cannot answer a question nobody asked, and you cannot check
work on a worksheet the tutor has never seen.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel

from tutor.speech.stt import wants_hint

log = logging.getLogger(__name__)

Intent = Literal["HINT_REQUEST", "WORK_CHECK", "ANSWER", "NONE"]

# "지금 내가 쓴 걸 봐 달라" — and ONLY that, named out loud.
#
# A work check costs a photo and a VLM run, so it requires BOTH halves: the
# student names their written work (풀이 / 내가 쓴 것) AND asks for it to be
# judged (맞아? / 봐줘 / 확인해줘). Either half alone is everyday speech —
# "아 이렇게 하면 되는구나" is thinking out loud, "다시 말해 봐줘" wants a
# repeat, "네 맞아요" is agreement — and every one of them used to stop the
# lesson and photograph the page.
_WORK_WORDS = (
    "풀이",                                    # 풀이 맞아? · 풀이 봐줘 · 내 풀이 어때?
    "내가 쓴", "제가 쓴", "내가 푼", "제가 푼", "내가 한", "제가 한",
    # STT mangles the same short word the same few ways, and a student whose
    # "풀이 맞아?" came back as "프리 맞아?" gets no camera and no idea why.
    # These spellings are not Korean words, so widening here costs nothing:
    # the check-word half of the rule still has to agree.
    "프리",                                    # 풀이 → 프리
    "플이",                                    # 내 풀이 → 네플이 / 내플이
    "푸리", "풀리",
)
_CHECK_WORDS = ("맞", "봐", "보", "확인", "틀", "어때")


# "어디가 틀렸어?" / "왜 그게 안 돼?" — a question ABOUT a verdict already
# given, not a request for a new one. It presupposes the error the tutor just
# reported, so the answer is an explanation, not another photo of the same page.
_ABOUT_THE_VERDICT = ("어디가", "어디서", "어느 부분", "왜 ", "왜냐", "무엇이 잘못", "뭐가 잘못")

# "…대입하면 되니까, 음, 뭐지?" — the sentence walks through a step and DIES on a
# self-question. By the time this transcript exists the student has also gone
# quiet (the VAD waited out the silence before endpointing), so this is not
# muttering that will resolve itself: it is the thread being lost, out loud,
# followed by nothing. Classified NONE, the tutor sat silent at exactly the
# moment a person would have leaned in — a live session did, and the student's
# next words were about the tutor, not the maths. Trailing position only:
# "이게 뭐지 싶었는데 알았어요" sails past, because the sentence recovers.
_LOST_TRAILING = re.compile(
    r"(뭐지|뭐더라|뭐였지|뭐였더라|뭘까|어떻게 하더라|어떻게 했더라|어떻게 하지|"
    r"어떻게 되더라|기억이 안\s*나)\s*[?.!…~요]*$"
)


def trails_off_lost(text: str) -> bool:
    """Did the utterance end on a lost self-question and then stop?"""
    return bool(_LOST_TRAILING.search(text.strip()))


def asks_about_the_verdict(text: str) -> bool:
    return _has(text, _ABOUT_THE_VERDICT)


def refers_to_work(text: str) -> bool:
    """Did they explicitly name their written work AND ask for it to be looked
    at? This is the ONLY door to the camera outside a hint request — the LLM
    classifier and the answer evaluator can suggest a work check, but neither
    is allowed to open this door on a transcript that keeps both halves shut.
    """
    return _has(text, _WORK_WORDS) and _has(text, _CHECK_WORDS)

# "5예요"는 네 토큰을 넘지 않는다. 길이가 내용만큼 중요한 이유는 아래 주석 참고.
_MAX_ANSWER_TOKENS = 4
# a lone latin letter is a variable ("x는 2요"); "help" is not
_VARIABLE_RE = re.compile(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])")


class IntentGuess(BaseModel):
    intent: Intent
    reason: str = ""


_SYSTEM = """You decide what a Korean student's spoken utterance is FOR during a
math tutoring session. The tutor has a camera pointed at the student's worksheet;
taking a photo costs time, so only ask for one when the answer depends on what is
currently written.

Return exactly one intent:

- "HINT_REQUEST" — they want help getting started or getting unstuck on the
  problem itself. "이 문제 힌트 줄래?", "도와줘", "이거 어떻게 풀어요?",
  "무슨 공식 써요?". The tutor must look at the problem. An utterance that
  works through a step and then DIES on a lost self-question — "…대입하면
  되니까, 음, 뭐지?", "여기서 어떻게 하더라?" — is also a HINT_REQUEST: the
  student has lost the thread and gone quiet, and the silence after it is
  them waiting.
- "WORK_CHECK" — they explicitly NAME their written work (풀이, 내가 쓴 것) and
  ask you to judge it. "풀이 맞아요?", "풀이 확인해줘", "제 풀이 봐 주세요",
  "내가 쓴 거 봐줘". Taking a photo costs the student a pause, so this label
  requires the naming: "이렇게 하는 거 맞아요?" or "여기서 뭐가 틀렸어요?"
  without 풀이/쓴 것 is NOT a work check. Agreeing with you is not one either.
- "ANSWER" — they are replying to the question the tutor just asked, out loud.
  "5예요", "마이너스 3이요", "x는 2요", "양변에서 5를 빼요", and also "모르겠어요"
  or "힌트 더 주세요" (declining to answer is still answering). Nothing new is
  written, so no photo is needed. A plain "네, 맞아요" or "그렇게 하면 돼요" is
  an ANSWER — the student is agreeing with you, not asking you to look.
- "NONE" — not addressed to the tutor: thinking out loud that CARRIES ON or
  lands somewhere ("아 이렇게 하면 되는구나"), muttering, small talk, or a
  fragment that asks for nothing. The tutor stays quiet and keeps listening.
  An utterance that ENDS lost is not NONE — see HINT_REQUEST.

Two hard constraints, given to you as facts in the context:
- If no question is pending, "ANSWER" is impossible — there is nothing to answer.
- If the tutor has no problem in front of it, "WORK_CHECK" is impossible; a
  student asking about their work before any photo exists is a "HINT_REQUEST",
  because the camera has to see the page first either way.

Prefer "NONE" over guessing when the utterance asks for nothing. Interrupting a
student who is working correctly is worse than staying silent.

Return ONLY the JSON object."""


def _has(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _answer_shaped(text: str) -> bool:
    """"5예요", "마이너스 3이요", "x는 2요" — a value, not a request.

    Short AND carrying a number or a lone variable. Length matters as much as
    content: "3x + 5 = 20 이거 어떻게 풀어요?" is full of digits too.
    """
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens or len(tokens) > _MAX_ANSWER_TOKENS:
        return False
    return any(ch.isdigit() for ch in text) or bool(_VARIABLE_RE.search(text))


def rule_intent(text: str, *, has_problem: bool, has_pending: bool) -> Intent | None:
    """The confident cases, decided without an LLM. None means "ask the model"."""
    # First, because it outranks the keywords: "5 맞아요?" is a student saying
    # five, not asking for a photo of their page. The evaluator's own WORK_CHECK
    # verdict is the net for the ones this reads too eagerly.
    if has_pending and _answer_shaped(text):
        return "ANSWER"
    if has_pending and asks_about_the_verdict(text):
        # The tutor has just judged this page and is waiting: "내 풀이 어디가
        # 틀렸어?" asks about THAT judgement, not for a new one. Routing it to
        # the camera re-photographs the same page to re-derive the same
        # verdict, and answers a where-question with another hint. As an
        # ANSWER it reaches the evaluator, which reads it as a question and
        # hands it to explain() — the path that answers "where".
        return "ANSWER"
    if refers_to_work(text):
        # Nothing seen yet: "풀이 맞아?" still has to start with a photo.
        return "WORK_CHECK" if has_problem else "HINT_REQUEST"
    if wants_hint(text):  # 힌트 / 도와 / 모르겠 / hint / help
        # "모르겠어요" against a pending question is an answer — the evaluator
        # reads it as "escalate", which is the behaviour that already works.
        return "ANSWER" if has_pending else "HINT_REQUEST"
    if trails_off_lost(text):
        # Same routing as 모르겠어요, for the student who never says the word:
        # working aloud, losing the thread, and stopping IS being stuck.
        return "ANSWER" if has_pending else "HINT_REQUEST"
    if has_pending:
        # A question is on the table and they said something that does not point
        # at their page: it is their answer. No classifier call — this is the
        # turn that has to feel like conversation, and AnswerEvaluator sees the
        # transcript anyway and can still redirect a mis-read one to a photo.
        return "ANSWER"
    return None


def classify(
    text: str, *, has_problem: bool, has_pending: bool, llm=None
) -> Intent:
    """One label per utterance. Rules first, one small LLM call for the rest."""
    ruled = rule_intent(text, has_problem=has_problem, has_pending=has_pending)
    if ruled is not None:
        log.info("intent %s (rule): %r", ruled, text[:40])
        return ruled

    # Conservative fallback for echo mode and for a classifier that fails: today
    # an utterance with no pending hint and no hint keyword produces no response
    # at all, and staying quiet is the behaviour worth preserving (spec rule 7).
    fallback: Intent = "ANSWER" if has_pending else "NONE"
    if llm is None:
        return fallback
    try:
        guess = llm.complete_json(
            purpose="intent",
            system=_SYSTEM,
            user=_context(text, has_problem, has_pending),
            schema=IntentGuess,
        )
    except Exception:
        log.exception("intent classification failed; falling back to %s", fallback)
        return fallback

    intent = _enforce(guess.intent, has_problem=has_problem, has_pending=has_pending)
    log.info("intent %s (llm): %r — %s", intent, text[:40], guess.reason[:60])
    return intent


def _enforce(intent: Intent, *, has_problem: bool, has_pending: bool) -> Intent:
    """The model may not pick an intent the session cannot carry out."""
    if intent == "WORK_CHECK":
        # The camera fires only on an explicit "풀이 봐줘" — and the rules
        # already claimed every transcript that says so, which means a
        # WORK_CHECK arriving from the model never has the words. A model
        # guessing otherwise is exactly how the tutor kept stopping to
        # photograph the page mid-conversation.
        intent = "ANSWER" if has_pending else "NONE"
    if intent == "ANSWER" and not has_pending:
        return "HINT_REQUEST" if has_problem else "NONE"
    return intent


class IntentClassifier:
    """The classifier as a dependency, like every other stage of the pipeline.

    No LLM (echo mode, or a Deps built without one) is a supported mode, not a
    broken one: the rules alone still route every example the tutor was built
    around, and everything else stays quiet.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def classify(self, text: str, *, has_problem: bool, has_pending: bool) -> Intent:
        return classify(
            text, has_problem=has_problem, has_pending=has_pending, llm=self.llm
        )


def _context(text: str, has_problem: bool, has_pending: bool) -> str:
    yes_no = lambda flag: "예" if flag else "아니오"  # noqa: E731
    return "\n".join(
        [
            f"학생이 방금 한 말: {text}",
            f"튜터가 지금 보고 있는 문제가 있는가: {yes_no(has_problem)}",
            f"튜터가 방금 한 질문이 답을 기다리고 있는가: {yes_no(has_pending)}",
        ]
    )

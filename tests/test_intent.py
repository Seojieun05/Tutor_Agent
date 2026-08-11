"""What an utterance is FOR.

The examples below are the ones the tutor was specified around, and the point
of the table is that the SAME sentence routes differently depending on what is
already on the table. The rules must settle all of them without an LLM: a
student saying "5예요" is the turn that has to feel like a conversation, and a
round trip to a classifier is exactly what would make it not.
"""

import pytest

from tutor.speech.intent import IntentClassifier, classify, rule_intent

NEW_PROBLEM = ["이 문제 힌트 줄래?", "이 문제 도와줘.", "모르겠어."]
WORK_CHECK = [
    "풀이 맞아?",
    "풀이 봐줘.",
    "풀이 확인해줘.",
    "내 풀이 보고 설명해줘.",
    "제 풀이 좀 봐 주세요.",
    "내가 쓴 거 봐줘.",
]
# What the STT actually returns when it mangles "풀이" — reported live, and
# a student cannot know their word came back spelled differently.
MISHEARD_WORK_CHECK = [
    "프리 맞아?",
    "프리 네플이 맞아?",
    "네플이 봐줘.",
    "내플이 확인해줘.",
    "푸리 맞나요?",
    "풀리 봐 주세요.",
]

SPOKEN_ANSWERS = ["5예요.", "마이너스 3이요.", "x는 2요."]

# The bug this narrowing fixes: these are a student agreeing with the tutor, and
# every one of them used to stop the lesson and photograph the page.
AGREEMENTS = [
    "네, 맞아요.",
    "맞아요.",
    "그렇게 하면 돼요.",
    "양변에서 5를 빼요.",
    "아, 그게 맞네요.",
    "제대로 한 것 같아요.",
]

# Half a work check is not one: a check word without the work named, or the
# work named without a check word. The camera must stay quiet for all of these
# no matter which door they arrive through — rules, classifier or evaluator.
NOT_A_WORK_CHECK = [
    "이렇게 하는 거 맞아?",   # check word, work never named
    "여기서 뭐가 틀렸어?",
    "다시 설명해 봐줘.",
    "잠깐만 봐봐.",
    "아 이렇게 하면 되는구나.",
    "풀이 과정이 원래 이래?",  # names the work, asks nothing to be judged... almost
]


class RecordingLLM:
    """Counts calls: most of these must never reach it."""

    def __init__(self, intent="HINT_REQUEST"):
        self.intent = intent
        self.calls = 0

    def complete_json(self, *, purpose, system, user, schema, images=()):
        self.calls += 1
        assert purpose == "intent"
        return schema.model_validate({"intent": self.intent, "reason": "test"})


class BrokenLLM:
    def complete_json(self, **kwargs):
        raise RuntimeError("no network")


# --- 1. new-problem hint requests -------------------------------------------


@pytest.mark.parametrize("text", NEW_PROBLEM)
def test_asking_for_help_is_a_hint_request(text):
    assert rule_intent(text, has_problem=True, has_pending=False) == "HINT_REQUEST"


@pytest.mark.parametrize("text", NEW_PROBLEM)
def test_help_still_works_before_any_photo(text):
    assert rule_intent(text, has_problem=False, has_pending=False) == "HINT_REQUEST"


# --- 2. work checks ---------------------------------------------------------


@pytest.mark.parametrize("text", WORK_CHECK)
def test_asking_about_their_own_work_is_a_work_check(text):
    assert rule_intent(text, has_problem=True, has_pending=False) == "WORK_CHECK"


@pytest.mark.parametrize("text", WORK_CHECK)
def test_a_work_check_outranks_a_pending_question(text):
    """The bug this whole split exists for.

    A hint is pending, so the old code graded "풀이 맞아?" as an answer
    and re-explained its own question at a worksheet it never looked at.
    """
    assert rule_intent(text, has_problem=True, has_pending=True) == "WORK_CHECK"


@pytest.mark.parametrize("text", WORK_CHECK)
def test_a_work_check_with_nothing_seen_yet_takes_a_photo_first(text):
    """"풀이 맞아?" before any problem exists still has to start with the page."""
    assert rule_intent(text, has_problem=False, has_pending=False) == "HINT_REQUEST"


@pytest.mark.parametrize("text", MISHEARD_WORK_CHECK)
def test_a_mangled_풀이_still_reaches_the_camera(text):
    """STT hears "풀이" as 프리/네플이 often enough that a student loses the
    turn to a spelling they never chose. These forms are not Korean words, so
    accepting them costs nothing — the check-word half still has to agree."""
    assert rule_intent(text, has_problem=True, has_pending=False) == "WORK_CHECK"


@pytest.mark.parametrize("text", NOT_A_WORK_CHECK)
def test_half_a_work_check_never_reaches_the_camera_by_rule(text):
    """Both halves or nothing: the work named AND a check asked. "이렇게 하는 거
    맞아?" alone used to photograph the page mid-sentence."""
    for pending in (False, True):
        assert rule_intent(text, has_problem=True, has_pending=pending) != "WORK_CHECK"


@pytest.mark.parametrize("text", NOT_A_WORK_CHECK)
def test_the_model_cannot_open_the_camera_door_either(text):
    """A classifier voting WORK_CHECK on a transcript without the words is
    overruled — the camera fires on "풀이 봐줘" and on nothing else."""
    llm = RecordingLLM("WORK_CHECK")
    assert classify(text, has_problem=True, has_pending=False, llm=llm) != "WORK_CHECK"


# --- 3. spoken answers: the fast path ---------------------------------------


@pytest.mark.parametrize("text", SPOKEN_ANSWERS)
def test_a_spoken_value_answers_the_pending_question(text):
    assert rule_intent(text, has_problem=True, has_pending=True) == "ANSWER"


@pytest.mark.parametrize("text", SPOKEN_ANSWERS)
def test_answers_never_pay_for_a_classifier_call(text):
    llm = RecordingLLM()
    assert classify(text, has_problem=True, has_pending=True, llm=llm) == "ANSWER"
    assert llm.calls == 0


def test_a_value_with_a_check_word_is_still_an_answer():
    """"5 맞아요?" is a student saying five, not asking for a photo."""
    assert rule_intent("5 맞아요?", has_problem=True, has_pending=True) == "ANSWER"


@pytest.mark.parametrize("text", AGREEMENTS)
def test_agreeing_with_the_tutor_never_takes_a_photo(text):
    """The regression this whole narrowing is for. A bare "맞아" used to mean
    "check my work", so simply agreeing stopped the lesson for a capture."""
    assert rule_intent(text, has_problem=True, has_pending=True) == "ANSWER"


@pytest.mark.parametrize("text", AGREEMENTS)
def test_agreeing_with_nothing_pending_is_not_a_work_check_either(text):
    """It may be nothing at all, but it must never be a capture."""
    assert rule_intent(text, has_problem=True, has_pending=False) != "WORK_CHECK"


@pytest.mark.parametrize("text", ["모르겠어요", "힌트 더 주세요"])
def test_declining_to_answer_is_still_answering(text):
    """The evaluator reads these as 'escalate' — that ladder already works, and
    routing them to the camera instead would reset it."""
    assert rule_intent(text, has_problem=True, has_pending=True) == "ANSWER"


def test_nothing_is_answerable_without_a_pending_question():
    assert rule_intent("5예요.", has_problem=True, has_pending=False) is None


# --- 4. silence -------------------------------------------------------------


def test_small_talk_with_no_problem_says_nothing():
    assert classify("오늘 날씨 좋다", has_problem=False, has_pending=False) == "NONE"


def test_thinking_out_loud_does_not_interrupt_a_working_student():
    """Spec rule 7. Without an LLM the ambiguous case must stay quiet."""
    assert classify("음 그러니까 이제", has_problem=True, has_pending=False) == "NONE"


# --- 5. the LLM backstop ----------------------------------------------------


def test_a_phrasing_the_keywords_miss_reaches_the_classifier():
    llm = RecordingLLM("HINT_REQUEST")
    assert classify("이거 어떻게 풀어요?", has_problem=True, has_pending=False, llm=llm) == (
        "HINT_REQUEST"
    )
    assert llm.calls == 1


def test_the_model_cannot_answer_a_question_nobody_asked():
    llm = RecordingLLM("ANSWER")
    assert classify("이거 어떻게 풀어요?", has_problem=True, has_pending=False, llm=llm) == (
        "HINT_REQUEST"
    )


def test_a_model_work_check_without_the_words_stays_quiet():
    """It reached the LLM at all only because the rules found neither half of
    "풀이 봐줘" in it — so whatever the model says, this is not a capture."""
    llm = RecordingLLM("WORK_CHECK")
    assert classify("이거 어때요", has_problem=False, has_pending=False, llm=llm) == "NONE"
    assert classify("이거 어때요", has_problem=True, has_pending=False, llm=llm) == "NONE"


def test_a_broken_classifier_falls_back_instead_of_killing_the_turn():
    assert classify("웅얼웅얼", has_problem=True, has_pending=True, llm=BrokenLLM()) == "ANSWER"
    assert classify("웅얼웅얼", has_problem=True, has_pending=False, llm=BrokenLLM()) == "NONE"


def test_the_classifier_dependency_works_without_an_llm():
    """Echo mode is a supported configuration, not a broken one."""
    classifier = IntentClassifier()
    assert classifier.classify("힌트 주세요", has_problem=False, has_pending=False) == (
        "HINT_REQUEST"
    )
    assert classifier.classify("풀이 맞아?", has_problem=True, has_pending=True) == (
        "WORK_CHECK"
    )

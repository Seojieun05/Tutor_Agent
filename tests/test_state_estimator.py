import pytest

from tutor.knowledge.models import Answer, ReferenceSolution, SolutionStep
from tutor.llm.echo import EchoLLMClient
from tutor.state.estimator import StudentStateEstimator, hint_was_effective
from tutor.state.models import StudentState
from tutor.store.session_store import HintRecord
from tutor.vision.recognizer import Recognition

REFERENCE = ReferenceSolution(
    steps=[
        SolutionStep(idx=1, description="이항", expression="3*x = 15"),
        SolutionStep(idx=2, description="나눗셈", expression="x = 5"),
    ],
    final_answer=Answer(kind="SCALAR", value="5"),
    concepts=["linear_equation"],
    verified=True,
    origin="db",
)


def pending_hint() -> HintRecord:
    return HintRecord(
        id=1, problem_hash="p", step=1, level=1, action="SOCRATIC_QUESTION",
        hint_text="힌트", effective=None,
    )


class TestHintWasEffective:
    prev = StudentState(status="CONCEPT_ERROR", misconception="sign_flip_on_move", last_correct_step=0)

    def test_progress(self):
        new = self.prev.model_copy(update={"last_correct_step": 1})
        assert hint_was_effective(self.prev, new)

    def test_misconception_resolved(self):
        new = self.prev.model_copy(update={"misconception": None})
        assert hint_was_effective(self.prev, new)

    def test_status_improved(self):
        new = self.prev.model_copy(update={"status": "CORRECT"})
        assert hint_was_effective(self.prev, new)

    def test_none_of_the_above(self):
        assert not hint_was_effective(self.prev, self.prev.model_copy())

    def test_error_to_error_not_improvement(self):
        new = self.prev.model_copy(update={"status": "CALCULATION_ERROR", "misconception": "sign_flip_on_move"})
        assert not hint_was_effective(self.prev, new)

    def test_uncertain_is_not_improvement(self):
        # a blurry frame loses the misconception field too — neither counts
        new = self.prev.model_copy(update={"status": "UNCERTAIN"})
        assert not hint_was_effective(self.prev, new)


class TestUncertainEcho:
    """A model shown a previous UNCERTAIN tends to answer UNCERTAIN. Live,
    one blurry photo early in a problem turned every later work check into
    UNCERTAIN → probe, forever — with a legible page and real work on it."""

    WORK = Recognition(
        problem_text="다음 일차방정식을 푸시오: 3x + 5 = 20",
        equations=["3*x + 5 = 20"],
        student_work=["3*x = 20 + 5"],       # wrong sign: a diagnosable page
        confidence=0.95,
    )

    def test_an_uncertain_echo_gets_a_second_unbiased_look(self, db):
        llm = EchoLLMClient({"estimate": [
            {"current_step": "?", "last_correct_step": 0, "status": "UNCERTAIN",
             "misconception": None, "attempt_count": 2, "previous_hint_effective": None},
            {"current_step": "이항", "last_correct_step": 0, "status": "CONCEPT_ERROR",
             "misconception": "sign_flip_on_move", "attempt_count": 2,
             "previous_hint_effective": False},
        ]})
        est = StudentStateEstimator(llm, db)
        state = est.estimate(
            rec=self.WORK, reference=REFERENCE,
            prev_state=StudentState(status="UNCERTAIN"),   # the bias source
            prev_work=None, history=[],
        )
        assert llm.calls.count("estimate") == 2            # the fresh-eyes retry
        assert state.status == "CONCEPT_ERROR"             # and it saw the page

    def test_uncertain_without_a_previous_state_stands(self, db):
        """No bias to shed → nothing to retry: a first-look UNCERTAIN is real."""
        llm = EchoLLMClient({"estimate": [
            {"current_step": "?", "last_correct_step": 0, "status": "UNCERTAIN",
             "misconception": None, "attempt_count": 1, "previous_hint_effective": None},
        ]})
        est = StudentStateEstimator(llm, db)
        state = est.estimate(
            rec=self.WORK, reference=REFERENCE,
            prev_state=None, prev_work=None, history=[],
        )
        assert llm.calls.count("estimate") == 1
        assert state.status == "UNCERTAIN"

    def test_an_old_misconception_repeated_on_changed_work_gets_fresh_eyes(self, db):
        """The other parrot, live: work corrected from r to r**3, and the
        model repeated the r-relation misconception word for word. A changed
        page with an unchanged diagnosis is suspect — look again unbiased."""
        stale_misc = "등비수열의 관계를 r로 잘못 설정함"
        llm = EchoLLMClient({"estimate": [
            {"current_step": "비 세우기", "last_correct_step": 0,
             "status": "CONCEPT_ERROR", "misconception": stale_misc,
             "attempt_count": 3, "previous_hint_effective": False},
            {"current_step": "비 세우기", "last_correct_step": 1, "status": "CORRECT",
             "misconception": None, "attempt_count": 3, "previous_hint_effective": True},
        ]})
        est = StudentStateEstimator(llm, db)
        state = est.estimate(
            rec=self.WORK, reference=REFERENCE,
            prev_state=StudentState(status="CONCEPT_ERROR", misconception=stale_misc),
            prev_work=["다른 줄"],                      # the page CHANGED
            history=[],
        )
        assert llm.calls.count("estimate") == 2
        assert state.status == "CORRECT"

    def test_the_same_diagnosis_on_the_same_page_is_not_an_echo(self, db):
        """Unchanged work showing the same mistake SHOULD keep its diagnosis —
        no retry, no second-guessing."""
        stale_misc = "등비수열의 관계를 r로 잘못 설정함"
        llm = EchoLLMClient({"estimate": [
            {"current_step": "비 세우기", "last_correct_step": 0,
             "status": "CONCEPT_ERROR", "misconception": stale_misc,
             "attempt_count": 3, "previous_hint_effective": False},
        ]})
        est = StudentStateEstimator(llm, db)
        state = est.estimate(
            rec=self.WORK, reference=REFERENCE,
            prev_state=StudentState(status="CONCEPT_ERROR", misconception=stale_misc),
            prev_work=list(self.WORK.student_work),     # identical page
            history=[],
        )
        assert llm.calls.count("estimate") == 1
        assert state.status == "CONCEPT_ERROR"

    def test_a_stubborn_uncertain_gets_the_standby_models_opinion(self, db):
        """Live: the routed small model answered UNCERTAIN even with fresh
        eyes, and the turn looped into "다시 보여 줄래요?" forever. One
        expensive look beats an endless recapture."""
        uncertain = {"current_step": "?", "last_correct_step": 0, "status": "UNCERTAIN",
                     "misconception": None, "attempt_count": 2,
                     "previous_hint_effective": None}
        small = EchoLLMClient({"estimate": [uncertain, uncertain]})
        strong = EchoLLMClient({"estimate": [
            {"current_step": "이항", "last_correct_step": 0, "status": "CONCEPT_ERROR",
             "misconception": "sign_flip_on_move", "attempt_count": 2,
             "previous_hint_effective": False},
        ]})
        est = StudentStateEstimator(small, db, second_opinion=strong)
        state = est.estimate(
            rec=self.WORK, reference=REFERENCE,
            prev_state=StudentState(status="UNCERTAIN"),
            prev_work=None, history=[],
        )
        assert small.calls.count("estimate") == 2      # first look + fresh eyes
        assert strong.calls.count("estimate") == 1     # then the standby
        assert state.status == "CONCEPT_ERROR"

    def test_when_the_standby_agrees_uncertain_is_honest(self, db):
        uncertain = {"current_step": "?", "last_correct_step": 0, "status": "UNCERTAIN",
                     "misconception": None, "attempt_count": 1,
                     "previous_hint_effective": None}
        small = EchoLLMClient({"estimate": [uncertain]})
        strong = EchoLLMClient({"estimate": [dict(uncertain)]})
        est = StudentStateEstimator(small, db, second_opinion=strong)
        state = est.estimate(
            rec=self.WORK, reference=REFERENCE,
            prev_state=None, prev_work=None, history=[],
        )
        assert strong.calls.count("estimate") == 1
        assert state.status == "UNCERTAIN"             # a real recapture case


class TestPreChecks:
    def test_low_confidence_uncertain_no_llm(self, db):
        llm = EchoLLMClient()
        est = StudentStateEstimator(llm, db, conf_threshold=0.6)
        rec = Recognition(problem_text="p", equations=[], student_work=["3x=15"], confidence=0.3)
        state = est.estimate(
            rec=rec, reference=REFERENCE, prev_state=None, prev_work=None, history=[]
        )
        assert state.status == "UNCERTAIN"
        assert llm.calls == []

    def test_empty_work_stuck_no_llm(self, db):
        llm = EchoLLMClient()
        est = StudentStateEstimator(llm, db)
        rec = Recognition(problem_text="p", equations=[], student_work=[])
        state = est.estimate(
            rec=rec, reference=REFERENCE, prev_state=None, prev_work=None, history=[]
        )
        assert state.status == "STUCK"
        assert llm.calls == []

    def test_erased_work_keeps_misconception(self, db):
        # erasing everything must not read as "misconception resolved"
        est = StudentStateEstimator(EchoLLMClient(), db)
        prev = StudentState(status="CONCEPT_ERROR", misconception="sign_flip_on_move")
        rec = Recognition(problem_text="p", equations=[], student_work=[])
        state = est.estimate(
            rec=rec, reference=REFERENCE, prev_state=prev, prev_work=["3*x = 25"],
            history=[pending_hint()],
        )
        assert state.misconception == "sign_flip_on_move"
        assert not hint_was_effective(prev, state)

    def test_unchanged_work_after_hint_no_llm(self, db):
        llm = EchoLLMClient()
        est = StudentStateEstimator(llm, db)
        prev = StudentState(status="CONCEPT_ERROR", misconception="sign_flip_on_move", attempt_count=1)
        rec = Recognition(problem_text="p", equations=[], student_work=["3*x = 20 + 5"], confidence=0.9)
        state = est.estimate(
            rec=rec,
            reference=REFERENCE,
            prev_state=prev,
            prev_work=["3*x = 20 + 5"],
            history=[pending_hint()],
        )
        assert llm.calls == []
        assert state.previous_hint_effective is False
        assert state.attempt_count == 2
        assert state.status == "CONCEPT_ERROR"


class TestRuleBasedProgress:
    """The symbolic fast path: matching work lines need no LLM diagnosis."""

    def test_progressed_work_is_correct_without_llm(self, db):
        llm = EchoLLMClient()
        est = StudentStateEstimator(llm, db)
        rec = Recognition(
            problem_text="p", equations=["3*x + 5 = 20"],
            student_work=["3*x = 15"], confidence=0.9,
        )
        state = est.estimate(
            rec=rec, reference=REFERENCE, prev_state=None, prev_work=None, history=[]
        )
        assert llm.calls == []
        assert state.status == "CORRECT"
        assert state.last_correct_step == 1

    def test_solved_work_reaches_last_step(self, db):
        est = StudentStateEstimator(EchoLLMClient(), db)
        rec = Recognition(
            problem_text="p", equations=["3*x + 5 = 20"],
            student_work=["3*x = 15", "x = 5"], confidence=0.9,
        )
        state = est.estimate(
            rec=rec, reference=REFERENCE, prev_state=None, prev_work=None, history=[]
        )
        assert (state.status, state.last_correct_step) == ("CORRECT", 2)

    def test_same_solution_set_is_not_the_same_step(self, db):
        # 'x = 5' as the only line is step 2, never step 1 ('3*x = 15')
        est = StudentStateEstimator(EchoLLMClient(), db)
        rec = Recognition(
            problem_text="p", equations=["3*x + 5 = 20"],
            student_work=["x = 5"], confidence=0.9,
        )
        state = est.estimate(
            rec=rec, reference=REFERENCE, prev_state=None, prev_work=None, history=[]
        )
        assert state.last_correct_step == 2

    def test_restated_problem_is_stuck_without_llm(self, db):
        llm = EchoLLMClient()
        est = StudentStateEstimator(llm, db)
        rec = Recognition(
            problem_text="p", equations=["3*x + 5 = 20"],
            student_work=["3*x + 5 = 20"], confidence=0.9,
        )
        state = est.estimate(
            rec=rec, reference=REFERENCE, prev_state=None, prev_work=None, history=[]
        )
        assert llm.calls == []
        assert (state.status, state.last_correct_step) == ("STUCK", 0)

    def test_wrong_work_still_needs_llm(self, db):
        llm = EchoLLMClient()
        est = StudentStateEstimator(llm, db)
        rec = Recognition(
            problem_text="p", equations=["3*x + 5 = 20"],
            student_work=["3*x = 20 + 5"], confidence=0.9,
        )
        est.estimate(rec=rec, reference=REFERENCE, prev_state=None, prev_work=None, history=[])
        assert llm.calls == ["estimate"]


class TestLLMPath:
    def test_llm_called_and_post_rules_applied(self, db):
        llm = EchoLLMClient(
            {
                "estimate": [
                    {
                        "current_step": "이항",
                        "last_correct_step": 99,  # gets clamped
                        "status": "CONCEPT_ERROR",
                        "misconception": "sign_flip_on_move",
                        "attempt_count": 7,
                        "previous_hint_effective": None,
                    }
                ]
            }
        )
        est = StudentStateEstimator(llm, db)
        rec = Recognition(
            problem_text="p", equations=[], student_work=["3*x = 20 + 5"], confidence=0.9
        )
        state = est.estimate(
            rec=rec, reference=REFERENCE, prev_state=None, prev_work=None, history=[]
        )
        assert llm.calls == ["estimate"]
        assert state.last_correct_step == 2  # clamped to len(steps)
        assert state.attempt_count == 1  # post-rule, prev is None

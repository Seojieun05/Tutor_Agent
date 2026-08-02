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
    return HintRecord(id=1, step=1, level=1, action="SOCRATIC_QUESTION", hint_text="힌트", effective=None)


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

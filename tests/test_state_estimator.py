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

    def test_a_better_status_cannot_hide_regressed_progress(self):
        prev = StudentState(status="CALCULATION_ERROR", last_correct_step=2)
        new = StudentState(status="CORRECT", last_correct_step=1)
        assert not hint_was_effective(prev, new)


class TestProgressDoesNotRewindOnACorrectCrop:
    def test_correct_newest_line_preserves_the_proven_prefix(self, db):
        est = StudentStateEstimator(EchoLLMClient(), db)
        prev = StudentState(status="CALCULATION_ERROR", last_correct_step=2)
        reported = StudentState(
            current_step="현재 사진에 보이는 줄은 맞음",
            status="CORRECT",
            last_correct_step=1,
        )

        state = est._post_rules(reported, REFERENCE, prev, rec=None)

        assert state.status == "CORRECT"
        assert state.last_correct_step == 2


class TestAFrontierResultMustBeWritten:
    """Live on problem 12: the page showed
    "2*(a_1+a_4+a_7) = r**3*(a_1+a_4+a_7)" — one division short of r³ = 2 —
    the diagnosis credited the r³ step as done, and the tutor asked for a_1
    while r³ was still unwritten. A route may replace a step's DERIVATION,
    never its RESULT: an unevidenced frontier retreats one step, and the
    replaced derivations beneath it stay credited."""

    GEO_REF = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="첫 묶음을 a_1로 나타내기",
                         expression="2*a_1*(1 + r**3 + r**6) = 6, a_1*(1 + r**3 + r**6) = 3"),
            SolutionStep(idx=2, description="둘째 묶음을 같은 꼴로 나타내기",
                         expression="a_1*r**3*(1 + r**3 + r**6) = 6"),
            SolutionStep(idx=3, description="두 식을 나눠 r³ 구하기",
                         expression="r**3 = 2"),
            SolutionStep(idx=4, description="첫 식에 대입해 a_1 구하기",
                         expression="a_1*(1 + 2 + 4) = 3, a_1 = 3/7"),
        ],
        final_answer=Answer(kind="SCALAR", value="24/7"),
        concepts=["geometric_sequence"], verified=True, origin="db",
    )
    LINE = "2*(a_1 + a_4 + a_7) = r**3*(a_1 + a_4 + a_7)"

    def estimated(self, db, work, transcript=None, prev=None):
        llm = EchoLLMClient({"estimate": [{
            "current_step": "공비 관계 파악", "last_correct_step": 3,
            "status": "CORRECT", "misconception": None,
        }]})
        est = StudentStateEstimator(llm, db)
        return est.estimate(
            rec=Recognition(problem_text="p", student_work=work, confidence=0.95),
            reference=self.GEO_REF, prev_state=prev, prev_work=None,
            history=[], transcript=transcript,
        )

    def test_the_setup_line_alone_keeps_the_result_step_open(self, db):
        state = self.estimated(db, [self.LINE], transcript="이거 맞아요?")
        assert state.last_correct_step == 2   # steps 1-2: route-replaced, kept

    def test_the_written_result_keeps_the_credit(self, db):
        state = self.estimated(db, [self.LINE, "r**3 = 2"])
        assert state.last_correct_step == 3

    def test_the_spoken_result_keeps_the_credit(self, db):
        state = self.estimated(db, [self.LINE], transcript="r 세제곱은 2예요")
        assert state.last_correct_step == 3

    def test_an_already_proven_frontier_is_not_reopened(self, db):
        prev = StudentState(status="CORRECT", last_correct_step=3)
        state = self.estimated(db, [self.LINE], prev=prev)
        assert state.last_correct_step == 3


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

    def test_a_stubborn_uncertain_asks_for_a_new_photo(self, db):
        """When fresh eyes still cannot decide, the honest move is the camera:
        the live cases were cropped frames, not a model that needed a bigger
        model behind it. Two looks, then ask to be shown the page again."""
        uncertain = {"current_step": "?", "last_correct_step": 0, "status": "UNCERTAIN",
                     "misconception": None, "attempt_count": 2,
                     "previous_hint_effective": None}
        llm = EchoLLMClient({"estimate": [uncertain, uncertain]})
        est = StudentStateEstimator(llm, db)
        state = est.estimate(
            rec=self.WORK, reference=REFERENCE,
            prev_state=StudentState(status="UNCERTAIN"),
            prev_work=None, history=[],
        )
        assert llm.calls.count("estimate") == 2        # first look + fresh eyes
        assert state.status == "UNCERTAIN"             # and no third opinion


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

    def test_expanded_derivative_advances_from_proven_prefix_without_llm(self, db):
        """Live problem 13: step 3 substitutes f'(x), while the verified
        reference leaves it symbolic.  A cropped photo no longer shows steps
        1-2, but their proven prefix plus this line must advance to step 3."""
        reference = ReferenceSolution(
            steps=[
                SolutionStep(
                    idx=1,
                    description="f'(x)로 접선 l의 기울기 구하기",
                    expression="f'(x) = 2*x - 4, f'(1) = -2",
                ),
                SolutionStep(
                    idx=2,
                    description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                    expression="l: y = -2*x - 4",
                ),
                SolutionStep(
                    idx=3,
                    description="곱의 미분법으로 g'(x) 쓰기",
                    expression=(
                        "g'(x) = (3*x**2 - 2)*f(x) "
                        "+ (x**3 - 2*x)*f'(x)"
                    ),
                ),
                SolutionStep(
                    idx=4,
                    description="g'(1) 계산",
                    expression="g'(1) = -4",
                ),
            ],
            final_answer=Answer(kind="SCALAR", value="49"),
            concepts=["differentiation"],
            verified=True,
            origin="db",
        )
        rec = Recognition(
            problem_text="13. 함수 f(x)의 접선과 함수 g(x)의 접선",
            equations=[
                "f(x) = x**2 - 4*x - 3",
                "g(x) = (x**3 - 2*x)*f(x)",
            ],
            student_work=[
                "g'(x) = (3*x**2 - 2)*f(x) "
                "+ (x**3 - 2*x)*(2*x - 4)"
            ],
            confidence=1.0,
        )
        llm = EchoLLMClient()
        state = StudentStateEstimator(llm, db).estimate(
            rec=rec,
            reference=reference,
            prev_state=StudentState(status="CORRECT", last_correct_step=2),
            prev_work=["l: y = -2*x - 4"],
            history=[],
        )

        assert llm.calls == []
        assert (state.status, state.last_correct_step) == ("CORRECT", 3)

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

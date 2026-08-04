from tutor.hints.guard import leaks_answer
from tutor.hints.generator import HintGenerator
from tutor.knowledge.models import (
    Answer,
    MatchResult,
    ReferenceSolution,
    SolutionStep,
    Tier,
)
from tutor.llm.echo import EchoLLMClient
from tutor.policy.engine import Action, Decision
from tutor.vision.recognizer import Recognition

LIN_REF = ReferenceSolution(
    steps=[
        SolutionStep(idx=1, description="양변에서 5를 뺀다", expression="3*x = 15"),
        SolutionStep(idx=2, description="양변을 3으로 나눈다", expression="x = 5"),
    ],
    final_answer=Answer(kind="SCALAR", value="5"),
    concepts=["linear_equation"],
    verified=True,
    origin="db",
)

QUAD_REF = ReferenceSolution(
    steps=[SolutionStep(idx=1, description="인수분해", expression="(x - 2)*(x - 3) = 0")],
    final_answer=Answer(kind="ROOT_SET", value=["2", "3"]),
    concepts=["quadratic_equation"],
    verified=True,
    origin="db",
)

DERIV_REF = ReferenceSolution(
    steps=[SolutionStep(idx=1, description="거듭제곱 법칙", expression="3*x**2 + 2*x")],
    final_answer=Answer(kind="EXPRESSION", value="3*x**2 + 2*x"),
    concepts=["differentiation"],
    verified=True,
    origin="db",
)


class TestLeakGuard:
    def test_scalar_literal_assignment(self):
        assert leaks_answer("정답은 x = 5 이에요", LIN_REF, 1)

    def test_scalar_bare_number_and_float(self):
        assert leaks_answer("답이 5인지 볼까요?", LIN_REF, 1)
        assert leaks_answer("결과는 5.0이에요", LIN_REF, 1)

    def test_scalar_clean_hint_passes(self):
        assert not leaks_answer("x만 남기려면 어떤 항을 옮겨야 할까요?", LIN_REF, 1)

    def test_future_step_verbatim(self):
        assert leaks_answer("다음은 3*x = 15 로 쓰면 돼요", LIN_REF, 0)
        # ...but the CURRENT target step may be revealed at L4
        assert not leaks_answer("다음은 3*x = 15 로 쓰면 돼요", LIN_REF, 1)

    def test_root_set_individual_roots(self):
        assert leaks_answer("근 중 하나는 2예요", QUAD_REF, 1)
        assert leaks_answer("x = 3 이 근이죠", QUAD_REF, 1)
        assert not leaks_answer("곱해서 6, 더해서 -5가 되는 두 수를 찾아보세요", QUAD_REF, 1)

    def test_expression_symbolic_rewrite(self):
        assert leaks_answer("답은 2*x + 3*x**2 이에요", DERIV_REF, 0)
        assert not leaks_answer("각 항에 거듭제곱 법칙을 적용해 보세요", DERIV_REF, 0)

    def test_expression_implicit_multiplication(self):
        ten_x = ReferenceSolution(
            steps=[SolutionStep(idx=1, description="거듭제곱 법칙", expression="10*x")],
            final_answer=Answer(kind="EXPRESSION", value="10*x"),
            concepts=["differentiation"],
            verified=True,
            origin="db",
        )
        assert leaks_answer("답은 10x 입니다", ten_x, 0)

    def test_scalar_numeric_compound(self):
        assert leaks_answer("15/3을 계산하면 답이 나와요", LIN_REF, 1)

    def test_exponent_digits_are_not_values(self):
        # 'x**2' contains the digit 2 but must not count as the root 2
        assert not leaks_answer("x**2 항을 어떻게 인수분해할까요?", QUAD_REF, 1)
        assert not leaks_answer("x^3의 지수를 확인해 보세요", QUAD_REF, 1)


def decision(level=1, misconception=None, target=1):
    from tutor.policy.engine import LEVEL_ACTIONS

    return Decision(LEVEL_ACTIONS[level], level, target, misconception, "test")


def lin_match(db_problem=None):
    return MatchResult(
        tier=Tier.EXACT,
        problem=db_problem,
        concepts=["linear_equation"],
        reference=LIN_REF,
    )


class TestGenerator:
    def test_db_template_first_no_llm(self, db):
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        text = gen.generate(decision(1), lin_match(), LIN_REF, Recognition(problem_text="p"), [])
        assert text
        assert llm.calls == []  # verified DB pedagogy, no LLM
        assert not leaks_answer(text, LIN_REF, 1)

    def test_misconception_template_with_safe_term(self, db):
        # answer is 5 and b=5 → the {term}=5 template would leak; generator must skip it
        problem = db.find_by_text_hash("nonexistent")  # None is fine
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        match = lin_match(problem)
        match = match.model_copy(update={"bindings": {"a": "3", "b": "5", "c": "20"}})
        text = gen.generate(
            decision(1, misconception="sign_flip_on_move"),
            match,
            LIN_REF,
            Recognition(problem_text="p"),
            [],
        )
        assert not leaks_answer(text, LIN_REF, 1)

    def test_l4_uses_description_not_answer(self, db):
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        text = gen.generate(
            decision(4, target=2), lin_match(), LIN_REF, Recognition(problem_text="p"), []
        )
        assert text
        assert "x = 5" not in text
        assert not leaks_answer(text, LIN_REF, 2)

    def test_fixed_actions(self, db):
        gen = HintGenerator(EchoLLMClient(), db)
        d = Decision(Action.ASK_RECAPTURE, 0, 1, None, "r")
        assert "카메라" in gen.generate(d, lin_match(), LIN_REF, Recognition(problem_text=""), [])
        d = Decision(Action.WAIT, 0, 1, None, "r")
        assert gen.generate(d, lin_match(), LIN_REF, Recognition(problem_text=""), []) == ""

    def test_completed_problem_gets_praise(self, db):
        gen = HintGenerator(EchoLLMClient(), db)
        text = gen.generate(
            decision(1, target=3), lin_match(), LIN_REF, Recognition(problem_text="p"), []
        )
        assert "훌륭" in text
        assert not leaks_answer(text, LIN_REF, 3)

    def test_llm_fallback_leak_gets_regenerated(self, db):
        # no templates match concepts=[] → LLM path; first phrase leaks, second is clean
        llm = EchoLLMClient(
            {"phrase": [{"hint": "정답은 x = 5!"}, {"hint": "다음 단계로 무엇을 할까요?"}]}
        )
        gen = HintGenerator(llm, db)
        match = MatchResult(tier=Tier.NEW, concepts=["unknown_concept"], reference=LIN_REF)
        text = gen.generate(decision(1), match, LIN_REF, Recognition(problem_text="p"), [])
        assert llm.calls == ["phrase", "phrase"]
        assert not leaks_answer(text, LIN_REF, 1)


class TestStudentAnswerInThePrompt:
    """Hints build on what the student just said, at every call site."""

    class RecordingLLM(EchoLLMClient):
        """Captures the phrase prompt so we can assert on what the model sees."""

        def __init__(self, responses=None):
            super().__init__(responses)
            self.prompts: list[str] = []

        def run_with_tools(self, *, purpose, system, user, images=(), schema, max_rounds=6):
            if purpose == "phrase":
                self.prompts.append(user)
            return super().run_with_tools(
                purpose=purpose, system=system, user=user, images=images,
                schema=schema, max_rounds=max_rounds,
            )

    def _llm_path_match(self):
        # concepts with no seeded templates → the LLM phrasing path
        return MatchResult(tier=Tier.NEW, concepts=["unknown_concept"], reference=LIN_REF)

    def test_hint_without_an_answer_still_works(self, db):
        """Regression: _phrase() lacked the parameter, so EVERY phrased hint
        raised TypeError — including plain hint requests that pass nothing."""
        gen = HintGenerator(self.RecordingLLM(), db)
        text = gen.generate(
            decision(1), self._llm_path_match(), LIN_REF, Recognition(problem_text="p"), []
        )
        assert text

    def test_the_answer_reaches_the_model(self, db):
        llm = self.RecordingLLM()
        gen = HintGenerator(llm, db)
        gen.generate(
            decision(1),
            self._llm_path_match(),
            LIN_REF,
            Recognition(problem_text="3x + 5 = 20"),
            [],
            student_answer="5를 빼면 되는 거 아니에요?",
        )
        assert llm.prompts, "the phrase call never happened"
        assert "5를 빼면 되는 거 아니에요?" in llm.prompts[0]

    def test_a_leaking_hint_is_regenerated_with_the_answer_intact(self, db):
        llm = self.RecordingLLM(
            {"phrase": [{"hint": "정답은 x = 5!"}, {"hint": "그다음은 무엇을 할까요?"}]}
        )
        gen = HintGenerator(llm, db)
        text = gen.generate(
            decision(1), self._llm_path_match(), LIN_REF, Recognition(problem_text="p"), [],
            student_answer="5를 빼요",
        )
        assert len(llm.prompts) == 2  # first leaked, second is the retry
        assert all("5를 빼요" in p for p in llm.prompts)
        assert not leaks_answer(text, LIN_REF, 1)

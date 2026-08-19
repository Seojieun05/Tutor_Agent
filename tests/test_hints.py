import pytest

from tutor.hints.guard import leaks_answer
from tutor.hints.generator import (
    HintGenerator,
    PhrasedHint,
    SafeWordEmitter,
    guided_step_question,
    mentions_future_step,
)
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


class TestSafeWordEmitter:
    def test_safe_words_leave_before_the_sentence_is_complete(self):
        out = []
        emitter = SafeWordEmitter(out.append, LIN_REF, 1, [])
        emitter.feed("지금 두 번째 줄에서 어떤 등식의 성질을 ")
        assert out, "the rolling quarantine held the entire sentence"
        assert "".join(out) != emitter.seen
        emitter.feed("써야 할까요?")
        text, blocked = emitter.finish(emitter.seen)
        assert not blocked
        assert "".join(out) == text

    def test_answer_word_never_leaves_quarantine(self):
        out = []
        emitter = SafeWordEmitter(out.append, LIN_REF, 1, [])
        unsafe = "두 번째 줄을 천천히 보면 정답은 바로 5예요 "
        emitter.feed(unsafe)
        text, blocked = emitter.finish(unsafe)
        spoken = "".join(out)
        assert blocked
        assert "5" not in spoken and "5" not in text
        # the closure CONTINUES the amputated sentence instead of confessing
        assert "이어서 완성해 볼까요" in spoken

    def test_l1_step_announcement_never_reaches_the_live_stream(self):
        out = []
        emitter = SafeWordEmitter(
            out.append, LIN_REF, 1, [], forbid_step_announcement=True
        )
        announced = "곱의 미분법으로 g'(x) 쓰기 차례예요. "
        emitter.feed(announced)
        text, blocked = emitter.finish(announced)

        assert blocked
        assert "차례" not in "".join(out) and "차례" not in text
        assert text.endswith("?")

    def test_future_step_question_never_leaves_the_live_stream(self):
        reference = ReferenceSolution(
            steps=[
                SolutionStep(idx=1, description="접선 l의 기울기 구하기",
                             expression="f'(1) = -2"),
                SolutionStep(idx=2, description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                             expression="l: y = -2*x - 4"),
            ],
            final_answer=Answer(kind="SCALAR", value="49"), concepts=[],
        )
        out = []
        emitter = SafeWordEmitter(
            out.append, reference, 1, [], forbid_future_step=True
        )
        future = (
            "우선 구하신 기울기 마이너스 2와 점 (1, -6)을 사용해서 "
            "접선 l의 방정식을 어떻게 나타낼 수 있을까요? "
        )
        emitter.feed(future)
        text, blocked = emitter.finish(future)

        assert blocked
        assert "방정식" not in "".join(out) and "방정식" not in text


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
    def test_live_generator_replaces_an_answer_before_it_is_emitted(self):
        class NoTemplates:
            def hint_templates_for(self, *args, **kwargs):
                return []

            def get_misconception(self, *args, **kwargs):
                return None

        llm = EchoLLMClient({"phrase": [{"hint": "천천히 보면 정답은 바로 5예요"}]})
        gen = HintGenerator(llm, NoTemplates())
        out = []
        text = gen.generate(
            decision(1),
            MatchResult(tier=Tier.NEW, concepts=[]),
            LIN_REF,
            Recognition(problem_text="일차방정식"),
            [],
            on_delta=out.append,
        )
        assert "5" not in "".join(out)
        assert "5" not in text
        # blocked at the door (nothing yet released) → the FRESH opener
        assert out and "직접" in text

    def test_db_template_first_no_llm(self, db):
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        text = gen.generate(decision(1), lin_match(), LIN_REF, Recognition(problem_text="p"), [])
        assert text
        assert text.startswith("우선 ")
        assert llm.calls == []  # verified DB pedagogy, no LLM
        assert not leaks_answer(text, LIN_REF, 1)

    def test_later_l1_template_connects_with_now(self, db):
        text = HintGenerator(EchoLLMClient(), db).generate(
            decision(1, target=2), lin_match(), LIN_REF,
            Recognition(problem_text="p"), [],
        )
        assert text.startswith("이제 ")

    def test_product_rule_step_is_invited_without_announcing_the_method(self):
        text = guided_step_question("곱의 미분법으로 g'(x) 쓰기", 3)
        assert text == "이제 g(x)가 두 식의 곱이라는 점을 보고, 어떻게 미분하면 좋을까요?"
        assert "곱의 미분법" not in text and "차례" not in text

    def test_a_target_step_one_hint_cannot_ask_for_step_two(self, db):
        reference = ReferenceSolution(
            steps=[
                SolutionStep(idx=1, description="접선 l의 기울기 구하기",
                             expression="f'(1) = -2"),
                SolutionStep(idx=2, description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                             expression="l: y = -2*x - 4"),
                SolutionStep(idx=3, description="곱의 미분법으로 g'(x) 쓰기",
                             expression="g'(x) = u'(x)*v(x) + u(x)*v'(x)"),
            ],
            final_answer=Answer(kind="SCALAR", value="49"), concepts=[],
        )
        future = (
            "우선 구하신 기울기 마이너스 2와 점 (1, -6)을 사용해서 "
            "접선 l의 방정식을 어떻게 나타낼 수 있을까요?"
        )
        assert mentions_future_step(future, reference, 1)

        class NoTemplates:
            def hint_templates_for(self, *args, **kwargs):
                return []

            def get_misconception(self, *args, **kwargs):
                return None

        llm = EchoLLMClient({"phrase": [
            {"hint": future},
            {"hint": "우선 접선 l의 기울기는 얼마인지 확인해 볼까요?"},
        ]})
        text = HintGenerator(llm, NoTemplates()).generate(
            Decision(Action.SOCRATIC_QUESTION, 1, 1, "force_llm", "test"),
            MatchResult(tier=Tier.EXACT, concepts=[], reference=reference),
            reference, Recognition(problem_text="p"), [],
        )

        assert llm.calls == ["phrase", "phrase"]
        assert "방정식" not in text

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
        d = Decision(Action.WAIT, 0, 1, None, "r")
        assert gen.generate(d, lin_match(), LIN_REF, Recognition(problem_text=""), []) == ""

    @pytest.mark.parametrize("mode, word, absent", [
        ("upload", "사진", "카메라"),
        ("camera", "카메라", "올려"),
    ])
    def test_recapture_asks_for_the_picture_the_student_can_actually_give(
        self, db, mode, word, absent
    ):
        """Telling a student to hold their worksheet up to a camera that is not
        connected is worse than saying nothing at all."""
        gen = HintGenerator(EchoLLMClient(), db, input_mode=mode)
        d = Decision(Action.ASK_RECAPTURE, 0, 1, None, "r")
        text = gen.generate(d, lin_match(), LIN_REF, Recognition(problem_text=""), [])
        assert word in text and absent not in text

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

    def test_llm_step_announcement_gets_regenerated_as_a_question(self, db):
        llm = EchoLLMClient({"phrase": [
            {"hint": "곱의 미분법으로 g'(x) 쓰기 차례예요."},
            {"hint": "이제 g(x)가 두 식의 곱이라는 점을 보고, 어떻게 미분하면 좋을까요?"},
        ]})
        reference = ReferenceSolution(
            steps=[
                SolutionStep(idx=1, description="첫 값 구하기", expression="a = 1"),
                SolutionStep(idx=2, description="둘째 값 구하기", expression="b = 2"),
                SolutionStep(idx=3, description="곱의 미분법으로 g'(x)를 씁니다.",
                             expression="g'(x) = u'(x)*v(x) + u(x)*v'(x)"),
            ],
            final_answer=Answer(kind="SCALAR", value="1"),
            concepts=["unknown_concept"], verified=True, origin="db",
        )
        text = HintGenerator(llm, db).generate(
            decision(1, target=3),
            MatchResult(tier=Tier.NEW, concepts=["unknown_concept"], reference=reference),
            reference, Recognition(problem_text="p"), [],
        )

        assert llm.calls == ["phrase", "phrase"]
        assert text.startswith("이제 ") and "차례" not in text


class TestTheirWorkReachesThePrompt:
    """The wording models see what the student actually wrote — from the most
    recent photo, no extra capture — so "어디가 잘못된 거야?" can be answered
    by pointing at THEIR line instead of explaining the step in the abstract."""

    WORKED = Recognition(
        problem_text="다음 일차방정식을 푸시오: 3x + 5 = 20",
        equations=["3*x + 5 = 20"],
        student_work=["3*x = 20 + 5", "3*x = 25"],
        confidence=0.95,
    )

    def _llm_match(self):
        return MatchResult(tier=Tier.NEW, concepts=["unknown_concept"], reference=LIN_REF)

    def test_the_hint_prompt_carries_their_lines(self, db):
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        gen.generate(decision(1), self._llm_match(), LIN_REF, self.WORKED, [])
        assert "3*x = 20 + 5" in llm.prompts_for("phrase")[0]

    def test_the_explain_prompt_carries_their_lines(self, db):
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        gen.explain(
            student_question="어디가 잘못된 거예요?",
            tutor_question="어떤 항을 옮겨야 할까요?",
            match=self._llm_match(),
            reference=LIN_REF,
            rec=self.WORKED,
            target_step=1,
        )
        assert "3*x = 20 + 5" in llm.prompts_for("explain")[0]

    def test_an_empty_page_adds_nothing(self, db):
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        gen.generate(
            decision(1), self._llm_match(), LIN_REF,
            Recognition(problem_text="p", student_work=[]), [],
        )
        assert "쓴 풀이" not in llm.prompts_for("phrase")[0]


class TestTheHintAimsAtTheMistake:
    """Live: a student wrote the product rule correctly but differentiated
    -2x as -2x. The estimator named exactly that, and the hint still aimed at
    the target step — praising the structure they had already built and
    pointing at a term they had already written right, while the slip went
    unmentioned. A named mistake outranks the step."""

    class Recording(EchoLLMClient):
        def __init__(self, responses=None):
            super().__init__(responses)
            self.prompt = ""

        def run_with_tools(self, *, purpose, system, user, images=(), schema, max_rounds=6):
            if purpose == "phrase":
                self.prompt = user
            return super().run_with_tools(
                purpose=purpose, system=system, user=user, images=images,
                schema=schema, max_rounds=max_rounds,
            )

    def _match(self):
        return MatchResult(tier=Tier.NEW, concepts=["unknown_concept"], reference=LIN_REF)

    def test_the_misconception_is_named_as_the_hints_job(self, db):
        llm = self.Recording()
        HintGenerator(llm, db).generate(
            decision(2, misconception="2x의 미분을 2가 아닌 2x로 계산함"),
            self._match(), LIN_REF, Recognition(problem_text="p"), [],
        )
        assert "2x의 미분을 2가 아닌 2x로 계산함" in llm.prompt
        assert "이번 힌트가 다뤄야 할 바로 그것" in llm.prompt
        # and the step is demoted to context, not the target
        assert "참고로 학생이 향하는 단계" in llm.prompt

    def test_without_a_misconception_the_step_is_still_the_target(self, db):
        llm = self.Recording()
        HintGenerator(llm, db).generate(
            decision(2), self._match(), LIN_REF, Recognition(problem_text="p"), [],
        )
        assert "학생이 지금 해내야 하는 단계" in llm.prompt
        assert "참고로 학생이 향하는 단계" not in llm.prompt

    def test_l4_still_gets_to_say_the_step(self, db):
        """The escape hatch is unchanged: at L4 the step may be spoken."""
        llm = self.Recording()
        HintGenerator(llm, db).generate(
            decision(4, misconception="부호를 반대로 옮김"),
            self._match(), LIN_REF, Recognition(problem_text="p"), [],
        )
        assert "알려줘도 되는 다음 단계" in llm.prompt


class TestTheBoard:
    """What the tutor WRITES travels with what it says, through the same gate:
    writing "x = 5" while carefully not saying it is still giving the answer."""

    SEEN = Recognition(
        problem_text="다음 일차방정식을 푸시오: 3x + 5 = 20",
        equations=["3*x + 5 = 20"],
    )
    # the same page with a line of their own on it
    WORKED = SEEN.model_copy(update={"student_work": ["3*x = 20 + 5"]})

    def _llm_match(self):
        # concepts with no seeded templates → the LLM phrasing path
        return MatchResult(tier=Tier.NEW, concepts=["unknown_concept"], reference=LIN_REF)

    def test_the_board_rides_on_the_hint(self, db):
        llm = EchoLLMClient({"phrase": [
            {"hint": "어느 항을 옮기면 x만 남을까요?",
             "board": [{"expr": "3*x + 5 = 20", "note": "이 식에서 출발"}]}
        ]})
        gen = HintGenerator(llm, db)
        text = gen.generate(decision(1), self._llm_match(), LIN_REF, self.SEEN, [])
        assert text == "어느 항을 옮기면 x만 남을까요?"   # still a str to everyone else
        assert [(b.expr, b.note) for b in text.board] == [("3*x + 5 = 20", "이 식에서 출발")]

    def test_a_bare_expression_still_makes_a_board_line(self, db):
        """A model that answers with plain strings does not lose its board."""
        llm = EchoLLMClient({"phrase": [
            {"hint": "여기를 보세요", "board": ["3*x + 5 = 20"]}
        ]})
        text = HintGenerator(llm, db).generate(
            decision(1), self._llm_match(), LIN_REF, self.SEEN, []
        )
        assert [(b.expr, b.note) for b in text.board] == [("3*x + 5 = 20", "")]

    def test_a_note_that_is_a_sentence_is_dropped_but_the_line_stays(self, db):
        """The note is a margin scribble. Past a label's length it is the hint
        said twice, and the board turns back into a transcript."""
        llm = EchoLLMClient({"phrase": [
            {"hint": "생각해 볼까요?", "board": [{
                "expr": "3*x + 5 = 20",
                "note": "이 식에서 상수항 5를 양변에서 빼면 x만 남게 되는데 그 과정을 떠올려 보세요",
            }]}
        ]})
        text = HintGenerator(llm, db).generate(
            decision(1), self._llm_match(), LIN_REF, self.SEEN, []
        )
        assert [(b.expr, b.note) for b in text.board] == [("3*x + 5 = 20", "")]

    def test_a_leaking_note_cancels_the_board_like_a_leaking_expression(self, db):
        """The note is words on the screen, so it answers to the same guard."""
        llm = EchoLLMClient({"phrase": [
            {"hint": "다음은요?", "board": [{"expr": "3*x + 5 = 20", "note": "답은 5"}]}
        ]})
        text = HintGenerator(llm, db).generate(
            decision(1), self._llm_match(), LIN_REF, self.SEEN, []
        )
        assert text.board == ()

    def test_one_leaking_line_cancels_the_whole_board(self, db):
        """Dropping just the leaking line left fragments on screen ("a₁" alone,
        half a derivation) — half a board reads worse than none, so a leak
        anywhere empties it. The hint itself still ships."""
        llm = EchoLLMClient({"phrase": [
            {"hint": "다음에는 무엇을 하면 좋을까요?", "board": ["x = 5", "3*x + 5 = 20"]}
        ]})
        gen = HintGenerator(llm, db)
        text = gen.generate(decision(1), self._llm_match(), LIN_REF, self.SEEN, [])
        assert text == "다음에는 무엇을 하면 좋을까요?"
        assert text.board == ()                        # all or nothing

    def test_the_students_own_line_never_reaches_the_board(self, db):
        """Live: the student wrote a wrong 등비수열 relation, the model put it
        on the board, and it appeared under "튜터 풀이" in the tutor's own
        emphasis style — a wrong line in the tutor's hand reads as the tutor
        endorsing it. Their page already has it; words point at it."""
        llm = EchoLLMClient({"phrase": [
            {"hint": "두 번째 줄을 다시 볼까요?",
             "board": ["3*x = 20 + 5", "3*x + 5 = 20"]},   # theirs, then the problem's
        ]})
        gen = HintGenerator(llm, db)
        text = gen.generate(decision(1), self._llm_match(), LIN_REF, self.WORKED, [])
        assert [b.expr for b in text.board] == ["3*x + 5 = 20"]

    def test_a_reformatted_copy_is_still_a_copy(self, db):
        """Spacing and the multiplication sign are the only things that differ
        between how the VLM reads a line and how the model writes it."""
        llm = EchoLLMClient({"phrase": [
            {"hint": "여기를 보세요", "board": ["3x=20+5"]},   # they wrote "3*x = 20 + 5"
        ]})
        gen = HintGenerator(llm, db)
        text = gen.generate(decision(1), self._llm_match(), LIN_REF, self.WORKED, [])
        assert text.board == ()

    def test_fragments_and_korean_never_reach_the_board(self, db):
        """The live 등비수열 board: bare terms ("a_1") and Korean sentences
        are not mathematics to write — they are filtered before the guard."""
        llm = EchoLLMClient({"phrase": [
            {"hint": "공비가 몇 번 곱해질까요?",
             "board": ["a_1", "일반항을 써 보세요", "a_4 = a_1 * r**3"]}
        ]})
        gen = HintGenerator(llm, db)
        text = gen.generate(decision(1), self._llm_match(), LIN_REF, self.SEEN, [])
        assert [b.expr for b in text.board] == ["a_4 = a_1 * r**3"]

    def test_the_hint_no_longer_decides_the_drawing(self, db):
        """Drawing moved to the illustrator, which runs while this hint is
        SPOKEN and can read it. Two deciders drew two pictures for one
        sentence, so the phrasing model stopped being one of them."""
        llm = EchoLLMClient({"phrase": [
            {"hint": "그래프의 개형을 떠올려 볼까요?", "board": [],
             "graph": ["x**2 - 4*x + 3"]},             # ignored now
        ]})
        text = HintGenerator(llm, db).generate(
            decision(1), self._llm_match(), LIN_REF, self.SEEN, []
        )
        assert not hasattr(text, "graph") or not getattr(text, "graph", ())

    def test_paths_without_a_board_read_as_an_empty_one(self, db):
        gen = HintGenerator(EchoLLMClient(), db)
        text = gen.generate(
            Decision(Action.WAIT, 0, 1, None, "r"), lin_match(), LIN_REF, self.SEEN, []
        )
        assert getattr(text, "board", ()) == ()        # how the session reads plain strs


class TestStudentAnswerInThePrompt:
    """Hints build on what the student just said, at every call site."""

    def _llm_path_match(self):
        # concepts with no seeded templates → the LLM phrasing path
        return MatchResult(tier=Tier.NEW, concepts=["unknown_concept"], reference=LIN_REF)

    def test_hint_without_an_answer_still_works(self, db):
        """Regression: _phrase() lacked the parameter, so EVERY phrased hint
        raised TypeError — including plain hint requests that pass nothing."""
        gen = HintGenerator(EchoLLMClient(), db)
        text = gen.generate(
            decision(1), self._llm_path_match(), LIN_REF, Recognition(problem_text="p"), []
        )
        assert text

    def test_the_answer_reaches_the_model(self, db):
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        gen.generate(
            decision(1),
            self._llm_path_match(),
            LIN_REF,
            Recognition(problem_text="3x + 5 = 20"),
            [],
            student_answer="5를 빼면 되는 거 아니에요?",
        )
        phrased = llm.prompts_for("phrase")
        assert phrased, "the phrase call never happened"
        assert "5를 빼면 되는 거 아니에요?" in phrased[0]

    def test_a_leaking_hint_is_regenerated_with_the_answer_intact(self, db):
        llm = EchoLLMClient(
            {"phrase": [{"hint": "정답은 x = 5!"}, {"hint": "그다음은 무엇을 할까요?"}]}
        )
        gen = HintGenerator(llm, db)
        text = gen.generate(
            decision(1), self._llm_path_match(), LIN_REF, Recognition(problem_text="p"), [],
            student_answer="5를 빼요",
        )
        phrased = llm.prompts_for("phrase")
        assert len(phrased) == 2  # first leaked, second is the retry
        assert all("5를 빼요" in p for p in phrased)
        assert not leaks_answer(text, LIN_REF, 1)


class TestVisibleNumbersAreNotSecrets:
    """In `3x + 5 = 20` the answer is 5 and so is a coefficient.

    The guard used to reject any hint containing 5, which banned the most
    natural L1 question on that problem ("5를 어떻게 없앨까요?") and pushed the
    tutor onto a generic template exactly where it had something useful to say.
    Telling a student a number they are looking at is not telling them anything
    — but saying it AS the answer, or computing it, still is.
    """

    REF = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="양변에서 5를 뺀다", expression="3*x = 15"),
            SolutionStep(idx=2, description="양변을 3으로 나눈다", expression="x = 5"),
        ],
        final_answer=Answer(kind="SCALAR", value="5"),
        concepts=["linear_equation"], verified=True, origin="db",
    )
    GIVEN = ["3x + 5 = 20 일 때 x는?", "3*x + 5 = 20"]

    @pytest.mark.parametrize("hint", [
        "5를 어떻게 없애면 x만 남을까요?",
        "양변에서 5를 빼면 어떻게 될까요?",
        "x 옆에 있는 더하기 5를 어떻게 정리하면 좋을까요?",
        "왼쪽의 20에서 5를 어떻게 다뤄야 할까요?",
    ])
    def test_a_number_on_their_page_may_be_named(self, hint):
        assert not leaks_answer(hint, self.REF, 1, given=self.GIVEN)

    @pytest.mark.parametrize("hint", [
        "답은 5예요.",
        "정답은 5입니다.",
        "x는 5예요.",
        "x = 5 가 나와요.",
        "계산하면 5가 돼요.",
    ])
    def test_saying_it_as_the_answer_is_still_a_leak(self, hint):
        assert leaks_answer(hint, self.REF, 1, given=self.GIVEN)

    @pytest.mark.parametrize("hint", ["15/3을 하면 되겠죠?", "(20 - 5)/3 를 해보세요."])
    def test_computing_it_is_still_a_leak(self, hint):
        """The given numbers are theirs; the arithmetic between them is the step."""
        assert leaks_answer(hint, self.REF, 1, given=self.GIVEN)

    def test_a_number_not_on_their_page_is_still_a_leak(self):
        """Same hint, a problem where 5 is not written down."""
        assert leaks_answer("5를 생각해 보세요.", self.REF, 1, given=["3*x + 7 = 22"])

    def test_without_the_problem_the_guard_stays_strict(self):
        """Callers that cannot say what is visible lose nothing they had."""
        assert leaks_answer("5를 어떻게 없애면 x만 남을까요?", self.REF, 1)

    def test_a_later_step_is_still_rejected(self):
        assert leaks_answer("x = 5 로 정리돼요.", self.REF, 1, given=self.GIVEN)

    def test_the_generator_now_keeps_the_good_question(self, db):
        """End to end: the phrasing that used to be thrown away survives."""
        class Writer:
            def run_with_tools(self, **kw):
                return PhrasedHint(hint="x 옆에 있는 5를 어떻게 없애면 좋을까요?")
            complete_json = run_with_tools

        text = HintGenerator(Writer(), db).generate(
            Decision(Action.SOCRATIC_QUESTION, 1, 1, None, "t"),
            # a concept with no seeded template, so the LLM path is taken
            MatchResult(tier=Tier.NEW, concepts=["unknown_concept"], reference=self.REF),
            self.REF,
            Recognition(problem_text="3x + 5 = 20 일 때 x는?", equations=["3*x + 5 = 20"]),
            [],
        )
        assert "5를 어떻게 없애면" in text


class TestAnEquationStepHidesItsContent:
    """Live, on problem 13: the reference's later step read "m: y = -13x + 19"
    and the illustrator drew the curve "-13*x + 19" — the student was still
    deriving m. The guard skipped every step containing "=", so the RHS of a
    step the student had not reached passed as long as it was not spelled
    verbatim. An equation step is now checked piece by piece."""

    TANGENT_REF = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="미분", expression="f'(x) = 3*x**2 - 6*x"),
            SolutionStep(idx=5, description="기울기", expression="f'(1) = -3"),
            SolutionStep(idx=6, description="접선 m", expression="m: y = -13*x + 19"),
        ],
        final_answer=Answer(kind="SCALAR", value="27/4"),
        concepts=["tangent_line"],
        verified=True,
        origin="db",
    )

    def test_the_live_leak_is_caught(self):
        # the exact expression the illustrator sent to the plot
        assert leaks_answer("-13*x + 19", self.TANGENT_REF, 5)

    def test_a_rewrite_of_the_rhs_is_caught_too(self):
        assert leaks_answer("19 - 13*x", self.TANGENT_REF, 5)

    def test_the_reached_step_is_still_allowed(self):
        # idx=5 IS the target step: naming its content is the L4 privilege,
        # filtered upstream, not the guard's business
        assert not leaks_answer("-3", self.TANGENT_REF, 5)
        assert not leaks_answer("3*x**2 - 6*x", self.TANGENT_REF, 5)

    def test_a_bare_variable_name_is_not_content(self):
        # "y를 구해 볼까요?" must not trip on the "y" left of the equals sign
        assert not leaks_answer("y를 어떻게 구할까요?", self.TANGENT_REF, 5)


class TestATemplateNeverWearsASentence:
    """Live: a solver-written reference had steps like "함수 f(x)의 도함수
    f'(x)를 구합니다." and the concept template glued a particle straight onto
    it — "…구합니다.가 먼저예요". A step that reads as a sentence cannot fill
    {step}; the template is skipped and the phrasing model, which can inflect,
    gets the turn instead."""

    def db_with_template(self):
        from tutor.knowledge.db import KnowledgeDB
        from tutor.knowledge.models import HintTemplate
        db = KnowledgeDB(":memory:")
        db.insert_concept("differentiation", "미분")
        db.insert_hint_template(HintTemplate(
            id="t1", concept_id="differentiation", level=2,
            template_text="{step}가 먼저예요.",
        ))
        return db

    def reference(self, description):
        return ReferenceSolution(
            steps=[SolutionStep(idx=1, description=description, expression="f'(x) = 2*x - 4")],
            final_answer=Answer(kind="SCALAR", value="49"),
            concepts=["differentiation"], verified=True, origin="db",
        )

    def generate(self, db, reference):
        from tutor.hints.generator import HintGenerator
        from tutor.knowledge.models import MatchResult, Tier
        from tutor.policy.engine import Action, Decision
        from tutor.vision.recognizer import Recognition
        gen = HintGenerator(EchoLLMClient(), db)
        return gen.generate(
            Decision(Action.CONCEPT_HINT, 2, 1, None, "t"),
            MatchResult(tier=Tier.EXACT, concepts=["differentiation"], reference=reference),
            reference, Recognition(problem_text="p"), [],
        )

    def test_a_sentence_step_skips_the_template(self):
        db = self.db_with_template()
        text = self.generate(db, self.reference("함수 f(x)의 도함수 f'(x)를 구합니다."))
        assert "구합니다.가" not in text and "구합니다가" not in text

    def test_a_noun_step_still_rides_it(self):
        db = self.db_with_template()
        text = self.generate(db, self.reference("접선 l의 기울기 구하기"))
        assert text == "접선 l의 기울기 구하기가 먼저예요."

    def test_trailing_punctuation_is_stripped_either_way(self):
        db = self.db_with_template()
        text = self.generate(db, self.reference("접선 l의 기울기 구하기."))
        assert text == "접선 l의 기울기 구하기가 먼저예요."


class TestATutorAsksInItsOwnVoice:
    """The deterministic L1 conjugates the step verb instead of gluing a
    particle onto a label — the audit that forced this heard "바꿔를 어떻게
    쓰면 좋을까요?", "찾기를 해 볼까요?" and "f'(1)를 구해 볼까요?"."""

    @pytest.mark.parametrize("description,expected", [
        # a phrase ending on a connective flows straight into the verb
        ("구하는 값을 밑 3으로 바꿔 쓰기", "이제 구하는 값을 밑 3으로 바꿔 써 볼까요?"),
        ("둘째 조건에 대입해 정리하기", "이제 둘째 조건에 대입해 정리해 볼까요?"),
        # an adverbial tail (…으로) already carries its particle
        ("ㄱ: k = 0 일 때 위치를 적분으로 구하기", "이제 ㄱ: k = 0 일 때 위치를 적분으로 구해 볼까요?"),
        ("넓이를 정적분으로 세우기", "이제 넓이를 정적분으로 세워 볼까요?"),
        # step verbs are conjugated, never "X기를 해 볼까요"
        ("극값을 갖는 x 찾기", "이제 극값을 갖는 x를 찾아볼까요?"),
        ("참인 보기 모으기", "이제 참인 보기를 모아 볼까요?"),
        ("점근선 확인", "이제 점근선을 확인해 볼까요?"),
        ("극댓값을 a로 나타내기", "이제 극댓값을 a로 나타내 볼까요?"),
    ])
    def test_step_labels_become_natural_questions(self, description, expected):
        assert guided_step_question(description, 3) == expected

    @pytest.mark.parametrize("description,expected", [
        # the particle follows the SOUND of the tail: 일 → 을, 세제곱 → 삼 → 을
        ("두 값을 더해 f'(1) 구하기", "이제 두 값을 더해 f'(1)을 구해 볼까요?"),
        ("두 식을 나눠 r³ 구하기", "이제 두 식을 나눠 r³을 구해 볼까요?"),
        ("첫 식에 대입해 a_1 구하기", "이제 첫 식에 대입해 a_1을 구해 볼까요?"),
        # letter names: 엘/엠/엔/알 close on a consonant, the rest stay open
        ("접선 l의 기울기와 m 구하기", "이제 접선 l의 기울기와 m을 구해 볼까요?"),
        ("교점의 x좌표에서 x 구하기", "이제 교점의 x좌표에서 x를 구해 볼까요?"),
    ])
    def test_the_particle_follows_the_spoken_tail(self, description, expected):
        assert guided_step_question(description, 2) == expected


class TestTheSeedGateRefusesAnnouncers:
    """warm_kb refuses an L1 template that announces the step: the first
    batch was seeded as "{step} 차례예요 …" and sat in the DB as dead rows,
    silently skipped by the generator on every single turn."""

    def test_an_announcing_l1_is_refused_and_a_question_is_kept(self, db):
        from tutor.scripts.warm_kb import seed_hint_templates

        n = seed_hint_templates(db, {"hint_templates": [
            {"id": "gate-l1-bad", "concept_id": "linear_equation", "level": 1,
             "template_text": "{step} 차례예요. 어떻게 시작하면 좋을까요?"},
            {"id": "gate-l1-good", "concept_id": "linear_equation", "level": 1,
             "template_text": "등식의 양쪽에서 무엇을 똑같이 할 수 있을까요?"},
            {"id": "gate-l2-ok", "concept_id": "linear_equation", "level": 2,
             "template_text": "이항은 부호를 바꿔 옮기는 거예요. 어느 항부터 옮기면 좋을까요?"},
        ]})

        assert n == 2
        kept = {t.id for t in db.hint_templates_for(["linear_equation"], None, 1)}
        assert "gate-l1-bad" not in kept
        assert "gate-l1-good" in kept


class TestADiagnosedMistakeOutranksBoilerplate:
    """Live: the policy named the exact slip ("2x의 미분을 2x로 계산") and the
    concept template answered with its stock line about tangent slopes. A
    diagnosed misconception may only be answered by its own pedagogy or by
    the phrasing model that was handed the diagnosis — never by a concept
    line that ignores it."""

    CONCEPT_LINE = "곡선 위 한 점에서의 접선의 기울기는 그 점의 미분계수와 같아요. 지금은 어느 점의 미분계수가 필요할까요?"

    def seeded(self, db):
        from tutor.knowledge.models import HintTemplate
        db.insert_hint_template(HintTemplate(
            id="t-l2-diff", concept_id="differentiation", level=2,
            template_text=self.CONCEPT_LINE,
        ))
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        rec = Recognition(problem_text="접선 문제", equations=[],
                          concepts=["differentiation"], confidence=0.95)
        match = MatchResult(tier=Tier.CONCEPT, concepts=["differentiation"])
        return gen, llm, rec, match

    def test_the_concept_template_stands_when_nothing_is_diagnosed(self, db):
        gen, llm, rec, match = self.seeded(db)
        decision = Decision(Action.CONCEPT_HINT, 2, 3, None, "escalate")
        text = gen.generate(decision, match, None, rec, history=[])
        # served straight from the DB's concept pedagogy, no model asked
        concept_lines = {
            t.template_text
            for t in db.hint_templates_for(["differentiation"], None, 2)
            if t.concept_id is not None
        }
        assert str(text) in concept_lines
        assert "phrase" not in llm.calls

    def test_a_misconception_takes_the_turn_to_the_phrasing_model(self, db):
        gen, llm, rec, match = self.seeded(db)
        decision = Decision(
            Action.CONCEPT_HINT, 2, 3,
            "x^3 - 2x의 도함수를 구할 때 2x의 미분을 2x로 잘못 계산함",
            "hint L1 ineffective: escalate to L2",
        )
        text = gen.generate(decision, match, None, rec, history=[])
        assert str(text) != self.CONCEPT_LINE
        assert "phrase" in llm.calls


class TestAPrewrittenLadderIsWrittenAsALadder:
    """Live on problem 12 the ten prewritten lines were written in parallel,
    each blind to the other nine: steps 2 and 3 both asked "두 묶음을 비교해
    볼까요", and a step-4 L2 taught step 1's division. A prewrite call now
    sees what earlier steps established, what the other lines already say,
    and which questions belong to later steps."""

    REF = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="첫 묶음을 a_1로 나타내기",
                         expression="a_1*(1 + r**3 + r**6) = 3"),
            SolutionStep(idx=2, description="둘째 묶음을 같은 꼴로 나타내기",
                         expression="a_1*r**3*(1 + r**3 + r**6) = 6"),
            SolutionStep(idx=3, description="두 식을 나눠 r³ 구하기",
                         expression="r**3 = 2"),
        ],
        final_answer=Answer(kind="SCALAR", value="24/7"),
        concepts=["geometric_sequence"], verified=True, origin="db",
    )

    def written(self, db, step_idx, siblings=()):
        from tutor.knowledge.models import Problem
        llm = EchoLLMClient({"phrase": [
            {"hint": "이제 두 식을 어떻게 이용하면 좋을까요?"}
        ]})
        gen = HintGenerator(llm, db)
        problem = Problem(
            id="p12", problem_type="geometric_sequence",
            problem_text="등비수열 문제", equations=[],
            answer=Answer(kind="SCALAR", value="24/7"),
            concepts=["geometric_sequence"], verified=True,
        )
        gen.prewrite(
            problem=problem, reference=self.REF,
            rec=Recognition(problem_text="등비수열 문제",
                            concepts=["geometric_sequence"]),
            step_idx=step_idx, level=1, siblings=siblings,
        )
        return llm.prompts[-1]

    def test_the_first_line_is_told_which_questions_come_later(self, db):
        prompt = self.written(db, step_idx=1)
        assert "뒤에 올 단계들" in prompt
        assert "두 식을 나눠" in prompt        # step 3's name, so step 1 leaves it
        assert "r**3 = 2" not in prompt        # names only, never expressions

    def test_a_later_line_is_told_what_is_already_established(self, db):
        prompt = self.written(db, step_idx=3)
        assert "이미 구해 놓은 것" in prompt
        assert "a_1*(1 + r**3 + r**6) = 3" in prompt

    def test_sibling_lines_are_listed_as_do_not_repeat(self, db):
        prompt = self.written(
            db, step_idx=3, siblings=["2단계 L1: 두 번째 묶음도 나타내 볼까요?"]
        )
        assert "두 번째 묶음도 나타내 볼까요?" in prompt
        assert "같은 질문을" in prompt


class TestConceptBoilerplateNeverOutranksAReference:
    """Live, problem 10 at step 1: the answer to the first concept question
    escalated to L2 while the solver's reference already knew the asymptote
    WAS the step — and the exponential concept template re-asked its generic
    fixed-point trivia anyway. With a reference in hand, L2+ concept lines
    give way to pedagogy that can see the step; without one they still
    serve, because they are all there is."""

    LINE = "지수함수는 밑이 무엇이든 반드시 한 점을 지나요. 그 점을 이 문제 어디에 쓸 수 있을까요?"

    def seeded(self, db):
        from tutor.knowledge.models import HintTemplate
        db.insert_hint_template(HintTemplate(
            id="t-l2-expo", concept_id="exponential_function", level=2,
            template_text=self.LINE,
        ))
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        rec = Recognition(problem_text="지수함수 문제", equations=[],
                          concepts=["exponential_function"], confidence=0.95)
        match = MatchResult(tier=Tier.CONCEPT, concepts=["exponential_function"])
        return gen, llm, rec, match

    EXPO_REF = ReferenceSolution(
        steps=[SolutionStep(idx=1, description="점근선 확인",
                            expression="y = a**x - 2 의 점근선은 y = -2")],
        final_answer=Answer(kind="SCALAR", value="2**(5/2)"),
        concepts=["exponential_function"], verified=True, origin="db",
    )

    def test_step_one_is_no_shield_once_a_reference_exists(self, db):
        gen, llm, rec, match = self.seeded(db)
        decision = Decision(Action.CONCEPT_HINT, 2, 1, None, "escalate")
        text = gen.generate(decision, match, self.EXPO_REF, rec, history=[])
        assert str(text) != self.LINE
        assert "phrase" in llm.calls

    def test_without_a_reference_the_template_still_serves(self, db):
        gen, llm, rec, match = self.seeded(db)
        decision = Decision(Action.CONCEPT_HINT, 2, 1, None, "escalate")
        text = gen.generate(decision, match, None, rec, history=[])
        assert str(text) == self.LINE
        assert "phrase" not in llm.calls


class TestAPrewrittenLineServesFirst:
    """A hint written at warm time for THIS problem's THIS step outranks the
    conjugated label and the concept template — model quality at template
    price — but never a diagnosed misconception, never a TEMPLATE-tier
    cousin (different numbers), and never twice."""

    LINE = "접선의 기울기가 이 곡선 어디에 숨어 있는지 살펴볼까요?"

    def ready(self, db, tier=Tier.EXACT):
        from tutor.knowledge.models import Problem
        problem = Problem(
            id="pw-1", problem_type="derivative_applications",
            problem_text="접선 문제", equations=[],
            answer=Answer(kind="SCALAR", value="1"),
            concepts=["differentiation"], verified=True,
        )
        db.save_prewritten_hint("pw-1", 1, 1, self.LINE)
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        rec = Recognition(problem_text="접선 문제", equations=[],
                          concepts=["differentiation"], confidence=0.95)
        match = MatchResult(tier=tier, concepts=["differentiation"], problem=problem)
        return gen, llm, rec, match

    def decision(self, misconception=None):
        return Decision(Action.SOCRATIC_QUESTION, 1, 1, misconception, "t")

    def test_the_written_line_is_served_without_a_model(self, db):
        gen, llm, rec, match = self.ready(db)
        text = gen.generate(self.decision(), match, None, rec, history=[])
        assert str(text) == self.LINE
        assert llm.calls == []

    def test_a_template_tier_cousin_never_reads_it(self, db):
        # same shape, different numbers: the prewritten words may name values
        gen, llm, rec, match = self.ready(db, tier=Tier.TEMPLATE)
        text = gen.generate(self.decision(), match, None, rec, history=[])
        assert str(text) != self.LINE

    def test_a_diagnosed_misconception_still_goes_to_the_model(self, db):
        gen, llm, rec, match = self.ready(db)
        text = gen.generate(
            self.decision("2x의 미분을 2x로 계산"), match, None, rec, history=[]
        )
        assert str(text) != self.LINE
        assert "phrase" in llm.calls

    def test_a_line_already_said_falls_through(self, db):
        from tutor.store.session_store import HintRecord
        gen, llm, rec, match = self.ready(db)
        said = [HintRecord(id=1, problem_hash="h", step=1, level=1,
                           action="SOCRATIC_QUESTION", hint_text=self.LINE,
                           effective=None)]
        text = gen.generate(self.decision(), match, None, rec, history=said)
        assert str(text) != self.LINE


def test_problem_13_curated_l2_is_source_data_and_survives_presolve(db):
    from tutor.knowledge.matching import Matcher
    from tutor.scripts.warm_kb import load_presolve, presolve

    entries = load_presolve()
    spec = entries["13"]
    source_hint = next(
        h for h in entries["13"]["prewritten_hints"]
        if h["step"] == 2 and h["level"] == 2
    )
    line = source_hint["text"]

    assert presolve(db, None, "13", entries)
    assert db.prewritten_hint("presolved-13", 2, 2) == line
    assert db.prewritten_hint("presolved-13-v1", 2, 2) == line
    stored = db.prewritten_hint_artifact("presolved-13", 2, 2)
    assert stored is not None
    assert stored.text == line
    assert [b.expr for b in stored.board] == ["y - y_1 = -2*(x - x_1)"]
    assert "y - y_1" not in line
    assert "다음과 같은 꼴" in line
    assert "어떻게 관계" not in line
    assert "y = -2*x - 4" not in line

    rec = Recognition(
        problem_text=spec["problem_text"],
        equations=spec["equations"][0],
        problem_type=spec["problem_type"],
        concepts=spec["concepts"],
    )
    match = Matcher(db).match(rec)
    llm = EchoLLMClient()
    served = HintGenerator(llm, db).generate(
        decision(2, target=2), match, match.reference, rec, []
    )

    assert match.tier is Tier.EXACT
    assert str(served) == line
    assert [b.expr for b in served.board] == ["y - y_1 = -2*(x - x_1)"]
    assert llm.calls == []


def test_problem_10_curated_l1_l2_hints_survive_presolve_and_all_serve(db):
    """The demo problem has a complete, deterministic L1/L2 ladder.

    Every source line must pass the same future-step and answer-leak screens as
    a live hint; otherwise the demo silently falls through to generic pedagogy.
    """
    from tutor.knowledge.matching import Matcher
    from tutor.scripts.warm_kb import load_presolve, presolve
    from tutor.store.session_store import HintRecord

    entries = load_presolve()
    spec = entries["10"]
    source_hints = spec["prewritten_hints"]

    assert len(source_hints) == 12
    assert {(h["step"], h["level"]) for h in source_hints} == {
        (step, level) for step in range(1, 7) for level in (1, 2)
    }
    assert presolve(db, None, "10", entries)

    rec = Recognition(
        problem_text=spec["problem_text"],
        equations=spec["equations"][0],
        problem_type=spec["problem_type"],
        concepts=spec["concepts"],
    )
    match = Matcher(db).match(rec)
    llm = EchoLLMClient()
    generator = HintGenerator(llm, db)

    assert match.tier is Tier.EXACT
    for source in source_hints:
        served = generator.generate(
            decision(source["level"], target=source["step"]),
            match,
            match.reference,
            rec,
            [],
        )
        assert str(served) == source["text"], (source["step"], source["level"])

    first_l1 = next(h for h in source_hints if (h["step"], h["level"]) == (1, 1))
    first_l2 = next(h for h in source_hints if (h["step"], h["level"]) == (1, 2))
    after_surrender = generator.generate(
        decision(2, target=1),
        match,
        match.reference,
        rec,
        [HintRecord(
            id=1,
            problem_hash="problem-10",
            step=1,
            level=1,
            action="SOCRATIC_QUESTION",
            hint_text=first_l1["text"],
            effective=False,
        )],
        student_answer="잘 모르겠어요",
    )
    assert str(after_surrender) == first_l2["text"]
    assert llm.calls == []


def test_problem_13_curated_first_l1_survives_presolve(db):
    from tutor.knowledge.matching import Matcher
    from tutor.scripts.warm_kb import load_presolve, presolve

    entries = load_presolve()
    spec = entries["13"]
    source_hint = next(
        h for h in spec["prewritten_hints"]
        if h["step"] == 1 and h["level"] == 1
    )

    assert presolve(db, None, "13", entries)
    assert db.prewritten_hint("presolved-13", 1, 1) == source_hint["text"]
    assert db.prewritten_hint("presolved-13-v1", 1, 1) == source_hint["text"]

    rec = Recognition(
        problem_text=spec["problem_text"],
        equations=spec["equations"][0],
        problem_type=spec["problem_type"],
        concepts=spec["concepts"],
    )
    match = Matcher(db).match(rec)
    llm = EchoLLMClient()
    served = HintGenerator(llm, db).generate(
        decision(1, target=1), match, match.reference, rec, []
    )

    assert str(served) == source_hint["text"]
    assert "어떤 값을 먼저 계산" in served
    assert "f'(x)로" not in served
    assert llm.calls == []


def test_problem_13_surrender_after_derivative_gets_targeted_l2(db):
    """Once f'(x) is established, surrendering on the slope question must not
    fall back to a broad derivative-applications template that asks the student
    to differentiate all over again.
    """
    from tutor.knowledge.matching import Matcher
    from tutor.scripts.warm_kb import load_presolve, presolve
    from tutor.store.session_store import HintRecord

    entries = load_presolve()
    spec = entries["13"]
    source_hint = next(
        h for h in spec["prewritten_hints"]
        if h["step"] == 1 and h["level"] == 2
    )

    assert presolve(db, None, "13", entries)
    rec = Recognition(
        problem_text=spec["problem_text"],
        equations=spec["equations"][0],
        problem_type=spec["problem_type"],
        concepts=spec["concepts"],
    )
    match = Matcher(db).match(rec)
    failed = [HintRecord(
        id=1,
        problem_hash="problem-13",
        step=1,
        level=1,
        action="SOCRATIC_QUESTION",
        hint_text="이제 구한 f'(x)를 이용해 접선 l의 기울기를 구해 볼까요?",
        effective=False,
    )]
    llm = EchoLLMClient()

    served = HintGenerator(llm, db).generate(
        Decision(Action.CONCEPT_HINT, 2, 1, None, "hint L1 ineffective: escalate to L2"),
        match,
        match.reference,
        rec,
        failed,
        student_answer="모르겠는데",
    )

    assert str(served) == source_hint["text"]
    assert "구한 f'(x)에서 x에 1을 넣으면" in served
    assert "무엇을 미분" not in served
    assert llm.calls == []


def test_problem_13_uses_socratic_transitions_and_an_intercept_step(db):
    from tutor.knowledge.matching import Matcher
    from tutor.scripts.warm_kb import load_presolve, presolve

    entries = load_presolve()
    spec = entries["13"]
    assert len(spec["steps"]) == 8
    assert spec["steps"][6]["description"] == "두 직선 l, m의 y절편 구하기"
    assert "넓이" in spec["steps"][7]["description"]

    curated = {
        (hint["step"], hint["level"]): hint["text"]
        for hint in spec["prewritten_hints"]
    }
    assert "어떤 값을 대입" in curated[(4, 1)]
    assert "g'(1)을 계산해 볼까요" not in curated[(4, 1)]
    assert "y축과 만나는 위치" in curated[(7, 1)]
    assert "밑변과 높이" in curated[(8, 1)]

    assert presolve(db, None, "13", entries)
    rec = Recognition(
        problem_text=spec["problem_text"], equations=spec["equations"][0],
        problem_type=spec["problem_type"], concepts=spec["concepts"],
    )
    match = Matcher(db).match(rec)
    for target in (4, 7, 8):
        llm = EchoLLMClient()
        served = HintGenerator(llm, db).generate(
            decision(1, target=target), match, match.reference, rec, []
        )
        assert str(served) == curated[(target, 1)]
        assert llm.calls == []


def test_later_step_never_uses_a_broad_concept_template(db):
    from tutor.knowledge.models import HintTemplate

    db.insert_hint_template(HintTemplate(
        id="broad-l2",
        concept_id="step_sensitive_concept",
        misconception_id=None,
        level=2,
        template_text="처음 사용한 개념부터 다시 떠올려 볼까요?",
    ))
    reference = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="기울기 구하기", expression="m = -2"),
            SolutionStep(idx=2, description="직선의 방정식 쓰기", expression="y = -2*x - 4"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["step_sensitive_concept"],
        verified=True,
        origin="db",
    )
    llm = EchoLLMClient({"phrase": [{
        "hint": "점과 기울기를 점-기울기 형태에 넣어 볼까요?"
    }]})
    rec = Recognition(
        problem_text="직선의 방정식을 구하시오",
        concepts=["step_sensitive_concept"],
    )
    match = MatchResult(
        tier=Tier.CONCEPT,
        concepts=["step_sensitive_concept"],
        reference=reference,
    )

    text = HintGenerator(llm, db).generate(
        decision(2, target=2), match, reference, rec, []
    )

    assert str(text) == "점과 기울기를 점-기울기 형태에 넣어 볼까요?"
    assert "처음 사용한 개념" not in text
    assert llm.calls == ["phrase"]


class TestPrewriteScreensItsOwnPen:
    """prewrite() = the live phrasing path with the clock removed: same
    screens, one retry, and None rather than a line that fails them."""

    def gear(self, db, phrases):
        from tutor.knowledge.models import Problem
        problem = Problem(
            id="pw-2", problem_type="derivative_applications",
            problem_text="접선 문제", equations=[],
            answer=Answer(kind="SCALAR", value="1"),
            concepts=["differentiation"], verified=True,
        )
        reference = ReferenceSolution(
            steps=[SolutionStep(idx=1, description="기울기 구하기",
                                expression="f'(1) = -2")],
            final_answer=Answer(kind="SCALAR", value="1"),
            concepts=["differentiation"], verified=True, origin="db",
        )
        llm = EchoLLMClient({"phrase": phrases})
        gen = HintGenerator(llm, db)
        rec = Recognition(problem_text="접선 문제", equations=[], confidence=0.95)
        return gen, problem, reference, rec

    def test_a_clean_line_is_returned(self, db):
        gen, problem, reference, rec = self.gear(
            db, [{"hint": "기울기라는 말이 문제 어디에 숨어 있을까요?"}]
        )
        line = gen.prewrite(problem=problem, reference=reference, rec=rec,
                            step_idx=1, level=1)
        assert line == "기울기라는 말이 문제 어디에 숨어 있을까요?"

    def test_an_announcing_l1_is_retried_then_dropped(self, db):
        gen, problem, reference, rec = self.gear(
            db, [{"hint": "기울기 구하기 차례예요."},
                 {"hint": "다음 단계는 기울기 구하기예요."}]
        )
        line = gen.prewrite(problem=problem, reference=reference, rec=rec,
                            step_idx=1, level=1)
        assert line is None


class TestTheTargetsOwnWordsAreNotTheFuture:
    """Sibling steps rhyme: l의 방정식 (step 2) and m의 방정식 (step 5) share
    방정식, 지나 and both digits. The words that are ALSO the target's own
    must not count as a future-step mention — counting them silenced the
    correct step-2 question and left the slot to step-blind boilerplate."""

    REF = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="f'(x)로 접선 l의 기울기 구하기",
                         expression="f'(x) = 2*x - 4, f'(1) = -2"),
            SolutionStep(idx=2, description="점 (1, -6)을 지나는 l의 방정식 쓰기",
                         expression="l: y = -2*x - 4"),
            SolutionStep(idx=3, description="곱의 미분법으로 g'(x) 쓰기",
                         expression="g'(x) = (3*x**2 - 2)*f(x) + (x**3 - 2*x)*f'(x)"),
            SolutionStep(idx=4, description="g'(1) 계산", expression="g'(1) = -4"),
            SolutionStep(idx=5, description="점 (1, 6)을 지나는 m의 방정식 쓰기",
                         expression="m: y = -4*x + 10"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation"], verified=True, origin="db",
    )

    def test_the_step_two_question_is_not_a_mention_of_step_five(self):
        assert not mentions_future_step(
            "이제 점 (1, -6)을 지나는 l의 방정식을 어떻게 쓰면 좋을까요?",
            self.REF, 2,
        )

    def test_a_step_one_hint_asking_for_the_equation_is_still_caught(self):
        # the original live failure this guard exists for, unchanged
        assert mentions_future_step(
            "우선 구하신 기울기 마이너스 2와 점 (1, -6)을 사용해서 "
            "접선 l의 방정식을 어떻게 나타낼 수 있을까요?",
            self.REF, 1,
        )


class TestACorrectingTurnIsNeverCanned:
    """The last hint at this step did not help: what follows must engage with
    THEIR work, which no prewritten or concept line can do. Live, the
    prewritten L2 served here and '다시 살펴보세요' only ever happened at L3 —
    the one level nothing was prewritten for."""

    def test_the_prewritten_line_yields_to_the_model(self, db):
        from tutor.store.session_store import HintRecord
        helper = TestAPrewrittenLineServesFirst()
        gen, llm, rec, match = helper.ready(db)
        failed = [HintRecord(id=1, problem_hash="h", step=1, level=1,
                             action="SOCRATIC_QUESTION", hint_text="첫 질문",
                             effective=False)]
        text = gen.generate(helper.decision(), match, None, rec, history=failed)
        assert str(text) != helper.LINE
        assert "phrase" in llm.calls

    def test_the_model_is_told_to_point_at_the_spoken_error(self, db):
        from tutor.store.session_store import HintRecord

        class Recording(EchoLLMClient):
            def __init__(self):
                super().__init__()
                self.prompt = ""

            def run_with_tools(
                self, *, purpose, system, user, images=(), schema, max_rounds=6
            ):
                if purpose == "phrase":
                    self.prompt = user
                return super().run_with_tools(
                    purpose=purpose, system=system, user=user, images=images,
                    schema=schema, max_rounds=max_rounds,
                )

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
            ],
            final_answer=Answer(kind="SCALAR", value="49"),
            concepts=["differentiation"],
            verified=True,
            origin="db",
        )
        failed = [HintRecord(
            id=1, problem_hash="h", step=2, level=2,
            action="CONCEPT_HINT",
            hint_text="기울기가 -2인 직선의 꼴에 접점을 넣어 볼까요?",
            effective=False,
        )]
        llm = Recording()
        HintGenerator(llm, db).generate(
            decision(3, target=2),
            MatchResult(tier=Tier.NEW, concepts=["differentiation"], reference=reference),
            reference,
            Recognition(problem_text="접선 문제"),
            failed,
            student_answer="그럼 2x 마이너스 4 맞아?",
            answer_error="2x-4를 접선의 방정식으로 혼동함; 앞에서 구한 f'(x)임",
        )

        assert "오답 뒤 재검토" in llm.prompt
        assert "답변 판정기가 특정한 오류" in llm.prompt
        assert "2x-4를 접선의 방정식으로 혼동" in llm.prompt
        assert "이미 완료되어 비교 근거로 써도 되는 단계" in llm.prompt
        assert "f'(x) = 2*x - 4, f'(1) = -2" in llm.prompt
        assert "새 풀이 절차를 처음부터 이어 설명하지 마세요" in llm.prompt


class TestABlockedStreamTrailsOffOnPurpose:
    """Live at L2: "기울기가 k이고 한 점을 지나는 직선은 y … 여기서 답은
    말하지 않을게요. 지금까지 한 풀이에서 가장 확실한 줄은 어디인가요?" — an
    amputated sentence stitched to a confession that changed the subject.
    The closure now HANDS THE SENTENCE OVER instead: trailing off into
    "직접 이어서 완성해 볼까요?" is a cloze prompt, not an apology."""

    def test_a_dangling_sentence_becomes_a_cloze(self):
        out = []
        emitter = SafeWordEmitter(out.append, LIN_REF, 1, [])
        emitter.feed("기울기가 k이고 한 점을 지나는 직선은 y 는 그래서 정답은 5 ")
        text, blocked = emitter.finish("기울기가 k이고 한 점을 지나는 직선은 y 는 그래서 정답은 5")
        assert blocked
        assert text.endswith("이 다음은 직접 이어서 완성해 볼까요?")
        assert " … " in text                     # the trail-off is audible
        assert "말하지 않을게요" not in text       # no confession

    def test_a_stream_blocked_at_the_door_opens_fresh(self):
        out = []
        emitter = SafeWordEmitter(out.append, LIN_REF, 1, [])
        text, blocked = emitter.finish("정답은 5")   # nothing ever released
        assert blocked
        assert text == SafeWordEmitter.FRESH
        assert "…" not in text                   # nothing to trail off FROM


class TestASurrenderGetsTheWellMadeLine:
    """"잘 모르겠는데" resolves the last hint as ineffective — but it leaves
    no error to point at, so the correcting detour does not apply: the
    prewritten concept line for the step serves, instead of a live model
    aimed at a mistake that does not exist."""

    L2 = "지나는 한 점의 좌표와 그 점에서의 기울기를 알 때, 접선의 방정식을 만드는 원리를 떠올려 볼까요?"

    def ready(self, db):
        from tutor.knowledge.models import Problem
        problem = Problem(
            id="pw-s", problem_type="derivative_applications",
            problem_text="접선 문제", equations=[],
            answer=Answer(kind="SCALAR", value="1"),
            concepts=["differentiation"], verified=True,
        )
        db.save_prewritten_hint("pw-s", 1, 2, self.L2)
        llm = EchoLLMClient()
        gen = HintGenerator(llm, db)
        rec = Recognition(problem_text="접선 문제", equations=[],
                          concepts=["differentiation"], confidence=0.95)
        match = MatchResult(tier=Tier.EXACT, concepts=["differentiation"], problem=problem)
        failed = [__import__("tutor.store.session_store", fromlist=["HintRecord"]).HintRecord(
            id=1, problem_hash="h", step=1, level=1,
            action="SOCRATIC_QUESTION", hint_text="첫 질문", effective=False)]
        return gen, llm, rec, match, failed

    def test_surrender_escalation_serves_the_prewritten_l2(self, db):
        gen, llm, rec, match, failed = self.ready(db)
        text = gen.generate(
            Decision(Action.CONCEPT_HINT, 2, 1, None, "escalate"),
            match, None, rec, history=failed, student_answer="잘 모르겠는데",
        )
        assert str(text) == self.L2
        assert "phrase" not in llm.calls          # no live model needed

    def test_a_wrong_attempt_still_goes_to_the_model(self, db):
        gen, llm, rec, match, failed = self.ready(db)
        text = gen.generate(
            Decision(Action.CONCEPT_HINT, 2, 1, None, "escalate"),
            match, None, rec, history=failed, student_answer="기울기는 5 아니에요?",
        )
        assert str(text) != self.L2
        assert "phrase" in llm.calls              # an attempt: point at it


class TestTheBoardCanQuoteWhatWasEarned:
    """Live on problem 13 step 6: asked to intersect the two tangents, the
    board read "y_1 = y_2". The model was never told what l and m ARE — only
    the NAME of the step it was aiming at — so a placeholder was the most it
    could write. Every phrasing turn now carries the established results, and
    the guard already permits them: they sit at or below the target."""

    REF = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="l의 방정식 쓰기",
                         expression="l: y = -2*(x - 1) - 6 = -2*x - 4"),
            SolutionStep(idx=2, description="m의 방정식 쓰기",
                         expression="m: y = -4*(x - 1) + 6 = -4*x + 10"),
            SolutionStep(idx=3, description="두 직선의 교점의 x좌표 구하기",
                         expression="-2*x - 4 = -4*x + 10, x = 7"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["derivative_applications"], verified=True, origin="db",
    )

    def prompt_at(self, db, target):
        llm = EchoLLMClient({"phrase": [{"hint": "이제 두 식을 같게 놓아 볼까요?"}]})
        HintGenerator(llm, db).generate(
            # a diagnosed mistake is what sends a turn to the phrasing model
            Decision(Action.SOCRATIC_QUESTION, 1, target, "force_llm", "t"),
            MatchResult(tier=Tier.CONCEPT, concepts=["derivative_applications"],
                        reference=self.REF),
            self.REF,
            Recognition(problem_text="두 접선 문제",
                        concepts=["derivative_applications"], confidence=0.95),
            history=[],
        )
        return llm.prompts_for("phrase")[0]

    def test_the_earned_equations_reach_the_model(self, db):
        prompt = self.prompt_at(db, target=3)
        assert "-2*x - 4" in prompt and "-4*x + 10" in prompt

    def test_the_first_step_is_told_nothing_is_established(self, db):
        prompt = self.prompt_at(db, target=1)
        assert "아직 아무것도 구해지지" in prompt
        assert "-2*x - 4" not in prompt

    def test_the_board_line_the_model_can_now_write_passes_the_guard(self, db):
        assert not leaks_answer("-2*x - 4 = -4*x + 10", self.REF, 3, set())


class TestTheStepsOwnResultIsNotAHint:
    """Live, correcting g'(x): "첫 번째 괄호를 3 x 제곱 빼기 2로 고쳐서 g′(x)를
    다시 정리해 볼까요?" — with the board writing (x^3-2x)' = 3x^2-2 beside it.
    That IS the step. The answer-leak guard never saw it: it screens the final
    answer and steps BEYOND the target, and the target's own result was left
    sayable because L4 exists to say it. Everything below L4 must not."""

    REF = ReferenceSolution(
        steps=[
            SolutionStep(idx=1, description="f'(x)", expression="f'(x) = 2*x - 4"),
            SolutionStep(idx=2, description="곱의 미분법으로 g'(x) 쓰기",
                         expression="g'(x) = (3*x**2 - 2)*f(x) + (x**3 - 2*x)*f'(x)"),
        ],
        final_answer=Answer(kind="SCALAR", value="49"),
        concepts=["differentiation"], verified=True, origin="db",
    )
    SEEN = ["13. 함수", "f(x) = x**2 - 4*x - 3", "g(x) = (x**3 - 2*x)*f(x)"]
    LEAK = ("작성하신 식에서 앞부분 x 세제곱 빼기 2x를 미분할 때, 마이너스 2x의 "
            "미분은 마이너스 2가 돼요. 첫 번째 괄호를 3 x 제곱 빼기 2로 고쳐서 "
            "g′(x)를 다시 정리해 볼까요?")

    def test_the_spoken_result_is_caught(self):
        from tutor.hints.guard import states_step_result
        assert states_step_result(self.LEAK, self.REF, 2, self.SEEN)

    def test_the_written_result_is_caught(self):
        from tutor.hints.guard import states_step_result
        assert states_step_result("(x**3 - 2*x)' = 3*x**2 - 2", self.REF, 2, self.SEEN)

    @pytest.mark.parametrize("hint", [
        "이제 g(x)가 두 식의 곱이라는 점을 보고, 어떻게 미분하면 좋을까요?",
        "x 세제곱 빼기 2x 부분을 다시 볼까요?",          # the problem's own, quotable
        "두 번째 줄에서 부호가 어떻게 됐어요?",
    ])
    def test_a_real_hint_still_passes(self, hint):
        from tutor.hints.guard import states_step_result
        assert not states_step_result(hint, self.REF, 2, self.SEEN)

    def rec(self):
        return Recognition(
            problem_text="13. 두 접선",
            equations=["f(x) = x**2 - 4*x - 3", "g(x) = (x**3 - 2*x)*f(x)"],
            concepts=["differentiation"], confidence=0.95,
        )

    def generated(self, db, level):
        llm = EchoLLMClient({"phrase": [
            {"hint": self.LEAK,
             "board": [{"expr": "(x**3 - 2*x)' = 3*x**2 - 2", "note": ""}]},
            {"hint": self.LEAK, "board": []},
        ]})
        action = Action.PARTIAL_STEP if level >= 4 else Action.CONCEPT_HINT
        return HintGenerator(llm, db).generate(
            Decision(action, level, 2, "2x의 미분을 2x로 계산", "t"),
            MatchResult(tier=Tier.CONCEPT, concepts=["differentiation"],
                        reference=self.REF),
            self.REF, self.rec(), history=[],
        )

    def test_below_l4_the_leak_never_reaches_the_student(self, db):
        out = self.generated(db, level=2)
        assert "3 x 제곱 빼기 2" not in str(out)
        assert getattr(out, "board", ()) == ()

    def test_l4_is_still_allowed_to_say_its_step(self, db):
        out = self.generated(db, level=4)
        assert "3 x 제곱 빼기 2" in str(out)

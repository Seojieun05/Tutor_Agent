"""ConceptTagger: two whitelisted layers, enforced in Python.

problem_type = one coarse type; concepts = 0-4 curriculum ideas needed to
solve. Whatever the model returns, nothing off the whitelist survives.
"""

import pytest

from tutor.knowledge.concepts import (
    ALLOWED_CONCEPT_IDS,
    MAX_CONCEPTS,
    concepts_for_prompt,
    normalize_concepts,
)
from tutor.knowledge.tagger import ConceptTagger, ProblemTags
from tutor.knowledge.taxonomy import (
    ALLOWED_PROBLEM_TYPES,
    normalize_problem_type,
    problem_types_for_prompt,
)
from tutor.llm.echo import EchoLLMClient
from tutor.vision.recognizer import Recognition

COUNTING = Recognition(
    problem_text="GOOD와 2002의 문자와 숫자를 번갈아 오도록 배열하는 경우의 수를 구하시오.",
    student_work=["4! = 24"],
)


class TestWhitelists:
    def test_the_two_layers_are_disjoint_in_purpose(self):
        # the example from the spec: counting is a TYPE, permutation a CONCEPT
        assert "counting" in ALLOWED_PROBLEM_TYPES
        assert "permutation" in ALLOWED_CONCEPT_IDS
        assert "permutation_with_identical_elements" in ALLOWED_CONCEPT_IDS

    def test_strategies_are_not_concepts(self):
        for strategy in ("alternating_arrangement", "case_split", "work_backwards"):
            assert strategy not in ALLOWED_CONCEPT_IDS
            assert strategy not in ALLOWED_PROBLEM_TYPES

    def test_unknown_is_always_available(self):
        assert normalize_problem_type("nonsense_type") == "unknown"
        assert normalize_problem_type(None) == "unknown"
        assert normalize_problem_type("") == "unknown"

    def test_normalize_concepts_drops_inventions_and_caps(self):
        assert normalize_concepts(["permutation", "made_up", "combination"]) == [
            "permutation",
            "combination",
        ]
        assert normalize_concepts(["permutation", "permutation"]) == ["permutation"]
        assert len(normalize_concepts(sorted(ALLOWED_CONCEPT_IDS))) == MAX_CONCEPTS

    def test_prompts_list_every_allowed_id(self):
        types_prompt = problem_types_for_prompt()
        assert all(t in types_prompt for t in ALLOWED_PROBLEM_TYPES)
        concepts_prompt = concepts_for_prompt()
        assert all(c in concepts_prompt for c in ALLOWED_CONCEPT_IDS)


class TestTagger:
    def test_valid_tags_pass_through(self):
        llm = EchoLLMClient(
            {"tag": [{"problem_type": "counting",
                      "concepts": ["permutation", "permutation_with_identical_elements"]}]}
        )
        tags = ConceptTagger(llm).tag(COUNTING)
        assert tags.problem_type == "counting"
        assert tags.concepts == ["permutation", "permutation_with_identical_elements"]
        assert llm.calls == ["tag"]

    def test_invented_ids_are_dropped_not_stored(self):
        llm = EchoLLMClient(
            {"tag": [{"problem_type": "arrangement_problem",
                      "concepts": ["permutation", "alternating_arrangement", "case_split"]}]}
        )
        tags = ConceptTagger(llm).tag(COUNTING)
        assert tags.problem_type == "unknown"  # not on the type whitelist
        assert tags.concepts == ["permutation"]  # strategies dropped

    def test_too_many_concepts_are_capped(self):
        llm = EchoLLMClient(
            {"tag": [{"problem_type": "counting",
                      "concepts": ["permutation", "combination", "factorial",
                                   "probability", "linear_equation", "geometry"]}]}
        )
        tags = ConceptTagger(llm).tag(COUNTING)
        assert len(tags.concepts) <= MAX_CONCEPTS

    def test_empty_classification_is_allowed(self):
        llm = EchoLLMClient({"tag": [{"problem_type": "unknown", "concepts": []}]})
        tags = ConceptTagger(llm).tag(COUNTING)
        assert (tags.problem_type, tags.concepts) == ("unknown", [])

    def test_validate_is_pure_and_reusable(self):
        raw = ProblemTags(problem_type="counting", concepts=["permutation", "nope"])
        assert ConceptTagger.validate(raw).concepts == ["permutation"]

    def test_tagger_never_sees_student_work(self):
        """Tags describe the problem: student writing must not move them."""
        context = ConceptTagger._context(COUNTING)
        assert "GOOD" in context
        assert "4! = 24" not in context


class TestTaggingInTheSession:
    """Where the tagger sits in the pipeline: once per problem, before matching."""

    @staticmethod
    def _session(db, llm):
        from tutor.config import Settings
        from tutor.hints.generator import HintGenerator
        from tutor.knowledge.matching import Matcher
        from tutor.server.session import Deps, Session
        from tutor.solver.grok_solver import GrokSolver
        from tutor.speech.stt import EchoTranscriber
        from tutor.speech.tts import NullSpeaker
        from tutor.state.estimator import StudentStateEstimator
        from tutor.store.session_store import SessionStore
        from tutor.vision.recognizer import Recognizer

        deps = Deps(
            settings=Settings(),
            recognizer=Recognizer(llm),
            matcher=Matcher(db),
            solver=GrokSolver(llm, db),
            estimator=StudentStateEstimator(llm, db),
            hint_gen=HintGenerator(llm, db),
            transcriber=EchoTranscriber(),
            speaker=NullSpeaker(),
            tagger=ConceptTagger(llm),
            store=SessionStore(),
        )
        return Session(object(), deps), llm

    async def test_tagged_once_then_carried_across_captures(self, db):
        llm = EchoLLMClient(
            {"tag": [{"problem_type": "linear_equation", "concepts": ["linear_equation"]}]}
        )
        session, llm = self._session(db, llm)

        first = Recognition(problem_text="3x + 5 = 20을 푸시오", equations=["3*x + 5 = 20"])
        await session._problem_context(first)
        assert (first.problem_type, first.concepts) == ("linear_equation", ["linear_equation"])

        # same problem, the student wrote another line
        second = Recognition(
            problem_text="3x + 5 = 20을 푸시오",
            equations=["3*x + 5 = 20"],
            student_work=["3*x = 15"],
        )
        await session._problem_context(second)

        assert llm.calls.count("tag") == 1  # cached: no LLM call per hint
        assert second.problem_type == "linear_equation"  # tags carried over
        assert second.concepts == ["linear_equation"]

    async def test_a_different_problem_is_tagged_again(self, db):
        llm = EchoLLMClient(
            {
                "tag": [
                    {"problem_type": "linear_equation", "concepts": ["linear_equation"]},
                    {"problem_type": "counting", "concepts": ["permutation"]},
                ]
            }
        )
        session, llm = self._session(db, llm)
        await session._problem_context(
            Recognition(problem_text="3x + 5 = 20", equations=["3*x + 5 = 20"])
        )
        other = Recognition(problem_text="배열하는 경우의 수를 구하시오")
        await session._problem_context(other)

        assert llm.calls.count("tag") == 2
        assert other.problem_type == "counting"

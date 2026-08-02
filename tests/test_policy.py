"""The R1-R10 rule table, parametrized, plus invariants."""

import pytest

from tutor.policy.engine import Action, decide
from tutor.state.models import StudentState
from tutor.store.session_store import HintRecord


def state(**kw) -> StudentState:
    return StudentState(**{"status": "CONCEPT_ERROR", "last_correct_step": 0, **kw})


def hint(step=1, level=1, action="SOCRATIC_QUESTION", effective=None, hint_id=1):
    return HintRecord(
        id=hint_id, step=step, level=level, action=action, hint_text="…", effective=effective
    )


class TestRules:
    def test_r10_recognition_failed(self):
        d = decide(state(), [], "RECOGNITION_FAILED")
        assert d.action == Action.ASK_RECAPTURE

    def test_r1_uncertain_recapture_first(self):
        d = decide(state(status="UNCERTAIN"), [], "HINT_REQUEST")
        assert d.action == Action.ASK_RECAPTURE

    def test_r2_uncertain_probe_after_recapture(self):
        history = [hint(level=0, action="ASK_RECAPTURE")]
        d = decide(state(status="UNCERTAIN"), history, "HINT_REQUEST")
        assert d.action == Action.PROBE

    @pytest.mark.parametrize("status", ["CORRECT", "CONCEPT_ERROR", "STUCK"])
    def test_r3_r4_state_update_never_interrupts(self, status):
        d = decide(state(status=status), [], "STATE_UPDATE")
        assert d.action == Action.WAIT
        assert d.level == 0

    def test_r5_correct_student_asking_gets_l1(self):
        d = decide(state(status="CORRECT", last_correct_step=1), [], "HINT_REQUEST")
        assert d.action == Action.SOCRATIC_QUESTION
        assert d.level == 1
        assert d.target_step == 2

    def test_r6_r7_first_hint_is_l1(self):
        d = decide(state(status="STUCK"), [], "HINT_REQUEST")
        assert (d.level, d.action) == (1, Action.SOCRATIC_QUESTION)
        assert d.target_step == 1

    def test_r8_escalate_one_level_after_ineffective(self):
        history = [hint(level=1, effective=False)]
        d = decide(state(), history, "HINT_REQUEST")
        assert (d.level, d.action) == (2, Action.CONCEPT_HINT)

    def test_r8_cap_at_l4(self):
        history = [
            hint(level=1, effective=False, hint_id=1),
            hint(level=2, action="CONCEPT_HINT", effective=False, hint_id=2),
            hint(level=3, action="PROCEDURAL_HINT", effective=False, hint_id=3),
            hint(level=4, action="PARTIAL_STEP", effective=False, hint_id=4),
        ]
        d = decide(state(), history, "HINT_REQUEST")
        assert (d.level, d.action) == (4, Action.PARTIAL_STEP)

    def test_r9_effective_hint_repeats_level(self):
        history = [hint(level=2, action="CONCEPT_HINT", effective=True)]
        d = decide(state(), history, "HINT_REQUEST")
        assert (d.level, d.action) == (2, Action.CONCEPT_HINT)

    def test_unresolved_effective_none_does_not_escalate(self):
        history = [hint(level=1, effective=None)]
        d = decide(state(), history, "HINT_REQUEST")
        assert d.level == 1


class TestInvariants:
    def test_never_skip_a_level(self):
        # even after an L1 failure the next is exactly L2, not L3/L4
        history = [hint(level=1, effective=False)]
        assert decide(state(), history, "HINT_REQUEST").level == 2

    def test_l4_needs_three_failures(self):
        history = []
        levels = []
        for i in range(5):
            d = decide(state(), history, "HINT_REQUEST")
            levels.append(d.level)
            history.append(
                hint(level=d.level, action=d.action.value, effective=False, hint_id=i + 1)
            )
        assert levels == [1, 2, 3, 4, 4]

    def test_fading_resets_to_l1_on_progress(self):
        history = [hint(step=1, level=3, action="PROCEDURAL_HINT", effective=True)]
        d = decide(state(last_correct_step=1), history, "HINT_REQUEST")
        assert d.target_step == 2
        assert d.level == 1

    def test_no_action_reveals_answer(self):
        assert not hasattr(Action, "GIVE_ANSWER")
        for level in range(5):
            from tutor.policy.engine import LEVEL_ACTIONS

            assert LEVEL_ACTIONS[level].value in {
                "WAIT",
                "SOCRATIC_QUESTION",
                "CONCEPT_HINT",
                "PROCEDURAL_HINT",
                "PARTIAL_STEP",
            }

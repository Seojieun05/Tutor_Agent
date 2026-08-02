import pytest

from tutor.state.models import StudentState
from tutor.store.session_store import SessionStore


def test_state_round_trip():
    store = SessionStore()
    assert store.get_state() is None
    store.set_state(StudentState(status="STUCK"))
    assert store.get_state().status == "STUCK"
    store.clear_state()
    assert store.get_state() is None


def test_hint_lifecycle():
    store = SessionStore()
    hint_id = store.append_hint(
        problem_hash="pA", step=1, level=1, action="SOCRATIC_QUESTION", hint_text="힌트"
    )
    assert store.get_history()[0].effective is None
    assert store.pending_hint("pA").id == hint_id
    store.mark_hint_effective(hint_id, False)
    assert store.get_history()[0].effective is False
    assert store.pending_hint("pA") is None


def test_pending_ignores_level0_actions():
    store = SessionStore()
    store.append_hint(
        problem_hash="pA", step=1, level=0, action="ASK_RECAPTURE", hint_text="다시 보여줘요"
    )
    assert store.pending_hint("pA") is None


def test_history_scoped_by_problem():
    """Hints for problem A must never mix into problem B's history."""
    store = SessionStore()
    store.append_hint(problem_hash="pA", step=1, level=1, action="SOCRATIC_QUESTION", hint_text="a")
    store.append_hint(problem_hash="pA", step=1, level=2, action="CONCEPT_HINT", hint_text="b")
    store.append_hint(problem_hash="pB", step=1, level=1, action="SOCRATIC_QUESTION", hint_text="c")
    assert [h.level for h in store.get_history(problem_hash="pA")] == [1, 2]
    assert [h.level for h in store.get_history(problem_hash="pB")] == [1]
    assert [h.step for h in store.get_history(step=1, problem_hash="pB")] == [1]
    # a pending hint on A is invisible from B
    assert store.pending_hint("pB").hint_text == "c"
    store.mark_hint_effective(store.pending_hint("pB").id, True)
    assert store.pending_hint("pB") is None
    assert store.pending_hint("pA") is not None


def test_mark_unknown_id():
    with pytest.raises(KeyError):
        SessionStore().mark_hint_effective(99, True)

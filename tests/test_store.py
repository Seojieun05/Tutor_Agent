import pytest

from tutor.state.models import StudentState
from tutor.store.session_store import SessionStore


def test_state_round_trip():
    store = SessionStore()
    assert store.get_state() is None
    store.set_state(StudentState(status="STUCK"))
    assert store.get_state().status == "STUCK"


def test_hint_lifecycle():
    store = SessionStore()
    hint_id = store.append_hint(step=1, level=1, action="SOCRATIC_QUESTION", hint_text="힌트")
    assert store.get_history()[0].effective is None
    assert store.pending_hint().id == hint_id
    store.mark_hint_effective(hint_id, False)
    assert store.get_history()[0].effective is False
    assert store.pending_hint() is None


def test_pending_ignores_level0_actions():
    store = SessionStore()
    store.append_hint(step=1, level=0, action="ASK_RECAPTURE", hint_text="다시 보여줘요")
    assert store.pending_hint() is None


def test_history_filter_by_step():
    store = SessionStore()
    store.append_hint(step=1, level=1, action="SOCRATIC_QUESTION", hint_text="a")
    store.append_hint(step=2, level=1, action="SOCRATIC_QUESTION", hint_text="b")
    assert [h.step for h in store.get_history(step=2)] == [2]


def test_mark_unknown_id():
    with pytest.raises(KeyError):
        SessionStore().mark_hint_effective(99, True)

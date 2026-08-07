"""Turn taking with a scripted VAD: no microphone, no torch, deterministic.

Frames are 512 samples @16 kHz = 32 ms, so the ms budgets below convert as
ceil(ms / 32) frames.
"""

import math

import pytest

from tutor.speech.turn import TurnConfig, TurnDetector, TurnState, TurnTaker

CFG = TurnConfig()
FRAME_MS = CFG.frame_ms  # 32.0


class ScriptedVAD:
    """is_speech() replays a caller-supplied sequence of booleans."""

    def __init__(self):
        self.script: list[bool] = []
        self.resets = 0
        self.seen = 0

    def is_speech(self, frame, sample_rate=None) -> bool:
        self.seen += 1
        return self.script.pop(0) if self.script else False

    def reset(self) -> None:
        self.resets += 1


def frames(n: int) -> list[bytes]:
    return [bytes(CFG.frame_bytes) for _ in range(n)]


def ms_to_frames(ms: float) -> int:
    return math.ceil(ms / FRAME_MS)


def run(detector: TurnDetector, vad: ScriptedVAD, script: list[bool]) -> list[bytes]:
    """Feed one frame per scripted flag; return every utterance emitted."""
    out = []
    for flag in script:
        vad.script.append(flag)
        utterance = detector.feed(bytes(CFG.frame_bytes))
        if utterance is not None:
            out.append(utterance)
    return out


def speech(n: int) -> list[bool]:
    return [True] * n


def silence(n: int) -> list[bool]:
    return [False] * n


@pytest.fixture
def detector():
    vad = ScriptedVAD()
    return TurnDetector(vad=vad, config=TurnConfig()), vad


# --- endpointing ------------------------------------------------------------


def test_speech_then_silence_emits_one_utterance(detector):
    det, vad = detector
    out = run(det, vad, silence(3) + speech(30) + silence(ms_to_frames(CFG.silence_ms)))
    assert len(out) == 1
    assert det.speaking is False  # state machine rearmed for the next turn


def test_utterance_ends_only_after_the_full_silence_window(detector):
    det, vad = detector
    quiet = ms_to_frames(CFG.silence_ms) - 1
    assert run(det, vad, speech(30) + silence(quiet)) == []  # 800 ms not reached yet
    assert len(run(det, vad, silence(1))) == 1


def test_mid_sentence_pause_does_not_split_the_turn(detector):
    det, vad = detector
    pause = ms_to_frames(CFG.silence_ms) - 2  # a long "uhh" that is still one turn
    out = run(
        det,
        vad,
        speech(20) + silence(pause) + speech(20) + silence(ms_to_frames(CFG.silence_ms)),
    )
    assert len(out) == 1
    # everything from onset to endpoint is in one PCM blob, pause included
    assert len(out[0]) > (20 + pause + 20) * CFG.frame_bytes


def test_short_noise_is_discarded_not_sent(detector):
    det, vad = detector
    # onset fires (3 of 5), but total speech stays under min_speech_ms=250 ms
    burst = ms_to_frames(CFG.min_speech_ms) - 2
    out = run(det, vad, speech(burst) + silence(ms_to_frames(CFG.silence_ms)))
    assert out == []


def test_single_noisy_frame_never_opens_a_turn(detector):
    det, vad = detector
    run(det, vad, [True] + silence(6) + [True] + silence(6))
    assert det.speaking is False


# --- prefix padding ---------------------------------------------------------


def test_prefix_padding_keeps_audio_from_before_onset(detector):
    det, vad = detector
    onset = CFG.onset_frames
    out = run(det, vad, silence(20) + speech(30) + silence(ms_to_frames(CFG.silence_ms)))
    kept_frames = len(out[0]) // CFG.frame_bytes
    # onset is only declared on the Nth speech frame, so those N frames plus at
    # least prefix_ms of earlier audio must survive in front of the utterance
    assert kept_frames >= CFG.prefix_frames + (30 - onset)
    assert CFG.prefix_frames * CFG.frame_ms >= CFG.prefix_ms


def test_prefix_never_exceeds_its_budget(detector):
    det, vad = detector
    out = run(det, vad, silence(200) + speech(10) + silence(ms_to_frames(CFG.silence_ms)))
    kept_ms = len(out[0]) // CFG.frame_bytes * CFG.frame_ms
    spoken_ms = (10 + ms_to_frames(CFG.silence_ms)) * CFG.frame_ms
    assert kept_ms - spoken_ms <= CFG.prefix_ms + CFG.frame_ms


# --- guards -----------------------------------------------------------------


def test_endless_speech_is_force_committed(detector):
    _, vad = detector
    cfg = TurnConfig(max_utterance_ms=1000)
    det = TurnDetector(vad=vad, config=cfg)
    out = run(det, vad, speech(ms_to_frames(1200)))
    assert len(out) == 1  # committed at the cap instead of buffering forever
    assert len(out[0]) // cfg.frame_bytes * cfg.frame_ms <= cfg.max_utterance_ms + cfg.prefix_ms


def test_wrong_frame_size_is_rejected(detector):
    det, _ = detector
    with pytest.raises(ValueError):
        det.feed(bytes(CFG.frame_bytes - 2))


def test_new_turn_resets_the_stateful_vad(detector):
    det, vad = detector
    before = vad.resets
    run(det, vad, speech(30) + silence(ms_to_frames(CFG.silence_ms)))
    assert vad.resets > before


# --- the four-state conversation gate ---------------------------------------


@pytest.fixture
def taker():
    """The gate with barge-in OFF — the local-mic device, where the tutor's
    voice reaches the tutor's own microphone with nothing in between."""
    vad = ScriptedVAD()
    det = TurnDetector(vad=vad, config=TurnConfig(barge_in=False))
    seen: list[TurnState] = []
    return TurnTaker(det, on_change=seen.append), vad, seen


@pytest.fixture
def interruptible():
    """The gate with barge-in ON — the browser, where echo cancellation makes
    an open mic during playback survivable."""
    vad = ScriptedVAD()
    det = TurnDetector(vad=vad, config=TurnConfig(barge_in=True))
    seen: list[TurnState] = []
    cuts: list[int] = []
    return (
        TurnTaker(det, on_change=seen.append, on_barge_in=lambda: cuts.append(1)),
        vad, seen, cuts,
    )


def feed(taker: TurnTaker, vad: ScriptedVAD, script: list[bool], now_ms: float = 0.0):
    out = []
    for flag in script:
        vad.script.append(flag)
        utterance = taker.feed(bytes(CFG.frame_bytes), now_ms=now_ms)
        if utterance is not None:
            out.append(utterance)
    return out


def test_states_follow_a_full_turn(taker):
    tt, vad, seen = taker
    assert tt.state is TurnState.LISTENING

    feed(tt, vad, speech(10))
    assert tt.state is TurnState.USER_SPEAKING

    out = feed(tt, vad, speech(20) + silence(ms_to_frames(CFG.silence_ms)))
    assert len(out) == 1
    assert tt.state is TurnState.PROCESSING  # utterance handed to the pipeline

    tt.agent_speaking()
    assert tt.state is TurnState.AGENT_SPEAKING

    tt.agent_finished(now_ms=1000)
    assert tt.state is TurnState.AGENT_SPEAKING  # tail guard still running
    feed(tt, vad, silence(1), now_ms=1000 + CFG.tail_guard_ms)
    assert tt.state is TurnState.LISTENING

    assert seen[:3] == [
        TurnState.USER_SPEAKING,
        TurnState.PROCESSING,
        TurnState.AGENT_SPEAKING,
    ]


def test_vad_never_runs_while_the_agent_speaks_without_barge_in(taker):
    """BARGE_IN=0 restores the old guarantee, which the local device needs."""
    tt, vad, _ = taker
    tt.agent_speaking()
    seen_before = vad.seen
    assert feed(tt, vad, speech(60)) == []  # the tutor's own voice
    assert vad.seen == seen_before  # frames never reached the model
    assert tt.state is TurnState.AGENT_SPEAKING


def test_no_turn_survives_the_agent_response(taker):
    tt, vad, _ = taker
    feed(tt, vad, speech(10))  # half an utterance buffered
    assert tt.state is TurnState.USER_SPEAKING
    tt.agent_speaking()
    tt.agent_finished(now_ms=0)
    feed(tt, vad, silence(1), now_ms=CFG.tail_guard_ms)
    assert tt.state is TurnState.LISTENING
    assert tt.detector.speaking is False
    assert tt.detector._frames == []


def test_tail_guard_ignores_speech_until_it_expires(taker):
    tt, vad, _ = taker
    tt.agent_speaking()
    tt.agent_finished(now_ms=0)
    # room decay right after playback must not open a turn
    assert feed(tt, vad, speech(5), now_ms=CFG.tail_guard_ms - 1) == []
    assert tt.state is TurnState.AGENT_SPEAKING
    feed(tt, vad, silence(1), now_ms=CFG.tail_guard_ms)
    assert tt.state is TurnState.LISTENING


def test_agent_finished_outside_speaking_is_ignored(taker):
    tt, vad, _ = taker
    tt.processing()
    tt.agent_finished(now_ms=0)  # e.g. a stray idle event
    feed(tt, vad, silence(1), now_ms=10_000)
    assert tt.state is TurnState.PROCESSING  # still waiting for the tutor


def test_listen_recovers_a_turn_that_got_no_response(taker):
    tt, vad, _ = taker
    feed(tt, vad, speech(30) + silence(ms_to_frames(CFG.silence_ms)))
    assert tt.state is TurnState.PROCESSING
    tt.listen()  # what the device does on transcript(wants_hint=False) or timeout
    assert tt.state is TurnState.LISTENING
    out = feed(tt, vad, speech(30) + silence(ms_to_frames(CFG.silence_ms)))
    assert len(out) == 1  # and the next turn works normally


def test_multi_turn_conversation_needs_no_button(taker):
    tt, vad, _ = taker
    turn = speech(30) + silence(ms_to_frames(CFG.silence_ms))
    now = 0.0
    for _ in range(3):
        assert len(feed(tt, vad, turn, now_ms=now)) == 1
        assert tt.state is TurnState.PROCESSING
        tt.agent_speaking()
        tt.agent_finished(now_ms=now)
        now += CFG.tail_guard_ms
        feed(tt, vad, silence(1), now_ms=now)
        assert tt.state is TurnState.LISTENING


# --- barge-in ----------------------------------------------------------------
#
# Cutting the tutor off is the one place where "did the VAD hear speech?" is not
# enough. The microphone can hear the loudspeaker, so the bar to take the floor
# mid-sentence is higher than the bar to start a turn in silence — and the words
# that did it began before anyone noticed, so they have to come back with it.


def bfeed(tt, vad, script, now_ms=0.0):
    out = []
    for flag in script:
        vad.script.append(flag)
        utterance = tt.feed(bytes(CFG.frame_bytes), now_ms=now_ms)
        if utterance is not None:
            out.append(utterance)
    return out


def test_a_short_burst_does_not_take_the_floor(interruptible):
    """A cough, or a syllable of the tutor leaking past echo cancellation."""
    tt, vad, _, cuts = interruptible
    tt.agent_speaking()
    bfeed(tt, vad, speech(CFG.onset_frames))  # enough to start a turn in silence
    assert tt.state is TurnState.AGENT_SPEAKING
    assert cuts == []


def test_sustained_speech_takes_the_floor(interruptible):
    tt, vad, _, cuts = interruptible
    tt.agent_speaking()
    bfeed(tt, vad, speech(CFG.barge_in_frames))
    assert tt.state is TurnState.USER_SPEAKING
    assert cuts == [1], "the session was not told to stop the audio"


def test_the_interrupting_words_are_not_lost(interruptible):
    """The student's question starts before the barge-in is noticed. A turn
    that begins at the moment we react hears '그렇게 해요?' and misses '왜'."""
    tt, vad, _, _ = interruptible
    tt.agent_speaking()
    bfeed(tt, vad, speech(CFG.barge_in_frames))
    committed = bfeed(
        tt, vad, speech(20) + silence(ms_to_frames(CFG.silence_ms) + 2)
    )
    assert committed, "the interrupting utterance never came back"
    spoken_frames = len(committed[0]) // CFG.frame_bytes
    assert spoken_frames > CFG.barge_in_frames + 20, spoken_frames


def test_the_bar_drops_back_once_the_floor_is_taken(interruptible):
    """Only the FIRST onset is hard. Mid-turn pauses must not need shouting."""
    tt, vad, _, _ = interruptible
    tt.agent_speaking()
    bfeed(tt, vad, speech(CFG.barge_in_frames))
    assert tt.detector.onset_required is None


def test_the_floor_is_taken_once_per_utterance(interruptible):
    tt, vad, _, cuts = interruptible
    tt.agent_speaking()
    bfeed(tt, vad, speech(CFG.barge_in_frames + 30))
    assert cuts == [1]


def test_thinking_is_still_not_interruptible(interruptible):
    """Nothing is being said, so there is nothing to cut off — and a half turn
    must not survive into the answer."""
    tt, vad, _, cuts = interruptible
    tt.processing()
    bfeed(tt, vad, speech(60))
    assert tt.state is TurnState.PROCESSING
    assert cuts == []


def test_a_normal_turn_still_starts_easily(interruptible):
    """The higher bar applies only over the tutor's voice."""
    tt, vad, _, _ = interruptible
    bfeed(tt, vad, speech(CFG.onset_frames))
    assert tt.state is TurnState.USER_SPEAKING

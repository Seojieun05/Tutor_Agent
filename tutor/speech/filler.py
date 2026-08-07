"""Say something while the tutor thinks.

A hint costs a photo, a recognition, sometimes a solver run, a diagnosis, a
policy decision, a phrasing call and a TTS round trip — several seconds during
which a student sitting next to a real tutor would have heard "음, 어디 보자".
Instead they hear nothing, and silence from something that was just talking
reads as broken rather than as thinking.

Two pieces, and the second is what makes the first honest:

    FillerBank    picks a phrase, never the same one twice in a row
    CachedSpeech  wraps the real speaker so a phrase we have said before costs
                  no TTS call at all — the audio is already bytes in memory,
                  or on disk from a previous run

Only registered phrases are cached. Hints are unique to a student and a step;
caching them would grow without bound and could put one student's hint in
another's ear. The fixed lines — the fillers, "다시 올려 줄래요?", the closing
praise — repeat all day and are what the cache is for.

The filler is not free: it delays the real answer by however long it plays. It
is worth it only when the wait would have been longer than the phrase, which is
why the session waits FILLER_DELAY_MS before starting one and stays silent when
the thinking finishes first.
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
from pathlib import Path

log = logging.getLogger(__name__)

# Short on purpose: this plays BEFORE the answer, so every syllable is added
# latency. Ending on a falling tone stops it sounding like a question the
# student is expected to answer.
FILLER_PHRASES: tuple[str, ...] = (
    "음, 어디 보자.",
    "네, 한번 볼게요.",
    "잠깐만요.",
    "음, 볼게요.",
    "어디 보자.",
)


class FillerBank:
    """Which phrase to say next."""

    def __init__(self, phrases: tuple[str, ...] = FILLER_PHRASES, rng=None):
        self.phrases = tuple(p for p in phrases if p.strip())
        self._rng = rng or random.Random()
        self._last: str | None = None

    def pick(self) -> str:
        if not self.phrases:
            return ""
        choices = [p for p in self.phrases if p != self._last] or list(self.phrases)
        self._last = self._rng.choice(choices)
        return self._last


class CachedSpeech:
    """A speaker that remembers the phrases it says over and over.

    Wraps any object with the speaker interface (`synthesize`, `speak`, `play`,
    `audio_format`) and is one itself, so both speech paths — the server's own
    ffplay and the browser's `synthesize`-then-ship — get the cache for free
    without either of them knowing it exists.
    """

    def __init__(self, speaker, cacheable=(), cache_dir: Path | None = None, voice: str = ""):
        self.speaker = speaker
        self.cache_dir = cache_dir
        self.voice = voice
        self._cacheable = set(cacheable)
        self._audio: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    audio_format = property(lambda self: getattr(self.speaker, "audio_format", "mp3"))

    # --- the speaker interface ----------------------------------------------

    def synthesize(self, text: str) -> bytes | None:
        if not text:
            return None
        if text not in self._cacheable:
            return self.speaker.synthesize(text)

        with self._lock:
            cached = self._audio.get(text)
        if cached is not None:
            self.hits += 1
            return cached

        path = self._path(text)
        if path is not None and path.exists():
            try:
                audio = path.read_bytes()
                with self._lock:
                    self._audio[text] = audio
                self.hits += 1
                return audio
            except OSError as e:  # noqa: PERF203 — a bad cache file is not fatal
                log.warning("could not read the cached phrase %s: %s", path, e)

        self.misses += 1
        audio = self.speaker.synthesize(text)
        if audio:
            with self._lock:
                self._audio[text] = audio
            self._write(path, audio)
        return audio

    def speak(self, text: str) -> None:
        if not text or text not in self._cacheable:
            self.speaker.speak(text)
            return
        audio = self.synthesize(text)
        play = getattr(self.speaker, "play", None)
        if audio and callable(play):
            play(audio)
        else:  # a speaker that cannot play bytes (echo mode) still says it
            self.speaker.speak(text)

    def play(self, audio: bytes) -> None:
        play = getattr(self.speaker, "play", None)
        if callable(play):
            play(audio)

    # --- warming -------------------------------------------------------------

    def warm(self, phrases=()) -> int:
        """Render everything up front so the first turn is not the slow one.

        Called off the event loop at startup. Never raises: a TTS outage must
        cost the fillers, not the server.
        """
        ready = 0
        for text in phrases or sorted(self._cacheable):
            try:
                if self.synthesize(text):
                    ready += 1
            except Exception as e:  # noqa: BLE001 — warming is best-effort
                log.warning("could not pre-render %r: %s", text, e)
        return ready

    def register(self, *phrases: str) -> None:
        self._cacheable.update(p for p in phrases if p)

    # --- disk ----------------------------------------------------------------

    def _path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        # The voice is part of the key: changing TTS_VOICE must not replay the
        # old one, and two voices should be able to share a cache directory.
        key = hashlib.sha1(f"{self.voice}\x00{text}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}.{self.audio_format}"

    def _write(self, path: Path | None, audio: bytes) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a half-written file read by the next run would
            # play as a click, or not at all.
            temp = path.with_suffix(path.suffix + ".part")
            temp.write_bytes(audio)
            temp.replace(path)
        except OSError as e:
            log.warning("could not cache the phrase at %s: %s", path, e)

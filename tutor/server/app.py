"""WebSocket server: wires dependencies and serves one Session per connection."""

from __future__ import annotations

import asyncio
import logging

from websockets.asyncio.server import serve

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.server.session import Deps, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.tools.registry import ToolRegistry
from tutor.vision.recognizer import Recognizer

log = logging.getLogger(__name__)


def build_shared(settings: Settings):
    """Build the per-server (shared) dependencies."""
    db = KnowledgeDB(settings.db_path)
    if not db.all_problems(verified_only=False):
        from tutor.scripts.seed_db import seed_database

        log.info("empty knowledge DB: seeding from bundled seeds")
        seed_database(db)

    registry = ToolRegistry(db)
    if settings.echo_mode:
        from tutor.speech.stt import EchoTranscriber
        from tutor.speech.tts import EchoSpeaker

        llm = EchoLLMClient()
        transcriber = EchoTranscriber(settings)
        speaker = EchoSpeaker(settings)
    else:
        from tutor.llm.client import GrokClient
        from tutor.speech.stt import XaiTranscriber
        from tutor.speech.tts import XaiSpeaker

        llm = GrokClient(settings, registry)
        transcriber = XaiTranscriber(settings)
        speaker = XaiSpeaker(settings)
    return db, llm, transcriber, speaker


def make_deps(settings: Settings, db, llm, transcriber, speaker) -> Deps:
    """Per-connection dependencies (fresh SessionStore each time)."""
    return Deps(
        settings=settings,
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db, settings.recog_conf_threshold),
        hint_gen=HintGenerator(llm, db),
        transcriber=transcriber,
        speaker=speaker,
        store=SessionStore(),
    )


async def amain(settings: Settings) -> None:
    db, llm, transcriber, speaker = build_shared(settings)

    async def handler(ws):
        log.info("device connected: %s", getattr(ws, "remote_address", "?"))
        session = Session(ws, make_deps(settings, db, llm, transcriber, speaker))
        try:
            await session.run()
        finally:
            log.info("device disconnected")

    # 16 MiB: a high-quality JPEG frame easily exceeds the 1 MiB default
    async with serve(handler, settings.ws_host, settings.ws_port, max_size=16 * 1024 * 1024):
        mode = "ECHO (no XAI_API_KEY: canned hints, no API calls)" if settings.echo_mode else "LIVE"
        print(f"Visual Socratic Tutor server on ws://{settings.ws_host}:{settings.ws_port} [{mode}]")
        await asyncio.Future()


def main(settings: Settings) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    try:
        asyncio.run(amain(settings))
    except KeyboardInterrupt:
        print("\nbye")

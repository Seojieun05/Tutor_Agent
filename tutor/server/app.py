"""WebSocket server: wires dependencies and serves one Session per connection.

Two device kinds share the port (and, over SSH, a single forwarded tunnel):

    ws://host:8765/          XIAO (or simulator): VAD on the device
    ws://host:8765/browser   browser: VAD on the server (BrowserSession)
    http://host:8765/        the browser client page + its AudioWorklet
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.server.session import Deps, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.tools.registry import ToolRegistry
from tutor.vision.recognizer import Recognizer

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
# An explicit map, not a path join: nothing outside these two files is servable.
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/worklet.js": ("worklet.js", "text/javascript; charset=utf-8"),
}


def serve_static(connection, request):
    """Answer plain HTTP on the WebSocket port; None lets the upgrade proceed."""
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    path = request.path.split("?", 1)[0]
    entry = STATIC.get(path)
    if entry is None:
        return connection.respond(404, "not found\n")
    body = (WEB_DIR / entry[0]).read_bytes()
    return Response(
        200,
        "OK",
        Headers(
            {
                "Content-Type": entry[1],
                "Content-Length": str(len(body)),
                # the page and the worklet must never be served stale while tuning
                "Cache-Control": "no-store",
            }
        ),
        body,
    )


def build_shared(settings: Settings):
    """Build the per-server (shared) dependencies."""
    db = KnowledgeDB(settings.db_path)
    # Pedagogy, not problems, is the thing the tutor cannot work without: an
    # imported dataset fills `problems` but leaves hint templates empty, and
    # every hint then costs an LLM call. Seeding is idempotent (INSERT OR REPLACE).
    if not db.has_pedagogy():
        from tutor.scripts.seed_db import seed_database

        log.info("no hint templates in the knowledge DB: seeding bundled pedagogy")
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
        evaluator=AnswerEvaluator(llm, db),
        store=SessionStore(),
    )


async def amain(settings: Settings) -> None:
    db, llm, transcriber, speaker = build_shared(settings)

    async def handler(ws):
        path = ws.request.path.split("?", 1)[0]
        deps = make_deps(settings, db, llm, transcriber, speaker)
        if path.rstrip("/") == "/browser":
            from tutor.server.browser import BrowserSession

            log.info("browser connected: %s", getattr(ws, "remote_address", "?"))
            session = BrowserSession(ws, deps)
        else:
            log.info("device connected: %s", getattr(ws, "remote_address", "?"))
            session = Session(ws, deps)
        try:
            await session.run()
        finally:
            log.info("disconnected: %s", path)

    # 16 MiB: a high-quality JPEG frame easily exceeds the 1 MiB default
    async with serve(
        handler,
        settings.ws_host,
        settings.ws_port,
        max_size=16 * 1024 * 1024,
        process_request=serve_static,
    ):
        mode = "ECHO (no XAI_API_KEY: canned hints, no API calls)" if settings.echo_mode else "LIVE"
        print(f"Visual Socratic Tutor server on ws://{settings.ws_host}:{settings.ws_port} [{mode}]")
        print(f"  hands-free browser client: http://localhost:{settings.ws_port}/")
        await asyncio.Future()


def main(settings: Settings) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    try:
        asyncio.run(amain(settings))
    except KeyboardInterrupt:
        print("\nbye")

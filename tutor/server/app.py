"""WebSocket server: wires dependencies and serves one Session per connection.

Two device kinds share the port (and, over SSH, a single forwarded tunnel):

    ws://host:8765/          XIAO (or simulator): VAD on the device
    ws://host:8765/browser   browser: VAD on the server (BrowserSession)
    http://host:8765/        the browser client page + its AudioWorklet
"""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import sys
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.matching import Matcher
from tutor.knowledge.tagger import ConceptTagger
from tutor.server.camera import CameraConnection, CameraHub
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
    # the KB tool already loaded the embedding index: share that one instance
    # with the matcher's SEMANTIC tier instead of loading the model twice
    semantic = getattr(getattr(registry, "kb", None), "semantic", None)
    return db, llm, transcriber, speaker, semantic, build_vision_llm(settings, llm)


def build_vision_llm(settings: Settings, llm):
    """Whichever model reads the worksheet. Only the Recognizer sees this.

    A bad key or a missing package must not cost the student the whole lesson,
    so a failure here falls back to the chat model rather than refusing to start.
    """
    if settings.vision_provider != "gemini" or settings.echo_mode:
        return llm
    from tutor.llm.gemini_vision import GeminiVisionClient

    try:
        return GeminiVisionClient(settings)
    except Exception as e:  # noqa: BLE001 — degrade to Grok, loudly
        log.error("VISION_PROVIDER=gemini unavailable (%s); reading with %s instead",
                  e, settings.chat_model)
        return llm


def make_deps(
    settings: Settings, db, llm, transcriber, speaker, semantic=None, cameras=None,
    vision_llm=None,
) -> Deps:
    """Per-connection dependencies (fresh SessionStore each time)."""
    return Deps(
        settings=settings,
        recognizer=Recognizer(vision_llm or llm, settings),
        matcher=Matcher(db, semantic=semantic),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db, settings.recog_conf_threshold),
        hint_gen=HintGenerator(llm, db, settings.input_mode),
        transcriber=transcriber,
        speaker=speaker,
        evaluator=AnswerEvaluator(llm, db),
        tagger=ConceptTagger(llm),
        cameras=cameras,
        store=SessionStore(),
    )


def port_is_free(host: str, port: int) -> bool:
    """Ask the OS before spending 40s loading models we would then throw away."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as serve() does
        try:
            probe.bind((host, port))
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return False
            raise
    return True


async def amain(settings: Settings) -> None:
    if not port_is_free(settings.ws_host, settings.ws_port):
        raise OSError(errno.EADDRINUSE, "address already in use")
    db, llm, transcriber, speaker, semantic, vision_llm = build_shared(settings)

    cameras = CameraHub()

    async def handler(ws):
        path = ws.request.path.split("?", 1)[0]
        if path.rstrip("/") == "/camera":
            # A camera device (XIAO) has no session of its own: it is an eye
            # that voice sessions borrow. See tutor/server/camera.py.
            await CameraConnection(ws, cameras).run()
            return
        deps = make_deps(
            settings, db, llm, transcriber, speaker, semantic, cameras, vision_llm
        )
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
        ping_interval=20,
        ping_timeout=120,
    ):
        mode = "ECHO (no XAI_API_KEY: canned hints, no API calls)" if settings.echo_mode else "LIVE"
        # flush: redirected to a log file this banner would otherwise sit in the
        # buffer, and it is the one thing the operator is waiting to read.
        say = lambda line: print(line, flush=True)  # noqa: E731
        say(f"Visual Socratic Tutor server on ws://{settings.ws_host}:{settings.ws_port} [{mode}]")
        say(f"  hands-free browser client: http://localhost:{settings.ws_port}/")
        # Say where the tutor's voice will come out, before anyone waits for it.
        from tutor.speech.tts import can_play_locally

        if can_play_locally():
            say("  audio: plays on THIS machine (and in the browser client)")
        else:
            say("  audio: cannot play here — open the browser client and it plays there:")
            say(f"         http://localhost:{settings.ws_port}/  (press 시작)")
            say(f"         remote server? ssh -N -L {settings.ws_port}:localhost:"
                f"{settings.ws_port} <user>@<this-host> first")
        # The board needs a routable address, not localhost.
        if settings.input_mode == "camera":
            # The board needs a routable address, not localhost.
            from tutor.scripts.live_demo import lan_ip

            say(f"  worksheet: camera device (XIAO) — ws://{lan_ip()}:{settings.ws_port}/camera")
        else:
            say("  worksheet: uploaded in the browser page (choose, drag, or Ctrl+V)")
            say("             INPUT_MODE=camera to use a XIAO on /camera instead")
        await asyncio.Future()


def port_in_use_help(settings: Settings) -> str:
    """The commonest restart mistake: the previous server is still running."""
    return "\n".join(
        [
            f"포트 {settings.ws_port}이(가) 이미 사용 중입니다 — 이전 서버가 아직 떠 있어요.",
            "",
            f"  누가 쓰는지 확인:  ss -ltnp | grep {settings.ws_port}",
            "  이전 서버 종료:    pkill -f 'python server.py'",
            f"  또는 다른 포트로:  WS_PORT={settings.ws_port + 1} python server.py",
        ]
    )


def main(settings: Settings) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    try:
        asyncio.run(amain(settings))
    except KeyboardInterrupt:
        print("\nbye")
    except OSError as e:
        # a traceback here says nothing an operator can act on
        if e.errno != errno.EADDRINUSE:
            raise
        print(port_in_use_help(settings), file=sys.stderr, flush=True)
        raise SystemExit(1) from None

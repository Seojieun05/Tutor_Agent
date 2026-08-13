"""WebSocket server: wires dependencies and serves one Session per connection.

Two device kinds share the port (and, over SSH, a single forwarded tunnel):

    ws://host:8765/          local-mic device (or simulator): VAD on the device
    ws://host:8765/browser   browser: VAD on the server (BrowserSession)
    ws://host:8765/camera    a camera device: a phone running /phone
    http://host:8765/        the browser client page + its AudioWorklet
    https://host:8766/phone  the phone camera page — TLS only, see tls_context()
"""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import ssl
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from tutor.config import Settings
from tutor.console import say, soften_stdout
from tutor.hints.generator import HintGenerator
from tutor.hints.illustrator import Illustrator
from tutor.knowledge.db import KnowledgeDB
from tutor.knowledge.matching import Matcher
from tutor.server.camera import CameraConnection, CameraHub
from tutor.llm.echo import EchoLLMClient
from tutor.server.session import Deps, Session
from tutor.llm.timing import timed
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.filler import FillerBank
from tutor.speech.intent import IntentClassifier
from tutor.state.answer import AnswerEvaluator
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.tools.registry import ToolRegistry
from tutor.vision.recognizer import Recognizer

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
# An explicit map, not a path join: nothing outside these files is servable.
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/worklet.js": ("worklet.js", "text/javascript; charset=utf-8"),
    "/phone": ("phone.html", "text/html; charset=utf-8"),
    "/phone.html": ("phone.html", "text/html; charset=utf-8"),
    # the tutor, swimming, for the seconds it is thinking. VP9 with a real
    # alpha channel, so it drops onto the board with no background to strip.
    "/squid-thinking.webm": ("squid-thinking.webm", "video/webm"),
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


@dataclass
class Shared:
    """The per-server dependencies: built once, shared by every connection.

    A dataclass rather than the tuple this used to be. Every model that got
    its own routing knob added an element, callers unpacked by position, and
    tutor.scripts.warm_kb was quietly unpacking nine values out of eleven —
    broken since the illustrator landed, and invisible until it ran.
    """

    db: object
    llm: object          # the chat model: every unrouted purpose, and the standby
    transcriber: object
    speaker: object
    semantic: object | None
    vision_llm: object
    hint_llm: object
    eval_llm: object
    estimate_llm: object
    illustrate_llm: object
    solve_llm: object
    eval_second_llm: object | None


def build_shared(settings: Settings) -> Shared:
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
    speaker = wrap_with_cache(settings, speaker)
    # the KB tool already loaded the embedding index: share that one instance
    # with the matcher's SEMANTIC tier instead of loading the model twice
    semantic = getattr(getattr(registry, "kb", None), "semantic", None)
    vision_llm = build_vision_llm(settings, llm)
    hint_llm = build_hint_llm(settings, llm)
    eval_llm = build_eval_llm(settings, llm, registry)
    eval_second_llm = build_eval_second_llm(settings, llm, registry)
    estimate_llm = build_estimate_llm(settings, llm, registry)
    illustrate_llm = build_illustrate_llm(settings, llm)
    solve_llm = build_solve_llm(settings, llm, registry)
    # One log line per model call, always on. The tutor's latency is almost
    # entirely other people's servers, so the only useful question is which
    # call — and that is not answerable after the fact without this.
    return Shared(
        db=db,
        llm=timed(llm, settings.chat_model),
        transcriber=transcriber,
        speaker=speaker,
        semantic=semantic,
        vision_llm=timed(vision_llm),
        hint_llm=timed(hint_llm),
        eval_llm=timed(eval_llm),
        estimate_llm=timed(estimate_llm),
        illustrate_llm=timed(illustrate_llm),
        solve_llm=timed(solve_llm),
        eval_second_llm=(
            timed(eval_second_llm) if eval_second_llm is not None else None
        ),
    )


def wrap_with_cache(settings: Settings, speaker):
    """Memoize the lines the tutor repeats all day: the fillers and the fixed
    prompts. Hints are never cached — they belong to one student and one step."""
    if not settings.filler_enabled:
        return speaker
    from tutor.hints.generator import FIXED_ACTIONS
    from tutor.server.session import (
        PROBLEM_DONE,
        READOUT_CLOSERS,
        READOUT_OPENER,
        RETRY_PROMPTS,
        WORK_CHECK_DEFAULT,
        WORK_CHECK_OPENERS,
        WORK_CHECK_REACTIONS,
        WORK_CONFIRMED,
    )
    from tutor.speech.filler import FILLER_PHRASES, WORK_CHECK_NARRATIONS, CachedSpeech

    repeated = [
        *FILLER_PHRASES,
        *(t for t in FIXED_ACTIONS.values() if t),
        *RETRY_PROMPTS.values(),
        PROBLEM_DONE,
        # the whole work-check frame: openers, the mid-wait narrations, the
        # reactions and the confirmation — every fixed line the turn can say,
        # so the only TTS it ever waits for is the hint itself
        *WORK_CHECK_OPENERS,
        *WORK_CHECK_NARRATIONS,
        *(t for t in WORK_CHECK_REACTIONS.values() if t),
        WORK_CHECK_DEFAULT,
        WORK_CONFIRMED,
        # the readout frame: the problem text between them is the only line of
        # the narration that ever pays for TTS at speak time
        READOUT_OPENER,
        *READOUT_CLOSERS.values(),
    ]
    return CachedSpeech(
        speaker,
        cacheable=repeated,
        cache_dir=settings.tts_cache_dir,
        voice=settings.tts_voice,
    )


def build_vision_llm(settings: Settings, llm):
    """Whichever model reads the worksheet. Only the Recognizer sees this."""
    return _gemini_or(settings, llm, settings.vision_provider,
                      settings.gemini_vision_model, "VISION_PROVIDER")


def build_hint_llm(settings: Settings, llm):
    """Whichever model writes what the tutor says. Only HintGenerator sees this.

    The hint LEVEL is not this model's decision and the leak guard still checks
    its output, so swapping it changes the wording and nothing about how much
    is given away.
    """
    return _gemini_or(settings, llm, settings.hint_provider,
                      settings.gemini_hint_model, "HINT_PROVIDER")


def build_eval_llm(settings: Settings, llm, registry=None):
    """Whichever model grades a spoken answer. Only AnswerEvaluator sees this.

    The verdict feeds the same deterministic policy either way (correct → next
    step, wrong → escalate), so swapping the model changes grading judgement
    and nothing about what the tutor is allowed to do with it.

    Eval is the one Gemini client that gets the tool registry: grading turns
    on equivalence ("3x가 15" vs "x = 5"), which is the sympy tools' job.
    Vision and hints stay toolless by design — their context is prefetched.
    """
    return _gemini_or(settings, llm, settings.eval_provider,
                      settings.gemini_eval_model, "EVAL_PROVIDER", registry=registry)


def build_eval_second_llm(settings: Settings, llm, registry=None):
    """The judge consulted before a student is told they are wrong.

    None unless grading was routed to a smaller model: the same model asked
    twice is not an opinion. It gets the registry because overturning a
    verdict usually turns on equivalence — the whole tangent the student said
    against the slope the question asked for.
    """
    if settings.eval_provider != "gemini":
        return None
    return _gemini_or(settings, llm, "gemini", settings.gemini_eval_second_model,
                      "GEMINI_EVAL_SECOND_MODEL", registry=registry)


def build_illustrate_llm(settings: Settings, llm):
    """Whichever model decides what to draw. Only the Illustrator sees it.

    Toolless by design: it is shown the hint it illustrates and the page in
    front of the student, never the reference solution, so there is nothing
    for it to look up and nothing for it to sketch the answer from.
    """
    return _gemini_or(settings, llm, settings.illustrate_provider,
                      settings.gemini_illustrate_model, "ILLUSTRATE_PROVIDER")


def build_solve_llm(settings: Settings, llm, registry=None):
    """Whichever model writes the reference solution. Only the solver sees it.

    The registry rides along: `solve` may search the KB for a similar solved
    problem and verify its own final answer with sympy. What comes back is
    checked the same way whatever wrote it — never marked verified, and stored
    only if mathnorm.verify_answer agrees — so this knob trades speed against
    how finely the solution is cut, not against whether it is believed.
    """
    return _gemini_or(settings, llm, settings.solve_provider,
                      settings.gemini_solve_model, "SOLVE_PROVIDER",
                      registry=registry)


def build_estimate_llm(settings: Settings, llm, registry=None):
    """Whichever model diagnoses the written work. Only the estimator sees it.

    Estimate sits on the WORK_CHECK critical path — the student asked "풀이
    맞아?" and is waiting on this call for the verdict. The registry rides
    along so the misconception KB lookups survive the move, and the sympy
    arithmetic check (_arithmetic_check) still outranks whatever model runs:
    swapping it changes diagnosis speed and judgement, never the final say
    on arithmetic.
    """
    return _gemini_or(settings, llm, settings.estimate_provider,
                      settings.gemini_estimate_model, "ESTIMATE_PROVIDER",
                      registry=registry)


def _gemini_or(settings: Settings, llm, provider: str, model: str, knob: str,
               registry=None):
    """A bad key or a missing package must not cost the student the whole
    lesson, so a failure here falls back to the chat model rather than
    refusing to start — loudly, because reading with a model you did not
    choose is worse than knowing you are."""
    if provider != "gemini" or settings.echo_mode:
        return llm
    from tutor.llm.fallback import FallbackLLM
    from tutor.llm.gemini import GeminiClient

    try:
        chosen = GeminiClient(settings, model, role=knob, registry=registry)
    except Exception as e:  # noqa: BLE001 — degrade to Grok, loudly
        log.error("%s=gemini unavailable (%s); using %s instead",
                  knob, e, settings.chat_model)
        return llm
    # A key can list a model it has no quota for, and that only shows up on the
    # first real call — mid-lesson. Keep the old model on standby.
    return FallbackLLM(chosen, llm, label=f"{knob}={model}")


def make_deps(
    settings: Settings, db, llm, transcriber, speaker, semantic=None, cameras=None,
    vision_llm=None, hint_llm=None, eval_llm=None, estimate_llm=None,
    illustrate_llm=None, eval_second_llm=None, solve_llm=None,
) -> Deps:
    """Per-connection dependencies (fresh SessionStore each time)."""
    return Deps(
        settings=settings,
        recognizer=Recognizer(vision_llm or llm, settings),
        matcher=Matcher(db, semantic=semantic),
        solver=GrokSolver(solve_llm or llm, db),
        estimator=StudentStateEstimator(estimate_llm or llm, db, settings.recog_conf_threshold),
        hint_gen=HintGenerator(hint_llm or llm, db, settings.input_mode),
        # the drawing hand: it runs while the tutor speaks, so its seconds
        # come out of the silence rather than out of the student's wait
        illustrator=Illustrator(illustrate_llm or hint_llm or llm),
        transcriber=transcriber,
        speaker=speaker,
        evaluator=AnswerEvaluator(eval_llm or llm, db, second_opinion=eval_second_llm),
        cameras=cameras,
        fillers=FillerBank() if settings.filler_enabled else None,
        classifier=IntentClassifier(llm),
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


def tls_context(settings: Settings) -> ssl.SSLContext | None:
    """The phone's secure context, or None.

    This never replaces the plain listener — it is an ADDITIONAL port, because
    localhost is already a secure context without any of this. Only the phone,
    reaching the laptop by LAN IP, has no other way to get at getUserMedia.
    """
    if not settings.tls_enabled:
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(settings.tls_cert, settings.tls_key)
    return ctx


class PortInUse(OSError):
    """Which port, so the advice names the one that is actually taken."""

    def __init__(self, port: int):
        super().__init__(errno.EADDRINUSE, "address already in use")
        self.port = port


async def amain(settings: Settings) -> None:
    ports = [settings.ws_port]
    if settings.tls_enabled:
        ports.append(settings.tls_listen_port)
    for port in ports:
        if not port_is_free(settings.ws_host, port):
            raise PortInUse(port)
    shared = build_shared(settings)
    db, llm, transcriber, speaker = (
        shared.db, shared.llm, shared.transcriber, shared.speaker
    )

    cameras = CameraHub()

    async def warm_fillers() -> None:
        """Render the repeated phrases before anyone asks for one.

        In a thread and unawaited: the first student should not pay for this,
        and neither should the port being open.
        """
        warm = getattr(speaker, "warm", None)
        if not callable(warm):
            return
        ready = await asyncio.to_thread(warm)
        log.info("pre-rendered %d spoken phrases (cache: %s)", ready, settings.tts_cache_dir)

    warming = asyncio.create_task(warm_fillers())

    async def handler(ws):
        path = ws.request.path.split("?", 1)[0]
        if path.rstrip("/") == "/camera":
            # A camera device has no session of its own: it is an eye
            # that voice sessions borrow. See tutor/server/camera.py.
            await CameraConnection(ws, cameras).run()
            return
        deps = make_deps(
            settings, db, llm, transcriber, speaker, shared.semantic, cameras,
            shared.vision_llm, shared.hint_llm, shared.eval_llm,
            shared.estimate_llm, shared.illustrate_llm, shared.eval_second_llm,
            shared.solve_llm,
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
    common = dict(
        max_size=16 * 1024 * 1024,
        process_request=serve_static,
        ping_interval=20,
        ping_timeout=120,
    )
    tls = tls_context(settings)
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(
            serve(handler, settings.ws_host, settings.ws_port, **common)
        )
        if tls is not None:
            # Same handler, same CameraHub: a camera on the TLS port and one on
            # the plain port are both just eyes to the voice session.
            await stack.enter_async_context(
                serve(handler, settings.ws_host, settings.tls_listen_port,
                      ssl=tls, **common)
            )
        mode = "ECHO (no XAI_API_KEY: canned hints, no API calls)" if settings.echo_mode else "LIVE"
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
        if settings.input_mode == "camera":
            # A phone needs a routable address, not localhost.
            from tutor.scripts.live_demo import lan_ip

            ip = lan_ip()
            say(f"  worksheet: camera device — ws://{ip}:{settings.ws_port}/camera")
            if tls is not None:
                say(f"  phone camera: https://{ip}:{settings.tls_listen_port}/phone")
                say("         (self-signed: accept the warning once, then allow the camera)")
            else:
                # Say it here rather than let the phone fail with a blank screen.
                say("  phone camera: off (no TLS_CERT/TLS_KEY in .env)")
                say("         python -m tutor.scripts.make_cert   (getUserMedia needs HTTPS)")
        else:
            say("  worksheet: uploaded in the browser page (choose, drag, or Ctrl+V)")
            say("             INPUT_MODE=camera to use the phone camera on /camera instead")
        await asyncio.Future()


def port_in_use_help(settings: Settings, port: int | None = None) -> str:
    """The commonest restart mistake: the previous server is still running."""
    port = port or settings.ws_port
    return "\n".join(
        [
            f"포트 {port}이(가) 이미 사용 중입니다 — 이전 서버가 아직 떠 있어요.",
            "",
            f"  누가 쓰는지 확인:  ss -ltnp | grep {port}",
            "  이전 서버 종료:    pkill -f 'python server.py'",
            f"  또는 다른 포트로:  WS_PORT={settings.ws_port + 1} python server.py",
        ]
    )


def main(settings: Settings) -> None:
    soften_stdout()  # the banner must never be the reason the server fails to start
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
        say(port_in_use_help(settings, getattr(e, "port", None)), sys.stderr)
        raise SystemExit(1) from None

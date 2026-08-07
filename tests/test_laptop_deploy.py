"""Running the server on the laptop, with a XIAO on the same Wi-Fi.

The demo topology: server.py on the laptop, the browser page on localhost,
the board connecting to the laptop's LAN IP. Nothing here may require the
optional RAG stack — a laptop should not need a 2 GB torch download to start.
"""

import builtins
from pathlib import Path

import pytest

from tutor.knowledge.db import KnowledgeDB
from tutor.tools import domain_kb
from tutor.tools.domain_kb import DomainKBTool, load_semantic_retriever
from tutor.tools.registry import ToolRegistry


class TestSemanticIsOptional:
    def test_a_missing_index_does_not_stop_the_server(self, db, monkeypatch, caplog):
        def missing(_db):
            raise FileNotFoundError("data/problem_embeddings.npz")

        monkeypatch.setattr(domain_kb, "SemanticRetriever", missing, raising=False)
        monkeypatch.setitem(
            __import__("sys").modules, "tutor.retrieval.semantic",
            type("M", (), {"SemanticRetriever": missing}),
        )
        with caplog.at_level("WARNING"):
            assert load_semantic_retriever(db) is None
        assert "build_embeddings" in caplog.text  # says how to get it back

    def test_a_missing_library_does_not_stop_the_server(self, db, monkeypatch, caplog):
        real_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name.startswith("tutor.retrieval"):
                raise ImportError("No module named 'sentence_transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_torch)
        with caplog.at_level("WARNING"):
            assert load_semantic_retriever(db) is None
        assert "[rag]" in caplog.text  # says how to enable it

    def test_the_kb_tool_still_answers_without_it(self, db, monkeypatch):
        monkeypatch.setattr(domain_kb, "load_semantic_retriever", lambda _db: None)
        tool = DomainKBTool(db)
        assert tool.semantic is None
        # concept retrieval is unaffected...
        hits = tool.search(kind="problems", concepts=["linear_equation"])
        assert hits["problems"]
        # ...and a query that would have gone semantic degrades to empty, not a crash
        assert tool.search(kind="problems", query="전혀 없는 문제") == {"problems": []}

    def test_the_matcher_skips_the_semantic_tier(self, db, monkeypatch):
        from tutor.knowledge.matching import Matcher
        from tutor.knowledge.models import Tier
        from tutor.vision.recognizer import Recognition

        monkeypatch.setattr(domain_kb, "load_semantic_retriever", lambda _db: None)
        registry = ToolRegistry(db)
        matcher = Matcher(db, semantic=registry.kb.semantic)
        result = matcher.match(Recognition(problem_text="처음 보는 문제"))
        assert result.tier == Tier.NEW  # falls through, does not raise


class TestFirmwareSettings:
    """The preflight has to hand over settings you can paste into the sketch."""

    def test_it_reports_a_reachable_address(self):
        from tutor.scripts.live_demo import lan_ip

        ip = lan_ip()
        assert ip.count(".") == 3
        assert not ip.startswith("127.")  # the board cannot reach loopback

    def test_the_sketch_block_matches_the_running_port(self):
        from tutor.config import Settings
        from tutor.scripts.live_demo import firmware_settings

        block = firmware_settings(Settings(ws_port=8765))
        assert "#define SERVER_PORT   8765" in block
        assert "SERVER_HOST" in block and "/camera" not in block  # host, not path


class TestConsoleEncoding:
    """A Korean Windows console is cp949: no em dash, no emoji, and `print`
    raises rather than degrading. That killed the startup banner, the echo-mode
    tutor voice, and `--help` on two scripts."""

    def _cp949_console(self):
        import io

        raw = io.BytesIO()
        return raw, io.TextIOWrapper(raw, encoding="cp949", newline="")

    def test_say_degrades_instead_of_raising(self):
        from tutor.console import say

        raw, console = self._cp949_console()
        say("audio: cannot play here — [TUTOR 🔊] 힌트를 줄게요", console)  # must not raise
        console.flush()

        out = raw.getvalue().decode("cp949")
        assert "힌트를 줄게요" in out and "TUTOR" in out  # the message survives
        assert "—" not in out and "🔊" not in out       # only the unencodable go

    def test_say_leaves_encodable_text_alone(self):
        from tutor.console import say

        raw, console = self._cp949_console()
        say("  camera device (XIAO): ws://10.0.0.2:8765/camera", console)
        console.flush()
        assert raw.getvalue().decode("cp949").strip() == (
            "camera device (XIAO): ws://10.0.0.2:8765/camera"
        )

    def test_softening_a_stream_that_cannot_be_reconfigured_is_harmless(self):
        """pytest replaces sys.stdout with its own capture object."""
        from tutor.console import soften_stdout

        soften_stdout()  # no exception


class TestPhoneCamera:
    """The phone is a camera device served over TLS, and nothing more."""

    def test_the_page_is_servable(self):
        from tutor.server.app import STATIC, WEB_DIR

        assert STATIC["/phone"] == STATIC["/phone.html"]
        assert (WEB_DIR / STATIC["/phone"][0]).exists()

    def test_the_page_connects_to_the_camera_socket(self):
        """Not /browser: the phone has no session, it is borrowed like the board."""
        from tutor.server.app import STATIC, WEB_DIR

        page = (WEB_DIR / STATIC["/phone"][0]).read_text(encoding="utf-8")
        assert "/camera" in page and "/browser" not in page
        assert "capture_request" in page      # it answers, it does not push
        assert "playsinline" in page          # or iOS refuses to show the preview

    def test_the_default_camera_is_the_rear_one_by_facing(self):
        """Only an explicit choice uses a deviceId; the default must never be a
        device index — videoinput[0] is the selfie camera on some phones."""
        from tutor.server.app import STATIC, WEB_DIR

        page = (WEB_DIR / STATIC["/phone"][0]).read_text(encoding="utf-8")
        assert 'facingMode: { ideal: "environment" }' in page

    def test_the_camera_keeps_its_own_aspect_ratio(self):
        """Width only. Asking for a height too pins an aspect ratio, and the
        browser crops the sensor to reach it — 2560×1440 asks a 4:3 camera for
        16:9 and cuts the sides off the page."""
        from tutor.server.app import STATIC, WEB_DIR

        page = (WEB_DIR / STATIC["/phone"][0]).read_text(encoding="utf-8")
        script = page.split("<script>")[1]
        # the constraint constant only — an IMAGE header reporting the captured
        # height is a different thing entirely
        block = script.split("const SIZE = {")[1].split("};")[0]
        # comments out: they discuss `height`, and prose must not decide a test
        code = "\n".join(line.split("//")[0] for line in block.splitlines())
        assert "width: { ideal: 2560 }" in code   # a small stream is a cropped one
        # A wider frame is asked for softly. Naming a height instead would PIN
        # the ratio and crop the sensor to fake it, which cut the sides off.
        assert "aspectRatio: { ideal: 16 / 9 }" in code
        assert "height" not in code

    def test_the_lens_is_pinned_to_the_main_rear_camera(self):
        """The focus fix, and not a choice the student should have to make. A
        four-camera phone exposes all of them and only some focus on paper held
        close; `facingMode` gets the browser's default rear camera, which on
        this hardware could not. The lowest-numbered rear lens is the main one.

        NOT videoinput[0] — that index is the selfie camera on this phone."""
        from tutor.server.app import STATIC, WEB_DIR

        page = (WEB_DIR / STATIC["/phone"][0]).read_text(encoding="utf-8")
        script = page.split("<script>")[1]
        assert "enumerateDevices" in script and "rearLensId" in script
        assert "deviceId: { exact: wanted }" in script
        assert "id=\"lens\"" not in page          # no picker
        assert "localStorage" not in script       # nothing to remember

    def test_the_lens_index_survives_the_android_label_format(self):
        """"camera2 0, facing back": parseInt() on the whole string is NaN, and
        the "2" in "camera2" is not the index. Getting this wrong silently
        selected the macro lens instead of the main one."""
        import re

        from tutor.server.app import STATIC, WEB_DIR

        page = (WEB_DIR / STATIC["/phone"][0]).read_text(encoding="utf-8")
        # the regexes the page uses, applied here to the labels it will meet
        primary, fallback = r"(\d+)\s*,", r"(\d+)(?!.*\d)"
        assert primary in page and fallback in page

        def index(label):
            m = re.search(primary, label) or re.search(fallback, label)
            return int(m.group(1)) if m else 99

        assert index("camera2 0, facing back") == 0
        assert index("camera2 3, facing back") == 3
        assert index("camera2 10, facing back") == 10
        assert index("Back Camera") == 99          # unnumbered sorts last

    def test_focus_control_is_present_and_stays_inside_the_frame(self):
        """Continuous AF plus tap-to-focus. The tap is normalized, and a tap on
        the letterbox must not hand the camera a point outside the picture."""
        from tutor.server.app import STATIC, WEB_DIR

        page = (WEB_DIR / STATIC["/phone"][0]).read_text(encoding="utf-8")
        script = page.split("<script>")[1]
        assert 'focusMode: "continuous"' in script and "applyConstraints" in script
        assert "pointsOfInterest" in script
        assert "Math.min(1, Math.max(0, v))" in script

    def test_the_preview_shows_the_frame_that_gets_sent(self):
        """object-fit:cover crops a 16:9 stream to the middle of a portrait box
        and scales it up — it looks zoomed in AND soft while the photo actually
        sent is neither, so the student cannot judge framing or focus at all."""
        from tutor.server.app import STATIC, WEB_DIR

        page = (WEB_DIR / STATIC["/phone"][0]).read_text(encoding="utf-8")
        style = page.split("<style>")[1].split("</style>")[0]
        assert "object-fit:contain" in style.replace(" ", "").replace("\n", "")
        assert "object-fit:cover" not in style.replace(" ", "").replace("\n", "")

    def test_tls_is_off_until_a_cert_is_configured(self):
        from tutor.config import Settings
        from tutor.server.app import tls_context

        assert tls_context(Settings()) is None
        # half a pair is not a configuration
        assert Settings(tls_cert=Path("a.crt")).tls_enabled is False

    def test_the_tls_port_sits_next_to_the_plain_one(self):
        from tutor.config import Settings

        assert Settings(ws_port=8765).tls_listen_port == 8766
        assert Settings(ws_port=8765, tls_port=9000).tls_listen_port == 9000

    def test_a_generated_cert_actually_loads(self, tmp_path):
        """The cert has to satisfy ssl, not just exist."""
        from tutor.config import Settings
        from tutor.scripts.make_cert import write_selfsigned
        from tutor.server.app import tls_context

        cert, key = tmp_path / "tutor.crt", tmp_path / "tutor.key"
        if not write_selfsigned(cert, key, "192.168.0.2"):
            pytest.skip("neither cryptography nor openssl available")

        settings = Settings(tls_cert=cert, tls_key=key)
        assert settings.tls_enabled
        assert tls_context(settings) is not None

    def test_the_server_decodes_what_the_page_writes(self):
        """The page builds frames by hand in JS, as the board does in C.

        These bytes came out of encodeFrame() in a real browser running
        tutor/web/phone.html; kept here so a change to the framing breaks loudly
        instead of leaving the phone silently unable to answer a capture.
        """
        from tutor.protocol.frames import ImageFrame, decode

        frame = decode(
            bytes.fromhex(
                "01000000417b22636170747572655f6964223a2263616d2d37222c22666f726d61"
                "74223a226a706567222c227769647468223a323536302c22686569676874223a31"
                "3434307dffd80102030405ffd9"
            )
        )
        assert isinstance(frame, ImageFrame)
        assert frame.header.capture_id == "cam-7"
        assert (frame.header.width, frame.header.height) == (2560, 1440)
        assert frame.jpeg == bytes.fromhex("ffd80102030405ffd9")

    def test_the_cert_covers_the_ip_not_just_a_name(self):
        """Browsers reject an IP URL whose cert has no IP SAN, warning or not."""
        from tutor.scripts.make_cert import openssl_command, san_entries

        assert "IP:192.168.0.2" in san_entries("192.168.0.2")
        assert "subjectAltName=IP:192.168.0.2,IP:127.0.0.1,DNS:localhost" in (
            openssl_command(Path("c"), Path("k"), "192.168.0.2")
        )

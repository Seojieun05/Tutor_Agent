"""Running the server on the laptop, with a XIAO on the same Wi-Fi.

The demo topology: server.py on the laptop, the browser page on localhost,
the board connecting to the laptop's LAN IP. Nothing here may require the
optional RAG stack — a laptop should not need a 2 GB torch download to start.
"""

import builtins

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

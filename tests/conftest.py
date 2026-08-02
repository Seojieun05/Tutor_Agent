import pytest

from tutor.knowledge.db import KnowledgeDB
from tutor.scripts.seed_db import seed_database


@pytest.fixture
def db() -> KnowledgeDB:
    db = KnowledgeDB(":memory:")
    report = seed_database(db)
    assert all(ok for _, ok, _ in report), f"seed rejected: {report}"
    return db

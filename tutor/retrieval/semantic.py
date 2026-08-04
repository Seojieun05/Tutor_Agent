from pathlib import Path
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from tutor.knowledge.db import KnowledgeDB

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")

MODEL_NAME = "intfloat/multilingual-e5-small"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = ROOT / "data" / "problem_embeddings.npz"


class SemanticRetriever:
    def __init__(
        self,
        db: KnowledgeDB,
        index_path: str | Path = DEFAULT_INDEX_PATH,
    ):
        self.db = db

        data = np.load(index_path)

        self.ids = data["ids"].astype(str)
        self.embeddings = data["embeddings"].astype(np.float32)

        self.model = get_embedding_model(MODEL_NAME)

        # DB 문제를 id로 바로 찾기 위한 map
        self.problems = {
            p.id: p
            for p in db.all_problems()
        }

    def search(
        self,
        query: str,
        limit: int = 5,
    ):
        query_embedding = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        # embeddings가 normalize되어 있으므로
        # dot product == cosine similarity
        scores = self.embeddings @ query_embedding

        k = min(limit, len(scores))

        indices = np.argpartition(
            -scores,
            k - 1,
        )[:k]

        indices = indices[
            np.argsort(-scores[indices])
        ]

        results = []

        for idx in indices:
            problem_id = self.ids[idx]
            problem = self.problems.get(problem_id)

            if problem is None:
                continue

            results.append(
                (problem, float(scores[idx]))
            )

        return results
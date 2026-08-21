from pathlib import Path
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from tutor.knowledge.db import KnowledgeDB

# 임베딩 모델은 한 번만 로드해 재사용.
@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")

# 다국어 문장 임베딩 모델.
MODEL_NAME = "intfloat/multilingual-e5-small"

ROOT = Path(__file__).resolve().parents[2]
# 미리 만들어 둔 문제 임베딩 인덱스 파일.
DEFAULT_INDEX_PATH = ROOT / "data" / "problem_embeddings.npz"


# 문장 임베딩으로 비슷한 문제를 찾는 검색기(SEMANTIC 등급 매칭과 KB 툴이 함께 쓴다).
class SemanticRetriever:
    # 인덱스를 읽고 모델을 올린 뒤, DB 문제를 id로 바로 찾을 수 있게 담아 둔다.
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

    # 질의문과 가장 가까운 문제들을 (문제, 유사도) 목록으로 돌려준다.
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
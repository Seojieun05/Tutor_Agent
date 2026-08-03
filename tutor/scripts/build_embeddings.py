from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from tutor.knowledge.db import KnowledgeDB


MODEL_NAME = "intfloat/multilingual-e5-small"

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "knowledge.db"
OUT_PATH = ROOT / "data" / "problem_embeddings.npz"


def main():
    db = KnowledgeDB(DB_PATH)
    problems = db.all_problems()

    print(f"problems: {len(problems)}")

    passages = []

    for p in problems:
        text = p.problem_text

        if p.equations:
            text += "\n수식: " + " ; ".join(p.equations)

        # E5 retrieval format
        passages.append("passage: " + text)

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    embeddings = model.encode(
        passages,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.savez_compressed(
        OUT_PATH,
        ids=np.array([p.id for p in problems]),
        embeddings=embeddings.astype(np.float32),
    )

    print(f"saved: {OUT_PATH}")
    print(f"shape: {embeddings.shape}")

    db.close()


if __name__ == "__main__":
    main()
"""Preflight + instructions for the live (real Grok) demo."""

from __future__ import annotations

import shutil
from pathlib import Path

from tutor.config import load_settings
from tutor.knowledge.db import KnowledgeDB
from tutor.scripts.gen_assets import ASSETS_DIR


def main() -> None:
    settings = load_settings()
    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "XAI_API_KEY",
            not settings.echo_mode,
            "set in .env" if not settings.echo_mode else "missing -> server runs in ECHO mode",
        )
    )
    db = KnowledgeDB(settings.db_path)
    n = len(db.all_problems())
    checks.append(
        ("knowledge DB", n > 0, f"{n} verified problems ({settings.db_path})"
         if n else "empty -> run: python -m tutor.scripts.seed_db")
    )
    checks.append(
        ("ffplay", shutil.which("ffplay") is not None,
         "found" if shutil.which("ffplay") else "missing -> sudo apt install ffmpeg")
    )
    assets = list(Path(ASSETS_DIR).glob("*.jpg"))
    checks.append(
        ("simulator assets", bool(assets), f"{len(assets)} images"
         if assets else "missing -> run: python -m tutor.scripts.gen_assets")
    )

    print("live demo preflight:")
    for name, ok, detail in checks:
        print(f"  {'OK ' if ok else 'FAIL'} {name:<18} {detail}")

    print(
        "\ndemo script:\n"
        "  1. terminal A: python server.py\n"
        "  2. terminal B: python -m simulator.device_sim --server ws://localhost:8765 \\\n"
        "       --images simulator/assets/lin_001_wrong_sign.jpg \\\n"
        "                simulator/assets/lin_001_step1_ok.jpg \\\n"
        "                simulator/assets/lin_001_solved.jpg \\\n"
        "       --wav simulator/assets/hint.wav\n"
        "  3. press h  -> EXACT match, L1 Socratic hint (Korean TTS)\n"
        "  4. press h  -> same work, escalates to L2\n"
        "  5. press n, then h -> progress detected, back to L1 (fading)\n"
    )


if __name__ == "__main__":
    main()

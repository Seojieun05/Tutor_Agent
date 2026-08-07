"""Preflight for a live demo: what is missing, and how the phone reaches this box.

    python -m tutor.scripts.live_demo

Run it on the machine that will run server.py. It checks the pieces that
silently degrade (API key, pedagogy, audio, embedding index) and prints this
machine's LAN address — the number that is otherwise guessed wrong more than
anything else.
"""

from __future__ import annotations

import platform
import shutil
import socket
from pathlib import Path

from tutor.config import Settings, load_settings
from tutor.console import soften_stdout
from tutor.knowledge.db import KnowledgeDB
from tutor.scripts.gen_assets import ASSETS_DIR


def lan_ip() -> str:
    """The address other devices on the Wi-Fi can reach.

    Opening a UDP socket to a public address sends nothing — it just makes the
    OS pick the outbound interface, which is the one the phone will use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def firewall_hint(port: int) -> str:
    """Inbound is blocked by default on Windows, which is where the phone fails."""
    if platform.system() == "Windows":
        return (
            "  Windows 방화벽에서 인바운드 허용 (관리자 PowerShell, 1회):\n"
            f'    New-NetFirewallRule -DisplayName "Tutor {port}" -Direction Inbound '
            f"-Protocol TCP -LocalPort {port} -Action Allow"
        )
    return (
        f"  방화벽에서 {port} 인바운드 허용이 필요할 수 있습니다 "
        f"(ufw: sudo ufw allow {port}/tcp)"
    )


def main() -> None:
    soften_stdout()  # a cp949 console cannot encode this preflight's punctuation
    settings = load_settings()
    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "XAI_API_KEY",
            not settings.echo_mode,
            ".env에 설정됨" if not settings.echo_mode else "없음 → ECHO 모드로 동작 (정해진 힌트)",
        )
    )

    db = KnowledgeDB(settings.db_path)
    problems = len(db.all_problems())
    checks.append(
        ("knowledge DB", problems > 0,
         f"검증된 문제 {problems}개 ({settings.db_path})"
         if problems else "비어 있음 → python -m tutor.scripts.seed_db")
    )
    checks.append(
        ("hint templates", db.has_pedagogy(),
         "있음" if db.has_pedagogy() else "없음 → 서버가 첫 실행에 자동 시딩합니다")
    )

    index = Path("data/problem_embeddings.npz")
    try:
        import sentence_transformers  # noqa: F401

        has_lib = True
    except ImportError:
        has_lib = False
    semantic_ok = has_lib and index.exists()
    checks.append(
        ("semantic 검색 (선택)", semantic_ok,
         "사용 가능" if semantic_ok else
         "꺼짐 → 나머지는 정상 동작 (켜려면: pip install -e '.[rag]' + build_embeddings)")
    )

    ffplay = shutil.which("ffplay")
    checks.append(
        ("ffplay (선택)", ffplay is not None,
         "있음 — 서버 기기에서도 재생 가능" if ffplay
         else "없음 — 브라우저 클라이언트로 들으면 됩니다")
    )

    assets = list(Path(ASSETS_DIR).glob("*.jpg"))
    checks.append(
        ("시뮬레이터 이미지", bool(assets),
         f"{len(assets)}장" if assets else "없음 → python -m tutor.scripts.gen_assets")
    )

    print("사전 점검:")
    for name, ok, detail in checks:
        print(f"  {'OK  ' if ok else '주의'} {name:<22} {detail}")

    ip = lan_ip()
    tls_port = settings.tls_listen_port
    print(f"\n서버 주소: ws://{ip}:{settings.ws_port}  (브라우저: http://localhost:{settings.ws_port}/)")
    if settings.tls_enabled:
        print(f"폰 카메라: https://{ip}:{tls_port}/phone")
    else:
        print("폰 카메라: 꺼짐 → python -m tutor.scripts.make_cert 로 인증서를 만들고 .env에 추가")
    print(f"\n{firewall_hint(settings.ws_port)}")
    print(
        f"  폰이 같은 Wi-Fi에 있는지 확인:  폰 브라우저에서  http://{ip}:{settings.ws_port}/\n"
    )

    print(
        "데모 순서:\n"
        "  1. python server.py\n"
        "  2. 브라우저에서 http://localhost:%d/ 열고 시작 누르기 (마이크·스피커)\n"
        "  3. 폰에서 https://%s:%d/phone 열고 시작 → 서버 로그에 'camera connected'\n"
        "     (폰 없이 확인: python -m simulator.camera_device --server ws://localhost:%d "
        "--images simulator/assets/lin_001_wrong_sign.jpg)\n"
        '  4. 문제를 카메라에 비추고 "힌트 주세요"라고 말하기\n'
        % (settings.ws_port, ip, tls_port, settings.ws_port)
    )


if __name__ == "__main__":
    main()

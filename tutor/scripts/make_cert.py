"""A self-signed certificate for the phone camera page.

    python -m tutor.scripts.make_cert

`getUserMedia` only exists in a secure context. A phone opening
http://192.168.x.x:8765/phone has none — `navigator.mediaDevices` is not
merely blocked there, it is undefined — so the camera page can never work over
the plain port. This mints a certificate for THIS machine's LAN address and
prints the two .env lines that switch the TLS listener on.

The subjectAltName is the whole point: browsers stopped reading CN years ago,
and a certificate without an `IP:` SAN entry is rejected for an IP address URL
no matter how many warnings the student clicks through.

Self-signed means one warning on the phone, once ("고급 → 계속"). That is the
price of a LAN address; the alternative is a public tunnel.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from tutor.config import PROJECT_ROOT, load_settings
from tutor.console import soften_stdout
from tutor.scripts.live_demo import lan_ip

CERT_DIR = PROJECT_ROOT / "certs"
DAYS = 825  # the longest a leaf certificate may live before browsers refuse it


def san_entries(ip: str) -> list[str]:
    """localhost too: the same cert then serves a laptop test of the page."""
    return [f"IP:{ip}", "IP:127.0.0.1", "DNS:localhost"]


def _with_cryptography(cert: Path, key: Path, ip: str) -> bool:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return False

    import datetime
    import ipaddress

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Visual Socratic Tutor")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private.public_key())
        .serial_number(x509.random_serial_number())
        # a minute of backdating covers a phone whose clock runs slightly behind
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=DAYS))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.IPAddress(ipaddress.ip_address(ip)),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.DNSName("localhost"),
                ]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(private, hashes.SHA256())
    )
    key.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    return True


def openssl_command(cert: Path, key: Path, ip: str) -> list[str]:
    return [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(cert), "-days", str(DAYS),
        "-subj", "/CN=Visual Socratic Tutor",
        "-addext", "subjectAltName=" + ",".join(san_entries(ip)),
    ]


def _with_openssl(cert: Path, key: Path, ip: str) -> bool:
    if shutil.which("openssl") is None:
        return False
    # openssl scribbles key-generation dots over stderr; worth seeing only if it fails
    done = subprocess.run(openssl_command(cert, key, ip), capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"openssl 실패 (exit {done.returncode}):\n{done.stderr}")
    return True


def write_selfsigned(cert: Path, key: Path, ip: str) -> bool:
    """Whichever backend this machine has. False if it has neither."""
    cert.parent.mkdir(parents=True, exist_ok=True)
    return _with_cryptography(cert, key, ip) or _with_openssl(cert, key, ip)


def main() -> None:
    soften_stdout()  # this docstring becomes --help, and cp949 cannot hold it
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=None, help="LAN address to certify (default: detected)")
    parser.add_argument("--out", type=Path, default=CERT_DIR)
    args = parser.parse_args()

    settings = load_settings()
    ip = args.ip or lan_ip()
    cert, key = args.out / "tutor.crt", args.out / "tutor.key"

    if not write_selfsigned(cert, key, ip):
        sys.exit(
            "인증서를 만들 도구가 없습니다. 둘 중 하나를 하세요:\n"
            "  pip install -e \".[phone]\"        (권장)\n"
            "  또는 openssl 설치 후 직접 실행:\n    "
            + " ".join(openssl_command(cert, key, ip))
        )

    port = settings.tls_listen_port
    print(f"인증서 생성 완료 ({ip} 용, {DAYS}일):")
    print(f"  {cert}")
    print(f"  {key}")
    print("\n.env 에 아래 두 줄을 추가하세요:\n")
    print(f"    TLS_CERT={cert}")
    print(f"    TLS_KEY={key}")
    print("\n그 다음 서버를 다시 띄우고, 폰에서 (같은 Wi-Fi):\n")
    print(f"    https://{ip}:{port}/phone")
    # No em dashes below: a Windows console on cp949 cannot encode them.
    print(
        "\n처음 한 번은 인증서 경고가 뜹니다. 자체 서명이라 정상입니다.\n"
        "  Chrome: 고급 → 계속 진행 / Safari: 세부사항 → 웹사이트 방문\n"
        f"  {port} 인바운드 방화벽 허용도 필요할 수 있습니다 "
        f"(python -m tutor.scripts.live_demo 참고).\n"
    )


if __name__ == "__main__":
    main()

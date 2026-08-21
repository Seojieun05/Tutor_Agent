"""Visual Socratic Tutor server entrypoint.

Reads .env at startup (see .env comments), then serves the device WebSocket.
Runs in echo mode (no API calls, canned Korean hints) when XAI_API_KEY is unset.
"""

from tutor.config import load_settings
from tutor.server.app import main

# 실행 진입점: .env를 읽어 Settings를 만들고 디바이스 WebSocket 서버를 띄운다.
if __name__ == "__main__":
    main(load_settings())

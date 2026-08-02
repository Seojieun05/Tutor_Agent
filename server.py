"""Visual Socratic Tutor server entrypoint.

Reads .env at startup (see .env comments), then serves the device WebSocket.
Runs in echo mode (no API calls, canned Korean hints) when XAI_API_KEY is unset.
"""

from tutor.config import load_settings
from tutor.server.app import main

if __name__ == "__main__":
    main(load_settings())

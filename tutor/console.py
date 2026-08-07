"""Printing to a console whose encoding cannot hold what we want to print.

A Korean Windows console is cp949: it has no em dash and no emoji, and `print`
raises UnicodeEncodeError rather than degrading. That has killed the startup
banner and the echo-mode tutor voice on this project already — losing one
character is a nuisance, dying because of it is not acceptable.

Only for text meant for a human watching a terminal. Logging has its own
handling, and anything that must survive byte-for-byte belongs in a file.
"""

from __future__ import annotations

import sys


def soften_stdout() -> None:
    """Let this process print characters the console cannot encode.

    Changes only the error handler, NOT the encoding: Korean still renders
    correctly on cp949, and the few characters it lacks become '?' instead of a
    UnicodeEncodeError. Call it from a script's entry point — argparse writes
    --help straight to sys.stdout, so say() cannot cover that one.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (pytest capture, a pipe wrapper)


def say(line: str, stream=None) -> None:
    """Print a line, replacing whatever the console cannot encode.

    flush: redirected to a log file the output would otherwise sit in the
    buffer, and it is usually the one thing the operator is waiting to read.
    """
    stream = stream or sys.stdout
    try:
        print(line, file=stream, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        print(line.encode(encoding, "replace").decode(encoding), file=stream, flush=True)

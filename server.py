"""FastAPI service to run the session title generator on Render web services."""

from __future__ import annotations

import threading
from typing import Optional

from fastapi import FastAPI

from session_title_generator import run_generator_loop

app = FastAPI(title="Session Title Generator", version="1.0.0")

_generator_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


@app.on_event("startup")
def start_generator() -> None:
    global _generator_thread
    if _generator_thread and _generator_thread.is_alive():
        return

    _stop_event.clear()
    _generator_thread = threading.Thread(
        target=run_generator_loop,
        kwargs={"start_offset": 0, "stop_event": _stop_event},
        daemon=True,
        name="session-title-generator",
    )
    _generator_thread.start()


@app.on_event("shutdown")
def stop_generator() -> None:
    _stop_event.set()
    if _generator_thread and _generator_thread.is_alive():
        _generator_thread.join(timeout=15)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "running",
        "message": "Session title generator is active."
    }

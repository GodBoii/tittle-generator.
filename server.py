"""FastAPI service to run the session title generator on Render web services."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from fastapi import FastAPI

from session_title_generator import run_generator_loop

logger = logging.getLogger(__name__)

app = FastAPI(title="Session Title Generator", version="1.0.0")

_generator_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _run_with_exception_handling() -> None:
    """Wrapper to catch and log any exceptions in the generator thread."""
    try:
        logger.info("🧵 Generator thread starting...")
        run_generator_loop(start_offset=0, stop_event=_stop_event)
        logger.info("🧵 Generator thread exited normally")
    except Exception as exc:
        logger.error(f"💥 FATAL: Generator thread crashed: {exc}")
        logger.exception("Full traceback:")


@app.on_event("startup")
def start_generator() -> None:
    global _generator_thread
    
    if _generator_thread and _generator_thread.is_alive():
        logger.info("🧵 Generator thread already running")
        return

    logger.info("🧵 Starting new generator thread...")
    _stop_event.clear()
    _generator_thread = threading.Thread(
        target=_run_with_exception_handling,
        daemon=True,
        name="session-title-generator",
    )
    _generator_thread.start()
    logger.info(f"🧵 Thread started: {_generator_thread.name}, alive={_generator_thread.is_alive()}")


@app.on_event("shutdown")
def stop_generator() -> None:
    logger.info("🛑 Shutdown requested, stopping generator...")
    _stop_event.set()
    if _generator_thread and _generator_thread.is_alive():
        _generator_thread.join(timeout=15)
        logger.info("🛑 Generator thread stopped")


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    thread_alive = _generator_thread.is_alive() if _generator_thread else False
    return {
        "status": "running",
        "message": "Session title generator is active.",
        "thread_alive": thread_alive,
        "thread_name": _generator_thread.name if _generator_thread else None,
    }


@app.get("/status")
def status():
    """Detailed status endpoint for debugging."""
    return {
        "thread_exists": _generator_thread is not None,
        "thread_alive": _generator_thread.is_alive() if _generator_thread else False,
        "thread_name": _generator_thread.name if _generator_thread else None,
        "thread_daemon": _generator_thread.daemon if _generator_thread else None,
        "stop_event_set": _stop_event.is_set(),
    }

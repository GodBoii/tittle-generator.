"""Generate concise titles for newly created chat sessions.

This script finds the next `agno_sessions` row that has not yet received a
title in the `session_titles` table, extracts the first user message, sends it
to Google's Gemini model (via agno), and persists the generated 3–4 word title.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
import supabase
from agno.agent import Agent
from agno.models.google import Gemini


# -----------------------------------------------------------------------------
# Environment & client setup
# -----------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be configured.")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is required for Google AI Studio access.")

supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Constants & Gemini agent
# -----------------------------------------------------------------------------

MIN_TITLE_WORDS = 3
MAX_TITLE_WORDS = 4
AGNO_SESSIONS_TABLE = "agno_sessions"
SESSION_TITLES_TABLE = "session_titles"

title_agent = Agent(
    model=Gemini(id="gemini-2.5-flash-lite-preview-09-2025"),
    instructions=(
        "Generate a concise title of exactly three or four words that captures "
        "the user's intent. Respond with plain text only, without quotes or "
        "additional formatting."
    ),
    markdown=False,
    debug_mode=False,
)


# -----------------------------------------------------------------------------
# Data access helpers
# -----------------------------------------------------------------------------

def fetch_session_by_offset(offset: int) -> Optional[Dict[str, Any]]:
    """Return the session at the specified offset ordered by created_at."""

    response = (
        supabase_client.from_(AGNO_SESSIONS_TABLE)
        .select("session_id, user_id, session_data, runs, metadata, summary, created_at")
        .order("created_at", desc=False)
        .range(offset, offset)
        .execute()
    )

    sessions: List[Dict[str, Any]] = response.data or []

    if sessions:
        session = sessions[0]
        logger.info(
            "Fetched session %s (offset %d) for processing",
            session.get("session_id"),
            offset,
        )
        return session

    logger.info("No additional sessions available for processing (offset %d)", offset)
    return None


def session_title_exists(session_id: str, user_id: str) -> bool:
    """Check whether a title already exists for the given session/user."""

    try:
        result = (
            supabase_client.from_(SESSION_TITLES_TABLE)
            .select("session_id")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Supabase query failed while checking title existence: %s", exc)
        return False

    return bool(result.data)


def save_title_entry(session_id: str, user_id: str, title: str) -> None:
    """Insert a new title row into the session_titles table."""

    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "tittle": title,
    }
    logger.info("Saving title for session %s: %s", session_id, title)
    supabase_client.from_(SESSION_TITLES_TABLE).insert(payload).execute()


# -----------------------------------------------------------------------------
# Message extraction utilities
# -----------------------------------------------------------------------------

def extract_first_user_message(session_row: Dict[str, Any]) -> Optional[str]:
    """Extract the first user-authored message from stored payloads."""

    session_data = _normalize_json(session_row.get("session_data"))
    message = _extract_from_messages(session_data)
    if message:
        return message

    runs = _normalize_json(session_row.get("runs"))
    message = _extract_from_runs(runs)
    if message:
        return message

    metadata = _normalize_json(session_row.get("metadata"))
    if isinstance(metadata, dict):
        for key in ("first_user_message", "initial_prompt", "prompt"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    summary = _normalize_json(session_row.get("summary"))
    if isinstance(summary, dict):
        for key in ("first_user_message", "initial_prompt", "prompt"):
            value = summary.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if isinstance(summary, str) and summary.strip():
        return summary.strip()

    return None


def _normalize_json(value: Any) -> Any:
    """Decode JSON strings returned by Supabase into Python objects."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.debug("Failed to decode JSON string; returning raw value")
            return value
    return value


def _extract_from_messages(session_data: Any) -> Optional[str]:
    """Handle message-based payloads inside session_data."""

    if isinstance(session_data, dict):
        candidates: Iterable[Any] = (
            session_data.get("messages")
            or session_data.get("conversation")
            or []
        )
    elif isinstance(session_data, list):
        candidates = session_data
    else:
        return None

    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role") or entry.get("speaker")
        if (role or "").lower() != "user":
            continue
        content = entry.get("content") or entry.get("text") or entry.get("message")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
    return None


def _extract_from_runs(runs: Any) -> Optional[str]:
    """Handle nested run history payloads."""

    if not isinstance(runs, list):
        return None

    for run in runs:
        if not isinstance(run, dict):
            continue
        input_payload = run.get("input")
        candidate = _coerce_to_text(input_payload)
        if candidate:
            return candidate
    return None


def _coerce_to_text(value: Any) -> Optional[str]:
    """Convert various structures into plain text where possible."""

    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, dict):
        for key in ("input_content", "content", "text", "message"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if isinstance(value, list):
        for item in value:
            text = _coerce_to_text(item)
            if text:
                return text
    return None


# -----------------------------------------------------------------------------
# LLM interaction and processing
# -----------------------------------------------------------------------------

def clean_user_message(message: str) -> str:
    """Trim whitespace and collapse runs for a clean prompt."""

    cleaned = re.sub(r"\s+", " ", message.strip())
    return cleaned[:2000]


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", text) if token]


def generate_title_with_llm(message: str) -> Optional[str]:
    """Use Gemini to create a three-to-four-word title."""

    prompt = clean_user_message(message)
    if not prompt:
        return None

    try:
        run_output = title_agent.run(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini call failed: %s", exc)
        return None

    response = getattr(run_output, "content", run_output)
    if isinstance(response, list):
        response_text = " ".join(str(part) for part in response)
    else:
        response_text = str(response)

    candidate_words = _tokenize(response_text)
    if len(candidate_words) < MIN_TITLE_WORDS:
        candidate_words += _tokenize(prompt)

    if not candidate_words:
        return None

    if len(candidate_words) > MAX_TITLE_WORDS:
        candidate_words = candidate_words[:MAX_TITLE_WORDS]
    elif len(candidate_words) < MIN_TITLE_WORDS:
        candidate_words = candidate_words[:MIN_TITLE_WORDS]

    title = " ".join(word.capitalize() for word in candidate_words)
    return title[:120]


def process_next_session(offset: int) -> Tuple[bool, int]:
    """Process sessions starting at offset; returns (title_saved, next_offset)."""

    current_offset = offset

    while True:
        session = fetch_session_by_offset(current_offset)
        if not session:
            logger.debug("Reached end of sessions; restarting from beginning")
            return False, 0

        session_id = session.get("session_id")
        user_id = session.get("user_id")
        next_offset = current_offset + 1

        if not session_id or not user_id:
            logger.warning("Skipping session without IDs: %s", session)
            current_offset = next_offset
            continue

        if session_title_exists(session_id, user_id):
            logger.debug("Title already exists for session %s", session_id)
            current_offset = next_offset
            continue

        message = extract_first_user_message(session)
        if not message:
            logger.warning("No user message found for session %s; skipping", session_id)
            current_offset = next_offset
            continue

        title = generate_title_with_llm(message)
        if not title:
            logger.warning("LLM did not return a usable title for session %s", session_id)
            current_offset = next_offset
            continue

        try:
            save_title_entry(session_id, user_id, title)
            return True, next_offset
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to persist title for session %s: %s", session_id, exc)
            current_offset = next_offset


# -----------------------------------------------------------------------------
# Runner utilities
# -----------------------------------------------------------------------------

def run_generator_loop(
    *, start_offset: int = 0, stop_event: Optional[threading.Event] = None
) -> None:
    """Continuously process sessions until stop_event (if any) is set."""

    if stop_event:
        logger.info("Starting generator loop with external stop control")
    else:
        logger.info("Starting generator loop (CLI mode)")

    offset = start_offset
    processed = 0
    skipped = 0
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stop signal received; exiting generator loop")
            break

        saved, offset = process_next_session(offset)
        if saved:
            processed += 1
            logger.info(
                "Generated a title; total processed=%d, continuing with offset %d",
                processed,
                offset,
            )
        else:
            logger.info("Verified all sessions; restarting from beginning")
            offset = 0


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    try:
        run_generator_loop()
    except KeyboardInterrupt:
        logger.info("Session title generator stopped by user")


if __name__ == "__main__":
    main()

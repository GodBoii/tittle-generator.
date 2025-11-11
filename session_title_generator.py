"""Generate concise titles for newly created chat sessions.

This script finds the next `agno_sessions` row that has not yet received a
title in the `session_titles` table, extracts the first user message, sends it
to Google's Gemini model (via agno), and persists the generated 3–4 word title.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
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

# Silence noisy httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Set to DEBUG for detailed logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


# -----------------------------------------------------------------------------
# Constants & Gemini agent
# -----------------------------------------------------------------------------

MIN_TITLE_WORDS = 3
MAX_TITLE_WORDS = 4
AGNO_SESSIONS_TABLE = "agno_sessions"
SESSION_TITLES_TABLE = "session_titles"
TITLE_AGENT_MODEL_ID = "gemini-2.5-flash-lite-preview-09-2025"
TITLE_AGENT_INSTRUCTIONS = (
    "Generate a concise title of exactly three or four words that captures "
    "the user's intent. Respond with plain text only, without quotes or "
    "additional formatting."
)


def _build_title_agent() -> Agent:
    return Agent(
        model=Gemini(id=TITLE_AGENT_MODEL_ID),
        instructions=TITLE_AGENT_INSTRUCTIONS,
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
        return sessions[0]

    return None


def session_title_exists(session_id: str, user_id: str) -> bool:
    """Check whether a usable title already exists for the given session/user."""

    try:
        result = (
            supabase_client.from_(SESSION_TITLES_TABLE)
            .select("session_id, tittle")
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ Failed to check title existence: %s", exc)
        return False

    rows = result.data or []
    for row in rows:
        title_value = row.get("tittle")
        if isinstance(title_value, str) and title_value.strip():
            return True
    return False


def save_title_entry(
    session_id: str,
    user_id: str,
    title: str,
    *,
    session_created_at: Optional[Any] = None,
) -> None:
    """Insert a new title row into the session_titles table."""

    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "tittle": title,
    }
    
    # Store original session timestamp if available
    if session_created_at is not None:
        payload["session_created_at"] = session_created_at
    
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
    return value


def _normalize_created_at(value: Any) -> Optional[str]:
    """Convert Supabase created_at values into ISO-8601 strings."""

    if value is None:
        return None

    # Handle numeric timestamps (seconds since epoch)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None

        if stripped.isdigit():
            try:
                return datetime.fromtimestamp(float(stripped), tz=timezone.utc).isoformat()
            except (OverflowError, OSError, ValueError):
                return None

        # Accept common timestamp layouts from Supabase/Postgres
        normalized = stripped.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).astimezone(timezone.utc).isoformat()
        except ValueError:
            pass

        for fmt in (
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S.%f",
        ):
            try:
                dt = datetime.strptime(stripped, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                continue

        # Fall back to returning the original string; Postgres may still accept it
        return stripped

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

    agent: Optional[Agent] = None
    run_output = None
    try:
        agent = _build_title_agent()
        run_output = agent.run(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ LLM error: %s", str(exc)[:100])
        return None
    finally:
        agent = None
        gc.collect()

    if not run_output:
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

    # Encourage garbage collection before returning to keep memory stable.
    del run_output, response, candidate_words
    gc.collect()

    return title[:120]


def process_next_session(offset: int, total_sessions: int) -> Tuple[bool, int, str]:
    """Process one session at offset; returns (title_saved, next_offset, status_msg)."""

    logger.debug(f"🔍 fetch_session_by_offset({offset})")
    session = fetch_session_by_offset(offset)
    
    if not session:
        logger.info(f"📭 No session found at offset {offset} - reached end of data")
        return False, 0, "end_of_data"

    session_id = session.get("session_id")
    user_id = session.get("user_id")
    next_offset = offset + 1
    
    logger.debug(f"🔍 Session {session_id[:8]}... at offset {offset}")

    if not session_id or not user_id:
        logger.warning(f"⚠️  Session at offset {offset} missing IDs: session_id={session_id}, user_id={user_id}")
        return False, next_offset, "missing_ids"

    if session_title_exists(session_id, user_id):
        logger.debug(f"⏭️  Session {session_id[:8]}... already has title")
        return False, next_offset, "already_titled"

    message = extract_first_user_message(session)
    if not message:
        logger.warning(f"⚠️  Session {session_id[:8]}... has no extractable user message")
        logger.debug(f"🔍 Session data keys: {list(session.keys())}")
        return False, next_offset, "no_message"

    logger.debug(f"🔍 Generating title for message: {message[:50]}...")
    title = generate_title_with_llm(message)
    if not title:
        logger.warning(f"⚠️  LLM failed to generate title for session {session_id[:8]}...")
        return False, next_offset, "llm_failed"

    try:
        logger.debug(f"💾 Saving title: {title}")
        save_title_entry(
            session_id,
            user_id,
            title,
            session_created_at=session.get("created_at"),
        )
        return True, next_offset, f"✓ {title}"
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ Save failed [offset %d]: %s", offset, str(exc)[:100])
        logger.exception("Full save error:")
        return False, next_offset, "save_error"


# -----------------------------------------------------------------------------
# Runner utilities
# -----------------------------------------------------------------------------

def run_generator_loop(
    *, start_offset: int = 0, stop_event: Optional[threading.Event] = None
) -> None:
    """Continuously process sessions until stop_event (if any) is set."""

    logger.info("🚀 Title generator started")
    logger.info(f"🔧 Thread name: {threading.current_thread().name}")
    logger.info(f"🔧 Thread daemon: {threading.current_thread().daemon}")
    logger.info(f"🔧 Stop event provided: {stop_event is not None}")

    # Get total session count once
    try:
        count_response = supabase_client.from_(AGNO_SESSIONS_TABLE).select("session_id", count="exact").limit(1).execute()
        total_sessions = count_response.count or 0
        logger.info(f"📊 Total sessions in database: {total_sessions}")
    except Exception as exc:
        logger.warning(f"⚠️  Could not fetch session count: {exc}")
        total_sessions = 0

    offset = start_offset
    processed = 0
    skipped = 0
    errors = 0
    loop_count = 0
    iteration = 0
    
    logger.info(f"🔁 Starting main loop from offset {offset}")
    
    while True:
        iteration += 1
        
        if iteration % 10 == 0:
            logger.info(f"💓 Loop heartbeat: iteration {iteration}, offset {offset}")
        
        if stop_event and stop_event.is_set():
            logger.info("🛑 Stop signal received")
            break

        logger.debug(f"🔍 Processing offset {offset}")
        
        try:
            saved, next_offset, status = process_next_session(offset, total_sessions)
            logger.debug(f"🔍 Result: saved={saved}, next_offset={next_offset}, status={status}")
        except Exception as exc:
            logger.error(f"💥 FATAL ERROR in process_next_session at offset {offset}: {exc}")
            logger.exception("Full traceback:")
            # Continue to next offset on error
            offset += 1
            errors += 1
            continue
        
        if status == "end_of_data":
            loop_count += 1
            logger.info(f"🔄 Completed pass #{loop_count} | Processed: {processed} | Skipped: {skipped} | Errors: {errors}")
            logger.info(f"🔄 Restarting from offset 0...")
            offset = 0
            skipped = 0
            errors = 0
            gc.collect()
            continue
        
        if saved:
            processed += 1
            logger.info(f"[{offset}/{total_sessions}] {status}")
        else:
            if status in ("no_message", "llm_failed", "save_error", "missing_ids"):
                errors += 1
                logger.warning(f"[{offset}/{total_sessions}] ⚠️  {status}")
            else:
                skipped += 1
                if iteration <= 50:  # Log first 50 skips
                    logger.debug(f"[{offset}/{total_sessions}] {status}")
        
        offset = next_offset
        logger.debug(f"🔍 Next offset will be: {offset}")
        gc.collect()
    
    logger.info(f"🏁 Generator loop exited after {iteration} iterations")


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

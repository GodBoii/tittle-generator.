import json
import logging
from typing import Any

from session_title_generator import process_next_session

logger = logging.getLogger("vercel-session-title")
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def handler(request: Any):
    saved, next_offset = process_next_session()
    if saved:
        logger.info("Title generated; next offset %d", next_offset)
        body = {"status": "generated", "nextOffset": next_offset}
    else:
        logger.info("No sessions required title generation")
        body = {"status": "idle"}

    return json.dumps(body), 200, {"Content-Type": "application/json"}

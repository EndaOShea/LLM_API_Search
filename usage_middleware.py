"""Usage-counting middleware for the composite MCP server.

Logs one JSONL record per MCP request — just timestamp + path. No client
identification, no IPs, no hashes. Lets us answer "how many requests, and
when" without touching personal data.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

USAGE_LOG_PATH = Path(os.environ.get("USAGE_LOG_PATH", "/var/log/llm-mcp/usage.jsonl"))


class UsageLoggingMiddleware(BaseHTTPMiddleware):
    """Append one JSONL record per MCP request: {ts, path}."""

    def __init__(self, app, log_path: Path = USAGE_LOG_PATH):
        super().__init__(app)
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Only count MCP traffic — skip index page, /health, /stats, favicon.
        if request.url.path.endswith("/mcp"):
            record = {"ts": int(time.time()), "path": request.url.path}
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

        return response


def summarize_usage(log_path: Path = USAGE_LOG_PATH) -> dict:
    """Read the JSONL log and return total + windowed counts.

    Counts are computed by streaming the file; no in-memory state to drift.
    """
    if not log_path.exists():
        return {"total": 0, "today": 0, "last_7d": 0, "by_path": {}}

    now = int(time.time())
    day_start = now - 86_400
    week_start = now - 7 * 86_400

    total = today = last_7d = 0
    by_path: dict[str, int] = {}

    with log_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            ts = rec.get("ts", 0)
            if ts >= day_start:
                today += 1
            if ts >= week_start:
                last_7d += 1
            path = rec.get("path", "unknown")
            by_path[path] = by_path.get(path, 0) + 1

    return {
        "total": total,
        "today": today,
        "last_7d": last_7d,
        "by_path": by_path,
    }

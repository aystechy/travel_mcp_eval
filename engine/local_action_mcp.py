"""Local FastMCP action server: simulated side effects persisted to ``logs/simulated_actions.jsonl``.

Tools: ``book_hotel``, ``create_calendar_event``, ``send_confirmation``.
Nothing leaves the machine: bookings/events/emails are only appended to the JSONL log.
Confirmation IDs are deterministic (SHA-256 of the canonical arguments).

Run standalone:  python -m engine.local_action_mcp            (stdio)
                 python -m engine.local_action_mcp --http 8765 (Streamable HTTP)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "logs" / "simulated_actions.jsonl"
_LOCK = threading.Lock()


def _confirmation(prefix: str, payload: dict) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return f"{prefix}-{digest[:10].upper()}"


def create_action_server(log_path: str | os.PathLike | None = None, context: dict | None = None) -> FastMCP:
    """Build the action server. ``context`` (e.g. run_id/episode_id) is stamped onto every record."""
    path = Path(log_path or os.environ.get("TRAVEL_ACTIONS_LOG") or DEFAULT_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx = context if context is not None else {}
    mcp = FastMCP("local-actions")

    def _append(kind: str, payload: dict, confirmation: str) -> dict:
        rec = {"ts": time.time(), "kind": kind, "confirmation": confirmation, **ctx, "payload": payload}
        with _LOCK, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    @mcp.tool
    def book_hotel(hotel_id: str, hotel_name: str, city: str, checkin: str, checkout: str,
                   guests: int, guest_name: str, total_price: float | None = None) -> dict:
        """Simulate a hotel booking. Returns a confirmation number."""
        payload: dict[str, Any] = {"hotel_id": hotel_id, "hotel_name": hotel_name, "city": city,
                                   "checkin": checkin, "checkout": checkout, "guests": guests,
                                   "guest_name": guest_name, "total_price": total_price}
        conf = _confirmation("BK", payload)
        _append("booking", payload, conf)
        return {"status": "confirmed", "confirmation": conf, **payload}

    @mcp.tool
    def create_calendar_event(title: str, start_date: str, end_date: str, location: str = "",
                              description: str = "") -> dict:
        """Simulate creating a calendar event."""
        payload = {"title": title, "start_date": start_date, "end_date": end_date,
                   "location": location, "description": description}
        conf = _confirmation("CAL", payload)
        _append("calendar_event", payload, conf)
        return {"status": "created", "event_id": conf, **payload}

    @mcp.tool
    def send_confirmation(to: str, subject: str, body: str) -> dict:
        """Simulate sending a confirmation email (never actually sent)."""
        payload = {"to": to, "subject": subject, "body": body}
        conf = _confirmation("MSG", payload)
        _append("email", payload, conf)
        return {"status": "queued_simulated", "message_id": conf, "to": to, "subject": subject}

    return mcp


mcp = create_action_server()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", type=int, default=None, help="serve Streamable HTTP on this port")
    a = ap.parse_args()
    if a.http:
        mcp.run(transport="http", port=a.http)
    else:
        mcp.run()

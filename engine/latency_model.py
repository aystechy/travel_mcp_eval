"""Frozen, calibrated tool-latency model (stdlib only; no network, no waiting).

Simulated MCP tools answer in milliseconds. This module adds the delay a real server would have
caused, drawn from timings measured once by ``engine.calibrate_latency`` and frozen in
``config/latency_profile.json``. It is applied after an episode, so it never affects a decision.

* Draws are seeded by ``(profile, seed, server, tool, args)``: the same call gets the same delay under
  every policy (common random numbers), and a run is reproducible.
* Delays follow the executor: steps of one DAG level run concurrently (max), levels run in turn (sum).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_PROFILE = Path(__file__).resolve().parent.parent / "config" / "latency_profile.json"


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def profile_id(tools: dict) -> str:
    return hashlib.sha256(_canon(tools).encode()).hexdigest()[:12]


class LatencyModel:
    def __init__(self, profile: dict):
        self.tools: dict[str, dict] = profile["tools"]
        self.id: str = profile["profile_id"]
        if profile_id(self.tools) != self.id:
            raise ValueError("latency profile was edited after it was frozen (profile_id mismatch)")

    def draw(self, server: str, tool: str, args: dict | None, seed: int = 0) -> float | None:
        """Modelled delay in ms for one call, or None if the tool is not in the profile."""
        entry = self.tools.get(f"{server}.{tool}")
        if entry is None:
            return None
        samples = entry["samples_ms"]
        h = hashlib.sha256(_canon([self.id, seed, server, tool, args or {}]).encode()).hexdigest()
        return samples[int(h[:12], 16) % len(samples)]

    def execution_ms(self, trace: list[dict], seed: int = 0) -> float:
        """Modelled tool time of one execution: sum over DAG levels of the slowest call in the level."""
        level: dict[str, int] = {}
        slowest: dict[int, float] = {}
        for s in trace:  # the executor appends steps level by level, so dependencies come first
            lv = 1 + max((level.get(d, 0) for d in s.get("depends_on", [])), default=0)
            level[s["step_id"]] = lv
            # A call without recorded args never reached a server.
            if s.get("kind") == "tool" and s.get("status") != "skipped" and "args" in s:
                ms = self.draw(s["server"], s["tool"], s["args"], seed)
                if ms is not None:
                    slowest[lv] = max(slowest.get(lv, 0.0), ms)
        return round(sum(slowest.values()), 3)


def load_profile(path: str | Path | None = None) -> LatencyModel | None:
    """The frozen profile, or None if none exists."""
    p = Path(path or DEFAULT_PROFILE)
    return LatencyModel(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None

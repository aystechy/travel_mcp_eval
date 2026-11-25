"""Multi-server MCP client with a SHA-256 disk replay cache.

Transports
  * live    - Streamable HTTP via ``fastmcp.Client(url)`` (bearer token from ``auth_env``)
  * sim     - in-memory FastMCP simulators (``engine.sim_servers``)
  * actions - in-memory ``engine.local_action_mcp`` (side effects; NEVER replay-cached)

Source modes: ``sim`` | ``live`` (per-call fallback to sim) | ``replay`` (cache only) | ``auto``
(replay -> live -> sim). Every call returns ``(result, meta)`` where ``meta["source"]`` is one of
``sim | live | replay | replay_sim | sim_fallback``. ``replay`` means a recorded *live* response;
``replay_sim`` is a recorded simulator response, so simulated data can never pass as recorded live data.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from fastmcp import Client

from engine.local_action_mcp import create_action_server
from engine.sim_servers import BUILDERS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "servers.json"
DEFAULT_REPLAY_DIR = ROOT / "cache" / "replay"
REPLAY_KEY_VERSION = 1
SOURCES = ("sim", "live", "replay", "auto")


class ReplayMiss(KeyError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def replay_key(server: str, tool: str, args: dict) -> str:
    payload = {"v": REPLAY_KEY_VERSION, "server": server, "tool": tool, "args": args or {}}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


class ReplayCache:
    def __init__(self, directory: str | os.PathLike = DEFAULT_REPLAY_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, server: str, tool: str, args: dict, result: Any, source: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "server": server, "tool": tool, "args": args,
                                   "source": source, "ts": time.time(), "result": result},
                                  default=str), encoding="utf-8")
        os.replace(tmp, p)


def load_config(config: str | os.PathLike | dict | None = None) -> dict:
    if isinstance(config, dict):
        return config
    return json.loads(Path(config or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def live_is_configured(config: str | os.PathLike | dict | None = None) -> bool:
    """False while no server maps any canonical tool to a live one (live would be 100 % simulator)."""
    return any(s.get("tool_map") for s in load_config(config)["servers"].values())


def _extract(result: Any) -> Any:
    """Normalise a fastmcp CallToolResult into plain JSON data."""
    sc = getattr(result, "structured_content", None)
    if isinstance(sc, dict):
        if set(sc) == {"result"}:
            return sc["result"]
        return sc
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
    return getattr(result, "data", None)


class MCPProxy:
    def __init__(self, config: str | os.PathLike | dict | None = None, source: str = "sim",
                 replay_dir: str | os.PathLike | None = None, actions_log: str | os.PathLike | None = None,
                 record: bool = True, live_timeout: float = 20.0):
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        self.config = load_config(config)
        self.servers: dict[str, dict] = self.config["servers"]
        self.source = source
        self.replay = ReplayCache(replay_dir or DEFAULT_REPLAY_DIR)
        self.record = record
        self.live_timeout = live_timeout
        self.action_context: dict[str, Any] = {}
        self._actions_log = actions_log
        self._stack: AsyncExitStack | None = None
        self._sim: dict[str, Client] = {}
        self._live: dict[str, Client | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.stats = {"sim": 0, "live": 0, "replay": 0, "replay_sim": 0, "sim_fallback": 0}

    async def __aenter__(self) -> "MCPProxy":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack:
            await self._stack.__aexit__(*exc)
        self._sim.clear()
        self._live.clear()

    def _lock(self, name: str) -> asyncio.Lock:
        return self._locks.setdefault(name, asyncio.Lock())

    async def _sim_client(self, server: str) -> Client:
        async with self._lock("sim:" + server):
            if server not in self._sim:
                kind = self.servers[server]["sim"]
                app = (create_action_server(self._actions_log, self.action_context)
                       if kind == "actions" else BUILDERS[kind]())
                self._sim[server] = await self._stack.enter_async_context(Client(app))
            return self._sim[server]

    async def _live_client(self, server: str) -> Client | None:
        async with self._lock("live:" + server):
            if server in self._live:
                return self._live[server]
            cfg = self.servers[server]
            client = None
            if cfg.get("transport") == "streamable_http" and cfg.get("url"):
                token = os.environ.get(cfg["auth_env"]) if cfg.get("auth_env") else None
                if cfg.get("auth_env") and not token:
                    client = None  # credential missing -> sim fallback
                else:
                    try:
                        c = Client(cfg["url"], auth=token, timeout=self.live_timeout)
                        client = await asyncio.wait_for(self._stack.enter_async_context(c), self.live_timeout)
                    except Exception:  # noqa: BLE001 - connection failure -> sim fallback
                        client = None
            self._live[server] = client
            return client

    async def list_tools(self, server: str) -> list[dict]:
        client = await self._sim_client(server)
        tools = await client.list_tools()
        return [{"name": t.name, "description": (t.description or "").strip(),
                 "input_schema": getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)} for t in tools]

    async def catalog(self, include_actions: bool = True) -> dict[str, list[dict]]:
        """Canonical tool catalog (from the simulators, which define the canonical contract)."""
        names = [s for s in self.servers if include_actions or self.servers[s]["sim"] != "actions"]
        lists = await asyncio.gather(*(self.list_tools(s) for s in names))
        return dict(zip(names, lists))

    async def call(self, server: str, tool: str, args: dict | None = None) -> tuple[Any, dict]:
        if server not in self.servers:
            raise KeyError(f"unknown server {server!r}")
        args = dict(args or {})
        cfg = self.servers[server]
        is_action = cfg["sim"] == "actions"
        key = replay_key(server, tool, args)
        t0 = time.perf_counter()
        meta: dict[str, Any] = {"cache_key": key}

        def done(result: Any, source: str) -> tuple[Any, dict]:
            meta["source"] = source
            meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            self.stats[source] = self.stats.get(source, 0) + 1
            if self.record and not is_action and not source.startswith("replay"):
                old = None if source == "live" else self.replay.get(key)
                if not (old and old.get("source") == "live"):  # never replace a live recording with simulated data
                    self.replay.put(key, server, tool, args, result, source)
            return result, meta

        if not is_action and self.source in ("replay", "auto"):
            hit = self.replay.get(key)
            is_live = hit is not None and hit.get("source") == "live"
            if is_live or (hit is not None and self.source == "replay"):  # auto skips simulator recordings
                return done(hit["result"], "replay" if is_live else "replay_sim")
            if self.source == "replay":
                raise ReplayMiss(f"no replay entry for {server}.{tool} {canonical_json(args)}")

        if not is_action and self.source in ("live", "auto"):
            remote = (cfg.get("tool_map") or {}).get(tool)
            if remote:
                client = await self._live_client(server)
                if client is not None:
                    renames = (cfg.get("arg_map") or {}).get(tool, {})
                    live_args = {renames.get(k, k): v for k, v in args.items()}
                    try:
                        res = await client.call_tool(remote, live_args, timeout=self.live_timeout)
                        return done(_extract(res), "live")
                    except Exception as e:  # noqa: BLE001
                        meta["fallback_reason"] = f"live_error: {type(e).__name__}"
                else:
                    meta["fallback_reason"] = "live_unavailable"
            else:
                meta["fallback_reason"] = "unmapped_tool"
            client = await self._sim_client(server)
            res = await client.call_tool(tool, args)
            return done(_extract(res), "sim_fallback")

        client = await self._sim_client(server)
        res = await client.call_tool(tool, args)
        return done(_extract(res), "sim")


def catalog_fingerprint(catalog: dict[str, list[dict]]) -> str:
    """SHA-256 of tool names + input schemas; cached plans are bound to this fingerprint."""
    slim = {s: sorted((t["name"], canonical_json(t.get("input_schema"))) for t in tools)
            for s, tools in catalog.items()}
    return hashlib.sha256(canonical_json(slim).encode()).hexdigest()[:16]

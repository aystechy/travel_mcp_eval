"""One-time calibration of real MCP tool latency. Benchmark runs never import this module.

    python -m engine.calibrate_latency collect --calls 10   # repeat at other times of day
    python -m engine.calibrate_latency freeze               # write config/latency_profile.json

``collect`` makes slow, sequential, read-only searches and appends only timings to
``config/latency_samples.jsonl``: responses are discarded, because they are untrusted text and may carry
provider storage restrictions. It stops a server at once when throttled and after 3 errors in a row.
Timed: one tool call on an open session (the connection handshake is excluded).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

from engine.latency_model import DEFAULT_PROFILE, profile_id
from engine.mcp_proxy import load_config
from engine.perturbations import load_dataset
from engine.settings import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "config" / "latency_samples.jsonl"
TASKS = ROOT / "data" / "tasks_100.json"
MIN_SAMPLES = 10
SPACING = {"kiwi": 3.0, "trivago": 5.0, "ferryhopper": 3.0, "mapbox": 2.0}  # gap between calls (s)
# Ferry routes that returned sailings when the servers were checked.
FERRIES = [("Piraeus", i) for i in ("Santorini", "Mykonos", "Paros", "Naxos")] + [("Barcelona", "Menorca")]

# Tools that could not be measured reuse the samples of a comparable measured tool.
ASSUMED = {"peek.search_experiences": ("kiwi.search_flights", "server lists tools but every search failed"),
           "weather.forecast": ("mapbox.geocode", "no server URL configured")}


def _flight(t: dict, r: random.Random) -> dict:
    y, m, d = t["depart_date"].split("-")
    return {"flyFrom": t["origin"]["iata"], "flyTo": t["destination"]["gateway_iata"],
            "departureDate": f"{d}/{m}/{y}", "currency": "GBP",
            "max_sector_stopovers": t["constraints"].get("max_stops", 1)}


def _hotel(t: dict, r: random.Random) -> dict:
    d = t["destination"]
    return {"latitude": d["lat"], "longitude": d["lon"], "arrival": t["depart_date"], "departure": t["return_date"],
            "adults": t["travelers"], "rooms": (t["travelers"] + 1) // 2, "currency": "GBP", "country": "GB",
            "language": "EN_GB"}


def _ferry(t: dict, r: random.Random) -> dict:
    a, b = r.choice(FERRIES)
    return {"legs": [{"departureLocation": a, "arrivalLocation": b,
                      "departureDate": f"2027-{r.randint(5, 9):02d}-{r.randint(1, 28):02d}"}]}


def _directions(t: dict, r: random.Random) -> dict:
    d = t["destination"]
    return {"coordinates": [{"longitude": round(d["lon"] + r.uniform(-.06, .06), 4),
                             "latitude": round(d["lat"] + r.uniform(-.04, .04), 4)},
                            {"longitude": d["lon"], "latitude": d["lat"]}], "routing_profile": "mapbox/driving"}


def _geocode(t: dict, r: random.Random) -> dict:
    return {"q": t["destination"]["city"]}


# (server, canonical tool, live tool, argument builder)
PROBES: list[tuple[str, str, str, Callable[[dict, random.Random], dict]]] = [
    ("kiwi", "search_flights", "search-flight", _flight),
    ("trivago", "search_hotels", "trivago-accommodation-radius-search", _hotel),
    ("ferryhopper", "search_ferries", "search_ferry_trips", _ferry),
    ("mapbox", "directions", "directions_tool", _directions),
    ("mapbox", "geocode", "search_and_geocode_tool", _geocode),
]


def read_samples(path: Path = SAMPLES) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _live_client(cfg: dict, token: str | None):
    from fastmcp import Client
    return Client(cfg["url"], auth=token, timeout=40)


async def _probe_server(server: str, calls: int, tasks: list[dict], rng: random.Random, session: str, path: Path,
                        cfg: dict, client_factory, gap: float) -> None:
    token = os.environ.get(cfg["auth_env"]) if cfg.get("auth_env") else None
    if cfg.get("auth_env") and not token:
        print(f"{server}: skipped ({cfg['auth_env']} is not set)", flush=True)
        return
    errors = 0
    async with client_factory(cfg, token) as client:
        for canonical, live, build in [(c, lv, b) for s, c, lv, b in PROBES if s == server]:
            for i, task in enumerate(rng.sample(tasks, calls)):
                args = build(task, rng)
                error = None
                t0 = time.perf_counter()
                try:
                    res = await client.call_tool(live, args, raise_on_error=False)
                    if res.is_error:
                        error = " ".join(getattr(b, "text", "") or "" for b in res.content)[:500]
                except Exception as e:  # noqa: BLE001
                    error = f"{type(e).__name__}: {e}"
                ms = round((time.perf_counter() - t0) * 1000, 1)
                low = (error or "").lower()
                throttled = any(x in low for x in ("429", "too many requests", "rate limit"))
                row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "session": session, "server": server,
                       "tool": canonical, "ok": error is None, "ms": ms}
                if error is not None:  # category only: error text is untrusted and never stored
                    row["error"] = "throttled" if throttled else "tool_error"
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{server}.{canonical} {i + 1}/{calls} {row.get('error', 'ok')} {ms:.0f} ms", flush=True)
                errors = errors + 1 if error is not None else 0
                if throttled or errors >= 3:
                    print(f"{server}: stopped ({'throttled' if throttled else '3 errors in a row'})", flush=True)
                    return
                await asyncio.sleep(gap)


async def collect(calls: int, servers: set[str] | None = None, path: Path = SAMPLES,
                  config: dict | None = None, client_factory=_live_client, spacing: dict | None = None) -> None:
    cfg = load_config(config)["servers"]
    tasks = load_dataset(TASKS)["tasks"]
    session = time.strftime("%Y-%m-%dT%H:%M:%S")
    rng = random.Random(session)  # each session sends different queries
    for server in dict.fromkeys(p[0] for p in PROBES):
        if servers and server not in servers:
            continue
        try:
            await _probe_server(server, calls, tasks, rng, session, path, cfg[server], client_factory,
                                (spacing or SPACING).get(server, 3.0))
        except Exception as e:  # noqa: BLE001 - one unreachable server must not stop the others
            print(f"{server}: connection failed ({type(e).__name__})", flush=True)


def _stats(v: list[float]) -> dict:
    q = statistics.quantiles(v, n=20, method="inclusive")
    return {"median_ms": round(statistics.median(v), 1), "p95_ms": round(q[18], 1), "min_ms": v[0], "max_ms": v[-1]}


def build_profile(rows: list[dict], frozen_at: str) -> dict:
    tools: dict[str, dict] = {}
    for server, tool, live, _ in PROBES:
        mine = [r for r in rows if (r["server"], r["tool"]) == (server, tool)]
        ok = sorted(r["ms"] for r in mine if r["ok"])
        if len(ok) < MIN_SAMPLES:
            raise ValueError(f"{server}.{tool}: {len(ok)} good samples, need {MIN_SAMPLES}")
        tools[f"{server}.{tool}"] = {"basis": "measured", "live_tool": live, "n": len(ok),
                                     "errors": len(mine) - len(ok), **_stats(ok), "samples_ms": ok}
    for key, (like, why) in ASSUMED.items():
        src = tools[like]["samples_ms"]
        tools[key] = {"basis": "assumed", "assumed_like": like, "reason": why, "n": 0, "errors": 0,
                      **_stats(src), "samples_ms": src}
    return {"profile_id": profile_id(tools), "frozen_at": frozen_at, "unit": "ms",
            "collected": {"sessions": sorted({r["session"] for r in rows}), "first": min(r["ts"] for r in rows),
                          "last": max(r["ts"] for r in rows)},
            "method": "sequential read-only calls on an open Streamable HTTP session; handshake excluded; "
                      "failed calls excluded; responses discarded",
            "tools": tools}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Calibrate real MCP tool latency (one-time)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="make slow live calls and append timings")
    c.add_argument("--calls", type=int, default=10, choices=range(1, 31), metavar="1-30", help="calls per tool")
    c.add_argument("--servers", default="", help="comma-separated subset")
    f = sub.add_parser("freeze", help="write the frozen profile")
    f.add_argument("--replace", action="store_true", help="replace an existing frozen profile")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        asyncio.run(collect(a.calls, {s for s in a.servers.split(",") if s}))
        return 0
    if DEFAULT_PROFILE.exists() and not a.replace:
        print(f"{DEFAULT_PROFILE} is frozen. Use --replace to start a new calibration version.", file=sys.stderr)
        return 2
    profile = build_profile(read_samples(), time.strftime("%Y-%m-%d"))
    DEFAULT_PROFILE.write_text(json.dumps(profile, indent=1) + "\n", encoding="utf-8")
    print(f"froze profile {profile['profile_id']} ({profile['frozen_at']})")
    for k, v in profile["tools"].items():
        print(f"  {k:28s} {v['basis']:8s} n={v['n']:2d} median={v['median_ms']:.0f} p95={v['p95_ms']:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate ``data/tasks_100.json``: 100 feasible base tasks over 100 distinct routes.

    python data/build_tasks_100.py

Deterministic for a given SEED. 5 UK origins x 15 mainland cities = 75 city trips, plus 25 of the
40 origin x island routes. Every budget is the cheapest valid itinerary (simulator prices) plus a
random 15-60 % margin, so every unperturbed task is solvable in ``--source sim``. Rebuild the file
if simulator prices change. Prices and budgets are synthetic, not real-world prices.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastmcp import Client  # noqa: E402

from engine.plan_executor import recompute_total, validate_itinerary  # noqa: E402
from engine.sim_servers import BUILDERS, PLACES  # noqa: E402

SEED = 2027
OUT = ROOT / "data" / "tasks_100.json"
MARGIN = (1.15, 1.60)
ORIGINS = {"Manchester": "MAN", "London": "LHR", "Edinburgh": "EDI", "Birmingham": "BHX", "Bristol": "BRS"}
HUBS = {"Barcelona": "BCN", "Lisbon": "LIS", "Amsterdam": "AMS", "Rome": "FCO", "Berlin": "BER", "Paris": "CDG",
        "Madrid": "MAD", "Athens": "ATH", "Porto": "OPO", "Milan": "MXP", "Vienna": "VIE", "Prague": "PRG",
        "Dublin": "DUB", "Naples": "NAP", "Split": "SPU"}
ISLANDS = {"Santorini": ("ATH", "Piraeus"), "Mykonos": ("ATH", "Piraeus"), "Paros": ("ATH", "Piraeus"),
           "Naxos": ("ATH", "Piraeus"), "Ibiza": ("BCN", "Barcelona"), "Menorca": ("BCN", "Barcelona"),
           "Capri": ("NAP", "Naples"), "Hvar": ("SPU", "Split")}
TRAVELLERS = ["Alex Morgan", "Priya Patel", "Liam O'Connor", "Hannah Lindqvist", "David Chen", "Sofia Rossi",
              "Marcus Vance", "Chloe Dubois", "Tariq Al-Mansoor", "Eleanor Wright"]
PARTY = {1: "a solo traveller", 2: "two", 3: "three", 4: "four"}


def catalogue() -> dict[str, dict]:
    cities = {name: ("hub", gw, None) for name, gw in HUBS.items()}
    for name, (gw, port) in ISLANDS.items():
        cities[name] = ("island", gw, {"from_port": port, "to_port": name})
    return {n: {"city": n, "country": PLACES[n][2], "type": kind, "gateway_iata": gw, "ferry": ferry,
                "lat": CITY_COORDS.get(n, PLACES[n])[0], "lon": CITY_COORDS.get(n, PLACES[n])[1]}
            for n, (kind, gw, ferry) in cities.items()}


# PLACES holds the *port* under these names; the dataset needs the city centre.
CITY_COORDS = {"Naples": (40.8518, 14.2681), "Split": (43.5081, 16.4402)}


async def cheapest_itinerary(task: dict, servers: dict | None = None) -> dict | None:
    """Cheapest itinerary meeting the task's constraints (budget ignored), or None if there is none."""
    servers = servers or {s: BUILDERS[s]() for s in ("flights", "hotels", "ferries")}

    async def call(server: str, tool: str, **args):
        async with Client(servers[server]) as c:
            return (await c.call_tool(tool, args)).structured_content

    o, d, c = task["origin"]["iata"], task["destination"], task["constraints"]
    gw, dep, ret = d["gateway_iata"], task["depart_date"], task["return_date"]
    nights = (date.fromisoformat(ret) - date.fromisoformat(dep)).days
    out: dict = {}
    for key, a, b, day in (("flight_out", o, gw, dep), ("flight_back", gw, o, ret)):
        offers = (await call("flights", "search_flights", origin=a, destination=b, date=day))["offers"]
        out[key] = min((x for x in offers if x["stops"] <= c.get("max_stops", 99)),
                       key=lambda x: x["price"], default=None)
    hotels = (await call("hotels", "search_hotels", city=d["city"], checkin=dep, checkout=ret))["hotels"]
    out["hotel"] = min((h for h in hotels if h["rating"] >= c.get("min_hotel_rating", 0)),
                       key=lambda h: h["price_per_night"] or h["total_price"] / nights, default=None)
    if d["ferry"]:
        p, q = d["ferry"]["from_port"], d["ferry"]["to_port"]
        for key, a, b, day in (("ferry_out", p, q, dep), ("ferry_back", q, p, ret)):
            sailings = (await call("ferries", "search_ferries", from_port=a, to_port=b, date=day))["sailings"]
            out[key] = min(sailings, key=lambda x: x["price"], default=None)
    if any(v is None for v in out.values()):
        return None
    out["total_cost"] = recompute_total(task, out)
    return out


def routes(rng: random.Random) -> list[tuple[str, str]]:
    """100 distinct (origin, destination) routes; every 4th is an island, so any prefix is balanced."""
    hubs = [(o, d) for o in ORIGINS for d in HUBS]
    rng.shuffle(hubs)
    isl: list[tuple[str, str]] = []  # each island from 3 origins, the first drawn island from a 4th
    for name in rng.sample(list(ISLANDS), len(ISLANDS)):
        isl += [(o, name) for o in rng.sample(list(ORIGINS), 3 if isl else 4)]
    rng.shuffle(isl)
    return [isl[i // 4] if i % 4 == 3 else hubs[i - i // 4] for i in range(100)]


def describe(t: dict, nights: int) -> str:
    c, d = t["constraints"], t["destination"]
    text = f"{nights}-night trip from {t['origin']['city']} to {d['city']} for {PARTY[t['travelers']]}"
    if d["ferry"]:
        text += f", with ferry transfer from {d['ferry']['from_port']}"
    if c.get("max_stops") == 0:
        text += ", direct flights only"
    if "min_hotel_rating" in c:
        text += f", hotel rated {c['min_hotel_rating']} or higher"
    return text


async def build() -> dict:
    rng = random.Random(SEED)
    servers = {s: BUILDERS[s]() for s in ("flights", "hotels", "ferries")}
    cities = catalogue()
    nights = [3] * 25 + [4] * 30 + [5] * 25 + [7] * 20
    party = [1] * 17 + [2] * 51 + [3] * 16 + [4] * 16
    ratings = [4.0] * 20 + [3.5] * 14 + [None] * 66
    for pool in (nights, party, ratings):
        rng.shuffle(pool)
    tasks, direct = [], 0
    for i, (o, d) in enumerate(routes(rng)):
        who = rng.choice(TRAVELLERS)
        t = {"id": f"T{i + 1:03d}", "intent": "trip", "description": "",
             "origin": {"city": o, "iata": ORIGINS[o]}, "destination": json.loads(json.dumps(cities[d])),
             "depart_date": "", "return_date": "", "travelers": party[i], "budget": 0, "currency": "GBP",
             "constraints": {"max_stops": 1},
             "traveler": {"name": who,
                          "email": who.lower().replace("'", "").replace("-", "").replace(" ", ".") + "@example.com"}}
        if ratings[i]:
            t["constraints"]["min_hotel_rating"] = ratings[i]

        def redate() -> None:
            dep = date(2027, 4, 1) + timedelta(days=rng.randrange(200))
            t["depart_date"], t["return_date"] = dep.isoformat(), (dep + timedelta(days=nights[i])).isoformat()

        redate()
        # 25 direct-only tasks, on mainland routes that have direct flights both ways
        if cities[d]["type"] == "hub" and direct < 25 and rng.random() < 0.45:
            t["constraints"]["max_stops"] = 0
            if await cheapest_itinerary(t, servers) is None:
                t["constraints"]["max_stops"] = 1
            else:
                direct += 1
        for _ in range(20):  # on some dates no hotel meets the rating: draw new dates
            best = await cheapest_itinerary(t, servers)
            if best is not None:
                break
            redate()
        else:
            raise RuntimeError(f"{t['id']}: no feasible itinerary")
        t["budget"] = math.ceil(best["total_cost"] * rng.uniform(*MARGIN) / 10) * 10
        t["description"] = describe(t, nights[i])
        tasks.append(t)
        report = validate_itinerary(t, best)
        if not report.ok:
            raise RuntimeError(f"{t['id']}: {report.violations}")
    return {"metadata": {"dataset_name": "travel-mcp-tasks-100", "seed": SEED, "num_tasks": len(tasks),
                         "num_cities": len(cities), "budget_margin": list(MARGIN),
                         "budget_basis": "cheapest valid itinerary at simulator prices (--source sim)",
                         "generator": "data/build_tasks_100.py"},
            "cities": cities, "tasks": tasks}


if __name__ == "__main__":
    data = asyncio.run(build())
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(data['tasks'])} tasks to {OUT}")

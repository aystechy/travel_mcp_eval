"""LLM planning: prompt construction, plan parsing, and a deterministic heuristic (mock) planner.

The heuristic planner deliberately behaves like a typical LLM planner: it writes *literal* task
values into tool args and derives a literal hotel price cap from the budget. The plan cache's
abstraction step lifts exact literals back into ``task.*`` references; derived literals (the cap)
stay stale under perturbation - a realistic source of reuse failures.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date

from pydantic import ValidationError

from engine.plan_executor import Plan, step_dependencies, _STEP_REF_RE, _strings

SYSTEM_PROMPT = """You are a travel-planning compiler. Output ONLY one JSON object: an executable plan.

PLAN SCHEMA
{"version":1,
 "steps":[ {"id":"snake_id","kind":"tool","server":"<server>","tool":"<tool>","args":{...}}
          |{"id":"...","kind":"select","query":"<JMESPath>"}
          |{"id":"...","kind":"compute","expr":"<arithmetic>","vars":{"name": <value>}} ],
 "output":{...}, "actions":[{"id":"...","server":"actions","tool":"<tool>","args":{...}}]}

VALUES (in args, vars, output, action args)
- {"$jp":"<JMESPath>"} evaluated over {"task":<task>,"steps":{<step_id>:<result>},"output":<output>}
- Prefer {"$jp":"task.depart_date"} etc. over copying task values.
- Inside JMESPath queries write task values as ${task.path}, e.g.
  "steps.flights_out.offers[?stops <= ${task.constraints.max_stops}] | sort_by(@, &price) | [0]"
- Strings may interpolate: "Trip to {{ task.destination.city }}".
- compute expr: numbers, + - * / // %, comparisons, and/or, x if c else y, and functions
  min max abs round sum len ceil floor days_between(a,b) safe_div(a,b,default) coalesce(a,b,...).

REQUIRED OUTPUT KEYS
flight_out, flight_back (offer objects), hotel (hotel object), total_cost (number), nights, currency,
plus ferry_out/ferry_back when task.destination.ferry is set (fly to gateway_iata, then ferry
from_port -> to_port; reverse on return), experience if constraints.include_experience,
transfer_minutes (directions gateway_iata -> city, or -> ferry from_port) if constraints.max_transfer_minutes.

COST MODEL (GBP; validated independently)
total_cost = (flight_out.price + flight_back.price + ferry prices + experience.price) * travelers
           + hotel_nightly * nights * rooms
hotel_nightly = coalesce(hotel.price_per_night, hotel.total_price / nights); rooms = ceil(travelers / 2).
Respect task.budget and every key in task.constraints (max_stops, min_hotel_rating, refundable_only,
include_experience, max_transfer_minutes). Flights go origin.iata -> destination.gateway_iata.

ACTIONS (run only after validation): book_hotel, create_calendar_event, send_confirmation (to traveler.email).

TOOL RESULTS (what steps.<tool_step_id> contains)
"""

# Result shape of each tool (item fields of the list, or the object's own fields). Tested against the simulators.
TOOL_RESULTS = {
    "search_flights": "{offers:[{id,carrier,flight_no,from,to,date,depart_time,stops,duration_min,price}]}",
    "search_hotels": "{nights,hotels:[{id,name,city,rating,price_per_night|null,total_price,refundable,"
                     "checkin,checkout}]}",
    "search_ferries": "{sailings:[{id,operator,from_port,to_port,date,depart_time,duration_min,price}]}",
    "search_experiences": "{experiences:[{id,title,category,city,date,rating,price}]}",
    "directions": "{found,distance_km,duration_min}",
    "geocode": "{found,name,lat,lon,country}",
    "forecast": "{date,temp_max_c,temp_min_c,precip_mm,summary}",
}

# Deliberately incomplete (one-way, no constraints, ferry, experience or actions): it teaches the plan language
# without giving away a benchmark answer.
EXAMPLE_PLAN = {
    "version": 1,
    "steps": [
        {"id": "flights", "kind": "tool", "server": "kiwi", "tool": "search_flights",
         "args": {"origin": {"$jp": "task.origin.iata"}, "destination": {"$jp": "task.destination.gateway_iata"},
                  "date": {"$jp": "task.depart_date"}}},
        {"id": "flight", "kind": "select", "query": "steps.flights.offers | sort_by(@, &price) | [0]"},
        {"id": "total", "kind": "compute", "expr": "round(price * n, 2)",
         "vars": {"price": {"$jp": "steps.flight.price"}, "n": {"$jp": "task.travelers"}}},
    ],
    "output": {"flight_out": {"$jp": "steps.flight"}, "total_cost": {"$jp": "steps.total"}},
}

SYSTEM_PROMPT += "\n".join(f"{k}: {v}" for k, v in TOOL_RESULTS.items()) + """

RULES
- You do not know any price, flight or hotel. Never invent them: every output object and total_cost must be a
  {"$jp":"steps.<id>"} reference to a select or compute step that uses tool results.
- Pick one item from a result list with a select step.

EXAMPLE of the plan language only (one-way flight; your plan must cover every required key and constraint):
""" + json.dumps(EXAMPLE_PLAN, separators=(",", ":")) + "\n"


def compact_catalog(catalog: dict[str, list[dict]], include_actions: bool = True) -> dict:
    out = {}
    for server, tools in catalog.items():
        if not include_actions and server == "actions":
            continue
        out[server] = {}
        for t in tools:
            props = (t.get("input_schema") or {}).get("properties", {})
            req = set((t.get("input_schema") or {}).get("required", []))
            out[server][t["name"]] = {k: (v.get("type") or "any") + ("" if k in req else "?")
                                      for k, v in props.items()}
    return out


def planner_task_view(task: dict) -> dict:
    """Task fields shown to the planner (drops bookkeeping fields)."""
    return {k: v for k, v in task.items() if k not in ("perturbation", "base_id")}


def build_user_prompt(task: dict, catalog: dict, feedback: list[str] | None = None,
                      previous: str | None = None) -> str:
    parts = ["TOOLS (server -> tool -> args; ? = optional):",
             json.dumps(compact_catalog(catalog), separators=(",", ":")),
             "TASK:", json.dumps(planner_task_view(task), separators=(",", ":"))]
    if previous:
        parts += ["YOUR PREVIOUS PLAN:", previous]
    if feedback:
        parts += ["PREVIOUS PLAN FAILED VALIDATION. Fix these violations:", "\n".join(f"- {f}" for f in feedback)]
    parts.append("Return the plan JSON now.")
    return "\n".join(parts)


class PlanParseError(ValueError):
    pass


GROUNDED_KEYS = ("flight_out", "flight_back", "hotel", "ferry_out", "ferry_back", "experience", "total_cost",
                 "transfer_minutes")


def check_grounded(plan: Plan) -> None:
    """Reject outputs the model wrote itself instead of deriving from tool results (null is allowed)."""
    by_id = {s.id: s for s in plan.steps}

    def uses_tool(sid: str) -> bool:
        seen, stack = set(), [sid]
        while stack:
            step = by_id.get(stack.pop())
            if step is None or step.id in seen:
                continue
            if step.kind == "tool":
                return True
            seen.add(step.id)
            stack += step_dependencies(step, set(by_id))
        return False

    for key in GROUNDED_KEYS:
        value = plan.output.get(key)
        if value is None:
            continue
        ref = isinstance(value, dict) and ("$jp" in value or "$expr" in value)
        if not ref or not any(uses_tool(i) for s in _strings(value) for i in _STEP_REF_RE.findall(s)):
            raise PlanParseError(f'output.{key} must be {{"$jp":"steps.<id>"}} pointing to a step that uses tool '
                                 f"results, not a value you wrote")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_plan(text: str) -> Plan:
    """Extract and validate a plan from raw model text (tolerates code fences / prose)."""
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text or "")] + [text or ""]
    last_err: Exception | None = None
    for c in candidates:
        start, end = c.find("{"), c.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            data = json.loads(c[start:end + 1])
            if isinstance(data, dict) and "plan" in data and "steps" not in data:
                data = data["plan"]
            plan = Plan.model_validate(data)
            check_grounded(plan)
            return plan
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
    raise PlanParseError(f"could not parse plan: {last_err}")


# --------------------------------------------------------------------------------------------
# Heuristic planner (deterministic stand-in for an LLM)
# --------------------------------------------------------------------------------------------
HOTEL_BUDGET_SHARE = 0.5


def heuristic_plan(task: dict) -> dict:
    o, d = task["origin"], task["destination"]
    c = task.get("constraints") or {}
    dep, ret, n = task["depart_date"], task["return_date"], int(task["travelers"])
    gw, ferry = d["gateway_iata"], d.get("ferry")
    nights = (date.fromisoformat(ret) - date.fromisoformat(dep)).days
    rooms = max(1, math.ceil(n / 2))
    # Derived literal: hotel spend cap per room for the whole stay (stale if budget changes).
    hotel_cap = round(task["budget"] * HOTEL_BUDGET_SHARE / rooms, 2)

    steps: list[dict] = [{"id": "geo_dest", "kind": "tool", "server": "mapbox", "tool": "geocode",
                          "args": {"query": d["city"]}}]
    fo = {"origin": o["iata"], "destination": gw, "date": dep}
    fb = {"origin": gw, "destination": o["iata"], "date": ret}
    if "max_stops" in c:
        fo["max_stops"] = fb["max_stops"] = c["max_stops"]
    hq = {"city": d["city"], "checkin": dep, "checkout": ret}
    if "min_hotel_rating" in c:
        hq["min_rating"] = c["min_hotel_rating"]
    steps += [
        {"id": "flights_out", "kind": "tool", "server": "kiwi", "tool": "search_flights", "args": fo},
        {"id": "flights_back", "kind": "tool", "server": "kiwi", "tool": "search_flights", "args": fb},
        {"id": "hotels", "kind": "tool", "server": "trivago", "tool": "search_hotels", "args": hq},
        {"id": "weather", "kind": "tool", "server": "weather", "tool": "forecast", "on_error": "null",
         "args": {"lat": {"$jp": "steps.geo_dest.lat"}, "lon": {"$jp": "steps.geo_dest.lon"}, "date": dep}},
    ]
    if ferry:
        steps += [
            {"id": "ferry_out_search", "kind": "tool", "server": "ferryhopper", "tool": "search_ferries",
             "args": {"from_port": ferry["from_port"], "to_port": ferry["to_port"], "date": dep}},
            {"id": "ferry_back_search", "kind": "tool", "server": "ferryhopper", "tool": "search_ferries",
             "args": {"from_port": ferry["to_port"], "to_port": ferry["from_port"], "date": ret}},
        ]
    if "include_experience" in c:
        steps.append({"id": "experiences", "kind": "tool", "server": "peek", "tool": "search_experiences",
                      "args": {"city": d["city"], "date": dep, "category": c["include_experience"]}})
    if "max_transfer_minutes" in c:
        steps.append({"id": "transfer", "kind": "tool", "server": "mapbox", "tool": "directions",
                      "args": {"from_place": gw, "to_place": ferry["from_port"] if ferry else d["city"],
                               "profile": "driving"}})

    stops = "[?stops <= ${task.constraints.max_stops}]" if "max_stops" in c else ""
    hotel_filters = []
    if "min_hotel_rating" in c:
        hotel_filters.append("rating >= ${task.constraints.min_hotel_rating}")
    if c.get("refundable_only"):
        hotel_filters.append("refundable == ${task.constraints.refundable_only}")
    hf = " && ".join(hotel_filters)
    capped = " && ".join([*hotel_filters, f"total_price <= `{hotel_cap}`"])
    steps += [
        {"id": "best_out", "kind": "select", "query": f"steps.flights_out.offers{stops} | sort_by(@, &price) | [0]"},
        {"id": "best_back", "kind": "select", "query": f"steps.flights_back.offers{stops} | sort_by(@, &price) | [0]"},
        {"id": "hotel_pick", "kind": "select",
         "query": f"steps.hotels.hotels[?{capped}] | sort_by(@, &rating) | reverse(@) | [0]"},
        {"id": "hotel_cheapest", "kind": "select",
         "query": f"steps.hotels.hotels{'[?' + hf + ']' if hf else ''} | sort_by(@, &total_price) | [0]"},
        {"id": "hotel", "kind": "select", "query": "steps.hotel_pick || steps.hotel_cheapest"},
    ]
    if ferry:
        steps += [
            {"id": "ferry_out", "kind": "select", "query": "steps.ferry_out_search.sailings | sort_by(@, &price) | [0]"},
            {"id": "ferry_back", "kind": "select", "query": "steps.ferry_back_search.sailings | sort_by(@, &price) | [0]"},
        ]
    if "include_experience" in c:
        steps.append({"id": "experience", "kind": "select",
                      "query": "steps.experiences.experiences[?category == ${task.constraints.include_experience}]"
                               " | sort_by(@, &price) | [0]"})
    steps += [
        {"id": "nights", "kind": "compute", "expr": "days_between(ret, dep)", "vars": {"ret": ret, "dep": dep}},
        {"id": "hotel_nightly", "kind": "compute", "on_error": "null", "expr": "coalesce(ppn, safe_div(total, nights))",
         "vars": {"ppn": {"$jp": "steps.hotel.price_per_night"}, "total": {"$jp": "steps.hotel.total_price"},
                  "nights": {"$jp": "steps.nights"}}},
        {"id": "total", "kind": "compute", "on_error": "null",
         "expr": "round((fo + fb + coalesce(fyo, 0) + coalesce(fyb, 0) + coalesce(ex, 0)) * n"
                 " + nightly * nights * rooms, 2)",
         "vars": {"fo": {"$jp": "steps.best_out.price"}, "fb": {"$jp": "steps.best_back.price"},
                  "fyo": {"$jp": "steps.ferry_out.price"}, "fyb": {"$jp": "steps.ferry_back.price"},
                  "ex": {"$jp": "steps.experience.price"}, "n": n, "rooms": rooms,
                  "nightly": {"$jp": "steps.hotel_nightly"}, "nights": {"$jp": "steps.nights"}}},
    ]
    output = {"flight_out": {"$jp": "steps.best_out"}, "flight_back": {"$jp": "steps.best_back"},
              "hotel": {"$jp": "steps.hotel"}, "nights": {"$jp": "steps.nights"},
              "total_cost": {"$jp": "steps.total"}, "currency": task.get("currency", "GBP"),
              "weather": {"$jp": "steps.weather"}}
    if ferry:
        output.update(ferry_out={"$jp": "steps.ferry_out"}, ferry_back={"$jp": "steps.ferry_back"})
    if "include_experience" in c:
        output["experience"] = {"$jp": "steps.experience"}
    if "max_transfer_minutes" in c:
        output["transfer_minutes"] = {"$jp": "steps.transfer.duration_min"}

    name, email = task["traveler"]["name"], task["traveler"]["email"]
    actions = [
        {"id": "book", "tool": "book_hotel", "args": {
            "hotel_id": {"$jp": "output.hotel.id"}, "hotel_name": {"$jp": "output.hotel.name"},
            "city": d["city"], "checkin": dep, "checkout": ret, "guests": n, "guest_name": name,
            "total_price": {"$jp": "output.hotel.total_price"}}},
        {"id": "calendar", "tool": "create_calendar_event", "args": {
            "title": f"Trip to {d['city']}", "start_date": dep, "end_date": ret, "location": d["city"],
            "description": "Outbound {{ output.flight_out.flight_no }}, return {{ output.flight_back.flight_no }}"}},
        {"id": "email", "tool": "send_confirmation", "args": {
            "to": email, "subject": f"Your trip to {d['city']} is booked",
            "body": f"Hi {name}, your stay at {{{{ output.hotel.name }}}} is confirmed. "
                    "Total {{ output.total_cost }} {{ output.currency }}."}},
    ]
    return {"version": 1, "steps": steps, "output": output, "actions": actions,
            "meta": {"planner": "heuristic"}}


def extract_task_from_prompt(prompt: str) -> dict:
    lines = prompt.splitlines()
    idx = lines.index("TASK:")
    return json.loads(lines[idx + 1])

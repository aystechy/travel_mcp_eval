"""Plan cache: abstraction (literal -> task reference), derived coverage, compatibility, stats.

INVARIANT: coverage/compatibility are derived from the plan structure, never from LLM metadata.
Only plans that passed validation are stored.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jmespath

from engine.plan_executor import Plan, _strings

# arg-name hints used to disambiguate literals equal to several task slots
_HINTS = {
    "origin": ["origin.iata", "destination.gateway_iata"],
    "destination": ["destination.gateway_iata", "origin.iata"],
    "date": ["depart_date", "return_date"],
    "checkin": ["depart_date"], "checkout": ["return_date"],
    "start_date": ["depart_date"], "end_date": ["return_date"],
    "ret": ["return_date"], "dep": ["depart_date"],
    "city": ["destination.city"], "query": ["destination.city"], "location": ["destination.city"],
    "from_port": ["destination.ferry.from_port", "destination.ferry.to_port"],
    "to_port": ["destination.ferry.to_port", "destination.ferry.from_port"],
    "from_place": ["destination.gateway_iata"],
    "to_place": ["destination.ferry.from_port", "destination.city"],
    "max_stops": ["constraints.max_stops"], "min_rating": ["constraints.min_hotel_rating"],
    "category": ["constraints.include_experience"],
    "guests": ["travelers"], "n": ["travelers"], "passengers": ["travelers"],
    "to": ["traveler.email"], "guest_name": ["traveler.name"], "budget": ["budget"],
}
# numeric literals are lifted only when the arg name names the slot (avoids 1 == travelers == max_stops)
_SUBSTRING_SLOTS = ["traveler.name", "destination.city", "origin.city"]
_CONSTRAINT_REF = re.compile(r"task\.constraints\.([A-Za-z_][A-Za-z0-9_]*)")
_TASK_REF = re.compile(r"task\.([A-Za-z_][A-Za-z0-9_.]*)")


def task_slots(task: dict) -> dict[str, Any]:
    slots: dict[str, Any] = {}

    def walk(prefix: str, obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, (str, int, float, bool)):
            slots[prefix] = obj

    walk("", {k: v for k, v in task.items() if k not in ("id", "base_id", "perturbation", "description")})
    return slots


def _lift_literal(key: str | None, value: Any, slots: dict[str, Any]) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    matches = [p for p, v in slots.items() if v == value and type(v) is type(value)]
    if not matches:
        return value
    hints = _HINTS.get(key or "", [])
    for h in hints:
        if h in matches:
            return {"$jp": f"task.{h}"}
    if isinstance(value, str) and len(value) >= 3:
        return {"$jp": f"task.{matches[0]}"}
    return value  # ambiguous short/numeric literal without a naming hint: leave as is


def _template_substrings(text: str, slots: dict[str, Any]) -> str:
    """Turn embedded task values (names, cities) into ``{{ task.* }}`` interpolations."""
    for s in sorted(_SUBSTRING_SLOTS, key=lambda s: -len(str(slots.get(s, "")))):
        v = slots.get(s)
        if isinstance(v, str) and len(v) >= 4 and v in text:
            text = text.replace(v, "{{ task.%s }}" % s)
    return text


def _abstract_value(key: str | None, value: Any, slots: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$jp"} or ("$expr" in value and set(value) <= {"$expr", "vars"}):
            if "vars" in value:
                return {**value, "vars": {k: _abstract_value(k, v, slots) for k, v in value["vars"].items()}}
            return value
        return {k: _abstract_value(k, v, slots) for k, v in value.items()}
    if isinstance(value, list):
        return [_abstract_value(key, v, slots) for v in value]
    lifted = _lift_literal(key, value, slots)
    if lifted is value and isinstance(value, str):
        return _template_substrings(value, slots)
    return lifted


def abstract_plan(plan: Plan | dict, task: dict) -> dict:
    """Return a task-parameterised template: literal task values become ``{"$jp": "task.*"}``."""
    p = copy.deepcopy(plan.model_dump() if isinstance(plan, Plan) else plan)
    slots = task_slots(task)
    for step in p["steps"]:
        step["args"] = _abstract_value(None, step.get("args") or {}, slots)
        step["vars"] = _abstract_value(None, step.get("vars") or {}, slots)
    for action in p.get("actions", []):
        action["args"] = _abstract_value(None, action.get("args") or {}, slots)
    p["output"] = {k: (_abstract_value(k, v, slots) if k != "currency" else v) for k, v in p["output"].items()}
    return p


@dataclass
class Structure:
    covers: set[str]
    has_ferry: bool
    task_refs: set[str]


def plan_structure(plan: dict) -> Structure:
    strings = list(_strings(plan))
    covers = {k for s in strings for k in _CONSTRAINT_REF.findall(s)}
    refs = {r.rstrip(".") for s in strings for r in _TASK_REF.findall(s)}
    has_ferry = any(st.get("kind") == "tool" and st.get("tool") == "search_ferries" for st in plan["steps"])
    return Structure(covers, has_ferry, refs)


def compatibility(task: dict, entry: dict, fingerprint: str | None = None) -> tuple[bool, list[str]]:
    reasons = []
    st = plan_structure(entry["template"])
    if fingerprint and entry.get("fingerprint") and entry["fingerprint"] != fingerprint:
        reasons.append("tool_schema_changed")
    task_cons = set(task.get("constraints") or {})
    missing = task_cons - st.covers
    if missing:
        reasons.append(f"uncovered_constraints:{','.join(sorted(missing))}")
    extra = st.covers - task_cons
    if extra:  # plan filters on a constraint the task does not have -> would resolve to null
        reasons.append(f"extra_constraint_refs:{','.join(sorted(extra))}")
    needs_ferry = bool(task["destination"].get("ferry"))
    if needs_ferry != st.has_ferry:
        reasons.append("structure_mismatch:" + ("needs_ferry" if needs_ferry else "unexpected_ferry"))
    for ref in sorted(st.task_refs):
        if ref.startswith("constraints."):
            continue
        if jmespath.search(ref, task) is None:
            reasons.append(f"unresolved_ref:task.{ref}")
    return not reasons, reasons


class PlanCache:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else None
        self.entries: list[dict] = []
        if self.path and self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8")).get("entries", [])

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"entries": self.entries}, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def slot_key(task: dict) -> str:
        cons = ",".join(sorted(task.get("constraints") or {}))
        return f"{task.get('intent', 'trip')}|{task['origin']['iata']}|{task['destination']['city']}|{cons}"

    @staticmethod
    def _score(task: dict, e: dict) -> int:
        s = 0
        s += 4 if e["destination_city"] == task["destination"]["city"] else 0
        s += 2 if e["origin"] == task["origin"]["iata"] else 0
        s += 1 if e["destination_type"] == task["destination"].get("type") else 0
        s += 1 if set(e["constraint_keys"]) == set(task.get("constraints") or {}) else 0
        return s

    def candidates(self, task: dict) -> list[dict]:
        cands = [e for e in self.entries if e["intent"] == task.get("intent", "trip") and (
            e["origin"] == task["origin"]["iata"] or e["destination_city"] == task["destination"]["city"])]
        return sorted(cands, key=lambda e: -self._score(task, e))  # stable: insertion order breaks ties

    def lookup(self, task: dict, fingerprint: str | None = None) -> tuple[str, dict | None, list[str]]:
        """Return ``(status, entry, reasons)`` with status in ``hit | stale | miss``."""
        cands = self.candidates(task)
        if not cands:
            return "miss", None, ["no_candidate"]
        first_reasons: list[str] = []
        for e in cands:
            ok, reasons = compatibility(task, e, fingerprint)
            if ok:
                return "hit", e, []
            if not first_reasons:
                first_reasons = reasons
        return "stale", cands[0], first_reasons

    def store(self, plan: Plan, task: dict, fingerprint: str | None = None) -> dict:
        template = abstract_plan(plan, task)
        key = self.slot_key(task)
        old = next((e for e in self.entries if e["key"] == key), None)
        entry = {
            "key": key, "intent": task.get("intent", "trip"), "origin": task["origin"]["iata"],
            "destination_city": task["destination"]["city"], "destination_type": task["destination"].get("type"),
            "constraint_keys": sorted(task.get("constraints") or {}), "fingerprint": fingerprint,
            "created_from": task["id"], "created_ts": time.time(), "template": template,
            "stats": old["stats"] if old else {"uses": 0, "successes": 0, "failures": 0, "by_perturbation": {}},
        }
        if old:
            self.entries[self.entries.index(old)] = entry
        else:
            self.entries.append(entry)
        self.save()
        return entry

    def record(self, entry: dict, perturbation: str, success: bool) -> None:
        st = entry["stats"]
        st["uses"] += 1
        st["successes" if success else "failures"] += 1
        bp = st["by_perturbation"].setdefault(perturbation, [0, 0])
        bp[0] += int(success)
        bp[1] += 1
        self.save()

    @staticmethod
    def success_rate(entry: dict) -> tuple[float, int]:
        st = entry["stats"]
        n = st["uses"]
        return ((st["successes"] / n) if n else 1.0), n

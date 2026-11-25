"""Deterministic task perturbations (pure functions of ``(task, kind, seed)``).

P0  identity (the original task; warms the plan cache)
P1  shift both trip dates by +/-1 day
P2  shift both trip dates by +/-3 days
P3  swap destination city (may cross hub <-> island, i.e. change plan structure)
P4  scale budget by +/-20%
P5  add or tighten a constraint (may introduce a constraint the cached plan never covered)
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

KINDS = ("P0", "P1", "P2", "P3", "P4", "P5")
DEFAULT_TASKS = Path(__file__).resolve().parent.parent / "data" / "sample_tasks.json"


def _h(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12], 16)


def load_dataset(path: str | Path = DEFAULT_TASKS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _shift(d: str, days: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def _constraint_mutation(task: dict, seed: int) -> tuple[dict, str]:
    cons = task.get("constraints") or {}
    options: list[tuple[dict, str]] = []
    if cons.get("max_stops", 99) > 0:
        options.append(({"max_stops": 0}, "max_stops -> 0"))
    new_rating = round(min(4.5, (cons.get("min_hotel_rating") or 3.5) + 0.5), 1)
    if new_rating > (cons.get("min_hotel_rating") or 0):
        options.append(({"min_hotel_rating": new_rating}, f"min_hotel_rating -> {new_rating}"))
    options.append(({"refundable_only": True}, "refundable_only"))
    cat = ["food", "culture", "outdoor"][_h(task["id"], "cat", seed) % 3]
    options.append(({"include_experience": cat}, f"include_experience={cat}"))
    options.append(({"max_transfer_minutes": 75}, "max_transfer_minutes <= 75"))
    return options[_h(task["id"], "P5", seed) % len(options)]


def perturb(task: dict, kind: str, seed: int = 0, cities: dict | None = None) -> dict:
    """Return a new perturbed task. The input task is never mutated."""
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation {kind!r}; expected one of {KINDS}")
    t = copy.deepcopy(task)
    t["base_id"] = task.get("base_id", task["id"])
    t["id"] = f"{t['base_id']}-{kind}"
    sign = 1 if _h(task["id"], kind, seed) % 2 == 0 else -1
    detail = "identity"
    if kind in ("P1", "P2"):
        days = sign * (1 if kind == "P1" else 3)
        t["depart_date"], t["return_date"] = _shift(t["depart_date"], days), _shift(t["return_date"], days)
        detail = f"dates {days:+d}d"
    elif kind == "P3":
        catalog = cities or load_dataset()["cities"]
        names = sorted(n for n in catalog if n != t["destination"]["city"])
        new = names[_h(task["id"], "P3", seed) % len(names)]
        old = t["destination"]["city"]
        t["destination"] = copy.deepcopy(catalog[new])
        detail = f"destination {old} -> {new} ({catalog[new]['type']})"
    elif kind == "P4":
        factor = 1 + sign * 0.2
        t["budget"] = round(task["budget"] * factor, 2)
        detail = f"budget x{factor:.1f} ({task['budget']} -> {t['budget']})"
    elif kind == "P5":
        mutation, detail = _constraint_mutation(task, seed)
        t["constraints"] = {**(t.get("constraints") or {}), **mutation}
    t["perturbation"] = {"kind": kind, "detail": detail, "seed": seed}
    return t

"""Model-free CPU plan executor.

A plan is a JSON DAG of steps:
  * ``tool``    - call an MCP tool (``server``/``tool``) with resolved ``args``
  * ``select``  - JMESPath ``query`` over the execution context
  * ``compute`` - safe arithmetic ``expr`` over resolved ``vars``

Value resolution inside args/vars/output:
  * ``{"$jp": "<jmespath>"}``                 -> JMESPath over ``{"task", "steps", "output"}``
  * ``{"$expr": "<expr>", "vars": {...}}``    -> ``safe_eval``
  * ``"text {{ jmespath }} text"``            -> string interpolation
  * ``${jmespath}`` inside any JMESPath query -> rendered as an escaped JSON literal

INVARIANT: this module never imports an LLM client. Execution and validation are deterministic.
"""
from __future__ import annotations

import ast
import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Awaitable, Callable, Literal

import jmespath
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------------------------
# Plan schema
# --------------------------------------------------------------------------------------------
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STEP_REF_RE = re.compile(r"steps\.([A-Za-z_][A-Za-z0-9_]*)")
_TEMPLATE_RE = re.compile(r"\$\{([^{}]*)\}")
_INTERP_RE = re.compile(r"\{\{\s*(.+?)\s*\}\}")


class PlanError(ValueError):
    """Structural problem with a plan (bad ids, cycles, unknown deps)."""


class Step(BaseModel):
    id: str
    kind: Literal["tool", "select", "compute"]
    server: str | None = None
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    query: str | None = None
    expr: str | None = None
    vars: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    on_error: Literal["fail", "null"] = "fail"

    @model_validator(mode="after")
    def _check(self) -> "Step":
        if not _ID_RE.match(self.id):
            raise ValueError(f"invalid step id {self.id!r}")
        if self.kind == "tool" and not (self.server and self.tool):
            raise ValueError(f"tool step {self.id} needs server and tool")
        if self.kind == "select" and not self.query:
            raise ValueError(f"select step {self.id} needs query")
        if self.kind == "compute" and not self.expr:
            raise ValueError(f"compute step {self.id} needs expr")
        return self


class Action(BaseModel):
    id: str
    server: str = "actions"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    version: int = 1
    steps: list[Step]
    output: dict[str, Any]
    actions: list[Action] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique(self) -> "Plan":
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step ids")
        return self


def _strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def step_dependencies(step: Step, known: set[str] | None = None) -> set[str]:
    """Declared deps plus every ``steps.<id>`` reference found in the step body.

    Inferred references to steps that do not exist (``known`` given) are optional: they resolve
    to null at runtime (e.g. ``coalesce(ferry_price, 0)`` in a hub plan). Declared deps must exist.
    """
    refs: set[str] = set()
    for s in _strings([step.args, step.query or "", step.expr or "", step.vars]):
        refs.update(_STEP_REF_RE.findall(s))
    if known is not None:
        refs &= known
    refs |= set(step.depends_on)
    refs.discard(step.id)
    return refs


def topo_levels(plan: Plan) -> list[list[Step]]:
    """Kahn's algorithm grouped into levels (steps in one level are independent)."""
    by_id = {s.id: s for s in plan.steps}
    deps = {s.id: step_dependencies(s, set(by_id)) for s in plan.steps}
    for sid, d in deps.items():
        unknown = d - by_id.keys()
        if unknown:
            raise PlanError(f"step {sid} depends on unknown step(s) {sorted(unknown)}")
    levels: list[list[Step]] = []
    done: set[str] = set()
    remaining = dict(deps)
    while remaining:
        ready = sorted(sid for sid, d in remaining.items() if d <= done)
        if not ready:
            raise PlanError(f"cycle detected among {sorted(remaining)}")
        levels.append([by_id[sid] for sid in ready])
        done.update(ready)
        for sid in ready:
            remaining.pop(sid)
    return levels


# --------------------------------------------------------------------------------------------
# Safe AST math
# --------------------------------------------------------------------------------------------
class SafeEvalError(ValueError):
    pass


def _days_between(a: str, b: str) -> int:
    return (date.fromisoformat(str(a)[:10]) - date.fromisoformat(str(b)[:10])).days


def _safe_div(a, b, default=None):
    try:
        return a / b
    except (ZeroDivisionError, TypeError):
        return default


SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sum": sum,
    "len": len,
    "ceil": math.ceil,
    "floor": math.floor,
    "days_between": _days_between,
    "safe_div": _safe_div,
}
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_CMPOPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}
_MAX_EXPR_LEN = 2000
_MAX_NODES = 400


def safe_eval(expr: str, variables: dict[str, Any] | None = None) -> Any:
    """Evaluate a whitelisted arithmetic expression.

    ``coalesce(a, b, ...)`` is lazy: it returns the first argument that evaluates without error
    to a non-None value, so ``coalesce(ppn, total / nights, 0)`` is a safe fallback chain.
    """
    if not isinstance(expr, str) or len(expr) > _MAX_EXPR_LEN:
        raise SafeEvalError("expression missing or too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"syntax error: {e.msg}") from None
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise SafeEvalError("expression too complex")
    env = dict(variables or {})

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool, str)) or node.value is None:
                return node.value
            raise SafeEvalError("constant type not allowed")
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[node.id]
            raise SafeEvalError(f"unknown name {node.id!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            a, b = ev(node.left), ev(node.right)
            if a is None or b is None:
                raise SafeEvalError("null operand")
            if isinstance(node.op, ast.Pow) and (not isinstance(b, (int, float)) or abs(b) > 10):
                raise SafeEvalError("exponent too large")
            try:
                return _BINOPS[type(node.op)](a, b)
            except ZeroDivisionError:
                raise SafeEvalError("division by zero") from None
            except TypeError as e:
                raise SafeEvalError(f"type error: {e}") from None
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.Not):
                return not v
            if v is None:
                raise SafeEvalError("null operand")
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = True
                for v in node.values:
                    result = ev(v)
                    if not result:
                        return result
                return result
            result = False
            for v in node.values:
                result = ev(v)
                if result:
                    return result
            return result
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                if type(op) not in _CMPOPS:
                    raise SafeEvalError("comparison not allowed")
                right = ev(comp)
                try:
                    if not _CMPOPS[type(op)](left, right):
                        return False
                except TypeError:
                    raise SafeEvalError("incomparable operands") from None
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [ev(e) for e in node.elts]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise SafeEvalError("only plain calls to whitelisted functions")
            if any(isinstance(a, ast.Starred) for a in node.args):
                raise SafeEvalError("starred args not allowed")
            name = node.func.id
            if name == "coalesce":
                for a in node.args:
                    try:
                        v = ev(a)
                    except SafeEvalError:
                        continue
                    if v is not None:
                        return v
                return None
            if name not in SAFE_FUNCS:
                raise SafeEvalError(f"function {name!r} not allowed")
            args = [ev(a) for a in node.args]
            try:
                return SAFE_FUNCS[name](*args)
            except (TypeError, ValueError, ZeroDivisionError) as e:
                raise SafeEvalError(f"{name}() failed: {e}") from None
        raise SafeEvalError(f"node {type(node).__name__} not allowed")

    return ev(tree)


# --------------------------------------------------------------------------------------------
# Value resolution
# --------------------------------------------------------------------------------------------
def _jp_literal(value: Any) -> str:
    return "`" + json.dumps(value, sort_keys=True).replace("`", "\\`") + "`"


def render_query(query: str, ctx: dict) -> str:
    """Replace ``${expr}`` with the escaped JSON literal of ``expr`` evaluated on ``ctx``."""
    return _TEMPLATE_RE.sub(lambda m: _jp_literal(jmespath.search(m.group(1).strip(), ctx)), query)


def jp(query: str, ctx: dict) -> Any:
    return jmespath.search(render_query(query, ctx), ctx)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def resolve(value: Any, ctx: dict) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$jp"}:
            return jp(value["$jp"], ctx)
        if "$expr" in value and set(value) <= {"$expr", "vars"}:
            env = {k: resolve(v, ctx) for k, v in (value.get("vars") or {}).items()}
            return safe_eval(value["$expr"], env)
        return {k: resolve(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, ctx) for v in value]
    if isinstance(value, str) and "{{" in value:
        return _INTERP_RE.sub(lambda m: _fmt(jp(m.group(1), ctx)), value)
    return value


# --------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------
ToolCaller = Callable[[str, str, dict], Awaitable[tuple[Any, dict]]]
EventHook = Callable[[dict], None] | None


@dataclass
class ExecutionResult:
    ok: bool
    output: dict | None
    step_results: dict[str, Any]
    trace: list[dict]
    error: str | None = None
    latency_ms: float = 0.0
    tool_calls: int = 0
    edges: list[tuple[str, str]] = field(default_factory=list)


def _preview(v: Any, n: int = 240) -> str:
    s = json.dumps(v, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


async def execute_plan(
    plan: Plan, task: dict, call_tool: ToolCaller, on_event: EventHook = None
) -> ExecutionResult:
    t0 = time.perf_counter()
    emit = on_event or (lambda e: None)
    try:
        levels = topo_levels(plan)
    except PlanError as e:
        return ExecutionResult(False, None, {}, [], error=f"plan_error: {e}")
    known = {s.id for s in plan.steps}
    edges = [(d, s.id) for s in plan.steps for d in sorted(step_dependencies(s, known))]
    ctx: dict[str, Any] = {"task": task, "steps": {}}
    trace: list[dict] = []
    failed: str | None = None
    tool_calls = 0

    async def run_step(step: Step) -> dict:
        rec = {
            "step_id": step.id, "kind": step.kind, "server": step.server, "tool": step.tool,
            "depends_on": sorted(step_dependencies(step, known)), "source": "cpu",
        }
        emit({"type": "step_start", **rec})
        s0 = time.perf_counter()
        try:
            if step.kind == "tool":
                args = resolve(step.args, ctx)
                rec["args"] = args
                result, meta = await call_tool(step.server, step.tool, args)
                rec["source"] = meta.get("source", "unknown")
            elif step.kind == "select":
                result = jp(step.query, ctx)
            else:
                env = {k: resolve(v, ctx) for k, v in step.vars.items()}
                result = safe_eval(step.expr, env)
            rec.update(status="ok", result_preview=_preview(result))
        except Exception as e:  # noqa: BLE001 - every step failure is recorded, not raised
            result = None
            if step.on_error == "null":
                rec.update(status="ok_null", error=f"{type(e).__name__}: {e}")
            else:
                rec.update(status="error", error=f"{type(e).__name__}: {e}")
        rec["latency_ms"] = round((time.perf_counter() - s0) * 1000, 3)
        rec["_result"] = result
        emit({"type": "step_end", **{k: v for k, v in rec.items() if k != "_result"}})
        return rec

    for level in levels:
        if failed:
            for step in level:
                rec = {"step_id": step.id, "kind": step.kind, "server": step.server, "tool": step.tool,
                       "depends_on": sorted(step_dependencies(step, known)), "status": "skipped",
                       "latency_ms": 0.0, "source": "none"}
                trace.append(rec)
                emit({"type": "step_skipped", **rec})
            continue
        tools = [s for s in level if s.kind == "tool"]
        local = [s for s in level if s.kind != "tool"]
        recs = list(await asyncio.gather(*(run_step(s) for s in tools)))
        tool_calls += len(tools)
        for s in local:
            recs.append(await run_step(s))
        for rec in recs:
            ctx["steps"][rec["step_id"]] = rec.pop("_result")
            trace.append(rec)
            if rec["status"] == "error" and not failed:
                failed = f"step {rec['step_id']} failed: {rec.get('error')}"

    output = None
    if not failed:
        try:
            output = resolve(plan.output, ctx)
        except Exception as e:  # noqa: BLE001
            failed = f"output resolution failed: {type(e).__name__}: {e}"
    return ExecutionResult(
        ok=failed is None, output=output, step_results=dict(ctx["steps"]), trace=trace,
        error=failed, latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        tool_calls=tool_calls, edges=edges,
    )


# --------------------------------------------------------------------------------------------
# Deterministic validator (independent of the plan's own arithmetic)
# --------------------------------------------------------------------------------------------
@dataclass
class ValidationReport:
    ok: bool
    violations: list[str]
    recomputed_total: float | None = None


def rooms_for(travelers: int) -> int:
    return max(1, math.ceil(int(travelers) / 2))


def nights_for(task: dict) -> int:
    return _days_between(task["return_date"], task["depart_date"])


def recompute_total(task: dict, out: dict) -> float:
    """Canonical cost model (GBP):
    (flight_out + flight_back + ferry_out + ferry_back + experience) * travelers
    + hotel_nightly * nights * rooms,   hotel_nightly = price_per_night or total_price / nights.
    """
    n, nights, rooms = int(task["travelers"]), nights_for(task), rooms_for(task["travelers"])
    per_person = 0.0
    for key in ("flight_out", "flight_back", "ferry_out", "ferry_back", "experience"):
        item = out.get(key)
        if item:
            per_person += float(item["price"])
    hotel = out["hotel"]
    nightly = hotel.get("price_per_night")
    if nightly is None:
        nightly = float(hotel["total_price"]) / nights
    return round(per_person * n + float(nightly) * nights * rooms, 2)


def validate_itinerary(task: dict, out: dict | None, tolerance: float = 1.0) -> ValidationReport:
    v: list[str] = []
    if not isinstance(out, dict):
        return ValidationReport(False, ["no_output: plan produced no output"])
    for key in ("flight_out", "flight_back", "hotel"):
        if not isinstance(out.get(key), dict):
            v.append(f"missing_{key}: no {key} selected")
    if not isinstance(out.get("total_cost"), (int, float)) or isinstance(out.get("total_cost"), bool):
        v.append("missing_total_cost: total_cost is not numeric")
    dest, origin, cons = task["destination"], task["origin"], task.get("constraints") or {}
    gateway = dest["gateway_iata"]
    fo, fb, hotel = out.get("flight_out"), out.get("flight_back"), out.get("hotel")

    if isinstance(fo, dict):
        if (fo.get("from"), fo.get("to")) != (origin["iata"], gateway):
            v.append(f"flight_out_route: {fo.get('from')}->{fo.get('to')} != {origin['iata']}->{gateway}")
        if fo.get("date") != task["depart_date"]:
            v.append(f"flight_out_date: {fo.get('date')} != {task['depart_date']}")
    if isinstance(fb, dict):
        if (fb.get("from"), fb.get("to")) != (gateway, origin["iata"]):
            v.append(f"flight_back_route: {fb.get('from')}->{fb.get('to')} != {gateway}->{origin['iata']}")
        if fb.get("date") != task["return_date"]:
            v.append(f"flight_back_date: {fb.get('date')} != {task['return_date']}")
    if isinstance(hotel, dict):
        if hotel.get("city") != dest["city"]:
            v.append(f"hotel_city: {hotel.get('city')} != {dest['city']}")
        if (hotel.get("checkin"), hotel.get("checkout")) != (task["depart_date"], task["return_date"]):
            v.append("hotel_dates: stay does not match trip dates")
        if hotel.get("price_per_night") is None and hotel.get("total_price") is None:
            v.append("hotel_price: no price information")

    ferry = dest.get("ferry")
    if ferry:
        fy_o, fy_b = out.get("ferry_out"), out.get("ferry_back")
        if not isinstance(fy_o, dict) or not isinstance(fy_b, dict):
            v.append("missing_ferry: island destination requires ferry legs")
        else:
            if (fy_o.get("from_port"), fy_o.get("to_port"), fy_o.get("date")) != (
                ferry["from_port"], ferry["to_port"], task["depart_date"]):
                v.append("ferry_out_mismatch: wrong ferry route/date")
            if (fy_b.get("from_port"), fy_b.get("to_port"), fy_b.get("date")) != (
                ferry["to_port"], ferry["from_port"], task["return_date"]):
                v.append("ferry_back_mismatch: wrong ferry route/date")

    # constraints
    if "max_stops" in cons:
        for key, f in (("flight_out", fo), ("flight_back", fb)):
            if isinstance(f, dict) and f.get("stops", 99) > cons["max_stops"]:
                v.append(f"max_stops: {key} has {f.get('stops')} stops > {cons['max_stops']}")
    if "min_hotel_rating" in cons and isinstance(hotel, dict):
        if (hotel.get("rating") or 0) < cons["min_hotel_rating"]:
            v.append(f"min_hotel_rating: {hotel.get('rating')} < {cons['min_hotel_rating']}")
    if cons.get("refundable_only") and isinstance(hotel, dict) and not hotel.get("refundable"):
        v.append("refundable_only: hotel is non-refundable")
    if "include_experience" in cons:
        exp = out.get("experience")
        if not isinstance(exp, dict) or exp.get("category") != cons["include_experience"]:
            v.append(f"include_experience: no '{cons['include_experience']}' experience selected")
    if "max_transfer_minutes" in cons:
        tm = out.get("transfer_minutes")
        if not isinstance(tm, (int, float)) or tm > cons["max_transfer_minutes"]:
            v.append(f"max_transfer_minutes: {tm} > {cons['max_transfer_minutes']}")

    recomputed = None
    if not any(x.startswith(("missing_flight", "missing_hotel", "hotel_price")) for x in v):
        try:
            recomputed = recompute_total(task, out)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            v.append(f"cost_model: cannot recompute total ({e})")
    if recomputed is not None:
        if isinstance(out.get("total_cost"), (int, float)) and abs(out["total_cost"] - recomputed) > tolerance:
            v.append(f"total_mismatch: plan says {out['total_cost']}, recomputed {recomputed}")
        if recomputed > task["budget"] + 1e-9:
            v.append(f"over_budget: {recomputed} > {task['budget']}")
    return ValidationReport(not v, v, recomputed)


# --------------------------------------------------------------------------------------------
# Commit phase (side effects) - only after a passing validation
# --------------------------------------------------------------------------------------------
class UnvalidatedCommitError(RuntimeError):
    pass


async def commit_actions(
    plan: Plan, task: dict, result: ExecutionResult, report: ValidationReport,
    call_tool: ToolCaller, on_event: EventHook = None, context_extra: dict | None = None,
) -> list[dict]:
    if not (report.ok and result.ok):
        raise UnvalidatedCommitError("actions may only be committed after a passing validation")
    emit = on_event or (lambda e: None)
    ctx = {"task": task, "steps": result.step_results, "output": result.output, **(context_extra or {})}
    receipts = []
    for action in plan.actions:
        args = resolve(action.args, ctx)
        s0 = time.perf_counter()
        emit({"type": "action_start", "step_id": action.id, "server": action.server, "tool": action.tool})
        res, meta = await call_tool(action.server, action.tool, args)
        rec = {"step_id": action.id, "server": action.server, "tool": action.tool, "result": res,
               "source": meta.get("source"), "latency_ms": round((time.perf_counter() - s0) * 1000, 3)}
        emit({"type": "action_end", **rec})
        receipts.append(rec)
    return receipts

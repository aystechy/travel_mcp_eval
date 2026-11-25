"""Offline verification suite (provider=mock, source=sim). Run: python -m pytest -q tests/"""
from __future__ import annotations

import asyncio
import copy
import json
import re
from datetime import date
from pathlib import Path

import pytest

from engine.batch_runner import RunConfig, run_benchmark
from engine.mcp_proxy import MCPProxy, ReplayMiss, replay_key
from engine.perturbations import KINDS, load_dataset, perturb
from engine.plan_cache import PlanCache, abstract_plan, compatibility, plan_structure
from engine.plan_executor import (Plan, PlanError, SafeEvalError, UnvalidatedCommitError, ValidationReport,
                                  commit_actions, execute_plan, recompute_total, render_query, safe_eval,
                                  topo_levels, validate_itinerary)
from engine.planner import heuristic_plan, parse_plan

ROOT = Path(__file__).resolve().parent.parent
DATA = load_dataset(ROOT / "data" / "sample_tasks.json")
TASKS = {t["id"]: t for t in DATA["tasks"]}


def run(coro):
    return asyncio.run(coro)


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


# ------------------------------------------------------------------------------ safe math
def test_safe_eval_arithmetic_and_functions():
    assert safe_eval("round((a + b) * n + c * 2, 2)", {"a": 100.5, "b": 99.5, "n": 2, "c": 10}) == 420.0
    assert safe_eval("days_between('2027-05-18', '2027-05-14')") == 4
    assert safe_eval("ceil(n / 2)", {"n": 3}) == 2
    assert safe_eval("x if x > 3 else 3", {"x": 1}) == 3
    assert safe_eval("min([4, 2, 9])") == 2


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "a.__class__", "a[0]", "(lambda: 1)()",
    "[x for x in [1]]", "open('f')", "2 ** 1000", "eval('1')", "max(*[1, 2])",
])
def test_safe_eval_rejects_unsafe(expr):
    with pytest.raises(SafeEvalError):
        safe_eval(expr, {"a": [1]})


def test_fallback_math_coalesce_chain():
    # missing nightly price -> fall back to total / nights
    assert safe_eval("coalesce(ppn, safe_div(total, nights))", {"ppn": None, "total": 600.0, "nights": 4}) == 150.0
    # division by zero inside coalesce is skipped, not raised
    assert safe_eval("coalesce(None, total / nights, 42)", {"total": 600.0, "nights": 0}) == 42
    assert safe_eval("safe_div(1, 0, -1)") == -1
    # optional legs absent -> 0
    assert safe_eval("(fo + coalesce(fy, 0)) * n", {"fo": 50, "fy": None, "n": 2}) == 100
    with pytest.raises(SafeEvalError):
        safe_eval("a + b", {"a": None, "b": 1})
    with pytest.raises(SafeEvalError):
        safe_eval("1 / 0")


def test_validator_cost_model_uses_total_price_fallback():
    task = copy.deepcopy(TASKS["T03"])  # 1 traveller, 4 nights
    out = {"flight_out": {"price": 100.0}, "flight_back": {"price": 80.0},
           "hotel": {"price_per_night": None, "total_price": 400.0}}
    assert recompute_total(task, out) == 180.0 + 400.0
    task["travelers"] = 3  # 2 rooms
    out["hotel"]["price_per_night"] = 50.0
    assert recompute_total(task, out) == 180.0 * 3 + 50.0 * 4 * 2


# ------------------------------------------------------------------------------ executor
def test_jmespath_template_renders_escaped_literal():
    ctx = {"task": {"x": "a`b", "n": 2}, "steps": {}}
    q = render_query("items[?name == ${task.x} && k <= ${task.n}]", ctx)
    assert q == r'items[?name == `"a\`b"` && k <= `2`]'


def test_topo_levels_and_cycle_detection():
    plan = Plan.model_validate({"steps": [
        {"id": "a", "kind": "compute", "expr": "1"},
        {"id": "b", "kind": "compute", "expr": "x + 1", "vars": {"x": {"$jp": "steps.a"}}},
        {"id": "c", "kind": "compute", "expr": "1"}], "output": {}})
    assert [[s.id for s in lvl] for lvl in topo_levels(plan)] == [["a", "c"], ["b"]]
    cyc = Plan.model_validate({"steps": [
        {"id": "a", "kind": "select", "query": "steps.b"},
        {"id": "b", "kind": "select", "query": "steps.a"}], "output": {}})
    with pytest.raises(PlanError):
        topo_levels(cyc)


def test_executor_runs_heuristic_plan_and_validates(tmp_path):
    task = TASKS["T02"]  # island: ferry legs required

    async def go():
        async with MCPProxy(source="sim", replay_dir=tmp_path / "r", actions_log=tmp_path / "a.jsonl") as p:
            plan = parse_plan(json.dumps(heuristic_plan(task)))
            res = await execute_plan(plan, task, p.call)
            return res, validate_itinerary(task, res.output)

    res, rep = run(go())
    assert res.ok, res.error
    assert rep.ok, rep.violations
    assert res.output["ferry_out"]["to_port"] == "Naxos"
    assert abs(res.output["total_cost"] - rep.recomputed_total) <= 1.0
    assert all(t["source"] in ("sim", "cpu") for t in res.trace)
    assert not (tmp_path / "a.jsonl").exists()  # execution alone never commits actions


def test_validator_flags_violations():
    task = copy.deepcopy(TASKS["T01"])
    out = {"flight_out": {"from": "MAN", "to": "BCN", "date": task["depart_date"], "stops": 2, "price": 300},
           "flight_back": {"from": "BCN", "to": "MAN", "date": task["return_date"], "stops": 0, "price": 300},
           "hotel": {"city": "Barcelona", "checkin": task["depart_date"], "checkout": task["return_date"],
                     "price_per_night": 200, "total_price": 800, "rating": 4.0},
           "total_cost": 999.0}
    rep = validate_itinerary(task, out)
    codes = {v.split(":")[0] for v in rep.violations}
    assert {"max_stops", "total_mismatch", "over_budget"} <= codes
    assert rep.recomputed_total == 600 * 2 + 200 * 4
    assert not validate_itinerary(task, None).ok


# ------------------------------------------------------------------------------ perturbations
def test_perturbations_deterministic_and_pure():
    task = TASKS["T01"]
    snapshot = copy.deepcopy(task)
    for kind in KINDS:
        a, b = perturb(task, kind, seed=3), perturb(task, kind, seed=3)
        assert a == b, kind
        assert a["id"] == f"T01-{kind}" and a["perturbation"]["kind"] == kind
    assert task == snapshot  # never mutated


@pytest.mark.parametrize("tid", list(TASKS))
def test_perturbation_semantics(tid):
    t = TASKS[tid]
    d0, r0 = date.fromisoformat(t["depart_date"]), date.fromisoformat(t["return_date"])
    for kind, mag in (("P1", 1), ("P2", 3)):
        p = perturb(t, kind, seed=7)
        dd = (date.fromisoformat(p["depart_date"]) - d0).days
        assert abs(dd) == mag and (date.fromisoformat(p["return_date"]) - r0).days == dd
    p3 = perturb(t, "P3", seed=7, cities=DATA["cities"])
    assert p3["destination"]["city"] != t["destination"]["city"]
    assert p3["destination"] == DATA["cities"][p3["destination"]["city"]]
    p4 = perturb(t, "P4", seed=7)
    assert p4["budget"] in (round(t["budget"] * 0.8, 2), round(t["budget"] * 1.2, 2))
    p5 = perturb(t, "P5", seed=7)
    assert p5["constraints"] != t["constraints"]
    assert set(t["constraints"]) <= set(p5["constraints"])


def test_p3_crosses_structure_somewhere():
    types = {(t["destination"]["type"], perturb(t, "P3", 7, DATA["cities"])["destination"]["type"])
             for t in TASKS.values()}
    assert any(a != b for a, b in types), "P3 should sometimes swap hub <-> island"


# ------------------------------------------------------------------------------ plan cache
def test_abstraction_lifts_literals_but_keeps_derived_ones():
    task = TASKS["T01"]
    tpl = abstract_plan(Plan.model_validate(heuristic_plan(task)), task)
    steps = {s["id"]: s for s in tpl["steps"]}
    assert steps["flights_out"]["args"]["date"] == {"$jp": "task.depart_date"}
    assert steps["flights_back"]["args"]["date"] == {"$jp": "task.return_date"}
    assert steps["flights_back"]["args"]["origin"] == {"$jp": "task.destination.gateway_iata"}
    assert steps["hotels"]["args"]["city"] == {"$jp": "task.destination.city"}
    assert steps["flights_out"]["args"]["max_stops"] == {"$jp": "task.constraints.max_stops"}
    assert "`" in steps["hotel_pick"]["query"]  # derived budget cap stays a literal (stale on P4)
    actions = {a["id"]: a for a in tpl["actions"]}
    assert actions["calendar"]["args"]["title"] == "Trip to {{ task.destination.city }}"
    st = plan_structure(tpl)
    assert st.covers == {"max_stops"} and not st.has_ferry


def test_cache_hit_miss_stale():
    cache = PlanCache(None)
    base = TASKS["T01"]
    assert cache.lookup(perturb(base, "P0", 7))[0] == "miss"
    cache.store(Plan.model_validate(heuristic_plan(base)), perturb(base, "P0", 7), "fp1")
    for kind in ("P1", "P2", "P4"):
        status, entry, _ = cache.lookup(perturb(base, kind, 7), "fp1")
        assert status == "hit", kind
    # new constraint the plan never referenced -> stale
    t5 = copy.deepcopy(base)
    t5["constraints"] = {**t5["constraints"], "refundable_only": True}
    status, _, reasons = cache.lookup(t5, "fp1")
    assert status == "stale" and any(r.startswith("uncovered_constraints") for r in reasons)
    # hub -> island needs ferry legs -> structure mismatch
    t3 = copy.deepcopy(base)
    t3["destination"] = DATA["cities"]["Naxos"]
    status, _, reasons = cache.lookup(t3, "fp1")
    assert status == "stale" and any(r.startswith("structure_mismatch") for r in reasons)
    # constraint referenced by plan but absent from task -> stale
    t0 = copy.deepcopy(base)
    t0["constraints"] = {}
    assert any(r.startswith("extra_constraint_refs") for r in cache.lookup(t0, "fp1")[2])
    # tool schema change invalidates
    ok, reasons = compatibility(perturb(base, "P1", 7), cache.entries[0], "fp2")
    assert not ok and "tool_schema_changed" in reasons


def test_reused_template_executes_on_perturbed_task(tmp_path):
    base = TASKS["T01"]
    tpl = abstract_plan(Plan.model_validate(heuristic_plan(base)), base)
    p1 = perturb(base, "P1", 7)

    async def go():
        async with MCPProxy(source="sim", replay_dir=tmp_path / "r", actions_log=tmp_path / "a.jsonl") as p:
            return await execute_plan(Plan.model_validate(tpl), p1, p.call)

    res = run(go())
    assert res.ok
    assert res.output["flight_out"]["date"] == p1["depart_date"]
    assert res.output["hotel"]["checkout"] == p1["return_date"]
    assert validate_itinerary(p1, res.output).ok


# ------------------------------------------------------------------------------ proxy / replay
def test_replay_cache_roundtrip(tmp_path):
    args = {"origin": "MAN", "destination": "BCN", "date": "2027-05-14"}

    async def go():
        async with MCPProxy(source="sim", replay_dir=tmp_path) as p:
            r1, m1 = await p.call("kiwi", "search_flights", args)
        async with MCPProxy(source="replay", replay_dir=tmp_path) as p:
            r2, m2 = await p.call("kiwi", "search_flights", dict(reversed(list(args.items()))))
            with pytest.raises(ReplayMiss):
                await p.call("kiwi", "search_flights", {**args, "date": "2030-01-01"})
        return r1, m1, r2, m2

    r1, m1, r2, m2 = run(go())
    assert m1["source"] == "sim" and m2["source"] == "replay_sim" and r1 == r2  # never shown as live data
    assert m1["cache_key"] == replay_key("kiwi", "search_flights", args)
    assert re.fullmatch(r"[0-9a-f]{64}", m1["cache_key"])


# ------------------------------------------------------------------------------ actions
def test_actions_only_commit_after_validation(tmp_path):
    log = tmp_path / "actions.jsonl"
    task = TASKS["T01"]

    async def go():
        async with MCPProxy(source="sim", replay_dir=tmp_path / "r", actions_log=log) as p:
            plan = Plan.model_validate(heuristic_plan(task))
            res = await execute_plan(plan, task, p.call)
            with pytest.raises(UnvalidatedCommitError):
                await commit_actions(plan, task, res, ValidationReport(False, ["x"]), p.call)
            assert not log.exists() or not read_jsonl(log)
            rep = validate_itinerary(task, res.output)
            assert rep.ok, rep.violations
            p.action_context.update(run_id="r1", episode_id="e1")
            return await commit_actions(plan, task, res, rep, p.call)

    receipts = run(go())
    recs = read_jsonl(log)
    assert [r["kind"] for r in recs] == ["booking", "calendar_event", "email"]
    assert all(r["run_id"] == "r1" and r["episode_id"] == "e1" for r in recs)
    assert recs[0]["payload"]["city"] == "Barcelona"
    assert recs[2]["payload"]["to"] == task["traveler"]["email"]
    assert "{{" not in recs[2]["payload"]["body"]
    assert receipts[0]["result"]["confirmation"].startswith("BK-")


# ------------------------------------------------------------------------------ end-to-end
def _cfg(tmp_path, **kw) -> RunConfig:
    base = dict(task_ids=("T01", "T02"), logs_dir=str(tmp_path / "logs"), cache_dir=str(tmp_path / "cache"),
                run_id="test-run")
    base.update(kw)
    return RunConfig(**base)


def test_end_to_end_policies(tmp_path):
    records = run(run_benchmark(_cfg(tmp_path)))
    assert len(records) == 3 * 2 * len(KINDS)
    by = {(r["policy"], r["task_id"]): r for r in records}

    for r in records:
        if r["policy"] == "replan":
            assert r["cache_status"] == "bypass" and r["llm_calls"] >= 1 and r["total_tokens"] > 0

    # cache warm-up then token-free reuse on date shifts
    for pol in ("cache", "budget_aware"):
        assert by[(pol, "T01-P0")]["cache_status"] == "miss" and by[(pol, "T01-P0")]["success"]
        p1 = by[(pol, "T01-P1")]
        assert p1["cache_status"] == "hit" and p1["reuse_ok"] and p1["total_tokens"] == 0 and p1["success"]

    # budget-aware spends fewer tokens than full re-planning without losing success
    agg = {pol: [r for r in records if r["policy"] == pol] for pol in ("replan", "budget_aware")}
    assert sum(r["total_tokens"] for r in agg["budget_aware"]) < sum(r["total_tokens"] for r in agg["replan"])
    assert sum(r["success"] for r in agg["budget_aware"]) >= sum(r["success"] for r in agg["replan"])

    logs = Path(tmp_path / "logs")
    runs = read_jsonl(logs / "runs.jsonl")
    events = read_jsonl(logs / "events.jsonl")
    actions = read_jsonl(logs / "simulated_actions.jsonl")
    assert len(runs) == len(records)
    step_events = [e for e in events if e.get("type") == "step_end" and e.get("kind") == "tool"]
    assert step_events and all(e["source"] == "sim" for e in step_events)
    assert {"episode_start", "cache_lookup", "validation", "episode_end"} <= {e["type"] for e in events}

    # actions: exactly 3 per successful episode, none for failed ones
    ok_eps = {r["episode_id"] for r in runs if r["success"]}
    assert len(actions) == 3 * len(ok_eps)
    assert {a["episode_id"] for a in actions} == ok_eps
    for r in runs:
        assert len(r["actions"]) == (3 if r["success"] else 0)


def test_budget_aware_degrades_when_budget_exhausted(tmp_path):
    records = run(run_benchmark(_cfg(tmp_path, policies=("budget_aware",), task_ids=("T01",),
                                     perturbations=("P0",), episode_token_budget=100)))
    r = records[0]
    assert r["decision_path"] == "degrade_budget"
    assert not r["success"] and r["total_tokens"] == 0 and r["llm_calls"] == 0
    assert not read_jsonl(Path(tmp_path / "logs" / "simulated_actions.jsonl"))


def test_executor_is_model_free():
    src = (ROOT / "engine" / "plan_executor.py").read_text(encoding="utf-8")
    imports = [l for l in src.splitlines() if l.startswith(("import ", "from "))]
    assert not any(m in l for l in imports for m in ("llm_clients", "planner", "openai", "anthropic", "httpx"))


def test_graceful_stop_file_ends_run_after_current_episode(tmp_path):
    from engine.batch_runner import STOP_FILE
    from engine.llm_clients import MockClient

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / STOP_FILE).touch()  # stale STOP from an earlier run must be ignored

    class StopAfterFirstCall(MockClient):
        async def _raw(self, system, user):
            (logs / STOP_FILE).touch()
            return await super()._raw(system, user)

    records = run(run_benchmark(_cfg(tmp_path, policies=("replan",), task_ids=("T01",)),
                                client=StopAfterFirstCall()))
    assert len(records) == 1
    end = [e for e in read_jsonl(logs / "events.jsonl") if e.get("type") == "run_end"][-1]
    assert end["stopped"] is True and end["episodes"] == 1
    assert not (logs / STOP_FILE).exists()


def test_pid_file_identity_and_stale_cleanup(tmp_path):
    import os
    from engine.runctl import PID_FILE, active_run, clear_pid_file, read_pid_file, write_pid_file

    write_pid_file(tmp_path, "r1")
    info = read_pid_file(tmp_path)
    assert info["pid"] == os.getpid() and info["run_id"] == "r1"
    assert active_run(tmp_path)["run_id"] == "r1"  # this live process is the "runner"

    # reused PID: same pid, different creation time -> not our runner, stale file removed
    (tmp_path / PID_FILE).write_text(json.dumps({**info, "created": info["created"] - 3600}), encoding="utf-8")
    assert active_run(tmp_path) is None and not (tmp_path / PID_FILE).exists()

    # dead PID -> stale file removed
    (tmp_path / PID_FILE).write_text(json.dumps({"pid": 2**22 + 12345, "created": 1.0, "run_id": "x"}),
                                     encoding="utf-8")
    assert active_run(tmp_path) is None and not (tmp_path / PID_FILE).exists()

    # clear only removes a file owned by the given pid
    write_pid_file(tmp_path, "r2")
    clear_pid_file(tmp_path, pid=os.getpid() + 1)
    assert (tmp_path / PID_FILE).exists()
    clear_pid_file(tmp_path, pid=os.getpid())
    assert not (tmp_path / PID_FILE).exists()


def test_delete_run_removes_only_that_run(tmp_path):
    from engine.runctl import PID_FILE, RunActiveError, delete_run, write_pid_file

    for rid in ("run-a", "run-b"):
        run(run_benchmark(_cfg(tmp_path, run_id=rid, policies=("cache",), task_ids=("T01",),
                               perturbations=("P0", "P1"))))
    logs, cache = tmp_path / "logs", tmp_path / "cache"
    before = {n: read_jsonl(logs / n) for n in ("runs.jsonl", "events.jsonl", "simulated_actions.jsonl")}
    assert all(any(r.get("run_id") == "run-a" for r in rows) for rows in before.values())
    assert list(cache.glob("plan_cache_run-a_*.json"))

    write_pid_file(logs, "live")  # an active runner blocks deletion
    with pytest.raises(RunActiveError):
        delete_run(logs, "run-a", cache)
    (logs / PID_FILE).unlink()

    counts = delete_run(logs, "run-a", cache)
    for name, rows in before.items():
        after = read_jsonl(logs / name)
        assert not any(r.get("run_id") == "run-a" for r in after)
        assert after == [r for r in rows if r.get("run_id") != "run-a"]  # run-b untouched, order kept
        assert counts[name] == len(rows) - len(after)
    assert counts["plan_cache_files"] == 1 and not list(cache.glob("plan_cache_run-a_*.json"))
    assert list(cache.glob("plan_cache_run-b_*.json"))
    assert list((cache / "replay").rglob("*.json"))  # shared replay cache kept


def test_run_label_recorded_and_normalised(tmp_path):
    from engine.batch_runner import parse_args

    cfg = parse_args(["--label", "  qwen 7b,\n tight   budget  " + "x" * 100, "--logs-dir", str(tmp_path)])
    assert cfg.label.startswith("qwen 7b, tight budget x") and len(cfg.label) == 80
    records = run(run_benchmark(_cfg(tmp_path, label="my label", policies=("cache",), task_ids=("T01",),
                                     perturbations=("P0",))))
    assert records[0]["label"] == "my label"
    start = next(e for e in read_jsonl(tmp_path / "logs" / "events.jsonl") if e["type"] == "run_start")
    assert start["config"]["label"] == "my label"


def test_ui_fragments_poll_only_while_a_run_is_active(tmp_path, monkeypatch):
    """Idle app: no fragment timers. Active run: 1 s timers. Run gone: timers off again."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from engine.runctl import PID_FILE, write_pid_file

    seen: list = []
    real_fragment = st.fragment

    def recording_fragment(func=None, *, run_every=None, **kw):
        seen.append(run_every)
        return real_fragment(func, run_every=run_every, **kw)

    monkeypatch.setattr(st, "fragment", recording_fragment)

    def poll_values() -> set:
        seen.clear()
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.session_state["logs_dir"] = str(tmp_path)
        at.run()
        assert not at.exception
        return set(seen)

    assert poll_values() == {None}  # idle: nothing re-runs on a timer
    write_pid_file(tmp_path, "active-run")  # this test process stands in for a live runner
    assert poll_values() == {None, 1} and seen.count(1) == 3  # run controls, Live, Live DAG; not the form
    (tmp_path / PID_FILE).unlink()
    assert poll_values() == {None}


def test_about_tab_renders_and_reproduce_command_round_trips(tmp_path):
    import shlex
    from streamlit.testing.v1 import AppTest
    from engine.batch_runner import parse_args

    run(run_benchmark(_cfg(tmp_path, run_id="about-run", policies=("cache",), task_ids=("T01",),
                           perturbations=("P0", "P1"), label='say "hi"', seed=11)))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["logs_dir"] = str(tmp_path / "logs")
    at.run()
    assert not at.exception
    about = next(t for t in at.tabs if t.label == "About")
    assert any("```mermaid" in m.value for m in about.markdown)
    cmd = about.code[0].value
    argv = shlex.split(cmd)[3:]  # drop "python -m engine.batch_runner"
    cfg = parse_args(argv)
    assert (cfg.provider, cfg.source, cfg.seed, cfg.policies, cfg.perturbations, cfg.task_ids) == \
        ("mock", "sim", 11, ("cache",), ("P0", "P1"), ("T01",))
    assert cfg.label == "say 'hi'" and cfg.episode_token_budget == 12000


def test_tasks_100_dataset_is_feasible_and_reproducible():
    from collections import Counter
    from data.build_tasks_100 import OUT, build, cheapest_itinerary
    from engine.perturbations import KINDS, load_dataset, perturb

    ds = load_dataset(OUT)
    tasks, cities = ds["tasks"], ds["cities"]
    assert len({t["id"] for t in tasks}) == 100
    assert len({(t["origin"]["city"], t["destination"]["city"]) for t in tasks}) == 100  # distinct routes
    assert Counter(t["destination"]["type"] for t in tasks) == {"hub": 75, "island": 25}
    assert not any(c["country"] == "GB" for c in cities.values())  # P3 never swaps to a UK city
    for t in tasks:
        assert t["destination"] == cities[t["destination"]["city"]]
        assert set(t["constraints"]) <= {"max_stops", "min_hotel_rating"}
        assert bool(t["destination"]["ferry"]) == (t["destination"]["type"] == "island")
        best = run(cheapest_itinerary(t))
        assert best and validate_itinerary(t, best).ok, t["id"]
        for kind in KINDS[1:]:  # no perturbation is a no-op
            p = perturb(t, kind, 7, cities)
            assert any(p[k] != t[k] for k in ("depart_date", "destination", "budget", "constraints")), (t["id"], kind)
    assert run(build()) == ds  # the saved file matches its generator

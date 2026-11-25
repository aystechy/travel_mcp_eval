"""Planner prompt, grounding check, repair prompt and failed-call accounting (offline)."""
import asyncio
import json
import re
from pathlib import Path

import pytest

from engine.batch_runner import RunConfig, run_benchmark
from engine.llm_clients import LLMClient, MockClient, plan_with_llm
from engine.mcp_proxy import MCPProxy
from engine.perturbations import load_dataset
from engine.plan_executor import Plan, execute_plan, validate_itinerary
from engine.planner import (EXAMPLE_PLAN, SYSTEM_PROMPT, TOOL_RESULTS, PlanParseError, build_user_prompt,
                            extract_task_from_prompt, heuristic_plan, parse_plan)

ROOT = Path(__file__).resolve().parent.parent
TASK = load_dataset(ROOT / "data" / "sample_tasks.json")["tasks"][0]
CALLS = {
    "search_flights": ("kiwi", {"origin": "MAN", "destination": "BCN", "date": "2027-05-14"}),
    "search_hotels": ("trivago", {"city": "Barcelona", "checkin": "2027-05-14", "checkout": "2027-05-18"}),
    "search_ferries": ("ferryhopper", {"from_port": "Piraeus", "to_port": "Naxos", "date": "2027-05-14"}),
    "search_experiences": ("peek", {"city": "Barcelona", "date": "2027-05-14"}),
    "directions": ("mapbox", {"from_place": "BCN", "to_place": "Barcelona"}),
    "geocode": ("mapbox", {"query": "Barcelona"}),
    "forecast": ("weather", {"lat": 41.4, "lon": 2.2, "date": "2027-05-14"}),
}


def test_documented_results_match_the_simulators():
    async def go():
        async with MCPProxy(source="sim") as proxy:
            return {t: (await proxy.call(s, t, a))[0] for t, (s, a) in CALLS.items()}

    results = asyncio.run(go())
    assert set(results) == set(TOOL_RESULTS)
    for tool, doc in TOOL_RESULTS.items():
        result = results[tool]
        items = {k: v[0] for k, v in result.items() if isinstance(v, list)}
        for lst, fields in re.findall(r"(\w+):\[\{([^}]*)\}\]", doc):
            assert {f.split("|")[0] for f in fields.split(",")} == set(items[lst]), tool
        flat = re.sub(r",?\w+:\[\{[^}]*\}\]", "", doc).strip("{}")
        assert {f for f in flat.split(",") if f} <= set(result), tool


def test_example_runs_but_is_not_a_complete_answer():
    plan = parse_plan(json.dumps(EXAMPLE_PLAN))

    async def go():
        async with MCPProxy(source="sim") as proxy:
            return await execute_plan(plan, TASK, proxy.call)

    res = asyncio.run(go())
    assert res.ok and res.output["flight_out"]["from"] == "MAN"
    assert not validate_itinerary(TASK, res.output).ok
    assert json.dumps(EXAMPLE_PLAN, separators=(",", ":")) in SYSTEM_PROMPT


@pytest.mark.parametrize("key,value", [
    ("hotel", {"name": "Hotel Barcelona", "price_per_night": 120}),
    ("total_cost", 740),
    ("flight_out", {"$jp": "task.origin"}),
    ("flight_out", {"$jp": "steps.no_such_step"}),
    ("total_cost", {"$jp": "steps.made_up"}),
])
def test_invented_outputs_are_rejected(key, value):
    plan = heuristic_plan(TASK)
    plan["steps"].append({"id": "made_up", "kind": "compute", "expr": "740"})
    plan["output"][key] = value
    with pytest.raises(PlanParseError, match=f"output.{key}"):
        parse_plan(json.dumps(plan))


def test_heuristic_plans_and_nulls_pass_the_grounding_check():
    for task in load_dataset(ROOT / "data" / "sample_tasks.json")["tasks"]:
        plan = heuristic_plan(task)
        plan["output"].setdefault("experience", None)
        parse_plan(json.dumps(plan))


def test_repair_prompt_shows_the_failed_plan_and_keeps_the_task_readable():
    previous = Plan.model_validate(heuristic_plan(TASK)).model_dump_json(exclude_defaults=True)
    prompt = build_user_prompt(TASK, {}, ["over_budget: 2 > 1"], previous)
    assert previous in prompt and "over_budget" in prompt
    assert extract_task_from_prompt(prompt)["id"] == TASK["id"]


class FailingClient(LLMClient):
    provider = "mock"
    retry_delays = ()

    async def _raw(self, system, user):
        await asyncio.sleep(0.05)
        raise TimeoutError("slow")


def test_failed_calls_count_towards_latency():
    plan, resps, err = asyncio.run(plan_with_llm(FailingClient(), TASK, {}))
    assert plan is None and "TimeoutError" in err and len(resps) == 1
    assert resps[0].latency_ms >= 40 and resps[0].total_tokens == 0 and resps[0].error


def test_warm_up_is_logged_and_its_failure_does_not_stop_the_run(tmp_path):
    class Client(MockClient):
        async def warm_up(self):
            raise ConnectionError("offline")

    cfg = RunConfig(tasks_limit=1, policies=("replan",), perturbations=("P0",), logs_dir=tmp_path / "logs",
                    cache_dir=tmp_path / "cache")
    records = asyncio.run(run_benchmark(cfg, client=Client()))
    assert records[0]["success"]
    events = [json.loads(x) for x in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "offline" in next(e for e in events if e["type"] == "warm_up")["error"]

"""Live DAG drawing: pipeline states and step graph (pure functions, offline)."""
import asyncio
import xml.etree.ElementTree as ET

import pytest

import dag_view as d
from engine.batch_runner import RunConfig, run_benchmark

HIT = {"type": "cache_lookup", "status": "hit"}


def start(ts=0.0):
    return {"type": "llm_call_start", "ts": ts}


def llm_end(ok=True, ms=2000):
    return {"type": "llm_call_end", "ok": ok, "calls": [{"latency_ms": ms}]}


def ran(att, ok, violations=()):
    return [{"type": "step_start", "attempt": att, "step_id": "a"},
            {"type": "step_end", "attempt": att, "step_id": "a", "status": "ok"},
            {"type": "validation", "attempt": att, "ok": ok, "violations": list(violations)}]


def end(ok):
    return {"type": "episode_end", "success": ok}


def test_reuse_skips_the_llm():
    p = d.pipeline([HIT, *ran("reuse#1", True), end(True)], False)
    assert p.reused and p.loops == 0 and p.nodes["llm"] == ("skip", "0 tokens") and p.nodes["validate"][0] == "ok"


def test_failed_reuse_loops_back_to_a_running_llm():
    p = d.pipeline([HIT, *ran("reuse#1", False, ["x"]), start(10)], True, now=27)
    assert p.looping and p.loops == 1 and p.nodes["llm"] == ("active", "17 s")
    assert p.nodes["validate"] == ("fail", "1 violation") and p.nodes["book"][0] == "wait"


def test_running_steps_and_stopped_run():
    evs = [{"type": "cache_lookup", "status": "miss"}, start(), llm_end(),
           {"type": "step_start", "attempt": "replan#1", "step_id": "a"}]
    assert d.pipeline(evs, True).nodes["execute"] == ("active", "0 of 1 steps")
    assert d.pipeline(evs, False).nodes["execute"][0] == "skip"  # a dead run is never shown as working
    assert d.pipeline(evs[:2], False).nodes["llm"] == ("skip", "stopped")


def test_invalid_plan_budget_stop_and_execution_error():
    bad = d.pipeline([{"type": "cache_lookup", "status": "bypass"}, start(), llm_end(False), end(False)], False)
    assert bad.nodes["llm"][0] == "fail" and bad.nodes["execute"][0] == "wait" and bad.nodes["cache"][0] == "skip"
    stop = d.pipeline([HIT, *ran("reuse#1", False, ["x"]), {"type": "decision", "decision": "degrade_budget"}, end(False)],
                      False)
    assert stop.nodes["llm"] == ("skip", "no token budget") and stop.nodes["book"] == ("skip", "nothing booked")
    err = d.pipeline([HIT, start(), llm_end(), *ran("replan#1", False, ["execution_error: boom"]), end(False)], False)
    assert err.nodes["execute"][0] == "fail" and err.nodes["validate"][0] == "skip"


def test_untrusted_text_is_escaped():
    trace = [{"step_id": "a", "kind": "tool", "server": "<b>x</b>", "tool": chr(34) + chr(92) + 'N', "status": "ok", "latency_ms": 2500}]
    src = d.step_graph(trace).source
    assert "2.5 s" in src and 'label="' in src  # a quoted string, not an HTML label
    p = d.pipeline([{"type": "cache_lookup", "status": "<script>"}], False)
    svg = d.pipeline_svg(p)
    assert "<script>" not in svg
    ET.fromstring(svg)


def test_columns_follow_dependency_levels():
    trace = [{"step_id": "a", "kind": "tool", "tool": "search_flights", "server": "kiwi", "status": "ok"},
             {"step_id": "b", "kind": "select", "status": "ok", "depends_on": ["a"]},
             {"step_id": "c", "kind": "tool", "tool": "unknown", "server": "s", "status": "error"}]
    src = d.step_graph(trace).source
    assert src.count("rank=same") == 2 and "a -> b" in src and "✈" in src and "•" in src


@pytest.mark.parametrize("policy", ["replan", "budget_aware"])
def test_record_and_events_give_the_same_picture(tmp_path, policy):
    import json
    cfg = RunConfig(tasks_limit=2, policies=(policy,), logs_dir=tmp_path / "l", cache_dir=tmp_path / "c")
    records = asyncio.run(run_benchmark(cfg))
    events = [json.loads(x) for x in (tmp_path / "l" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in records:
        live = d.pipeline([e for e in events if e.get("episode_id") == r["episode_id"]], False)
        saved = d.pipeline(d.events_from_record(r), False)
        assert live.nodes == saved.nodes and (live.reused, live.loops) == (saved.reused, saved.loops), r["task_id"]
        ET.fromstring(d.pipeline_svg(saved))

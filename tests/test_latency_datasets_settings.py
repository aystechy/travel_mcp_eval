"""Offline tests: source provenance, latency calibration/model, datasets, settings."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine import calibrate_latency as cal
from engine import datasets as dsets
from engine import settings
from engine.batch_runner import RunConfig, parse_args, run_benchmark
from engine.latency_model import LatencyModel, profile_id
from engine.mcp_proxy import MCPProxy, ReplayCache, live_is_configured, replay_key

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "data" / "sample_tasks.json").read_text(encoding="utf-8"))
ARGS = {"origin": "MAN", "destination": "BCN", "date": "2027-05-14"}


def run(coro):
    return asyncio.run(coro)


def profile(**samples) -> dict:
    tools = {k.replace("__", "."): {"basis": "measured", "samples_ms": v} for k, v in samples.items()}
    return {"profile_id": profile_id(tools), "frozen_at": "2026-01-01", "tools": tools}


def cfg(tmp_path, **kw) -> RunConfig:
    return RunConfig(**{"logs_dir": str(tmp_path / "logs"), "cache_dir": str(tmp_path / "cache"), **kw})


# ------------------------------------------------------------------------------ provenance
def test_live_sources_are_refused_while_nothing_is_mapped(capsys):
    assert not live_is_configured()
    for source in ("live", "auto"):
        with pytest.raises(SystemExit):
            parse_args(["--source", source])
    assert "disabled" in capsys.readouterr().err
    assert parse_args(["--source", "replay"]).source == "replay"


def test_simulated_data_never_replaces_or_passes_as_live_recordings(tmp_path):
    key = replay_key("kiwi", "search_flights", ARGS)

    async def call(source):
        async with MCPProxy(source=source, replay_dir=tmp_path) as p:
            return (await p.call("kiwi", "search_flights", ARGS))[1]["source"]

    assert run(call("sim")) == "sim" and run(call("replay")) == "replay_sim"
    assert run(call("auto")) == "sim_fallback"  # auto does not serve simulator recordings as replay
    cache = ReplayCache(tmp_path)
    cache.put(key, "kiwi", "search_flights", ARGS, {"offers": []}, "live")
    assert run(call("sim")) == "sim"
    assert cache.get(key)["source"] == "live"
    assert run(call("replay")) == "replay" and run(call("auto")) == "replay"


# ------------------------------------------------------------------------------ latency model
def test_latency_model_follows_dag_levels_and_is_seeded():
    m = LatencyModel(profile(a__slow=[100.0], a__fast=[10.0]))
    tool = {"kind": "tool", "server": "a", "status": "ok", "args": {}}
    trace = [{**tool, "step_id": "s", "tool": "slow", "depends_on": []},
             {**tool, "step_id": "f", "tool": "fast", "depends_on": []},
             {"step_id": "pick", "kind": "select", "depends_on": ["s"], "status": "ok"},
             {**tool, "step_id": "g", "tool": "fast", "depends_on": ["pick"]},
             {**tool, "step_id": "skipped", "tool": "slow", "depends_on": ["g"], "status": "skipped"}]
    assert m.execution_ms(trace) == 110.0  # parallel: max(100, 10); then a later level: +10
    wide = LatencyModel(profile(a__t=[float(i) for i in range(1000)]))
    assert wide.draw("a", "t", {"x": 1, "y": 2}, 7) == wide.draw("a", "t", {"y": 2, "x": 1}, 7)
    assert len({wide.draw("a", "t", {"x": i}, 7) for i in range(20)}) > 10
    assert wide.draw("a", "unknown", {}, 7) is None


def test_edited_profile_is_rejected():
    p = profile(a__t=[1.0])
    p["tools"]["a.t"]["samples_ms"] = [2.0]
    with pytest.raises(ValueError):
        LatencyModel(p)


def test_modelled_latency_is_recorded_and_never_changes_outcomes(tmp_path):
    from engine.sim_servers import BUILDERS  # noqa: F401 - canonical tools below come from the simulators
    tools = ("kiwi__search_flights", "trivago__search_hotels", "ferryhopper__search_ferries", "mapbox__geocode",
             "mapbox__directions", "peek__search_experiences", "weather__forecast")
    path = tmp_path / "p.json"
    path.write_text(json.dumps(profile(**{t: [1000.0] for t in tools})), encoding="utf-8")
    common = dict(task_ids=("T01", "T02"), commit=False)
    a = run(run_benchmark(cfg(tmp_path / "a", latency_profile=str(path), **common)))
    b = run(run_benchmark(cfg(tmp_path / "b", latency_profile=str(tmp_path / "none.json"), **common)))
    keys = ("policy", "task_id", "success", "total_tokens", "decision_path", "output", "violations")
    assert [[r[k] for k in keys] for r in a] == [[r[k] for k in keys] for r in b]
    assert all(r["modelled_tool_ms"] is None and r["latency_profile_id"] is None for r in b)
    for r in a:
        assert r["latency_profile_id"] and r["modelled_tool_ms"] >= 1000 * (r["tool_calls"] > 0)
        assert r["modelled_tool_ms"] % 1000 == 0 and r["latency_ms"] < 60_000
        assert sum(x["modelled_tool_ms"] for x in r["attempts"]) == r["modelled_tool_ms"]
    twice = [r for r in a if len(r["attempts"]) == 2]
    assert twice and all(r["modelled_tool_ms"] > r["attempts"][0]["modelled_tool_ms"] for r in twice)


# ------------------------------------------------------------------------------ calibration (fake servers)
class FakeClient:
    def __init__(self, texts):
        self.texts, self.calls = texts, 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def call_tool(self, name, args, raise_on_error=False):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return SimpleNamespace(is_error=text is not None, content=[SimpleNamespace(text=text or "SECRET BODY")])


def _collect(tmp_path, monkeypatch, texts, calls=10, servers=None):
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "pk.test-token-value")
    path = tmp_path / "samples.jsonl"
    run(cal.collect(calls, servers, path=path, client_factory=lambda c, t: FakeClient(texts),
                    spacing=dict.fromkeys(cal.SPACING, 0)))
    return path


def test_calibration_stores_timings_only_and_freezes(tmp_path, monkeypatch):
    path = _collect(tmp_path, monkeypatch, [None])
    text = path.read_text(encoding="utf-8")
    assert "SECRET" not in text and "pk.test" not in text
    rows = cal.read_samples(path)
    assert len(rows) == 50 and set(rows[0]) == {"ts", "session", "server", "tool", "ok", "ms"}
    p = cal.build_profile(rows, "2026-01-01")
    assert p["tools"]["kiwi.search_flights"]["n"] == 10
    assert p["tools"]["peek.search_experiences"]["basis"] == "assumed"
    LatencyModel(p)
    with pytest.raises(ValueError):
        cal.build_profile(rows[:30], "2026-01-01")


def test_calibration_stops_when_throttled_or_failing(tmp_path, monkeypatch):
    rows = cal.read_samples(_collect(tmp_path, monkeypatch, [None, "HTTP 429 slow down"], servers={"kiwi"}))
    assert [r.get("error") for r in rows] == [None, "throttled"]
    rows = cal.read_samples(_collect(tmp_path, monkeypatch, ["boom"], servers={"trivago"}))
    assert [r["error"] for r in rows if r["server"] == "trivago"] == ["tool_error"] * 3


def test_every_simulated_tool_has_a_latency_entry():
    async def names():
        async with MCPProxy(source="sim") as p:
            cat = await p.catalog(include_actions=False)
        return {f"{s}.{t['name']}" for s, ts in cat.items() for t in ts}

    covered = {f"{s}.{t}" for s, t, _, _ in cal.PROBES} | set(cal.ASSUMED)
    assert run(names()) == covered


# ------------------------------------------------------------------------------ datasets
def test_dataset_roundtrip_derives_city_fields_and_marks_budgets():
    rows = dsets.to_rows(DATA)
    same, errors, changed = dsets.apply_rows(DATA, rows, dsets.origins())
    assert not errors and not changed and dsets.content_hash(same) == dsets.content_hash(DATA)
    original = copy.deepcopy(DATA)
    rows[0]["destination"] = "Naxos"
    rows[1]["budget"] = 9999.0
    new, errors, changed = dsets.apply_rows(DATA, rows, dsets.origins())
    assert not errors and changed == ["T01", "T02"] and DATA == original
    t = new["tasks"][0]
    assert t["destination"] == DATA["cities"]["Naxos"] and "ferry transfer from Piraeus" in t["description"]
    assert new["metadata"]["needs_recalibration"] == ["T01"]
    assert new["tasks"][1]["description"] == DATA["tasks"][1]["description"]
    assert dsets.content_hash(new) != dsets.content_hash(DATA)


@pytest.mark.parametrize("field,value", [("return_date", "2020-01-01"), ("budget", 0), ("budget", float("nan")),
                                         ("travelers", 0), ("travelers", 1.5), ("destination", "Atlantis"),
                                         ("origin", "Atlantis"), ("max_stops", 5), ("min_hotel_rating", 9),
                                         ("depart_date", None)])
def test_dataset_rejects_invalid_rows(field, value):
    rows = dsets.to_rows(DATA)
    rows[0][field] = value
    _, errors, _ = dsets.apply_rows(DATA, rows, dsets.origins())
    assert errors and errors[0].startswith("T01:")


def test_dataset_copy_never_overwrites_and_runs(tmp_path):
    rows = dsets.to_rows(DATA)
    rows[0]["travelers"] = 3
    new, _, _ = dsets.apply_rows(DATA, rows, dsets.origins())
    path = dsets.save_copy(new, "My Trips!", "sample_tasks", data_dir=tmp_path)
    assert path == tmp_path / "custom" / "my_trips.json"
    for name in ("my trips", "sample_tasks", "../.."):
        with pytest.raises(ValueError):
            dsets.save_copy(new, name, "sample_tasks", data_dir=tmp_path)
    r = run(run_benchmark(cfg(tmp_path, tasks_path=str(path), task_ids=("T01",), policies=("replan",),
                              perturbations=("P0",))))[0]
    assert (r["dataset"], r["dataset_hash"]) == ("my_trips", dsets.content_hash(new))


# ------------------------------------------------------------------------------ settings
def test_env_file_save_preserves_other_lines_and_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# keys\nOTHER=1\nOPENAI_MODEL=old\n", encoding="utf-8")
    settings.save_env_file({"OPENAI_MODEL": "new", "OPENAI_API_KEY": "sk-abc"}, env)
    assert env.read_text(encoding="utf-8") == "# keys\nOTHER=1\nOPENAI_MODEL=new\nOPENAI_API_KEY=sk-abc\n"
    for bad in ({"OPENAI_API_KEY": "a\nPATH=x"}, {"PATH": "x"}):
        with pytest.raises(ValueError):
            settings.save_env_file(bad, env)
    for k in settings.KEYS:
        monkeypatch.delenv(k, raising=False)
    assert settings.effective({}, env)["OPENAI_MODEL"] == ("new", ".env")
    assert settings.effective({}, env)["OLLAMA_MODEL"][1] == "default"
    monkeypatch.setenv("OPENAI_MODEL", "sys")
    assert settings.effective({}, env)["OPENAI_MODEL"] == ("sys", "environment")
    assert settings.effective({"OPENAI_MODEL": "s"}, env)["OPENAI_MODEL"] == ("s", "session")


def test_model_from_env_is_read_after_import(monkeypatch):
    from engine.llm_clients import make_client
    monkeypatch.setenv("OLLAMA_MODEL", "my-model")
    assert make_client("ollama").model == "my-model"
    assert make_client("ollama", "explicit").model == "explicit"


def test_secrets_are_redacted_and_missing_keys_reported(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    assert settings.redact("bad key sk-very-secret!") == "bad key ***!"
    ok, text = settings.check_connection("openai", dict.fromkeys(settings.KEYS, ""))
    assert not ok and "not set" in text


def test_dataset_and_settings_tabs_render_without_showing_secrets(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["logs_dir"] = str(tmp_path)
    at.session_state["dataset"] = "tasks_100"
    at.run()
    assert not at.exception
    assert any("tasks_100" in c.value and "100 tasks" in c.value for c in at.caption)
    assert {"Dataset", "Settings"} <= {t.label for t in at.tabs}
    shown = " ".join(str(e.value) for kind in ("markdown", "caption", "text_input", "table") for e in getattr(at, kind))
    assert "sk-very-secret" not in shown


def test_deepseek_client_request(monkeypatch):
    import openai
    from engine.llm_clients import make_client, price
    seen: dict = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                                   choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    monkeypatch.setattr(openai, "AsyncOpenAI", Fake)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    r = run(make_client("deepseek").complete("system json", "user"))
    assert (seen["base_url"], seen["model"]) == ("https://api.deepseek.com", "deepseek-v4-pro")
    assert seen["extra_body"] == {"thinking": {"type": "disabled"}} and r.total_tokens == 15
    assert price("deepseek", "deepseek-flash", 1_000_000, 0) == 0.3
    assert parse_args(["--provider", "deepseek"]).provider == "deepseek"
    assert {k for g in settings.GROUPS.values() for k in g} == set(settings.KEYS)


def test_settings_form_opens_on_demand(tmp_path):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["logs_dir"] = str(tmp_path)
    at.run()
    assert not [t for t in at.text_input if t.key and t.key.startswith("set_")]
    at.button(key="set_open_btn").click().run()
    at.selectbox(key="set_group").select("deepseek").run()
    assert not at.exception
    assert {t.key for t in at.text_input if t.key and t.key.startswith("set_")} == \
        {"set_DEEPSEEK_API_KEY", "set_DEEPSEEK_MODEL"}
    at.button(key="set_cancel").click().run()
    assert not [t for t in at.text_input if t.key and t.key.startswith("set_")]


# ------------------------------------------------------------------------------ real-time tool delays
def test_paced_run_waits_once_and_changes_nothing_else(tmp_path):
    tools = ("kiwi__search_flights", "trivago__search_hotels", "ferryhopper__search_ferries", "mapbox__geocode",
             "mapbox__directions", "peek__search_experiences", "weather__forecast")
    path = tmp_path / "p.json"
    path.write_text(json.dumps(profile(**{t: [400.0] for t in tools})), encoding="utf-8")
    common = dict(task_ids=("T01",), policies=("cache",), perturbations=("P0", "P1"), commit=False,
                  latency_profile=str(path))
    plain = run(run_benchmark(cfg(tmp_path / "a", **common)))
    paced = run(run_benchmark(cfg(tmp_path / "b", pace=True, **common)))
    keys = ("success", "total_tokens", "decision_path", "output", "modelled_tool_ms")
    assert [[r[k] for k in keys] for r in plain] == [[r[k] for k in keys] for r in paced]
    assert all(r["paced_wait_ms"] is None and not r["paced"] for r in plain)
    for r in paced:  # concurrent calls wait together, and the wait is not part of any measured time
        assert r["paced"] and r["modelled_tool_ms"] * 0.95 <= r["paced_wait_ms"] < r["modelled_tool_ms"] * 1.5
        wall = sum(s["latency_ms"] for s in r["dag"]["trace"] if s["kind"] == "tool")
        assert wall >= 400 * r["tool_calls"] * 0.95
        assert r["exec_latency_ms"] < 400 and 0 <= r["latency_ms"] < 400 + r["plan_latency_ms"]


def test_paced_run_needs_sim_and_a_profile(tmp_path):
    from engine.batch_runner import parse_args
    assert parse_args(["--pace"]).pace and not parse_args([]).pace
    for argv in (["--pace", "--source", "replay"], ["--pace", "--latency-profile", str(tmp_path / "none.json")]):
        with pytest.raises(SystemExit):
            parse_args(argv)
    with pytest.raises(ValueError, match="frozen latency profile"):
        run(run_benchmark(cfg(tmp_path, pace=True, latency_profile=str(tmp_path / "none.json"))))

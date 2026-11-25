"""Choosing a run in the sidebar must update every run-related tab."""
import asyncio
from pathlib import Path

import pytest

from engine.batch_runner import RunConfig, run_benchmark
from engine.runctl import clear_pid_file, write_pid_file

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    logs = tmp_path / "logs"
    for rid, tasks, perts in (("20261125-100000-aaaaaa", ("T01",), ("P0",)), ("20261125-110000-bbbbbb", ("T03",), ("P0", "P1"))):
        asyncio.run(run_benchmark(RunConfig(task_ids=tasks, perturbations=perts, policies=("cache",), logs_dir=logs,
                                            cache_dir=tmp_path / "c", run_id=rid)))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["logs_dir"] = str(logs)
    return at.run(), logs


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_every_tab_follows_the_selected_run(app):
    at, _ = app
    assert not at.exception and at.selectbox(key="run").value.endswith("bbbbbb")
    for rid, n, task in (("20261125-100000-aaaaaa", "1", "T01"), ("20261125-110000-bbbbbb", "2", "T03")):
        at.selectbox(key="run").select(rid).run()
        assert not at.exception
        assert metric(at, "Episodes done") == n and metric(at, "Episodes") == n  # Live and About
        assert any(f"cache · {task}-" in c.value for c in at.caption)  # Live DAG
        assert metric(at, "Bookings") == n  # Actions
        assert len(at.selectbox(key="insp").options) == int(n)  # Episode inspector
        assert all(task in o for o in at.selectbox(key="insp").options)
        assert any(f"`{rid}`" in m.value for m in at.markdown)


def test_cost_card_is_empty_without_an_api_charge(app):
    assert metric(app[0], "Est. cost") == "–"


def test_a_new_run_is_selected_once_and_the_user_can_switch_back(app):
    at, logs = app
    new = "20261125-120000-cccccc"
    write_pid_file(logs, new, ["test"])  # this process stands in for a live runner
    try:
        at.run()
        assert not at.exception and at.selectbox(key="run").value == new
        at.selectbox(key="run").select("20261125-100000-aaaaaa").run()
        assert not at.exception and metric(at, "Episodes done") == "1"
        at.run()
        assert at.selectbox(key="run").value.endswith("aaaaaa")
    finally:
        import os
        clear_pid_file(logs, os.getpid())

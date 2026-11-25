"""Provider faults are retried, stop the run or are excluded; they never count as model failures."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import openai
import pytest

from engine.batch_runner import RunConfig, run_benchmark, summarize
from engine.llm_clients import MockClient, classify_error

ROOT = Path(__file__).resolve().parent.parent


class HTTPError(Exception):
    def __init__(self, status, text="", headers=None):
        super().__init__(text)
        self.status_code = status
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


CREDIT = HTTPError(400, "Your credit balance is too low to access the Anthropic API.")


@pytest.mark.parametrize("error,provider,kind", [
    (CREDIT, "anthropic", "fatal"),
    (HTTPError(401), "openai", "fatal"),
    (HTTPError(402, "Insufficient Balance"), "deepseek", "fatal"),
    (HTTPError(404, "model not found"), "ollama", "fatal"),
    (HTTPError(429, "insufficient_quota"), "openai", "fatal"),
    (KeyError("DEEPSEEK_API_KEY"), "deepseek", "fatal"),
    (HTTPError(429, "rate limit"), "openai", "transient"),
    (HTTPError(529, "overloaded"), "anthropic", "transient"),
    (anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x.test")), "anthropic", "transient"),
    (httpx.ConnectError("refused"), "ollama", "transient"),
    (httpx.ReadTimeout("slow"), "ollama", None),  # the local model was too slow: a real failure
    (HTTPError(400, "prompt is too long"), "anthropic", None),
])
def test_classification(error, provider, kind):
    assert classify_error(error, provider) == kind


def test_openai_connection_errors_are_transient():
    request = httpx.Request("POST", "https://x.test")
    try:
        error = openai.APIConnectionError(request=request)
    except Exception:  # SDK built on httpx2
        error = openai.APIConnectionError(request=httpx2.Request("POST", "https://x.test"))
    assert classify_error(error, "openai") == "transient"


class Flaky(MockClient):
    """Raises the queued errors in order (None = answer normally; the last entry repeats)."""
    retry_delays = (0.02, 0.02, 0.02)

    def __init__(self, *errors):
        super().__init__()
        self.errors, self.calls = list(errors), 0

    async def _raw(self, system, user):
        self.calls += 1
        e = self.errors.pop(0) if len(self.errors) > 1 else self.errors[0]
        if e:
            raise e
        return await super()._raw(system, user)


def bench(tmp_path, client, **kw):
    cfg = RunConfig(task_ids=("T01", "T03"), policies=("replan",), logs_dir=tmp_path / "l", cache_dir=tmp_path / "c", **kw)
    records = asyncio.run(run_benchmark(cfg, client=client))
    events = [json.loads(x) for x in (tmp_path / "l" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return records, events[-1]


def test_transient_errors_are_retried_and_kept_out_of_measured_time(tmp_path):
    client = Flaky(HTTPError(429), HTTPError(503, headers={"retry-after": "0.5"}), None)
    records, end = bench(tmp_path, client, perturbations=("P0",))
    r = records[0]
    assert r["success"] and r["provider_error"] is None and r["llm_retries"] == 2 and r["llm_retry_ms"] > 500
    assert r["plan_latency_ms"] < 400 and r["latency_ms"] < r["llm_retry_ms"] + 400 and not end["stopped"]


def test_fatal_error_stops_the_run_at_once(tmp_path):
    client = Flaky(None, CREDIT)
    records, end = bench(tmp_path, client)
    assert len(records) == 2 and client.calls == 2  # no retry
    assert records[0]["success"] and records[1]["provider_error"]["kind"] == "fatal"
    assert records[1]["violations"][0].startswith("provider_error")
    assert end["stopped"] and "credit balance" in end["stop_reason"]
    text = summarize(records)
    assert "1 episode(s) excluded" in text and "1.0" in text


def test_outage_stops_after_three_lost_episodes_in_a_row(tmp_path):
    client = Flaky(HTTPError(503))
    records, end = bench(tmp_path, client)
    assert len(records) == 3 and client.calls == 12 and end["stopped"]
    assert "no valid episodes" in summarize(records)


def test_local_timeout_stays_a_model_failure(tmp_path):
    class Slow(Flaky):
        provider = "ollama"
    client = Slow(httpx.ReadTimeout("slow"))
    client.model = "x"
    records, end = bench(tmp_path, client)
    assert len(records) == 12 and not end["stopped"]
    assert all(r["provider_error"] is None and not r["success"] for r in records)


@pytest.mark.parametrize("legacy", [False, True])
def test_app_excludes_lost_episodes(tmp_path, legacy):
    from streamlit.testing.v1 import AppTest
    for logs, client in ((tmp_path / "mixed", Flaky(None, None, CREDIT)), (tmp_path / "lost", Flaky(CREDIT))):
        records, _ = bench(logs, client)
        path = logs / "l" / "runs.jsonl"
        if legacy:  # as logged before provider_error existed
            for r in records:
                if r.pop("provider_error"):
                    r["violations"] = ["plan_parse_error: llm_error: BadRequestError: credit"]
                r["provider"] = "anthropic"
            path.write_text("".join(json.dumps(r, default=str) + "\n" for r in records), encoding="utf-8")
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.session_state["logs_dir"] = str(logs / "l")
        at.run()
        assert not at.exception
        shown = [m.value for m in at.metric if m.label == "Success"]
        if len(records) > 1:
            assert shown == ["100%", "100%"] and any("1 of 3 episodes are excluded" in w.value for w in at.warning)
        else:
            assert shown == ["-", "–"]
        assert any("provider error" in e.value for e in at.error)

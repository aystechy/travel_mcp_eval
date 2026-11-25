"""LLM price table and cost accounting (offline)."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import anthropic
import openai
import pytest

from engine.batch_runner import RunConfig, run_benchmark
from engine.llm_clients import MockClient, make_client
from engine.prices import UNBILLED, find, load_prices, price
from engine.settings import DEFAULTS

ROOT = Path(__file__).resolve().parent.parent
HEAD = "provider,model,input,cached_input,cache_write,output,max_prompt_tokens,conditions,source_url,checked_on\n"


def test_every_row_is_sourced_and_default_models_are_listed():
    table = load_prices()
    for (provider, _), r in table.items():
        if provider not in UNBILLED:
            assert r["source_url"].startswith("https://") and r["checked_on"] and r["input"] > 0 and r["output"] > 0
            assert r["cached_input"] is None or r["cached_input"] < r["input"]
            assert r["cache_write"] is None or r["cache_write"] > r["input"]
    for p in ("openai", "anthropic", "deepseek", "mistral"):
        assert find(p, DEFAULTS[f"{p.upper()}_MODEL"])
        assert sum(k[0] == p for k in table) >= 3


def test_price_arithmetic_and_unknown_models():
    assert price("anthropic", "claude-sonnet-5-5", 1_000_000, 1_000_000) == 12.0
    # 0.5M plain at 2 + 0.3M cached at 0.1 + 0.2M written at 2.5
    assert price("anthropic", "claude-sonnet-5-5", 1_000_000, 0, 300_000, 200_000) == pytest.approx(1.53)
    assert price("openai", "gpt-5-pro", 1_000_000, 0, 1_000_000) == 15.0  # no listed discount: none assumed
    assert price("openai", "gpt-4o-mini-2030-01-01", 10, 10) is None
    assert price("anthropic", "claude-haiku-5-5", 100_000, 0) == 0.01
    assert price("anthropic", "claude-haiku-5-5", 100_001, 0) is None
    assert price("ollama", "anything", 10, 10) == 0.0 and price("mock", "heuristic-v1", 1_000_000, 0) == 3.0


@pytest.mark.parametrize("row", ["openai,m,-1,,,1,,,,", "openai,m,x,,,1,,,,", "openai,m,,,,1,,,,", "openai,*,1,,,1,,,,",
                                 "openai,m,nan,,,1,,,,", "openai,m,1,,,1,,,,\nopenai,m,1,,,1,,,,"])
def test_bad_tables_are_rejected(tmp_path, row):
    p = tmp_path / "p.csv"
    p.write_text(HEAD + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prices(p)


def fake_openai(monkeypatch, usage):
    seen = {}

    class Fake:
        def __init__(self, **kw):
            self.chat = NS(completions=NS(create=self.create))

        async def create(self, **kw):
            seen.update(kw)
            return NS(usage=usage, choices=[NS(message=NS(content="{}"))])
    monkeypatch.setattr(openai, "AsyncOpenAI", Fake)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    return seen


def complete(provider, model):
    return asyncio.run(make_client(provider, model).complete("s", "u"))


def test_openai_and_deepseek_cached_tokens_and_temperature(monkeypatch):
    seen = fake_openai(monkeypatch, NS(prompt_tokens=1000, completion_tokens=100,
                                       prompt_tokens_details=NS(cached_tokens=400, cache_write_tokens=100)))
    r = complete("openai", "gpt-6-luna")
    assert "temperature" not in seen and (r.cached_tokens, r.cache_write_tokens) == (400, 100)
    assert r.cost_usd == pytest.approx((500 * .1 + 400 * .01 + 100 * .125 + 100 * .5) / 1e6, abs=1e-6)
    assert r.cost_uncached_usd == pytest.approx((1000 * .1 + 100 * .5) / 1e6, abs=1e-6)
    complete("openai", "gpt-4o-mini")
    assert seen["temperature"] == 0
    fake_openai(monkeypatch, NS(prompt_tokens=1000, completion_tokens=0, prompt_cache_hit_tokens=900))
    r = complete("deepseek", "deepseek-flash")
    assert r.cached_tokens == 900 and r.cost_usd < r.cost_uncached_usd


@pytest.mark.parametrize("model,effort", [("mistral-small-2603", "none"), ("mistral-large-4-0", "low"),
                                          ("zai-glm-5-3", "low"), ("codestral-2508", None)])
def test_mistral_request_and_answer(monkeypatch, model, effort):
    seen = {}
    chunks = [{"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]}, {"type": "text", "text": "{}"}]

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)
            self.chat = NS(completions=NS(create=self.create))

        async def create(self, **kw):
            seen.update(kw)
            usage = NS(prompt_tokens=1000, completion_tokens=100, prompt_tokens_details=NS(cached_tokens=640))
            return NS(usage=usage, choices=[NS(message=NS(content=chunks if effort == "low" else "{}"))])
    monkeypatch.setattr(openai, "AsyncOpenAI", Fake)
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    r = complete("mistral", model)
    assert seen["base_url"] == "https://api.mistral.ai/v1" and seen["api_key"] == "x"
    assert (seen["extra_body"] or {}).get("reasoning_effort") == effort
    assert r.text == "{}" and r.cached_tokens == 640 and r.cost_usd is not None


def test_mistral_settings_and_connection_check(monkeypatch, tmp_path):
    import io

    from streamlit.testing.v1 import AppTest

    from engine import settings
    seen = {}

    class Opener:
        def open(self, request, timeout):
            seen.update(url=request.full_url, auth=request.get_header("Authorization"))
            return io.BytesIO(b'{"data": [{"id": "mistral-small-2603"}]}')
    with monkeypatch.context() as m:
        m.setattr(settings.urllib.request, "build_opener", lambda *a: Opener())
        ok, _ = settings.check_connection("mistral", {"MISTRAL_API_KEY": "k", "MISTRAL_MODEL": "mistral-small-2603"})
    assert ok and seen == {"url": "https://api.mistral.ai/v1/models", "auth": "Bearer k"}
    monkeypatch.setenv("MISTRAL_API_KEY", "mk-very-secret")
    assert "mk-very" not in settings.redact("bad key mk-very-secret")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["logs_dir"] = str(tmp_path)
    at.run()
    at.button(key="set_open_btn").click().run()
    at.selectbox(key="set_group").select("mistral").run()
    assert not at.exception and at.text_input(key="set_MISTRAL_MODEL")
    assert any("mistral" in s.options and "mock" in s.options for s in at.selectbox)


def test_anthropic_prompt_total_includes_cached_tokens(monkeypatch):
    class Fake:
        def __init__(self, **kw):
            self.messages = NS(create=self.create)

        async def create(self, **kw):
            return NS(content=[NS(type="text", text="{}")], usage=NS(
                input_tokens=100, output_tokens=10, cache_read_input_tokens=800, cache_creation_input_tokens=None))
    monkeypatch.setattr(anthropic, "AsyncAnthropic", Fake)
    r = complete("anthropic", "claude-sonnet-5-5")
    assert (r.prompt_tokens, r.cached_tokens) == (900, 800) and r.cost_usd == pytest.approx(0.00038)


@pytest.mark.parametrize("model,known", [("claude-sonnet-5-5", True), ("claude-unlisted", False)])
def test_run_logs_its_prices_and_the_app_never_guesses(tmp_path, model, known):
    from streamlit.testing.v1 import AppTest

    class Paid(MockClient):
        provider = "anthropic"
    cfg = RunConfig(task_ids=("T01",), policies=("cache",), logs_dir=tmp_path / "l", cache_dir=tmp_path / "c")
    records = asyncio.run(run_benchmark(cfg, client=Paid(model)))
    start = json.loads((tmp_path / "l" / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert start["prices_hash"] and bool(start["price"]) == known
    assert all((r["cost_usd"] is not None) == (known or r["llm_calls"] == 0) for r in records)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.session_state["logs_dir"] = str(tmp_path / "l")
    at.run()
    assert not at.exception
    shown = next(m.value for m in at.metric if m.label == "Est. cost")
    assert shown.startswith("$0.0") if known else shown == "–"

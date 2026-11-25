"""Unified async LLM wrapper: ``mock`` | ``ollama`` | ``openai`` | ``anthropic`` | ``deepseek``.

All clients return ``LLMResponse`` with token usage, wall latency and estimated cost so the
benchmark can compare the planning cost of re-planning vs. cache reuse on equal footing.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass

import httpx

from engine.prices import price
from engine.settings import DEFAULTS, redact
from engine.planner import (SYSTEM_PROMPT, PlanParseError, build_user_prompt, extract_task_from_prompt,
                            heuristic_plan, parse_plan)


def default_model(provider: str) -> str:
    """Model from the environment, read when a client is created (after .env is loaded)."""
    if provider == "mock":
        return "heuristic-v1"
    key = f"{provider.upper()}_MODEL"
    return os.environ.get(key) or DEFAULTS[key]


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float | None  # None: the model has no listed price
    error: str | None = None  # set when the call itself failed; its tokens are then unknown (0)
    error_kind: str | None = None  # fatal | transient (provider faults), or None (e.g. a local model was too slow)
    retries: int = 0
    retry_ms: float = 0.0  # failed tries and back-off before the final try; not part of latency_ms
    cached_tokens: int = 0  # part of prompt_tokens, served from the provider's prompt cache
    cache_write_tokens: int = 0  # part of prompt_tokens, written to it
    cost_uncached_usd: float | None = 0.0  # the same call without any prompt caching

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d["total_tokens"] = self.total_tokens
        return d


_CONNECTION_ERRORS = {"APIConnectionError", "TransportError", "ConnectionError", "TimeoutError"}


def classify_error(e: Exception, provider: str) -> str | None:
    """``transient`` (worth retrying), ``fatal`` (billing, key, model name, setup: every call would fail) or
    ``None`` (counts against the model: prompt too long, or a local model that ran into the time limit)."""
    names = {c.__name__ for c in type(e).__mro__}
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    text = str(e).lower()
    if status is None:
        if provider == "ollama" and names & {"ReadTimeout"}:
            return None
        return "transient" if names & _CONNECTION_ERRORS else "fatal"
    if "insufficient_quota" in text or "credit balance" in text:
        return "fatal"
    if status in (408, 409, 429) or status >= 500:
        return "transient"
    if status in (400, 413) and any(x in text for x in ("context length", "context window", "prompt is too long")):
        return None
    return "fatal"


def _retry_after(e: Exception) -> float:
    try:
        return min(60.0, max(0.0, float(e.response.headers.get("retry-after", 0))))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient:
    provider = "base"
    retry_delays: tuple[float, ...] = (2.0, 5.0, 15.0)  # seconds; SDK retries are off so each one is logged

    def __init__(self, model: str | None = None, timeout: float = 180.0):
        self.model = model or default_model(self.provider)
        self.timeout = timeout

    async def _raw(self, system: str, user: str) -> tuple:
        """(text, prompt tokens, completion tokens[, cached prompt tokens, cache-write prompt tokens])"""
        raise NotImplementedError

    async def warm_up(self) -> None:
        """Load the model before timing starts (local providers only)."""

    async def complete(self, system: str, user: str) -> LLMResponse:
        start, n = time.perf_counter(), 0
        while True:
            t0 = time.perf_counter()
            try:
                text, pt, ct, *cache = await self._raw(system, user)
            except Exception as e:  # noqa: BLE001 - reported to the caller with its kind and duration
                kind = classify_error(e, self.provider)
                if kind == "transient" and n < len(self.retry_delays):
                    await asyncio.sleep(max(self.retry_delays[n], _retry_after(e)))
                    n += 1
                    continue
                return LLMResponse("", self.provider, self.model, 0, 0, round((time.perf_counter() - t0) * 1000, 3),
                                   0.0, error=redact(f"llm_error: {type(e).__name__}: {e}"), error_kind=kind,
                                   retries=n, retry_ms=round((t0 - start) * 1000, 3))
            return LLMResponse(text, self.provider, self.model, pt, ct, round((time.perf_counter() - t0) * 1000, 3),
                               price(self.provider, self.model, pt, ct, *cache), retries=n,
                               retry_ms=round((t0 - start) * 1000, 3), cached_tokens=cache[0] if cache else 0,
                               cache_write_tokens=cache[1] if cache else 0,
                               cost_uncached_usd=price(self.provider, self.model, pt, ct))


class MockClient(LLMClient):
    """Deterministic heuristic planner behind the LLM interface. Token counts are estimated (chars/4)."""
    provider = "mock"

    async def _raw(self, system: str, user: str) -> tuple[str, int, int]:
        import json
        text = json.dumps(heuristic_plan(extract_task_from_prompt(user)), indent=1)
        return text, _est_tokens(system + user), _est_tokens(text)


class OllamaClient(LLMClient):
    provider = "ollama"
    num_ctx = 8192
    max_output = 4096  # stops runaway generation; thinking is disabled, as for DeepSeek

    async def warm_up(self) -> None:
        host = (os.environ.get("OLLAMA_HOST") or DEFAULTS["OLLAMA_HOST"]).rstrip("/")
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            # No prompt: loads the model with the same context size and generates nothing.
            r = await c.post(f"{host}/api/generate", json={"model": self.model, "keep_alive": "30m",
                                                          "options": {"num_ctx": self.num_ctx}})
            r.raise_for_status()

    async def _raw(self, system: str, user: str) -> tuple[str, int, int]:
        host = (os.environ.get("OLLAMA_HOST") or DEFAULTS["OLLAMA_HOST"]).rstrip("/")
        payload = {"model": self.model, "stream": False, "format": "json", "think": False, "keep_alive": "30m",
                   "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": self.max_output},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{host}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        text = data.get("message", {}).get("content", "")
        return text, int(data.get("prompt_eval_count") or _est_tokens(system + user)), int(
            data.get("eval_count") or _est_tokens(text))


def _openai_usage(u) -> tuple[int, int, int, int]:
    """Prompt, completion, cached and cache-write tokens (DeepSeek names its cache hits differently)."""
    if not u:
        return 0, 0, 0, 0
    d = getattr(u, "prompt_tokens_details", None)
    cached = getattr(u, "prompt_cache_hit_tokens", None) or getattr(d, "cached_tokens", None) or 0
    return u.prompt_tokens, u.completion_tokens, cached, getattr(d, "cache_write_tokens", None) or 0


class OpenAIClient(LLMClient):
    provider = "openai"

    async def _raw(self, system: str, user: str) -> tuple:
        # Reasoning models do not take a temperature; the others are kept deterministic.
        sampling = {"temperature": 0} if self.model.startswith("gpt-4") else {}
        from openai import AsyncOpenAI
        client = AsyncOpenAI(timeout=self.timeout, max_retries=0)
        r = await client.chat.completions.create(
            model=self.model, response_format={"type": "json_object"}, **sampling,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or "", *_openai_usage(r.usage)


class DeepSeekClient(LLMClient):
    """OpenAI-compatible. Thinking (on by default) is disabled: it ignores temperature and adds output tokens."""
    provider = "deepseek"

    async def _raw(self, system: str, user: str) -> tuple[str, int, int]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com",
                             timeout=self.timeout, max_retries=0)
        r = await client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=8192, response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or "", *_openai_usage(r.usage)


class AnthropicClient(LLMClient):
    provider = "anthropic"

    async def _raw(self, system: str, user: str) -> tuple[str, int, int]:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(timeout=self.timeout, max_retries=0)
        r = await client.messages.create(model=self.model, max_tokens=8192, system=system,
                                         messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")
        u = r.usage  # input_tokens leaves out cached and written tokens; output_tokens includes thinking
        cached, written = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        return text, u.input_tokens + cached + written, u.output_tokens, cached, written


CLIENTS = {c.provider: c for c in (MockClient, OllamaClient, OpenAIClient, AnthropicClient,
                                        DeepSeekClient)}


def make_client(provider: str, model: str | None = None) -> LLMClient:
    if provider not in CLIENTS:
        raise ValueError(f"provider must be one of {sorted(CLIENTS)}")
    return CLIENTS[provider](model)


async def plan_with_llm(client: LLMClient, task: dict, catalog: dict, feedback: list[str] | None = None,
                        parse_retries: int = 1, previous: str | None = None):
    """Ask the LLM for a plan. Returns ``(plan_or_None, [LLMResponse...], error_or_None)``."""
    user = build_user_prompt(task, catalog, feedback, previous)
    responses: list[LLMResponse] = []
    err: str | None = None
    for attempt in range(parse_retries + 1):
        resp = await client.complete(SYSTEM_PROMPT, user)
        responses.append(resp)
        if resp.error:
            return None, responses, resp.error
        try:
            return parse_plan(resp.text), responses, None
        except PlanParseError as e:
            err = str(e)
            user = build_user_prompt(task, catalog, feedback, previous) + (
                f"\nYour previous answer was not a valid plan ({err[:300]}). Return ONLY valid plan JSON.")
    return None, responses, err

"""Unified async LLM wrapper: ``mock`` | ``ollama`` | ``openai`` | ``anthropic`` | ``deepseek``.

All clients return ``LLMResponse`` with token usage, wall latency and estimated cost so the
benchmark can compare the planning cost of re-planning vs. cache reuse on equal footing.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import httpx

from engine.settings import DEFAULTS, redact
from engine.planner import (SYSTEM_PROMPT, PlanParseError, build_user_prompt, extract_task_from_prompt,
                            heuristic_plan, parse_plan)


def default_model(provider: str) -> str:
    """Model from the environment, read when a client is created (after .env is loaded)."""
    if provider == "mock":
        return "heuristic-v1"
    key = f"{provider.upper()}_MODEL"
    return os.environ.get(key) or DEFAULTS[key]


# USD per 1M tokens (input, output). ESTIMATES - verify against provider pricing before reporting.
# The mock planner is priced at a reference API rate so token savings translate into a cost axis.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "mock": (3.0, 15.0),
    "ollama": (0.0, 0.0),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai": (2.5, 10.0),
    "anthropic:claude-haiku-5-5": (1.0, 5.0),
    "anthropic": (3.0, 15.0),
    # DeepSeek peak, cache-miss prices (off-peak is half), read 2026-11-25.
    "deepseek:deepseek-v4-pro": (1.32, 3.96),
    "deepseek": (0.30, 1.20),
}


def price(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICING_PER_MTOK.get(f"{provider}:{model}", PRICING_PER_MTOK.get(provider, (0.0, 0.0)))
    return round((prompt_tokens * pin + completion_tokens * pout) / 1e6, 6)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    error: str | None = None  # set when the call itself failed; its tokens are then unknown (0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d["total_tokens"] = self.total_tokens
        return d


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient:
    provider = "base"

    def __init__(self, model: str | None = None, timeout: float = 180.0):
        self.model = model or default_model(self.provider)
        self.timeout = timeout

    async def _raw(self, system: str, user: str) -> tuple[str, int, int]:
        raise NotImplementedError

    async def warm_up(self) -> None:
        """Load the model before timing starts (local providers only)."""

    async def complete(self, system: str, user: str) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            text, pt, ct = await self._raw(system, user)
        except Exception as e:  # noqa: BLE001 - a failed call still took time, which must be counted
            return LLMResponse("", self.provider, self.model, 0, 0, round((time.perf_counter() - t0) * 1000, 3), 0.0,
                               error=redact(f"llm_error: {type(e).__name__}: {e}"))
        return LLMResponse(text, self.provider, self.model, pt, ct,
                           round((time.perf_counter() - t0) * 1000, 3), price(self.provider, self.model, pt, ct))


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


class OpenAIClient(LLMClient):
    provider = "openai"

    async def _raw(self, system: str, user: str) -> tuple[str, int, int]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(timeout=self.timeout)
        r = await client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        u = r.usage
        return r.choices[0].message.content or "", u.prompt_tokens if u else 0, u.completion_tokens if u else 0


class DeepSeekClient(LLMClient):
    """OpenAI-compatible. Thinking (on by default) is disabled: it ignores temperature and adds output tokens."""
    provider = "deepseek"

    async def _raw(self, system: str, user: str) -> tuple[str, int, int]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com",
                             timeout=self.timeout)
        r = await client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=8192, response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        u = r.usage
        return r.choices[0].message.content or "", u.prompt_tokens if u else 0, u.completion_tokens if u else 0


class AnthropicClient(LLMClient):
    provider = "anthropic"

    async def _raw(self, system: str, user: str) -> tuple[str, int, int]:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(timeout=self.timeout)
        r = await client.messages.create(model=self.model, max_tokens=8192, system=system,
                                         messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens


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

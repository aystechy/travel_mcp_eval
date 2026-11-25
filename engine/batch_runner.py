"""Standalone async benchmark worker.

For each policy (fresh plan cache per policy) and each task: run P0 (warms the cache), then P1..P5.
Writes step-level telemetry to ``logs/events.jsonl`` and episode records to ``logs/runs.jsonl``.

    python -m engine.batch_runner --provider mock --source sim
    python -m engine.batch_runner --provider ollama --policies budget_aware,replan --tasks-limit 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.datasets import content_hash
from engine.latency_model import DEFAULT_PROFILE, LatencyModel, load_profile
from engine.llm_clients import LLMClient, make_client, plan_with_llm
from engine.mcp_proxy import MCPProxy, catalog_fingerprint, live_is_configured
from engine.perturbations import KINDS, load_dataset, perturb
from engine.plan_cache import PlanCache
from engine.plan_executor import ExecutionResult, Plan, commit_actions, execute_plan, validate_itinerary
from engine.runctl import STOP_FILE, active_run, clear_pid_file, write_pid_file
from engine.settings import load_dotenv, redact

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ("replan", "cache", "budget_aware")


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, obj: dict) -> None:
        line = json.dumps(obj, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


@dataclass
class RunConfig:
    tasks_path: str = str(ROOT / "data" / "sample_tasks.json")
    policies: tuple[str, ...] = POLICIES
    perturbations: tuple[str, ...] = KINDS
    provider: str = "mock"
    model: str | None = None
    source: str = "sim"
    seed: int = 7
    episode_token_budget: int | None = 12000
    run_token_budget: int | None = None
    max_repairs: int = 1
    tasks_limit: int | None = None
    task_ids: tuple[str, ...] = ()
    logs_dir: str = str(ROOT / "logs")
    cache_dir: str = str(ROOT / "cache")
    replay_dir: str | None = None
    config_path: str | None = None
    latency_profile: str | None = None  # None = config/latency_profile.json if it exists
    pace: bool = False  # wait out the modelled delay of every tool call (demos, small checks)
    commit: bool = True
    reset_logs: bool = False
    label: str = ""  # free-text run description shown in the UI
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])


class Pacer:
    """Waits out each call's modelled delay. ``wait_ms`` is wall time with at least one call waiting, so
    concurrent waits count once and can be subtracted from the measured time."""

    def __init__(self, call, latency: LatencyModel, seed: int):
        self._call, self._latency, self._seed = call, latency, seed
        self._waiting, self._since, self.wait_ms = 0, 0.0, 0.0

    async def call(self, server: str, tool: str, args: dict):
        try:
            return await self._call(server, tool, args)
        finally:  # a failed call is charged too, as in LatencyModel.execution_ms
            if ms := self._latency.draw(server, tool, args, self._seed):
                await self._wait(ms)

    async def _wait(self, ms: float) -> None:
        if self._waiting == 0:
            self._since = time.perf_counter()
        self._waiting += 1
        try:
            await asyncio.sleep(ms / 1000)
        finally:
            self._waiting -= 1
            if self._waiting == 0:
                self.wait_ms += (time.perf_counter() - self._since) * 1000


@dataclass
class Attempt:
    mode: str  # reuse | replan
    ok: bool
    violations: list[str]
    result: ExecutionResult | None
    plan: Plan | None
    llm: list[dict] = field(default_factory=list)
    error: str | None = None
    recomputed_total: float | None = None
    paced_ms: float = 0.0  # simulated waiting, kept out of every measured time

    @property
    def exec_ms(self) -> float:
        return round(max(0.0, self.result.latency_ms - self.paced_ms), 3) if self.result else 0.0


class PolicyRunner:
    def __init__(self, cfg: RunConfig, policy: str, proxy: MCPProxy, client: LLMClient, catalog: dict,
                 fingerprint: str, events: JsonlWriter, runs: JsonlWriter,
                 latency: LatencyModel | None = None, dataset: dict | None = None):
        self.latency, self.dataset = latency, dataset or {}
        self.cfg, self.policy, self.proxy, self.client = cfg, policy, proxy, client
        self.catalog, self.fp, self.events, self.runs = catalog, fingerprint, events, runs
        self.cache = PlanCache(Path(cfg.cache_dir) / f"plan_cache_{cfg.run_id}_{policy}.json")
        self.run_tokens = 0
        self.ema_replan_tokens: float | None = None

    # ------------------------------------------------------------------ budget
    def remaining_budget(self, episode_tokens: int) -> float:
        rem = float("inf")
        if self.cfg.episode_token_budget is not None:
            rem = min(rem, self.cfg.episode_token_budget - episode_tokens)
        if self.cfg.run_token_budget is not None:
            rem = min(rem, self.cfg.run_token_budget - self.run_tokens)
        return rem

    def estimated_replan_tokens(self) -> float:
        return self.ema_replan_tokens if self.ema_replan_tokens is not None else 3000.0

    def can_afford_replan(self, episode_tokens: int) -> bool:
        return self.remaining_budget(episode_tokens) >= self.estimated_replan_tokens()

    def _observe_replan(self, tokens: int) -> None:
        self.ema_replan_tokens = tokens if self.ema_replan_tokens is None else 0.7 * self.ema_replan_tokens + 0.3 * tokens

    # ------------------------------------------------------------------ episode
    async def run_episode(self, task: dict) -> dict:
        episode_id = uuid.uuid4().hex[:10]
        pert = (task.get("perturbation") or {}).get("kind", "P0")
        base = {"run_id": self.cfg.run_id, "episode_id": episode_id, "task_id": task["id"],
                "base_id": task.get("base_id", task["id"]), "policy": self.policy, "perturbation": pert}
        trace: list[dict] = []
        tokens = {"prompt": 0, "completion": 0, "calls": 0, "cost": 0.0, "plan_ms": 0.0}
        attempts: list[Attempt] = []
        decisions: list[str] = []
        t_start = time.perf_counter()

        def emit(ev: dict, attempt: str | None = None) -> None:
            ev = {"ts": time.time(), **base, **({"attempt": attempt} if attempt else {}), **ev}
            self.events.append(ev)
            trace.append(ev)

        async def execute(plan: Plan, mode: str) -> Attempt:
            label = f"{mode}#{len(attempts) + 1}"
            pacer = Pacer(self.proxy.call, self.latency, self.cfg.seed) if self.cfg.pace else None
            paced = {"paced": True} if pacer else {}
            res = await execute_plan(plan, task, pacer.call if pacer else self.proxy.call,
                                     on_event=lambda e: emit({**e, **(paced if e.get("kind") == "tool" else {})}, label))
            rep = validate_itinerary(task, res.output) if res.ok else None
            violations = rep.violations if rep else [f"execution_error: {res.error}"]
            ok = bool(rep and rep.ok)
            emit({"type": "validation", "ok": ok, "violations": violations,
                  "recomputed_total": rep.recomputed_total if rep else None}, label)
            return Attempt(mode, ok, violations, res, plan, recomputed_total=rep.recomputed_total if rep else None,
                           paced_ms=pacer.wait_ms if pacer else 0.0)

        async def replan(feedback: list[str] | None, previous: Plan | None = None) -> Attempt:
            emit({"type": "llm_call_start", "provider": self.client.provider, "model": self.client.model,
                  "feedback": bool(feedback)})
            try:
                plan, resps, err = await plan_with_llm(
                    self.client, task, self.catalog, feedback,
                    previous=previous.model_dump_json(exclude_defaults=True) if previous else None)
            except Exception as e:  # noqa: BLE001 - provider/network errors are episode failures
                plan, resps, err = None, [], redact(f"llm_error: {type(e).__name__}: {e}")
            used = sum(r.total_tokens for r in resps)
            for r in resps:
                tokens["prompt"] += r.prompt_tokens
                tokens["completion"] += r.completion_tokens
                tokens["calls"] += 1
                tokens["cost"] += r.cost_usd
                tokens["plan_ms"] += r.latency_ms
            self.run_tokens += used
            if used:
                self._observe_replan(used)
            emit({"type": "llm_call_end", "tokens": used, "ok": plan is not None, "error": err,
                  "calls": [r.as_dict() for r in resps]})
            if plan is None:
                return Attempt("replan", False, [f"plan_parse_error: {err}"], None, None,
                               llm=[r.as_dict() for r in resps], error=err)
            att = await execute(plan, "replan")
            att.llm = [r.as_dict() for r in resps]
            return att

        def ep_tokens() -> int:
            return tokens["prompt"] + tokens["completion"]

        async def replan_with_repairs(budget_checked: bool) -> Attempt:
            att = await replan(None)
            attempts.append(att)
            repairs = 0
            while not att.ok and repairs < self.cfg.max_repairs and att.plan is not None:
                if budget_checked and not self.can_afford_replan(ep_tokens()):
                    decisions.append("repair_skipped_budget")
                    emit({"type": "decision", "decision": "repair_skipped_budget",
                          "remaining": self.remaining_budget(ep_tokens())})
                    break
                repairs += 1
                decisions.append("repair")
                att = await replan(att.violations, att.plan)
                attempts.append(att)
            return att

        emit({"type": "episode_start", "task": task})
        status, entry, reasons = ("bypass", None, []) if self.policy == "replan" else self.cache.lookup(task, self.fp)
        emit({"type": "cache_lookup", "status": status, "reasons": reasons,
              "entry": entry["key"] if entry else None})
        final: Attempt | None = None
        reuse_attempted = False

        if self.policy == "replan":
            decisions.append("replan")
            final = await replan_with_repairs(budget_checked=False)
        elif status == "hit":
            skip = False
            if self.policy == "budget_aware":
                rate, n = PlanCache.success_rate(entry)
                if n >= 2 and rate < 0.5 and self.can_afford_replan(ep_tokens()):
                    skip = True
                    decisions.append(f"skip_reuse_low_success({rate:.2f},n={n})")
                    emit({"type": "decision", "decision": "skip_reuse", "success_rate": rate, "uses": n})
            if not skip:
                reuse_attempted = True
                decisions.append("reuse")
                att = await execute(Plan.model_validate(entry["template"]), "reuse")
                attempts.append(att)
                self.cache.record(entry, f"{pert}", att.ok)
                final = att
                if not att.ok and self.policy == "budget_aware":
                    if self.can_afford_replan(ep_tokens()):
                        decisions.append("fallback_replan")
                        emit({"type": "decision", "decision": "fallback_replan",
                              "remaining": self.remaining_budget(ep_tokens()),
                              "estimate": self.estimated_replan_tokens()})
                        final = await replan_with_repairs(budget_checked=True)
                    else:
                        decisions.append("degrade_budget")
                        emit({"type": "decision", "decision": "degrade_budget",
                              "remaining": self.remaining_budget(ep_tokens())})
            else:
                final = await replan_with_repairs(budget_checked=True)
        else:  # stale / miss
            if self.policy == "budget_aware" and not self.can_afford_replan(ep_tokens()):
                decisions.append("degrade_budget")
                emit({"type": "decision", "decision": "degrade_budget",
                      "remaining": self.remaining_budget(ep_tokens())})
            else:
                decisions.append("replan")
                final = await replan_with_repairs(budget_checked=self.policy == "budget_aware")

        success = bool(final and final.ok)
        if success and final.mode == "replan" and self.policy != "replan":
            self.cache.store(final.plan, task, self.fp)
            emit({"type": "cache_store", "key": PlanCache.slot_key(task)})

        receipts: list[dict] = []
        if success and self.cfg.commit:
            self.proxy.action_context.clear()
            self.proxy.action_context.update({k: base[k] for k in ("run_id", "episode_id", "task_id", "policy",
                                                                     "perturbation")})
            rep = validate_itinerary(task, final.result.output)
            receipts = await commit_actions(final.plan, task, final.result, rep, self.proxy.call,
                                            on_event=lambda e: emit(e, "commit"))

        res = final.result if final else None
        out = res.output if res else None
        exec_ms = sum(a.exec_ms for a in attempts)
        paced_ms = sum(a.paced_ms for a in attempts)
        # Modelled, not measured: what the real servers would have added to these simulated calls.
        modelled = [self.latency.execution_ms(a.result.trace, self.cfg.seed) if a.result else 0.0
                    for a in attempts] if self.latency else None
        sources: dict[str, int] = {}
        for a in attempts:
            for st in (a.result.trace if a.result else []):
                if st.get("kind") == "tool" and st.get("status") != "skipped":
                    sources[st["source"]] = sources.get(st["source"], 0) + 1
        record = {
            **base, "ts": time.time(), "perturbation_detail": (task.get("perturbation") or {}).get("detail", "identity"),
            "provider": self.client.provider, "model": self.client.model, "source": self.cfg.source,
            "label": self.cfg.label, "dataset": self.dataset.get("name"), "dataset_hash": self.dataset.get("hash"),
            "cache_status": status, "cache_reasons": reasons, "cache_entry": entry["key"] if entry else None,
            "cross_task_reuse": bool(reuse_attempted and entry
                                     and not entry["created_from"].startswith(base["base_id"] + "-")),
            "reuse_attempted": reuse_attempted,
            "reuse_ok": bool(attempts and attempts[0].mode == "reuse" and attempts[0].ok),
            "replanned": any(a.mode == "replan" for a in attempts) or (final is not None and final.mode == "replan"),
            "decision_path": " > ".join(decisions) or "none",
            "llm_calls": tokens["calls"], "prompt_tokens": tokens["prompt"], "completion_tokens": tokens["completion"],
            "total_tokens": ep_tokens(), "cost_usd": round(tokens["cost"], 6),
            "plan_latency_ms": round(tokens["plan_ms"], 3), "exec_latency_ms": round(exec_ms, 3),
            "latency_ms": round((time.perf_counter() - t_start) * 1000 - paced_ms, 3),  # excludes simulated waiting
            "paced": self.cfg.pace, "paced_wait_ms": round(paced_ms, 3) if self.cfg.pace else None,
            "modelled_tool_ms": round(sum(modelled), 3) if modelled is not None else None,
            "latency_profile_id": self.latency.id if self.latency else None,
            "tool_calls": sum(a.result.tool_calls for a in attempts if a.result), "tool_sources": sources,
            "episode_token_budget": self.cfg.episode_token_budget,
            "over_token_budget": self.cfg.episode_token_budget is not None and ep_tokens() > self.cfg.episode_token_budget,
            "success": success, "violations": final.violations if final else ["budget_exhausted: no plan executed"],
            "attempts": [{"mode": a.mode, "ok": a.ok, "violations": a.violations, "llm": a.llm,
                          "exec_ms": a.exec_ms if a.result else None,
                          "modelled_tool_ms": modelled[i] if modelled is not None else None}
                         for i, a in enumerate(attempts)],
            "budget": task["budget"], "total_cost": (out or {}).get("total_cost"),
            "recomputed_total": final.recomputed_total if final else None,
            "actions": [{"tool": r["tool"], "confirmation": (r["result"] or {}).get("confirmation")
                         or (r["result"] or {}).get("event_id") or (r["result"] or {}).get("message_id")}
                        for r in receipts],
            "output": out, "plan": final.plan.model_dump() if final and final.plan else None,
            "dag": {"trace": res.trace if res else [], "edges": res.edges if res else []},
        }
        emit({"type": "episode_end", "success": success, "decision_path": record["decision_path"],
              "total_tokens": record["total_tokens"], "latency_ms": record["latency_ms"]})
        self.runs.append(record)
        return record


async def run_benchmark(cfg: RunConfig, client: LLMClient | None = None, progress: bool = False) -> list[dict]:
    logs = Path(cfg.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    if cfg.reset_logs:
        for name in ("events.jsonl", "runs.jsonl", "simulated_actions.jsonl"):
            (logs / name).write_text("", encoding="utf-8")
    events, runs = JsonlWriter(logs / "events.jsonl"), JsonlWriter(logs / "runs.jsonl")
    data = load_dataset(cfg.tasks_path)
    tasks = [t for t in data["tasks"] if not cfg.task_ids or t["id"] in cfg.task_ids]
    if cfg.tasks_limit:
        tasks = tasks[: cfg.tasks_limit]
    dataset = {"name": Path(cfg.tasks_path).stem, "hash": content_hash(data)}
    # The model describes real servers behind simulated calls; live calls are timed for real.
    latency = load_profile(cfg.latency_profile) if cfg.source in ("sim", "replay") else None
    if cfg.pace and (latency is None or cfg.source != "sim"):
        raise ValueError("real-time tool delays need --source sim and a frozen latency profile")
    client = client or make_client(cfg.provider, cfg.model)
    stop_file = logs / STOP_FILE
    stop_file.unlink(missing_ok=True)  # a stale STOP from an earlier run must not abort this one
    stopped = False
    records: list[dict] = []
    events.append({"ts": time.time(), "run_id": cfg.run_id, "type": "run_start",
                   "config": {k: v for k, v in cfg.__dict__.items()}, "n_tasks": len(tasks),
                   "dataset": dataset, "model": client.model, "latency_profile_id": latency.id if latency else None})
    async with MCPProxy(cfg.config_path, source=cfg.source,
                        replay_dir=cfg.replay_dir or Path(cfg.cache_dir) / "replay",
                        actions_log=logs / "simulated_actions.jsonl") as proxy:
        catalog = await proxy.catalog()
        t0, warm_error = time.perf_counter(), None
        try:
            await client.warm_up()
        except Exception as e:  # noqa: BLE001 - the episodes will report the real failure
            warm_error = redact(f"{type(e).__name__}: {e}")
        events.append({"ts": time.time(), "run_id": cfg.run_id, "type": "warm_up", "error": warm_error,
                       "latency_ms": round((time.perf_counter() - t0) * 1000, 3)})
        fp = catalog_fingerprint(catalog)
        for policy in cfg.policies:
            runner = PolicyRunner(cfg, policy, proxy, client, catalog, fp, events, runs, latency, dataset)
            for task in tasks:
                for kind in cfg.perturbations:
                    if stop_file.exists():  # graceful stop: checked only between episodes
                        stopped = True
                        break
                    rec = await runner.run_episode(perturb(task, kind, cfg.seed, data["cities"]))
                    records.append(rec)
                    if progress:
                        print(f"[{policy:12s}] {rec['task_id']:8s} {rec['cache_status']:6s} "
                              f"{'OK ' if rec['success'] else 'FAIL'} tok={rec['total_tokens']:5d} "
                              f"{rec['decision_path']}", flush=True)
                if stopped:
                    break
            if stopped:
                break
        stop_file.unlink(missing_ok=True)
        if stopped and progress:
            print("Stop requested: run ended after the current episode.", flush=True)
        events.append({"ts": time.time(), "run_id": cfg.run_id, "type": "run_end", "stopped": stopped,
                       "episodes": len(records), "tool_stats": proxy.stats})
    return records


def summarize(records: list[dict]) -> str:
    import pandas as pd
    if not records:
        return "no records"
    df = pd.DataFrame(records)
    g = df.groupby("policy").agg(episodes=("success", "size"), success_rate=("success", "mean"),
                                 mean_tokens=("total_tokens", "mean"), llm_calls=("llm_calls", "sum"),
                                 cost_usd=("cost_usd", "sum"), mean_latency_ms=("latency_ms", "mean"))
    p = df.pivot_table(index="policy", columns="perturbation", values="success", aggfunc="mean")
    return g.round(3).to_string() + "\n\nSuccess rate by perturbation:\n" + p.round(2).to_string()


def parse_args(argv: list[str] | None = None) -> RunConfig:
    ap = argparse.ArgumentParser(description="Plan-cache reuse vs LLM re-planning benchmark")
    ap.add_argument("--tasks", default=RunConfig.tasks_path)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--perturbations", default=",".join(KINDS))
    ap.add_argument("--provider", default="mock", choices=["mock", "ollama", "openai", "anthropic", "deepseek"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--source", default="sim", choices=["sim", "live", "replay", "auto"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--episode-token-budget", type=int, default=12000, help="<=0 disables")
    ap.add_argument("--run-token-budget", type=int, default=0, help="<=0 disables")
    ap.add_argument("--max-repairs", type=int, default=1)
    ap.add_argument("--tasks-limit", type=int, default=None)
    ap.add_argument("--task-ids", default="")
    ap.add_argument("--logs-dir", default=RunConfig.logs_dir)
    ap.add_argument("--cache-dir", default=RunConfig.cache_dir)
    ap.add_argument("--no-commit", action="store_true", help="skip the action commit phase")
    ap.add_argument("--reset-logs", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--label", default="", help="short free-text description of this run")
    ap.add_argument("--latency-profile", default=None, help="frozen latency profile (default: config/latency_profile.json)")
    ap.add_argument("--pace", action="store_true", help="wait out the modelled tool delays in real time (slow; demos)")
    a = ap.parse_args(argv)
    if a.pace and (a.source != "sim" or not Path(a.latency_profile or DEFAULT_PROFILE).exists()):
        ap.error("--pace needs --source sim and a frozen latency profile")
    if a.source in ("live", "auto") and not live_is_configured():
        ap.error(f"--source {a.source} is disabled: config/servers.json has no live tool mappings, so every "
                 "call would silently use the simulator (see docs/ARCHITECTURE.md)")
    policies = tuple(p for p in a.policies.split(",") if p)
    bad = set(policies) - set(POLICIES)
    if bad:
        ap.error(f"unknown policies {bad}")
    cfg = RunConfig(tasks_path=a.tasks, policies=policies,
                    perturbations=tuple(p for p in a.perturbations.split(",") if p), provider=a.provider,
                    model=a.model, source=a.source, seed=a.seed,
                    episode_token_budget=a.episode_token_budget if a.episode_token_budget > 0 else None,
                    run_token_budget=a.run_token_budget if a.run_token_budget > 0 else None,
                    max_repairs=a.max_repairs, tasks_limit=a.tasks_limit,
                    task_ids=tuple(t for t in a.task_ids.split(",") if t), logs_dir=a.logs_dir,
                    cache_dir=a.cache_dir, commit=not a.no_commit, reset_logs=a.reset_logs,
                    latency_profile=a.latency_profile, pace=a.pace,
                    label=" ".join(a.label.split())[:80])
    if a.run_id:
        cfg.run_id = a.run_id
    return cfg


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    cfg = parse_args(argv)
    other = active_run(cfg.logs_dir)
    if other is not None:
        print(f"Another run is active (PID {other['pid']}, run_id {other.get('run_id')}) on {cfg.logs_dir}. "
              f"Stop it first (touch {Path(cfg.logs_dir) / STOP_FILE}).", file=sys.stderr)
        return 2
    write_pid_file(cfg.logs_dir, cfg.run_id, sys.argv)
    try:
        print(f"run_id={cfg.run_id} pid={os.getpid()} provider={cfg.provider} source={cfg.source} "
              f"policies={cfg.policies}", flush=True)
        records = asyncio.run(run_benchmark(cfg, progress=True))
        print("\n" + summarize(records))
        return 0
    finally:
        clear_pid_file(cfg.logs_dir, os.getpid())


if __name__ == "__main__":
    sys.exit(main())

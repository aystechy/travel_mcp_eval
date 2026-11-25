# travel_mcp_eval — CLAUDE.md

Research benchmark: **budget-aware plan-cache reuse vs. full LLM re-planning** for MCP travel agents.
An LLM writes a JSON plan (a DAG of MCP tool calls + JMESPath selections + safe math). The plan is
abstracted into a task-parameterised template, cached, and re-executed on perturbed tasks (P1–P5)
by a model-free CPU executor. A deterministic validator decides whether reuse succeeded.

## Layout
- `engine/plan_executor.py` — Plan schema (pydantic), DAG topo-executor, `$jp`/`$expr` resolution,
  `safe_eval` (AST whitelist), `validate_itinerary` (deterministic validator).
- `engine/plan_cache.py` — `abstract_plan` (lift literals → `task.*` refs), structural coverage,
  compatibility check, JSON persistence, per-entry success stats.
- `engine/planner.py` — planner prompt, plan parsing/repair, `HeuristicPlanner` (mock LLM),
  `TOOL_RESULTS` (result fields shown to LLMs; tested against the simulators), `check_grounded` (rejects invented outputs).
- `engine/llm_clients.py` — unified `LLMClient` for `mock | ollama | openai | anthropic | deepseek | mistral`.
- `engine/mcp_proxy.py` — multi-server client: Streamable HTTP (live) + in-memory FastMCP (sim/actions),
  SHA-256 disk replay cache in `cache/replay/`.
- `engine/sim_servers.py` — deterministic in-memory stand-ins for the 6 travel servers.
- `engine/local_action_mcp.py` — FastMCP action server (book_hotel, create_calendar_event, send_confirmation).
- `engine/perturbations.py` — deterministic P1–P5. `engine/batch_runner.py` — async CLI, policies, logging.
- `engine/latency_model.py` — frozen, seeded tool-latency model (stdlib); `engine/calibrate_latency.py` — the only
  code that contacts real MCP servers, run manually, stores timings only.
- `engine/datasets.py` — dataset hash, table view, validated copies; `engine/settings.py` — `.env`, precedence,
  redaction, connection test (both stdlib only).
- `app.py` — Streamlit; reads log files, edits dataset copies and settings (invariant 9).
- `dag_view.py` — Live DAG drawing (pipeline SVG, step graph); pure functions, no Streamlit or engine imports.
- `docs/ARCHITECTURE.md` — diagrams, design decisions, calibration, real-server findings (for the report).
- `docs/ABOUT.md` — About tab text (edit here, not in `app.py`); `<!-- RUN_CARD -->` marks where the live
  run card renders. Keep it in sync when policies, perturbations or defaults change.

## Architectural invariants (do not break)
1. **Executor is model-free.** `plan_executor.py` never imports `llm_clients`/`planner`. Reuse costs 0 LLM tokens.
2. **Validator is deterministic and independent.** It recomputes `total_cost` from components and never
   trusts the plan's own total or the LLM. Same task + same tool outputs ⇒ same verdict.
3. **Coverage is derived, not declared.** Cache compatibility comes from scanning the plan for
   `task.constraints.<key>` refs and `search_ferries` steps — never from LLM-written metadata.
4. **Actions commit only after validation.** `plan.actions` run only when `validate_itinerary` returns no
   violations. Planning, reuse and failed episodes must never write to `simulated_actions.jsonl`.
5. **Perturbations are pure functions** of `(task, kind, seed)`; never mutate the input task.
6. **`safe_eval` stays a whitelist** (numbers, arithmetic, comparisons, bool ops, ternary, whitelisted
   funcs). No attribute access, subscripts, lambdas, comprehensions or `eval`/`exec` anywhere.
7. **JMESPath templating** `${expr}` is rendered as a JSON literal in backticks (escaped), never raw text.
8. **Replay-cache key** = sha256 of canonical JSON `{server, tool, args}` (sorted keys). Changing the key
   format invalidates `cache/replay/`; bump `REPLAY_KEY_VERSION` if you must.
9. **UI is decoupled.** `app.py` never imports the engine for execution; it reads `logs/*.jsonl`
   and may spawn `batch_runner` as a subprocess. It may import only `runctl`, `datasets`, `settings` and `prices`
   (stdlib only), and writes only new files in `data/custom/` and, on request from a local browser, `.env`.
   Built-in datasets and existing copies are never overwritten. Logs are append-only JSONL, one object per line;
   the sole exception is user-confirmed `runctl.delete_run`, refused while any run is active.
10. **Secrets** come only from env / `.env` / the Settings session (passed via the subprocess environment, never
    argv); never display or log them, never write them into logs or the cache; `redact()` provider errors.
11. **Sim vs live is always recorded.** Every tool call event carries `source` ∈
    `sim | live | replay | replay_sim | sim_fallback` (`replay` = recorded live only). Never present sim results as live.
12. **Modelled latency is never presented as measured.** It is computed after the episode, never affects decisions,
    and is logged separately (`modelled_tool_ms`, `latency_profile_id`). `--pace` waits are simulated: subtracted from
    every measured episode time, logged as `paced_wait_ms`, and never added to `modelled_tool_ms`. Never edit the frozen profile by hand.
13. **`live`/`auto` stay refused** until `config/servers.json` has mappings backed by real request/response adapters.

14. **Provider faults are not model failures.** `classify_error` marks them (`provider_error` in the record); transient ones are
    retried (logged, kept out of measured time), fatal ones stop the run, and the episodes are excluded from success rates
    and reported as a count. Local-model timeouts, over-long prompts and bad plans stay model failures.

## Policies (batch_runner)
- `replan` — always LLM-plan each episode (baseline), up to `--max-repairs` re-plans with violations.
- `cache` — reuse compatible cached plan; LLM only on miss/stale; no fallback after a failed reuse.
- `budget_aware` — reuse if compatible; skip reuse if entry success-rate < 0.5 (n≥2) and budget allows;
  on failed validation re-plan only if `remaining_budget ≥ EMA(replan tokens)`; else degrade.

## Commands
Run everything inside the `travelenv` venv (git-ignored): `.\travelenv\Scripts\Activate.ps1`, or call
`travelenv/Scripts/python` directly (`travelenv/bin/python` on macOS/Linux).
```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/                         # full verification suite (offline, sim only)
python -m engine.batch_runner --provider mock --source sim --label "baseline"  # full offline benchmark
python -m engine.batch_runner --provider ollama --policies budget_aware,replan --tasks-limit 3
python -m engine.batch_runner --tasks data/tasks_100.json --provider ollama   # 100 tasks, 1,800 episodes
python -m engine.calibrate_latency collect --calls 10  # LIVE, slow; then `freeze [--replace]`
python -m engine.local_action_mcp                  # run action server standalone (stdio)
streamlit run app.py                               # observability UI
```
Run IDs stay `YYYYMMDD-HHMMSS-xxxxxx` (used in file names; never embed model names, which may contain `:`).
Provider/model/source/label are display metadata only (`run_start.config`, each run record).
Run control (`engine/runctl.py`, stdlib only; the one engine module `app.py` may import):
the CLI runner writes `<logs_dir>/batch_runner.pid` (pid + process creation time) and removes it on
exit; a second runner on the same logs dir refuses to start. Graceful stop: create `<logs_dir>/STOP`;
checked only between episodes, logged as `run_end.stopped: true`, then deleted. Force stop kills the
PID only after verifying its creation time (Windows reuses PIDs).
`--source`: `sim` (deterministic, default) or `replay` (disk cache only; miss = error). `live` and `auto`
exist in `MCPProxy` but the CLI refuses them while nothing is mapped (invariant 13).

## Conventions
- Python ≥3.11, async via `asyncio`; file I/O always `encoding="utf-8"`.
- Tests must stay offline: `provider=mock`, `source=sim`, logs in `tmp_path`.
- Endpoint URLs in `config/servers.json` were checked on 2026-11-25 and serve only the calibration.
- Calibration is manual and slow: never call live servers from tests, the app or a benchmark run, and never
  store or log their responses (untrusted text, unclear storage rights).
- Prices live in `config/llm_prices.csv` (loader `engine/prices.py`), read by hand from provider pages with `source_url` and
  `checked_on`. Exact model match; an unlisted model costs `None`, never a guess. Recheck before reporting cost.

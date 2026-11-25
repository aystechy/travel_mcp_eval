# Reuse or Replan? (`travel_mcp_eval`)

A research benchmark for MCP travel agents. It asks whether a plan an LLM has already written can be
cached and re-executed on a changed task, cutting LLM cost without losing task success.

An LLM writes a JSON plan (a DAG of MCP tool calls). The plan is stored as a template and re-run on
perturbed tasks without calling the LLM. A deterministic validator decides whether each itinerary passes.

| Policy | Rule |
|---|---|
| `replan` | Call the LLM every episode (baseline) |
| `cache` | Reuse a compatible plan; call the LLM only when there is none; accept a failed reuse |
| `budget_aware` | Reuse, and re-plan after a failure if the remaining token budget allows |

Each task runs unchanged (P0) and with five perturbations (P1-P5: dates, destination, budget,
constraints). The default run is 10 tasks x 6 variants x 3 policies = 180 episodes.

## Quick start

Requires Python 3.11+. No API keys are needed for the offline run.

The project runs in a virtual environment named `travelenv`. Create and activate it once:

```powershell
# Windows (PowerShell)
python -m venv travelenv
.\travelenv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv travelenv
source travelenv/bin/activate
```

In each new terminal, run only the activate line. If PowerShell blocks the script, first run
`Set-ExecutionPolicy -Scope Process RemoteSigned`. With `travelenv` active:

```bash
python -m pip install -r requirements.txt                     # first time only
python -m pytest -q tests/                                    # offline tests
python -m engine.batch_runner --provider mock --source sim --label "baseline"
python -m streamlit run app.py                                          # results UI
```

You can also start, stop and delete runs from the UI sidebar.

## Real LLMs

Enter keys and model names in the **Settings** tab, or copy `.env.example` to `.env`, then (with `travelenv`
active):

```bash
python -m engine.batch_runner --provider ollama --policies budget_aware,replan --tasks-limit 3
python -m engine.batch_runner --provider anthropic --tasks data/tasks_100.json
```

| Option | Values |
|---|---|
| `--provider` | `mock` (offline heuristic), `ollama`, `openai`, `anthropic`, `deepseek` |
| `--model` | Overrides the provider's default model |
| `--source` | `sim` (simulators, default) or `replay` (recorded responses only). `live` and `auto` are disabled |
| `--tasks` | Dataset file; default `data/sample_tasks.json` |
| `--policies`, `--perturbations` | Comma-separated subsets |
| `--pace` | Wait out the modelled tool delays in real time (demos, small runs; needs `sim`) |
| `--tasks-limit` | Run only the first N tasks |
| `--episode-token-budget` | Default 12000; 0 disables |

Run `python -m engine.batch_runner --help` for all options.

## Datasets

The **Dataset** tab shows `sample_tasks` (10 tasks) and `tasks_100` (100 tasks, 1,800 episodes) as editable
tables. Edits are saved as copies in `data/custom/`; the built-in sets never change. Select **Use for next
run** to run the one shown. Each run logs its dataset name and content hash.

## Tool latency

Simulated tools answer in milliseconds, so real-server delays were measured **once** and frozen in
`config/latency_profile.json`. Runs add this *modelled* tool time to finished episodes without waiting, and
report it separately from measured time. No run contacts a real server. Method and results:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Layout

| Path | Contents |
|---|---|
| `engine/` | Planner, executor and validator, plan cache, MCP client, simulators, perturbations, batch runner |
| `app.py` | Streamlit UI; displays the logs and launches runs |
| `config/servers.json` | MCP server endpoints (used only for calibration) |
| `config/latency_*.json*` | Calibration timings and the frozen latency profile |
| `data/` | `sample_tasks.json` (10 tasks), `tasks_100.json` and its generator, saved copies in `custom/` |
| `docs/ABOUT.md` | Full description of the experiment (the About tab) |
| `docs/ARCHITECTURE.md` | Architecture diagrams, design decisions, calibration and real-server findings |
| `tests/` | Offline test suite |
| `travelenv/` | Virtual environment (created locally, not committed) |
| `logs/` | Run output: `runs.jsonl` (one record per episode), `events.jsonl` (trace), `simulated_actions.jsonl` |
| `cache/` | Plan caches and recorded tool responses |

Design rules for contributors are in `CLAUDE.md`.

## Limitations

- **Tool data, prices and budgets are synthetic.** Results do not show how agents cope with real prices,
  availability or server failures.
- **Real MCP servers are not used in runs.** Checked on 25 November 2026: Kiwi, trivago, Ferryhopper and Mapbox
  responded; Peek searches failed; no weather server was configured. Their formats need adapters, their data
  changes between runs, and rights to store results are unclear or restricted.
- **Tool latency is modelled,** covers normal delays only, and is assumed for experiences and weather.
- Flight and ferry times are not validated, so same-day connections may be unrealistic.
- The `mock` planner is not an LLM: its token counts are estimates and its latency is near zero.
- Prices are estimates; check `engine/llm_clients.py` before reporting cost.
- Bookings, calendar events and emails are simulated; nothing is booked or sent.
- One seed per run, and the default set has only 10 tasks; use `tasks_100` and several seeds before drawing conclusions.

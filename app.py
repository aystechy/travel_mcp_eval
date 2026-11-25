"""Streamlit observability UI for travel_mcp_eval.

Decoupled by design: this app reads logs/*.jsonl and may spawn engine.batch_runner as a subprocess.
It never executes plans in-process. It writes only dataset copies (data/custom) and, on request, .env.

    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import dag_view  # noqa: E402  (pure drawing code)
from engine import datasets as dsets, settings  # noqa: E402  (stdlib only, no execution)
from engine.runctl import (STOP_FILE, RunActiveError, active_run, delete_run,  # noqa: E402
                           force_stop, request_stop)  # process control only, never execution

ROOT = Path(__file__).resolve().parent
POLICIES = ["replan", "cache", "budget_aware"]
UNBILLED = ("mock", "ollama")  # no API charge
PERTS = ["P0", "P1", "P2", "P3", "P4", "P5"]
# Reference categorical palette, slots 1-4 (slots 1-3 validated all-pairs; 1-4 adjacent for stacks).
POLICY_COLOR = {"replan": "#2a78d6", "cache": "#eb6834", "budget_aware": "#1baf7a"}
STATUS_ORDER = ["hit", "stale", "miss", "bypass"]
STATUS_COLOR = {"hit": "#2a78d6", "stale": "#eb6834", "miss": "#1baf7a", "bypass": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"

st.set_page_config(page_title="Plan-cache vs re-planning", layout="wide")


# ----------------------------------------------------------------------------- data loading
def read_jsonl(path: Path, offset: int = 0, run_id: str | None = None) -> tuple[list[dict], int]:
    """Read complete JSONL lines from ``offset`` (of one run only, if given); tolerate a partial last line."""
    needle = run_id.encode() if run_id else None
    if not path.exists():
        return [], 0
    rows: list[dict] = []
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1
    for line in data[:end].splitlines():
        if needle and needle not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if run_id is None or row.get("run_id") == run_id:
            rows.append(row)
    return rows, offset + end


@st.cache_data(ttl=2, show_spinner=False)
def load_runs(path: str, mtime: float) -> pd.DataFrame:
    rows, _ = read_jsonl(Path(path))
    return pd.DataFrame(rows)


@st.cache_data(ttl=2, show_spinner=False)
def load_actions(path: str, mtime: float) -> pd.DataFrame:
    rows, _ = read_jsonl(Path(path))
    return pd.DataFrame(rows)


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def episode_labels(frame: pd.DataFrame, cols: list[str]) -> dict[str, str]:
    """episode_id -> label, bound per render so a later run's data never relabels an old widget."""
    return {r["episode_id"]: " · ".join(str(r[c]) for c in cols) for r in frame.to_dict("records")}


def last_run_end(path: Path, tail_bytes: int = 65536) -> dict | None:
    """Most recent ``run_end`` event, read from the file tail only."""
    if not path.exists():
        return None
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - tail_bytes))
        lines = f.read().splitlines()
    for line in reversed(lines):
        if b'"run_end"' in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Reuse or Replan?")
logs_dir = Path(st.sidebar.text_input("Logs directory", str(ROOT / "logs"), key="logs_dir"))
EVENTS, RUNS, ACTIONS = logs_dir / "events.jsonl", logs_dir / "runs.jsonl", logs_dir / "simulated_actions.jsonl"

runs_df = load_runs(str(RUNS), mtime(RUNS))
run_ids = list(dict.fromkeys(runs_df["run_id"].tolist()))[::-1] if not runs_df.empty else []
def short_model(model: str | None) -> str:
    """'qwen2.5:7b-instruct-q4_K_M' -> 'qwen2.5:7b'; names without a tag are kept as they are."""
    if not model:
        return "default"
    name, sep, tag = str(model).partition(":")
    return f"{name}:{tag.split('-')[0]}" if sep else name


def run_labels(frame: pd.DataFrame) -> dict[str, str]:
    """run_id -> 'YYYY-MM-DD HH:MM · provider / model · source · N eps · label' (display only)."""
    labels = {}
    if frame.empty:
        return labels
    for rid, g in frame.groupby("run_id", sort=False):
        first = g.iloc[0]
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(g["ts"].min())))
        label = first.get("label") if isinstance(first.get("label"), str) else ""
        parts = [when, f"{first.get('provider')} / {short_model(first.get('model'))}", str(first.get("source")),
                 f"{len(g)} eps"] + (["paced"] if first.get("paced") == True else []) + ([label] if label else [])
        labels[rid] = " · ".join(parts)
    return labels


RUN_LABELS = run_labels(runs_df)
_active = active_run(logs_dir)
ACTIVE_RUN_ID = (_active or {}).get("run_id")
if ACTIVE_RUN_ID:
    if ACTIVE_RUN_ID not in run_ids:  # no finished episode yet
        run_ids.insert(0, ACTIVE_RUN_ID)
        RUN_LABELS[ACTIVE_RUN_ID] = f"{ACTIVE_RUN_ID} · running"
    if st.session_state.get("auto_selected") != ACTIVE_RUN_ID:  # show a new run once; the user may then pick another
        st.session_state["auto_selected"] = st.session_state["run"] = ACTIVE_RUN_ID
sel_run = st.sidebar.selectbox("Run", run_ids or ["(no runs yet)"], key="run",
                               format_func=lambda r, L=RUN_LABELS: L.get(r, r))
if run_ids:
    st.sidebar.caption(f"Run ID `{sel_run}`")
df = runs_df[runs_df["run_id"] == sel_run].copy() if run_ids and not runs_df.empty else pd.DataFrame()


@st.cache_data(ttl=5, show_spinner=False)
def _run_start_config(events_path: str, mtime_: float, run_id: str) -> dict:
    p = Path(events_path)
    if p.exists():
        with p.open("rb") as f:
            for line in f:
                if b'"run_start"' in line and run_id.encode() in line:
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("run_id") == run_id:
                        return ev.get("config") or {}
    return {}


def run_start_config(run_id: str) -> dict:
    """The run's full RunConfig as logged in its run_start event ({} if unavailable)."""
    return _run_start_config(str(EVENTS), mtime(EVENTS), run_id)


def run_cache_dir(run_id: str) -> Path:
    """Plan-cache dir the run used (from its run_start event), defaulting to ./cache."""
    return Path(run_start_config(run_id).get("cache_dir") or ROOT / "cache")


@st.dialog("Delete run", icon=":material/delete:")
def confirm_delete(run_id: str):
    rows = runs_df[runs_df["run_id"] == run_id]
    meta = rows.iloc[0] if len(rows) else {}
    st.markdown(f"Permanently delete run `{run_id}`?")
    st.markdown(f"- {len(rows)} episodes · provider `{meta.get('provider', '?')}` · "
                f"tool source `{meta.get('source', '?')}`\n"
                "- Its lines in `runs.jsonl`, `events.jsonl` and `simulated_actions.jsonl`, "
                "and its plan-cache files\n- The shared replay cache is kept")
    st.caption("This cannot be undone.")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Cancel", key="delete_cancel"):
            st.rerun()
        if st.button("Delete", type="primary", icon=":material/delete:", key="delete_confirm"):
            try:
                counts = delete_run(logs_dir, run_id, run_cache_dir(run_id))
            except RunActiveError as e:
                st.error(str(e))
                return
            st.session_state["flash"] = (f"Deleted run `{run_id}`: {counts['runs.jsonl']} episodes, "
                                         f"{counts['events.jsonl']} events, "
                                         f"{counts['simulated_actions.jsonl']} actions, "
                                         f"{counts['plan_cache_files']} plan-cache files.")
            st.session_state.pop("ev", None)  # live panel re-reads the rewritten events file
            load_runs.clear()
            load_actions.clear()
            st.rerun()


def run_is_active() -> bool:
    """A runner holds the PID file, or a child we just launched has not written it yet."""
    launching = st.session_state.get("proc")
    return active_run(logs_dir) is not None or (launching is not None and launching.poll() is None)


RUN_ACTIVE = run_is_active()
POLL = 1 if RUN_ACTIVE else None  # fragments auto-refresh only while a run is active
run_busy = RUN_ACTIVE
if run_ids and st.sidebar.button("Delete run", icon=":material/delete:", disabled=run_busy, key="delete_run",
                                 help="Stop the active run first." if run_busy
                                 else "Delete the selected run and its logs."):
    confirm_delete(sel_run)
if msg := st.session_state.pop("flash", None):
    st.sidebar.success(msg, icon=":material/check:")


def active_dataset() -> tuple[str, Path, dict]:
    """The dataset the next run will use (falls back to the default if a saved copy was removed)."""
    available = dsets.list_datasets()
    name = st.session_state.get("dataset")
    if name not in available:
        name = dsets.DEFAULT_DATASET.stem
    return name, available[name], load_tasks(str(available[name]), mtime(available[name]))


@st.cache_data(show_spinner=False, max_entries=20)
def load_tasks(path: str, mtime_: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


with st.sidebar.expander("Launch benchmark run", expanded=not run_ids):
    @st.fragment(run_every=POLL)
    def run_controls(cmd: list[str], can_start: bool):
        # Run state comes from the runner's PID file, so it survives browser refreshes and also
        # picks up runs started from the CLI. The Popen handle only covers the ~1 s before the
        # child writes its PID file.
        info = active_run(logs_dir)
        launching = st.session_state.get("proc")
        if launching is not None and launching.poll() is not None:
            st.session_state["proc"] = launching = None  # reap the exited child
        running = info is not None or launching is not None
        if RUN_ACTIVE and not running:
            st.rerun()  # the run just ended: full rerun shows the final state and stops all timers
        if st.button("Start run", type="primary", icon=":material/play_arrow:",
                     disabled=running or not can_start):
            (logs_dir / STOP_FILE).unlink(missing_ok=True)
            log = (logs_dir / "batch_runner.out").open("w", encoding="utf-8")
            # Session-only settings reach the runner through its environment, never the command line.
            env = {**os.environ, **st.session_state.get("session_settings", {})}
            st.session_state["proc"] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            st.session_state["force_stopped"] = None
            st.rerun()
        with st.container(horizontal=True):
            if st.button("Stop", icon=":material/stop:", disabled=info is None,
                         help="Finish the current episode, then end the run cleanly."):
                request_stop(logs_dir)
                st.session_state["stop_requested"] = info["run_id"]
                st.rerun()
            if st.button("Force stop", icon=":material/dangerous:", disabled=info is None,
                         help="Kill the runner immediately. The current episode is not recorded."):
                if force_stop(logs_dir):
                    st.session_state["force_stopped"] = info["run_id"]
                else:
                    st.error("Could not stop the runner process.")
                st.rerun()
        if info is not None:
            stopping = ((logs_dir / STOP_FILE).exists()
                        or st.session_state.get("stop_requested") == info["run_id"])
            st.caption(f"Run `{info['run_id']}` (PID {info['pid']}): "
                       + ("stopping after the current episode..." if stopping else "running"))
        elif launching is not None:
            st.caption(f"Starting (PID {launching.pid})...")
        elif st.session_state.get("force_stopped"):
            st.caption(f"Run `{st.session_state['force_stopped']}`: force-stopped")
        else:
            end = last_run_end(EVENTS)
            if end:
                st.caption(f"Last run `{end.get('run_id')}`: "
                           f"{'stopped' if end.get('stopped') else 'finished'}"
                           + (f" after {end['episodes']} episodes" if "episodes" in end else ""))

    @st.fragment  # changing a launch setting reruns only this form, not the whole app
    def launch_form():
        ds_name, ds_path, ds_data = active_dataset()
        n_tasks = len(ds_data["tasks"])
        st.caption(f"Dataset: `{ds_name}` ({n_tasks} tasks). Change it in the Dataset tab.")
        provider = st.selectbox("Provider", ["mock", "ollama", "openai", "anthropic", "deepseek"])
        if provider != "mock":
            conf = settings.effective(st.session_state.get("session_settings"))
            key_name = f"{provider.upper()}_API_KEY"
            missing = key_name in conf and not conf[key_name][0]
            st.caption(f"Model: `{conf[f'{provider.upper()}_MODEL'][0]}`"
                       + (f". `{key_name}` is not set (Settings tab)." if missing else ""))
        st.caption("Tools: FastMCP simulators, calibrated latency", help=(
            "Synthetic data served by FastMCP. Delays come from a one-time calibration of real MCP servers "
            "(assumed for experiences and weather) and are added after each episode."))
        pace = st.toggle("Real-time tool delays", key="launch_pace", help=(
            "Makes simulated tools wait for their modelled delay, so Live DAG shows realistic timing. For demos "
            "and small runs. Results and modelled times do not change."))
        policies = st.multiselect("Policies", POLICIES, default=POLICIES)
        perts = st.multiselect("Perturbations", PERTS, default=PERTS)
        limit = st.number_input("Tasks limit (0 = all)", 0, n_tasks, 0, key=f"limit_{ds_name}_{n_tasks}")
        ep_budget = st.number_input("Episode token budget (0 = off)", 0, 200000, 12000, step=500)
        run_budget = st.number_input("Run token budget (0 = off)", 0, 10_000_000, 0, step=1000)
        run_label = st.text_input("Description (optional)", max_chars=80, placeholder="e.g. qwen 7b, tight budget")
        cmd = [sys.executable, "-m", "engine.batch_runner", "--provider", provider, "--source", "sim",
               "--policies", ",".join(policies), "--perturbations", ",".join(perts),
               "--episode-token-budget", str(ep_budget), "--run-token-budget", str(run_budget),
               "--logs-dir", str(logs_dir)] \
            + (["--tasks", str(ds_path)] if ds_path != dsets.DEFAULT_DATASET else []) + (["--tasks-limit", str(limit)] if limit else []) \
            + (["--label", run_label.strip()] if run_label.strip() else []) + (["--pace"] if pace else [])
        if pace:
            n_eps = (limit or n_tasks) * len(policies) * len(perts)
            st.caption(f"Up to {n_eps:,} episodes; waiting adds roughly {max(1, round(n_eps * 14 / 60)):,} minutes "
                       "(about 14 s per episode; more if plans are repaired).")
        run_controls(cmd, bool(policies and perts))

    launch_form()

# ----------------------------------------------------------------------------- helpers
def style_ax(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def has_latency_model(frame: pd.DataFrame) -> bool:
    return "modelled_tool_ms" in frame and bool(frame["modelled_tool_ms"].notna().any())


def latency_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean seconds per episode: measured LLM and local time, modelled tool time, and totals at 0x-2x."""
    f = frame[frame["modelled_tool_ms"].notna()]
    s = pd.DataFrame({"policy": f["policy"], "LLM (measured)": f["plan_latency_ms"] / 1000,
                      "local (measured)": (f["latency_ms"] - f["plan_latency_ms"]).clip(lower=0) / 1000,
                      "tools (modelled)": f["modelled_tool_ms"].astype(float) / 1000})
    for k in (0, 0.5, 1, 2):
        s[f"total, tools ×{k:g}"] = s["LLM (measured)"] + s["local (measured)"] + k * s["tools (modelled)"]
    return s.groupby("policy").mean().reindex([p for p in POLICIES if p in set(f["policy"])])


def new_fig(w=6.4, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def grouped_bars(ax, table: pd.DataFrame, fmt: str | None = None):
    cols = [p for p in PERTS if p in table.columns]
    pols = [p for p in POLICIES if p in table.index]
    width = 0.8 / max(len(pols), 1)
    gap = width * 0.12
    for i, pol in enumerate(pols):
        xs = [j + (i - (len(pols) - 1) / 2) * width for j in range(len(cols))]
        ax.bar(xs, table.loc[pol, cols].values, width - gap, color=POLICY_COLOR[pol], label=pol,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(len(cols)), cols)
    ax.legend(frameon=False, fontsize=8, ncols=len(pols), loc="upper left", bbox_to_anchor=(0, -0.12),
              labelcolor=INK2)


def fig_download(fig, name: str, key: str):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=SURFACE)
    st.download_button(f"Download {name}.png", buf.getvalue(), f"{name}.png", "image/png", key=key)


# ----------------------------------------------------------------------------- tabs
tab_live, tab_dag, tab_res, tab_plots, tab_act, tab_ep, tab_data, tab_set, tab_about = st.tabs(
    ["Live", "Live DAG", "Results", "Paper plots", "Actions", "Episode inspector", "Dataset", "Settings", "About"])

def _dismiss(key: str) -> None:
    st.session_state.setdefault("dismissed", set()).add(key)


def run_status_banner(run_id: str | None, run_events: list[dict], episodes: int) -> None:
    """Status of the latest run: starting / running / stopping / finished / stopped / force-stopped.

    Each (run, state) pair can be dismissed separately, so a dismissed "running" banner still
    lets the final "finished" or "force-stopped" banner appear.
    """
    if not run_id:
        return
    info = active_run(logs_dir)
    ended = next((e for e in reversed(run_events) if e.get("type") == "run_end"), None)
    n = f"{episodes} episode{'s' if episodes != 1 else ''}"
    if info is not None and info.get("run_id") == run_id:
        stopping = (logs_dir / STOP_FILE).exists() or st.session_state.get("stop_requested") == run_id
        state, alert, icon, title, body = (
            ("stopping", st.warning, ":material/hourglass_top:", "Stopping",
             f"Run `{run_id}` will stop after the current episode ({n} done).") if stopping else
            ("running", st.info, ":material/progress_activity:", "Running",
             f"Run `{run_id}` is in progress (PID {info['pid']}): {n} done."))
    elif ended is not None:
        if ended.get("stopped"):
            state, alert, icon, title, body = ("stopped", st.warning, ":material/stop_circle:", "Stopped",
                                               f"Run `{run_id}` was stopped after {n}.")
        else:
            state, alert, icon, title, body = ("finished", st.success, ":material/check_circle:", "Finished",
                                               f"Run `{run_id}` completed {n}.")
    elif st.session_state.get("proc") is not None and episodes == 0:
        state, alert, icon, title, body = ("starting", st.info, ":material/progress_activity:", "Starting",
                                           "Launching the batch runner...")
    elif st.session_state.get("force_stopped") == run_id:
        state, alert, icon, title, body = ("killed", st.error, ":material/dangerous:", "Force-stopped",
                                           f"Run `{run_id}` was killed after {n}. "
                                           "The episode in progress was not recorded.")
    else:  # no run_end and no live process: killed elsewhere or crashed
        state, alert, icon, title, body = ("aborted", st.error, ":material/error:", "Ended unexpectedly",
                                           f"Run `{run_id}` stopped after {n} without finishing "
                                           "(force-stopped or crashed). See `logs/batch_runner.out`.")
    key = f"{run_id}:{state}"
    if key in st.session_state.get("dismissed", set()):
        return
    with st.container(horizontal=True, vertical_alignment="center"):
        alert(body, icon=icon, title=title)
        st.button("", key=f"dismiss_{key}", icon=":material/close:", type="tertiary",
                  on_click=_dismiss, args=(key,))


def live_run_header(run_id: str | None, run_events: list[dict]) -> None:
    """Description, provider/model, requested tool source and the ACTUAL source split of tool calls."""
    if not run_id:
        return
    start = next((e for e in run_events if e.get("type") == "run_start"), None)
    cfg = (start or {}).get("config") or {}
    rec = runs_df[runs_df["run_id"] == run_id].iloc[0] if (not runs_df.empty and run_id in RUN_LABELS) else {}
    llm = next((e for e in run_events if e.get("type") == "llm_call_start"), {})
    provider = cfg.get("provider") or rec.get("provider", "?")
    model = cfg.get("model") or llm.get("model") or rec.get("model")
    source = cfg.get("source") or rec.get("source", "?")
    label = cfg.get("label") or (rec.get("label") if isinstance(rec.get("label"), str) else "")
    counts: dict[str, int] = {}
    for e in run_events:
        if e.get("type") == "step_end" and e.get("kind") == "tool":
            counts[e.get("source", "unknown")] = counts.get(e.get("source", "unknown"), 0) + 1
    total = sum(counts.values())
    split = " · ".join(f"{k} {v / total:.0%}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))         if total else "no tool calls yet"
    st.subheader(label or f"Run {run_id}")
    st.caption(f"{'Run `' + run_id + '` · ' if label else ''}`{provider}` / `{model or 'default'}` · "
               f"requested source `{source}` · tool data: {split}")


def selected_events() -> list[dict]:
    """Events of the selected run, read incrementally (large fields are dropped to save memory)."""
    key = (str(EVENTS), sel_run)
    state = st.session_state.get("ev")
    if not state or state.get("key") != key or (EVENTS.exists() and EVENTS.stat().st_size < state["offset"]):
        state = st.session_state["ev"] = {"key": key, "offset": 0, "rows": []}
    new, state["offset"] = read_jsonl(EVENTS, state["offset"], run_id=sel_run)
    state["rows"] += [{k: v for k, v in r.items() if k not in ("result_preview", "args")} for r in new]
    return state["rows"]


with tab_live:
    @st.fragment(run_every=POLL)
    def live_panel():
        info = active_run(logs_dir)
        if info is not None and st.session_state.get("auto_selected") != info.get("run_id"):
            st.rerun()  # a run just started: a full rerun selects it
        rows = selected_events()
        if not rows:
            st.info("No events for this run yet. Start a run from the sidebar or with `python -m engine.batch_runner`.")
            return
        ends = [r for r in rows if r.get("type") == "episode_end"]
        live_run_header(sel_run, rows)
        c = st.columns(4) # st.columns(5)
        # c[0].metric("Run", last_run or "-")
        c[0].metric("Episodes done", len(ends))
        c[1].metric("Success", f"{(sum(e['success'] for e in ends) / len(ends) * 100):.0f}%" if ends else "-")
        c[2].metric("LLM tokens", f"{sum(e.get('total_tokens', 0) for e in ends):,}")
        c[3].metric("Tool calls", sum(1 for r in rows if r.get("type") == "step_end" and r.get("kind") == "tool"))
        run_status_banner(sel_run, rows, len(ends))
        view = pd.DataFrame(rows[-40:][::-1])
        # llm_call_end carries its own "tokens", episode_end the episode's "total_tokens".
        view["tokens"] = pd.to_numeric(view.reindex(columns=["tokens"])["tokens"].combine_first(
            view.reindex(columns=["total_tokens"])["total_tokens"])).astype("Int64")
        cols = [k for k in ("type", "policy", "task_id", "attempt", "step_id", "tool", "status", "source",
                            "tokens", "latency_ms", "paced", "decision", "ok") if k in view.columns]
        st.dataframe(view[cols], hide_index=True, width="stretch", height=520, column_config={
            "tokens": st.column_config.NumberColumn("tokens", format="localized", help=(
                "LLM tokens. On llm_call_end: that planning call (with any parse retry). On episode_end: "
                "the episode total, which already includes those calls, so do not sum this column."))})

    live_panel()

with tab_dag:
    follow = st.toggle("Follow the latest episode of the selected run", value=True)

    def show_episode(p: dag_view.Pipeline, trace: list[dict], edges: list | None = None) -> None:
        st.image(dag_view.pipeline_svg(p), width=1000, alt=p.summary)
        if trace:
            st.graphviz_chart(dag_view.step_graph(trace, edges), width="content",
                              alt=f"Step graph of the executed plan, {len(trace)} steps")
            st.caption(dag_view.LEGEND + (" Tool times include the simulated real-time delay."
                                          if any(t.get("paced") for t in trace) else ""))
        elif p.done:
            st.info("No valid plan was executed in this episode.")
        else:
            st.info("Waiting for a plan. Steps appear here once one is available.")

    @st.fragment(run_every=POLL)
    def live_dag():
        rows = selected_events()
        eps = [r for r in rows if r.get("type") == "episode_start"]
        if not eps:
            st.info("No episodes in this run yet.")
            return
        ep = eps[-1]
        evs = [r for r in rows if r.get("episode_id") == ep["episode_id"]]
        att = dag_view.last_attempt(evs)
        steps: dict[str, dict] = {}
        for r in evs:
            if r.get("attempt") == att and r.get("type", "").startswith("step_"):
                steps[r["step_id"]] = {**steps.get(r["step_id"], {}), **r,
                                       "status": r.get("status", "running")}
        st.caption(f"{ep['policy']} · {ep['task_id']} · attempt {att or '-'}")
        show_episode(dag_view.pipeline(evs, RUN_ACTIVE and sel_run == ACTIVE_RUN_ID), list(steps.values()))

    if follow:
        live_dag()
    elif df.empty:
        st.info("No completed episodes for this run.")
    else:
        dag_labels = episode_labels(df, ["policy", "task_id", "decision_path"])
        eid = st.selectbox("Episode", df["episode_id"], format_func=lambda e, L=dag_labels: L.get(e, str(e)))
        rec = df[df.episode_id == eid].iloc[0].to_dict()
        trace = [{**t, "paced": rec.get("paced") == True} for t in rec["dag"]["trace"]]
        show_episode(dag_view.pipeline(dag_view.events_from_record(rec), False), trace, rec["dag"]["edges"])

with tab_res:
    if df.empty:
        st.info("No completed episodes for this run.")
    else:
        summary = df.groupby("policy").agg(
            episodes=("success", "size"), success_rate=("success", "mean"),
            mean_tokens=("total_tokens", "mean"), llm_calls=("llm_calls", "sum"),
            cost_usd=("cost_usd", "sum"), mean_latency_ms=("latency_ms", "mean"),
            reuse_rate=("reuse_attempted", "mean")).reindex([p for p in POLICIES if p in set(df.policy)])
        st.subheader("Policy summary")
        st.dataframe(summary.style.format({"success_rate": "{:.0%}", "reuse_rate": "{:.0%}",
                                           "mean_tokens": "{:,.0f}", "cost_usd": "${:.3f}",
                                           "mean_latency_ms": "{:.0f}"}), width="stretch")
        a, b = st.columns(2)
        a.subheader("Success by perturbation")
        a.dataframe(df.pivot_table(index="policy", columns="perturbation", values="success", aggfunc="mean")
                    .style.format("{:.0%}"), width="stretch")
        b.subheader("Mean tokens by perturbation")
        b.dataframe(df.pivot_table(index="policy", columns="perturbation", values="total_tokens",
                                   aggfunc="mean").style.format("{:,.0f}"), width="stretch")
        if has_latency_model(df):
            st.subheader("Latency with modelled tool time")
            st.dataframe(latency_table(df).style.format("{:.1f}"), width="stretch",
                         alt="Mean episode latency per policy, with modelled tool time at four scales")
            ids = ", ".join(f"`{i}`" for i in sorted(df["latency_profile_id"].dropna().unique()))
            st.caption("Mean seconds per episode. LLM and local time are **measured**; tool time is **modelled** from "
                       f"frozen calibration profile {ids}, not measured in this run. The total columns multiply the "
                       "modelled tool time by 0, 0.5, 1 and 2: a policy ranking that holds across them does not "
                       "depend on the calibration.")
            if "paced" in df and (paced := df[df["paced"] == True]).shape[0]:
                st.caption(f"Real-time delays were on: episodes waited {paced['paced_wait_ms'].mean() / 1000:.1f} s on "
                           f"average, against {paced['modelled_tool_ms'].mean() / 1000:.1f} s modelled. The wait is "
                           "**simulated**, is left out of every measured time, and is not added again here.")
        st.subheader("Decision paths")
        st.dataframe(df.groupby(["policy", "decision_path"]).agg(n=("success", "size"),
                                                                 success=("success", "mean"))
                     .reset_index(), hide_index=True, width="stretch")
        st.subheader("Episodes")
        show = ["policy", "task_id", "perturbation_detail", "cache_status", "decision_path", "success",
                "total_tokens", "latency_ms", "total_cost", "budget"]
        st.dataframe(df[show], hide_index=True, width="stretch")

with tab_plots:
    if df.empty:
        st.info("No data for plots.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = new_fig()
            t = df.pivot_table(index="policy", columns="perturbation", values="success", aggfunc="mean") * 100
            grouped_bars(ax, t)
            style_ax(ax, "Task success rate by perturbation", "success (%)")
            ax.set_ylim(0, 105)
            st.pyplot(fig)
            fig_download(fig, "success_by_perturbation", "d1")
            plt.close(fig)
        with c2:
            fig, ax = new_fig()
            t = df.pivot_table(index="policy", columns="perturbation", values="total_tokens", aggfunc="mean")
            grouped_bars(ax, t)
            style_ax(ax, "LLM tokens per episode", "mean tokens")
            st.pyplot(fig)
            fig_download(fig, "tokens_by_perturbation", "d2")
            plt.close(fig)
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = new_fig()
            g = df.groupby("policy").agg(cost=("cost_usd", "sum"), succ=("success", "mean"))
            for pol in [p for p in POLICIES if p in g.index]:
                ax.scatter(g.loc[pol, "cost"], g.loc[pol, "succ"] * 100, s=90, color=POLICY_COLOR[pol],
                           edgecolors=SURFACE, linewidths=2, label=pol, zorder=3)
                ax.annotate(pol, (g.loc[pol, "cost"], g.loc[pol, "succ"] * 100), xytext=(8, 4),
                            textcoords="offset points", fontsize=8, color=INK2)
            style_ax(ax, "Cost vs. success (Pareto view)", "success (%)")
            ax.set_xlabel("total planning cost (USD, estimated)", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
            st.pyplot(fig)
            fig_download(fig, "cost_vs_success", "d3")
            plt.close(fig)
        with c4:
            fig, ax = new_fig()
            ct = (df.groupby(["policy", "cache_status"]).size().unstack(fill_value=0)
                  .reindex([p for p in POLICIES if p in set(df.policy)]))
            bottom = pd.Series(0, index=ct.index)
            for s in [s for s in STATUS_ORDER if s in ct.columns]:
                ax.bar(ct.index, ct[s], 0.5, bottom=bottom, color=STATUS_COLOR[s], label=s,
                       edgecolor=SURFACE, linewidth=2)
                bottom += ct[s]
            style_ax(ax, "Cache lookup outcomes", "episodes")
            ax.legend(frameon=False, fontsize=8, ncols=4, loc="upper left", bbox_to_anchor=(0, -0.12),
                      labelcolor=INK2)
            st.pyplot(fig)
            fig_download(fig, "cache_outcomes", "d4")
            plt.close(fig)
        fig, ax = new_fig(12.8, 3.2)
        lat = df.groupby("policy")[["plan_latency_ms", "exec_latency_ms"]].mean().reindex(
            [p for p in POLICIES if p in set(df.policy)])
        y = range(len(lat))
        ax.barh(list(y), lat["plan_latency_ms"], 0.5, color="#6250d6", label="LLM planning", edgecolor=SURFACE,
                linewidth=2)
        ax.barh(list(y), lat["exec_latency_ms"], 0.5, left=lat["plan_latency_ms"], color="#e87ba4",
                label="CPU execution + tools", edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(list(y), lat.index)
        style_ax(ax, "Mean episode latency breakdown", "")
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_xlabel("ms", color=INK2, fontsize=9)
        ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper left", bbox_to_anchor=(0, -0.22),
                  labelcolor=INK2)
        st.pyplot(fig)
        if set(df["provider"]) == {"mock"}:
            st.caption("Provider is `mock`: planning latency is near zero and not representative of a real LLM. "
                       "Token and cost axes use estimated counts at a reference API price.")
        fig_download(fig, "latency_breakdown", "d5")
        plt.close(fig)
        if has_latency_model(df):
            t = latency_table(df)
            fig, ax = new_fig(12.8, 3.2)
            left = pd.Series(0.0, index=t.index)
            for col, color in (("LLM (measured)", "#6250d6"), ("local (measured)", "#e87ba4"),
                               ("tools (modelled)", "#b9b8b1")):
                ax.barh(list(range(len(t))), t[col], 0.5, left=left, color=color, label=col, edgecolor=SURFACE,
                        linewidth=2, hatch="///" if "modelled" in col else None)
                left += t[col]
            ax.set_yticks(list(range(len(t))), t.index)
            style_ax(ax, "Mean episode latency with modelled tool time", "")
            ax.yaxis.grid(False)
            ax.xaxis.grid(True, color=GRID, linewidth=0.6)
            ax.set_xlabel("seconds", color=INK2, fontsize=9)
            ax.invert_yaxis()
            ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper left", bbox_to_anchor=(0, -0.22), labelcolor=INK2)
            st.pyplot(fig)
            st.caption("Hatched: modelled from the frozen calibration of real MCP servers, not measured in this run.")
            fig_download(fig, "latency_modelled", "d6")
            plt.close(fig)
        with st.expander("Table view of plotted data"):
            st.dataframe(df.groupby(["policy", "perturbation"]).agg(
                success=("success", "mean"), tokens=("total_tokens", "mean"), cost=("cost_usd", "sum"),
                plan_ms=("plan_latency_ms", "mean"), exec_ms=("exec_latency_ms", "mean")).reset_index(),
                hide_index=True, width="stretch")

with tab_act:
    acts = load_actions(str(ACTIONS), mtime(ACTIONS))
    if acts.empty:
        st.info("No simulated actions yet (they are written only after a passing validation).")
    else:
        only_run = st.checkbox("Only selected run", value=bool(run_ids))
        if only_run and "run_id" in acts:
            acts = acts[acts["run_id"] == sel_run]
        payload = pd.json_normalize(acts["payload"].tolist()) if len(acts) else pd.DataFrame()
        full = pd.concat([acts.drop(columns=["payload"]).reset_index(drop=True), payload], axis=1)
        k = st.columns(3)
        for col, (kind, label) in zip(k, [("booking", "Bookings"), ("calendar_event", "Calendar events"),
                                         ("email", "Emails")]):
            col.metric(label, int((full["kind"] == kind).sum()))
        s1, s2, s3 = st.tabs(["Bookings", "Calendar", "Emails"])
        base_cols = ["confirmation", "policy", "task_id", "perturbation"]
        for tab, kind, cols in ((s1, "booking", ["hotel_name", "city", "checkin", "checkout", "guests",
                                                  "guest_name", "total_price"]),
                                (s2, "calendar_event", ["title", "start_date", "end_date", "location",
                                                        "description"]),
                                (s3, "email", ["to", "subject", "body"])):
            sub = full[full["kind"] == kind]
            tab.dataframe(sub[[c for c in base_cols + cols if c in sub.columns]], hide_index=True,
                          width="stretch")
        st.caption("All actions are simulated by engine/local_action_mcp.py; nothing is booked or sent.")

with tab_ep:
    if df.empty:
        st.info("No episodes.")
    else:
        insp_labels = episode_labels(df, ["policy", "task_id", "cache_status", "decision_path"])
        eid = st.selectbox("Episode ", df["episode_id"], key="insp",
                           format_func=lambda e, L=insp_labels: L.get(e, str(e)))
        rec = df[df.episode_id == eid].iloc[0].to_dict()
        ok = rec["success"]
        st.markdown(f"**{'✅ Success' if ok else '❌ Failed'}** · {rec['perturbation_detail']} · "
                    f"tokens {rec['total_tokens']:,} · cost ${rec['cost_usd']:.4f} · {rec['latency_ms']:.0f} ms")
        if rec.get("violations"):
            st.warning("\n".join(f"- {v}" for v in rec["violations"]))
        a, b = st.columns(2)
        a.markdown("**Output**")
        a.json(rec.get("output") or {}, expanded=False)
        b.markdown("**Attempts**")
        b.json(rec.get("attempts") or [], expanded=False)
        st.markdown("**Executed plan**")
        st.json(rec.get("plan") or {}, expanded=False)


# ----------------------------------------------------------------------------- dataset
with tab_data:
    @st.fragment  # table edits rerun only this panel
    def dataset_panel():
        available = dsets.list_datasets()
        if pending := st.session_state.pop("ds_view_next", None):
            st.session_state["ds_view"] = pending
        if st.session_state.get("ds_view") not in available:
            st.session_state["ds_view"] = active_dataset()[0]
        view = st.selectbox("Dataset", list(available), key="ds_view",
                            format_func=lambda n: n + ("" if n in dsets.BUILTIN else " (saved copy)"))
        data = load_tasks(str(available[view]), mtime(available[view]))
        in_use = active_dataset()[0]
        st.caption(f"{len(data['tasks'])} tasks · {len(data['cities'])} destinations · content hash "
                   f"`{dsets.content_hash(data)}` · prices and budgets are synthetic"
                   + (" · used for the next run" if view == in_use else ""))
        if msg := st.session_state.pop("ds_flash", None):
            st.success(msg, icon=":material/check:")
        if RUN_ACTIVE:
            st.info("Editing is locked while a run is active.", icon=":material/lock:")
        airports = dsets.origins()
        frame = pd.DataFrame(dsets.to_rows(data), columns=list(dsets.COLUMNS))
        frame["max_stops"] = frame["max_stops"].astype("Int64")
        frame["min_hotel_rating"] = frame["min_hotel_rating"].astype("Float64")
        cc = st.column_config
        edited = st.data_editor(
            frame, hide_index=True, width="stretch", height=460, num_rows="fixed", disabled=RUN_ACTIVE or
            [c for c in dsets.COLUMNS if c not in dsets.EDITABLE],
            key=f"ds_editor_{view}_{dsets.content_hash(data)}_{st.session_state.get('ds_version', 0)}",
            alt="Editable task table", column_config={
                "id": cc.TextColumn("Task", pinned=True),
                "origin": cc.SelectboxColumn("Origin", options=list(airports), required=True),
                "destination": cc.SelectboxColumn("Destination", options=sorted(data["cities"]), required=True,
                                                  help="Airport, ferry port and coordinates follow the city."),
                "depart_date": cc.DateColumn("Depart", required=True),
                "return_date": cc.DateColumn("Return", required=True),
                "travelers": cc.NumberColumn("Travellers", min_value=1, max_value=dsets.MAX_TRAVELLERS, step=1,
                                             required=True),
                "budget": cc.NumberColumn("Budget (GBP)", min_value=1.0, required=True, format="%.0f"),
                "max_stops": cc.SelectboxColumn("Max stops", options=[0, 1, 2], help="Blank = no limit"),
                "min_hotel_rating": cc.NumberColumn("Min rating", min_value=1.0, max_value=5.0, step=0.1,
                                                    format="%.1f", help="Blank = no limit"),
                "budget_check": cc.TextColumn("Budget check", help=(
                    "'check budget': the trip changed after its budget was set, so it may be too low or too high.")),
                "description": cc.TextColumn("Description", width="large")})
        new, errors, changed = dsets.apply_rows(data, edited.to_dict("records"), airports)
        if changed:
            st.caption(f"Unsaved changes in {len(changed)} task{'s' if len(changed) != 1 else ''}: "
                       + ", ".join(changed[:12]) + ("..." if len(changed) > 12 else ""))
        for e in errors[:8]:
            st.error(e, icon=":material/error:")
        with st.container(horizontal=True, vertical_alignment="bottom"):
            name = st.text_input("Save as", key=f"ds_name_{st.session_state.get('ds_version', 0)}", max_chars=40,
                                 placeholder="e.g. summer_trips", disabled=RUN_ACTIVE)
            if st.button("Save as copy", icon=":material/save:",
                         disabled=RUN_ACTIVE or not changed or bool(errors)):
                try:
                    path = dsets.save_copy(new, name, based_on=view)
                except (ValueError, OSError) as e:
                    st.error(str(e))
                else:
                    st.session_state["ds_view_next"] = path.stem
                    st.session_state["ds_version"] = st.session_state.get("ds_version", 0) + 1
                    st.session_state["ds_flash"] = f"Saved `{path.stem}`. Select **Use for next run** to run it."
                    st.rerun()
            if st.button("Use for next run", type="primary", icon=":material/check_circle:",
                         disabled=view == in_use or bool(changed),
                         help="Save or undo your changes first." if changed else None):
                st.session_state["dataset"] = view
                st.rerun()  # full rerun: the launch form shows the new dataset
        st.caption("The built-in datasets are never changed; edits are saved as copies in `data/custom/`. "
                   "Budgets are not recalculated when a trip changes.")

    dataset_panel()


# ----------------------------------------------------------------------------- settings
def is_local_browser() -> bool:
    """True only when the browser is on the machine running the app (fails closed)."""
    host = urlparse(st.context.url or "").hostname
    return st.context.ip_address in (None, "127.0.0.1", "::1") and host in ("localhost", "127.0.0.1", "::1")


def _open_settings(value: bool) -> None:
    st.session_state["set_open"] = value


with tab_set:
    @st.fragment
    def settings_panel():
        local = is_local_browser()
        session = st.session_state.setdefault("session_settings", {})
        conf = settings.effective(session)
        if msg := st.session_state.pop("set_flash", None):
            st.success(msg, icon=":material/check:")
        if RUN_ACTIVE:
            st.info("Settings are locked while a run is active.", icon=":material/lock:")
        left, right = st.columns(2, gap="large")
        with left:
            st.table(pd.DataFrame([{"setting": k, "value": ("set" if v else "not set") if k in settings.SECRETS else f"`{v}`"}
                                   for k, (v, _) in conf.items()]).set_index("setting"),
                     border="horizontal", alt="Current settings; keys are shown only as set or not set")
            st.caption("Priority: this session, then system environment, then `.env`, then the default. "
                       "The `mock` provider needs no settings.")

            with st.container(horizontal=True, vertical_alignment="bottom"):
                options = [g for g in settings.GROUPS if local or g != "ollama"]
                target = st.selectbox("Connection", options, key="set_target")
                if st.button("Test connection", icon=":material/network_check:",
                             help="Checks the key and model name. Uses no LLM tokens."):
                    with st.spinner("Checking..."):
                        ok, text = settings.check_connection(target, {k: v for k, (v, _) in conf.items()})
                    (st.success if ok else st.error)(f"{target}: {text}")
            st.caption("The Mapbox token is used only by the latency calibration.")

        with right:
            if not st.session_state.get("set_open"):
                st.button("Add or update", icon=":material/edit:", disabled=RUN_ACTIVE, key="set_open_btn",
                          help="Enter a key, model name or address.", on_click=_open_settings, args=(True,))
            if st.session_state.get("set_open") and not RUN_ACTIVE:
                group = st.selectbox("Provider to update", list(settings.GROUPS), key="set_group")
                with st.form(f"settings_form_{group}", clear_on_submit=True):
                    values = {}
                    for k in settings.GROUPS[group]:
                        values[k] = st.text_input(k, key=f"set_{k}", placeholder="leave blank to keep",
                                                  type="password" if k in settings.SECRETS else "default",
                                                  autocomplete="new-password" if k in settings.SECRETS else "off")
                    where = st.segmented_control(
                        "Apply to", ["This session only", "Save to .env"] if local else ["This session only"],
                        default="This session only", required=True,
                        help="Session values are kept in memory and are lost when the page is closed or reloaded.")
                    if not local:
                        st.caption("Saving to `.env` is available only in a browser on the computer running this app, "
                                   "because a saved key would be used by everyone.")
                    if st.form_submit_button("Apply", type="primary", icon=":material/check:"):
                        try:
                            given = {k: settings.clean(v) for k, v in values.items() if v and v.strip()}
                            if where == "Save to .env" and given:
                                settings.save_env_file(given)
                                for k in given:
                                    session.pop(k, None)
                                shadowed = [k for k in given if os.environ.get(k)]
                                st.session_state["set_flash"] = f"Saved {', '.join(given)} to .env." + (
                                    f" System environment variables still override: {', '.join(shadowed)}." if shadowed else "")
                            elif given:
                                session.update(given)
                                st.session_state["set_flash"] = f"Using {', '.join(given)} for this session."
                        except (ValueError, OSError) as e:
                            st.error(f"Not saved: {e}")
                        else:
                            if given:
                                st.session_state["set_open"] = False
                                st.rerun()  # full rerun: the launch form shows the model
                st.button("Cancel", icon=":material/close:", key="set_cancel", on_click=_open_settings, args=(False,))
            if session and st.button("Clear session settings", icon=":material/delete:", disabled=RUN_ACTIVE):
                session.clear()
                st.rerun()
    settings_panel()


# ----------------------------------------------------------------------------- about
ABOUT_MD = ROOT / "docs" / "ABOUT.md"
RUN_CARD_MARKER = "<!-- RUN_CARD -->"


def reproduce_command(cfg: dict) -> str:
    """CLI that re-runs a logged RunConfig (logs/cache dirs left at their defaults)."""
    parts = ["python -m engine.batch_runner", f"--provider {cfg.get('provider', 'mock')}",
             f"--source {cfg.get('source', 'sim')}", f"--seed {cfg.get('seed', 7)}"]
    if cfg.get("model"):
        parts.append(f"--model {cfg['model']}")
    for flag, key in (("--policies", "policies"), ("--perturbations", "perturbations"), ("--task-ids", "task_ids")):
        if cfg.get(key):
            parts.append(f"{flag} {','.join(cfg[key])}")
    parts.append(f"--episode-token-budget {cfg.get('episode_token_budget') or 0}")
    if cfg.get("run_token_budget"):
        parts.append(f"--run-token-budget {cfg['run_token_budget']}")
    if cfg.get("max_repairs", 1) != 1:
        parts.append(f"--max-repairs {cfg['max_repairs']}")
    if cfg.get("tasks_path"):
        p = Path(cfg["tasks_path"])
        if p.resolve() != dsets.DEFAULT_DATASET.resolve():
            shown = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
            parts.append(f'--tasks "{shown.as_posix()}"')
    if cfg.get("latency_profile"):
        parts.append(f'--latency-profile "{Path(cfg["latency_profile"]).as_posix()}"')
    if cfg.get("tasks_limit"):
        parts.append(f"--tasks-limit {cfg['tasks_limit']}")
    if cfg.get("pace"):
        parts.append("--pace")
    if not cfg.get("commit", True):
        parts.append("--no-commit")
    if cfg.get("label"):
        parts.append('--label "' + cfg["label"].replace('"', "'") + '"')
    return " ".join(parts)  # one line: runs unchanged in PowerShell and bash


def task_list(frame: pd.DataFrame) -> str:
    ids = sorted(frame["base_id"].unique())
    return f"{len(ids)} ({', '.join(ids)})" if len(ids) <= 12 else f"{len(ids)} ({ids[0]} to {ids[-1]})"


def run_card() -> None:
    with st.container(border=True):
        if df.empty:
            st.markdown("**Selected run:** none yet. Start one from the sidebar to see its details here.")
            return
        cfg = run_start_config(sel_run)
        first = df.iloc[0]
        label = first.get("label") if isinstance(first.get("label"), str) else ""
        st.markdown(f"**Selected run** · `{sel_run}`" + (f" · {label}" if label else ""))
        with st.container(horizontal=True):
            st.metric("Episodes", len(df), border=True)
            st.metric("Success", f"{df['success'].mean():.0%}", border=True)
            st.metric("LLM tokens", f"{int(df['total_tokens'].sum()):,}", border=True)
            paid = first.get("provider") not in UNBILLED
            st.metric("Est. cost", f"${df['cost_usd'].sum():.3f}" if paid else "–", border=True, help=(
                "Token prices are estimates." if paid else "No API charge: the mock planner and local models are not billed."))
        sources: dict[str, int] = {}
        for per_episode in df.get("tool_sources", pd.Series(dtype=object)):
            if isinstance(per_episode, dict):
                for src, n in per_episode.items():
                    sources[src] = sources.get(src, 0) + int(n)
        total = sum(sources.values())
        ranked = sorted(sources.items(), key=lambda kv: -kv[1])
        split = ", ".join(f"{k} {v / total:.0%}" for k, v in ranked) if total else "no tool calls"
        budget = cfg.get("episode_token_budget")
        rows = {
            "Planner": f"{first.get('provider')} / {first.get('model')}",
            "Tool source": f"requested {first.get('source')}; actual data: {split}",
            "Policies": ", ".join(cfg.get("policies") or sorted(df["policy"].unique())),
            "Perturbations": ", ".join(cfg.get("perturbations") or sorted(df["perturbation"].unique())),
            "Dataset": (f"{first['dataset']} (content hash {first['dataset_hash']})"
                        if isinstance(first.get("dataset"), str) else "not logged"),
            "Tasks": task_list(df),
            "Tool latency": (f"modelled, profile {first['latency_profile_id']}"
                             + (", with real-time simulated delays" if cfg.get("pace") else "")
                             if isinstance(first.get("latency_profile_id"), str) else "not modelled"),
            "Seed": cfg.get("seed", "not logged"),
            "Token budget": f"{budget:,} per episode" if budget else ("off" if cfg else "not logged"),
        }
        if cfg.get("run_token_budget"):
            rows["Token budget"] += f", {cfg['run_token_budget']:,} per run"
        st.table(pd.DataFrame({"value": [str(v) for v in rows.values()]}, index=list(rows.keys())),
                 border="horizontal", hide_header=True, alt="Configuration of the selected run")
        notes = []
        if first.get("provider") == "mock":
            notes.append("the planner is the `mock` heuristic, so token counts are estimates and latency is not "
                         "representative")
        if sources.get("replay_sim"):
            notes.append("replayed tool data was recorded from the simulator, not from live servers")
        if sources.get("sim_fallback"):
            notes.append(f"{sources['sim_fallback']} tool calls fell back to the simulator")
        if notes:
            st.caption("Note: " + "; ".join(notes) + ".")
        if cfg:
            st.markdown("Reproduce this run:")
            if not cfg.get("model") and first.get("provider") != "mock":
                cfg = {**cfg, "model": first.get("model")}  # the default may differ on another machine
            st.code(reproduce_command(cfg), language="text", wrap_lines=True)


with tab_about:
    try:
        about = ABOUT_MD.read_text(encoding="utf-8")
    except OSError:
        about = f"# About\n\n`{ABOUT_MD.relative_to(ROOT)}` is missing."
    before, marker, after = about.partition(RUN_CARD_MARKER)
    st.markdown(before)
    if marker:
        run_card()
    st.markdown(after)

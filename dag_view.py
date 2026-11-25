"""Drawing code for the Live DAG tab: episode pipeline (inline SVG) and step graph (Graphviz).

Pure functions over logged events: no Streamlit, no engine imports. Plans are written by an LLM, so every
string taken from them is escaped.
"""
from __future__ import annotations

import html
import time
from dataclasses import dataclass

import graphviz

# ----------------------------------------------------------------------------- episode pipeline
STAGES = ("task", "cache", "llm", "execute", "validate", "book")
TITLES = {"task": "Task", "cache": "Cache", "llm": "LLM plan", "execute": "Execute", "validate": "Validate",
          "book": "Book"}
# state -> (colour on the light card, status symbol, meaning for screen readers)
STATES = {"ok": ("#2e9d4f", "✓", "done"), "fail": ("#d8483f", "✗", "failed"), "active": ("#3b82d6", "…", "in progress"),
          "info": ("#c98a12", "!", "note"), "skip": ("#8c8b85", "–", "skipped"), "wait": ("#8c8b85", "", "not reached")}
ICONS = {  # 24x24 line icons
    "task": "M7 4h10a1 1 0 0 1 1 1v15a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zM9 9h6M9 13h6M9 17h4",
    "cache": "M5 6c0-4 14-4 14 0s-14 4-14 0zM5 6v12c0 4 14 4 14 0V6M5 12c0 4 14 4 14 0",
    "llm": "M12 3l2.2 6.3L20 12l-5.8 2.7L12 21l-2.2-6.3L4 12l5.8-2.7zM19 3v4M17 5h4",
    "execute": "M8 5l11 7-11 7z",
    "validate": "M12 3l8 3v6c0 5-3.5 8-8 9-4.500-1-8-4-8-9V6zM8.500 12l2.500 2.500 4.500-5",
    "book": "M4 8V6h16v2a4 4 0 0 0 0 8v2H4v-2a4 4 0 0 0 0-8zM14 6v12",
}


@dataclass
class Pipeline:
    nodes: dict[str, tuple[str, str]]  # stage -> (state, detail)
    reused: bool = False  # a cached plan was executed without the LLM
    loops: int = 0  # LLM calls made after a failed validation
    looping: bool = False  # such a call is running now
    done: bool = False

    @property
    def summary(self) -> str:
        parts = [f"{TITLES[s]}: {STATES[st][2]}" + (f" ({d})" if d else "") for s, (st, d) in self.nodes.items()]
        if self.reused:
            parts.append("cached plan reused")
        if self.loops:
            parts.append(f"re-planned {self.loops} time(s) after failed validation")
        return "; ".join(parts)


def last_attempt(evs: list[dict]) -> str | None:
    names = [r["attempt"] for r in evs if r.get("attempt") and r["attempt"] != "commit"]
    return names[-1] if names else None


def pipeline(evs: list[dict], running: bool, now: float | None = None) -> Pipeline:
    """Where an episode is, from its events. ``running``: the benchmark process is alive."""
    now = time.time() if now is None else now
    last = {r["type"]: r for r in evs}
    end = last.get("episode_end")
    live = running and end is None
    att = last_attempt(evs)
    mine = [r for r in evs if r.get("attempt") == att] if att else []
    decisions = {r.get("decision") for r in evs if r["type"] == "decision"}
    attempts = {r["attempt"] for r in evs if r.get("attempt")}
    reused = any(a.startswith("reuse") for a in attempts)
    n: dict[str, tuple[str, str]] = {s: ("wait", "") for s in STAGES}
    n["task"] = ("ok", "")

    if hit := last.get("cache_lookup"):
        status = hit["status"]
        n["cache"] = {"hit": ("ok", "hit"), "bypass": ("skip", "not used")}.get(status, ("info", status))
        if "skip_reuse" in decisions:
            n["cache"] = ("info", "hit, not trusted")
    elif live:
        n["cache"] = ("active", "")

    calls = [r for r in evs if r["type"] in ("llm_call_start", "llm_call_end")]
    starts = [r for r in calls if r["type"] == "llm_call_start"]
    ends = [r for r in calls if r["type"] == "llm_call_end"]
    thinking = len(starts) > len(ends)
    failed_seen, loops = False, 0
    for r in evs:
        if r["type"] == "validation":
            failed_seen = not r["ok"]
        elif r["type"] == "llm_call_start" and failed_seen:
            loops += 1
    if thinking:
        n["llm"] = ("active", f"{max(0, now - starts[-1]['ts']):.0f} s") if live else ("skip", "stopped")
    elif ends:
        secs = sum(c.get("latency_ms") or 0 for e in ends for c in e.get("calls", [])) / 1000
        made = sum(len(e.get("calls", [])) for e in ends)
        n["llm"] = ("ok" if ends[-1]["ok"] else "fail",
                    ("" if ends[-1]["ok"] else "invalid plan, ") + (f"{made} calls, " if made > 1 else "") + (f"{secs:.1f} s" if secs >= 0.05 else "<0.1 s"))
    elif "degrade_budget" in decisions:
        n["llm"] = ("skip", "no token budget")
    elif reused:
        n["llm"] = ("skip", "0 tokens")

    began = {r["step_id"] for r in mine if r["type"] == "step_start"}
    over = {r["step_id"] for r in mine if r["type"] in ("step_end", "step_skipped")}
    errors = [r for r in mine if r.get("status") == "error"]
    check = next((r for r in reversed(mine) if r["type"] == "validation"), None)
    if att:
        if check is None:
            n["execute"] = (("active" if live else "skip"),
                            f"{len(over)} of {len(began | over)} steps" if began else ("" if live else "stopped"))
        else:
            total = len(began | over)
            n["execute"] = ("fail", "tool error" if errors else "") if errors else ("ok", f"{total} step{'s' * (total != 1)}" if total else "")
            violations = check.get("violations") or []
            ran = not any(str(v).startswith("execution_error") for v in violations)
            if not ran and not errors:
                n["execute"] = ("fail", "plan error")
            count = len(violations)
            n["validate"] = (("ok", "passed") if check["ok"] else
                             ("fail", f"{count} violation{'s' * (count != 1)}") if ran else ("skip", "nothing to check"))

    booked = [r for r in evs if r.get("attempt") == "commit" and r["type"] != "step_start"]
    if end is not None:
        n["book"] = (("ok", "simulated") if booked else ("skip", "off")) if end["success"] else ("skip", "nothing booked")
    elif booked:
        n["book"] = ("active" if live else "skip", "")
    return Pipeline(n, reused, loops, thinking and live and failed_seen, end is not None)


def events_from_record(rec: dict) -> list[dict]:
    """Minimal events for a finished episode, rebuilt from its run record."""
    evs: list[dict] = [{"type": "cache_lookup", "status": rec["cache_status"]}]
    path = str(rec.get("decision_path", ""))
    if "skip_reuse" in path:
        evs.append({"type": "decision", "decision": "skip_reuse"})
    if "degrade_budget" in path:
        evs.append({"type": "decision", "decision": "degrade_budget"})
    attempts = rec.get("attempts") or []
    for i, a in enumerate(attempts, 1):
        label = f"{a['mode']}#{i}"
        ran = a.get("exec_ms") is not None
        if a["mode"] == "replan":
            evs += [{"type": "llm_call_start", "ts": 0.0}, {"type": "llm_call_end", "ok": ran, "calls": a.get("llm") or []}]
        if not ran:
            continue
        if i == len(attempts):  # the record keeps the trace of the final attempt only
            evs += [{"type": "step_start", "attempt": label, **s} for s in (rec.get("dag") or {}).get("trace", [])]
        evs.append({"type": "validation", "attempt": label, "ok": a["ok"], "violations": a["violations"]})
    if rec.get("actions"):
        evs.append({"type": "step_end", "attempt": "commit", "step_id": "commit"})
    evs.append({"type": "episode_end", "success": bool(rec["success"])})
    return evs


def pipeline_svg(p: Pipeline) -> str:
    """Episode pipeline as a standalone SVG for ``st.image`` (``st.html`` strips SVG). Alt text: ``p.summary``.

    Drawn on its own light card, so it is legible in either theme and in print."""
    xs = {s: 80 + 168 * i for i, s in enumerate(STAGES)}
    y, r, idle = 78, 27, "#8c8b85"
    reached = {s: st not in ("wait", "skip") for s, (st, _) in p.nodes.items()}
    out: list[str] = []

    def arrow(d: str, used: bool, color: str, label: str = "", lx: float = 0, ly: float = 0) -> None:
        dash = "" if used else ' stroke-dasharray="3 5" opacity="0.45"'
        head = "url(#pl-" + {"#2e9d4f": "ok", "#d8483f": "fail"}.get(color, "idle") + ")"
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{2.5 if used else 1.5}"{dash} '
                   f'marker-end="{head}"/>')
        if label and used:
            out.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" font-size="12" fill="currentColor">'
                       f"{html.escape(label)}</text>")

    called = p.nodes["llm"][0] in ("ok", "fail", "active") or p.nodes["llm"][1] == "stopped"
    used = {("task", "cache"): reached["cache"] or p.nodes["cache"][0] == "skip", ("cache", "llm"): called,
            ("llm", "execute"): called and reached["execute"] and not (p.reused and p.looping),
            ("execute", "validate"): reached["validate"], ("validate", "book"): reached["book"]}
    for (a, b), u in used.items():
        arrow(f"M{xs[a] + r + 5} {y}H{xs[b] - r - 9}", u, idle)
    # Reuse skips the LLM (above); a failed check goes back to it (below).
    arrow(f"M{xs['cache'] + 10} {y - r - 5}C{xs['cache'] + 60} 8 {xs['execute'] - 60} 8 {xs['execute'] - 10} {y - r - 8}",
          p.reused, "#2e9d4f" if p.reused else idle, "reuse: 0 tokens", xs["llm"], 14)
    arrow(f"M{xs['validate']} 150C{xs['validate'] - 40} 196 {xs['llm'] + 40} 196 {xs['llm']} 154", p.loops > 0,
          "#d8483f" if p.loops else idle, "re-plan" + (f" ×{p.loops}" if p.loops > 1 else ""), xs["execute"], 204)

    for s in STAGES:
        state, detail = p.nodes[s]
        color, symbol, _ = STATES[state]
        faint = ' opacity="0.5"' if state in ("wait", "skip") else ""
        if state == "active":
            out.append(f'<circle class="pl-pulse" cx="{xs[s]}" cy="{y}" r="{r + 5}" fill="none" stroke="{color}" '
                       f'stroke-width="3"/>')
        out.append(f'<g{faint}><circle cx="{xs[s]}" cy="{y}" r="{r}" fill="{color}" fill-opacity="0.14" stroke="{color}" '
                   f'stroke-width="2"/><path d="{ICONS[s]}" transform="translate({xs[s] - 14} {y - 14}) scale(1.17)" '
                   f'fill="none" stroke="{color}" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"/></g>')
        if symbol:
            out.append(f'<circle cx="{xs[s] + 21}" cy="{y - 21}" r="10" fill="{color}"/><text x="{xs[s] + 21}" '
                       f'y="{y - 16.5}" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">{symbol}</text>')
        out.append(f'<text x="{xs[s]}" y="{y + 46}" text-anchor="middle" font-size="14" font-weight="600" '
                   f'fill="currentColor">{TITLES[s]}</text><text x="{xs[s]}" y="{y + 63}" text-anchor="middle" '
                   f'font-size="12" fill="currentColor" opacity="0.8">{html.escape(detail)}</text>')
    markers = "".join(f'<marker id="pl-{k}" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" '
                      f'orient="auto"><path d="M0 0L8 4L0 8z" fill="{c}"/></marker>'
                      for k, c in (("ok", "#2e9d4f"), ("fail", "#d8483f"), ("idle", idle)))
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 212" width="1000" height="212" '
            f'font-family="Segoe UI, Helvetica, Arial, sans-serif" color="#0b0b0b">'
            "<style>.pl-pulse{animation:p 1.6s ease-in-out infinite}@keyframes p{50%{opacity:.2}}"
            "@media (prefers-reduced-motion:reduce){.pl-pulse{animation:none}}</style>"
            f'<defs>{markers}</defs><rect x="1" y="1" width="998" height="210" rx="10" fill="#fcfcfb" stroke="#e4e3de"/>'
            f"{''.join(out)}</svg>")


# ----------------------------------------------------------------------------- step graph
# Plain symbols (text presentation), not emoji: they look alike on every computer and in print.
TOOL_ICON = {"search_flights": "✈︎", "search_hotels": "⌂", "search_ferries": "⚓︎", "search_experiences": "★",
             "geocode": "⌖", "directions": "➤", "forecast": "☀︎"}
KIND_ICON = {"select": "▽", "compute": "∑"}
STEP_STYLE = {"ok": ("#d6f0d6", "✓"), "ok_null": ("#fdeec4", "!"), "error": ("#f6d2d2", "✗"), "skipped": ("#ecebe7", "–"),
              "running": ("#dbe8f8", "…")}
LEGEND = ("✈︎ flights · ⌂ hotels · ⚓︎ ferries · ★ experiences · ⌖ geocode · ➤ directions · ☀︎ weather · "
          "▽ select · ∑ compute. Status: ✓ ok, ! empty fallback, ✗ error, – skipped, … running. Steps in one column "
          "have no dependency on each other; tool calls in a column run at the same time.")


def _duration(ms) -> str:
    if not isinstance(ms, (int, float)):
        return ""
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms" if ms >= 1 else "<1 ms"


def step_graph(trace: list[dict], edges: list | None = None) -> graphviz.Digraph:
    """Executed steps, one column per DAG level."""
    g = graphviz.Digraph(graph_attr={"rankdir": "LR", "bgcolor": "transparent", "fontname": "Helvetica", "nodesep": "0.15",
                                     "ranksep": "0.4"},
                         node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica", "fontsize": "10",
                                    "fontcolor": "#0b0b0b", "color": "#a5a49d", "margin": "0.1,0.05"},
                         edge_attr={"color": "#a5a49d", "arrowsize": "0.6"})
    level: dict[str, int] = {}
    columns: dict[int, list[str]] = {}
    for s in trace:
        sid = s["step_id"]
        if sid in level:
            continue
        level[sid] = 1 + max((level.get(d, 0) for d in s.get("depends_on", [])), default=0)
        columns.setdefault(level[sid], []).append(sid)
        fill, mark = STEP_STYLE.get(s.get("status", "running"), ("#ffffff", ""))
        is_tool = s.get("kind") == "tool"
        icon = (TOOL_ICON.get(s.get("tool") or "", "•") if is_tool else KIND_ICON.get(s.get("kind") or "", "•"))
        took = _duration(s.get("latency_ms")) if is_tool and s.get("status") != "skipped" else ""
        what = (s.get("server") or "") if is_tool else ""
        detail = " · ".join(x for x in (what, took) if x)
        label = graphviz.escape(f"{icon}  {sid}  {mark}".rstrip()) + (r"\n" + graphviz.escape(detail) if detail else "")
        g.node(sid, label=label, fillcolor=fill)
    for ids in columns.values():
        with g.subgraph() as column:
            column.attr(rank="same")
            for sid in ids:
                column.node(sid)
    pairs = edges if edges else [(d, s["step_id"]) for s in trace for d in s.get("depends_on", [])]
    for a, b in sorted({tuple(p) for p in pairs}):
        if a in level and b in level:
            g.edge(a, b)
    return g

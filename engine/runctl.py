"""Run control for batch_runner: PID file + process identity checks (stdlib only, no engine imports).

The runner writes ``<logs_dir>/batch_runner.pid`` on start and removes it on exit. The file stores the
process creation time so a reused PID (common on Windows) is never mistaken for the runner.
Graceful stop is the ``STOP`` file; force stop terminates the verified PID.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

PID_FILE = "batch_runner.pid"
STOP_FILE = "STOP"  # touch <logs_dir>/STOP to stop gracefully after the current episode


def process_start_time(pid: int) -> float | None:
    """Creation time (epoch seconds) of a *running* process, or None if it is not running.

    On POSIX without psutil the creation time is unknown; returns 0.0 for a live process.
    """
    if pid <= 0:
        return None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k32.GetProcessTimes(h, *(ctypes.byref(t) for t in times)):
                return None
            ft = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            return ft / 1e7 - 11644473600.0
        finally:
            k32.CloseHandle(h)
    try:
        import psutil  # optional
        p = psutil.Process(pid)
        return None if p.status() == psutil.STATUS_ZOMBIE else p.create_time()
    except ImportError:
        try:
            os.kill(pid, 0)  # POSIX only: signal 0 = existence check
            return 0.0
        except OSError:
            return None
    except Exception:  # noqa: BLE001 - NoSuchProcess / AccessDenied
        return None


def write_pid_file(logs_dir: str | os.PathLike, run_id: str, argv: list[str] | None = None) -> Path:
    path = Path(logs_dir) / PID_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    info = {"pid": pid, "created": process_start_time(pid), "run_id": run_id,
            "started_ts": time.time(), "argv": argv or []}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(info), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_pid_file(logs_dir: str | os.PathLike) -> dict | None:
    try:
        return json.loads((Path(logs_dir) / PID_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clear_pid_file(logs_dir: str | os.PathLike, pid: int | None = None) -> None:
    """Remove the PID file (only if it still belongs to ``pid`` when given)."""
    info = read_pid_file(logs_dir)
    if info is not None and (pid is None or info.get("pid") == pid):
        (Path(logs_dir) / PID_FILE).unlink(missing_ok=True)


def is_running(info: dict | None, tolerance_s: float = 2.0) -> bool:
    """True only if the PID is alive AND is the same process that wrote the file."""
    if not info or not isinstance(info.get("pid"), int):
        return False
    created = process_start_time(info["pid"])
    if created is None:
        return False
    if info.get("created") and created:  # both known -> verify identity
        return abs(created - info["created"]) <= tolerance_s
    return True


def active_run(logs_dir: str | os.PathLike) -> dict | None:
    """PID-file info of a verified live runner; a stale file (dead or reused PID) is removed."""
    info = read_pid_file(logs_dir)
    if info is None:
        return None
    if is_running(info):
        return info
    clear_pid_file(logs_dir, info.get("pid"))
    return None


LOG_FILES = ("runs.jsonl", "events.jsonl", "simulated_actions.jsonl")


class RunActiveError(RuntimeError):
    pass


def _rewrite_without(path: Path, run_id: str) -> int:
    """Atomically rewrite a JSONL file without the lines of ``run_id``. Returns lines removed."""
    if not path.exists():
        return 0
    needle = run_id.encode()
    removed = 0
    tmp = path.with_name(path.name + ".tmp")
    with path.open("rb") as src, tmp.open("wb") as dst:
        for line in src:
            if needle in line:  # cheap pre-filter, then confirm on the parsed object
                try:
                    if json.loads(line).get("run_id") == run_id:
                        removed += 1
                        continue
                except (json.JSONDecodeError, AttributeError):
                    pass
            dst.write(line)
    for attempt in range(20):  # Windows: a reader may hold the file open for a moment
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 19:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.1)
    return removed


def delete_run(logs_dir: str | os.PathLike, run_id: str, cache_dir: str | os.PathLike | None = None) -> dict:
    """Remove every trace of one run: its lines in the three JSONL logs and its plan-cache files.

    The only sanctioned exception to append-only logs. Refused while any runner is active on this
    logs dir, because a concurrent append during the rewrite could be lost. The shared replay cache
    (keyed by tool args, not by run) is left untouched.
    """
    if not run_id or not isinstance(run_id, str):
        raise ValueError("run_id required")
    if active_run(logs_dir) is not None:
        raise RunActiveError("a run is active on this logs directory; stop it before deleting runs")
    logs = Path(logs_dir)
    counts = {name: _rewrite_without(logs / name, run_id) for name in LOG_FILES}
    removed_cache = []
    if cache_dir and Path(cache_dir).is_dir():
        for f in Path(cache_dir).glob(f"plan_cache_{run_id}_*.json"):
            f.unlink(missing_ok=True)
            removed_cache.append(f.name)
    counts["plan_cache_files"] = len(removed_cache)
    return counts


def request_stop(logs_dir: str | os.PathLike) -> None:
    (Path(logs_dir) / STOP_FILE).touch()


def force_stop(logs_dir: str | os.PathLike, wait_s: float = 10.0) -> bool:
    """Terminate the verified runner. Returns True if it is gone afterwards."""
    info = active_run(logs_dir)
    if info is None:
        return True
    try:
        os.kill(info["pid"], signal.SIGTERM)  # TerminateProcess on Windows
    except OSError:
        pass
    deadline = time.time() + wait_s
    while time.time() < deadline and is_running(info):
        time.sleep(0.1)
    gone = not is_running(info)
    if gone:
        clear_pid_file(logs_dir, info["pid"])
        (Path(logs_dir) / STOP_FILE).unlink(missing_ok=True)
    return gone

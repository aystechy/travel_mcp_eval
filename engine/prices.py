"""LLM prices from ``config/llm_prices.csv`` (stdlib only, no network).

Models are matched exactly (``*`` is allowed only for providers that are not billed). A model that is not listed has an
unknown price: ``None``, never a guess. Prices are read by hand from the providers' pages (``source_url``, ``checked_on``).
"""
from __future__ import annotations

import csv
import hashlib
import math
from functools import lru_cache
from pathlib import Path

PRICES_PATH = Path(__file__).resolve().parent.parent / "config" / "llm_prices.csv"
NUMBERS = ("input", "cached_input", "cache_write", "output", "max_prompt_tokens")
UNBILLED = ("mock", "ollama")


def load_prices(path: str | Path = PRICES_PATH) -> dict[tuple[str, str], dict]:
    """(provider, model) -> row. Raises ValueError on a missing, invalid or duplicate price."""
    out: dict[tuple[str, str], dict] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for n, raw in enumerate(csv.DictReader(f), 2):
            row: dict = {k: (v or "").strip() for k, v in raw.items() if k}
            key = (row.get("provider", ""), row.get("model", ""))
            if not all(key) or key in out or (key[1] == "*" and key[0] not in UNBILLED):
                raise ValueError(f"{path} line {n}: missing, duplicate or not allowed: {key}")
            for k in NUMBERS:
                try:
                    row[k] = float(row[k]) if row.get(k) else None
                except ValueError:
                    raise ValueError(f"{path} line {n}: {k} is not a number") from None
                if row[k] is not None and not (math.isfinite(row[k]) and row[k] >= 0):
                    raise ValueError(f"{path} line {n}: {k} must be 0 or more")
            if row["input"] is None or row["output"] is None:
                raise ValueError(f"{path} line {n}: input and output prices are required")
            out[key] = row
    return out


@lru_cache(maxsize=4)
def _load(path: str, mtime_ns: int) -> tuple[dict, str]:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return load_prices(path), hashlib.sha256(data).hexdigest()[:12]


def _table(path: str | Path) -> tuple[dict, str]:
    return _load(str(path), Path(path).stat().st_mtime_ns)


def find(provider: str, model: str | None, path: str | Path = PRICES_PATH) -> dict | None:
    table = _table(path)[0]
    return table.get((provider, model or "")) or table.get((provider, "*"))


def file_hash(path: str | Path = PRICES_PATH) -> str:
    return _table(path)[1]


def price(provider: str, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0,
          cache_write_tokens: int = 0, path: str | Path = PRICES_PATH) -> float | None:
    """Estimated USD of one call, or None if unknown. ``prompt_tokens`` includes cached and written tokens."""
    row = find(provider, model, path)
    if row is None or (row["max_prompt_tokens"] is not None and prompt_tokens > row["max_prompt_tokens"]):
        return None
    cached = min(max(cached_tokens, 0), prompt_tokens)
    written = min(max(cache_write_tokens, 0), prompt_tokens - cached)

    def rate(k: str) -> float:  # a discount that is not listed is not assumed
        return row["input"] if row[k] is None else row[k]

    usd = ((prompt_tokens - cached - written) * row["input"] + cached * rate("cached_input")
           + written * rate("cache_write") + completion_tokens * row["output"])
    return round(usd / 1e6, 6)

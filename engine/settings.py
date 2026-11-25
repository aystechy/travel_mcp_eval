"""LLM/API settings shared by the runner and the UI (stdlib only, no engine imports).

Effective value of a variable: this browser session > system environment > ``.env`` > default. Secret
values are never returned to the UI for display, logged, or put on a command line.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
SECRETS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "MAPBOX_ACCESS_TOKEN")
PLAIN = ("OLLAMA_HOST", "OLLAMA_MODEL", "OPENAI_MODEL", "ANTHROPIC_MODEL", "DEEPSEEK_MODEL", "MISTRAL_MODEL")
GROUPS = {"openai": ("OPENAI_API_KEY", "OPENAI_MODEL"), "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"),
          "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"),
          "mistral": ("MISTRAL_API_KEY", "MISTRAL_MODEL"), "ollama": ("OLLAMA_HOST", "OLLAMA_MODEL"),
          "mapbox": ("MAPBOX_ACCESS_TOKEN",)}
KEYS = SECRETS + PLAIN
DEFAULTS = {"OLLAMA_HOST": "http://localhost:11434", "OLLAMA_MODEL": "qwen2.5:7b-instruct-q4_K_M",
            "OPENAI_MODEL": "gpt-4o-mini", "ANTHROPIC_MODEL": "claude-sonnet-5-5",
            "DEEPSEEK_MODEL": "deepseek-flash", "MISTRAL_MODEL": "mistral-small-2603"}


def read_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    """Non-empty KEY=VALUE pairs of a .env file."""
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    out[k.strip()] = v.strip()
    return out


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Fill missing environment variables from .env (existing variables win)."""
    for k, v in read_env_file(path).items():
        os.environ.setdefault(k, v)


def clean(value: str) -> str:
    """A value safe for one .env line / environment variable."""
    value = value.strip()
    if not value.isprintable():
        raise ValueError("values cannot contain line breaks or control characters")
    return value


def save_env_file(values: dict[str, str], path: Path = ENV_PATH) -> None:
    """Set the given keys in .env, keeping every other line and comment. Atomic."""
    values = {k: clean(v) for k, v in values.items()}
    if unknown := set(values) - set(KEYS):
        raise ValueError(f"not a setting: {sorted(unknown)}")
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    left, out = dict(values), []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key not in values:
            out.append(line)
        elif key in left:  # later duplicates of the key are dropped
            out.append(f"{key}={left.pop(key)}")
    out += [f"{k}={v}" for k, v in left.items()]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def effective(session: dict[str, str] | None = None, path: Path = ENV_PATH) -> dict[str, tuple[str, str]]:
    """key -> (value, origin); origin is session | environment | .env | default | not set."""
    env_file, out = read_env_file(path), {}
    for k in KEYS:
        candidates = (((session or {}).get(k), "session"), (os.environ.get(k), "environment"),
                      (env_file.get(k), ".env"), (DEFAULTS.get(k), "default"))
        out[k] = next(((v, origin) for v, origin in candidates if v), ("", "not set"))
    return out


def redact(text: str) -> str:
    """Replace any configured secret that appears in ``text`` (e.g. in a provider error)."""
    for k in SECRETS:
        v = os.environ.get(k)
        if v and len(v) >= 8:
            text = text.replace(v, "***")
    return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # never forward a credential to another host
        return None


def check_connection(name: str, values: dict[str, str], timeout: float = 10.0) -> tuple[bool, str]:
    """Cheap authenticated request (model list or token check). Uses no LLM tokens."""
    need = next((k for k in GROUPS[name] if k in SECRETS), None)
    if need and not values[need]:
        return False, f"{need} is not set"
    headers: dict[str, str] = {}
    if name == "ollama":
        url = values["OLLAMA_HOST"].rstrip("/") + "/api/tags"
        if not url.startswith(("http://", "https://")):
            return False, "the address must start with http:// or https://"
    elif name == "openai":
        url, headers = "https://api.openai.com/v1/models", {"Authorization": f"Bearer {values['OPENAI_API_KEY']}"}
    elif name == "deepseek":
        url, headers = "https://api.deepseek.com/models", {"Authorization": f"Bearer {values['DEEPSEEK_API_KEY']}"}
    elif name == "mistral":
        url, headers = "https://api.mistral.ai/v1/models", {"Authorization": f"Bearer {values['MISTRAL_API_KEY']}"}
    elif name == "anthropic":
        url = "https://api.anthropic.com/v1/models?limit=1000"
        headers = {"x-api-key": values["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}
    elif name == "mapbox":
        url = f"https://api.mapbox.com/tokens/v2?access_token={values['MAPBOX_ACCESS_TOKEN']}"
    else:
        raise ValueError(name)
    try:
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            data = json.loads(r.read(5_000_000))
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}" + (" (key rejected)" if e.code in (401, 403) else "")
    except Exception as e:  # noqa: BLE001 - never echo the exception text: it can contain the URL
        return False, f"could not connect ({type(e).__name__})"
    if name == "mapbox":
        valid = data.get("code") == "TokenValid"
        return valid, "token is valid" if valid else "token was rejected"
    model = values[f"{name.upper()}_MODEL"]
    models = [m.get("name") for m in data.get("models", [])] if name == "ollama" else \
        [m.get("id") for m in data.get("data", [])]
    if model in models or f"{model}:latest" in models:
        return True, f"connected; model `{model}` is available"
    return False, f"connected, but model `{model}` is not in the provider's list"

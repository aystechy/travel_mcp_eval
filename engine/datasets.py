"""Task datasets: listing, content hash, table view, validated edited copies (stdlib only).

The built-in datasets are never modified. Edits are saved as new files in ``data/custom/``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CUSTOM_DIR = DATA_DIR / "custom"
BUILTIN = ("sample_tasks", "tasks_100")
DEFAULT_DATASET = DATA_DIR / "sample_tasks.json"
COLUMNS = ("id", "origin", "destination", "depart_date", "return_date", "travelers", "budget", "max_stops",
           "min_hotel_rating", "budget_check", "description")
EDITABLE = COLUMNS[1:9]
MAX_TRAVELLERS = 8
MAX_NIGHTS = 30
PARTY = {1: "a solo traveller", 2: "two", 3: "three", 4: "four"}


def list_datasets(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """name -> path: built-in datasets first, then saved copies."""
    out = {n: data_dir / f"{n}.json" for n in BUILTIN if (data_dir / f"{n}.json").exists()}
    custom = data_dir / "custom"
    if custom.exists():
        out.update({p.stem: p for p in sorted(custom.glob("*.json"))})
    return out


def content_hash(data: dict) -> str:
    """Hash of what the benchmark reads (cities + tasks), so metadata and formatting do not matter."""
    payload = json.dumps({"cities": data["cities"], "tasks": data["tasks"]}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def origins(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Origin city -> airport, over the built-in datasets."""
    out: dict[str, str] = {}
    for n in BUILTIN:
        p = data_dir / f"{n}.json"
        if p.exists():
            for t in json.loads(p.read_text(encoding="utf-8"))["tasks"]:
                out.setdefault(t["origin"]["city"], t["origin"]["iata"])
    return dict(sorted(out.items()))


def to_rows(data: dict) -> list[dict]:
    pending = set(data.get("metadata", {}).get("needs_recalibration", []))
    return [{"id": t["id"], "origin": t["origin"]["city"], "destination": t["destination"]["city"],
             "depart_date": date.fromisoformat(t["depart_date"]), "return_date": date.fromisoformat(t["return_date"]),
             "travelers": t["travelers"], "budget": float(t["budget"]),
             "max_stops": t["constraints"].get("max_stops"),
             "min_hotel_rating": t["constraints"].get("min_hotel_rating"),
             "budget_check": "check budget" if t["id"] in pending else "ok",
             "description": t["description"]} for t in data["tasks"]]


def describe(t: dict) -> str:
    nights = (date.fromisoformat(t["return_date"]) - date.fromisoformat(t["depart_date"])).days
    c, d, n = t["constraints"], t["destination"], t["travelers"]
    text = f"{nights}-night trip from {t['origin']['city']} to {d['city']} for {PARTY.get(n, f'{n} travellers')}"
    if d["ferry"]:
        text += f", with ferry transfer from {d['ferry']['from_port']}"
    if c.get("max_stops") == 0:
        text += ", direct flights only"
    if "min_hotel_rating" in c:
        text += f", hotel rated {c['min_hotel_rating']} or higher"
    return text


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or str(v) in ("", "NaT", "<NA>", "None")


def _day(v) -> date:
    if isinstance(v, datetime):  # includes pandas Timestamp
        return v.date()
    return v if isinstance(v, date) else date.fromisoformat(str(v)[:10])


def _whole(v) -> int:
    f = float(v)
    if f != int(f):
        raise ValueError("not a whole number")
    return int(f)


def apply_rows(data: dict, rows: list[dict], airports: dict[str, str]) -> tuple[dict, list[str], list[str]]:
    """Apply edited table rows. Returns (new dataset, errors, ids of changed tasks); never mutates ``data``.

    Airports, ports and coordinates are derived from the chosen city, so they cannot disagree with it.
    """
    new = copy.deepcopy(data)
    pending = set(new.get("metadata", {}).get("needs_recalibration", []))
    errors: list[str] = []
    changed: list[str] = []
    by_id = {r["id"]: r for r in rows}
    if set(by_id) != {t["id"] for t in new["tasks"]} or len(rows) != len(new["tasks"]):
        return new, ["tasks cannot be added, removed or renumbered"], []
    for t in new["tasks"]:
        r, before = by_id[t["id"]], copy.deepcopy(t)

        def bad(msg: str) -> None:
            errors.append(f"{before['id']}: {msg}")

        if r["origin"] not in airports:
            bad(f"unknown origin {r['origin']!r}")
        elif r["origin"] != t["origin"]["city"]:
            t["origin"] = {"city": r["origin"], "iata": airports[r["origin"]]}
        if r["destination"] not in new["cities"]:
            bad(f"unknown destination {r['destination']!r}")
        elif r["destination"] != t["destination"]["city"]:
            t["destination"] = copy.deepcopy(new["cities"][r["destination"]])
        if t["origin"]["city"] == t["destination"]["city"]:
            bad("origin and destination are the same")
        try:
            dep, ret = _day(r["depart_date"]), _day(r["return_date"])
            if not 1 <= (ret - dep).days <= MAX_NIGHTS:
                bad(f"return must be 1 to {MAX_NIGHTS} nights after departure")
            t["depart_date"], t["return_date"] = dep.isoformat(), ret.isoformat()
        except (ValueError, TypeError):
            bad("dates are missing or invalid")
        try:
            t["travelers"] = _whole(r["travelers"])
            if not 1 <= t["travelers"] <= MAX_TRAVELLERS:
                bad(f"travellers must be 1 to {MAX_TRAVELLERS}")
        except (ValueError, TypeError):
            bad("travellers must be a whole number")
        try:
            b = round(float(r["budget"]), 2)
            if not (math.isfinite(b) and b > 0):
                raise ValueError
            t["budget"] = int(b) if b == int(b) else b
        except (ValueError, TypeError):
            bad("budget must be a positive number")
        try:
            if _blank(r["max_stops"]):
                t["constraints"].pop("max_stops", None)
            else:
                t["constraints"]["max_stops"] = _whole(r["max_stops"])
                if not 0 <= t["constraints"]["max_stops"] <= 2:
                    bad("max stops must be 0 to 2 or blank")
        except (ValueError, TypeError):
            bad("max stops must be 0 to 2 or blank")
        try:
            if _blank(r["min_hotel_rating"]):
                t["constraints"].pop("min_hotel_rating", None)
            else:
                t["constraints"]["min_hotel_rating"] = round(float(r["min_hotel_rating"]), 1)
                if not 1 <= t["constraints"]["min_hotel_rating"] <= 5:
                    bad("hotel rating must be 1 to 5 or blank")
        except (ValueError, TypeError):
            bad("hotel rating must be 1 to 5 or blank")
        if t != before:
            changed.append(t["id"])
            if t["budget"] != before["budget"]:
                pending.discard(t["id"])  # the user set this budget for the edited trip
            else:
                pending.add(t["id"])
            if not errors:
                # Keep a hand-written description when only the budget changed.
                if any(t[k] != before[k] for k in t if k != "budget"):
                    t["description"] = describe(t)
    new.setdefault("metadata", {})["needs_recalibration"] = sorted(pending)
    return new, errors, changed


def save_copy(data: dict, name: str, based_on: str, data_dir: Path = DATA_DIR) -> Path:
    """Write a new dataset file. Existing datasets are never overwritten, so a run's dataset stays fixed."""
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_-")[:40]
    if not slug:
        raise ValueError("enter a name")
    if slug in list_datasets(data_dir) or slug in BUILTIN:
        raise ValueError(f"a dataset named {slug!r} already exists")
    out = copy.deepcopy(data)
    out["metadata"] = {**out.get("metadata", {}), "dataset_name": slug, "based_on": based_on,
                       "num_tasks": len(out["tasks"])}
    for key in ("generator", "seed", "budget_margin", "budget_basis"):  # no longer true of an edited copy
        out["metadata"].pop(key, None)
    path = data_dir / "custom" / f"{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path

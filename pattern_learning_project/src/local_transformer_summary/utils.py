from __future__ import annotations

import csv
import gzip
import json
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

EVENT_TYPE_ORDER = {
    "serious_injury": 1,
    "normal_injury": 2,
    "near_miss": 3,
    "hazard_identification": 4,
    "audit_unsafe_condition": 5,
    "audit_unsafe_act": 6,
    "audit_other": 7,
    "task_overdue": 8,
    "task_open": 9,
    "task_completed_or_closed": 10,
    "task_other": 11,
    "other": 99,
}

SECTION_ORDER = [
    "serious_injury",
    "normal_injury",
    "near_miss",
    "hazard_identification",
    "audit_unsafe_condition",
    "audit_unsafe_act",
    "audit_other",
    "task_overdue",
    "task_open",
    "task_completed_or_closed",
    "task_other",
    "other",
]

SAFETY_KEYWORDS = [
    "unsafe", "hazard", "near miss", "injury", "incident", "risk", "control",
    "corrective", "overdue", "audit", "inspection", "observation", "fall",
    "slip", "trip", "struck", "caught", "pinch", "fire", "burn", "chemical",
    "forklift", "vehicle", "mobile equipment", "pedestrian", "electrical", "loto",
]

SUMMARY_FIELDS = [
    "unsafe_conditions_summary",
    "serious_injury_summary",
    "normal_injury_summary",
    "near_miss_summary",
    "hazards_summary",
    "audits_summary",
    "actions_summary",
    "recurring_themes",
    "dates_to_review",
    "data_gaps_or_cautions",
]

FACT_COUNT_FIELDS = [
    "event_count",
    "text_event_count",
    "serious_injury_count",
    "normal_injury_count",
    "near_miss_count",
    "hazard_identification_count",
    "audit_count",
    "unsafe_condition_audit_count",
    "unsafe_act_audit_count",
    "safe_condition_audit_count",
    "safe_act_audit_count",
    "task_count",
    "open_action_count",
    "overdue_action_count",
    "completed_or_closed_task_count",
    "safety_keyword_text_count",
]


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compact_text(value: object, max_chars: int = 1000) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def parse_dt(value: str | None):
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:19])
    except Exception:
        pass
    try:
        ts = pd.to_datetime(text, errors="coerce")
        return None if pd.isna(ts) else ts.to_pydatetime()
    except Exception:
        return None


def period_info(dt: datetime, freq: str) -> tuple[str, str, str]:
    freq = freq.upper()
    if freq == "Y":
        label = f"{dt.year:04d}"
        start = datetime(dt.year, 1, 1)
        end = datetime(dt.year, 12, 31)
    elif freq == "Q":
        q = (dt.month - 1) // 3 + 1
        sm = 3 * (q - 1) + 1
        em = sm + 2
        label = f"{dt.year:04d}Q{q}"
        start = datetime(dt.year, sm, 1)
        end = datetime(dt.year, em, 28) + timedelta(days=4)
        end = end.replace(day=1) - timedelta(days=1)
    elif freq == "M":
        label = f"{dt.year:04d}-{dt.month:02d}"
        start = datetime(dt.year, dt.month, 1)
        end = datetime(dt.year, dt.month, 28) + timedelta(days=4)
        end = end.replace(day=1) - timedelta(days=1)
    elif freq == "W":
        iso = dt.isocalendar()
        label = f"{iso.year:04d}-W{iso.week:02d}"
        start = dt - timedelta(days=dt.weekday())
        end = start + timedelta(days=6)
    else:
        raise ValueError("PERIOD must be one of Y, Q, M, W")
    return label, start.date().isoformat(), end.date().isoformat()


def open_text_csv(path: Path | str):
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def location_leaf(location_path: str, fallback: str = "Unknown Location") -> str:
    if not location_path or not str(location_path).strip():
        return fallback
    parts = [p.strip() for p in str(location_path).split(">") if p.strip()]
    return parts[-1] if parts else fallback


def classify_row(row: dict, include_safety_keywords: bool = False) -> dict:
    source = (row.get("source_type") or "").strip().lower()
    subtype = (row.get("source_subtype") or "").strip().lower()
    category = (row.get("category") or "").strip().lower()
    status = (row.get("status") or "").strip().lower()
    audit_type = (row.get("audit_type") or "").strip().lower()

    any_injury = parse_bool(row.get("any_injury"))
    severe_actual = parse_bool(row.get("severe_actual"))
    is_open_task = parse_bool(row.get("is_open_task"))
    is_overdue_task = parse_bool(row.get("is_overdue_task"))

    is_serious = source == "incident" and severe_actual
    is_normal = source == "incident" and any_injury and not severe_actual
    is_near_miss = source == "near_miss" or "near miss" in subtype or "near miss" in category
    is_hazard = source == "hazard_identification" or category == "hazard identification" or subtype == "hazard identification"
    is_audit = source == "audit"
    is_task = source == "task"
    is_unsafe_condition = is_audit and ("unsafe condition" in audit_type or "unsafe condition" in subtype)
    is_unsafe_act = is_audit and ("unsafe act" in audit_type or "unsafe act" in subtype)
    is_safe_condition = is_audit and ("safe condition" in audit_type or "safe condition" in subtype)
    is_safe_act = is_audit and ("safe act" in audit_type or "safe act" in subtype)
    has_completion = bool(str(row.get("completion_date") or "").strip())
    is_completed_task = is_task and ("closed" in status or "complete" in status or has_completion)

    if is_serious:
        review_type = "serious_injury"
    elif is_normal:
        review_type = "normal_injury"
    elif is_near_miss:
        review_type = "near_miss"
    elif is_hazard:
        review_type = "hazard_identification"
    elif is_unsafe_condition:
        review_type = "audit_unsafe_condition"
    elif is_unsafe_act:
        review_type = "audit_unsafe_act"
    elif is_audit:
        review_type = "audit_other"
    elif is_task and is_overdue_task:
        review_type = "task_overdue"
    elif is_task and is_open_task:
        review_type = "task_open"
    elif is_completed_task:
        review_type = "task_completed_or_closed"
    elif is_task:
        review_type = "task_other"
    else:
        review_type = "other"

    clean_text = row.get("clean_text") or ""
    has_text = parse_bool(row.get("has_text")) or bool(clean_text.strip())
    if include_safety_keywords and clean_text:
        lo_text = clean_text.lower()
        has_safety_keyword = any(k in lo_text for k in SAFETY_KEYWORDS)
    else:
        has_safety_keyword = False

    return {
        "any_injury": any_injury,
        "severe_actual": severe_actual,
        "is_open_task": is_open_task,
        "is_overdue_task": is_overdue_task,
        "is_serious": is_serious,
        "is_normal": is_normal,
        "is_near_miss": is_near_miss,
        "is_hazard": is_hazard,
        "is_audit": is_audit,
        "is_task": is_task,
        "is_unsafe_condition": is_unsafe_condition,
        "is_unsafe_act": is_unsafe_act,
        "is_safe_condition": is_safe_condition,
        "is_safe_act": is_safe_act,
        "is_completed_task": is_completed_task,
        "has_text": has_text,
        "has_safety_keyword": has_safety_keyword,
        "review_event_type": review_type,
        "review_priority": EVENT_TYPE_ORDER.get(review_type, 99),
    }


def blank_fact() -> dict:
    out = {field: 0 for field in FACT_COUNT_FIELDS}
    out["first_event_date"] = None
    out["last_event_date"] = None
    return out


def update_min_max(fact: dict, dt: datetime) -> None:
    date_s = dt.isoformat(sep=" ")
    if fact["first_event_date"] is None or date_s < fact["first_event_date"]:
        fact["first_event_date"] = date_s
    if fact["last_event_date"] is None or date_s > fact["last_event_date"]:
        fact["last_event_date"] = date_s


def make_event_detail(row: dict, event_dt: datetime, flags: dict, max_chars: int) -> str:
    date = event_dt.date().isoformat() if event_dt else ""
    event_id = row.get("event_id") or ""
    subtype = row.get("source_subtype") or row.get("category") or ""
    status = row.get("status") or ""
    text = row.get("clean_text") or row.get("title") or row.get("description") or ""
    return f"[{date}] {flags['review_event_type']} | {event_id} | {subtype} | status={status} | {compact_text(text, max_chars)}"


def facts_to_frame(facts: Dict[Tuple[str, str, str, str, str, str], dict]) -> pd.DataFrame:
    fact_rows = []
    for key, val in facts.items():
        location_id, location_label, location_path, period, period_start, period_end = key
        row = {
            "location_id": location_id,
            "location_label": location_label,
            "location_path": location_path,
            "period": period,
            "period_start": period_start,
            "period_end": period_end,
        }
        row.update(val)
        row["review_priority_score"] = (
            10 * row["serious_injury_count"]
            + 4 * row["normal_injury_count"]
            + 3 * row["near_miss_count"]
            + row["hazard_identification_count"]
            + 2 * row["unsafe_condition_audit_count"]
            + 2 * row["unsafe_act_audit_count"]
            + 2 * row["overdue_action_count"]
            + 0.5 * row["open_action_count"]
        )
        fact_rows.append(row)
    if not fact_rows:
        return pd.DataFrame()
    return pd.DataFrame(fact_rows).sort_values(["review_priority_score", "event_count"], ascending=False)


def keyword_themes(text: str, max_terms: int = 12) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
    stop = set(
        "title description status source event date dates with from that this were have been into during "
        "employee employees record records location task tasks audit audits incident incidents near miss closed open pending "
        "injury injuries safety action actions corrective hazard hazards observation observations observed identified identification "
        "summary summarize period immediateaction immediate offpremiseslocation premiseslocation first second third shift area "
        "department plant site review completed complete normal serious condition unsafe safe act ehs".split()
    )
    counts = Counter(w for w in words if w not in stop and not w.isdigit())
    return ", ".join([w for w, _ in counts.most_common(max_terms)])


def date_list(text: str, max_dates: int = 20) -> str:
    dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    return ", ".join(sorted(set(dates))[:max_dates]) if dates else "No dates found in sampled evidence."


class ProgressLogger:
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start = time.time()
        print(f"[{self._ts()}] START {step_name}", flush=True)

    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, message: str) -> None:
        elapsed = time.time() - self.start
        print(f"[{self._ts()}] {self.step_name}: {message} elapsed={elapsed:,.1f}s", flush=True)

    def done(self, message: str = "done") -> None:
        elapsed = time.time() - self.start
        print(f"[{self._ts()}] DONE {self.step_name}: {message} elapsed={elapsed:,.1f}s", flush=True)


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

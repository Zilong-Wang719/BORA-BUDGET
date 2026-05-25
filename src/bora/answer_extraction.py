from __future__ import annotations

import re

from bora.common import safe_float


EXPLICIT_PATTERNS = [
    r"\bfinal\s*answer\s*[:：]\s*(.+)$",
    r"\bthe\s+final\s+answer\s+is\s+(.+)$",
    r"\banswer\s*[:：]\s*(.+)$",
    r"\banswer\s+is\s+(.+)$",
]


def extract_last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else None


def extract_numeric_answer(text: str) -> str | None:
    if not text:
        return None
    whole_hour_time = re.search(r"\b(\d{1,2}):00\b", text)
    if whole_hour_time:
        return whole_hour_time.group(1)
    return extract_last_number(text)


def clean_candidate(raw: str) -> str:
    candidate = (raw or "").strip()
    if not candidate:
        return candidate
    candidate = re.sub(r"^[\s>*`_~#-]+", "", candidate)
    candidate = candidate.strip()
    candidate = candidate.rstrip(" .。!！")
    return candidate.strip()


def extract_explicit_answer(text: str, *, prefer_numeric: bool = True) -> str | None:
    if not text:
        return None
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines):
        boxed = re.search(r"\\boxed\{([^{}]+)\}", line)
        if boxed:
            candidate = clean_candidate(boxed.group(1))
            if not candidate:
                continue
            if prefer_numeric:
                return extract_numeric_answer(candidate) or candidate
            return candidate

        normalized = re.sub(r"[*`_~]", "", line)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        for pattern in EXPLICIT_PATTERNS:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = clean_candidate(match.group(1).splitlines()[0])
            if not candidate:
                continue
            if prefer_numeric:
                return extract_numeric_answer(candidate) or candidate
            return candidate
    return None


def extract_tagged_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    pattern = re.compile(r"\[([A-Z_]+)\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)", flags=re.DOTALL)
    for name, content in pattern.findall(text or ""):
        sections[name] = content.strip()
    return sections


def parse_bool(text: str | None) -> bool | None:
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "done", "complete", "completed"}:
        return True
    if lowered in {"false", "no", "not yet", "incomplete"}:
        return False
    return None


def parse_score(text: str | None) -> float | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    numeric = safe_float(stripped.split()[0])
    if numeric is not None:
        return numeric
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return safe_float(match.group(0))
    return None

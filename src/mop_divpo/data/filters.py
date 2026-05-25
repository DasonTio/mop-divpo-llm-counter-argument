import re
from collections import defaultdict
from typing import Optional

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
_MATH_CHARS = set("$\\{}^_=")


def count_words(text: str) -> int:
    return len(text.split())


def is_empty(text: str) -> bool:
    return not text or not text.strip()


def is_too_short(text: str, min_words: int) -> bool:
    return count_words(text) < min_words


def is_too_long(text: str, max_words: int) -> bool:
    return count_words(text) > max_words


def has_deleted_marker(text: str) -> bool:
    return "[deleted]" in text or "[removed]" in text


def is_mostly_links(text: str, threshold: float = 0.4) -> bool:
    urls = _URL_RE.findall(text)
    if not urls:
        return False
    url_chars = sum(len(u) for u in urls)
    return url_chars / max(len(text), 1) > threshold


def is_mostly_code(text: str, threshold: float = 0.4) -> bool:
    blocks = _CODE_FENCE_RE.findall(text)
    if not blocks:
        return False
    code_chars = sum(len(b) for b in blocks)
    return code_chars / max(len(text), 1) > threshold


def is_formula_heavy(text: str, threshold: float = 0.12) -> bool:
    math_chars = sum(1 for c in text if c in _MATH_CHARS)
    return math_chars / max(len(text), 1) > threshold


class FilterStats:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)
        self.total_seen = 0
        self.total_kept = 0

    def record(self, reason: Optional[str]) -> None:
        self.total_seen += 1
        if reason is None:
            self.total_kept += 1
        else:
            self.counts[reason] += 1

    def to_dict(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "total_kept": self.total_kept,
            "total_filtered": self.total_seen - self.total_kept,
            "filter_counts": dict(self.counts),
        }

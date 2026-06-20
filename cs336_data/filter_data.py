from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tldextract import TLDExtract
from warcio.archiveiterator import ArchiveIterator

from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality import classify_quality, gopher_quality_filter
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech


MIN_DOCUMENT_WORDS = 80
MAX_DOCUMENT_WORDS = 50_000
MIN_DOCUMENT_CHARS = 300
MAX_DOCUMENT_CHARS = 500_000
MAX_MASKED_PII = 25
QUALITY_THRESHOLD = 0.60
HARMFUL_THRESHOLD = 0.80

_WORD_RE = re.compile(r"\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{39,}")
_LONG_TOKEN_RE = re.compile(r"\S{500,}")

_BAD_URL_PARTS = {
    "account",
    "cart",
    "category",
    "checkout",
    "comment",
    "comments",
    "contact",
    "feed",
    "forum",
    "gallery",
    "login",
    "logout",
    "member",
    "privacy",
    "profile",
    "register",
    "rss",
    "search",
    "signin",
    "signup",
    "tag",
    "tags",
    "terms",
    "user",
    "wp-admin",
}

_BAD_URL_SUFFIXES = {
    ".avi",
    ".css",
    ".gif",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".svg",
    ".webm",
    ".xml",
    ".zip",
}


@dataclass(frozen=True)
class FilterConfig:
    min_words: int = MIN_DOCUMENT_WORDS
    max_words: int = MAX_DOCUMENT_WORDS
    min_chars: int = MIN_DOCUMENT_CHARS
    max_chars: int = MAX_DOCUMENT_CHARS
    max_masked_pii: int = MAX_MASKED_PII
    use_gopher: bool = True
    use_quality_classifier: bool = False
    quality_threshold: float = QUALITY_THRESHOLD
    use_harmful_classifiers: bool = False
    harmful_threshold: float = HARMFUL_THRESHOLD
    keep_metadata: bool = False


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    empty: int = 0
    bad_url: int = 0
    malformed: int = 0
    length: int = 0
    repeated_text: int = 0
    gopher: int = 0
    low_quality_classifier: int = 0
    harmful: int = 0
    too_much_pii: int = 0
    pii_masked_documents: int = 0
    pii_replacements: int = 0
    errors: int = 0

    def add(self, other: "FilterStats") -> None:
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FilteredDocument:
    text: str
    url: str
    domain: str
    masked_pii: int

    def to_output_line(self, *, keep_metadata: bool = False) -> str:
        if not keep_metadata:
            return self.text
        return json.dumps(
            {
                "url": self.url,
                "domain": self.domain,
                "masked_pii": self.masked_pii,
                "text": self.text,
            },
            ensure_ascii=False,
        )


_extract_domain = TLDExtract(suffix_list_urls=(), cache_dir=None)


def normalize_document_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lstrip("\ufeff")


def url_domain(url: str) -> str:
    parsed = _extract_domain(url)
    return ".".join(part for part in (parsed.domain, parsed.suffix) if part)


def looks_like_bad_url(url: str) -> bool:
    lower_url = url.lower()
    if any(lower_url.endswith(suffix) for suffix in _BAD_URL_SUFFIXES):
        return True

    parts = {part for part in re.split(r"[^a-z0-9-]+", lower_url) if part}
    return bool(parts & _BAD_URL_PARTS)


def mask_pii(text: str) -> tuple[str, int]:
    text, num_emails = mask_emails(text)
    text, num_phones = mask_phone_numbers(text)
    text, num_ips = mask_ips(text)
    return text, num_emails + num_phones + num_ips


def filter_document(text: str, url: str, config: FilterConfig = FilterConfig()) -> tuple[FilteredDocument | None, str]:
    normalized = normalize_document_text(text)
    if not normalized:
        return None, "empty"

    if not url or looks_like_bad_url(url):
        return None, "bad_url"

    if _REPEATED_CHAR_RE.search(normalized) or _LONG_TOKEN_RE.search(normalized):
        return None, "repeated_text"

    num_chars = len(normalized)
    words = _WORD_RE.findall(normalized)
    if (
        num_chars < config.min_chars
        or num_chars > config.max_chars
        or len(words) < config.min_words
        or len(words) > config.max_words
    ):
        return None, "length"

    if config.use_gopher and not gopher_quality_filter(normalized):
        return None, "gopher"

    if config.use_quality_classifier:
        label, score = classify_quality(normalized)
        if label != "wiki" and score >= config.quality_threshold:
            return None, "low_quality_classifier"

    if config.use_harmful_classifiers:
        nsfw_label, nsfw_score = classify_nsfw(normalized)
        toxic_label, toxic_score = classify_toxic_speech(normalized)
        if (nsfw_label == "nsfw" and nsfw_score >= config.harmful_threshold) or (
            toxic_label == "toxic" and toxic_score >= config.harmful_threshold
        ):
            return None, "harmful"

    masked, num_masked = mask_pii(normalized)
    if num_masked > config.max_masked_pii:
        return None, "too_much_pii"

    return FilteredDocument(text=masked, url=url, domain=url_domain(url), masked_pii=num_masked), "kept"


def iter_wet_records(input_path: Path, *, max_records: int | None = None):
    with gzip.open(input_path, "rb") as input_stream:
        for record in ArchiveIterator(input_stream):
            if record.rec_type != "conversion":
                continue
            if max_records is not None and max_records <= 0:
                return
            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            payload = record.content_stream().read()
            if max_records is not None:
                max_records -= 1
            yield url, payload.decode("utf-8", errors="replace")


def filter_wet_file(
    input_path: Path,
    output_path: Path,
    *,
    config: FilterConfig = FilterConfig(),
    discarded_examples_path: Path | None = None,
    max_discarded_examples: int = 0,
    max_records: int | None = None,
) -> FilterStats:
    stats = FilterStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if discarded_examples_path is not None:
        discarded_examples_path.parent.mkdir(parents=True, exist_ok=True)

    discarded_counts: Counter[str] = Counter()
    discarded_file = (
        open(discarded_examples_path, "w", encoding="utf-8")
        if discarded_examples_path is not None and max_discarded_examples > 0
        else None
    )
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            for url, text in iter_wet_records(input_path, max_records=max_records):
                stats.seen += 1
                try:
                    document, reason = filter_document(text, url, config)
                except Exception:
                    stats.errors += 1
                    continue

                if document is None:
                    setattr(stats, reason, getattr(stats, reason) + 1)
                    if discarded_file is not None and discarded_counts[reason] < max_discarded_examples:
                        discarded_file.write(
                            json.dumps(
                                {
                                    "reason": reason,
                                    "url": url,
                                    "text": normalize_document_text(text)[:2000],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        discarded_counts[reason] += 1
                    continue

                stats.kept += 1
                stats.pii_replacements += document.masked_pii
                if document.masked_pii:
                    stats.pii_masked_documents += 1
                output_file.write(document.to_output_line(keep_metadata=config.keep_metadata) + "\n")
    finally:
        if discarded_file is not None:
            discarded_file.close()

    return stats


def write_report(path: Path, stats: FilterStats, config: FilterConfig, inputs: list[Path], outputs: list[Path]) -> None:
    kept_after_filter = {
        "input": stats.seen,
        "nonempty": stats.seen - stats.empty,
        "url": stats.seen - stats.empty - stats.bad_url,
        "text_shape": stats.seen - stats.empty - stats.bad_url - stats.repeated_text - stats.length,
        "gopher": stats.seen - stats.empty - stats.bad_url - stats.repeated_text - stats.length - stats.gopher,
        "quality_classifier": (
            stats.seen
            - stats.empty
            - stats.bad_url
            - stats.repeated_text
            - stats.length
            - stats.gopher
            - stats.low_quality_classifier
        ),
        "harmful_classifiers": (
            stats.seen
            - stats.empty
            - stats.bad_url
            - stats.repeated_text
            - stats.length
            - stats.gopher
            - stats.low_quality_classifier
            - stats.harmful
        ),
        "pii": stats.kept,
    }
    report: dict[str, Any] = {
        "stats": stats.to_dict(),
        "kept_after_filter": kept_after_filter,
        "discarded": stats.seen - stats.kept,
        "kept_fraction": stats.kept / stats.seen if stats.seen else 0.0,
        "discard_fraction_by_filter": {},
        "config": asdict(config),
        "inputs": [str(path) for path in inputs],
        "outputs": [str(path) for path in outputs],
    }
    discarded = max(stats.seen - stats.kept, 1)
    for key, value in stats.to_dict().items():
        if key in {"seen", "kept", "pii_masked_documents", "pii_replacements"}:
            continue
        report["discard_fraction_by_filter"][key] = value / discarded

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from cs336_data.common import MODAL_SHARED_PATH, get_shared_assets_path


MIN_GOPHER_WORDS = 50
MAX_GOPHER_WORDS = 100_000
MIN_MEAN_WORD_LENGTH = 3.0
MAX_MEAN_WORD_LENGTH = 10.0
MAX_ELLIPSIS_LINE_FRACTION = 0.30
MIN_ALPHABETIC_WORD_FRACTION = 0.80

QUALITY_MODEL_ENV_VARS = ("CS336_FASTTEXT_QUALITY_MODEL", "CS336_QUALITY_MODEL_PATH")
QUALITY_MODEL_RELATIVE_PATH = Path("classifiers/quality.bin")
QUALITY_POSITIVE_LABEL = "wiki"
QUALITY_NEGATIVE_LABEL = "cc"

_WORD_RE = re.compile(r"\S+")


class _FastTextModel(Protocol):
    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], tuple[float, ...]]: ...


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _fraction_lines_ending_with_ellipsis(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    return sum(line.endswith("...") for line in lines) / len(lines)


def gopher_quality_filter(text: str) -> bool:
    """Return whether text passes the assignment's subset of Gopher quality rules."""
    words = _words(text)
    num_words = len(words)
    if num_words < MIN_GOPHER_WORDS or num_words > MAX_GOPHER_WORDS:
        return False

    mean_word_length = sum(len(word) for word in words) / num_words
    if mean_word_length < MIN_MEAN_WORD_LENGTH or mean_word_length > MAX_MEAN_WORD_LENGTH:
        return False

    if _fraction_lines_ending_with_ellipsis(text) > MAX_ELLIPSIS_LINE_FRACTION:
        return False

    alphabetic_word_fraction = sum(any(char.isalpha() for char in word) for word in words) / num_words
    if alphabetic_word_fraction < MIN_ALPHABETIC_WORD_FRACTION:
        return False

    return True


def _candidate_model_paths() -> list[Path]:
    candidates: list[Path] = []
    for env_var in QUALITY_MODEL_ENV_VARS:
        if value := os.environ.get(env_var):
            candidates.append(Path(value).expanduser())

    candidates.extend(
        [
            get_shared_assets_path() / QUALITY_MODEL_RELATIVE_PATH,
            MODAL_SHARED_PATH / QUALITY_MODEL_RELATIVE_PATH,
            Path("local-shared-data") / QUALITY_MODEL_RELATIVE_PATH,
        ]
    )
    return candidates


def get_quality_model_path() -> Path:
    for path in _candidate_model_paths():
        if path.is_file():
            return path

    searched = "\n".join(f"  - {path}" for path in _candidate_model_paths())
    raise FileNotFoundError(
        "Could not find the fastText quality classifier.\n"
        "Train it first with `.venv/bin/python scripts/train_quality_classifier.py`, "
        "or set "
        f"{QUALITY_MODEL_ENV_VARS[0]} to the model path.\n"
        f"Searched:\n{searched}"
    )


def _prepare_text_for_fasttext(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_fasttext_label(label: str) -> str:
    if label.startswith("__label__"):
        return label.removeprefix("__label__")
    return label


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def format_fasttext_example(label: str, text: str) -> str:
    normalized_text = _prepare_text_for_fasttext(text)
    if not normalized_text:
        raise ValueError("Cannot create a fastText training example from empty text.")
    return f"__label__{label} {normalized_text}"


def chunk_text_for_quality_training(text: str, *, min_words: int = 30, max_words: int = 180) -> list[str]:
    """Split long documents into one-line examples suitable for fastText."""
    words = _words(_prepare_text_for_fasttext(text))
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    for start in range(0, len(words), max_words):
        chunk_words = words[start : start + max_words]
        if len(chunk_words) >= min_words:
            chunks.append(" ".join(chunk_words))
    return chunks


def build_quality_training_lines(
    positive_texts: Iterable[str],
    negative_texts: Iterable[str],
    *,
    min_words: int = 30,
    max_words: int = 180,
    balance_classes: bool = True,
) -> list[str]:
    positive_examples = [
        format_fasttext_example(QUALITY_POSITIVE_LABEL, chunk)
        for text in positive_texts
        for chunk in chunk_text_for_quality_training(text, min_words=min_words, max_words=max_words)
    ]
    negative_examples = [
        format_fasttext_example(QUALITY_NEGATIVE_LABEL, chunk)
        for text in negative_texts
        for chunk in chunk_text_for_quality_training(text, min_words=min_words, max_words=max_words)
    ]

    if not positive_examples:
        raise ValueError("No positive quality examples were provided.")
    if not negative_examples:
        raise ValueError("No negative quality examples were provided.")

    if balance_classes:
        positive_examples, negative_examples = _balance_examples(positive_examples, negative_examples)

    lines: list[str] = []
    for positive, negative in zip(positive_examples, negative_examples, strict=True):
        lines.append(positive)
        lines.append(negative)
    return lines


def _balance_examples(left: list[str], right: list[str]) -> tuple[list[str], list[str]]:
    size = max(len(left), len(right))
    balanced_left = [left[idx % len(left)] for idx in range(size)]
    balanced_right = [right[idx % len(right)] for idx in range(size)]
    return balanced_left, balanced_right


def _read_text_files(paths: Sequence[os.PathLike[str] | str]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if _prepare_text_for_fasttext(text):
            texts.append(text)
    return texts


def _default_positive_text_paths() -> list[Path]:
    return [Path("tests/fixtures/high_quality_wiki_reference.txt")]


def _default_negative_texts() -> list[str]:
    negative_texts = _read_text_files([Path("tests/fixtures/low_quality_cc.txt")])
    negative_texts.extend(
        [
            "Login Register Search FAQ Memberlist Usergroups Profile Private messages Contact us "
            "Copyright all rights reserved powered by forum software.",
            "Subscribe now Cookie policy Privacy policy Terms of service Advertisement Sponsored links "
            "Share this page Follow us on social media.",
            "404 Not Found The requested URL was not found on this server. Please check the address "
            "or return to the home page.",
            "Click here click here free download free download online games wallpapers ringtones "
            "latest offers sign up today.",
        ]
    )
    return negative_texts


def train_quality_classifier_from_texts(
    positive_texts: Iterable[str],
    negative_texts: Iterable[str],
    output_path: os.PathLike[str] | str,
    *,
    epoch: int = 20,
    learning_rate: float = 0.5,
    word_ngrams: int = 2,
    dim: int = 100,
    min_count: int = 1,
    thread: int = 1,
) -> Path:
    """Train and save a fastText quality classifier from positive and negative text examples."""
    import fasttext

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    training_lines = build_quality_training_lines(positive_texts, negative_texts)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".quality.txt", delete=False) as f:
        train_path = Path(f.name)
        for line in training_lines:
            f.write(line + "\n")

    try:
        model = fasttext.train_supervised(
            input=str(train_path),
            lr=learning_rate,
            epoch=epoch,
            wordNgrams=word_ngrams,
            dim=dim,
            minCount=min_count,
            loss="softmax",
            thread=thread,
            verbose=0,
        )
        model.save_model(str(output))
    finally:
        train_path.unlink(missing_ok=True)

    _load_quality_model.cache_clear()
    return output


def train_default_quality_classifier(output_path: os.PathLike[str] | str | None = None) -> Path:
    """Train a small local fallback classifier from bundled fixtures and benign low-quality samples."""
    output = Path(output_path) if output_path is not None else get_shared_assets_path() / QUALITY_MODEL_RELATIVE_PATH
    positive_texts = _read_text_files(_default_positive_text_paths())
    negative_texts = _default_negative_texts()
    return train_quality_classifier_from_texts(positive_texts, negative_texts, output)


@lru_cache(maxsize=1)
def _load_quality_model() -> _FastTextModel:
    import fasttext

    return fasttext.load_model(str(get_quality_model_path()))


def classify_quality(text: str) -> tuple[str, float]:
    """Return a wiki/cc quality label and fastText confidence score."""
    normalized_text = _prepare_text_for_fasttext(text)
    if not normalized_text:
        return QUALITY_NEGATIVE_LABEL, 0.0

    labels, scores = _load_quality_model().predict(normalized_text, k=1)
    if not labels or not scores:
        return QUALITY_NEGATIVE_LABEL, 0.0

    label = _normalize_fasttext_label(labels[0])
    if label not in {QUALITY_POSITIVE_LABEL, QUALITY_NEGATIVE_LABEL}:
        label = QUALITY_POSITIVE_LABEL if "wiki" in label.lower() else QUALITY_NEGATIVE_LABEL
    return label, _clamp_score(scores[0])

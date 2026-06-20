from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

from cs336_data.common import MODAL_SHARED_PATH, get_shared_assets_path


class _FastTextModel(Protocol):
    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], tuple[float, ...]]: ...


NSFW_MODEL_ENV_VARS = ("CS336_FASTTEXT_NSFW_MODEL", "CS336_NSFW_MODEL_PATH")
TOXIC_MODEL_ENV_VARS = ("CS336_FASTTEXT_TOXIC_MODEL", "CS336_TOXIC_MODEL_PATH")

NSFW_MODEL_RELATIVE_PATH = Path("classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
TOXIC_MODEL_RELATIVE_PATH = Path("classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

HarmfulContentTask = Literal["nsfw", "toxic"]


def _candidate_model_paths(env_vars: tuple[str, ...], relative_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for env_var in env_vars:
        if value := os.environ.get(env_var):
            candidates.append(Path(value).expanduser())

    candidates.extend(
        [
            get_shared_assets_path() / relative_path,
            MODAL_SHARED_PATH / relative_path,
            Path("local-shared-data") / relative_path,
        ]
    )
    return candidates


def _get_model_path(task: HarmfulContentTask) -> Path:
    if task == "nsfw":
        env_vars = NSFW_MODEL_ENV_VARS
        relative_path = NSFW_MODEL_RELATIVE_PATH
        description = "NSFW"
    else:
        env_vars = TOXIC_MODEL_ENV_VARS
        relative_path = TOXIC_MODEL_RELATIVE_PATH
        description = "toxic speech"

    for path in _candidate_model_paths(env_vars, relative_path):
        if path.is_file():
            return path

    searched = "\n".join(f"  - {path}" for path in _candidate_model_paths(env_vars, relative_path))
    raise FileNotFoundError(
        f"Could not find the Dolma fastText {description} classifier.\n"
        "Run `uv run scripts/download_data.py --offline-only`, or set "
        f"{env_vars[0]} to the model path.\n"
        f"Searched:\n{searched}"
    )


@lru_cache(maxsize=1)
def _load_nsfw_model() -> _FastTextModel:
    import fasttext

    return fasttext.load_model(str(_get_model_path("nsfw")))


@lru_cache(maxsize=1)
def _load_toxic_model() -> _FastTextModel:
    import fasttext

    return fasttext.load_model(str(_get_model_path("toxic")))


def _prepare_text_for_fasttext(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_fasttext_label(label: str) -> str:
    if label.startswith("__label__"):
        return label.removeprefix("__label__")
    return label


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _is_negative_label(label: str) -> bool:
    normalized = label.lower().replace("_", "-")
    return (
        normalized.startswith("non-")
        or normalized.startswith("not-")
        or normalized in {"safe", "clean", "neutral", "normal", "0", "false"}
    )


def _canonical_nsfw_label(label: str) -> str:
    return "non-nsfw" if _is_negative_label(label) else "nsfw"


def _canonical_toxic_label(label: str) -> str:
    return "non-toxic" if _is_negative_label(label) else "toxic"


def _classify_text(text: str, task: HarmfulContentTask) -> tuple[str, float]:
    normalized_text = _prepare_text_for_fasttext(text)
    if not normalized_text:
        return ("non-nsfw", 0.0) if task == "nsfw" else ("non-toxic", 0.0)

    model = _load_nsfw_model() if task == "nsfw" else _load_toxic_model()
    labels, scores = model.predict(normalized_text, k=1)
    if not labels or not scores:
        return ("non-nsfw", 0.0) if task == "nsfw" else ("non-toxic", 0.0)

    label = _normalize_fasttext_label(labels[0])
    score = _clamp_score(scores[0])
    if task == "nsfw":
        return _canonical_nsfw_label(label), score
    return _canonical_toxic_label(label), score


def classify_nsfw(text: str) -> tuple[str, float]:
    """Return an NSFW/non-NSFW label and fastText confidence score."""
    return _classify_text(text, "nsfw")


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """Return a toxic/non-toxic label and fastText confidence score."""
    return _classify_text(text, "toxic")

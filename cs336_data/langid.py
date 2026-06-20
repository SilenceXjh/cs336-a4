from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from cs336_data.common import MODAL_SHARED_PATH, get_shared_assets_path


class _FastTextModel(Protocol):
    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], tuple[float, ...]]: ...


MODEL_ENV_VARS = ("CS336_FASTTEXT_LID_MODEL", "CS336_LID_MODEL_PATH")
MODEL_RELATIVE_PATH = Path("classifiers/lid.176.bin")
UNKNOWN_LANGUAGE = "unknown"


def _candidate_model_paths() -> list[Path]:
    candidates: list[Path] = []
    for env_var in MODEL_ENV_VARS:
        if value := os.environ.get(env_var):
            candidates.append(Path(value).expanduser())

    candidates.extend(
        [
            get_shared_assets_path() / MODEL_RELATIVE_PATH,
            MODAL_SHARED_PATH / MODEL_RELATIVE_PATH,
            Path("local-shared-data") / MODEL_RELATIVE_PATH,
        ]
    )
    return candidates


def get_language_identification_model_path() -> Path:
    for path in _candidate_model_paths():
        if path.is_file():
            return path

    searched = "\n".join(f"  - {path}" for path in _candidate_model_paths())
    raise FileNotFoundError(
        "Could not find the fastText language identification model lid.176.bin.\n"
        "Run `uv run scripts/download_data.py --offline-only`, or set "
        f"{MODEL_ENV_VARS[0]} to the model path.\n"
        f"Searched:\n{searched}"
    )


@lru_cache(maxsize=1)
def _load_language_identification_model() -> _FastTextModel:
    import fasttext

    return fasttext.load_model(str(get_language_identification_model_path()))


def _prepare_text_for_fasttext(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_fasttext_label(label: str) -> str:
    if label.startswith("__label__"):
        return label.removeprefix("__label__")
    return label


def identify_language(text: str) -> tuple[str, float]:
    """Return the top fastText language prediction and confidence score."""
    normalized_text = _prepare_text_for_fasttext(text)
    if not normalized_text:
        return UNKNOWN_LANGUAGE, 0.0

    labels, scores = _load_language_identification_model().predict(normalized_text, k=1)
    if not labels or not scores:
        return UNKNOWN_LANGUAGE, 0.0

    language = _normalize_fasttext_label(labels[0])
    score = float(scores[0])
    return language, max(0.0, min(1.0, score))

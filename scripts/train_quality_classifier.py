from __future__ import annotations

import argparse
from pathlib import Path

from cs336_data.common import get_shared_assets_path
from cs336_data.quality import (
    QUALITY_MODEL_RELATIVE_PATH,
    build_quality_training_lines,
    train_quality_classifier_from_texts,
)


DEFAULT_POSITIVE_TEXT = Path("tests/fixtures/high_quality_wiki_reference.txt")
DEFAULT_NEGATIVE_TEXT = Path("tests/fixtures/low_quality_cc.txt")


def _read_texts(paths: list[Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if text.strip():
            texts.append(text)
    return texts


def _write_training_file(path: Path, positive_texts: list[str], negative_texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = build_quality_training_lines(positive_texts, negative_texts)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fastText webpage quality classifier.")
    parser.add_argument(
        "--positive-text-file",
        action="append",
        type=Path,
        default=[],
        help="Text file containing high-quality examples. May be passed multiple times.",
    )
    parser.add_argument(
        "--negative-text-file",
        action="append",
        type=Path,
        default=[],
        help="Text file containing low-quality/Common Crawl examples. May be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=get_shared_assets_path() / QUALITY_MODEL_RELATIVE_PATH,
        help="Path where the trained fastText .bin model should be written.",
    )
    parser.add_argument(
        "--save-train-file",
        type=Path,
        default=None,
        help="Optional path to save the generated fastText supervised training file.",
    )
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--thread", type=int, default=1)
    args = parser.parse_args()

    positive_paths = args.positive_text_file or [DEFAULT_POSITIVE_TEXT]
    negative_paths = args.negative_text_file or [DEFAULT_NEGATIVE_TEXT]
    positive_texts = _read_texts(positive_paths)
    negative_texts = _read_texts(negative_paths)

    if args.save_train_file is not None:
        _write_training_file(args.save_train_file, positive_texts, negative_texts)

    output = train_quality_classifier_from_texts(
        positive_texts,
        negative_texts,
        args.output,
        epoch=args.epoch,
        learning_rate=args.lr,
        word_ngrams=args.word_ngrams,
        dim=args.dim,
        min_count=args.min_count,
        thread=args.thread,
    )
    print(f"Wrote quality classifier to {output}")


if __name__ == "__main__":
    main()

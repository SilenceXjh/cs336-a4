from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from cs336_data.extract import iter_text_from_warc
from cs336_data.langid import identify_language


DEFAULT_WARC_PATH = Path("local-shared-data/CC/example.warc.gz")


@dataclass(frozen=True)
class LanguageSample:
    index: int
    url: str
    predicted_language: str
    score: float
    text: str

    @property
    def excerpt(self) -> str:
        return self.text[: self._excerpt_chars].replace("|", "\\|")

    _excerpt_chars: int = 500


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def collect_candidate_records(
    warc_path: Path,
    *,
    limit: int | None,
    min_chars: int,
) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for url, text in iter_text_from_warc(warc_path, limit=limit):
        normalized_text = normalize_text(text)
        if len(normalized_text) >= min_chars:
            records.append((url, normalized_text))
    return records


def sample_language_predictions(
    warc_path: Path,
    *,
    num_examples: int,
    seed: int,
    limit: int | None,
    min_chars: int,
    excerpt_chars: int,
) -> list[LanguageSample]:
    records = collect_candidate_records(warc_path, limit=limit, min_chars=min_chars)
    if not records:
        raise ValueError(f"No records with at least {min_chars} extracted characters found in {warc_path}")

    rng = random.Random(seed)
    selected_records = rng.sample(records, min(num_examples, len(records)))
    samples: list[LanguageSample] = []
    for index, (url, text) in enumerate(selected_records, start=1):
        predicted_language, score = identify_language(text)
        samples.append(
            LanguageSample(
                index=index,
                url=url,
                predicted_language=predicted_language,
                score=score,
                text=text[:excerpt_chars],
                _excerpt_chars=excerpt_chars,
            )
        )
    return samples


def write_markdown(samples: list[LanguageSample], output_file: TextIO) -> None:
    counts = Counter(sample.predicted_language for sample in samples)
    english_count = counts["en"]
    english_fraction = english_count / len(samples) if samples else 0.0

    output_file.write("# Language Identification Sample\n\n")
    output_file.write(f"Total sampled documents: {len(samples)}\n\n")
    output_file.write(f"Predicted English fraction: {english_count}/{len(samples)} = {english_fraction:.2%}\n\n")
    output_file.write("Predicted language counts:\n\n")
    for language, count in sorted(counts.items()):
        output_file.write(f"- `{language}`: {count}\n")
    output_file.write("\n")

    output_file.write("| # | Manual Language | Prediction | Score | URL | Excerpt |\n")
    output_file.write("|---:|---|---|---:|---|---|\n")
    for sample in samples:
        output_file.write(
            f"| {sample.index} |  | `{sample.predicted_language}` | {sample.score:.4f} | "
            f"{sample.url} | {sample.excerpt} |\n"
        )


def write_tsv(samples: list[LanguageSample], output_file: TextIO) -> None:
    writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
    writer.writerow(["index", "manual_language", "predicted_language", "score", "url", "excerpt"])
    for sample in samples:
        writer.writerow(
            [
                sample.index,
                "",
                sample.predicted_language,
                f"{sample.score:.6f}",
                sample.url,
                sample.text,
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample extracted WARC documents and run fastText language identification for assignment 2.3(c)."
    )
    parser.add_argument(
        "--warc-path",
        type=Path,
        default=DEFAULT_WARC_PATH,
        help=f"Input WARC file. Defaults to {DEFAULT_WARC_PATH}.",
    )
    parser.add_argument("--num-examples", type=int, default=20, help="Number of documents to sample.")
    parser.add_argument("--seed", type=int, default=336, help="Random seed used for sampling.")
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Only scan the first N response records before sampling. Use 0 to scan the whole WARC.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Minimum extracted text length for a record to be eligible for sampling.",
    )
    parser.add_argument("--excerpt-chars", type=int, default=500, help="Maximum excerpt length per sampled record.")
    parser.add_argument(
        "--format",
        choices=("markdown", "tsv"),
        default="markdown",
        help="Output format. Markdown is convenient for the writeup; TSV is convenient for manual labeling.",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output path. Defaults to stdout.")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    samples = sample_language_predictions(
        args.warc_path,
        num_examples=args.num_examples,
        seed=args.seed,
        limit=limit,
        min_chars=args.min_chars,
        excerpt_chars=args.excerpt_chars,
    )

    writer = write_markdown if args.format == "markdown" else write_tsv
    if args.output is None:
        import sys

        writer(samples, sys.stdout)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as output_file:
            writer(samples, output_file)


if __name__ == "__main__":
    main()

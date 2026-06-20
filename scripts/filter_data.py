from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path

from tqdm import tqdm

from cs336_data.common import get_shared_assets_path
from cs336_data.filter_data import FilterConfig, FilterStats, filter_wet_file, write_report


def _default_workers() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _find_wet_files(input_dir: Path, max_files: int | None) -> list[Path]:
    wet_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() or path.is_symlink()
        if path.name.endswith((".warc.wet.gz", ".wet.gz", ".gz"))
    )
    if max_files is not None:
        wet_files = wet_files[:max_files]
    return wet_files


def _filter_one(args: tuple[Path, Path, FilterConfig, Path | None, int, int | None]) -> tuple[Path, FilterStats]:
    input_path, output_path, config, discarded_path, max_discarded_examples, max_records = args
    stats = filter_wet_file(
        input_path,
        output_path,
        config=config,
        discarded_examples_path=discarded_path,
        max_discarded_examples=max_discarded_examples,
        max_records=max_records,
    )
    return output_path, stats


def _merge_shards(shard_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        for shard_path in shard_paths:
            with open(shard_path, encoding="utf-8") as shard_file:
                for line in shard_file:
                    output_file.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Common Crawl English WET files for language modeling.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=get_shared_assets_path() / "english-wet-data",
        help="Directory containing English-filtered Common Crawl WET .gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_shared_assets_path() / "filtered-lm-data",
        help="Directory for output shards and reports.",
    )
    parser.add_argument("--output-file", type=Path, default=None, help="Merged text output path.")
    parser.add_argument("--report-file", type=Path, default=None, help="JSON report path.")
    parser.add_argument("--workers", type=int, default=_default_workers(), help="Number of local worker processes.")
    parser.add_argument("--max-files", type=int, default=None, help="Only process the first N WET files.")
    parser.add_argument("--max-records-per-file", type=int, default=None, help="Only process the first N records per file.")
    parser.add_argument("--min-words", type=int, default=FilterConfig.min_words)
    parser.add_argument("--max-words", type=int, default=FilterConfig.max_words)
    parser.add_argument("--min-chars", type=int, default=FilterConfig.min_chars)
    parser.add_argument("--max-chars", type=int, default=FilterConfig.max_chars)
    parser.add_argument("--max-masked-pii", type=int, default=FilterConfig.max_masked_pii)
    parser.add_argument("--disable-gopher", action="store_true", help="Skip Gopher heuristic quality filtering.")
    parser.add_argument(
        "--use-quality-classifier",
        action="store_true",
        help="Use the fastText wiki/cc quality classifier. This loads the model in each worker.",
    )
    parser.add_argument("--quality-threshold", type=float, default=FilterConfig.quality_threshold)
    parser.add_argument(
        "--use-harmful-classifiers",
        action="store_true",
        help="Use the NSFW and toxic-speech fastText classifiers. This is slow and memory-heavy.",
    )
    parser.add_argument("--harmful-threshold", type=float, default=FilterConfig.harmful_threshold)
    parser.add_argument("--keep-metadata", action="store_true", help="Write JSONL with URL/domain metadata instead of text lines.")
    parser.add_argument(
        "--write-discarded-examples",
        action="store_true",
        help="Write a small JSONL sample of discarded documents per shard for manual inspection.",
    )
    parser.add_argument(
        "--max-discarded-examples-per-filter",
        type=int,
        default=5,
        help="Number of discarded examples to save for each filter in each shard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    shard_dir = output_dir / "shards"
    discarded_dir = output_dir / "discarded_examples"
    output_file = args.output_file or output_dir / ("filtered.jsonl" if args.keep_metadata else "filtered.txt")
    report_file = args.report_file or output_dir / "filter_report.json"

    wet_files = _find_wet_files(input_dir, args.max_files)
    if not wet_files:
        raise FileNotFoundError(f"No WET .gz files found in {input_dir}")

    config = FilterConfig(
        min_words=args.min_words,
        max_words=args.max_words,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_masked_pii=args.max_masked_pii,
        use_gopher=not args.disable_gopher,
        use_quality_classifier=args.use_quality_classifier,
        quality_threshold=args.quality_threshold,
        use_harmful_classifiers=args.use_harmful_classifiers,
        harmful_threshold=args.harmful_threshold,
        keep_metadata=args.keep_metadata,
    )

    shard_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for input_path in wet_files:
        shard_path = shard_dir / f"{input_path.name}.jsonl" if args.keep_metadata else shard_dir / f"{input_path.name}.txt"
        discarded_path = discarded_dir / f"{input_path.name}.discarded.jsonl" if args.write_discarded_examples else None
        tasks.append(
            (
                input_path,
                shard_path,
                config,
                discarded_path,
                args.max_discarded_examples_per_filter,
                args.max_records_per_file,
            )
        )

    total_stats = FilterStats()
    completed_shards: list[Path] = []
    workers = max(1, args.workers)
    if workers == 1:
        iterator = (_filter_one(task) for task in tasks)
        for shard_path, stats in tqdm(iterator, total=len(tasks), desc="Filtering WET files"):
            completed_shards.append(shard_path)
            total_stats.add(stats)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_filter_one, task) for task in tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Filtering WET files"):
                shard_path, stats = future.result()
                completed_shards.append(shard_path)
                total_stats.add(stats)

    completed_shards = sorted(completed_shards)
    _merge_shards(completed_shards, output_file)
    write_report(report_file, total_stats, config, wet_files, completed_shards)

    print(f"Wrote filtered data to {output_file}")
    print(f"Wrote report to {report_file}")
    print(f"Kept {total_stats.kept}/{total_stats.seen} documents ({total_stats.kept / total_stats.seen:.2%})")


if __name__ == "__main__":
    main()

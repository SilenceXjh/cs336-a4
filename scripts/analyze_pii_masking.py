from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from cs336_data.extract import iter_text_from_warc
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers

DEFAULT_WARC_PATH = Path("local-shared-data/CC/example.warc.gz")


@dataclass(frozen=True)
class PIISample:
    index: int
    url: str
    email_count: int
    phone_count: int
    ip_count: int
    masked_excerpt: str

    @property
    def total_count(self) -> int:
        return self.email_count + self.phone_count + self.ip_count


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def mask_all(text: str) -> tuple[str, int, int, int]:
    masked_text, email_count = mask_emails(text)
    masked_text, phone_count = mask_phone_numbers(masked_text)
    masked_text, ip_count = mask_ips(masked_text)
    return masked_text, email_count, phone_count, ip_count


def collect_pii_samples(
    warc_path: Path,
    *,
    limit: int | None,
    min_chars: int,
) -> list[PIISample]:
    samples: list[PIISample] = []
    for record_index, (url, text) in enumerate(iter_text_from_warc(warc_path, limit=limit), start=1):
        normalized_text = normalize_text(text)
        if len(normalized_text) < min_chars:
            continue

        masked_text, email_count, phone_count, ip_count = mask_all(normalized_text)
        if email_count + phone_count + ip_count == 0:
            continue

        samples.append(
            PIISample(
                index=record_index,
                url=url,
                email_count=email_count,
                phone_count=phone_count,
                ip_count=ip_count,
                masked_excerpt=masked_text[:600].replace("|", "\\|"),
            )
        )
    return samples


def write_markdown(samples: list[PIISample], total_candidates: int, output_file: TextIO) -> None:
    output_file.write("# PII Masking Sample\n\n")
    output_file.write(f"Documents with at least one replacement: {total_candidates}\n\n")
    output_file.write(f"Sampled documents shown: {len(samples)}\n\n")
    output_file.write("| WARC # | Emails | Phones | IPs | URL | Masked excerpt |\n")
    output_file.write("|---:|---:|---:|---:|---|---|\n")
    for sample in samples:
        output_file.write(
            f"| {sample.index} | {sample.email_count} | {sample.phone_count} | {sample.ip_count} | "
            f"{sample.url} | {sample.masked_excerpt} |\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample extracted WARC documents where email, phone, or IPv4 masking changed the text."
    )
    parser.add_argument("--warc-path", type=Path, default=DEFAULT_WARC_PATH)
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--limit", type=int, default=500, help="Use 0 to scan the whole WARC.")
    parser.add_argument("--min-chars", type=int, default=80)
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    candidates = collect_pii_samples(args.warc_path, limit=limit, min_chars=args.min_chars)
    rng = random.Random(args.seed)
    selected = rng.sample(candidates, min(args.num_examples, len(candidates)))
    write_markdown(selected, len(candidates), output_file=sys.stdout)


if __name__ == "__main__":
    main()

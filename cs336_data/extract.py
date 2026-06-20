from __future__ import annotations

import argparse
import gzip
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """Extract visible plain text from raw HTML bytes."""
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html = html_bytes.decode(encoding, errors="replace")

    return extract_plain_text(html)


def iter_text_from_warc(warc_path: Path, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """Yield (URL, extracted text) pairs from response records in a WARC file."""
    from warcio.archiveiterator import ArchiveIterator

    opener = gzip.open if warc_path.suffix == ".gz" else open
    count = 0
    with opener(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue

            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            html_bytes = record.content_stream().read()
            yield url, extract_text_from_html_bytes(html_bytes)

            count += 1
            if limit is not None and count >= limit:
                break


def _write_extracted_records(
    records: Iterator[tuple[str, str]],
    output_file: TextIO,
) -> None:
    for index, (url, text) in enumerate(records):
        if index:
            output_file.write("\n\n")
        output_file.write(f"===== RECORD {index} =====\n")
        output_file.write(f"URL: {url}\n\n")
        output_file.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract plain text from response records in a WARC file.")
    parser.add_argument("warc_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="Write extracted text to this file. Defaults to stdout.")
    parser.add_argument("--limit", type=int, help="Only extract the first N response records.")
    args = parser.parse_args()

    records = iter_text_from_warc(args.warc_path, limit=args.limit)
    if args.output is None:
        _write_extracted_records(records, sys.stdout)
    else:
        with args.output.open("w", encoding="utf-8") as output_file:
            _write_extracted_records(records, output_file)


if __name__ == "__main__":
    main()

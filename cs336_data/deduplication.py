from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import Counter
from collections import defaultdict
from pathlib import Path

import mmh3
from xopen import xopen


def _hash_line(line: bytes) -> bytes:
    return hashlib.blake2b(line, digest_size=16).digest()


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    line_counts: Counter[bytes] = Counter()
    for input_file in input_files:
        with xopen(input_file, "rb") as f:
            for line in f:
                line_counts[_hash_line(line)] += 1

    for input_file in input_files:
        input_path = Path(input_file)
        destination = output_path / input_path.name
        with xopen(input_file, "rb") as src, xopen(destination, "wb") as dst:
            for line in src:
                if line_counts[_hash_line(line)] == 1:
                    dst.write(line)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char) and not unicodedata.category(char).startswith("P")
    )
    return re.sub(r"\s+", " ", text).strip()


def _word_ngrams(text: str, ngram_length: int) -> set[str]:
    words = _normalize_text(text).split()
    if ngram_length <= 0:
        raise ValueError("ngram_length must be positive")
    if len(words) < ngram_length:
        return set()
    return {" ".join(words[i : i + ngram_length]) for i in range(len(words) - ngram_length + 1)}


def _minhash_signature(shingles: set[str], num_hashes: int) -> tuple[int, ...]:
    if num_hashes <= 0:
        raise ValueError("num_hashes must be positive")
    if not shingles:
        return tuple([2**64 - 1] * num_hashes)

    signature = []
    for seed in range(num_hashes):
        signature.append(min(mmh3.hash64(shingle, seed=seed, signed=False)[0] for shingle in shingles))
    return tuple(signature)


def _candidate_duplicate_pairs(signatures: list[tuple[int, ...]], num_bands: int) -> set[tuple[int, int]]:
    if num_bands <= 0:
        raise ValueError("num_bands must be positive")
    if not signatures:
        return set()
    if len(signatures[0]) % num_bands != 0:
        raise ValueError("num_hashes must be evenly divisible by num_bands")

    rows_per_band = len(signatures[0]) // num_bands
    buckets: defaultdict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for doc_idx, signature in enumerate(signatures):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            buckets[(band_idx, signature[start:end])].append(doc_idx)

    pairs: set[tuple[int, int]] = set()
    for bucket in buckets.values():
        for i, left in enumerate(bucket):
            for right in bucket[i + 1 :]:
                pairs.add((left, right))
    return pairs


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _read_text(path: Path) -> str:
    with xopen(path) as f:
        return f.read()


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    input_paths = [Path(path) for path in input_files]
    documents = [_read_text(path) for path in input_paths]
    shingle_sets = [_word_ngrams(document, ngrams) for document in documents]
    signatures = [_minhash_signature(shingles, num_hashes) for shingles in shingle_sets]

    union_find = _UnionFind(len(input_paths))
    for left, right in _candidate_duplicate_pairs(signatures, num_bands):
        if _jaccard_similarity(shingle_sets[left], shingle_sets[right]) >= jaccard_threshold:
            union_find.union(left, right)

    clusters: defaultdict[int, list[int]] = defaultdict(list)
    for doc_idx in range(len(input_paths)):
        clusters[union_find.find(doc_idx)].append(doc_idx)
    representatives = {
        min(members, key=lambda idx: (input_paths[idx].name, str(input_paths[idx])))
        for members in clusters.values()
    }

    for doc_idx, input_path in enumerate(input_paths):
        if doc_idx not in representatives:
            continue
        destination = output_path / input_path.name
        with xopen(destination, "w") as dst:
            dst.write(documents[doc_idx])

import gzip
import os
import random
import shutil
import tempfile
import time
import urllib.request
from functools import cached_property
from io import BytesIO
from pathlib import Path

from collections.abc import Callable
import modal
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from cs336_data.common import get_shared_assets_path
from cs336_data.langid import identify_language
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image
from furu import Furu

BASE_URL = "https://data.commoncrawl.org/"
ENGLISH_PROBABILITY_THRESHOLD = 0.7


def _urlretrieve_with_retries(url: str, filename: Path, *, attempts: int = 10) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            urllib.request.urlretrieve(url, filename)
            return
        except Exception as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(min(2**attempt, 30))
    assert last_error is not None
    raise last_error


def _is_english(text: str) -> bool:
    language, score = identify_language(text)
    return language == "en" and score >= ENGLISH_PROBABILITY_THRESHOLD



class _EnglishWetFile(Furu[Path]):
    chunk_urls: tuple[str, ...]

    def _create(self) -> Path:
        output_path = self.data_dir / "data.warc.wet.gz"

        self.logger.info("Loading English language identifier")
        is_english: Callable[[str], bool] = _is_english

        total_text = 0
        skipped_text = 0
        self.logger.info("Processing WET chunk (%d files)", len(self.chunk_urls))

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir="/tmp",
            suffix=f".{output_path.name}",
        ) as temp_output_file:
            temp_output_path = Path(temp_output_file.name)

        with gzip.open(temp_output_path, "wb") as output_stream:
            writer = WARCWriter(output_stream, gzip=False)
            for wet_url in self.chunk_urls:
                local_wet_path = Path("/tmp") / wet_url.split("/")[-1]
                if not local_wet_path.exists():
                    self.logger.info("Downloading %s to %s", wet_url, local_wet_path)
                    _urlretrieve_with_retries(wet_url, local_wet_path)
                else:
                    self.logger.info("Using cached WET file %s", local_wet_path)
                with gzip.open(local_wet_path, "rb") as input_stream:
                    for rec in ArchiveIterator(input_stream):
                        if rec.rec_type != "conversion":
                            writer.write_record(rec)
                            continue
                        payload = rec.content_stream().read()
                        text = payload.decode("utf-8", errors="replace")
                        total_text += len(text)

                        if is_english(text):
                            rec.raw_stream = BytesIO(payload)
                            writer.write_record(rec)
                        else:
                            skipped_text += len(text)
        shutil.copy2(temp_output_path, output_path)
        temp_output_path.unlink(missing_ok=True)

        self.logger.info(
            "Finished WET chunk: wrote %s, kept %.2f%% of text",
            output_path,
            100 * (total_text - skipped_text) / total_text if total_text else 0,
        )
        return output_path

    @cached_property
    def storage_root(self) -> Path:
        return get_shared_assets_path() / "furu"


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, max_containers=128)
def make_wet_file_on_modal(wet_file: _EnglishWetFile) -> Path:
    return wet_file.load_or_create()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


class EnglishWetFiles(Furu[list[Path]]):
    n_files: int = _env_int("CS336_ENGLISH_WET_N_FILES", 2500)
    group_size: int = _env_int("CS336_ENGLISH_WET_GROUP_SIZE", 4)
    shuffle_seed: int = 336
    crawl_id: str = "CC-MAIN-2026-17"

    def _create(self) -> list[Path]:
        explicit_wet_urls = os.environ.get("CS336_ENGLISH_WET_URLS")
        if explicit_wet_urls:
            wet_urls = [url.strip() for url in explicit_wet_urls.split(",") if url.strip()]
            self.logger.info("Using %d explicit WET URLs", len(wet_urls))
        else:
            assert self.n_files % self.group_size == 0
            wet_paths = f"{BASE_URL}crawl-data/{self.crawl_id}/wet.paths.gz"
            self.logger.info("Loading WET paths from %s", wet_paths)
            local_wet_paths = get_shared_assets_path() / "common-crawl" / self.crawl_id / "wet.paths.gz"
            _urlretrieve_with_retries(wet_paths, local_wet_paths)
            with gzip.open(local_wet_paths, "rt") as f:
                all_wet_paths = [line.strip() for line in f if line.strip()]
            selected_wet_paths = random.Random(self.shuffle_seed).sample(all_wet_paths, k=self.n_files)
            wet_urls = [BASE_URL + path for path in selected_wet_paths]

        self.logger.info("Selected %d WET files for crawl %s", len(wet_urls), self.crawl_id)

        wet_files: list[_EnglishWetFile] = []
        for chunk_idx in range(0, len(wet_urls), self.group_size):
            chunk_urls = tuple(wet_urls[chunk_idx : chunk_idx + self.group_size])
            wet_files.append(_EnglishWetFile(chunk_urls=chunk_urls))

        self.logger.info("Making %d english wet files", len(wet_files))

        wet_data_paths: list[Path] = []
        if modal.is_local():
            self.logger.info("downloading wet files locally")
            for wet_file_idx, wet_file in enumerate(wet_files):
                wet_data_paths.append(wet_file.load_or_create())
                self.logger.info(
                    "Completed %d/%d WET chunks",
                    wet_file_idx,
                    len(wet_files),
                )

            repo_path = get_shared_assets_path() / "english-wet-data"
            repo_path.mkdir(exist_ok=True)
            self.logger.info("Linking local WET outputs into %s", repo_path)
            for wet_data_idx, wet_data_path in enumerate(wet_data_paths):
                link_path = repo_path / f"{wet_data_idx:05d}-{wet_data_path.name}"
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(wet_data_path)
                self.logger.info("Linked WET chunk %d: %s -> %s", wet_data_idx, link_path, wet_data_path)
        else:
            self.logger.info("downloading wet files on remote")

            wet_data_paths = list(make_wet_file_on_modal.map(wet_files))
            self.logger.info("Completed %d remote WET chunks", len(wet_data_paths))

            repo_path = get_shared_assets_path() / "english-wet-data"
            repo_path.mkdir(exist_ok=False)
            self.logger.info("Linking remote WET outputs into %s", repo_path)

            source_link = repo_path / ".source"
            if source_link.exists() or source_link.is_symlink():
                self.logger.info("Replacing existing source link %s", source_link)
                source_link.unlink()
            source_link.symlink_to(self.data_dir)
            self.logger.info("Linked source data directory %s -> %s", source_link, self.data_dir)

            for wet_data_idx, wet_data_path in enumerate(wet_data_paths):
                link_path = repo_path / f"{wet_data_idx:05d}-{wet_data_path.name}"
                if link_path.exists() or link_path.is_symlink():
                    self.logger.info("Replacing existing WET chunk link %s", link_path)
                    link_path.unlink()
                link_path.symlink_to(wet_data_path)
                self.logger.info("Linked WET chunk %d: %s -> %s", wet_data_idx, link_path, wet_data_path)

        self.logger.info("Finished creating %d English WET files", len(wet_data_paths))
        return wet_data_paths

    @cached_property
    def storage_root(self) -> Path:
        return get_shared_assets_path() / "furu"

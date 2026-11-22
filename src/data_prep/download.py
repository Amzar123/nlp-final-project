"""Download raw datasets (plan step 0.2).

    python -m src.data_prep.download              # FEVER claims + wiki + ClaimBuster
    python -m src.data_prep.download --no-wiki    # skip the 1.7 GB wiki-pages.zip

Each file is streamed to `<name>.part` and renamed only after the byte count
matches Content-Length, so an interrupted download is never mistaken for a
finished one. Existing files are re-checked (zip central directory, last JSON
line, non-empty) and re-downloaded if they look truncated.
"""
import argparse
import http.client
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from ..paths import RAW_DIR

FEVER_MIRRORS = [
    "https://fever.ai/download/fever",
    # Listed as fallback in the plan. Returned HTTP 403 when checked on 2026-11-16.
    "https://s3-eu-west-1.amazonaws.com/fever.public",
]
FEVER_CLAIM_FILES = ["train.jsonl", "paper_dev.jsonl", "paper_test.jsonl"]
FEVER_WIKI_FILE = "wiki-pages.zip"

CLAIMBUSTER_BASE = "https://zenodo.org/records/3609356/files"
CLAIMBUSTER_FILES = ["groundtruth.csv", "crowdsourced.csv"]
CLAIMBUSTER_DIR = RAW_DIR / "claimbuster"

CHUNK_SIZE = 1 << 20
USER_AGENT = "claim-check-pipeline/0.1"


class DownloadError(RuntimeError):
    pass


class Downloader:
    """Streams each file to `<name>.part` and only renames it once the size and integrity checks pass."""

    @staticmethod
    def looks_complete(path):
        """Cheap truncation check that does not need the expected size."""
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if path.suffix == ".zip":
            # A truncated zip has no end-of-central-directory record.
            return zipfile.is_zipfile(path)
        if path.suffix == ".jsonl":
            return Downloader._last_line_is_json(path)
        return True

    @staticmethod
    def _last_line_is_json(path):
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(max(0, end - CHUNK_SIZE))
            tail = f.read().rstrip(b"\n")
        last = tail.rsplit(b"\n", 1)[-1]
        try:
            json.loads(last)
        except ValueError:
            return False
        return True

    @classmethod
    def _stream_to_file(cls, url, dest):
        from tqdm import tqdm

        part = dest.with_name(dest.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            expected = response.headers.get("Content-Length")
            expected = int(expected) if expected is not None else None
            written = 0
            with open(part, "wb") as out, tqdm(
                total=expected, unit="B", unit_scale=True, desc=dest.name
            ) as bar:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    bar.update(len(chunk))

        if written == 0:
            part.unlink(missing_ok=True)
            raise DownloadError("empty response body")
        if expected is not None and written != expected:
            raise DownloadError(f"got {written} bytes, expected {expected}. Partial file kept at {part}")
        part.replace(dest)
        if not cls.looks_complete(dest):
            raise DownloadError(f"{dest} was downloaded but fails the integrity check")
        return dest

    @classmethod
    def download(cls, urls, dest, force=False):
        """Try each URL in order. Raise DownloadError if all of them fail."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and not force:
            if cls.looks_complete(dest):
                return dest
            print(f"{dest} looks truncated, downloading again")

        errors = []
        for url in urls:
            print(f"Download {url} -> {dest}")
            try:
                return cls._stream_to_file(url, dest)
            except (urllib.error.URLError, http.client.HTTPException, OSError, DownloadError) as e:
                errors.append(f"  {url}: {e}")
                print(f"  failed: {e}")

        raise DownloadError(f"Could not download {dest.name}:\n" + "\n".join(errors))

    @classmethod
    def fever(cls, filename, dest_dir=RAW_DIR, force=False):
        urls = [f"{base}/{filename}" for base in FEVER_MIRRORS]
        return cls.download(urls, Path(dest_dir) / filename, force=force)

    @classmethod
    def claimbuster(cls, filename, dest_dir=CLAIMBUSTER_DIR, force=False):
        urls = [f"{CLAIMBUSTER_BASE}/{filename}?download=1"]
        return cls.download(urls, Path(dest_dir) / filename, force=force)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-wiki", action="store_true", help="skip wiki-pages.zip (1.7 GB)")
    parser.add_argument("--no-claimbuster", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download even if the file looks complete")
    args = parser.parse_args()

    jobs = [lambda f=f: Downloader.fever(f, force=args.force) for f in FEVER_CLAIM_FILES]
    if not args.no_wiki:
        jobs.append(lambda: Downloader.fever(FEVER_WIKI_FILE, force=args.force))
    if not args.no_claimbuster:
        jobs += [lambda f=f: Downloader.claimbuster(f, force=args.force) for f in CLAIMBUSTER_FILES]

    try:
        for job in jobs:
            path = job()
            print(f"OK {path} ({path.stat().st_size:,} bytes)")
    except DownloadError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()

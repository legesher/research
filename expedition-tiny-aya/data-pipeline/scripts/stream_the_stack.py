#!/usr/bin/env python3
"""Stream and filter Python files from bigcode/the-stack.

This script streams records from Hugging Face, filters for syntactically valid
Python source with permissive licenses, rejects exact-content duplicates via a
persistent SHA-256 index, and writes accepted samples plus metadata to disk.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


PERMISSIVE_LICENSES = {
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "mit": "MIT",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd 2-clause": "BSD-2-Clause",
    "bsd 2 clause": "BSD-2-Clause",
    'bsd-2-clause "simplified"': "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd 3 clause": "BSD-3-Clause",
    'bsd-3-clause "new" or "revised"': "BSD-3-Clause",
    "isc": "ISC",
    "unlicense": "Unlicense",
}

REJECT_LICENSE_PATTERNS = (
    "gpl",
    "lgpl",
    "agpl",
    "proprietary",
    "commercial",
    "unknown",
    "other",
    "no license",
)

CSV_HEADERS = [
    "record_index",
    "sha256",
    "repo_name",
    "file_path",
    "size",
    "license",
    "stars",
    "line_count",
    "code_file",
    "metadata_file",
]


@dataclass
class CandidateRecord:
    repo_name: str
    file_path: str
    content: str
    size: int
    license_name: str
    stars: int
    line_count: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and filter permissively licensed Python from bigcode/the-stack."
    )
    parser.add_argument("--dataset", default="bigcode/the-stack")
    parser.add_argument("--data-dir", default="data/python")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("expedition-tiny-aya/data-pipeline/output/the-stack-python"),
    )
    parser.add_argument("--min-lines", type=int, default=20)
    parser.add_argument("--max-lines", type=int, default=200)
    parser.add_argument("--min-stars", type=int, default=21)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def ensure_output_layout(output_dir: Path) -> dict[str, Path]:
    files_dir = output_dir / "files"
    metadata_dir = output_dir / "metadata"
    files_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return {
        "files": files_dir,
        "metadata": metadata_dir,
        "progress": output_dir / "progress.json",
        "csv": output_dir / "metadata.csv",
        "db": output_dir / "state.sqlite3",
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "last_index": 0,
            "processed": 0,
            "accepted": 0,
            "rejected": 0,
            "duplicates": 0,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_hashes (
            sha256 TEXT PRIMARY KEY
        )
        """
    )
    conn.commit()
    return conn


def ensure_csv(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writerow(row)


def safe_slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or fallback


def get_first(record: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        try:
            return int(float(cleaned))
        except ValueError:
            return default
    return default


def flatten_licenses(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        flattened: list[str] = []
        for nested in value.values():
            flattened.extend(flatten_licenses(nested))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(flatten_licenses(item))
        return flattened
    return [str(value)]


def normalize_license(value: Any) -> str | None:
    for raw_license in flatten_licenses(value):
        normalized = raw_license.strip().lower()
        if not normalized:
            continue
        if any(pattern in normalized for pattern in REJECT_LICENSE_PATTERNS):
            continue
        if normalized in PERMISSIVE_LICENSES:
            return PERMISSIVE_LICENSES[normalized]
        compact = normalized.replace("_", "-")
        if compact in PERMISSIVE_LICENSES:
            return PERMISSIVE_LICENSES[compact]
    return None


def count_lines(content: str) -> int:
    if not content:
        return 0
    return len(content.splitlines())


def is_valid_python(content: str) -> bool:
    try:
        ast.parse(content)
    except SyntaxError:
        return False
    return True


def build_candidate(record: dict[str, Any], min_lines: int, max_lines: int, min_stars: int) -> CandidateRecord | None:
    content = get_first(record, ("content", "code", "text"))
    if not isinstance(content, str) or not content.strip():
        return None

    line_count = count_lines(content)
    if line_count < min_lines or line_count > max_lines:
        return None

    if not is_valid_python(content):
        return None

    stars = to_int(get_first(record, ("max_stars_count", "stars", "star_count")))
    if stars < min_stars:
        return None

    license_name = normalize_license(
        get_first(
            record,
            (
                "max_stars_repo_licenses",
                "license",
                "licenses",
                "max_stars_repo_license",
            ),
        )
    )
    if not license_name:
        return None

    repo_name = str(
        get_first(record, ("max_stars_repo_name", "repo_name", "repository_name"), default="unknown-repo")
    )
    file_path = str(get_first(record, ("max_stars_repo_path", "path", "file_path"), default="unknown.py"))
    size = to_int(get_first(record, ("size", "bytes", "file_size")), default=len(content.encode("utf-8")))
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return CandidateRecord(
        repo_name=repo_name,
        file_path=file_path,
        content=content,
        size=size,
        license_name=license_name,
        stars=stars,
        line_count=line_count,
        sha256=sha256,
    )


def hash_exists(conn: sqlite3.Connection, sha256: str) -> bool:
    row = conn.execute("SELECT 1 FROM seen_hashes WHERE sha256 = ?", (sha256,)).fetchone()
    return row is not None


def remember_hash(conn: sqlite3.Connection, sha256: str) -> None:
    conn.execute("INSERT OR IGNORE INTO seen_hashes (sha256) VALUES (?)", (sha256,))
    conn.commit()


def write_record(record_index: int, candidate: CandidateRecord, paths: dict[str, Path]) -> None:
    repo_slug = safe_slug(candidate.repo_name, "repo")
    file_slug = safe_slug(Path(candidate.file_path).stem, "file")
    file_stem = f"{record_index:09d}_{repo_slug}_{file_slug}_{candidate.sha256[:12]}"

    code_path = paths["files"] / f"{file_stem}.py"
    metadata_path = paths["metadata"] / f"{file_stem}.json"

    code_path.write_text(candidate.content, encoding="utf-8")

    metadata = {
        "record_index": record_index,
        "sha256": candidate.sha256,
        "repo_name": candidate.repo_name,
        "file_path": candidate.file_path,
        "size": candidate.size,
        "license": candidate.license_name,
        "stars": candidate.stars,
        "line_count": candidate.line_count,
        "code_file": str(code_path),
        "metadata_file": str(metadata_path),
    }
    write_json_atomic(metadata_path, metadata)
    append_csv_row(paths["csv"], metadata)


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    write_json_atomic(path, progress)


def main() -> int:
    args = parse_args()
    paths = ensure_output_layout(args.output_dir)
    ensure_csv(paths["csv"])
    progress = load_progress(paths["progress"])
    conn = init_db(paths["db"])

    dataset = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        streaming=True,
        token=args.hf_token,
    )

    start_index = to_int(progress.get("last_index"), default=0)
    if start_index:
        dataset = dataset.skip(start_index)

    processed = to_int(progress.get("processed"))
    accepted = to_int(progress.get("accepted"))
    rejected = to_int(progress.get("rejected"))
    duplicates = to_int(progress.get("duplicates"))

    current_index = start_index
    try:
        for record in dataset:
            current_index += 1
            processed += 1

            candidate = build_candidate(
                record,
                min_lines=args.min_lines,
                max_lines=args.max_lines,
                min_stars=args.min_stars,
            )
            if candidate is None:
                rejected += 1
            elif hash_exists(conn, candidate.sha256):
                duplicates += 1
            else:
                write_record(current_index, candidate, paths)
                remember_hash(conn, candidate.sha256)
                accepted += 1

            if processed % args.checkpoint_every == 0:
                save_progress(
                    paths["progress"],
                    {
                        "last_index": current_index,
                        "processed": processed,
                        "accepted": accepted,
                        "rejected": rejected,
                        "duplicates": duplicates,
                    },
                )

            if args.limit is not None and accepted >= args.limit:
                break
    except KeyboardInterrupt:
        print("Interrupted. Saving progress.", file=sys.stderr)
    finally:
        save_progress(
            paths["progress"],
            {
                "last_index": current_index,
                "processed": processed,
                "accepted": accepted,
                "rejected": rejected,
                "duplicates": duplicates,
            },
        )
        conn.close()

    print(
        json.dumps(
            {
                "last_index": current_index,
                "processed": processed,
                "accepted": accepted,
                "rejected": rejected,
                "duplicates": duplicates,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

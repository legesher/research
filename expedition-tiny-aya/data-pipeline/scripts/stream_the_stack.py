#!/usr/bin/env python3
"""Stream and filter Python files from bigcode/the-stack.

This script streams records from Hugging Face, filters for syntactically valid
Python source with permissive licenses, rejects exact and near-duplicate code,
and writes accepted samples plus metadata to disk.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import pickle
import re
import sqlite3
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasketch import MinHash, MinHashLSH
from datasets import load_dataset

# Parameters (standard from StarCoder 2 / Lee et al. 2022)
NUM_PERM = 128
THRESHOLD = 0.7
SHINGLE_SIZE = 5

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


@dataclass
class NearDuplicateMatch:
    sha256: str
    record_index: int
    metadata_path: str
    cluster_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and filter permissively licensed Python from bigcode/the-stack."
    )
    parser.add_argument("--dataset", default="bigcode/the-stack-dedup")
    parser.add_argument("--data-dir", default="data/python")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("expedition-tiny-aya/data-pipeline/output/the-stack-python"),
    )
    parser.add_argument("--min-lines", type=int, default=10)
    parser.add_argument("--max-lines", type=int, default=1000)
    parser.add_argument("--min-stars", type=int, default=21)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-tests",
        action="store_true",
        default=False,
        help="Keep test files (test_*, tests/) instead of rejecting them",
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
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
        "near_duplicates": output_dir / "near_duplicate_clusters.jsonl",
        "stats": output_dir / "filter_stats.json",
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "last_index": 0,
            "processed": 0,
            "accepted": 0,
            "rejected": 0,
            "duplicates": 0,
            "near_duplicates": 0,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_hashes (
            sha256 TEXT PRIMARY KEY
        )
        """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS accepted_minhashes (
            sha256 TEXT PRIMARY KEY,
            record_index INTEGER NOT NULL,
            cluster_id TEXT NOT NULL,
            metadata_path TEXT NOT NULL,
            minhash BLOB NOT NULL
        )
        """)
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
        writer.writerow({header: row.get(header, "") for header in CSV_HEADERS})


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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
    except (SyntaxError, UnicodeDecodeError):
        return False
    return True


def tokenize_code(content: str) -> list[str]:
    tokens: list[str] = []
    ignored_types = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
    }
    for token_info in tokenize.generate_tokens(io.StringIO(content).readline):
        if token_info.type in ignored_types:
            continue
        token = token_info.string.strip()
        if token:
            tokens.append(token)
    return tokens


def shingle_tokens(tokens: list[str], size: int = SHINGLE_SIZE) -> list[str]:
    if not tokens:
        return []
    if len(tokens) < size:
        return [" ".join(tokens)]
    return [" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def build_minhash(content: str) -> MinHash:
    minhash = MinHash(num_perm=NUM_PERM)
    for shingle in shingle_tokens(tokenize_code(content)):
        minhash.update(shingle.encode("utf-8"))
    return minhash


def is_autogenerated(content: str) -> bool:
    lines = content.lstrip().splitlines()
    first_lines = lines[:5]

    for line in first_lines:
        text = line.strip().lower()

        if text.startswith("# generated by"):
            return True

        if text.startswith("# auto-generated"):
            return True

        if text.startswith("# this file is automatically generated"):
            return True

        if text.startswith("# do not edit"):
            return True

    return False


def is_mostly_comments(content: str, threshold: float = 0.8) -> bool:
    lines = content.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return True

    comment_lines = 0
    for line in lines:
        stripped_line = line.strip()

        if stripped_line == "" or stripped_line.startswith("#"):
            comment_lines += 1

    ratio = comment_lines / total_lines
    return ratio > threshold


def is_test_file(file_path: str) -> bool:
    path = Path(file_path)
    filename = path.stem

    if filename.startswith(("test_", "Test_")):
        return True

    for part in path.parts:
        if part == "test" or part == "tests":
            return True

    return False


def has_no_definitions(content: str) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
    return True


def build_candidate(
    record: dict[str, Any],
    min_lines: int,
    max_lines: int,
    min_stars: int,
    include_tests: bool = False,
) -> tuple[CandidateRecord | None, str | None]:
    content = get_first(record, ("content", "code", "text"))
    if not isinstance(content, str) or not content.strip():
        return None, "failed_parse"

    line_count = count_lines(content)
    if line_count < min_lines or line_count > max_lines:
        return None, "failed_length"

    if not is_valid_python(content):
        return None, "failed_parse"

    stars = to_int(get_first(record, ("max_stars_count", "stars", "star_count")))
    if stars < min_stars:
        return None, "failed_stars"

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
        return None, "failed_license"

    repo_name = str(
        get_first(
            record,
            ("max_stars_repo_name", "repo_name", "repository_name"),
            default="unknown-repo",
        )
    )
    file_path = str(
        get_first(
            record, ("max_stars_repo_path", "path", "file_path"), default="unknown.py"
        )
    )
    size = to_int(
        get_first(record, ("size", "bytes", "file_size")),
        default=len(content.encode("utf-8")),
    )
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Quality heuristics — each returns a specific reason for granular stats.
    # NOTE: is_mostly_comments() intentionally excluded from the filter chain.
    # Files with heavy native-language comments are valuable training signal
    # for multilingual code models. The has_no_definitions() check already
    # catches files that are pure comments with no real code.
    # Re-enable with: if is_mostly_comments(content): return None, "failed_mostly_comments"
    if is_autogenerated(content):
        return None, "failed_autogenerated"
    if has_no_definitions(content):
        return None, "failed_no_defs"
    if not include_tests and is_test_file(file_path):
        return None, "failed_test_file"

    return (
        CandidateRecord(
            repo_name=repo_name,
            file_path=file_path,
            content=content,
            size=size,
            license_name=license_name,
            stars=stars,
            line_count=line_count,
            sha256=sha256,
        ),
        None,
    )


def hash_exists(conn: sqlite3.Connection, sha256: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM seen_hashes WHERE sha256 = ?", (sha256,)
    ).fetchone()
    return row is not None


def remember_hash(conn: sqlite3.Connection, sha256: str) -> None:
    conn.execute("INSERT OR IGNORE INTO seen_hashes (sha256) VALUES (?)", (sha256,))
    conn.commit()


def load_lsh_state(
    conn: sqlite3.Connection,
    metadata_dir: Path,
) -> tuple[MinHashLSH, dict[str, NearDuplicateMatch]]:
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    accepted: dict[str, NearDuplicateMatch] = {}
    rows = conn.execute(
        "SELECT sha256, record_index, metadata_path, cluster_id, minhash FROM accepted_minhashes"
    ).fetchall()
    if not rows:
        backfill_lsh_state(conn, metadata_dir)
        rows = conn.execute(
            "SELECT sha256, record_index, metadata_path, cluster_id, minhash FROM accepted_minhashes"
        ).fetchall()
    for sha256, record_index, metadata_path, cluster_id, minhash_blob in rows:
        minhash = pickle.loads(minhash_blob)
        lsh.insert(sha256, minhash)
        accepted[sha256] = NearDuplicateMatch(
            sha256=sha256,
            record_index=record_index,
            metadata_path=metadata_path,
            cluster_id=cluster_id,
        )
    return lsh, accepted


def remember_minhash(
    conn: sqlite3.Connection,
    candidate: CandidateRecord,
    record_index: int,
    metadata_path: Path,
    minhash: MinHash,
    cluster_id: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO accepted_minhashes (
            sha256, record_index, cluster_id, metadata_path, minhash
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            candidate.sha256,
            record_index,
            cluster_id,
            str(metadata_path),
            pickle.dumps(minhash, protocol=pickle.HIGHEST_PROTOCOL),
        ),
    )
    conn.commit()


def backfill_lsh_state(conn: sqlite3.Connection, metadata_dir: Path) -> None:
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        code_file = metadata.get("code_file")
        sha256 = metadata.get("sha256")
        record_index = metadata.get("record_index")
        if not code_file or not sha256 or record_index is None:
            continue
        code_path = Path(code_file)
        if not code_path.exists():
            continue
        content = code_path.read_text(encoding="utf-8")
        remember_minhash(
            conn,
            CandidateRecord(
                repo_name=str(metadata.get("repo_name", "unknown-repo")),
                file_path=str(metadata.get("file_path", "unknown.py")),
                content=content,
                size=to_int(metadata.get("size")),
                license_name=str(metadata.get("license", "")),
                stars=to_int(metadata.get("stars")),
                line_count=to_int(metadata.get("line_count")),
                sha256=str(sha256),
            ),
            record_index=to_int(record_index),
            metadata_path=metadata_path,
            minhash=build_minhash(content),
            cluster_id=str(metadata.get("near_duplicate_cluster_id", sha256)),
        )


def select_cluster_matches(
    accepted: dict[str, NearDuplicateMatch],
    match_ids: list[str],
) -> list[NearDuplicateMatch]:
    matches = [accepted[match_id] for match_id in match_ids if match_id in accepted]
    return sorted(matches, key=lambda match: (match.record_index, match.sha256))


def update_cluster_metadata(metadata_path: Path, duplicate_summary: dict[str, Any]) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["near_duplicate_cluster_size"] = (
        int(metadata.get("near_duplicate_cluster_size", 1)) + 1
    )
    cluster_members = metadata.setdefault("near_duplicate_cluster_members", [])
    cluster_members.append(duplicate_summary)
    write_json_atomic(metadata_path, metadata)


def write_record(
    record_index: int, candidate: CandidateRecord, paths: dict[str, Path]
) -> Path:
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
        "near_duplicate_cluster_id": candidate.sha256,
        "near_duplicate_cluster_size": 1,
        "near_duplicate_cluster_members": [],
        "code_file": str(code_path),
        "metadata_file": str(metadata_path),
    }
    write_json_atomic(metadata_path, metadata)
    append_csv_row(paths["csv"], metadata)
    return metadata_path


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    write_json_atomic(path, progress)


def main() -> int:
    args = parse_args()
    paths = ensure_output_layout(args.output_dir)
    ensure_csv(paths["csv"])
    progress = load_progress(paths["progress"])
    conn = init_db(paths["db"])
    lsh, accepted_minhashes = load_lsh_state(conn, paths["metadata"])

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
    duplicates = to_int(progress.get("duplicates"))
    near_duplicates = to_int(progress.get("near_duplicates"))
    failed_parse = to_int(progress.get("failed_parse"))
    failed_length = to_int(progress.get("failed_length"))
    failed_stars = to_int(progress.get("failed_stars"))
    failed_license = to_int(progress.get("failed_license"))
    failed_autogenerated = to_int(progress.get("failed_autogenerated"))
    failed_no_defs = to_int(progress.get("failed_no_defs"))
    failed_test_file = to_int(progress.get("failed_test_file"))

    current_index = start_index
    try:
        for record in dataset:
            current_index += 1
            processed += 1

            candidate, reason = build_candidate(
                record,
                min_lines=args.min_lines,
                max_lines=args.max_lines,
                min_stars=args.min_stars,
                include_tests=args.include_tests,
            )
            if candidate is None:
                if reason == "failed_parse":
                    failed_parse += 1
                elif reason == "failed_length":
                    failed_length += 1
                elif reason == "failed_stars":
                    failed_stars += 1
                elif reason == "failed_license":
                    failed_license += 1
                elif reason == "failed_autogenerated":
                    failed_autogenerated += 1
                elif reason == "failed_no_defs":
                    failed_no_defs += 1
                elif reason == "failed_test_file":
                    failed_test_file += 1
            elif hash_exists(conn, candidate.sha256):
                duplicates += 1
            else:
                minhash = build_minhash(candidate.content)
                near_duplicate_ids = lsh.query(minhash)
                near_duplicate_matches = select_cluster_matches(
                    accepted_minhashes, near_duplicate_ids
                )
                if near_duplicate_matches:
                    near_duplicates += 1
                    duplicate_summary = {
                        "record_index": current_index,
                        "sha256": candidate.sha256,
                        "repo_name": candidate.repo_name,
                        "file_path": candidate.file_path,
                    }
                    for match in near_duplicate_matches:
                        update_cluster_metadata(
                            Path(match.metadata_path), duplicate_summary
                        )
                    append_jsonl(
                        paths["near_duplicates"],
                        {
                            "record_index": current_index,
                            "sha256": candidate.sha256,
                            "repo_name": candidate.repo_name,
                            "file_path": candidate.file_path,
                            "matched_clusters": [
                                {
                                    "cluster_id": match.cluster_id,
                                    "sha256": match.sha256,
                                    "record_index": match.record_index,
                                    "metadata_path": match.metadata_path,
                                }
                                for match in near_duplicate_matches
                            ],
                        },
                    )
                else:
                    metadata_path = write_record(current_index, candidate, paths)
                    remember_hash(conn, candidate.sha256)
                    remember_minhash(
                        conn,
                        candidate,
                        current_index,
                        metadata_path,
                        minhash,
                        candidate.sha256,
                    )
                    lsh.insert(candidate.sha256, minhash)
                    accepted_minhashes[candidate.sha256] = NearDuplicateMatch(
                        sha256=candidate.sha256,
                        record_index=current_index,
                        metadata_path=str(metadata_path),
                        cluster_id=candidate.sha256,
                    )
                    accepted += 1

            if processed % args.checkpoint_every == 0:
                save_progress(
                    paths["progress"],
                    {
                        "last_index": current_index,
                        "processed": processed,
                        "accepted": accepted,
                        "duplicates": duplicates,
                        "near_duplicates": near_duplicates,
                        "failed_parse": failed_parse,
                        "failed_length": failed_length,
                        "failed_stars": failed_stars,
                        "failed_license": failed_license,
                        "failed_autogenerated": failed_autogenerated,
                        "failed_no_defs": failed_no_defs,
                        "failed_test_file": failed_test_file,
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
                "duplicates": duplicates,
                "near_duplicates": near_duplicates,
                "failed_parse": failed_parse,
                "failed_length": failed_length,
                "failed_stars": failed_stars,
                "failed_license": failed_license,
                "failed_autogenerated": failed_autogenerated,
                "failed_no_defs": failed_no_defs,
                "failed_test_file": failed_test_file,
            },
        )
        conn.close()
    write_json_atomic(
        paths["stats"],
        {
            "processed": processed,
            "accepted": accepted,
            "duplicates": duplicates,
            "near_duplicates": near_duplicates,
            "failed_parse": failed_parse,
            "failed_length": failed_length,
            "failed_stars": failed_stars,
            "failed_license": failed_license,
            "failed_autogenerated": failed_autogenerated,
            "failed_no_defs": failed_no_defs,
            "failed_test_file": failed_test_file,
        },
    )
    print(
        json.dumps(
            {
                "last_index": current_index,
                "processed": processed,
                "accepted": accepted,
                "duplicates": duplicates,
                "near_duplicates": near_duplicates,
                "failed_parse": failed_parse,
                "failed_length": failed_length,
                "failed_stars": failed_stars,
                "failed_license": failed_license,
                "failed_autogenerated": failed_autogenerated,
                "failed_no_defs": failed_no_defs,
                "failed_test_file": failed_test_file,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reparse every Phase-3 session on `legesher/language-decoded-experiments`
and upload `_summary_reparsed_{template}.json` siblings next to the originals.

Each session under `phase3/conditions/<condition>/seed<seed>/` gets:

- inputs:       <prefix>_results_template{1,2}.json     (Kaggle's raw outputs)
- inputs:       <prefix>_summary_template{1,2}.json     (Kaggle's strict accs)
- new outputs:  <prefix>_summary_reparsed_template{1,2}.json  ← this script

The originals are never touched. The reparsed siblings let any analysis
downstream pick strict-vs-lenient per-cell deltas straight off HF without
re-running inference.

Workflow:

1. List every session folder under `phase3/conditions/` via the HF API.
2. For each session × template, hf_hub_download the `_results_*.json` to a
   local cache (parallelised with a small thread pool).
3. Run reparse_results' build_reparsed_summary against each downloaded file.
4. Stage the resulting `_summary_reparsed_*.json` files in an HfApi.create_commit
   batch and push as a single discussion-PR.

Usage:

    # Plan-only — list what would be processed (no downloads or uploads):
    python upload_reparsed_summaries.py --dry-run

    # Full run — creates one HF discussion PR with all new summaries:
    python upload_reparsed_summaries.py

    # Limit to a single session (handy for spot-checking):
    python upload_reparsed_summaries.py --only condition-2-es-5k/seed42

    # Re-run only sessions that don't already have a reparsed sibling:
    python upload_reparsed_summaries.py --skip-existing

Auth:
    Reads from HF auth cache (huggingface-cli login). Token must have WRITE
    scope on the `legesher/language-decoded-experiments` dataset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ID = "legesher/language-decoded-experiments"
REPO_TYPE = "dataset"
PHASE3_ROOT = "phase3/conditions"
HF_TIMEOUT = 30  # seconds — applies to the JSON tree API; downloads have their own
HF_TREE_RETRIES = 3  # transient connection resets during multi-call session walks
MAX_DOWNLOAD_WORKERS = 4  # ThreadPoolExecutor concurrency for hf_hub_download


def _basename(remote_path: str) -> str:
    """Last path segment of a forward-slash HF path."""
    return remote_path.split("/")[-1]


def _hf_tree(path_in_repo: str) -> list[dict]:
    """List one directory level under `path_in_repo` on the main branch.

    Wraps the HF REST tree endpoint with a connection timeout, a context
    manager, and a small exponential-backoff retry to absorb the intermittent
    connection-reset errors that show up when walking ~30 paths back-to-back.
    Raises SystemExit with the offending URL after `HF_TREE_RETRIES` failures."""
    url = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main/{path_in_repo}"
    last_err: Exception | None = None
    for attempt in range(HF_TREE_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=HF_TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # 4xx/5xx HTTP errors: don't retry 4xx (client error), but 5xx is
            # worth a couple attempts. Conservative: retry on 500/502/503/504,
            # bail immediately on anything else.
            if e.code not in (500, 502, 503, 504):
                raise SystemExit(
                    f"HF tree API failed for {path_in_repo!r}: HTTP {e.code} ({url})"
                )
            last_err = e
        except urllib.error.URLError as e:
            # Transient connection errors (ECONNRESET, timeout, DNS hiccup).
            last_err = e
        if attempt + 1 < HF_TREE_RETRIES:
            time.sleep(2**attempt)  # 1s, 2s, 4s
    raise SystemExit(
        f"HF tree API failed for {path_in_repo!r} after {HF_TREE_RETRIES} attempts: "
        f"{last_err} ({url})"
    )


def list_sessions() -> list[dict]:
    """Walk `phase3/conditions/<cond>/seed<N>/` and classify each file.

    Returns a list of dicts with keys:
      - `session` (str): the session folder path, e.g. 'phase3/conditions/cond-2-ur-5k/seed42'
      - `canonical_results` (list[str]): full HF paths of result files that
        belong to this session (basename matches `<cond>_<seed>_results_*`)
      - `stray_results` (list[str]): full HF paths of result files in this
        folder whose basename does NOT match the folder's seed — usually
        accidental double-uploads
      - `existing_reparsed` (list[str]): basenames of `_summary_reparsed_*.json`
        files already present in the folder (used by `--skip-existing`)
    """
    conds = [x["path"] for x in _hf_tree(PHASE3_ROOT) if x["type"] == "directory"]
    sessions: list[dict] = []
    for cp in sorted(conds):
        for s in _hf_tree(cp):
            if s["type"] != "directory":
                continue
            files = [f for f in _hf_tree(s["path"]) if f["type"] == "file"]
            cond, seed_folder = s["path"].split("/")[-2:]
            expected_seed_token = (
                "seednone" if seed_folder == "seednone" else seed_folder
            )
            filename_prefix = f"{cond}_{expected_seed_token}_results_"

            canonical_results = sorted(
                f["path"]
                for f in files
                if "_results_" in f["path"]
                and _basename(f["path"]).startswith(filename_prefix)
            )
            stray = sorted(
                f["path"]
                for f in files
                if "_results_" in f["path"]
                and not _basename(f["path"]).startswith(filename_prefix)
            )
            existing_reparsed = sorted(
                _basename(f["path"]) for f in files if "_summary_reparsed_" in f["path"]
            )
            sessions.append(
                {
                    "session": s["path"],
                    "canonical_results": canonical_results,
                    "stray_results": stray,
                    "existing_reparsed": existing_reparsed,
                }
            )
    return sessions


def filter_skip_existing(sessions: list[dict]) -> tuple[list[dict], int]:
    """Drop canonical results from each session whose reparsed sibling
    already exists on HF. Returns (mutated_sessions, n_skipped)."""
    # Defer import — only loaded when this function is reached, after the
    # caller has done preflight via reparse_results.verify_extractor_source().
    from reparse_results import reparsed_summary_path_remote

    n_skipped = 0
    filtered: list[dict] = []
    for s in sessions:
        existing = set(s["existing_reparsed"])
        kept = []
        for remote_results in s["canonical_results"]:
            expected_sibling = _basename(reparsed_summary_path_remote(remote_results))
            if expected_sibling in existing:
                n_skipped += 1
            else:
                kept.append(remote_results)
        filtered.append({**s, "canonical_results": kept})
    return filtered, n_skipped


def _download_and_reparse(
    remote_results_path: str, cache_dir: Path
) -> tuple[str, Path, dict]:
    """Worker function suitable for a thread pool.

    Downloads one `_results_*.json` from HF, reparses it, and writes the
    resulting summary body to a sibling file in `cache_dir`. Returns
    (remote upload path, local temp path, body dict)."""
    from huggingface_hub import hf_hub_download

    from reparse_results import (
        build_reparsed_summary,
        reparse_file,
        reparsed_summary_path_remote,
    )

    local_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_results_path,
            repo_type=REPO_TYPE,
            cache_dir=str(cache_dir),
        )
    )
    rows = reparse_file(local_path, only=None)
    body = build_reparsed_summary(Path(remote_results_path), rows)
    out_remote = reparsed_summary_path_remote(remote_results_path)
    tmp_out = cache_dir / _basename(out_remote)
    tmp_out.write_text(json.dumps(body, indent=2, ensure_ascii=False))
    return out_remote, tmp_out, body


def _print_plan(sessions: list[dict], skipped_existing: int = 0) -> int:
    """Print the planned work and return the total reparse count."""
    total = 0
    print("Plan:")
    for s in sessions:
        n_canonical = len(s["canonical_results"])
        n_stray = len(s["stray_results"])
        flags = []
        if n_stray:
            flags.append(f"skipping {n_stray} stray")
        if s["existing_reparsed"]:
            flags.append(
                f"{len(s['existing_reparsed'])} reparsed sibling(s) already on HF"
            )
        flag = f" ({'; '.join(flags)})" if flags else ""
        print(f"  {s['session']}: {n_canonical} canonical results{flag}")
        for r in s["stray_results"]:
            print(f"    ✗ stray: {_basename(r)}")
        total += n_canonical
    if skipped_existing:
        print(
            f"  (skipped {skipped_existing} canonical files because their reparsed "
            f"siblings already exist on HF — pass without --skip-existing to overwrite)"
        )
    print()
    print(f"Total reparses to perform: {total}")
    print(f"Total new HF files to upload: {total}")
    return total


def main() -> None:
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit. Still calls the HF tree API to "
        "enumerate sessions, but does not download, reparse, or upload anything.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Limit to a single session, e.g., 'condition-2-es-5k/seed42'.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip canonical results whose `_summary_reparsed_*.json` sibling "
        "already exists on HF. Default is to overwrite — useful when "
        "regenerating after an extractor change.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_DOWNLOAD_WORKERS,
        help=f"Thread-pool size for parallel HF downloads (default: {MAX_DOWNLOAD_WORKERS}).",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="add reparsed summaries (native-aware SIB-200 extractor, PR #49)",
    )
    args = parser.parse_args()

    # Preflight extractor source so users see the error immediately, not
    # after we've already listed HF sessions and started downloads.
    from reparse_results import verify_extractor_source

    verify_extractor_source()

    # Heavy import deferred until after preflight passes.
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    print(f"Listing sessions under {PHASE3_ROOT}/ on {REPO_ID}...")
    sessions = list_sessions()
    print(f"  Found {len(sessions)} session folders")
    print()

    if args.only:
        sessions = [s for s in sessions if s["session"].endswith(args.only)]
        if not sessions:
            raise SystemExit(f"No session matched --only={args.only!r}")

    skipped_existing = 0
    if args.skip_existing:
        sessions, skipped_existing = filter_skip_existing(sessions)

    total = _print_plan(sessions, skipped_existing=skipped_existing)
    print()

    if total == 0:
        msg = (
            "Nothing to do — every canonical result already has a reparsed sibling on HF. "
            "Pass without --skip-existing to overwrite."
            if args.skip_existing
            else "No canonical results to process. Check --only filter or HF state."
        )
        print(msg)
        return

    if args.dry_run:
        print("--dry-run set; not downloading, reparsing, or uploading. Exiting.")
        return

    # Flatten the work list for the thread pool.
    work: list[str] = [
        remote_results for s in sessions for remote_results in s["canonical_results"]
    ]

    operations: list[CommitOperationAdd] = []
    with tempfile.TemporaryDirectory(prefix="reparse-cache-") as cache_dir:
        cache = Path(cache_dir)
        print(
            f"Downloading + reparsing {len(work)} file(s) "
            f"(parallel, max_workers={args.max_workers})..."
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as pool:
            futures = {
                pool.submit(_download_and_reparse, remote, cache): remote
                for remote in work
            }
            for fut in concurrent.futures.as_completed(futures):
                remote = futures[fut]
                try:
                    out_remote, tmp_out, body = fut.result()
                except Exception as e:
                    raise SystemExit(
                        f"Failed to reparse {remote!r}: {type(e).__name__}: {e}"
                    ) from e
                meta = body["reparse_metadata"]
                print(
                    f"  ✓ {_basename(remote)} → {_basename(out_remote)} "
                    f"(cells_changed={meta['cells_changed']})"
                )
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=out_remote, path_or_fileobj=str(tmp_out)
                    )
                )

        if not operations:
            # Shouldn't be reachable given the `total == 0` guard above, but
            # belt-and-suspenders for the case where a future filter is added.
            print("No operations staged; nothing to upload.")
            return

        print()
        print(f"Creating HF PR with {len(operations)} files...")
        commit_info = api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            operations=operations,
            commit_message=args.commit_message,
            commit_description=(
                "Adds `_summary_reparsed_{template}.json` siblings to each "
                "Phase-3 (condition, seed) folder under `phase3/conditions/`. "
                "Each file mirrors the original `_summary_*.json` schema "
                "(`summary` + `parse_failure_rates`) plus a `reparse_metadata` "
                "block recording when it was generated, against which extractor "
                "version, and which cells changed.\n\n"
                "The originals are untouched. Downstream analysis can read "
                "either or both — strict-vs-lenient deltas are pre-computed "
                "in `reparse_metadata.delta_per_cell` per file."
            ),
            create_pr=True,
        )

    print()
    print(f"✓ PR created: {commit_info.pr_url}")
    print(f"  PR number:  {commit_info.pr_num}")


if __name__ == "__main__":
    main()

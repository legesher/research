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

1. List every session folder under `phase3/conditions/`.
2. For each session × template, hf_hub_download the `_results_*.json` to a
   local cache.
3. Run reparse_results' build_reparsed_summary against it.
4. Stage the resulting `_summary_reparsed_*.json` in an HfApi.create_commit
   batch and push as a single discussion-PR.

Usage:

    # Dry run — list what would be uploaded, no HF writes:
    python upload_reparsed_summaries.py --dry-run

    # Real run — creates one HF discussion PR with all new summaries:
    python upload_reparsed_summaries.py

    # Limit to a single session (handy for spot-checking):
    python upload_reparsed_summaries.py --only condition-2-es-5k/seed42 --dry-run

Auth:
    Reads from HF auth cache (huggingface-cli login). Token must have WRITE
    scope on the `legesher/language-decoded-experiments` dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

# Import the local reparser's machinery so the extractor version is identical
# to what `reparse_results.py --write-reparsed-summary` would use locally.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reparse_results import build_reparsed_summary, reparse_file  # noqa: E402

REPO_ID = "legesher/language-decoded-experiments"
REPO_TYPE = "dataset"
PHASE3_ROOT = "phase3/conditions"


def list_sessions() -> list[dict]:
    """Return [{'session': 'phase3/conditions/cond/seedN', 'results': [...names]}]."""

    def tree(path: str) -> list[dict]:
        url = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main/{path}"
        return json.loads(urllib.request.urlopen(url).read())

    conds = [x["path"] for x in tree(PHASE3_ROOT) if x["type"] == "directory"]
    sessions: list[dict] = []
    for cp in sorted(conds):
        for s in tree(cp):
            if s["type"] != "directory":
                continue
            files = [f for f in tree(s["path"]) if f["type"] == "file"]
            # Canonical results files have a basename that includes the folder's
            # own seed token. Filters out accidentally-duplicated files like
            # condition-2-es-5k_seed123_*.json sitting inside seed42/.
            cond, seed_folder = s["path"].split("/")[-2:]
            expected_seed_token = (
                "seednone" if seed_folder == "seednone" else seed_folder
            )
            # Match filename pattern: <cond>_<seed_token>_results_template*.json
            filename_prefix = f"{cond}_{expected_seed_token}_results_"

            def basename(p):
                return p.split("/")[-1]

            canonical_results = sorted(
                f["path"]
                for f in files
                if "_results_" in f["path"]
                and basename(f["path"]).startswith(filename_prefix)
            )
            stray = sorted(
                f["path"]
                for f in files
                if "_results_" in f["path"]
                and not basename(f["path"]).startswith(filename_prefix)
            )
            sessions.append(
                {
                    "session": s["path"],
                    "canonical_results": canonical_results,
                    "stray_results": stray,
                }
            )
    return sessions


def reparse_one(remote_results_path: str, local_cache_dir: Path) -> tuple[Path, dict]:
    """Download one `_results_*.json` and produce its reparsed-summary body.

    Returns (local input path, body dict). Caller is responsible for writing
    or uploading the body.
    """
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=remote_results_path,
        repo_type=REPO_TYPE,
        cache_dir=str(local_cache_dir),
    )
    rows = reparse_file(Path(local_path), only=None)
    body = build_reparsed_summary(Path(remote_results_path), rows, only=None)
    return Path(local_path), body


def reparsed_remote_path(remote_results_path: str) -> str:
    """`.../X_results_template1.json` → `.../X_summary_reparsed_template1.json`."""
    name = remote_results_path.split("/")[-1]
    new_name = name.replace("_results_", "_summary_reparsed_", 1)
    return remote_results_path.rsplit("/", 1)[0] + "/" + new_name


def main() -> None:
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would happen without touching HF.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Limit to a single session, e.g., 'condition-2-es-5k/seed42'.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="add reparsed summaries (native-aware SIB-200 extractor, PR #49)",
    )
    args = parser.parse_args()

    api = HfApi()
    print(f"Listing sessions under {PHASE3_ROOT}/ on {REPO_ID}...")
    sessions = list_sessions()
    print(f"  Found {len(sessions)} session folders")
    print()

    if args.only:
        sessions = [s for s in sessions if s["session"].endswith(args.only)]
        if not sessions:
            raise SystemExit(f"No session matched --only={args.only!r}")

    # Show plan
    total_inputs = 0
    print("Plan:")
    for s in sessions:
        n_canonical = len(s["canonical_results"])
        n_stray = len(s["stray_results"])
        flag = f" (skipping {n_stray} stray)" if n_stray else ""
        print(f"  {s['session']}: {n_canonical} canonical results{flag}")
        for r in s["stray_results"]:
            print(f"    ✗ stray: {r.split('/')[-1]}")
        total_inputs += n_canonical
    print()
    print(f"Total reparses to perform: {total_inputs}")
    print(f"Total new HF files to upload: {total_inputs}")
    print()

    if args.dry_run:
        print("--dry-run set; not downloading or uploading. Exiting.")
        return

    # Process each canonical results file and stage uploads.
    operations: list[CommitOperationAdd] = []
    with tempfile.TemporaryDirectory(prefix="reparse-cache-") as cache_dir:
        cache = Path(cache_dir)
        for s in sessions:
            for remote_results in s["canonical_results"]:
                print(f"  → reparsing {remote_results.split('/')[-1]}")
                _, body = reparse_one(remote_results, cache)
                out_remote = reparsed_remote_path(remote_results)
                tmp_out = cache / Path(out_remote).name
                tmp_out.write_text(json.dumps(body, indent=2, ensure_ascii=False))
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=out_remote, path_or_fileobj=str(tmp_out)
                    )
                )
                meta = body["reparse_metadata"]
                print(
                    f"     cells_changed={meta['cells_changed']}  "
                    f"output={out_remote.split('/')[-1]}"
                )

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

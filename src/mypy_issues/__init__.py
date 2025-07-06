from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import UTC, datetime

from .apply import (
    LOG as APPLY_LOGGER,
    MypyRevision,
    run_apply,
)
from .diff import diff
from .issues import (
    LOG as ISSUES_LOGGER,
    download_snippets,
    get_commit,
    get_pr,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("mypy_issues")


def fetch_issues() -> None:
    args = _make_fetch_parser().parse_args()
    token = _get_gh_token()

    ISSUES_LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    asyncio.run(download_snippets(token, limit=args.limit, since=args.since))


def run_mypy() -> None:
    parser = _make_apply_parser()
    args = parser.parse_args()
    APPLY_LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.pr is not None:
        if (
            args.left_rev != "guess"
            or args.left_origin != "pypi"
            or args.right_rev != "master"
            or args.right_origin != "python/mypy"
        ):
            parser.error("--pr is incompatible with other targets options.")

        token = _get_gh_token()
        left, right = _update_apply_args_to_pr(args, token)
    else:
        left = MypyRevision(
            rev=None if args.left_rev == "guess" else args.left_rev,
            origin=args.left_origin,
        )
        right = MypyRevision(rev=args.right_rev, origin=args.right_origin)
    run_apply(
        left=left if not args.only_right else None,
        right=right if not args.only_left else None,
        old_strategy=args.old_strategy,
        shard=args.shard,
        total_shards=args.total_shards,
    )


def run_diff() -> None:
    args = _make_diff_parser().parse_args()
    diff(
        interactive=args.interactive,
        print_snippets=not args.no_snippets,
        diff_originals=args.diff_originals,
        ignore_notes=args.ignore_notes,
    )


def _make_fetch_parser() -> argparse.ArgumentParser:
    def parse_datetime_or_detect(val: str) -> datetime | None:
        if val == "detect":
            return None
        try:
            return datetime.fromisoformat(val).replace(tzinfo=UTC)
        except ValueError:
            parser.error(f"Invalid datetime value: {val!r}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", help="Print more output"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max amount of valid snippets to generate",
    )
    parser.add_argument(
        "--since",
        type=parse_datetime_or_detect,
        default=None,
        help="Force synchronization after this cutoff date, YYYY-MM-DD, or 'detect' to only update since last fetch.",
    )
    return parser


def _make_apply_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", help="Print more output"
    )

    left_right_group = parser.add_mutually_exclusive_group()
    left_right_group.add_argument("--only-left", action="store_true")
    left_right_group.add_argument("--only-right", action="store_true")

    revs_group = parser.add_argument_group("Targets")
    revs_group.add_argument(
        "--left-rev",
        default="guess",
        help='Version to use as old ("guess" to infer from issue), git-style or semver',
    )
    revs_group.add_argument(
        "--left-origin",
        default="pypi",
        help="Origin to use as old, org/repo or 'pypi'. Ignored if --left-rev is semver",
    )
    revs_group.add_argument(
        "--right-rev",
        default="master",
        help="Version to use as new, git-style or semver",
    )
    revs_group.add_argument(
        "--right-origin",
        default="python/mypy",
        help="Origin to use as new, org/repo or 'pypi'",
    )
    revs_group.add_argument(
        "--pr",
        type=int,
        default=None,
        help="PR to check against master. Conflicts with other options in this group.",
    )

    parser.add_argument(
        "--old-strategy",
        default="skip",
        choices=["skip", "cap"],
        help='If "cap", run with newer mypy than guessed if the guessed version is too old',
    )

    parser.add_argument("--shard", type=int, default=0, help="Shard number (from 0)")
    parser.add_argument(
        "--total-shards", type=int, default=1, help="Total amount of shards to use"
    )
    return parser


def _update_apply_args_to_pr(
    args: argparse.Namespace, gh_token: str
) -> tuple[MypyRevision, MypyRevision]:
    while True:
        pr = get_pr(gh_token, args.pr)
        if pr.merged:
            # Get a merge commit, compare to its parent. The branch may already be gone.
            # No need to merge anything here.
            right = MypyRevision(rev=pr.merge_commit_sha, origin=pr.base.repo.full_name)

            org, repo = pr.base.repo.full_name.split("/")
            assert pr.merge_commit_sha is not None
            merge_commit = get_commit(gh_token, pr.merge_commit_sha, org=org, repo=repo)
            left = MypyRevision(
                rev=merge_commit.parents[0].sha, origin=pr.base.repo.full_name
            )
        else:
            if pr.mergeable is None:
                # Evaluation in progress
                LOG.info("Waiting for merge status to arrive...")
                time.sleep(2)
                continue
            assert pr.head.repo is not None, "PR does not originate from a repo?.."
            # If the PR is still alive, we merge its ref (usually python/mypy/master) into
            # its HEAD and compare against the latest version of the ref.
            if pr.mergeable:
                left = MypyRevision(rev=pr.base.ref, origin=pr.base.repo.full_name)
                right = MypyRevision(
                    rev=pr.head.sha,
                    origin=pr.head.repo.full_name,
                    merge_with=(pr.base.ref, pr.base.repo.full_name),
                )
            else:
                LOG.warning("PR not mergeable, comparing against its base directly")
                left = MypyRevision(rev=pr.base.sha, origin=pr.base.repo.full_name)
                right = MypyRevision(
                    rev=pr.head.sha, origin=pr.head.repo.full_name, merge_with=None
                )
        return left, right


def _make_diff_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        dest="interactive",
        help="Launch review TUI",
    )
    parser.add_argument(
        "--no-snippets", action="store_true", help="Only print discovered output diffs"
    )
    parser.add_argument(
        "--diff-originals",
        action="store_true",
        help="Print diffs of unmodified outputs if normalized versions differ",
    )
    parser.add_argument(
        "--ignore-notes",
        action="store_true",
        help=(
            "Consider snippets with only 'note:' lines differing as equal"
            " (does not affect reveal_type notes)"
        ),
    )
    return parser


def _get_gh_token() -> str:
    token = os.getenv("GH_ACCESS_TOKEN")
    if token is None:
        LOG.critical("Please pass a PAT as GH_ACCESS_TOKEN")
        sys.exit(1)
    return token

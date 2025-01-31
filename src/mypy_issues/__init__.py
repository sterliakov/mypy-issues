from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .apply import (
    LOG as APPLY_LOGGER,
    run_apply,
)
from .diff import diff
from .issues import (
    LOG as ISSUES_LOGGER,
    download_snippets,
    get_pr,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("mypy_issues")


def fetch_issues() -> None:
    args = _make_fetch_parser().parse_args()
    token = _get_gh_token()

    ISSUES_LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    asyncio.run(download_snippets(token, limit=args.limit))


def run_mypy() -> None:
    parser = _make_apply_parser()
    args = parser.parse_args()
    APPLY_LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if args.pr is not None:
        if (
            args.left_rev != "master"
            or args.left_origin != "python/mypy"
            or args.right_rev != "guess"
            or args.right_origin != "pypi"
        ):
            parser.error("--pr is incompatible with other targets options.")

        token = _get_gh_token()
        _update_apply_args_to_pr(args, token)
    run_apply(
        left=not args.only_right,
        right=not args.only_left,
        left_rev=args.left_rev,
        left_origin=args.left_origin,
        right_rev=None if args.right_rev == "guess" else args.right_rev,
        right_origin=args.right_origin,
        old_strategy=args.old_strategy,
    )


def run_diff() -> None:
    args = _make_diff_parser().parse_args()
    diff(
        interactive=args.interactive,
        print_snippets=not args.no_snippets,
        diff_originals=args.diff_originals,
    )


def _make_fetch_parser() -> argparse.ArgumentParser:
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
        default="master",
        help="Version to use as old, git-style or semver",
    )
    revs_group.add_argument(
        "--left-origin",
        default="python/mypy",
        help="Origin to use as old, org/repo or 'pypi'",
    )
    revs_group.add_argument(
        "--right-rev",
        default="guess",
        help='Version to use as new ("guess" to infer from issue), git-style or semver',
    )
    revs_group.add_argument(
        "--right-origin",
        default="pypi",
        help="Origin to use as new, org/repo or 'pypi'. Ignored if --right-rev is semver",
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
    return parser


def _update_apply_args_to_pr(args: argparse.Namespace, gh_token: str) -> None:
    pr = get_pr(gh_token, args.pr)
    args.left_origin = pr.head.repo.full_name
    args.left_rev = pr.head.sha
    args.right_origin = pr.base.repo.full_name
    args.right_rev = pr.base.sha


def _make_diff_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        dest="interactive",
        help="Do not launch review TUI",
    )
    parser.add_argument(
        "--no-snippets", action="store_true", help="Only print discovered output diffs"
    )
    parser.add_argument(
        "--diff-originals",
        action="store_true",
        help="Print diffs of unmodified outputs if normalized versions differ",
    )
    return parser


def _get_gh_token() -> str:
    token = os.getenv("GH_ACCESS_TOKEN")
    if token is None:
        LOG.critical("Please pass a PAT as GH_ACCESS_TOKEN")
        sys.exit(1)
    return token

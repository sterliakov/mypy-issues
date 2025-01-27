from __future__ import annotations

import argparse
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
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("mypy_issues")


def fetch_issues() -> None:
    args = _make_fetch_parser().parse_args()

    token = os.getenv("GH_ACCESS_TOKEN")
    if token is None:
        LOG.critical("Please pass a PAT as GH_ACCESS_TOKEN")
        sys.exit(1)

    ISSUES_LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    download_snippets(token, limit=args.limit)


def run_mypy() -> None:
    args = _make_apply_parser().parse_args()
    APPLY_LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    run_apply(
        left=not args.only_right,
        right=not args.only_left,
        left_rev=args.left_rev,
        right_rev=None if args.right_rev == "guess" else args.right_rev,
        old_strategy=args.old_strategy,
    )


def run_diff() -> None:
    args = _make_diff_parser().parse_args()
    diff(interactive=args.interactive, print_snippets=not args.no_snippets)


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

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only-left", action="store_true")
    group.add_argument("--only-right", action="store_true")

    parser.add_argument(
        "--left-rev",
        default="master",
        help="Version to use as old, git-style or semver",
    )
    parser.add_argument(
        "--right-rev",
        default="guess",
        help='Version to use as old ("guess" to infer from issue), git-style or semver',
    )
    parser.add_argument(
        "--old-strategy",
        default="skip",
        choices=["skip", "cap"],
        help='If "cap", run with newer mypy than guessed if the guessed version is too old',
    )
    return parser


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
    return parser

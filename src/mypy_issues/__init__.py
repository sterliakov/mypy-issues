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
    run_apply()


def run_diff() -> None:
    diff()


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
    return parser

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Final, NamedTuple

from githubkit import GitHub
from githubkit.utils import UNSET
from githubkit.versions.latest.models import GistSimplePropFiles, Issue
from markdown_it import MarkdownIt

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("issues")
LOG.setLevel(logging.DEBUG)

OUTPUT_ROOT: Final[Path] = Path("./downloaded").resolve()
INVENTORY_ROOT: Final[Path] = OUTPUT_ROOT / "inventory.json"

RUFF = "ruff"


def main() -> None:
    token = os.getenv("GH_ACCESS_TOKEN")
    assert token is not None, "Please pass a PAT"

    args = _parse_args()
    if not args.no_cleanup:
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    OUTPUT_ROOT.mkdir(exist_ok=True)

    gh = GitHub(token)
    inventory = []
    with ThreadPoolExecutor() as pool:
        for snippets in pool.map(
            partial(extract_snippets, gh=gh),
            list(get_issues(gh, limit=args.limit)),
        ):
            for snip in snippets:
                if store_snippet(snip):
                    inventory.append({  # noqa: PERF401
                        "filename": snip.filename,
                        "mypy_version": snip.mypy_version,
                        "created_at": int(snip.date.timestamp()),
                    })
    LOG.info("Stored %s snippets to %s.", len(inventory), OUTPUT_ROOT)
    with INVENTORY_ROOT.open("w") as fd:
        json.dump(inventory, fd, indent=4)


class Snippet(NamedTuple):
    issue: int
    date: datetime
    id: int
    body: str
    mypy_version: str | None

    @property
    def filename(self) -> str:
        return f"gh_{self.issue}_{self.id}.py"


def get_issues(
    gh: GitHub[Any], *, limit: int | None = None, since: datetime | None = None
) -> Iterator[Issue]:
    i = 0
    page = 1
    has_more = True
    while has_more and (limit is None or i < limit):
        issues = gh.rest.issues.list_for_repo(
            "python",
            "mypy",
            state="open",
            sort="created",
            direction="desc",
            per_page=100,
            since=since or UNSET,
            page=page,
        ).parsed_data
        # See https://github.com/yanyongyu/githubkit/pull/184
        for iss in issues:  # type: ignore[attr-defined]
            if iss.pull_request is UNSET:
                yield iss
                i += 1
        has_more = bool(issues)
        page += 1


def extract_snippets(issue: Issue, gh: GitHub[Any]) -> list[Snippet]:
    md = MarkdownIt("gfm-like")
    i = 0
    result = []
    if not isinstance(issue.body, str):
        return []
    version = _extract_mypy_version(issue.body)
    seen_snippets = set()
    for token in md.parse(issue.body):
        if (
            token.type == "fence"
            and token.tag == "code"
            and token.info in {"", "py", "python", "python3"}
            and _is_relevant(token.content)
        ):
            norm_body = _normalize(token.content)
            if norm_body in seen_snippets:
                continue
            seen_snippets.add(norm_body)
            result.append(
                Snippet(
                    issue=issue.number,
                    date=issue.created_at,
                    id=i,
                    body=token.content,
                    mypy_version=version,
                )
            )
            i += 1
    for gist_id in re.findall(
        r"https://mypy-play\.net/\?.+?gist=([\da-f]{32})", issue.body
    ):
        LOG.debug("Extracting gist %s...", gist_id)
        gist = gh.rest.gists.get(gist_id).parsed_data
        assert isinstance(gist.files, GistSimplePropFiles)
        (file,) = gist.files.model_dump().values()
        norm_body = _normalize(file["content"])
        if norm_body in seen_snippets:
            continue
        seen_snippets.add(norm_body)
        if _is_relevant(token.content):
            result.append(
                Snippet(
                    issue=issue.number,
                    date=issue.created_at,
                    id=i,
                    body=file["content"],
                    mypy_version=version,
                )
            )
            i += 1
    return result


def _extract_mypy_version(body: str) -> str | None:
    if m := re.search(r"Mypy versions? used:.*?(\d+\.\d+(\.\d+)?)", body):
        return m.group(1)
    if m := re.search(r"[mM]ypy\s*==?\s*?(\d+\.\d+(\.\d+)?)", body):
        return m.group(1)
    if re.search(r"Mypy versions? used:.*?master", body):
        return "master"
    return None


def _is_relevant(code: str) -> bool:
    return not code.startswith("$") and not re.search(
        r"^[\w-]+\.py:\d+: (error|warning|note):", code
    )


def _normalize(snippet: str) -> str:
    return re.sub(r"\n+", "\n", snippet.replace("\r", "")).strip()


def store_snippet(snip: Snippet) -> bool:
    dest = OUTPUT_ROOT / snip.filename
    dest.write_text(snip.body)
    try:
        subprocess.check_output([RUFF, "check", dest.resolve(), "--select", "PYI001"])
    except subprocess.CalledProcessError:
        LOG.info("Rejecting snippet %s: syntax error", dest.name)
        dest.rename(dest.with_name(dest.name + ".bak"))
        return False
    return True


def _parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cleanup", default=False, action="store_true")
    return parser.parse_args()

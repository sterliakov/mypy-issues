from __future__ import annotations

import json
import logging
import os
import re
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


def main() -> None:
    token = os.getenv("GH_ACCESS_TOKEN")
    assert token is not None, "Please pass a PAT"
    OUTPUT_ROOT.mkdir(exist_ok=True)

    gh = GitHub(token)
    count = 0
    inventory = []
    with ThreadPoolExecutor() as pool:
        for snippets in pool.map(
            partial(extract_snippets, gh=gh), get_issues(gh, limit=10)
        ):
            for snip in snippets:
                store_snippet(snip)
                inventory.append({
                    "filename": snip.filename,
                    "mypy_version": snip.mypy_version,
                })
                count += 1
    LOG.info("Stored %s snippets to %s.", count, OUTPUT_ROOT)
    with INVENTORY_ROOT.open("w") as fd:
        json.dump(inventory, fd, indent=4)


class Snippet(NamedTuple):
    issue: int
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
        ).parsed_data
        # See https://github.com/yanyongyu/githubkit/pull/184
        for iss in issues:  # type: ignore[attr-defined]
            yield iss
            i += 1
        has_more = bool(issues)


def extract_snippets(issue: Issue, gh: GitHub[Any]) -> list[Snippet]:
    md = MarkdownIt("gfm-like")
    i = 0
    result = []
    if not isinstance(issue.body, str):
        return []
    version = _extract_mypy_version(issue.body)
    for token in md.parse(issue.body):
        if (
            token.type == "fence"
            and token.tag == "code"
            and token.info in {"", "py", "python", "python3"}
            and _is_relevant(token.content)
        ):
            result.append(
                Snippet(
                    issue=issue.number, id=i, body=token.content, mypy_version=version
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
        if _is_relevant(token.content):
            result.append(
                Snippet(
                    issue=issue.number, id=i, body=file["content"], mypy_version=version
                )
            )
            i += 1
    return result


def _extract_mypy_version(body: str) -> str | None:
    if m := re.search(r"Mypy versions? used:.*?(\d+\.\d+(\.\d+)?)", body):
        return m.group(1)
    if re.search(r"Mypy versions? used:.*?master", body):
        return "master"
    return None


def _is_relevant(code: str) -> bool:
    return not code.startswith("$") and not re.search(
        r"^\w+\.py:\d+: (error|warning):", code
    )


def store_snippet(snip: Snippet) -> None:
    dest = OUTPUT_ROOT / snip.filename
    dest.write_text(snip.body)

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
from collections.abc import Iterator
from datetime import datetime
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Any, Final, NamedTuple

from githubkit import GitHub
from githubkit.utils import UNSET
from githubkit.versions.latest.models import GistSimplePropFiles, Issue
from markdown_it import MarkdownIt

from .config import INVENTORY_ROOT, SNIPPETS_ROOT, InventoryItem

LOG = logging.getLogger("issues")

RUFF: Final = "ruff"


def download_snippets(token: str, *, limit: int | None = None) -> None:
    shutil.rmtree(SNIPPETS_ROOT, ignore_errors=True)
    SNIPPETS_ROOT.mkdir()

    gh = GitHub(token)
    event = threading.Event()
    inventory: list[InventoryItem] = []
    with ThreadPool() as pool:
        for snippets in pool.imap(
            partial(extract_snippets, gh_token=token, event=event),
            _get_issues(gh, event),
        ):
            for snip in snippets:
                if store_snippet(snip):
                    inventory.append({
                        "filename": snip.filename,
                        "mypy_version": snip.mypy_version,
                        "created_at": int(snip.date.timestamp()),
                    })
                    if len(inventory) == limit:
                        event.set()
                        break
            if len(inventory) == limit:
                break
    LOG.info("Stored %s snippets to %s.", len(inventory), SNIPPETS_ROOT)
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


def _get_issues(gh: GitHub[Any], event: threading.Event) -> Iterator[Issue]:
    page = 1
    has_more = True
    while has_more and not event.is_set():
        issues = gh.rest.issues.list_for_repo(
            "python",
            "mypy",
            state="open",
            sort="created",
            direction="desc",
            per_page=100,
            page=page,
        ).parsed_data
        # See https://github.com/yanyongyu/githubkit/pull/184
        for iss in issues:  # type: ignore[attr-defined]
            if iss.pull_request is UNSET:
                yield iss
        has_more = bool(issues)
        page += 1


def extract_snippets(
    issue: Issue, gh_token: str, event: threading.Event
) -> list[Snippet]:
    if event.is_set():
        return []
    gh = GitHub(gh_token)
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
    dest = SNIPPETS_ROOT / snip.filename
    dest.write_text(snip.body)
    try:
        subprocess.check_output([RUFF, "check", dest.resolve(), "--select", "PYI001"])
    except subprocess.CalledProcessError:
        LOG.debug("Rejecting snippet %s: syntax error", dest.name)
        dest.rename(dest.with_name(dest.name + ".bak"))
        return False
    LOG.debug("Added snippet %s.", dest.name)
    return True

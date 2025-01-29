from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Any, NamedTuple

from githubkit import GitHub
from githubkit.utils import UNSET
from githubkit.versions.latest.models import GistSimplePropFiles, Issue, PullRequest
from markdown_it import MarkdownIt

from .config import INVENTORY_FILE, ISSUES_FILE, SNIPPETS_ROOT, InventoryItem

LOG = logging.getLogger("issues")


def download_snippets(
    token: str, *, limit: int | None = None, org: str = "python", repo: str = "mypy"
) -> None:
    SNIPPETS_ROOT.mkdir(exist_ok=True)

    gh = GitHub(token)
    event = threading.Event()
    inventory: list[InventoryItem] = []
    issues: dict[int, Issue] = {}
    seen: set[str] = set()
    since = None
    if INVENTORY_FILE.is_file() and ISSUES_FILE.is_file():
        with INVENTORY_FILE.open() as fd:
            inventory = json.load(fd)
        with ISSUES_FILE.open() as fd:
            issues = {
                int(n): Issue.model_validate(iss) for n, iss in json.load(fd).items()
            }
        since = max(
            (iss.created_at for iss in issues.values()),
            default=datetime.fromtimestamp(0, tz=UTC),
        )
        removed = {iss.number for iss in _get_closed_issues(gh, org, repo, since)}
        inventory = [
            snip
            for snip in inventory
            if int(snip["filename"].split("_")[1]) not in removed
        ]
        issues = {n: iss for n, iss in issues.items() if n not in removed}
        for n in removed:
            for file in SNIPPETS_ROOT.glob(f"gh_{n}_*.py"):
                file.unlink()
        seen = {snip["filename"] for snip in inventory}

    with ThreadPool() as pool:
        for snippets in pool.imap(
            partial(extract_snippets, gh_token=token, event=event),
            _get_issues(gh, event, org, repo, since),
        ):
            for snip, issue in snippets:
                if snip.filename in seen:
                    continue
                if store_snippet(snip):
                    inventory.append({
                        "filename": snip.filename,
                        "mypy_version": snip.mypy_version,
                        "created_at": int(snip.date.timestamp()),
                    })
                    if len(inventory) == limit:
                        event.set()
                        break
                issues[issue.number] = issue
            if len(inventory) == limit:
                break
    LOG.info("Stored %s snippets to %s.", len(inventory), SNIPPETS_ROOT)
    with INVENTORY_FILE.open("w") as fd:
        json.dump(inventory, fd, indent=4)
    with ISSUES_FILE.open("w") as fd:
        json.dump(
            {
                n: iss.model_dump(mode="json", exclude_unset=True)
                for n, iss in issues.items()
            },
            fd,
            indent=4,
        )


class Snippet(NamedTuple):
    issue: int
    date: datetime
    id: int
    body: str
    mypy_version: str | None

    @property
    def filename(self) -> str:
        return f"gh_{self.issue}_{self.id}.py"


def _get_issues(
    gh: GitHub[Any],
    event: threading.Event,
    org: str,
    repo: str,
    since: datetime | None = None,
) -> Iterator[Issue]:
    page = 1
    has_more = True
    while has_more and not event.is_set():
        issues = gh.rest.issues.list_for_repo(
            org,
            repo,
            state="open",
            sort="created",
            direction="desc",
            since=since or UNSET,
            per_page=100,
            page=page,
        ).parsed_data
        # See https://github.com/yanyongyu/githubkit/pull/184
        for iss in issues:  # type: ignore[attr-defined]
            if iss.pull_request is UNSET:
                yield iss
        has_more = bool(issues)
        page += 1


def _get_closed_issues(
    gh: GitHub[Any], org: str, repo: str, since: datetime
) -> Iterator[Issue]:
    page = 1
    while True:
        issues = gh.rest.issues.list_for_repo(
            org,
            repo,
            state="closed",
            sort="created",
            direction="desc",
            since=since,
            per_page=100,
            page=page,
        ).parsed_data
        if not issues:
            break  # type: ignore[unreachable]
        # See https://github.com/yanyongyu/githubkit/pull/184
        for iss in issues:  # type: ignore[attr-defined]
            if iss.pull_request is UNSET:
                yield iss
        page += 1


def get_pr(
    gh_token: str, pr_number: int, *, org: str = "python", repo: str = "mypy"
) -> PullRequest:
    gh = GitHub(gh_token)
    return gh.rest.pulls.get(org, repo, pr_number).parsed_data


def extract_snippets(
    issue: Issue, gh_token: str, event: threading.Event
) -> list[tuple[Snippet, Issue]]:
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
    return [(b, issue) for b in result]


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
        # fmt: off
        subprocess.check_output(
            [
                sys.executable, "-m", "ruff",
                "check",
                str(dest.resolve()),
                "--select", "F821",
                "--output-format", "concise",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            env={"NO_COLOR": "1"},
        )
        # fmt: on
    except subprocess.CalledProcessError as exc:
        if "SyntaxError" in exc.stdout:
            LOG.debug("Rejecting snippet %s: syntax error", dest.name)
            dest.rename(dest.with_name(dest.name + ".bak"))
            return False
        has_undef_names = any(
            "F821" in line
            for line in exc.stdout.splitlines()
            if "reveal_type" not in line
        )
        if has_undef_names and "__future__" not in snip.body:
            # Try to recover
            dest.write_text("from typing import *  # Added by us\n" + snip.body)
    LOG.debug("Added snippet %s.", dest.name)
    return True

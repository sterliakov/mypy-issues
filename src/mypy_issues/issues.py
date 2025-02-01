from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import sys
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Final, NamedTuple, ParamSpec, Protocol, TypeVar

from githubkit import GitHub
from githubkit.exception import GitHubException
from githubkit.utils import UNSET
from githubkit.versions.latest.models import (
    GistSimple,
    GistSimplePropFiles,
    Issue,
    IssueComment,
    PullRequest,
)
from markdown_it import MarkdownIt

from .config import (
    INVENTORY_FILE,
    ISSUES_FILE,
    SNIPPETS_ROOT,
    InventoryItem,
    IssueWithComments,
    load_inventory,
    load_issues,
    save_inventory,
    save_issues,
)

LOG = logging.getLogger("issues")

PAGE_SIZE: Final = 100
RETRIES: Final = 5

P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


async def download_snippets(
    token: str,
    *,
    limit: int | None = None,
    org: str = "python",
    repo: str = "mypy",
) -> None:
    SNIPPETS_ROOT.mkdir(exist_ok=True)

    # Github API will sometimes hang for a long time to fail with timeout.
    # Set smaller timeout and retry.
    gh = GitHub(token, timeout=2)
    inventory: list[InventoryItem] = []
    issues: dict[int, IssueWithComments] = {}
    seen: set[str] = set()
    since = None
    if INVENTORY_FILE.is_file() and ISSUES_FILE.is_file():
        inventory = load_inventory()
        issues = load_issues()
        since = max(
            (iss.issue.created_at for iss in issues.values()),
            default=datetime.fromtimestamp(0, tz=UTC),
        )
        removed = {iss.number async for iss in _get_closed_issues(gh, org, repo, since)}
        comments = await _get_comments(gh, org, repo, since)
        # Just sync issues with new comments from scratch, that should be cheap.
        removed.update(comments.keys())
        inventory, issues = _incremental_update(inventory, issues, removed)
        seen = {snip["filename"] for snip in inventory}

    event = asyncio.Event()
    async for batch in abatch(_get_issues(gh, event, org, repo, since), size=64):
        blocks = await asyncio.gather(*[
            extract_snippets(plain_issue, gh=gh, event=event) for plain_issue in batch
        ])
        for block in blocks:
            for snip, issue in block:
                if snip.filename in seen:
                    continue
                if store_snippet(snip):
                    inventory.append({
                        "filename": snip.filename,
                        "issue": issue.issue.number,
                        "mypy_version": snip.mypy_version,
                        "created_at": int(snip.date.timestamp()),
                    })
                    if len(inventory) == limit:
                        event.set()
                        break
                issues[issue.issue.number] = issue
            if len(inventory) == limit:
                break
        if len(inventory) == limit:
            break
    LOG.info("Stored %s snippets to %s.", len(inventory), SNIPPETS_ROOT)
    save_inventory(inventory)
    save_issues(issues)


def _incremental_update(
    inventory: list[InventoryItem],
    issues: dict[int, IssueWithComments],
    removed: set[int],
) -> tuple[list[InventoryItem], dict[int, IssueWithComments]]:
    inventory = [snip for snip in inventory if snip["issue"] not in removed]
    issues = {n: iss for n, iss in issues.items() if n not in removed}
    for n in removed:
        for file in SNIPPETS_ROOT.glob(f"gh_{n}_*.py"):
            file.unlink()
    return inventory, issues


class Snippet(NamedTuple):
    issue: int
    comment: int | None
    date: datetime
    id: int
    body: str
    mypy_version: str | None

    @property
    def filename(self) -> str:
        if self.comment is None:
            return f"gh_{self.issue}_body_{self.id}.py"
        return f"gh_{self.issue}_c{self.comment}_{self.id}.py"


async def _get_issues(
    gh: GitHub[Any],
    event: asyncio.Event,
    org: str,
    repo: str,
    since: datetime | None = None,
) -> AsyncIterator[Issue]:
    page = 1
    issues: list[Issue] = []
    while not event.is_set():
        issues = await retry(
            gh.rest.issues.async_list_for_repo,
            org,
            repo,
            state="open",
            sort="created",
            direction="asc",
            since=since or UNSET,
            per_page=PAGE_SIZE,
            page=page,
        )
        for iss in issues:
            if iss.pull_request is UNSET:
                yield iss
        if len(issues) < PAGE_SIZE:
            break
        page += 1


async def _get_closed_issues(
    gh: GitHub[Any], org: str, repo: str, since: datetime
) -> AsyncIterator[Issue]:
    page = 1
    issues: list[Issue] = []
    while True:
        issues = await retry(
            gh.rest.issues.async_list_for_repo,
            org,
            repo,
            state="closed",
            sort="created",
            direction="asc",
            since=since,
            per_page=PAGE_SIZE,
            page=page,
        )
        for iss in issues:
            if iss.pull_request is UNSET:
                yield iss
        if len(issues) < PAGE_SIZE:
            break
        page += 1


async def _get_comments(
    gh: GitHub[Any], org: str, repo: str, since: datetime
) -> dict[int, list[IssueComment]]:
    result: dict[int, list[IssueComment]] = {}
    page = 1
    comments: list[IssueComment] = []
    while True:
        comments = await retry(
            gh.rest.issues.async_list_comments_for_repo,
            org,
            repo,
            sort="created",
            direction="asc",
            since=since,
            per_page=PAGE_SIZE,
            page=page,
        )
        for comm in comments:
            _, issue_no = comm.issue_url.rsplit("/", 1)
            result.setdefault(int(issue_no), []).append(comm)
        if len(comments) < PAGE_SIZE:
            break
        page += 1
    return result


async def _get_comments_for_issue(
    issue: Issue, gh: GitHub[Any]
) -> AsyncIterator[IssueComment]:
    _, org, repo = issue.repository_url.rsplit("/", 2)
    page = 1
    comments: list[IssueComment] = []
    while True:
        comments = await retry(
            gh.rest.issues.async_list_comments,
            org,
            repo,
            issue.number,
            per_page=PAGE_SIZE,
            page=page,
        )
        for com in comments:
            yield com
        if len(comments) < PAGE_SIZE:
            break
        page += 1


async def _get_gist(gist: str, gh: GitHub[Any]) -> GistSimple:
    return await retry(gh.rest.gists.async_get, gist)


def get_pr(
    gh_token: str, pr_number: int, *, org: str = "python", repo: str = "mypy"
) -> PullRequest:
    gh = GitHub(gh_token)
    return gh.rest.pulls.get(org, repo, pr_number).parsed_data


async def extract_snippets(
    issue: Issue, gh: GitHub[Any], event: asyncio.Event
) -> list[tuple[Snippet, IssueWithComments]]:
    if event.is_set():
        return []
    result: list[Snippet] = []
    version = _extract_mypy_version(issue.body) if issue.body else None
    seen_snippets = {""}  # Ignore empty
    if issue.body:
        result += [
            Snippet(
                issue=issue.number,
                comment=None,
                date=issue.created_at,
                id=i,
                body=snip,
                mypy_version=version,
            )
            for i, snip in enumerate(
                await _extract_snippets(issue.body, gh, seen_snippets)
            )
        ]
    comments = []
    if issue.comments > 0 and not event.is_set():
        comments = [com async for com in _get_comments_for_issue(issue, gh) if com.body]
        snippets = await asyncio.gather(*[
            _extract_snippets(com.body or "", gh, seen_snippets) for com in comments
        ])
        result += [
            Snippet(
                issue=issue.number,
                comment=com.id,
                date=issue.created_at,
                id=i,
                body=snip,
                mypy_version=version,
            )
            for com, snips in zip(comments, snippets, strict=True)
            for i, snip in enumerate(snips)
        ]
    return [(b, IssueWithComments(issue, comments)) for b in result]


async def _extract_snippets(
    text: str, gh: GitHub[Any], seen_snippets: set[str]
) -> list[str]:
    result: list[str] = []
    md = MarkdownIt("gfm-like")
    for token in md.parse(text):
        if (
            token.type == "fence"
            and token.tag == "code"
            and token.info in {"", "py", "pyi", "python", "python3"}
            and _is_relevant(token.content)
        ):
            norm_body = _normalize(token.content)
            if norm_body in seen_snippets:
                continue
            seen_snippets.add(norm_body)
            result.append(token.content)

    gist_ids = [
        *re.findall(r"https://mypy-play\.net/\?.+?gist=([\da-f]{32})", text),
        *re.findall(r"https://gist\.github\.com/[\w-]+/([\da-f]{32})", text),
    ]
    if not gist_ids:
        return result
    gists = await asyncio.gather(*[_get_gist(gist_id, gh) for gist_id in gist_ids])
    for gist in gists:
        assert isinstance(gist.files, GistSimplePropFiles)
        (file,) = gist.files.model_dump().values()
        norm_body = _normalize(file["content"])
        if norm_body in seen_snippets:
            continue
        seen_snippets.add(norm_body)
        if _is_relevant(token.content):
            result.append(file["content"])
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
    return (
        not code.startswith("$")
        and not re.search(r"^[\w-]+\.py:\d+: (error|warning|note):", code)
        and not re.search(r"^\[(tool\.)?mypy\]", code)
    )


def _normalize(snippet: str) -> str:
    # strip comments (yes, this is invalid, should parse ast, but this is fast)
    # we don't care if something goes wrong, we check orig snippets - this is only
    # needed to reduce duplicates
    snippet = re.sub(r"#.+\n", "\n", snippet)
    # Strip trailing whitespace
    snippet = re.sub(r"\s+\n", "\n", snippet)
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
            dest.unlink()
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


async def abatch(aiterable: AsyncIterable[T], size: int) -> AsyncIterable[list[T]]:
    batch = []
    async for element in aiterable:
        batch.append(element)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class _WithParsedData(Protocol[T_co]):
    @property
    def parsed_data(self) -> T_co: ...


async def retry(
    fn: Callable[P, Awaitable[_WithParsedData[T]]], *args: P.args, **kwargs: P.kwargs
) -> T:
    for i in range(RETRIES):
        try:
            return (await fn(*args, **kwargs)).parsed_data
        except GitHubException:
            LOG.warning("Request failed, retrying %d more times...", RETRIES - 1 - i)
    raise RuntimeError(f"Request failed {RETRIES} times, aborting.")

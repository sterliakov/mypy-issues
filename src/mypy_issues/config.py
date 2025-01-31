from __future__ import annotations

import json
from pathlib import Path
from typing import Final, NamedTuple, TypedDict

from githubkit.versions.latest.models import Issue, IssueComment

SNIPPETS_ROOT: Final = Path("./downloaded").resolve()
INVENTORY_FILE: Final = SNIPPETS_ROOT / "inventory.json"
ISSUES_FILE: Final = SNIPPETS_ROOT / "issues.json"

RUN_OUTPUTS_ROOT: Final = Path("./outputs")
LEFT_OUTPUTS: Final = RUN_OUTPUTS_ROOT / "left"
RIGHT_OUTPUTS: Final = RUN_OUTPUTS_ROOT / "right"


class IssueWithComments(NamedTuple):
    issue: Issue
    comments: list[IssueComment]


class InventoryItem(TypedDict):
    filename: str
    issue: int
    mypy_version: str | None
    created_at: int


def save_inventory(inventory: list[InventoryItem]) -> None:
    with INVENTORY_FILE.open("w") as fd:
        json.dump(inventory, fd, indent=4)


def load_inventory() -> list[InventoryItem]:
    with INVENTORY_FILE.open() as fd:
        return json.load(fd)


def save_issues(issues: dict[int, IssueWithComments]) -> None:
    with ISSUES_FILE.open("w") as fd:
        json.dump(
            {
                n: {
                    "issue": iss.issue.model_dump(mode="json", exclude_unset=True),
                    "comments": [
                        com.model_dump(mode="json", exclude_unset=True)
                        for com in iss.comments
                    ],
                }
                for n, iss in issues.items()
            },
            fd,
            indent=4,
        )


def load_issues() -> dict[int, IssueWithComments]:
    with ISSUES_FILE.open() as fd:
        return {
            int(n): IssueWithComments(
                Issue.model_validate(iss["issue"]),
                [IssueComment.model_validate(com) for com in iss["comments"]],
            )
            for n, iss in json.load(fd).items()
        }

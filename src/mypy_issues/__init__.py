from __future__ import annotations

import os

from githubkit import GitHub


def main() -> None:
    token = os.getenv("GH_ACCESS_TOKEN")
    assert token is not None, "Please pass a PAT"
    gh = GitHub(token)
    issues = gh.rest.issues.list_for_repo(
        "python", "mypy", state="open", sort="created", direction="desc", per_page=50
    ).parsed_data
    print(issues)

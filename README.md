Compare `mypy` output on valid snippets from `mypy` issues.

## Online usage

To run against a PR, simply create an issue with PR number in title
(e.g. "Check #12345") and wait approx. 15 minutes for results to arrive
in comments.

Comment on the issue to run again (e.g. if upstream PR was updated).

Issues will be closed after 14 days of inactivity.

## Prerequisites

* Linux (this may or may not work on other OSes or under WSL)
* GitHub API token with "read public repositories" permission.
* `uv` installed and available on PATH
* `git` installed and available on PATH
* Non-metered connection and some disk space (~ 1 GB)

## Common flow

```bash
export GH_ACCESS_TOKEN="github_pat_xxxxx...x"
# Fetch GitHub issues database. ~1 min on my PC.
# Starts from scratch if rerun.
uv run fetch
# Run `mypy` on the acquired snippets. ~15 mins per version
# on my 4 cores, 30 mins in total.
uv run run
# Normalize and compare outputs.
# Beware: diffs you see may differ from real `mypy` output,
# they undergo severe preprocessing to soften version differences.
uv run diff | less
```

Every subcommand supports `--help`, please read.

## Use cases

### Compare `master` to the version against which the issue was reported

Supports `mypy >= 0.800` (issues from previous versions will be ignored or
analyzed with 0.800, depending on `--old-strategy` flag to `run`).
Guesses the version (assumes latest non-dev release existing at that time) if
not mentioned explicitly.

To compare the outputs manually, pass `-i` (`--interactive`) to `uv run diff`.
This will enable interactive "review" style of every issue.
To print a huge diff to stdout, do not pass that flag. Pipe through `less` - I warned you!

### Compare commit to the version against which the issue was reported

Same, but use `uv run run --left-rev=xxxxx`, supports semver (installed from PyPI if available)
and anything that `git checkout` can understand.

### Compare two revisions

This will not ignore issues for `mypy` below 0.800. Pass `--right-rev` to `uv run run`.

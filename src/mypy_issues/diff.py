from __future__ import annotations

import difflib
import re

from .config import LEFT_OUTPUTS, RIGHT_OUTPUTS, SNIPPETS_ROOT


def diff() -> None:
    def get_prefix(filename: str) -> str:
        _, iss, _ = filename.split("_", 2)
        return f"gh_{iss}"

    names = {get_prefix(f.name) for f in LEFT_OUTPUTS.iterdir()}
    print_snippets = True
    for prefix in sorted(names):
        if diffs := diff_one(prefix):
            issue_id = prefix.removeprefix("gh_")
            print()
            print("=" * 40)
            print(
                f"Issue #{issue_id}: https://github.com/python/mypy/issues/{issue_id}"
            )
            for d, snip in diffs:
                if print_snippets:
                    print(snip)
                print(d)


def diff_one(prefix: str) -> list[tuple[str, str]]:
    left_files = list(LEFT_OUTPUTS.glob(f"{prefix}_*"))
    right_files = list(RIGHT_OUTPUTS.glob(f"{prefix}_*"))
    if not left_files or not right_files:
        return []
    outs = []
    snips = []
    for left in left_files:
        right = RIGHT_OUTPUTS / left.name
        if not right.is_file():
            continue
        snips.append((SNIPPETS_ROOT / left.with_suffix(".py").name).read_text().strip())
        diff_iter = difflib.unified_diff(
            _normalize(right.read_text()),
            _normalize(left.read_text()),
            lineterm="",
        )
        diff = list(diff_iter)[2:]  # Remove ---/+++ header

        outs.append("\n".join(map(colorize, diff)))

    return [
        (block, snip) for block, snip in zip(outs, snips, strict=False) if block.strip()
    ]


def _normalize(text: str) -> list[str]:
    text = (
        text.replace("<nothing>", "Never")
        .replace("NoReturn", "Never")
        .replace('"Never"', "Never")
        .replace("*", "")  # Old-style * for inferred types
    )
    text = text.replace("'", '"')
    text = re.sub(r"`-?\d+", "", text)
    for typ in ["Type", "List", "Dict", "Set", "FrozenSet", "Tuple"]:
        text = re.sub(rf"\b{typ}\b", typ.lower(), text)
    text = text.replace("tuple[,", "tuple[(),")
    text = re.sub(r"Optional\[(\w+?)\]", r"\1 | None", text)
    text = re.sub(r'"Optional\[(.+?)\]"', r'"\1 | None"', text)
    text = re.sub(
        r'"Union\[(.+?)\]"', lambda m: '"' + _piped_union(m.group(1)) + '"', text
    )
    text = re.sub(r"Union\[([\w .,]+?)\]", lambda m: _piped_union(m.group(1)), text)
    text = re.sub(
        r'Skipping analyzing "(.+?)": found module but no type hints or library stubs',
        r'Cannot find implementation or library stub for module named "\1"',
        text,
    )
    text = re.sub(
        r'Library stubs not installed for "(.+)" \(or incompatible with Python 3\.\d+\)',
        r'Cannot find implementation or library stub for module named "\1"',
        text,
    )
    text = _rewrite_literals(text)
    lines = [
        _remove_code(line)
        for line in text.strip().splitlines()
        if not _is_extra_note(line)
    ]
    if lines and lines[-1].startswith(("Success: ", "Found ")):
        lines = lines[:-1]
    return sorted(lines)


def _is_extra_note(line: str) -> bool:
    return "Missing return statement" in line or (
        "note:" in line and "note: Revealed type" not in line
    )


def _remove_code(line: str) -> str:
    return re.sub(r"\s*\[[\w-]+\]$", "", line)


def _piped_union(s: str) -> str:
    r = ""
    level = 0
    for c in s:
        if c == "[":
            level += 1
        elif c == "]":
            level -= 1
        elif c == "," and level == 0:
            r += " |"
            continue
        r += c
    return r


def _rewrite_literals(s: str) -> str:
    while True:
        s1 = re.sub(
            r"Literal\[([\w'\"., ]+)\] \| Literal\[([\w'\"., ]+)\]",
            r"Literal[\1, \2]",
            s,
            count=1,
        )
        if s == s1:
            break
        s = s1
    return s1


def colorize(line: str) -> str:
    if not line:
        return line
    end = "\x1b[0m"
    color = {
        "+": "\x1b[1;32m",
        "-": "\x1b[1;31m",
    }.get(line[0])
    if color is None:
        return line
    return f"{color}{line}{end}"

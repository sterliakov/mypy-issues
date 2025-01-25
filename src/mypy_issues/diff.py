from __future__ import annotations

import difflib
import re

from pygments import highlight
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers import PythonLexer

from .config import LEFT_OUTPUTS, RIGHT_OUTPUTS, SNIPPETS_ROOT


def diff(*, print_snippets=True) -> None:
    def get_issue_no(filename: str) -> int:
        _, iss, _ = filename.split("_", 2)
        return int(iss)

    names = {get_issue_no(f.name) for f in LEFT_OUTPUTS.iterdir()}
    for issue_no in sorted(names, reverse=True):
        if diffs := diff_one(f"gh_{issue_no}"):
            print("=" * 80)
            print(
                f"Issue #{issue_no}: https://github.com/python/mypy/issues/{issue_no}"
            )
            for d, snip in diffs:
                if print_snippets:
                    print(
                        highlight(
                            snip,
                            PythonLexer(),
                            TerminalTrueColorFormatter(linenos=True, bg="dark"),
                        )
                    )
                    print("-" * 80)
                print(d)
                print()


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
        .replace(
            "Function is missing a return type annotation",
            "Function is missing a type annotation",
        )
    )
    text = text.replace("'", '"')
    text = re.sub(r"`-?\d+", "", text)
    text = re.sub(r"(.)gh_\d+_\d+\.", r"\1", text)
    for typ in ["Type", "List", "Dict", "Set", "FrozenSet", "Tuple"]:
        text = re.sub(rf"\b{typ}\b", typ.lower(), text)
    text = text.replace("tuple[,", "tuple[(),")
    text = text.replace('"type"', '"builtins.type"')
    text = re.sub(r"\bellipsis\b", "EllipsisType", text)
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
        r'Skipping analyzing "(.+)": module is installed, but missing library stubs or py\.typed marker',
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

    def lineno(s: str) -> int:
        try:
            _, n, _ = s.split(":", 2)
            return int(n)
        except ValueError:
            return -1

    return sorted(lines, key=lambda line: (lineno(line), line))


def _is_extra_note(line: str) -> bool:
    return (
        "Missing return statement" in line
        or ("note:" in line and "note: Revealed type" not in line)
        or "syntax error in type comment" in line
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

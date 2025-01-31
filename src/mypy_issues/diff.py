from __future__ import annotations

import contextlib
import difflib
import json
import os
import re
import sys
import termios
import tty
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar, Final, TextIO

from githubkit.versions.latest.models import Issue
from pygments import highlight
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers import PythonLexer

from .config import (
    ISSUES_FILE,
    LEFT_OUTPUTS,
    RIGHT_OUTPUTS,
    RUN_OUTPUTS_ROOT,
    SNIPPETS_ROOT,
)


def diff(
    *,
    interactive: bool = True,
    print_snippets: bool = True,
    diff_originals: bool = False,
) -> None:
    def get_issue_no(filename: str) -> int:
        _, iss, _ = filename.split("_", 2)
        return int(iss)

    with ISSUES_FILE.open() as fd:
        issues = {int(n): Issue.model_validate(iss) for n, iss in json.load(fd).items()}

    printer_cls = InteractivePrinter if interactive else NonInteractivePrinter
    printer = printer_cls(issues, print_snippets=print_snippets)
    printer.setup()
    names = {get_issue_no(f.name) for f in LEFT_OUTPUTS.iterdir()}
    for issue_no in sorted(names, reverse=True):
        if diffs := diff_one(f"gh_{issue_no}", diff_originals=diff_originals):
            printer.print_issue(diffs, issue_no)
    printer.finalize()


class Printer(ABC):
    def __init__(self) -> None:
        self.count = 0

    @abstractmethod
    def print_issue(self, diffs: list[tuple[str, str]], issue_number: int) -> None:
        self.count += 1

    def setup(self) -> None:  # noqa: B027
        pass

    def finalize(self) -> None:
        print(f"Found {self.count:d} issues with different output.")


class NonInteractivePrinter(Printer):
    sep_width: ClassVar[int] = 80

    def __init__(
        self, issues: dict[int, Issue], *, print_snippets: bool = True
    ) -> None:
        super().__init__()
        self.issues = issues
        self.print_snippets = print_snippets

    def print_issue(self, diffs: list[tuple[str, str]], issue_number: int) -> None:
        super().print_issue(diffs, issue_number)
        print("=" * self.sep_width)
        print(f"#{issue_number}: {self.issues[issue_number].title}")
        labels = ", ".join(
            lb if isinstance(lb, str) else (lb.name or "")
            for lb in self.issues[issue_number].labels
        )
        print(f"Labels: {labels}")
        for d, snip in diffs:
            if self.print_snippets:
                print(
                    highlight(
                        snip,
                        PythonLexer(),
                        TerminalTrueColorFormatter(linenos=True, bg="dark"),
                    )
                )
                print("-" * self.sep_width)
            print(d)
            print()


class InteractivePrinter(NonInteractivePrinter):
    files: Final = {
        "f": RUN_OUTPUTS_ROOT / "fixed.txt",
        "b": RUN_OUTPUTS_ROOT / "better.txt",
        "w": RUN_OUTPUTS_ROOT / "worse.txt",
        "s": RUN_OUTPUTS_ROOT / "same.txt",
        "n": RUN_OUTPUTS_ROOT / "notes.txt",
    }

    def __init__(
        self, issues: dict[int, Issue], *, print_snippets: bool = True
    ) -> None:
        super().__init__(issues, print_snippets=print_snippets)
        self.skip: set[int] = set()

    def setup(self) -> None:
        super().setup()
        if not any(f.is_file() for f in self.files.values()):
            return
        confirm = input(
            "Clean previous results (if no, will only show missing entries)? [y/N] "
        ).lower()
        if confirm in {"y", "yes"}:
            for f in self.files.values():
                f.unlink(missing_ok=True)
        else:
            self.skip = {
                int(iss.split(":")[0])
                for f in self.files.values()
                if f.is_file()
                for iss in f.read_text().splitlines()
            }
            self.count += len(self.skip)

    def finalize(self) -> None:
        super().finalize()
        print("Review results written to:")
        for file in self.files.values():
            lines = len(file.read_text().splitlines())
            print(f"{file.stem.title()}: {file.relative_to('.')} ({lines} issues)")

    def print_issue(self, diffs: list[tuple[str, str]], issue_number: int) -> None:
        if issue_number in self.skip:
            return
        print("\033[H\033[J", end="")
        super().print_issue(diffs, issue_number)
        while True:
            self._print_prompt()
            ch = self.getchar().lower()
            if ch == "c":
                self.print_context(issue_number)
            elif ch == "n":
                message = input("\nEnter your comment: ")
                with self.files[ch].open("a") as fd:
                    fd.write(f"{issue_number}: {message}\n")
                break
            elif ch in "fbws":
                with self.files[ch].open("a") as fd:
                    fd.write(f"{issue_number}\n")
                break

    def _print_prompt(self) -> None:
        print("f=fixed | b=better | w=worse | s=same | n=note | c=context: ")
        print("How's this (don't press Enter)? [fbwsinc] ")

    def print_context(self, issue_number: int) -> None:
        print("~" * self.sep_width)
        print(self.issues[issue_number].body)
        print("~" * self.sep_width)

    @contextlib.contextmanager
    def raw_terminal(self) -> Iterator[int]:
        """Switch to raw-mode terminal.

        Adapted from https://github.com/pallets/click/blob/main/src/click/_termui_impl.py
        """
        f: TextIO | None
        fd: int

        if not sys.stdin.isatty():
            f = open("/dev/tty")  # noqa: PLW1514, SIM115, PTH123
            fd = f.fileno()
        else:
            fd = sys.stdin.fileno()
            f = None

        try:
            old_settings = termios.tcgetattr(fd)

            try:
                tty.setraw(fd)
                yield fd
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                sys.stdout.flush()

                if f is not None:
                    f.close()
        except termios.error:
            pass

    def getchar(self, *, echo: bool = True) -> str:
        """Read one character from a terminal.

        Adapted from https://github.com/pallets/click/blob/main/src/click/_termui_impl.py
        """
        with self.raw_terminal() as fd:
            ch = os.read(fd, 32).decode("utf8", "replace")

            if echo and sys.stdout.isatty():
                sys.stdout.write(ch)

            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x04":  # Unix-like, Ctrl+D
                raise EOFError
            return ch


def diff_one(prefix: str, *, diff_originals: bool = False) -> list[tuple[str, str]]:
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
        right_out = right.read_text()
        left_out = left.read_text()
        if "syntax error in type comment" in right_out:
            # mypy too old, something went wrong
            continue
        snips.append((SNIPPETS_ROOT / left.with_suffix(".py").name).read_text().strip())
        if diff_originals:
            if _normalize(right_out) != _normalize(left_out):
                diff_iter = difflib.unified_diff(
                    right_out.splitlines(), left_out.splitlines(), lineterm=""
                )
            else:
                diff_iter = iter([])
        else:
            diff_iter = difflib.unified_diff(
                _normalize(right_out), _normalize(left_out), lineterm=""
            )
        diff = list(diff_iter)[2:]  # Remove ---/+++ header

        outs.append("\n".join(map(colorize, diff)))

    return [
        (block, snip) for block, snip in zip(outs, snips, strict=False) if block.strip()
    ]


def _normalize(text: str) -> list[str]:
    text = (
        text.replace("'", '"')
        .replace("<nothing>", "Never")
        .replace("NoReturn", "Never")
        .replace('"Never"', "Never")
        .replace('"super()"', "super()")
        .replace("super()", '"super()"')
        .replace('"NamedTuple()"', "NamedTuple()")
        .replace("NamedTuple()", '"NamedTuple()"')
        .replace(
            "Function is missing a return type annotation",
            "Function is missing a type annotation",
        )
    )
    text = re.sub(r"`-?\d+", "", text)
    text = re.sub(r"(.)gh_\d+_\d+\.", r"\1", text)
    for typ in ["Type", "List", "Dict", "Set", "FrozenSet", "Tuple"]:
        text = re.sub(rf"\b{typ}\b", typ.lower(), text)
    text = text.replace("tuple[,", "tuple[(),")
    text = text.replace("builtins.", "")
    text = re.sub(r"\bellipsis\b", "EllipsisType", text)
    text = re.sub(r"Optional\[(\w+?)\]", r"\1 | None", text)
    text = re.sub(r'"Optional\[(.+?)\]"', r'"\1 | None"', text)
    while "Union[" in text:
        text = re.sub(r"Union\[(.+)", lambda m: _piped_union(m.group(1)), text)
    for mod in [
        r'Skipping analyzing "(.+?)": found module but no type hints or library stubs',
        r'Skipping analyzing "(.+)": module is installed, but missing library stubs or py\.typed marker',
        r'Library stubs not installed for "(.+)" \(or incompatible with Python 3\.\d+\)',
        r'Library stubs not installed for "(.+)"',
    ]:
        text = re.sub(
            mod,
            r'Cannot find implementation or library stub for module named "\1"',
            text,
        )
    text = _rewrite_literals(text)
    text = re.sub(r"\*(?!\w)", "", text)  # Old-style inferred type asterisks
    text = re.sub(r"\bunused\b", "Unused", text)
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
        if level >= 0 and c == "[":
            level += 1
        elif c == "]":
            level -= 1
            if level == -1:
                continue
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

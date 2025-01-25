from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from bisect import bisect_left
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from functools import partial
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from tqdm import trange

from mypy_issues.config import (
    INVENTORY_ROOT,
    LEFT_OUTPUTS,
    RIGHT_OUTPUTS,
    SNIPPETS_ROOT,
    InventoryItem,
)

LOG = logging.getLogger("apply")
LOG.setLevel(logging.DEBUG)

LEFT: Final = Path("left_mypy")
RIGHT: Final = Path("right_mypy")

GIT: Final = "git"
UV: Final = "uv"

MYPY_CONFIG: Final = """
[mypy]
warn_unreachable = True
warn_unused_ignores = True
"""


class UnknownVersionError(RuntimeError):
    pass


class IncompatiblePythonError(RuntimeError):
    pass


def run_apply() -> None:
    with INVENTORY_ROOT.open() as fd:
        inventory = json.load(fd)
    inventory = list(add_versions(inventory))

    # Prevent interference from parent pyproject.toml
    (SNIPPETS_ROOT / "mypy.ini").write_text(MYPY_CONFIG)

    LOG.info("Running left (current) mypy...")
    run_left(inventory)
    LOG.info("Running left (current) mypy done.")
    LOG.info("Running right (referenced) mypy...")
    run_right(inventory)
    LOG.info("Running right (referenced) mypy done.")


def add_versions(inventory: list[InventoryItem]) -> Iterator[InventoryItem]:
    releases = _get_releases()
    dates = sorted(releases)
    for file in inventory:
        if file["mypy_version"] not in {None, "master"}:
            yield file
            continue
        latest_release_index = bisect_left(
            dates, datetime.fromtimestamp(file["created_at"], tz=UTC)
        )
        if latest_release_index == len(dates):
            continue
        yield file | {"mypy_version": releases[dates[latest_release_index]]}


def _get_releases() -> dict[datetime, str]:
    out = subprocess.check_output(
        [GIT, "log", "--tags", "--simplify-by-decoration", "--pretty=%at %D"],
        text=True,
        cwd=LEFT,
    )
    date_to_tag = {}
    for line in out.splitlines():
        if "tag:" in line:
            # 1721404787 tag: v1.11.0, tag: v1.11
            # 1719274966 origin/master, origin/HEAD
            # 1719125073 tag: v1.10.1, upstream/release-1.10
            first_part, *_ = line.split(",")
            timestamp, _, tag = first_part.split()
            date_to_tag[datetime.fromtimestamp(int(timestamp), tz=UTC)] = (
                tag.removesuffix("v")
            )
    return date_to_tag


def run_left(inventory: list[InventoryItem]) -> None:
    shutil.rmtree(LEFT_OUTPUTS)
    LEFT_OUTPUTS.mkdir()
    mypy = _setup_copy_from_source(LEFT, "master")
    run_on_files(mypy, [SNIPPETS_ROOT / f["filename"] for f in inventory], LEFT_OUTPUTS)


def run_right(inventory: list[InventoryItem]) -> None:
    shutil.rmtree(RIGHT_OUTPUTS)
    RIGHT_OUTPUTS.mkdir()

    def get_ver(item: InventoryItem) -> str:
        ver = item["mypy_version"]
        if not ver:
            return ""  # Should be comparable
        ver = ver.removesuffix("v")
        match ver.split("."):
            case ["1", _]:
                return f"{ver}.0"
            case _:
                return ver

    for ver, files_ in groupby(sorted(inventory, key=get_ver), key=get_ver):
        if not ver:
            continue
        files = list(files_)
        try:
            mypy = _setup_copy_from_pypi(RIGHT, ver)
        except UnknownVersionError:
            LOG.warning("Failed to switch to version %s", ver)
            continue
        run_on_files(
            mypy, [SNIPPETS_ROOT / f["filename"] for f in files], RIGHT_OUTPUTS
        )


def run_on_files(mypy: Path, files: Sequence[Path], dest_root: Path) -> None:
    with Pool() as pool, TemporaryDirectory() as tmp:
        task_iter = pool.imap_unordered(
            partial(run_on_file, mypy=mypy, dest_root=dest_root, temp_dir=tmp),
            files,
        )
        list(zip(trange(len(files)), task_iter, strict=True))


def run_on_file(target: Path, mypy: Path, dest_root: Path, temp_dir: str) -> None:
    out = _run_on_file(mypy, target, temp_dir)
    (dest_root / f"{target.stem}.txt").write_text(out)


def _run_on_file(mypy: Path, target: Path, temp_dir: str) -> str:
    try:
        # fmt: off
        res = subprocess.run(
            [
                str(mypy.resolve()),
                str(target.resolve()),
                "--cache-dir", f"{temp_dir}/{os.getpid()}",
                "--show-traceback",
                "--strict",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=SNIPPETS_ROOT,
        )
        # fmt: on
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    else:
        text = res.stdout + res.stderr
        if not text:
            # segfault
            return "CRASH"
        if "Traceback (most recent call last)" in text:
            return "CRASH"
        return text


def _setup_copy_from_source(dest: Path, rev: str = "master") -> Path:
    LOG.debug("Switching to mypy version %s", rev)
    wd = dest.resolve()
    if not dest.is_dir():
        subprocess.check_output(
            [
                GIT,
                "clone",
                "https://github.com/python/mypy",
                str(wd),
            ],
            stderr=subprocess.STDOUT,
        )
        _call_uv(["venv"], wd)
    else:
        subprocess.check_output(
            [GIT, "fetch", "--all"], cwd=wd, stderr=subprocess.STDOUT
        )

    try:
        subprocess.check_output(
            [GIT, "checkout", rev], cwd=wd, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        raise UnknownVersionError(f"Unknown version: {rev}") from exc

    LOG.debug("Installing mypy %s from source...", rev)
    _call_uv(["pip", "install", "-e", ".", "--reinstall"], wd)
    return dest / ".venv/bin/mypy"


def _setup_copy_from_pypi(dest: Path, rev: str) -> Path:
    for minor in [13, 10]:  # intermediate versions never help; below 8 is too old
        try:
            path = _setup_copy_from_pypi_with_python(dest, rev, f"3.{minor}")
        except IncompatiblePythonError:
            LOG.info("Python 3.%d too new for mypy %s.", minor, rev)
            continue
        else:
            LOG.info("Using mypy %s with python 3.%d", rev, minor)
            return path
    raise UnknownVersionError(f"Failed to find a suitable python for mypy {rev}")


def _setup_copy_from_pypi_with_python(dest: Path, rev: str, python: str) -> Path:
    dest.mkdir(exist_ok=True)
    wd = dest.resolve()
    venv = f".venv{python}"
    if not (dest / venv).is_dir():
        _call_uv(["venv", venv, "--python", python], wd)
    rev = rev.removeprefix("v")

    try:
        _call_uv(
            ["pip", "install", f"mypy=={rev}", "--python", f"{venv}/bin/python"], wd
        )
    except subprocess.CalledProcessError as exc:
        if "typed-ast" in (exc.stderr or "") + (exc.stdout or ""):
            raise IncompatiblePythonError(
                f"Python {python} is too new for mypy {rev}"
            ) from exc
        raise UnknownVersionError(f"Unknown version: {rev}") from exc
    else:
        return dest / venv / "bin/mypy"


def _call_uv(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(
        [UV, *cmd],
        cwd=cwd,
        stderr=subprocess.STDOUT,
        env={"PATH": os.environ["PATH"]},
        text=True,
    )

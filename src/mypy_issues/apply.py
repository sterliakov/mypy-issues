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
from typing import Final, Literal, TypeAlias

from tqdm import trange

from mypy_issues.config import (
    INVENTORY_FILE,
    LEFT_OUTPUTS,
    RIGHT_OUTPUTS,
    SNIPPETS_ROOT,
    InventoryItem,
)

LOG = logging.getLogger("apply")

OldStrategy: TypeAlias = Literal["skip", "cap"]

LEFT: Final = Path("left_mypy")
RIGHT: Final = Path("right_mypy")

GIT: Final = "git"
UV: Final = "uv"

MIN_SUPPORTED_MYPY: Final = (0, 800)
DEFAULT_PACKAGES: Final = ("attrs",)

MYPY_CONFIG: Final = """
[mypy]
warn_unreachable = True
warn_unused_ignores = True
"""


class UnknownVersionError(RuntimeError):
    pass


class IncompatiblePythonError(RuntimeError):
    pass


def run_apply(
    *,
    left: bool = True,
    right: bool = True,
    left_rev: str = "master",
    left_origin: str = "python/mypy",
    right_rev: str | None = None,
    right_origin: str = "pypi",
    old_strategy: OldStrategy = "skip",
) -> None:
    with INVENTORY_FILE.open() as fd:
        inventory = json.load(fd)

    if right_rev is None:
        _setup_copy_from_source("master")  # To have tags available
        inventory = list(add_versions(inventory, old_strategy))

    # Prevent interference from parent pyproject.toml
    (SNIPPETS_ROOT / "mypy.ini").write_text(MYPY_CONFIG)

    if left:
        LOG.info("Running left (current) mypy...")
        run_left(inventory, left_rev, left_origin)
        LOG.info("Running left (current) mypy done.")
    if right:
        LOG.info("Running right (referenced) mypy...")
        run_right(inventory, right_rev, right_origin)
        LOG.info("Running right (referenced) mypy done.")


def add_versions(
    inventory: list[InventoryItem], old_strategy: OldStrategy
) -> Iterator[InventoryItem]:
    releases = _get_releases()
    dates = sorted(releases)
    for file in inventory:
        ver = file["mypy_version"]
        if ver in {None, "master"}:
            latest_release_index = bisect_left(
                dates, datetime.fromtimestamp(file["created_at"], tz=UTC)
            )
            if latest_release_index == len(dates):
                continue
            ver = releases[dates[latest_release_index]]
        parsed = _parse_semver(ver)
        if parsed is None or parsed < MIN_SUPPORTED_MYPY:
            if old_strategy == "cap":
                parsed = MIN_SUPPORTED_MYPY
                ver = ".".join(map(str, parsed))
            else:
                continue
        yield file | {"mypy_version": ver}


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
                tag.removeprefix("v")
            )
    return date_to_tag


def run_left(inventory: list[InventoryItem], rev: str, origin: str) -> None:
    if LEFT_OUTPUTS.is_dir():
        shutil.rmtree(LEFT_OUTPUTS)
    LEFT_OUTPUTS.mkdir(parents=True)
    mypy = _setup_mypy(rev, origin)
    run_on_files(mypy, [SNIPPETS_ROOT / f["filename"] for f in inventory], LEFT_OUTPUTS)


def run_right(inventory: list[InventoryItem], rev: str | None, origin: str) -> None:
    if RIGHT_OUTPUTS.is_dir():
        shutil.rmtree(RIGHT_OUTPUTS)
    RIGHT_OUTPUTS.mkdir(parents=True)

    def get_ver(item: InventoryItem) -> tuple[int, int] | tuple[int, int, int]:
        ver = item["mypy_version"]
        assert ver
        parsed = _parse_semver(ver)
        assert parsed, ver
        return parsed

    if rev is None and origin == "pypi":
        # Use guessed version for each snippet
        for ver, files_ in groupby(sorted(inventory, key=get_ver), key=get_ver):
            files = list(files_)
            try:
                mypy = _setup_copy_from_pypi(".".join(map(str, ver)))
            except UnknownVersionError:
                LOG.warning("Failed to switch to version %s", ver)
                continue
            run_on_files(
                mypy, [SNIPPETS_ROOT / f["filename"] for f in files], RIGHT_OUTPUTS
            )
    elif rev is not None:
        mypy = _setup_mypy(rev, origin)
        run_on_files(
            mypy, [SNIPPETS_ROOT / f["filename"] for f in inventory], RIGHT_OUTPUTS
        )
    else:
        raise RuntimeError("Only 'pypi' origin supported for 'guess' revision.")


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


def _setup_mypy(rev: str, origin: str) -> Path:
    if origin == "pypi" and (parts := _parse_semver(rev)) is not None:
        # Prefer PyPI installs
        try:
            return _setup_copy_from_pypi(".".join(map(str, parts)))
        except UnknownVersionError:
            LOG.info("Failed to install mypy '%s' from PyPI, trying git...", rev)
    return _setup_copy_from_source(rev, origin)


def _setup_copy_from_source(rev: str = "master", origin: str = "python/mypy") -> Path:
    LOG.debug("Switching to mypy version %s", rev)
    dest = LEFT
    wd = dest.resolve()
    if not dest.is_dir():
        subprocess.check_output(
            [GIT, "clone", "https://github.com/python/mypy", str(wd)],
            stderr=subprocess.STDOUT,
        )
        _call_uv(["venv"], wd)
        _call_uv(["pip", "install", *DEFAULT_PACKAGES], wd)

    remotes = {
        r.split("\t")[0]
        for r in subprocess.check_output(
            [GIT, "remote", "-v"], cwd=wd, text=True
        ).splitlines()
        if "(fetch)" in r
    }
    remote_name = origin.replace("/", "__")
    if remote_name not in remotes:
        subprocess.check_output(
            [GIT, "remote", "add", remote_name, f"https://github.com/{origin}"],
            cwd=wd,
            stderr=subprocess.STDOUT,
        )

    subprocess.check_output([GIT, "fetch", "--all"], cwd=wd, stderr=subprocess.STDOUT)

    try:
        subprocess.check_output(
            [GIT, "reset", "--hard", f"origin/{rev}"], cwd=wd, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError:
        try:
            subprocess.check_output(
                [GIT, "reset", "--hard", rev], cwd=wd, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as exc:
            raise UnknownVersionError(f"Unknown version: {rev}") from exc

    LOG.debug("Installing mypy %s from source...", rev)
    _call_uv(["pip", "install", ".", "--reinstall"], wd)
    return dest / ".venv/bin/mypy"


def _setup_copy_from_pypi(rev: str) -> Path:
    for minor in [13, 10]:  # intermediate versions never help; below 8 is too old
        try:
            path = _setup_copy_from_pypi_with_python(rev, f"3.{minor}")
        except IncompatiblePythonError:
            LOG.info("Python 3.%d too new for mypy %s.", minor, rev)
            continue
        else:
            LOG.info("Using mypy %s with python 3.%d", rev, minor)
            return path
    raise UnknownVersionError(f"Failed to find a suitable python for mypy {rev}")


def _setup_copy_from_pypi_with_python(rev: str, python: str) -> Path:
    dest = RIGHT
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
        _call_uv(
            ["pip", "install", *DEFAULT_PACKAGES, "--python", f"{venv}/bin/python"], wd
        )
        return dest / venv / "bin/mypy"


def _call_uv(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(
        [UV, *cmd],
        cwd=cwd,
        stderr=subprocess.STDOUT,
        env={"PATH": os.environ["PATH"]},
        text=True,
    )


def _parse_semver(ver: str | None) -> tuple[int, int] | tuple[int, int, int] | None:
    match (ver or "").removeprefix("v").split("."):
        case ["0", minor]:
            return (0, int(minor))
        case ["1", minor]:
            return (1, int(minor), 0)
        case ["1", minor, patch]:
            return (1, int(minor), int(patch))
        case _:
            return None

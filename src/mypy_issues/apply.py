from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from bisect import bisect_left
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import partial
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("apply")
LOG.setLevel(logging.DEBUG)

LEFT = Path("left_mypy")
RIGHT = Path("right_mypy")

RUN_OUTPUTS = Path("outputs")

OUTPUT_ROOT: Final[Path] = Path("./downloaded").resolve()
INVENTORY_ROOT: Final[Path] = OUTPUT_ROOT / "inventory.json"

GIT = "git"
UV = "uv"


class UnknownVersionError(RuntimeError):
    pass


def main() -> None:
    with INVENTORY_ROOT.open() as fd:
        inventory = json.load(fd)
    inventory = list(add_versions(inventory))

    shutil.rmtree(RUN_OUTPUTS, ignore_errors=True)
    RUN_OUTPUTS.mkdir()

    run_left(inventory)
    run_right(inventory)


def add_versions(inventory: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
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


def run_left(inventory: list[dict[str, Any]]) -> None:
    left_dest = RUN_OUTPUTS / "left"
    left_dest.mkdir()
    _setup_copy_from_source(LEFT, "master", use_mypyc=True)
    run_on_files(LEFT, [OUTPUT_ROOT / f["filename"] for f in inventory], left_dest)


def run_right(inventory: list[dict[str, str]]) -> None:
    right_dest = RUN_OUTPUTS / "right"
    right_dest.mkdir()
    get_ver = itemgetter("mypy_version")
    for ver, files_ in groupby(sorted(inventory, key=get_ver), key=get_ver):
        files = list(files_)
        try:
            _setup_copy_from_pypi(RIGHT, ver)
        except UnknownVersionError:
            LOG.warning("Failed to switch to version %s", ver)
            continue
        run_on_files(RIGHT, [OUTPUT_ROOT / f["filename"] for f in files], right_dest)


def run_on_files(mypy: Path, files: Iterable[Path], dest_root: Path) -> None:
    with ProcessPoolExecutor() as pool, TemporaryDirectory() as tmp:
        list(
            pool.map(
                partial(run_on_file, mypy=mypy, dest_root=dest_root, temp_dir=tmp),
                files,
            )
        )


def run_on_file(target: Path, mypy: Path, dest_root: Path, temp_dir: str) -> None:
    out = _run_on_file(mypy, target, temp_dir)
    (dest_root / f"{target.stem}.txt").write_text(out)


def _run_on_file(mypy: Path, target: Path, temp_dir: str) -> str:
    try:
        # fmt: off
        res = subprocess.run(
            [
                "./.venv/bin/mypy",
                str(target.resolve()),
                "--strict",
                "--cache-dir", f"{temp_dir}/{os.getpid()}",
                "--show-traceback",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=mypy,
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


def _setup_copy_from_source(
    dest: Path, rev: str = "master", *, use_mypyc: bool = False
) -> None:
    LOG.info("Switching to mypy version %s", rev)
    wd = dest.resolve()
    if not dest.is_dir():
        subprocess.check_output([
            GIT,
            "clone",
            "https://github.com/python/mypy",
            str(wd),
        ])
        subprocess.check_output([UV, "venv"], cwd=wd)
    else:
        subprocess.check_output([GIT, "fetch", "--all"], cwd=wd)

    try:
        subprocess.check_output([GIT, "checkout", rev], cwd=wd)
    except subprocess.CalledProcessError as exc:
        raise UnknownVersionError(f"Unknown version: {rev}") from exc

    uv_env = {"PATH": os.environ["PATH"]}
    if use_mypyc:
        uv_env["MYPY_USE_MYPYC"] = "1"
        uv_env["MYPYC_OPT_LEVEL"] = "3"
    subprocess.check_output(
        [UV, "pip", "install", "-e", ".", "--reinstall"], cwd=wd, env=uv_env
    )


def _setup_copy_from_pypi(dest: Path, rev: str) -> None:
    wd = dest.resolve()
    if not (dest / ".venv").is_dir():
        subprocess.check_output([UV, "venv"], cwd=wd)
    dest.mkdir(exist_ok=True)
    rev = rev.removeprefix("v")

    for maybe_rev in [rev, rev.removesuffix(".0"), f"{rev}.0"]:
        try:
            subprocess.check_output(
                [UV, "pip", "install", f"mypy=={maybe_rev}"],
                cwd=wd,
                stderr=subprocess.STDOUT,
                env={"PATH": os.environ["PATH"]},
            )
        except subprocess.CalledProcessError:
            continue
        else:
            break
    else:
        raise UnknownVersionError(f"Unknown version: {rev}")

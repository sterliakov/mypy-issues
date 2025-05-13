from __future__ import annotations

import contextlib
import dataclasses
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

from .config import (
    LEFT_OUTPUTS,
    RIGHT_OUTPUTS,
    SNIPPETS_ROOT,
    InventoryItem,
    load_inventory,
)

LOG = logging.getLogger("apply")

OldStrategy: TypeAlias = Literal["skip", "cap"]

SOURCE_MYPY_FOLDER: Final = Path("left_mypy")
PYPI_MYPY_FOLDER: Final = Path("right_mypy")

GIT: Final = "git"
UV: Final = "uv"

MIN_SUPPORTED_MYPY: Final = (0, 800)
DEFAULT_PACKAGES: Final = ("attrs", "orjson")
TIMEOUT: Final = 20

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
    left: MypyRevision | None,
    right: MypyRevision | None,
    *,
    old_strategy: OldStrategy = "skip",
    shard: int = 0,
    total_shards: int = 1,
) -> None:
    inventory = load_inventory()
    inventory = inventory[shard::total_shards]

    if right is not None and right.rev is None:
        MypyRevision("master").setup()
        inventory = list(add_versions(inventory, old_strategy))

    # Prevent interference from parent pyproject.toml
    (SNIPPETS_ROOT / "mypy.ini").write_text(MYPY_CONFIG)

    if left is not None:
        LOG.info("Running left (current) mypy...")
        run_left(inventory, left)
        LOG.info("Running left (current) mypy done.")
    if right is not None:
        LOG.info("Running right (referenced) mypy...")
        run_right(inventory, right)
        LOG.info("Running right (referenced) mypy done.")


def add_versions(
    inventory: list[InventoryItem], old_strategy: OldStrategy
) -> Iterator[InventoryItem]:
    releases = _get_releases()
    known_versions = {v.removeprefix("v") for v in releases.values()}
    dates = sorted(releases)
    for file in inventory:
        ver = file["mypy_version"]
        old_ver = ver

        if ver is not None:
            orig_semver = _parse_semver(ver.removeprefix("v"))
            if (
                orig_semver is not None
                and ".".join(map(str, orig_semver)) not in known_versions
            ):
                LOG.info("Unknown mypy version given ('%s')", ver)
                ver = None

        if ver in {None, "master"}:
            latest_release_index = bisect_left(
                dates, datetime.fromtimestamp(file["created_at"], tz=UTC)
            )
            if latest_release_index == len(dates):
                latest_release_index -= 1
            ver = releases[dates[latest_release_index]]
            if old_ver is not None:
                LOG.info(
                    "Unable to handle mypy version '%s', replaced with '%s'",
                    old_ver,
                    ver,
                )
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
        cwd=SOURCE_MYPY_FOLDER,
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


def run_left(inventory: list[InventoryItem], revision: MypyRevision) -> None:
    if LEFT_OUTPUTS.is_dir():
        shutil.rmtree(LEFT_OUTPUTS)
    (LEFT_OUTPUTS / "crashes").mkdir(parents=True)
    mypy = revision.setup()
    run_on_files(mypy, [SNIPPETS_ROOT / f["filename"] for f in inventory], LEFT_OUTPUTS)


def run_right(inventory: list[InventoryItem], revision: MypyRevision) -> None:
    if RIGHT_OUTPUTS.is_dir():
        shutil.rmtree(RIGHT_OUTPUTS)
    (RIGHT_OUTPUTS / "crashes").mkdir(parents=True)

    def get_ver(item: InventoryItem) -> tuple[int, int] | tuple[int, int, int]:
        ver = item["mypy_version"]
        assert ver
        parsed = _parse_semver(ver)
        assert parsed, ver
        return parsed

    assert revision.merge_with is None
    if revision.rev is None:
        if revision.origin != "pypi":
            raise RuntimeError("Only 'pypi' origin supported for 'guess' revision.")
        # Use guessed version for each snippet
        for ver, files_ in groupby(sorted(inventory, key=get_ver), key=get_ver):
            files = list(files_)
            try:
                mypy = MypyRevision(".".join(map(str, ver)), revision.origin).setup()
            except UnknownVersionError:
                LOG.warning("Failed to switch to version %s", ver)
                continue
            run_on_files(
                mypy,
                [SNIPPETS_ROOT / f["filename"] for f in files],
                RIGHT_OUTPUTS,
            )
    else:
        mypy = revision.setup()
        run_on_files(
            mypy, [SNIPPETS_ROOT / f["filename"] for f in inventory], RIGHT_OUTPUTS
        )


def run_on_files(mypy: Path, files: Sequence[Path], dest_root: Path) -> None:
    with contextlib.chdir(SNIPPETS_ROOT), Pool() as pool, TemporaryDirectory() as tmp:
        task_iter = pool.imap_unordered(
            partial(run_on_file, mypy=mypy, dest_root=dest_root, temp_dir=tmp),
            files,
        )
        list(zip(trange(len(files)), task_iter, strict=True))


def run_on_file(target: Path, mypy: Path, dest_root: Path, temp_dir: str) -> None:
    out, err = _run_on_file(mypy, target, temp_dir)
    if out == "SKIP":
        return
    (dest_root / f"{target.stem}.txt").write_text(out)
    if err is not None:
        (dest_root / "crashes" / f"{target.stem}.txt").write_text(err)


def _run_on_file(mypy: Path, target: Path, temp_dir: str) -> tuple[str, str | None]:
    # fmt: off
    args = [
        target.name,
        "--cache-dir", f"{temp_dir}/{os.getpid()}",
        "--python-executable", str(mypy.with_name("python").absolute()),
        "--show-traceback",
        "--strict",
        "--allow-empty-bodies",
        "--skip-cache-mtime-checks",
    ]
    # fmt: on
    try:
        res = subprocess.run(
            [str(mypy.absolute()), *args],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
            cwd=SNIPPETS_ROOT,
        )
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", "TIMEOUT")
    else:
        text = res.stdout + res.stderr
        if (
            "error: invalid syntax" in text
            or "'ASTConverter' object has no attribute 'visit_" in text
        ):
            return ("SKIP", None)
        if not text or "Traceback (most recent call last)" in text:
            return ("CRASH", text or "Please rerun manually")
        return (text, None)


@dataclasses.dataclass
class MypyRevision:
    # None indicates "guess", cannot be directly installed
    rev: str | None
    origin: str = "python/mypy"
    # (branch, origin) pair
    merge_with: tuple[str, str] | None = None

    def setup(self) -> Path:
        assert self.rev is not None
        if self.origin == "pypi" and (parts := _parse_semver(self.rev)) is not None:
            # Prefer PyPI installs
            try:
                # After parsing we have a canonical representation
                canonical = dataclasses.replace(self, rev=".".join(map(str, parts)))
                return canonical.setup_copy_from_pypi()
            except UnknownVersionError:
                LOG.info(
                    "Failed to install mypy '%s' from PyPI, trying git...", self.rev
                )

        return self.setup_copy_from_source()

    def setup_copy_from_source(self) -> Path:
        assert self.rev is not None
        LOG.debug("Switching to mypy version %s (from source)", self.rev)
        for minor in [13, 10]:  # intermediate versions never help; below 8 is too old
            try:
                path = self._setup_copy_from_source_with_python(f"3.{minor}")
            except IncompatiblePythonError:
                LOG.info("Python 3.%d too new for mypy %s.", minor, self.rev)
                continue
            else:
                LOG.info("Using mypy %s with python 3.%d", self.rev, minor)
                return path
        raise UnknownVersionError(
            f"Failed to find a suitable python for mypy {self.rev}"
        )

    def _setup_copy_from_source_with_python(self, python: str) -> Path:
        assert self.rev is not None
        dest = SOURCE_MYPY_FOLDER
        venv = f".venv{python}"
        wd = dest.absolute()
        if not dest.is_dir():
            subprocess.check_output(
                [GIT, "clone", "https://github.com/python/mypy", str(wd)],
                stderr=subprocess.STDOUT,
            )
        if not (dest / venv).is_dir():
            _call_uv(["venv", venv, "--python", python], wd)
            _call_uv(
                ["pip", "install", "--python", f"{venv}/bin/python", *DEFAULT_PACKAGES],
                wd,
            )

        remotes = {
            r.split("\t")[0]
            for r in subprocess.check_output(
                [GIT, "remote", "-v"], cwd=wd, text=True
            ).splitlines()
            if "(fetch)" in r
        }
        origin = "python/mypy" if self.origin == "pypi" else self.origin
        rev = self.rev
        self._maybe_add_remote(origin, remotes, wd)

        if self.merge_with is not None:
            _, merge_origin = self.merge_with
            self._maybe_add_remote(merge_origin, remotes, wd)

        subprocess.check_output(
            [GIT, "fetch", "--all"], cwd=wd, stderr=subprocess.STDOUT
        )

        for full_rev in [f"origin/{rev}", rev, f"v{rev}"]:
            try:
                subprocess.check_output(
                    [GIT, "reset", "--hard", full_rev], cwd=wd, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError:
                continue
            else:
                with contextlib.suppress(subprocess.CalledProcessError):
                    subprocess.check_output(
                        [GIT, "submodule", "update", "--init", "mypy/typeshed"],
                        stderr=subprocess.STDOUT,
                        cwd=wd,
                    )
                break
        else:
            raise UnknownVersionError(f"Unknown version: {rev}")

        if self.merge_with is not None:
            merge_branch, merge_origin = self.merge_with
            LOG.info("Merging %s of %r...", merge_branch, merge_origin)
            _git_config_dummy(wd)
            subprocess.check_output(
                [
                    GIT,
                    "merge",
                    f"{self._remote_name(merge_origin)}/{merge_branch}",
                    "--no-edit",
                ],
                cwd=wd,
            )

        LOG.debug("Installing mypy %s from source...", rev)
        args = ["pip", "install", "--reinstall", "--python", f"{venv}/bin/python", "."]
        try:
            _call_uv(args, wd)
        except subprocess.CalledProcessError as exc:
            raise IncompatiblePythonError(
                f"Python {python} is too new for mypy {rev}"
            ) from exc
        return (dest / venv / "bin/mypy").absolute()

    def _remote_name(self, origin: str) -> str:
        return origin.replace("/", "__")

    def _maybe_add_remote(self, origin: str, found_remotes: set[str], wd: Path) -> None:
        remote_name = self._remote_name(origin)
        if remote_name not in found_remotes:
            subprocess.check_output(
                [GIT, "remote", "add", remote_name, f"https://github.com/{origin}"],
                cwd=wd,
                stderr=subprocess.STDOUT,
            )
            found_remotes.add(remote_name)

    def setup_copy_from_pypi(self) -> Path:
        assert self.rev is not None
        assert self.merge_with is None, "Cannot merge into PyPI versions"
        for minor in [13, 10]:  # intermediate versions never help; below 8 is too old
            try:
                path = self._setup_copy_from_pypi_with_python(f"3.{minor}")
            except IncompatiblePythonError:
                LOG.info("Python 3.%d too new for mypy %s.", minor, self.rev)
                continue
            else:
                LOG.info("Using mypy %s with python 3.%d", self.rev, minor)
                return path
        raise UnknownVersionError(
            f"Failed to find a suitable python for mypy {self.rev}"
        )

    def _setup_copy_from_pypi_with_python(self, python: str) -> Path:
        assert self.rev is not None
        dest = PYPI_MYPY_FOLDER
        dest.mkdir(exist_ok=True)
        wd = dest.absolute()
        venv = f".venv{python}"
        if not (dest / venv).is_dir():
            _call_uv(["venv", venv, "--python", python], wd)
        rev = self.rev.removeprefix("v")
        args = ["pip", "install", "--python", f"{venv}/bin/python", f"mypy=={rev}"]
        try:
            _call_uv(args, wd)
        except subprocess.CalledProcessError as exc:
            if "typed-ast" in (exc.stderr or "") + (exc.stdout or ""):
                raise IncompatiblePythonError(
                    f"Python {python} is too new for mypy {rev}"
                ) from exc
            raise UnknownVersionError(f"Unknown version: {rev}") from exc
        else:
            _call_uv(
                ["pip", "install", *DEFAULT_PACKAGES, "--python", f"{venv}/bin/python"],
                wd,
            )
            return (dest / venv / "bin/mypy").absolute()


def _call_uv(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(
        [UV, *cmd],
        cwd=cwd,
        stderr=subprocess.STDOUT,
        env={"PATH": os.environ["PATH"]},
        text=True,
    )


def _parse_semver(ver: str | None) -> tuple[int, int] | tuple[int, int, int] | None:
    try:
        match (ver or "").removeprefix("v").split("."):
            case ["0", minor]:
                return (0, int(minor))
            case ["1", minor]:
                return (1, int(minor), 0)
            case ["1", minor, patch]:
                return (1, int(minor), int(patch))
            case _:
                return None
    except ValueError:
        return None


def _git_config_dummy(wd: Path) -> None:
    args = {
        "name": "Never pushes",
        "email": "does@not.exist",
    }
    for key, value in args.items():
        subprocess.check_output([GIT, "config", f"user.{key}", value], cwd=wd)

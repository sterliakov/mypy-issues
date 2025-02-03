from __future__ import annotations

import contextlib
import io
import logging
import os
import shutil
import signal
import subprocess
import sys
import types
from bisect import bisect_left
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from functools import partial
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, Literal, TypeAlias

from tqdm import trange

from mypy_issues.config import (
    LEFT_OUTPUTS,
    RIGHT_OUTPUTS,
    SNIPPETS_ROOT,
    InventoryItem,
    load_inventory,
)

LOG = logging.getLogger("apply")

OldStrategy: TypeAlias = Literal["skip", "cap"]
MypyMain: TypeAlias = Callable[[list[str]], tuple[str, str, int]]
MypySpec: TypeAlias = tuple[Path | MypyMain, Path]

LEFT: Final = Path("left_mypy")
RIGHT: Final = Path("right_mypy")

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
    *,
    left: bool = True,
    right: bool = True,
    left_rev: str = "master",
    left_origin: str = "python/mypy",
    right_rev: str | None = None,
    right_origin: str = "pypi",
    old_strategy: OldStrategy = "skip",
    try_import: bool = False,
) -> None:
    inventory = load_inventory()

    if right and right_rev is None:
        _setup_copy_from_source("master")  # To have tags available
        inventory = list(add_versions(inventory, old_strategy))

    # Prevent interference from parent pyproject.toml
    (SNIPPETS_ROOT / "mypy.ini").write_text(MYPY_CONFIG)

    if left:
        LOG.info("Running left (current) mypy...")
        run_left(inventory, left_rev, left_origin, try_import=try_import)
        LOG.info("Running left (current) mypy done.")
    if right:
        LOG.info("Running right (referenced) mypy...")
        run_right(inventory, right_rev, right_origin, try_import=try_import)
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


def run_left(
    inventory: list[InventoryItem], rev: str, origin: str, *, try_import: bool = False
) -> None:
    if LEFT_OUTPUTS.is_dir():
        shutil.rmtree(LEFT_OUTPUTS)
    (LEFT_OUTPUTS / "crashes").mkdir(parents=True)
    with _setup_mypy(rev, origin, try_import=try_import) as mypy:
        run_on_files(
            mypy, [SNIPPETS_ROOT / f["filename"] for f in inventory], LEFT_OUTPUTS
        )


def run_right(
    inventory: list[InventoryItem],
    rev: str | None,
    origin: str,
    *,
    try_import: bool = False,
) -> None:
    if RIGHT_OUTPUTS.is_dir():
        shutil.rmtree(RIGHT_OUTPUTS)
    (RIGHT_OUTPUTS / "crashes").mkdir(parents=True)

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
                with _setup_mypy(
                    ".".join(map(str, ver)), origin, try_import=try_import
                ) as mypy:
                    run_on_files(
                        mypy,
                        [SNIPPETS_ROOT / f["filename"] for f in files],
                        RIGHT_OUTPUTS,
                    )
            except UnknownVersionError:
                LOG.warning("Failed to switch to version %s", ver)
                continue
    elif rev is not None:
        with _setup_mypy(rev, origin, try_import=try_import) as mypy:
            run_on_files(
                mypy, [SNIPPETS_ROOT / f["filename"] for f in inventory], RIGHT_OUTPUTS
            )
    else:
        raise RuntimeError("Only 'pypi' origin supported for 'guess' revision.")


def run_on_files(mypy: MypySpec, files: Sequence[Path], dest_root: Path) -> None:
    with contextlib.chdir(SNIPPETS_ROOT), Pool() as pool, TemporaryDirectory() as tmp:
        task_iter = pool.imap_unordered(
            partial(run_on_file, mypy=mypy, dest_root=dest_root, temp_dir=tmp),
            files,
        )
        list(zip(trange(len(files)), task_iter, strict=True))


def run_on_file(target: Path, mypy: MypySpec, dest_root: Path, temp_dir: str) -> None:
    out, err = _run_on_file(mypy, target, temp_dir)
    if out == "SKIP":
        return
    (dest_root / f"{target.stem}.txt").write_text(out)
    if err is not None:
        (dest_root / "crashes" / f"{target.stem}.txt").write_text(err)


def _run_on_file(mypy: MypySpec, target: Path, temp_dir: str) -> tuple[str, str | None]:
    mypy_func, mypy_file = mypy
    # fmt: off
    args = [
        target.name,
        "--cache-dir", f"{temp_dir}/{os.getpid()}",
        "--python-executable", str(mypy_file.with_name("python").absolute()),
        "--show-traceback",
        "--strict",
        "--allow-empty-bodies",
        "--skip-cache-mtime-checks",
    ]
    # fmt: on
    if isinstance(mypy_func, Path):
        return _run_subprocess_on_file(mypy_func, args)
    return _run_imported_on_file(mypy_func, args)


def _run_subprocess_on_file(mypy: Path, args: list[str]) -> tuple[str, str | None]:
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


def _run_imported_on_file(mypy: MypyMain, args: list[str]) -> tuple[str, str | None]:
    class AlarmError(Exception):
        pass

    def handle_alarm(_signum: int, _frame: types.FrameType | None) -> None:
        raise AlarmError

    signal.signal(signal.SIGALRM, handle_alarm)
    iout = io.StringIO()
    ierr = io.StringIO()
    signal.alarm(TIMEOUT)
    try:
        with contextlib.redirect_stdout(iout), contextlib.redirect_stderr(ierr):
            out, err, _ = mypy(args)
    except Exception:  # noqa: BLE001
        return _classify_output(ierr.getvalue())
    else:
        out += iout.getvalue()
        err += ierr.getvalue()
        text = out + err
        return _classify_output(text)
    finally:
        signal.alarm(0)


def _classify_output(text: str, *, force_error: bool = False) -> tuple[str, str | None]:
    if "AlarmError" in text:
        return ("TIMEOUT", "TIMEOUT")
    if (
        "error: invalid syntax" in text
        or "'ASTConverter' object has no attribute 'visit_" in text
    ):
        return ("SKIP", None)
    if force_error or not text or "Traceback (most recent call last)" in text:
        return ("CRASH", text or "Please rerun manually")
    return (text, None)


@contextlib.contextmanager
def _setup_mypy(
    rev: str, origin: str, *, try_import: bool = False
) -> Iterator[MypySpec]:
    mypy = None
    if origin == "pypi" and (parts := _parse_semver(rev)) is not None:
        # Prefer PyPI installs
        try:
            mypy = _setup_copy_from_pypi(".".join(map(str, parts)))
        except UnknownVersionError:
            LOG.info("Failed to install mypy '%s' from PyPI, trying git...", rev)

    if mypy is None:
        mypy = _setup_copy_from_source(
            rev, origin if origin != "pypi" else "python/mypy"
        )

    if try_import:
        with _maybe_import_mypy(mypy) as runner:
            yield (runner, mypy)
    else:
        yield (mypy, mypy)


def _setup_copy_from_source(rev: str = "master", origin: str = "python/mypy") -> Path:
    LOG.debug("Switching to mypy version %s (from source)", rev)
    for minor in [13, 10]:  # intermediate versions never help; below 8 is too old
        try:
            path = _setup_copy_from_source_with_python(rev, origin, f"3.{minor}")
        except IncompatiblePythonError:
            LOG.info("Python 3.%d too new for mypy %s.", minor, rev)
            continue
        else:
            LOG.info("Using mypy %s with python 3.%d", rev, minor)
            return path
    raise UnknownVersionError(f"Failed to find a suitable python for mypy {rev}")


def _setup_copy_from_source_with_python(rev: str, origin: str, python: str) -> Path:
    dest = LEFT
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
            ["pip", "install", "--python", f"{venv}/bin/python", *DEFAULT_PACKAGES], wd
        )

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

    LOG.debug("Installing mypy %s from source...", rev)
    args = ["pip", "install", "--reinstall", "--python", f"{venv}/bin/python", "."]
    try:
        _call_uv(args, wd)
    except subprocess.CalledProcessError as exc:
        raise IncompatiblePythonError(
            f"Python {python} is too new for mypy {rev}"
        ) from exc
    return (dest / venv / "bin/mypy").absolute()


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
    wd = dest.absolute()
    venv = f".venv{python}"
    if not (dest / venv).is_dir():
        _call_uv(["venv", venv, "--python", python], wd)
    rev = rev.removeprefix("v")
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
            ["pip", "install", *DEFAULT_PACKAGES, "--python", f"{venv}/bin/python"], wd
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
    match (ver or "").removeprefix("v").split("."):
        case ["0", minor]:
            return (0, int(minor))
        case ["1", minor]:
            return (1, int(minor), 0)
        case ["1", minor, patch]:
            return (1, int(minor), int(patch))
        case _:
            return None


@contextlib.contextmanager
def _maybe_import_mypy(mypy: Path) -> Iterator[MypyMain | Path]:
    ref_python = (
        subprocess.check_output(
            [str(mypy.with_name("python").absolute()), "-V"], text=True
        )
        .strip()
        .removeprefix("Python ")
    )
    our_python = ".".join(map(str, sys.version_info[:3]))
    if ref_python != our_python:
        LOG.warning(
            "Using different interpreter ('%s' vs '%s'), falling back to subprocess.",
            ref_python,
            our_python,
        )
        yield mypy
        return

    path = subprocess.check_output(
        [
            str(mypy.with_name("python").absolute()),
            "-c",
            "print(':'.join(__import__('sys').path))",
        ],
        text=True,
    ).strip()

    sys.modules.pop("mypy", None)
    for k in list(sys.modules):
        if k.startswith("mypy."):
            del sys.modules[k]
    old_path = sys.path
    sys.path[:] = path.split(":")

    try:
        from mypy.api import run
    except ImportError:
        LOG.warning("Failed to import mypy, falling back to subprocess", exc_info=True)
        yield mypy
    else:
        try:
            yield run
        finally:
            sys.path[:] = old_path


def _patch_fscache() -> None:
    """This is noticeably faster, but somehow changes output on a few issues."""
    from mypy import build

    old_load = build._load_json_file  # noqa: SLF001
    global_read_cache: dict[int, dict[str, Any]] = {}

    def _load_json_file(
        file: str, manager: build.BuildManager, log_success: str, log_error: str
    ) -> dict[str, Any] | None:
        cache = global_read_cache.setdefault(os.getpid(), {})
        if file in cache:
            return cache[file]
        res = old_load(file, manager, log_success, log_error)
        if res is not None:
            cache[file] = res
        return res

    build._load_json_file = _load_json_file  # noqa: SLF001

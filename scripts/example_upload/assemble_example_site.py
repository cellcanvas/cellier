"""Add the released example scripts to a built docs site.

For every release tag, this takes the tag's ``examples/`` directory, keeps the
regular Python scripts, pins each one to that release of cellier, and writes
the result to ``<site_dir>/<tag>/examples/``.  The newest stable release is
also copied to ``<site_dir>/latest/examples/``.  A user can then run::

    uv run https://cellcanvas.github.io/cellier/latest/examples/<script>.py

A tag is published when it

- matches ``v*`` and parses as a version at or after ``FIRST_VERSION``,
- is on PyPI (so the pinned script can be installed), and
- has an ``examples/`` directory.

Marimo notebooks and files that are not ``.py`` are left out.

Usage::

    python scripts/example_upload/assemble_example_site.py site
    python scripts/example_upload/assemble_example_site.py site --offline --first-version 0.0.26
    python scripts/example_upload/assemble_example_site.py site --working-tree-version 0.0.0

``--working-tree-version`` also checks that the tracked examples of the
working tree can be pinned, and fails if one cannot.  Nothing is published
from the working tree.

See plans/examples-distribution-v2.md.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from packaging.version import InvalidVersion, Version

sys.path.insert(0, str(Path(__file__).resolve().parent))

from set_example_cellier_version import (
    NoCellierDependencyError,
    has_script_block,
    set_cellier_version,
)

# The first release whose examples are published.
FIRST_VERSION = "0.0.29"

PACKAGE = "cellier"
EXAMPLES = "examples"
PYPI_URL = "https://pypi.org/pypi/{package}/{version}/json"

_MARIMO_APP = re.compile(r"^app\s*=\s*marimo\.App\(", re.MULTILINE)

NOT_PYTHON = "not a .py file"
MARIMO = "marimo notebook"
NO_BLOCK = "no metadata block"
NO_CELLIER = "metadata block has no cellier dependency"


class PyPIError(RuntimeError):
    """PyPI could not say whether a version exists."""


@dataclass
class Result:
    """What was done with the examples of one tag (or of the working tree)."""

    name: str
    version: Version
    published: list[str] = field(default_factory=list)
    left_out: dict[str, str] = field(default_factory=dict)
    skipped: str | None = None


def _git(repo: Path, *args: str) -> bytes:
    """Run a git command in a repository and return its output."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def release_tags(repo: Path, first_version: Version) -> list[tuple[str, Version]]:
    """Return the (tag, version) of each release to publish, oldest first."""
    tags = []
    for tag in _git(repo, "tag", "--list", "v*").decode().split():
        try:
            version = Version(tag)
        except InvalidVersion:
            print(f"warning: tag {tag} is not a version, skipped", file=sys.stderr)
            continue
        if version >= first_version:
            tags.append((tag, version))
    return sorted(tags, key=lambda item: item[1])


def is_on_pypi(version: Version, urlopen: Callable = urllib.request.urlopen) -> bool:
    """Whether a version of cellier is on PyPI.

    Raises
    ------
    PyPIError
        If the answer is anything other than "yes" or "not found".  A release
        that cannot be checked must not be dropped from the site silently.
    """
    url = PYPI_URL.format(package=PACKAGE, version=version)
    try:
        with urlopen(url, timeout=30):
            return True
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise PyPIError(f"{url} returned {error.code}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise PyPIError(f"could not reach {url}: {error}") from error


def is_marimo_notebook(text: str) -> bool:
    """Whether the source of a script is a marimo notebook."""
    return _MARIMO_APP.search(text) is not None


def pinned_example(text: str, version: Version) -> tuple[str | None, str | None]:
    """Return (pinned source, None), or (None, why the script is left out)."""
    if is_marimo_notebook(text):
        return None, MARIMO
    if not has_script_block(text):
        return None, NO_BLOCK
    try:
        return set_cellier_version(text, str(version)), None
    except NoCellierDependencyError:
        return None, NO_CELLIER


def _tag_examples(repo: Path, tag: str) -> dict[str, bytes] | None:
    """Return {path relative to examples/: content} at a tag, if it has any."""
    if not _git(repo, "ls-tree", "-d", "--name-only", tag, EXAMPLES).strip():
        return None
    # The bytes as committed, whatever line-ending conversion is configured.
    archive = _git(
        repo, "-c", "core.autocrlf=false", "archive", "--format=tar", tag, EXAMPLES
    )
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.isfile():
                relative = Path(member.name).relative_to(EXAMPLES).as_posix()
                files[relative] = tar.extractfile(member).read()
    return files


def _pin_all(files: dict[str, bytes], result: Result) -> dict[str, bytes]:
    """Pin the scripts to publish; record the rest in ``result.left_out``."""
    pinned = {}
    for relative, content in sorted(files.items()):
        if not relative.endswith(".py"):
            result.left_out[relative] = NOT_PYTHON
            continue
        text, reason = pinned_example(content.decode("utf-8"), result.version)
        if text is None:
            result.left_out[relative] = reason
            continue
        pinned[relative] = text.encode("utf-8")
        result.published.append(relative)
    return pinned


def _write(files: dict[str, bytes], directory: Path) -> None:
    """Write files into a directory, replacing whatever was there."""
    if directory.exists():
        shutil.rmtree(directory)
    for relative, content in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def assemble(
    site_dir: Path,
    repo: Path,
    first_version: Version,
    exists_on_pypi: Callable[[Version], bool] = is_on_pypi,
) -> list[Result]:
    """Write the pinned examples of every release into a site directory.

    Parameters
    ----------
    site_dir : Path
        The built site.  ``<tag>/examples/`` and ``latest/examples/`` are
        written under it.
    repo : Path
        The git repository to read the tags from.
    first_version : Version
        Tags before this version are ignored.
    exists_on_pypi : Callable[[Version], bool]
        Whether a version can be installed.  Pass ``lambda version: True`` to
        skip the check.

    Returns
    -------
    list[Result]
        One entry per tag at or after ``first_version``, oldest first.
    """
    results = []
    latest = None
    for tag, version in release_tags(repo, first_version):
        result = Result(tag, version)
        results.append(result)
        if not exists_on_pypi(version):
            result.skipped = "not on PyPI"
            continue
        files = _tag_examples(repo, tag)
        if files is None:
            result.skipped = f"no {EXAMPLES}/ directory"
            continue
        pinned = _pin_all(files, result)
        if not pinned:
            result.skipped = "no script to publish"
            continue
        _write(pinned, site_dir / tag / EXAMPLES)
        if not version.is_prerelease:
            latest = pinned

    latest_dir = site_dir / "latest" / EXAMPLES
    if latest is not None:
        _write(latest, latest_dir)
    elif latest_dir.exists():
        shutil.rmtree(latest_dir)
    return results


def check_working_tree(repo: Path, version: Version) -> Result:
    """Pin the tracked examples of the working tree, without publishing them.

    The result's ``left_out`` holds the scripts that a release would not
    publish.  The ones left out for ``NO_CELLIER`` are mistakes.
    """
    result = Result("working tree", version)
    tracked = _git(repo, "ls-files", "-z", "--", EXAMPLES).decode().split("\0")
    files = {}
    for name in filter(None, tracked):
        path = repo / name
        if path.is_file():
            files[Path(name).relative_to(EXAMPLES).as_posix()] = path.read_bytes()
    _pin_all(files, result)
    return result


def _report(results: list[Result]) -> None:
    """Print what was published and what was left out."""
    for result in results:
        if result.skipped is not None:
            print(f"{result.name}  ({result.version})  skipped: {result.skipped}")
        else:
            print(
                f"{result.name}  ({result.version})  "
                f"{len(result.published)} published, {len(result.left_out)} left out"
            )
        for relative, reason in result.left_out.items():
            print(f"    left out  {relative}  ({reason})")


def main(argv: list[str] | None = None) -> int:
    """Add the released example scripts to a built docs site."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site_dir", type=Path, help="the built site directory")
    parser.add_argument(
        "--first-version",
        type=Version,
        default=Version(FIRST_VERSION),
        help=f"ignore tags before this version (default {FIRST_VERSION})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not ask PyPI whether a version exists",
    )
    parser.add_argument(
        "--working-tree-version",
        type=Version,
        default=None,
        metavar="VERSION",
        help="also check that the working tree's examples can be pinned",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="the git repository to read (default: the one holding this script)",
    )
    args = parser.parse_args(argv)

    exists_on_pypi = (lambda version: True) if args.offline else is_on_pypi
    try:
        results = assemble(args.site_dir, args.repo, args.first_version, exists_on_pypi)
    except PyPIError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if not results:
        print(f"no release tag at or after {args.first_version}")
    _report(results)

    if args.working_tree_version is not None:
        result = check_working_tree(args.repo, args.working_tree_version)
        _report([result])
        bad = [
            relative
            for relative, reason in result.left_out.items()
            if reason == NO_CELLIER
        ]
        if bad:
            print(
                f"error: {len(bad)} example(s) have a metadata block with no "
                f"{PACKAGE} dependency: {', '.join(bad)}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

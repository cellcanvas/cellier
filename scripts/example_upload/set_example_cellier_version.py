"""Pin the cellier version in the inline metadata of the example scripts.

Each example declares its dependencies in a PEP 723 block::

    # /// script
    # requires-python = ">=3.12"
    # dependencies = [
    #     "cellier[pyside]",
    #     "scikit-image",
    # ]
    # ///

This rewrites the ``cellier`` entry to ``cellier[pyside]==<version>``, so a
published copy of the script runs against the release it was published with.
Nothing else in the file is touched.

Usage::

    python scripts/example_upload/set_example_cellier_version.py 0.0.29 examples/
    python scripts/example_upload/set_example_cellier_version.py 0.0.29 examples/ --check

``--check`` reports what would change and writes nothing.  The exit code is
non-zero if a script has a metadata block with no cellier dependency in it.

See plans/examples-distribution-v2.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PACKAGE = "cellier"

_BLOCK_START = re.compile(r"^# /// script\s*$")
_BLOCK_END = re.compile(r"^# ///\s*$")
_DEPENDENCIES_START = re.compile(r"^#\s*dependencies\s*=\s*\[")
_TABLE_HEADER = re.compile(r"^#\s*\[")
# A TOML basic string (with escapes) or literal string.
_QUOTED = re.compile(
    r"""(?P<quote>")(?P<body>(?:[^"\\]|\\.)*)"|(?P<lquote>')(?P<lbody>[^']*)'"""
)
_REQUIREMENT = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?P<extras>\[[^\]]*\])?(?P<rest>.*)$"
)
_VERSION = re.compile(r"^[0-9][0-9A-Za-z.!+]*$")


class NoCellierDependencyError(ValueError):
    """A script has a metadata block that does not depend on cellier."""


def _normalized(name: str) -> str:
    """Return a package name in its PEP 503 normalized form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _script_block(lines: list[str]) -> tuple[int, int] | None:
    """Return the (first, last) line indices of the metadata block, if any."""
    for start, line in enumerate(lines):
        if not _BLOCK_START.match(line):
            continue
        for end in range(start + 1, len(lines)):
            if not lines[end].startswith("#"):
                break
            if _BLOCK_END.match(lines[end]):
                return start, end
        return None
    return None


def has_script_block(text: str) -> bool:
    """Whether the text has a PEP 723 ``script`` metadata block."""
    return _script_block(text.splitlines()) is not None


def _pin(match: re.Match[str], version: str) -> tuple[str, bool]:
    """Pin one quoted requirement if it is cellier; return (text, pinned)."""
    quote = match["quote"] or match["lquote"]
    body = match["body"] if match["quote"] else match["lbody"]
    requirement = _REQUIREMENT.match(body)
    if requirement is None or _normalized(requirement["name"]) != PACKAGE:
        return match[0], False
    extras = re.sub(r"\s+", "", requirement["extras"] or "")
    # Keep an environment marker (`; python_version < "3.14"`); drop any
    # specifier or direct reference before it.
    _, separator, marker = requirement["rest"].partition(";")
    marker = f"; {marker.strip()}" if separator else ""
    return f"{quote}{PACKAGE}{extras}=={version}{marker}{quote}", True


def set_cellier_version(text: str, version: str) -> str:
    """Pin the cellier dependency in a script's inline metadata.

    Parameters
    ----------
    text : str
        The source of the script.
    version : str
        The version to pin, without a leading ``v`` (``"0.0.29"``).

    Returns
    -------
    str
        The source with the cellier entry of the block's ``dependencies``
        rewritten to ``cellier[<extras>]==<version>``.  Returned unchanged if
        the script has no metadata block.

    Raises
    ------
    NoCellierDependencyError
        If the script has a metadata block with no cellier dependency.
    ValueError
        If ``version`` does not look like a version.
    """
    if not _VERSION.match(version):
        raise ValueError(f"not a version: {version!r}")

    lines = text.splitlines(keepends=True)
    block = _script_block(lines)
    if block is None:
        return text
    start, end = block

    pinned = False
    in_dependencies = False
    for index in range(start + 1, end):
        line = lines[index]
        if not in_dependencies:
            # Only the top-level `dependencies`, which comes before any table.
            if _TABLE_HEADER.match(line):
                break
            if not _DEPENDENCIES_START.match(line):
                continue
            in_dependencies = True
            head, bracket, entries = line.partition("[")
        else:
            head, bracket, entries = "", "", line

        def replace(match: re.Match[str]) -> str:
            nonlocal pinned
            new, was_pinned = _pin(match, version)
            pinned = pinned or was_pinned
            return new

        lines[index] = head + bracket + _QUOTED.sub(replace, entries)
        # A `]` outside the quoted entries closes the array.
        if "]" in _QUOTED.sub("", entries):
            break

    if not pinned:
        raise NoCellierDependencyError(
            f"the metadata block has no {PACKAGE} dependency to pin"
        )
    return "".join(lines)


def _example_scripts(directory: Path) -> list[Path]:
    """Return the ``.py`` files under a directory, skipping hidden directories."""
    return sorted(
        path
        for path in directory.rglob("*.py")
        if not any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(directory).parts[:-1]
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Pin the cellier version in every example script under a directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the version to pin, e.g. 0.0.29")
    parser.add_argument("directory", type=Path, help="the examples directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change without writing anything",
    )
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")
    if not _VERSION.match(args.version):
        parser.error(f"not a version (give it without a leading 'v'): {args.version}")

    failed = []
    for path in _example_scripts(args.directory):
        # Bytes in and out, so the line endings are kept as they are.
        text = path.read_bytes().decode("utf-8")
        if not has_script_block(text):
            print(f"no metadata block  {path}")
            continue
        try:
            new_text = set_cellier_version(text, args.version)
        except NoCellierDependencyError:
            print(f"NO CELLIER ENTRY   {path}")
            failed.append(path)
            continue
        if new_text == text:
            print(f"unchanged          {path}")
            continue
        if args.check:
            print(f"would pin          {path}")
        else:
            path.write_bytes(new_text.encode("utf-8"))
            print(f"pinned             {path}")

    if failed:
        print(
            f"\n{len(failed)} script(s) have a metadata block with no "
            f"{PACKAGE} dependency.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

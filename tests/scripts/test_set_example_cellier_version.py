"""Tests for scripts/example_upload/set_example_cellier_version.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "example_upload"
    / "set_example_cellier_version.py"
).resolve()


def _load_script():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()

BODY = '\nimport cellier\n\nprint("cellier[pyside]", cellier.__version__)\n'


def _source(*dependency_lines: str) -> str:
    """A script with a multi-line dependencies array."""
    lines = [
        "# /// script",
        '# requires-python = ">=3.12"',
        "# dependencies = [",
        *dependency_lines,
        "# ]",
        "# ///",
    ]
    return "\n".join(lines) + "\n" + BODY


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ('"cellier[pyside]"', '"cellier[pyside]==0.0.29"'),
        ('"cellier"', '"cellier==0.0.29"'),
        ('"cellier[pyside]>=0.0.20"', '"cellier[pyside]==0.0.29"'),
        ('"cellier >= 0.0.20, < 1"', '"cellier==0.0.29"'),
        ('"cellier[pyside]==0.0.28"', '"cellier[pyside]==0.0.29"'),
        (
            '"cellier[logging,pyside,remote]"',
            '"cellier[logging,pyside,remote]==0.0.29"',
        ),
        ('"cellier[ pyside , remote ]"', '"cellier[pyside,remote]==0.0.29"'),
        ("'cellier[pyside]'", "'cellier[pyside]==0.0.29'"),
        ('"Cellier[pyside]"', '"cellier[pyside]==0.0.29"'),
        (
            '"cellier @ git+https://github.com/cellcanvas/cellier"',
            '"cellier==0.0.29"',
        ),
    ],
)
def test_entry_is_pinned(entry, expected):
    source = _source(f"#     {entry},", '#     "scikit-image",')
    result = script.set_cellier_version(source, "0.0.29")
    assert result == _source(f"#     {expected},", '#     "scikit-image",')


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (
            "\"cellier[pyside]>=0.0.20; python_version < '3.14'\"",
            "\"cellier[pyside]==0.0.29; python_version < '3.14'\"",
        ),
        (
            "'cellier ; sys_platform != \"win32\"'",
            "'cellier==0.0.29; sys_platform != \"win32\"'",
        ),
        (
            '"cellier[pyside]; sys_platform != \\"win32\\""',
            '"cellier[pyside]==0.0.29; sys_platform != \\"win32\\""',
        ),
    ],
)
def test_marker_is_kept(entry, expected):
    source = _source(f"#     {entry},", '#     "scikit-image",')
    result = script.set_cellier_version(source, "0.0.29")
    assert result == _source(f"#     {expected},", '#     "scikit-image",')


def test_single_line_array():
    source = (
        "# /// script\n"
        '# dependencies = ["numpy", "cellier[pyqt]", "scikit-image"]\n'
        "# ///\n" + BODY
    )
    expected = (
        "# /// script\n"
        '# dependencies = ["numpy", "cellier[pyqt]==1.2.3", "scikit-image"]\n'
        "# ///\n" + BODY
    )
    assert script.set_cellier_version(source, "1.2.3") == expected


def test_other_packages_are_untouched():
    source = _source(
        '#     "cellier-extra",',
        '#     "my.cellier",',
        '#     "cellier_tools[pyside]>=1",',
        '#     "cellier[pyside]",',
    )
    expected = _source(
        '#     "cellier-extra",',
        '#     "my.cellier",',
        '#     "cellier_tools[pyside]>=1",',
        '#     "cellier[pyside]==0.0.29",',
    )
    assert script.set_cellier_version(source, "0.0.29") == expected


def test_second_run_changes_nothing():
    once = script.set_cellier_version(_source('#     "cellier[pyside]",'), "0.0.29")
    assert script.set_cellier_version(once, "0.0.29") == once


def test_no_metadata_block_is_returned_unchanged():
    source = '"""An example."""\n' + BODY
    assert not script.has_script_block(source)
    assert script.set_cellier_version(source, "0.0.29") == source


def test_block_without_cellier_raises():
    source = _source('#     "numpy",', '#     "scikit-image",')
    assert script.has_script_block(source)
    with pytest.raises(script.NoCellierDependencyError):
        script.set_cellier_version(source, "0.0.29")


def test_only_the_dependencies_array_is_edited():
    source = (
        "# /// script\n"
        "# dependencies = [\n"
        '#     "cellier[pyside]",\n'
        "# ]\n"
        '# description = "cellier"\n'
        "# [tool.uv.sources]\n"
        '# cellier = { git = "https://github.com/cellcanvas/cellier" }\n'
        "# ///\n" + BODY
    )
    expected = source.replace('"cellier[pyside]",', '"cellier[pyside]==0.0.29",', 1)
    assert script.set_cellier_version(source, "0.0.29") == expected


def test_dependencies_of_a_table_are_not_the_script_dependencies():
    source = (
        "# /// script\n"
        '# requires-python = ">=3.12"\n'
        "# [tool.other]\n"
        '# dependencies = ["cellier"]\n'
        "# ///\n" + BODY
    )
    with pytest.raises(script.NoCellierDependencyError):
        script.set_cellier_version(source, "0.0.29")


def test_line_endings_are_kept():
    source = _source('#     "cellier[pyside]",').replace("\n", "\r\n")
    expected = _source('#     "cellier[pyside]==0.0.29",').replace("\n", "\r\n")
    assert script.set_cellier_version(source, "0.0.29") == expected


@pytest.mark.parametrize("version", ["v0.0.29", "", "0.0.29 ", '0.0.29"', "latest"])
def test_bad_version_is_rejected(version):
    with pytest.raises(ValueError, match="not a version"):
        script.set_cellier_version(_source('#     "cellier",'), version)


def _write_examples(root: Path) -> dict[str, Path]:
    paths = {
        "plain": root / "plain.py",
        "nested": root / "sub" / "nested.py",
        "no_block": root / "sub" / "no_block.py",
        "hidden": root / ".ipynb_checkpoints" / "hidden.py",
        "notebook": root / "notebook.ipynb",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["plain"].write_text(_source('#     "cellier",'))
    paths["nested"].write_text(_source('#     "cellier[pyside]",'))
    paths["no_block"].write_text(BODY)
    paths["hidden"].write_text(_source('#     "cellier",'))
    paths["notebook"].write_text(_source('#     "cellier",'))
    return paths


def test_main_pins_every_script(tmp_path, capsys):
    paths = _write_examples(tmp_path)

    assert script.main(["0.0.29", str(tmp_path)]) == 0

    assert paths["plain"].read_text() == _source('#     "cellier==0.0.29",')
    assert paths["nested"].read_text() == _source('#     "cellier[pyside]==0.0.29",')
    assert paths["no_block"].read_text() == BODY
    # Hidden directories and files that are not .py are left alone.
    assert paths["hidden"].read_text() == _source('#     "cellier",')
    assert paths["notebook"].read_text() == _source('#     "cellier",')
    out = capsys.readouterr().out
    assert f"no metadata block  {paths['no_block']}" in out


def test_main_check_writes_nothing(tmp_path, capsys):
    paths = _write_examples(tmp_path)

    assert script.main(["0.0.29", str(tmp_path), "--check"]) == 0

    assert paths["plain"].read_text() == _source('#     "cellier",')
    assert f"would pin          {paths['plain']}" in capsys.readouterr().out


def test_main_fails_on_a_block_without_cellier(tmp_path, capsys):
    paths = _write_examples(tmp_path)
    bad = tmp_path / "bad.py"
    bad.write_text(_source('#     "numpy",'))

    assert script.main(["0.0.29", str(tmp_path)]) == 1

    # The other scripts are still pinned.
    assert paths["plain"].read_text() == _source('#     "cellier==0.0.29",')
    captured = capsys.readouterr()
    assert f"NO CELLIER ENTRY   {bad}" in captured.out
    assert "1 script(s)" in captured.err


def test_main_rejects_a_tag(tmp_path):
    with pytest.raises(SystemExit):
        script.main(["v0.0.29", str(tmp_path)])


def test_repository_examples_can_be_pinned():
    """Every tracked example with a metadata block has a cellier entry."""
    examples = SCRIPT.parents[2] / "examples"
    for path in script._example_scripts(examples):
        text = path.read_text()
        if script.has_script_block(text):
            assert "==9.9.9" in script.set_cellier_version(text, "9.9.9"), path

"""Tests for scripts/example_upload/assemble_example_site.py."""

import importlib.util
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
from packaging.version import Version

SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "example_upload"
    / "assemble_example_site.py"
).resolve()


def _load_script():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()

FIRST = Version("0.1.0")


def _example(dependency: str = "cellier[pyside]") -> str:
    return (
        "# /// script\n"
        "# dependencies = [\n"
        f'#     "{dependency}",\n'
        "# ]\n"
        "# ///\n"
        "\n"
        "import cellier\n"
    )


MARIMO = _example("cellier[anywidget]") + "import marimo\n\napp = marimo.App()\n"


def _always(version):
    return True


class Repo:
    """A git repository in a temporary directory."""

    def __init__(self, path: Path):
        self.path = path
        path.mkdir()
        self.git("init", "--quiet")
        self.commit({"README.md": "a repository\n"})

    def git(self, *args: str) -> None:
        subprocess.run(
            [
                "git",
                "-C",
                str(self.path),
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                "-c",
                "core.autocrlf=false",
                *args,
            ],
            check=True,
            capture_output=True,
        )

    def commit(self, files: dict, tag=None) -> None:
        """Make the working tree hold exactly these files, commit, and tag."""
        for path in self.path.rglob("*"):
            if path.is_file() and ".git" not in path.relative_to(self.path).parts:
                path.unlink()
        for name, content in files.items():
            path = self.path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode())
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "-m", "a commit")
        if tag is not None:
            self.git("tag", tag)


@pytest.fixture
def repo(tmp_path):
    return Repo(tmp_path / "repo")


@pytest.fixture
def site(tmp_path):
    return tmp_path / "site"


def _tree(directory: Path) -> dict:
    return {
        path.relative_to(directory).as_posix(): path.read_text()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_tags_before_the_first_version_are_ignored(repo, site):
    repo.commit({"examples/a.py": _example()}, tag="v0.0.9")
    repo.commit({"examples/a.py": _example()}, tag="v0.1.0")

    results = script.assemble(site, repo.path, FIRST, _always)

    assert [result.name for result in results] == ["v0.1.0"]
    assert sorted(path.name for path in site.iterdir()) == ["latest", "v0.1.0"]


def test_scripts_are_pinned_to_their_own_release(repo, site):
    repo.commit(
        {"examples/a.py": _example(), "examples/sub/b.py": _example("cellier")},
        tag="v0.1.0",
    )
    repo.commit({"examples/a.py": _example("cellier[pyqt]")}, tag="v0.2.0")

    script.assemble(site, repo.path, FIRST, _always)

    assert _tree(site) == {
        "v0.1.0/examples/a.py": _example("cellier[pyside]==0.1.0"),
        "v0.1.0/examples/sub/b.py": _example("cellier==0.1.0"),
        # sub/b.py was deleted before v0.2.0, so it is gone from then on.
        "v0.2.0/examples/a.py": _example("cellier[pyqt]==0.2.0"),
        "latest/examples/a.py": _example("cellier[pyqt]==0.2.0"),
    }


def test_tags_are_ordered_by_version_not_by_name(repo, site):
    repo.commit({"examples/a.py": _example("cellier[ten]")}, tag="v0.10.0")
    repo.commit({"examples/a.py": _example("cellier[nine]")}, tag="v0.9.0")

    results = script.assemble(site, repo.path, FIRST, _always)

    assert [result.name for result in results] == ["v0.9.0", "v0.10.0"]
    assert _tree(site / "latest") == {"examples/a.py": _example("cellier[ten]==0.10.0")}


def test_prerelease_gets_a_directory_but_not_latest(repo, site):
    repo.commit({"examples/a.py": _example()}, tag="v0.1.0")
    repo.commit({"examples/a.py": _example("cellier[pyqt]")}, tag="v0.2.0rc1")

    script.assemble(site, repo.path, FIRST, _always)

    assert _tree(site) == {
        "v0.1.0/examples/a.py": _example("cellier[pyside]==0.1.0"),
        "v0.2.0rc1/examples/a.py": _example("cellier[pyqt]==0.2.0rc1"),
        "latest/examples/a.py": _example("cellier[pyside]==0.1.0"),
    }


def test_only_prereleases_means_no_latest(repo, site):
    repo.commit({"examples/a.py": _example()}, tag="v0.2.0rc1")
    # A stale latest/ from an earlier build must not survive.
    (site / "latest" / "examples").mkdir(parents=True)
    (site / "latest" / "examples" / "old.py").write_text("old")

    script.assemble(site, repo.path, FIRST, _always)

    assert sorted(_tree(site)) == ["v0.2.0rc1/examples/a.py"]


def test_tag_is_the_directory_and_the_pin_is_normalized(repo, site):
    repo.commit({"examples/a.py": _example()}, tag="v0.2.0-rc1")

    script.assemble(site, repo.path, FIRST, _always)

    assert _tree(site) == {
        "v0.2.0-rc1/examples/a.py": _example("cellier[pyside]==0.2.0rc1")
    }


def test_marimo_notebooks_and_other_files_are_left_out(repo, site):
    repo.commit(
        {
            "examples/a.py": _example(),
            "examples/a_marimo.py": MARIMO,
            "examples/a.ipynb": "{}",
            "examples/no_block.py": "import cellier\n",
            "examples/no_cellier.py": _example("numpy"),
        },
        tag="v0.1.0",
    )

    (result,) = script.assemble(site, repo.path, FIRST, _always)

    assert sorted(_tree(site / "v0.1.0")) == ["examples/a.py"]
    assert result.published == ["a.py"]
    assert result.left_out == {
        "a.ipynb": script.NOT_PYTHON,
        "a_marimo.py": script.MARIMO,
        "no_block.py": script.NO_BLOCK,
        "no_cellier.py": script.NO_CELLIER,
    }


def test_version_that_is_not_on_pypi_is_skipped(repo, site):
    repo.commit({"examples/a.py": _example()}, tag="v0.1.0")
    repo.commit({"examples/a.py": _example()}, tag="v0.2.0")

    results = script.assemble(
        site, repo.path, FIRST, lambda version: version != Version("0.2.0")
    )

    assert [result.skipped for result in results] == [None, "not on PyPI"]
    assert sorted(path.name for path in site.iterdir()) == ["latest", "v0.1.0"]
    assert _tree(site / "latest") == {
        "examples/a.py": _example("cellier[pyside]==0.1.0")
    }


def test_tag_without_examples_is_skipped(repo, site):
    repo.commit({"src/a.py": _example()}, tag="v0.1.0")
    repo.commit({"examples/only.ipynb": "{}"}, tag="v0.2.0")

    results = script.assemble(site, repo.path, FIRST, _always)

    assert [result.skipped for result in results] == [
        "no examples/ directory",
        "no script to publish",
    ]
    assert not site.exists()


def test_tags_that_are_not_versions_are_ignored(repo, site, capsys):
    repo.commit({"examples/a.py": _example()}, tag="vnext")
    repo.commit({"examples/a.py": _example()}, tag="other-1.0.0")

    assert script.assemble(site, repo.path, FIRST, _always) == []
    assert "vnext is not a version" in capsys.readouterr().err


def test_a_rebuild_replaces_the_previous_output(repo, site):
    repo.commit({"examples/a.py": _example()}, tag="v0.1.0")
    stale = site / "v0.1.0" / "examples" / "stale.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale")
    docs = site / "index.html"
    docs.write_text("docs")

    script.assemble(site, repo.path, FIRST, _always)

    assert not stale.exists()
    assert docs.read_text() == "docs"


def _http_error(code):
    return urllib.error.HTTPError("url", code, "message", None, None)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_is_on_pypi():
    urls = []

    def found(url, timeout):
        urls.append(url)
        return _Response()

    def raises(error):
        def urlopen(url, timeout):
            raise error

        return urlopen

    version = Version("0.2.0rc1")
    assert script.is_on_pypi(version, found)
    assert urls == ["https://pypi.org/pypi/cellier/0.2.0rc1/json"]
    assert not script.is_on_pypi(version, raises(_http_error(404)))
    with pytest.raises(script.PyPIError):
        script.is_on_pypi(version, raises(_http_error(503)))
    with pytest.raises(script.PyPIError):
        script.is_on_pypi(version, raises(urllib.error.URLError("no network")))


def test_main_fails_when_pypi_cannot_be_reached(repo, site, monkeypatch, capsys):
    repo.commit({"examples/a.py": _example()}, tag="v0.1.0")

    def unreachable(version):
        raise script.PyPIError("could not reach PyPI")

    monkeypatch.setattr(script, "is_on_pypi", unreachable)
    arguments = [str(site), "--repo", str(repo.path), "--first-version", "0.1.0"]

    assert script.main(arguments) == 1
    assert "could not reach PyPI" in capsys.readouterr().err
    assert script.main([*arguments, "--offline"]) == 0
    assert (site / "latest" / "examples" / "a.py").exists()


def test_working_tree_check(repo, site, capsys):
    repo.commit({"examples/a.py": _example(), "examples/a_marimo.py": MARIMO})
    # Untracked files are not part of a release, so they are not checked.
    (repo.path / "examples" / "untracked.py").write_text(_example("numpy"))
    arguments = [
        str(site),
        "--repo",
        str(repo.path),
        "--offline",
        "--working-tree-version",
        "0.0.0",
    ]

    assert script.main(arguments) == 0
    assert "working tree  (0.0.0)  1 published, 1 left out" in capsys.readouterr().out
    # The check publishes nothing and leaves the working tree as it was.
    assert not site.exists()
    assert (repo.path / "examples" / "a.py").read_text() == _example()

    repo.commit({"examples/a.py": _example(), "examples/bad.py": _example("numpy")})

    assert script.main(arguments) == 1
    assert "bad.py" in capsys.readouterr().err

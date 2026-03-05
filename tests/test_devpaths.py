from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from qubit_experiment import devpaths


def test_workspace_path_resolves_explicit_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    workspace = repo / "projects" / "selectiveRIP-dev"
    descriptor = workspace / "configs" / "descriptors" / "1port.yaml"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("instruments: []\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='tmp'\n", encoding="utf-8")
    monkeypatch.setattr(devpaths, "_REPO_ROOT", repo)

    resolved = devpaths.workspace_path(
        "configs/descriptors/1port.yaml",
        workspace="selectiveRIP-dev",
    )

    assert resolved == descriptor.resolve()


def test_workspace_path_can_infer_workspace_from_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    workspace = repo / "projects" / "selectiveRIP-dev"
    notebooks = workspace / "notebooks"
    descriptor = workspace / "configs" / "descriptors" / "1port.yaml"
    notebooks.mkdir(parents=True)
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("instruments: []\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='tmp'\n", encoding="utf-8")
    monkeypatch.setattr(devpaths, "_REPO_ROOT", repo)

    resolved = devpaths.workspace_path(
        "configs/descriptors/1port.yaml",
        start=notebooks,
    )

    assert resolved == descriptor.resolve()


def test_workspace_path_requires_workspace_when_start_is_outside_projects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='tmp'\n", encoding="utf-8")
    monkeypatch.setattr(devpaths, "_REPO_ROOT", repo)

    with pytest.raises(ValueError, match="Pass workspace="):
        devpaths.workspace_path("configs/descriptors/1port.yaml", start=repo)


def test_devpaths_submodule_is_exposed_from_package_namespace() -> None:
    package = import_module("qubit_experiment")

    assert hasattr(package.devpaths, "workspace_path")

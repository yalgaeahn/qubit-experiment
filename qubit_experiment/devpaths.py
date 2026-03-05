"""Helpers for resolving local development workspace paths.

These helpers support the local-only `projects/<workspace>/` sandboxes used for
real-data validation while keeping reusable code in `qubit_experiment/`.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Return the repository root for this editable checkout."""

    if not (_REPO_ROOT / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"Could not locate repository root from package path: {_REPO_ROOT}"
        )
    return _REPO_ROOT


def projects_root() -> Path:
    """Return the local-only development workspace root."""

    return repo_root() / "projects"


def infer_workspace_root(start: str | Path | None = None) -> Path | None:
    """Infer the active `projects/<workspace>` root from a starting directory."""

    origin = Path(start).expanduser().resolve() if start is not None else Path.cwd().resolve()
    root = projects_root()
    for candidate in (origin, *origin.parents):
        if candidate.parent == root:
            return candidate
    return None


def workspace_root(
    workspace: str | None = None,
    *,
    start: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Return a workspace root, explicitly or by inferring it from `start`."""

    if workspace is not None:
        path = (projects_root() / workspace).resolve()
    else:
        path = infer_workspace_root(start)
        if path is None:
            raise ValueError(
                "Could not infer a local workspace from the current directory. "
                "Pass workspace='selectiveRIP-dev' (or your workspace name) explicitly."
            )

    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def workspace_path(
    relative_path: str | Path,
    *,
    workspace: str | None = None,
    start: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a path inside `projects/<workspace>/`.

    Use `workspace=...` for notebooks launched outside the workspace directory.
    """

    path = Path(relative_path).expanduser()
    if path.is_absolute():
        raise ValueError("workspace_path expects a path relative to the workspace root.")

    resolved = (workspace_root(workspace, start=start, must_exist=must_exist) / path).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved

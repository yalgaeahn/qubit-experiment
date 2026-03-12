"""Unified Matplotlib plotting theme utilities for analysis modules."""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any, TypeVar

import matplotlib

ENV_THEME_VAR = "QUBIT_EXPERIMENT_PLOT_THEME"
DEFAULT_PLOT_THEME = "high_contrast_publication_light"

_DARK_PROP_CYCLE = [
    "#4FC3F7",
    "#FF8A65",
    "#CE93D8",
    "#80CBC4",
    "#FFD54F",
    "#81D4FA",
]

_LIGHT_PROP_CYCLE = [
    "#1F77B4",
    "#E4572E",
    "#2E8B57",
    "#9467BD",
    "#8C564B",
    "#17BECF",
]

THEME_PRESETS: dict[str, dict[str, Any]] = {
    "high_contrast_publication_dark": {
        "figure.facecolor": "#111111",
        "figure.edgecolor": "#111111",
        "axes.facecolor": "#151515",
        "axes.edgecolor": "#DCDCDC",
        "axes.labelcolor": "#EDEDED",
        "axes.titlecolor": "#F5F5F5",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.25,
        "axes.prop_cycle": matplotlib.cycler(color=_DARK_PROP_CYCLE),
        "text.color": "#EDEDED",
        "xtick.color": "#D9D9D9",
        "ytick.color": "#D9D9D9",
        "grid.color": "#777777",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.85,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.facecolor": "#111111",
        "legend.edgecolor": "#111111",
        "legend.labelcolor": "#EDEDED",
        "lines.linewidth": 2.1,
        "lines.markersize": 5.5,
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "savefig.facecolor": "#111111",
        "savefig.edgecolor": "#111111",
        "savefig.transparent": False,
    },
    "high_contrast_publication_light": {
        "figure.facecolor": "#FFFFFF",
        "figure.edgecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#202020",
        "axes.labelcolor": "#101010",
        "axes.titlecolor": "#080808",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.25,
        "axes.prop_cycle": matplotlib.cycler(color=_LIGHT_PROP_CYCLE),
        "text.color": "#111111",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#999999",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.85,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#FFFFFF",
        "legend.labelcolor": "#121212",
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
        "savefig.transparent": False,
    },
    "clean_scientific": {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "axes.titlecolor": "#111111",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.prop_cycle": matplotlib.cycler(color=_LIGHT_PROP_CYCLE),
        "text.color": "#111111",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#A0A0A0",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "font.size": 10.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "savefig.facecolor": "#FFFFFF",
        "savefig.transparent": False,
    },
    "minimal_no_grid": {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#DCDCDC",
        "axes.labelcolor": "#101010",
        "axes.titlecolor": "#101010",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.prop_cycle": matplotlib.cycler(color=_LIGHT_PROP_CYCLE),
        "text.color": "#101010",
        "xtick.color": "#2A2A2A",
        "ytick.color": "#2A2A2A",
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "font.size": 10.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "savefig.facecolor": "#FFFFFF",
        "savefig.transparent": False,
    },
}

_SEMANTIC_COLOR_TABLE: dict[str, dict[str, str]] = {
    "high_contrast_publication_dark": {
        "g": "#4FC3F7",
        "e": "#FF8A65",
        "f": "#CE93D8",
        "boundary": "#E8E8E8",
        "text_box": "#1E1E1E",
        "text_box_edge": "#606060",
        "text_box_text": "#F0F0F0",
        "fit": "#FFD54F",
        "warning": "#FFB74D",
        "ok": "#80CBC4",
        "fail": "#EF5350",
    },
    "high_contrast_publication_light": {
        "g": "#1F77B4",
        "e": "#D62728",
        "f": "#9467BD",
        "boundary": "#1D1D1D",
        "text_box": "#FFFFFF",
        "text_box_edge": "#A0A0A0",
        "text_box_text": "#111111",
        "fit": "#E4572E",
        "warning": "#E67E22",
        "ok": "#2E8B57",
        "fail": "#C0392B",
    },
    "clean_scientific": {
        "g": "#1F77B4",
        "e": "#D62728",
        "f": "#2CA02C",
        "boundary": "#1D1D1D",
        "text_box": "#FFFFFF",
        "text_box_edge": "#B0B0B0",
        "text_box_text": "#111111",
        "fit": "#E4572E",
        "warning": "#E67E22",
        "ok": "#2E8B57",
        "fail": "#C0392B",
    },
    "minimal_no_grid": {
        "g": "#2CA58D",
        "e": "#E4572E",
        "f": "#6A4C93",
        "boundary": "#2A2A2A",
        "text_box": "#FFFFFF",
        "text_box_edge": "#D0D0D0",
        "text_box_text": "#101010",
        "fit": "#3A86FF",
        "warning": "#D97706",
        "ok": "#2CA58D",
        "fail": "#B22222",
    },
}

_DEFAULT_THEME_NAME = DEFAULT_PLOT_THEME
_DEFAULT_RC_OVERRIDES: dict[str, Any] = {}

F = TypeVar("F", bound=Callable[..., Any])


def list_plot_themes() -> tuple[str, ...]:
    """Return all supported theme preset names."""
    return tuple(THEME_PRESETS.keys())


def _normalize_theme_name(theme: str) -> str:
    theme_name = str(theme).strip()
    if theme_name not in THEME_PRESETS:
        supported = ", ".join(sorted(THEME_PRESETS))
        raise ValueError(f"Unknown plot theme: {theme_name!r}. Supported themes: {supported}.")
    return theme_name


def _parse_env_theme() -> str | None:
    raw = os.getenv(ENV_THEME_VAR)
    if raw is None or raw.strip() == "":
        return None
    return _normalize_theme_name(raw)


def get_default_plot_theme() -> str:
    """Return the configured default plot theme (excluding env override)."""
    return _DEFAULT_THEME_NAME


def _resolve_theme_name(
    explicit_theme: str | None = None,
    options_theme: str | None = None,
) -> str:
    if explicit_theme is not None:
        return _normalize_theme_name(explicit_theme)
    if options_theme is not None:
        return _normalize_theme_name(options_theme)
    env_theme = _parse_env_theme()
    if env_theme is not None:
        return env_theme
    return _DEFAULT_THEME_NAME


def set_default_plot_theme(theme: str, rc_overrides: Mapping[str, Any] | None = None) -> None:
    """Set the process-local default theme and optional default rc overrides."""
    global _DEFAULT_THEME_NAME, _DEFAULT_RC_OVERRIDES
    _DEFAULT_THEME_NAME = _normalize_theme_name(theme)
    _DEFAULT_RC_OVERRIDES = dict(rc_overrides or {})


def _normalize_rc_overrides(rc_overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if rc_overrides is None:
        return {}
    return dict(rc_overrides)


def get_plot_theme_rc_params(
    theme: str | None = None,
    rc_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the resolved rcParams dictionary for a theme and overrides."""
    theme_name = _resolve_theme_name(explicit_theme=theme)
    rc = dict(THEME_PRESETS[theme_name])
    rc.update(_DEFAULT_RC_OVERRIDES)
    rc.update(_normalize_rc_overrides(rc_overrides))
    return rc


def get_semantic_color(name: str, theme: str | None = None) -> str:
    """Resolve semantic color token for the active theme."""
    theme_name = _resolve_theme_name(explicit_theme=theme)
    palette = _SEMANTIC_COLOR_TABLE.get(theme_name, {})
    if name in palette:
        return palette[name]
    fallback = _SEMANTIC_COLOR_TABLE[DEFAULT_PLOT_THEME]
    return fallback.get(name, "#FFFFFF")


def get_state_color(state: str, theme: str | None = None) -> str:
    """Return semantic color for logical state label g/e/f."""
    return get_semantic_color(str(state).lower().strip(), theme=theme)


@contextmanager
def plot_theme_context(
    theme: str | None = None,
    rc_overrides: Mapping[str, Any] | None = None,
):
    """Apply unified plotting theme in a temporary matplotlib rc context."""
    rc = get_plot_theme_rc_params(theme=theme, rc_overrides=rc_overrides)
    with matplotlib.rc_context(rc=rc):
        yield


def with_plot_theme(
    func: F | None = None,
    *,
    theme: str | None = None,
    rc_overrides: Mapping[str, Any] | None = None,
) -> F | Callable[[F], F]:
    """Decorator applying plot theme around function execution.

    Theme selection priority:
    1. Explicit decorator ``theme`` argument
    2. ``options.plot_theme`` (if function receives ``options``)
    3. Environment variable ``QUBIT_EXPERIMENT_PLOT_THEME``
    4. Process-local default set via ``set_default_plot_theme``
    """

    def _decorator(target: F) -> F:
        signature = inspect.signature(target)
        has_options_param = "options" in signature.parameters

        @functools.wraps(target)
        def _wrapped(*args: Any, **kwargs: Any):
            options = kwargs.get("options")
            if options is None and has_options_param:
                try:
                    bound = signature.bind_partial(*args, **kwargs)
                except TypeError:
                    bound = None
                if bound is not None:
                    options = bound.arguments.get("options")

            options_theme = getattr(options, "plot_theme", None) if options is not None else None
            options_overrides = (
                getattr(options, "plot_theme_overrides", None) if options is not None else None
            )
            resolved_theme = _resolve_theme_name(
                explicit_theme=theme,
                options_theme=options_theme,
            )
            merged_overrides: dict[str, Any] = {}
            merged_overrides.update(_normalize_rc_overrides(rc_overrides))
            merged_overrides.update(_normalize_rc_overrides(options_overrides))

            with plot_theme_context(theme=resolved_theme, rc_overrides=merged_overrides):
                return target(*args, **kwargs)

        return _wrapped  # type: ignore[return-value]

    if func is None:
        return _decorator
    return _decorator(func)


__all__ = [
    "ENV_THEME_VAR",
    "DEFAULT_PLOT_THEME",
    "get_default_plot_theme",
    "set_default_plot_theme",
    "plot_theme_context",
    "with_plot_theme",
    "list_plot_themes",
    "get_plot_theme_rc_params",
    "get_semantic_color",
    "get_state_color",
]

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from analysis.plot_theme import (
    DEFAULT_PLOT_THEME,
    ENV_THEME_VAR,
    get_semantic_color,
    plot_theme_context,
    set_default_plot_theme,
    with_plot_theme,
)


@dataclass
class _Options:
    plot_theme: str | None = None
    plot_theme_overrides: dict | None = None


@pytest.fixture(autouse=True)
def _reset_theme(monkeypatch):
    monkeypatch.delenv(ENV_THEME_VAR, raising=False)
    set_default_plot_theme(DEFAULT_PLOT_THEME)
    yield
    set_default_plot_theme(DEFAULT_PLOT_THEME)


@with_plot_theme
def _capture_facecolors(options=None):
    fig, ax = plt.subplots()
    fig_face = to_hex(fig.get_facecolor())
    ax_face = to_hex(ax.get_facecolor())
    plt.close(fig)
    return fig_face, ax_face


@with_plot_theme(theme="minimal_no_grid")
def _capture_explicit_theme(options=None):
    fig, ax = plt.subplots()
    fig_face = to_hex(fig.get_facecolor())
    plt.close(fig)
    return fig_face


def test_decorator_applies_theme_and_restores_rcparams() -> None:
    before = to_hex(plt.rcParams["figure.facecolor"])
    fig_face, ax_face = _capture_facecolors()
    after = to_hex(plt.rcParams["figure.facecolor"])

    assert fig_face == "#111111"
    assert ax_face == "#151515"
    assert after == before


def test_options_theme_overrides_default_and_env(monkeypatch) -> None:
    set_default_plot_theme("clean_scientific")
    monkeypatch.setenv(ENV_THEME_VAR, "minimal_no_grid")

    fig_face, ax_face = _capture_facecolors(
        _Options(plot_theme="high_contrast_publication_light")
    )
    assert fig_face == "#ffffff"
    assert ax_face == "#ffffff"


def test_explicit_decorator_theme_has_highest_priority(monkeypatch) -> None:
    monkeypatch.setenv(ENV_THEME_VAR, "high_contrast_publication_dark")
    face = _capture_explicit_theme(_Options(plot_theme="high_contrast_publication_light"))
    assert face == "#ffffff"


def test_options_rc_overrides_are_applied() -> None:
    target_axes_face = "#123456"

    fig_face, ax_face = _capture_facecolors(
        _Options(plot_theme_overrides={"axes.facecolor": target_axes_face})
    )
    assert fig_face == "#111111"
    assert ax_face == target_axes_face


def test_context_manager_and_decorator_can_be_nested() -> None:
    with plot_theme_context(theme="high_contrast_publication_light"):
        fig_face, _ = _capture_facecolors(_Options(plot_theme="high_contrast_publication_dark"))
        assert fig_face == "#111111"


def test_semantic_color_visible_under_decorator() -> None:
    @with_plot_theme
    def _boundary_color(options=None):
        return get_semantic_color("boundary")

    assert _boundary_color() == "#E8E8E8"

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from qubit_experiment.analysis.plot_theme import (
    DEFAULT_PLOT_THEME,
    ENV_THEME_VAR,
    get_default_plot_theme,
    get_plot_theme_rc_params,
    list_plot_themes,
    plot_theme_context,
    set_default_plot_theme,
)


@pytest.fixture(autouse=True)
def _reset_theme(monkeypatch):
    monkeypatch.delenv(ENV_THEME_VAR, raising=False)
    set_default_plot_theme(DEFAULT_PLOT_THEME)
    yield
    set_default_plot_theme(DEFAULT_PLOT_THEME)


def test_theme_list_and_default_present() -> None:
    themes = list_plot_themes()
    assert DEFAULT_PLOT_THEME in themes
    assert "high_contrast_publication_light" in themes
    assert "clean_scientific" in themes
    assert "minimal_no_grid" in themes


def test_invalid_theme_raises() -> None:
    with pytest.raises(ValueError, match="Unknown plot theme"):
        set_default_plot_theme("does_not_exist")


def test_plot_theme_context_applies_and_restores() -> None:
    original_fig_face = to_hex(plt.rcParams["figure.facecolor"])
    original_axes_face = to_hex(plt.rcParams["axes.facecolor"])

    with plot_theme_context(theme="high_contrast_publication_light"):
        assert to_hex(plt.rcParams["figure.facecolor"]) == "#ffffff"
        assert to_hex(plt.rcParams["axes.facecolor"]) == "#ffffff"

    assert to_hex(plt.rcParams["figure.facecolor"]) == original_fig_face
    assert to_hex(plt.rcParams["axes.facecolor"]) == original_axes_face


def test_set_default_theme_with_overrides_updates_rc_resolution() -> None:
    set_default_plot_theme(
        "clean_scientific",
        rc_overrides={"lines.linewidth": 3.25, "axes.facecolor": "#f0f0f0"},
    )

    assert get_default_plot_theme() == "clean_scientific"
    rc = get_plot_theme_rc_params()
    assert rc["lines.linewidth"] == 3.25
    assert to_hex(rc["axes.facecolor"]) == "#f0f0f0"


def test_env_override_takes_precedence(monkeypatch) -> None:
    set_default_plot_theme("clean_scientific")
    monkeypatch.setenv(ENV_THEME_VAR, "minimal_no_grid")

    rc = get_plot_theme_rc_params()
    assert rc["axes.spines.left"] is False
    assert to_hex(rc["figure.facecolor"]) == "#ffffff"


def test_invalid_env_theme_raises(monkeypatch) -> None:
    monkeypatch.setenv(ENV_THEME_VAR, "unknown_theme")
    with pytest.raises(ValueError, match="Unknown plot theme"):
        get_plot_theme_rc_params()

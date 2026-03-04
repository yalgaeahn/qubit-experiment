"""Analysis package exports."""

from analysis.plot_theme import (
    DEFAULT_PLOT_THEME,
    ENV_THEME_VAR,
    get_default_plot_theme,
    get_plot_theme_rc_params,
    get_semantic_color,
    get_state_color,
    list_plot_themes,
    plot_theme_context,
    set_default_plot_theme,
    with_plot_theme,
)

__all__ = [
    "DEFAULT_PLOT_THEME",
    "ENV_THEME_VAR",
    "get_default_plot_theme",
    "set_default_plot_theme",
    "plot_theme_context",
    "with_plot_theme",
    "list_plot_themes",
    "get_plot_theme_rc_params",
    "get_semantic_color",
    "get_state_color",
]

try:
    from analysis.plotting_helpers import (
        PlotRawDataOptions,
        plot_raw_complex_data_1d,
        timestamped_title,
    )

    __all__.extend(
        [
            "PlotRawDataOptions",
            "plot_raw_complex_data_1d",
            "timestamped_title",
        ]
    )
except ModuleNotFoundError:
    # Keep plot-theme utilities importable even without LabOne Q dependencies.
    pass

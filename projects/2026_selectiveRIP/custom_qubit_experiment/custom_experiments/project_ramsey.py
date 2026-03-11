"""Project-local Ramsey experiment wrapper.

Use this module as the starting point for SelectiveRIP-specific variants before
forking or heavily editing the shared package implementation.
"""

from __future__ import annotations

from qubit_experiment.experiments.ramsey import (
    RamseyWorkflowOptions,
    create_experiment,
    experiment_workflow,
)


def recommended_options(
    *,
    count: int = 1024,
    update: bool = False,
    use_cal_traces: bool = True,
    echo: bool = False,
    refocus_qop: str = "y180",
):
    """Build a Ramsey workflow options builder with project-local defaults."""
    options = experiment_workflow.options()
    options.count(count)
    options.update(update)
    options.use_cal_traces(use_cal_traces)
    options.echo(echo)
    options.refocus_qop(refocus_qop)
    return options


__all__ = [
    "RamseyWorkflowOptions",
    "create_experiment",
    "experiment_workflow",
    "recommended_options",
]


# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Experiment definition for a cavity pi/nopi cross-Kerr check.

This adapts the `cavity_pi_nopi` sequence from ``Bosonic_experiments.py`` to the
LabOne Q Applications workflow style. The experiment consists of a frequency
offset sweep on a cavity displacement pulse while toggling whether qubit pi
pulses are applied (``pi`` vs ``nopi``) and whether an additional conditional pi
is played for a cross-Kerr check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    ExperimentSignal,
    LinearSweepParameter,
    Pulse,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import compile_experiment, run_experiment

from laboneq_applications.analysis.exp_cavity_pi_nopi import analysis_workflow
from laboneq_applications.experiments.options import BaseExperimentOptions

if TYPE_CHECKING:
    from laboneq.dsl.session import Session
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults


@workflow.task_options(base_class=BaseExperimentOptions)
class CavityPiNoPiExperimentOptions:
    """Options for the cavity-pi/nopi experiment.

    Attributes:
        freq_start:
            Start of the frequency offset sweep in Hz.
        freq_stop:
            Stop of the frequency offset sweep in Hz.
        freq_count:
            Number of points in the frequency sweep (inclusive).
        base_frequency:
            Base cavity frequency in Hz. The sweep offsets are added to this
            value when programming the displacement pulse.
        relax_time:
            Optional relax time after each acquisition in seconds.
        auto_chunking:
            Whether to enable auto-chunking on the outermost sweep.
    """

    freq_start: float = workflow.option_field(
        0.0, description="Start of the frequency offset sweep in Hz."
    )
    freq_stop: float = workflow.option_field(
        0.0, description="Stop of the frequency offset sweep in Hz."
    )
    freq_count: int = workflow.option_field(
        1, description="Number of points in the frequency sweep (inclusive)."
    )
    base_frequency: float = workflow.option_field(
        0.0,
        description=(
            "Base cavity frequency in Hz. Sweep offsets are added to this value."
        ),
    )
    relax_time: float | None = workflow.option_field(
        None, description="Optional relax time after each acquisition in seconds."
    )
    auto_chunking: bool = workflow.option_field(
        True, description="Whether to enable auto-chunking on the frequency sweep."
    )


@workflow.workflow_options
class CavityPiNoPiWorkflowOptions:
    """Workflow options for the cavity pi/nopi experiment."""

    do_analysis: bool = workflow.option_field(
        True, description="Whether to run the analysis workflow."
    )


@workflow.workflow(name="exp_cavity_pi_nopi")
def experiment_workflow(
    session: Session,
    *,
    signal_map: dict[str, object],
    cavity_pulse: Pulse | dict,
    qubit_pi_pulse: Pulse | dict,
    readout_pulse: Pulse | dict,
    integration_kernel: Pulse | dict,
    conditional_pi_pulse: Pulse | dict | None = None,
    second_qubit_pi_pulse: Pulse | dict | None = None,
    options: CavityPiNoPiWorkflowOptions | None = None,
    experiment_options: CavityPiNoPiExperimentOptions | None = None,
) -> RunExperimentResults:
    """Workflow for the cavity pi/nopi experiment.

    Arguments:
        session:
            LabOne Q session used to compile and run the experiment.
        signal_map:
            Mapping from experiment signal names (``drive``, ``drive2``,
            ``cavity_drive``, ``measure``, ``acquire``) to logical signals on
            the device setup.
        cavity_pulse:
            Pulse played on ``cavity_drive`` during the frequency sweep.
        qubit_pi_pulse:
            Pi pulse played on ``drive`` for the ``pi`` cases.
        readout_pulse:
            Readout pulse to use on ``measure``.
        integration_kernel:
            Integration kernel for the acquisition on ``acquire``.
        conditional_pi_pulse:
            Optional conditional pi pulse for the cross-Kerr check. If ``None``,
            the cross-Kerr sweep still runs but no pulse is played.
        second_qubit_pi_pulse:
            Optional pi pulse for a second qubit on ``drive2`` for the ``pi``
            cases.
        options:
            Workflow options controlling whether analysis is executed.
        experiment_options:
            Experiment options configuring sweep ranges and repetition settings.
    """
    wf_opts = (
        CavityPiNoPiWorkflowOptions() if options is None else options
    )
    exp_opts = (
        CavityPiNoPiExperimentOptions()
        if experiment_options is None
        else experiment_options
    )

    frequency_offsets = np.linspace(
        exp_opts.freq_start, exp_opts.freq_stop, exp_opts.freq_count
    )

    exp = create_experiment(
        signal_map=signal_map,
        cavity_pulse=cavity_pulse,
        qubit_pi_pulse=qubit_pi_pulse,
        readout_pulse=readout_pulse,
        integration_kernel=integration_kernel,
        conditional_pi_pulse=conditional_pi_pulse,
        second_qubit_pi_pulse=second_qubit_pi_pulse,
        options=exp_opts,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(wf_opts.do_analysis):
        analysis_workflow(
            result,
            frequency_offsets=frequency_offsets,
            base_frequency=exp_opts.base_frequency,
        )
    workflow.return_(result)


def _convert_pulse(pulse: Pulse | dict, name: str) -> Pulse:
    """Create a Pulse from a dictionary if needed."""
    if isinstance(pulse, dict):
        return dsl.create_pulse(pulse, name=name)
    return pulse


@workflow.task
def create_experiment(
    *,
    signal_map: dict[str, object],
    cavity_pulse: Pulse | dict,
    qubit_pi_pulse: Pulse | dict,
    readout_pulse: Pulse | dict,
    integration_kernel: Pulse | dict,
    conditional_pi_pulse: Pulse | dict | None = None,
    second_qubit_pi_pulse: Pulse | dict | None = None,
    options: CavityPiNoPiExperimentOptions | None = None,
) -> Experiment:
    """Create a LabOne Q experiment for the cavity pi/nopi sequence."""
    opts = CavityPiNoPiExperimentOptions() if options is None else options

    cavity_pulse = _convert_pulse(cavity_pulse, "cavity_pulse")
    qubit_pi_pulse = _convert_pulse(qubit_pi_pulse, "qubit_pi_pulse")
    readout_pulse = _convert_pulse(readout_pulse, "readout_pulse")
    integration_kernel = _convert_pulse(integration_kernel, "integration_kernel")
    if conditional_pi_pulse is not None:
        conditional_pi_pulse = _convert_pulse(
            conditional_pi_pulse, "conditional_pi_pulse"
        )
    if second_qubit_pi_pulse is not None:
        second_qubit_pi_pulse = _convert_pulse(
            second_qubit_pi_pulse, "second_qubit_pi_pulse"
        )

    signals: list[ExperimentSignal] = [
        ExperimentSignal("cavity_drive"),
        ExperimentSignal("drive"),
        ExperimentSignal("measure"),
        ExperimentSignal("acquire"),
    ]
    if second_qubit_pi_pulse is not None:
        signals.append(ExperimentSignal("drive2"))

    exp_cavity_pi_nopi = Experiment(uid="cavity_pi_nopi", signals=signals)

    freq_sweep = LinearSweepParameter(
        uid="freq_offset",
        start=opts.freq_start,
        stop=opts.freq_stop,
        count=opts.freq_count,
    )
    pi_case_sweep = SweepParameter(uid="pi_case", values=[0, 1])
    crosskerr_sweep = SweepParameter(uid="crosskerr_case", values=[0, 1])

    with exp_cavity_pi_nopi.acquire_loop_rt(
        uid="shots",
        count=opts.count,
        averaging_mode=opts.averaging_mode or AveragingMode.CYCLIC,
        acquisition_type=opts.acquisition_type or AcquisitionType.INTEGRATION,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with exp_cavity_pi_nopi.sweep(
            uid="freq_sweep", parameter=freq_sweep, auto_chunking=opts.auto_chunking
        ):
            with exp_cavity_pi_nopi.sweep(uid="crosskerr", parameter=crosskerr_sweep):
                with exp_cavity_pi_nopi.sweep(uid="pi_toggle", parameter=pi_case_sweep):
                    with exp_cavity_pi_nopi.section(uid="cavity_pi_nopi"):
                        with exp_cavity_pi_nopi.match(
                            sweep_parameter=pi_case_sweep
                        ) as pi_match:
                            with pi_match.case(0):
                                exp_cavity_pi_nopi.play(
                                    signal="cavity_drive",
                                    pulse=cavity_pulse,
                                    pulse_parameters={
                                        "frequency": opts.base_frequency + freq_sweep
                                    },
                                )
                                exp_cavity_pi_nopi.reserve("drive")
                                if second_qubit_pi_pulse is not None:
                                    exp_cavity_pi_nopi.reserve("drive2")
                            with pi_match.case(1):
                                exp_cavity_pi_nopi.play(
                                    signal="drive", pulse=qubit_pi_pulse
                                )
                                if second_qubit_pi_pulse is not None:
                                    exp_cavity_pi_nopi.play(
                                        signal="drive2", pulse=second_qubit_pi_pulse
                                    )
                                exp_cavity_pi_nopi.play(
                                    signal="cavity_drive",
                                    pulse=cavity_pulse,
                                    pulse_parameters={
                                        "frequency": opts.base_frequency + freq_sweep
                                    },
                                )
                                exp_cavity_pi_nopi.play(
                                    signal="drive", pulse=qubit_pi_pulse
                                )
                                if second_qubit_pi_pulse is not None:
                                    exp_cavity_pi_nopi.play(
                                        signal="drive2", pulse=second_qubit_pi_pulse
                                    )

                        with exp_cavity_pi_nopi.match(
                            sweep_parameter=crosskerr_sweep
                        ) as ck_match:
                            with ck_match.case(0):
                                exp_cavity_pi_nopi.reserve(signal="drive")
                            with ck_match.case(1):
                                if conditional_pi_pulse is not None:
                                    exp_cavity_pi_nopi.play(
                                        signal="drive", pulse=conditional_pi_pulse
                                    )
                                else:
                                    exp_cavity_pi_nopi.reserve(signal="drive")

                    with exp_cavity_pi_nopi.section(
                        uid="measure", play_after="cavity_pi_nopi"
                    ):
                        exp_cavity_pi_nopi.play(
                            signal="measure",
                            pulse=readout_pulse,
                        )
                        exp_cavity_pi_nopi.acquire(
                            signal="acquire",
                            handle="cavity_pi_nopi",
                            kernel=integration_kernel,
                        )

                    relax_time = opts.relax_time
                    if relax_time is not None and relax_time > 0:
                        with exp_cavity_pi_nopi.section(
                            uid="relax", length=relax_time
                        ):
                            exp_cavity_pi_nopi.reserve(signal="measure")

    exp_cavity_pi_nopi.set_signal_map(signal_map)
    return exp_cavity_pi_nopi

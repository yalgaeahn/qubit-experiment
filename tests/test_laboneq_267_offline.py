"""Hardware-free LabOne Q 26.7 experiment contract checks."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from laboneq import workflow
from laboneq.dsl.device import create_connection
from laboneq.dsl.quantum.qpu import QPU
from laboneq.dsl.session import Session
from laboneq.serializers import load
from laboneq.simple import dsl
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qpu,
)

from qubit_experiment.qpu_types.bus_cavity.bus_types import BusCavity, BusCavityParameters
from qubit_experiment.qpu_types.bus_cavity.operations import BusCavityOperations
from qubit_experiment.qpu_types.fixed_transmon.demo_qpus import demo_platform
from qubit_experiment.qpu_types.fixed_transmon.operations import FixedTransmonOperations
from qubit_experiment.qpu_types.fixed_transmon.qubit_types import FixedTransmonQubitParameters
from qubit_experiment.qpu_types.twpa.demo_qpus import demo_platform as twpa_demo_platform

STANDARD_MODULES = (
    "amplitude_fine",
    "amplitude_rabi",
    "amplitude_rabi_chevron",
    "dispersive_shift",
    "drag_q_scaling",
    "echo",
    "ef_spectroscopy",
    "iq_cloud",
    "iq_time_trace",
    "lifetime_measurement",
    "linear_phase_delay",
    "qubit_gate_spectroscopy_amplitude",
    "qubit_spectroscopy",
    "qubit_spectroscopy_amplitude",
    "ramsey",
    "readout_amplitude_sweep",
    "readout_frequency_sweep",
    "readout_integration_delay_sweep",
    "readout_mid_sweep",
    "resonator_spectroscopy",
    "resonator_spectroscopy_amplitude",
    "signal_propagation_delay",
    "spin_locking",
    "time_rabi",
    "time_rabi_chevron",
    "time_traces",
    "two_qubit_readout_calibration",
    "three_qubit_readout_calibration",
    "measurement_qndness",
)

BUS_MODULES = {
    "coherence_spectroscopy": lambda q, b: dict(qubit=q[0], bus=b[0], delays=_delays(), CW_frequencies=_frequencies(), CW_amplitude=0.1, CW_phase=0.0),
    "coherence_spectroscopy_echo": lambda q, b: dict(qubit=q[0], bus=b[0], delays=_delays(), CW_frequencies=_frequencies(), CW_amplitude=0.1, CW_phase=0.0),
    "new_rip_echo": lambda q, b: dict(ctrl=q[0], targ=q[1], bus=b[0], delays=_delays(), rip_detunings=np.array([-11e6, -10e6, -9e6])),
    "residual_zz_echo": lambda q, b: dict(ctrl=q[0], targ=q[1], delays=_delays()),
    "three_qubit_ghz": lambda q, b: dict(qubits=q, bus=b),
    "three_qubit_state_tomography": lambda q, b: dict(qubits=q, bus=b),
    "three_qubit_virtual_z_validation": lambda q, b: dict(qubits=q, bus=b, phase_tuple=(0.0, 0.0, 0.0), stage="product"),
    "threeq_qst": lambda q, b: dict(qubits=q, bus=b),
    "twoq_qst": lambda q, b: dict(qubits=q[:2], bus=b[0]),
}

PROJECT_MODULES = (
    "cavity_T1_2",
    "photonnumber_calibration_6",
    "photonnumber_splitting",
    "project_ramsey",
    "residual_photon_calibration",
    "rip",
    "rip2",
    "rip4",
    "rip5",
    "rip_bell_state",
    "rip_selective",
    "rip_zz_echo_interaction",
    "rip_zzz_interaction",
    "two_qubit_state_tomography_3",
)


def _delays():
    return np.array([64e-9, 96e-9, 128e-9])


def _frequencies():
    return 5.54e9 + np.array([-1e6, 0.0, 1e6])


@pytest.fixture(scope="module")
def fixed_platform():
    platform = demo_platform(3)
    return platform.qpu, Session(device_setup=platform.setup, configure_logging=False)


@pytest.fixture(scope="module")
def bus_platform():
    platform = demo_platform(3)
    buses = []
    for index in range(3):
        uid = f"b{index}"
        platform.setup.add_connections(
            "device_hdawg",
            create_connection(to_signal=f"{uid}/drive", ports=f"SIGOUTS/{index}"),
            create_connection(to_signal=f"{uid}/drive_p", ports=f"SIGOUTS/{index + 3}"),
        )
        buses.append(
            BusCavity.from_logical_signal_group(
                uid,
                platform.setup.logical_signal_groups[uid],
                parameters=BusCavityParameters(
                    drive_lo_frequency=5.5e9,
                    resonance_frequency_bus=5.55e9,
                    rip_detuning=-10e6,
                    rip_length=400e-9,
                    rip_amplitude=0.1,
                    rip_pulse={"function": "NestedCosine"},
                    drive_p_lo_frequency=5.5e9,
                    resonance_frequency_bus_p=5.55e9,
                    rip_p_detuning=-10e6,
                    rip_p_length=400e-9,
                    rip_p_amplitude=0.1,
                    rip_p_pulse={"function": "const"},
                ),
            )
        )
    qpu = QPU(
        quantum_elements={"qubits": platform.qpu.quantum_elements, "bus": buses},
        quantum_operations=[FixedTransmonOperations, BusCavityOperations],
    )
    return (
        qpu,
        platform.qpu.quantum_elements,
        buses,
        Session(device_setup=platform.setup, configure_logging=False),
    )


def _standard_kwargs(module_name, qubits, *, uid=False):
    if module_name == "readout_amplitude_sweep":
        return {
            "qubit": qubits[0].uid if uid else qubits[0],
            "amplitudes": np.array([0.1, 0.2, 0.3]),
        }
    if module_name == "readout_frequency_sweep":
        return {"qubit": qubits[0].uid if uid else qubits[0]}
    module = importlib.import_module(f"qubit_experiment.experiments.{module_name}")
    signature = inspect.signature(module.create_experiment)
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if name in {"qubit", "qubits"}:
            selection = (
                qubits
                if module_name.startswith("three_qubit")
                else qubits[:2]
                if module_name.startswith("two_qubit")
                else qubits[0]
            )
            kwargs[name] = (
                [q.uid for q in selection]
                if uid and isinstance(selection, list)
                else selection.uid
                if uid
                else selection
            )
        elif name in {"amplitudes", "q_scalings"}:
            kwargs[name] = np.array([0.1, 0.2, 0.3])
        elif name in {"delays", "lengths"}:
            kwargs[name] = _delays()
        elif name == "frequencies":
            center = 7.1e9 if module_name.startswith(("readout", "resonator", "linear_phase", "dispersive")) else 6.5e9
            kwargs[name] = center + np.array([-1e6, 0.0, 1e6])
        elif name == "states":
            kwargs[name] = ("g", "e")
        elif name == "state":
            kwargs[name] = "g"
        elif name == "repetitions":
            kwargs[name] = np.array([1, 2, 3])
        elif name == "amplification_qop":
            kwargs[name] = "x180"
        elif name == "qpu":
            continue
        else:
            raise AssertionError(f"Unmapped {module_name} argument: {name}")
    return kwargs


@pytest.mark.parametrize("module_name", STANDARD_MODULES)
def test_standard_experiment_compiles_without_hardware(fixed_platform, module_name):
    qpu, session = fixed_platform
    module = importlib.import_module(f"qubit_experiment.experiments.{module_name}")
    experiment = module.create_experiment(
        qpu=qpu, **_standard_kwargs(module_name, qpu.quantum_elements)
    )
    session.compile(experiment)


@pytest.mark.parametrize("module_name", BUS_MODULES)
def test_bus_experiment_compiles_without_hardware(bus_platform, module_name):
    qpu, qubits, buses, session = bus_platform
    module = importlib.import_module(f"qubit_experiment.experiments.{module_name}")
    experiment = module.create_experiment(
        qpu=qpu, **BUS_MODULES[module_name](qubits, buses)
    )
    session.compile(experiment)


def _uids(value):
    if isinstance(value, list):
        return [_uids(item) for item in value]
    return value.uid if hasattr(value, "uid") else value


@pytest.mark.parametrize("module_name", BUS_MODULES)
def test_bus_workflow_builds_with_uids(bus_platform, module_name):
    qpu, qubits, buses, session = bus_platform
    module = importlib.import_module(f"qubit_experiment.experiments.{module_name}")
    kwargs = {
        name: _uids(value)
        for name, value in BUS_MODULES[module_name](qubits, buses).items()
    }
    if module_name == "new_rip_echo":
        kwargs["ramsey_detunings"] = 0.0
    elif module_name == "three_qubit_virtual_z_validation":
        kwargs.pop("phase_tuple")
        kwargs.pop("stage")
        kwargs["phase_values"] = [0.0, 0.5]
    module.experiment_workflow(session=session, qpu=qpu, **kwargs)


@pytest.mark.xfail(
    strict=True,
    reason="The repository defines neither direct_cr nor cr_cancel quantum operations",
)
def test_direct_cr_hamiltonian_tomography_compiles_without_hardware(bus_platform):
    qpu, qubits, _, session = bus_platform
    module = importlib.import_module(
        "qubit_experiment.experiments.direct_cr_hamiltonian_tomography"
    )
    experiment = module.create_experiment(
        qpu=qpu,
        ctrl=qubits[0],
        targ=qubits[1],
        amplitudes=np.array([0.05, 0.1, 0.15]),
        lengths=_delays(),
    )
    session.compile(experiment)


@pytest.mark.parametrize("module_name", STANDARD_MODULES)
def test_standard_workflow_builds_with_uids(fixed_platform, module_name):
    qpu, session = fixed_platform
    module = importlib.import_module(f"qubit_experiment.experiments.{module_name}")
    kwargs = _standard_kwargs(module_name, qpu.quantum_elements, uid=True)
    if module_name == "amplitude_fine":
        kwargs["target_angle"] = np.pi
        kwargs["phase_offset"] = 0.0
    elif module_name == "readout_frequency_sweep":
        kwargs["frequencies"] = 7.1e9 + np.array([-1e6, 0.0, 1e6])
    elif module_name == "time_traces":
        kwargs = {"qubits": qpu.quantum_elements[0].uid, "states": ("g", "e")}
    module.experiment_workflow(session=session, qpu=qpu, **kwargs)


@pytest.mark.parametrize(
    ("module_name", "kwargs"),
    [
        ("calibrate_cancellation", {"cancel_phase": [0.1, 0.2], "cancel_attenuation": [1.0, 2.0]}),
        ("measure_gain_curve", {"probe_frequency": [6.4e9, 6.5e9], "pump_power": [0.1, 0.2]}),
        ("scan_pump_parameters", {"pump_frequency": [4.1e9, 4.2e9], "pump_power": [0.1, 0.2]}),
        ("twpa_spectroscopy", {"frequencies": [6.4e9, 6.5e9]}),
    ],
)
def test_twpa_experiment_compiles_without_hardware(module_name, kwargs):
    platform = twpa_demo_platform(1)
    session = Session(device_setup=platform.setup, configure_logging=False)
    pa = platform.qpu.quantum_elements[0]
    module = importlib.import_module(f"qubit_experiment.experiments.{module_name}")
    experiment = module.create_experiment(
        qpu=platform.qpu, parametric_amplifier=pa, **kwargs
    )
    session.compile(experiment)


@pytest.mark.parametrize(
    ("module_name", "kwargs"),
    [
        ("calibrate_cancellation", {"cancel_phase": [0.1, 0.2], "cancel_attenuation": [1.0, 2.0]}),
        ("measure_gain_curve", {"probe_frequency": [6.4e9, 6.5e9], "pump_power": [0.1, 0.2]}),
        ("scan_pump_parameters", {"pump_frequency": [4.1e9, 4.2e9], "pump_power": [0.1, 0.2]}),
        ("twpa_spectroscopy", {"frequencies": [6.4e9, 6.5e9]}),
    ],
)
def test_twpa_workflow_builds_with_uid(module_name, kwargs):
    platform = twpa_demo_platform(1)
    session = Session(device_setup=platform.setup, configure_logging=False)
    module = importlib.import_module(f"qubit_experiment.experiments.{module_name}")
    module.experiment_workflow(
        session=session,
        qpu=platform.qpu,
        parametric_amplifier="twpa0",
        **kwargs,
    )


def test_twpa_cancellation_on_compiles_without_hardware():
    platform = twpa_demo_platform(1)
    session = Session(device_setup=platform.setup, configure_logging=False)
    module = importlib.import_module("qubit_experiment.experiments.calibrate_cancellation")
    experiment = module.create_experiment(
        qpu=platform.qpu,
        parametric_amplifier=platform.qpu.quantum_elements[0],
        cancel_phase=[0.1, 0.2],
        cancel_attenuation=[1.0, 2.0],
        cancellation_on=True,
    )
    session.compile(experiment)


def test_uid_lookup_temporary_parameters_and_qpu_update(fixed_platform):
    qpu, _ = fixed_platform
    original = qpu.quantum_elements[0].parameters.ge_drive_amplitude_pi
    temp_qpu = temporary_qpu.func(
        qpu, {"q0": {"ge_drive_amplitude_pi": 0.2}}
    )
    temp_qubit = temporary_quantum_elements_from_qpu.func(temp_qpu, "q0")
    assert temp_qubit.uid == "q0"
    assert temp_qubit.parameters.ge_drive_amplitude_pi == 0.2
    assert qpu.quantum_elements[0].parameters.ge_drive_amplitude_pi == original
    update_qpu.func(temp_qpu, {"q0": {"ge_drive_amplitude_pi": 0.3}})
    assert temp_qpu.quantum_elements[0].parameters.ge_drive_amplitude_pi == 0.3


def test_legacy_user_defined_values_migrate_to_custom():
    parameters = FixedTransmonQubitParameters(
        custom={"current": 2, "shared": "current"},
        user_defined={"legacy": 1, "shared": "legacy"},
    )
    assert parameters.custom == {"legacy": 1, "current": 2, "shared": "current"}
    assert parameters.user_defined == {}


@pytest.mark.parametrize("snapshot", ["20260303-0824_3Q_tomo", "20260303-0948_3Q_tomo"])
def test_saved_2510_qpu_parameters_load_in_267(monkeypatch, snapshot):
    class CombinedOperations(FixedTransmonOperations, BusCavityOperations):
        pass

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "qubit_experiment"))
    monkeypatch.setattr(sys.modules["__main__"], "CombinedOperations", CombinedOperations, raising=False)
    imported_before = set(sys.modules)
    try:
        qpu = load(root / "projects" / "2026_selectiveRIP" / "qpu_parameters" / snapshot)
        assert len(qpu.quantum_elements) == 6
        assert all(
            not q.parameters.user_defined
            for q in qpu.quantum_elements
            if q.uid.startswith("q")
        )
    finally:
        for name in set(sys.modules) - imported_before:
            if name == "qpu_types" or name.startswith("qpu_types."):
                sys.modules.pop(name, None)


@pytest.mark.parametrize("shape", [(16,), (1, 16), (16, 1)])
def test_raw_trace_single_acquisition_axis_order(shape):
    from qubit_experiment.analysis.iq_time_trace import _single_raw_trace

    data = np.arange(16, dtype=complex).reshape(shape)
    np.testing.assert_array_equal(_single_raw_trace(data, "trace"), np.arange(16))


def test_raw_trace_rejects_multiple_acquisitions_without_axis_metadata():
    from qubit_experiment.analysis.iq_time_trace import _single_raw_trace

    with pytest.raises(ValueError, match="Expected one RAW trace"):
        _single_raw_trace(np.ones((2, 16)), "trace")


@pytest.mark.parametrize("shape", [(8,), (1, 8), (8, 1)])
def test_single_shot_iq_cloud_accepts_singleton_axes(shape):
    from qubit_experiment.analysis.iq_cloud import _read_iq_cloud_shots_with_fallback

    handle = dsl.handles.calibration_trace_handle("q0", "g")
    result = {handle: SimpleNamespace(data=np.arange(8).reshape(shape))}
    shots = _read_iq_cloud_shots_with_fallback(
        result=result, qubit_uid="q0", prepared_label="g", num_qubits=1
    )
    np.testing.assert_array_equal(shots, np.arange(8))


def test_randomized_benchmarking_compiles_without_hardware(fixed_platform):
    qpu, session = fixed_platform
    from qubit_experiment.experiments import single_qubit_randomized_benchmarking as rb

    gate_map = rb.get_gate_map.func()
    sequences = rb.create_sq_rb_qasm.func([1, 2], gate_map, variations=1, seed=7)
    operations = rb.add_qasm_operations.func(qpu.quantum_operations, gate_map)
    experiment = rb.create_experiment(
        qpu,
        qpu.quantum_elements[0],
        sequences,
        quantum_operations=operations,
    )
    session.compile(experiment)


@pytest.mark.parametrize(
    ("module_name", "kwargs"),
    [
        ("coherence_tracking", {"qubits": "q0", "t1_delays": [64e-9, 96e-9]}),
        ("readout_length_sweep", {"qubit": "q0", "readout_lengths": [1e-6, 1.2e-6]}),
        ("single_qubit_randomized_benchmarking", {"qubits": "q0", "length_cliffords": [1, 2]}),
    ],
)
def test_workflow_only_modules_build_with_uids(fixed_platform, module_name, kwargs):
    qpu, session = fixed_platform
    module = importlib.import_module(f"qubit_experiment.experiments.{module_name}")
    module.experiment_workflow(session=session, qpu=qpu, **kwargs)


@pytest.mark.parametrize("kind", ["transmon", "bus", "twpa"])
def test_uid_workflow_executes_through_compile_without_hardware(
    fixed_platform, bus_platform, monkeypatch, kind
):
    @workflow.task
    def offline_run(session, compiled_experiment):  # noqa: ARG001
        return {"offline": True}

    if kind == "transmon":
        qpu, session = fixed_platform
        module = importlib.import_module("qubit_experiment.experiments.amplitude_rabi")
        kwargs = dict(
            qubits="q0",
            amplitudes=np.array([0.1, 0.2]),
            temporary_parameters={"q0": {"ge_drive_amplitude_pi": 0.3}},
        )
        options = module.experiment_workflow.options()
        options.do_analysis(False)
        options.update(False)
        kwargs["options"] = options
    elif kind == "bus":
        qpu, _, _, session = bus_platform
        project_root = Path(__file__).resolve().parents[1] / "projects" / "2026_selectiveRIP"
        monkeypatch.syspath_prepend(str(project_root))
        module = importlib.import_module(
            "custom_qubit_experiment.custom_experiments.rip"
        )
        kwargs = dict(
            ctrl="q0", targ="q1", bus="b0", delays=_delays(), detunings=-10e6
        )
    else:
        platform = twpa_demo_platform(1)
        qpu = platform.qpu
        session = Session(device_setup=platform.setup, configure_logging=False)
        module = importlib.import_module("qubit_experiment.experiments.twpa_spectroscopy")
        kwargs = dict(parametric_amplifier="twpa0", frequencies=[6.4e9, 6.5e9])
    monkeypatch.setattr(module, "run_experiment", offline_run)
    output = module.experiment_workflow(session=session, qpu=qpu, **kwargs).run().output
    assert output == {"offline": True}


def _project_kwargs(name, qubits, buses):
    if name == "project_ramsey":
        return {"qubits": qubits[0], "delays": _delays()}
    if name == "rip":
        return dict(
            ctrl=qubits[0], targ=qubits[1], bus=buses[0],
            delays=_delays(), detunings=-10e6,
        )
    if name == "rip2":
        return dict(
            ctrl=qubits[0], targ=qubits[1], bus=buses[0],
            bus_frequency=5.54e9, bus_amplitude=0.1, delays=_delays(),
        )
    if name == "cavity_T1_2":
        return dict(
            qubit=qubits[0],
            bus=buses[0],
            delay_time=400e-9,
            CW_amplitude=0.1,
            CW_frequency=5.54e9,
        )
    if name == "photonnumber_splitting":
        return dict(
            qubit=qubits[0],
            bus=buses[0],
            qubit_frequencies=qubits[0].parameters.resonance_frequency_ge
            + np.array([-1e6, 0.0, 1e6]),
            CW_frequencies=_frequencies(),
            CW_amplitude=0.1,
        )
    if name in {"photonnumber_calibration_6", "residual_photon_calibration"}:
        return dict(
            qubit=qubits[0],
            bus=buses[0],
            frequencies=qubits[0].parameters.resonance_frequency_ge
            + np.array([-1e6, 0.0, 1e6]),
            CW_amplitude=0.1,
            CW_frequency=5.54e9,
        )
    kwargs = dict(
        ctrl=qubits[0],
        targ=qubits[1],
        bus=buses[0],
        bus2=buses[1],
        bus_frequency=5.54e9,
        bus_amplitude=0.1,
        bus2_frequency=5.54e9,
        bus2_amplitude=0.1,
        delays=_delays(),
    )
    if name not in {"rip4", "rip5"}:
        kwargs.update(
            bus3=buses[2],
            bus3_frequency=5.54e9,
            bus3_amplitude=0.1,
        )
    if name in {"rip_bell_state", "rip_zz_echo_interaction", "rip_zzz_interaction"}:
        kwargs["spec"] = qubits[2]
    if name == "two_qubit_state_tomography_3":
        return {"ctrl": qubits[0], "targ": qubits[1], "bus": buses[0]}
    return kwargs


@pytest.mark.parametrize("module_name", PROJECT_MODULES)
def test_project_experiment_compiles_and_workflow_builds_with_uids(
    bus_platform, monkeypatch, module_name
):
    project_root = Path(__file__).resolve().parents[1] / "projects" / "2026_selectiveRIP"
    monkeypatch.syspath_prepend(str(project_root))
    module = importlib.import_module(
        f"custom_qubit_experiment.custom_experiments.{module_name}"
    )
    qpu, qubits, buses, session = bus_platform
    kwargs = _project_kwargs(module_name, qubits, buses)
    experiment = module.create_experiment(qpu=qpu, **kwargs)
    session.compile(experiment)
    module.experiment_workflow(
        session=session,
        qpu=qpu,
        **{name: _uids(value) for name, value in kwargs.items()},
    )

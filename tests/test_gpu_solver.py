import numpy as np
import pytest
import sympy as sp
import importlib

cp = pytest.importorskip("cupy")


def _gpu_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
@pytest.mark.parametrize("solver", ("rk4", "lserk4", "lserk4f", "dp5", "ab5", "anas5", "tsit5",
                                    "alshina6", "dop853", "tsit5a"))
def test_pi_rotation_reaches_requested_final_time(solver):
    from gqis import mesolve_2D

    dummy = sp.Symbol("dummy", real=True)
    hamiltonian = sp.Matrix([[0, sp.Rational(1, 2)], [sp.Rational(1, 2), 0]])
    rho0 = sp.Matrix([[1, 0], [0, 0]])
    tlist = np.linspace(0.0, np.pi, 101, dtype=np.float32)

    final_rho = mesolve_2D(hamiltonian, sp.Integer(0), [], sp.eye(2), tlist,
                           var_arrays={dummy: np.array([0.0], dtype=np.float32)}, rho0=rho0,
                           output_mode="final_rho", solver=solver)
    assert abs(float(final_rho[0, 0, 0])) < 2.0e-5


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
@pytest.mark.parametrize("solver", ("rk4", "lserk4", "lserk4f", "dp5", "ab5", "anas5", "tsit5",
                                    "alshina6", "dop853"))
def test_time_dependent_commuting_hamiltonian_uses_correct_stage_times(solver):
    from gqis import mesolve_2D

    dummy = sp.Symbol("dummy", real=True)
    drive_symbol = sp.Symbol("Drive", real=True)
    t = sp.Symbol("t", real=True)
    final_time = 1.7
    hamiltonian = (1 + drive_symbol) * sp.Matrix([[0, 1], [1, 0]]) / 2
    tlist = np.linspace(0.0, final_time, 81, dtype=np.float64)
    final_rho = mesolve_2D(
        hamiltonian, {drive_symbol: sp.Rational(2, 5) * sp.sin(t)}, [], sp.eye(2), tlist,
        var_arrays={dummy: np.array([0.0], dtype=np.float64)}, output_mode="final_rho",
        solver=solver, fp64=True,
    )
    integrated_frequency = final_time + 0.4 * (1.0 - np.cos(final_time))
    expected_rho00 = np.cos(0.5 * integrated_frequency)**2
    assert abs(float(final_rho[0, 0, 0]) - expected_rho00) < 2.0e-7


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
def test_forced_solver_unrolling_compiles_and_is_reported():
    from gqis import mesolve_2D

    dummy = sp.Symbol("dummy", real=True)
    final_rho, timing = mesolve_2D(
        sp.zeros(2), sp.Integer(0), [], sp.eye(2),
        np.linspace(0.0, 1.0, 5, dtype=np.float32),
        var_arrays={dummy: np.array([0.0], dtype=np.float32)},
        output_mode="final_rho", unroll=True, return_timing_info=True,
    )
    assert np.all(np.isfinite(final_rho))
    assert timing["solver"] == "rk4"
    assert timing["unroll"] is True


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
def test_different_solvers_share_one_symbolic_rhs(monkeypatch):
    solver_module = importlib.import_module("gqis.solver")
    solver_module._RHS_CODE_CACHE.clear()
    solver_module._KERNEL_CACHE.clear()
    original = solver_module.build_reduced_lindblad_rhs
    calls = 0

    def counted_generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_module, "build_reduced_lindblad_rhs", counted_generate)
    dummy = sp.Symbol("dummy", real=True)
    model = (sp.zeros(2), sp.Integer(0), [], sp.eye(2),
             np.linspace(0.0, 1.0, 5, dtype=np.float32))
    common = {"var_arrays": {dummy: np.array([0.0], dtype=np.float32)},
              "output_mode": "final_rho", "return_timing_info": True}
    _, rk4_timing = solver_module.mesolve_2D(*model, solver="rk4", **common)
    _, tsit5_timing = solver_module.mesolve_2D(*model, solver="tsit5", **common)

    assert calls == 1
    assert rk4_timing["cached_rhs"] == "miss"
    assert rk4_timing["cached_kernel"] == "miss"
    assert rk4_timing["rhs_codegen_s"] >= 0.0
    assert tsit5_timing["cached_rhs"] == "hit"
    assert tsit5_timing["cached_kernel"] == "miss"
    assert tsit5_timing["rhs_codegen_s"] == 0.0
    stages = ("ode_prepare_s", "ode_optimize_s", "code_emit_s", "kernel_build_s")
    assert all(rk4_timing[key] >= 0.0 for key in stages)
    assert rk4_timing["rhs_codegen_s"] == pytest.approx(
        rk4_timing["ode_prepare_s"] + rk4_timing["ode_optimize_s"] + rk4_timing["code_emit_s"])
    assert tsit5_timing["ode_prepare_s"] == tsit5_timing["code_emit_s"] == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
def test_compiled_rhs_gives_the_same_result_without_preparing_the_ode():
    solver_module = importlib.import_module("gqis.solver")
    eps, amp, t = sp.symbols("eps amp t", real=True)
    drive = sp.Symbol("Drive", real=True)
    model = (sp.Matrix([[eps + drive, 0.5], [0.5, -eps - drive]]) / 2, {drive: amp * sp.cos(t)},
             [0.1 * sp.Matrix([[0, 1], [0, 0]])], sp.diag(1, 0))
    sweep = {eps: np.linspace(-1, 1, 4, dtype=np.float32), amp: np.linspace(0, 2, 3, dtype=np.float32)}
    tlist = np.linspace(0.0, 10.0, 201, dtype=np.float32)
    direct = solver_module.mesolve_2D(*model, tlist, var_arrays=sweep)
    compiled = solver_module.compile_lindblad_rhs(*model, sweep)
    via, timing = solver_module.mesolve_2D(compiled, tlist=tlist, var_arrays=sweep, return_timing_info=True)
    np.testing.assert_array_equal(direct, via)
    assert timing["cached_rhs"] == "provided"
    assert timing["ode_prepare_s"] == timing["ode_optimize_s"] == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
def test_calibrated_timestep_meets_limits_against_fine_reference():
    """calibrate_timesteps refines a too-coarse grid until the map meets its limits."""
    from gqis import mesolve_2D
    import gqis.solver as core

    D, eps, Drive, A, t = sp.symbols("Delta eps Drive A t", real=True)
    sz = sp.Matrix([[1, 0], [0, -1]])
    sm = sp.Matrix([[0, 1], [0, 0]])
    sx = sp.Matrix([[0, 1], [1, 0]])
    w, period = 1.14, 2.0 * np.pi / 1.14
    gamma1 = 0.05 / period
    gamma2 = gamma1 / 2.0 + 0.04 / period
    axes = {eps: np.linspace(-10 * w, 10 * w, 64, dtype=np.float32),
            A: np.linspace(0.0, 10 * w, 64, dtype=np.float32)}

    def run(steps_per_period, calibrate=None):
        return mesolve_2D(0.5 * D * sx + 0.5 * Drive * sz, eps + A * sp.sin(w * t),
                          [sp.sqrt(gamma1) * sm, sp.sqrt(gamma2) * sz], sp.diag(0, 1),
                          np.linspace(0.0, 30 * period, 30 * steps_per_period + 1),
                          var_arrays=axes, const_values={D: 1.0}, calibrate_timesteps=calibrate,
                          return_timing_info=calibrate is not None)

    core._TIMESTEP_CALIBRATION_CACHE.clear()
    reference = np.abs(run(2048))
    result, timing = run(64, dict(rms_atol=1e-3, max_atol=1e-2, sample_stride=8))
    error = np.abs(np.abs(result) - reference)
    assert timing["timestep_calibration"]["steps"] > 30 * 64  # the coarse start was refined
    assert np.sqrt(np.mean(error**2)) <= 1e-3
    assert error.max() <= 1e-2


def _small_driven_qubit():
    """16 x 16 sweep of a driven dissipative qubit over 10 periods, for quick calibration checks."""
    from gqis import mesolve_2D

    D, eps, Drive, A, t = sp.symbols("Delta eps Drive A t", real=True)
    sz, sm, sx = sp.Matrix([[1, 0], [0, -1]]), sp.Matrix([[0, 1], [0, 0]]), sp.Matrix([[0, 1], [1, 0]])
    w = 1.14
    period = 2.0 * np.pi / w
    gamma1 = 0.05 / period
    gamma2 = gamma1 / 2.0 + 0.04 / period
    axes = {eps: np.linspace(-6 * w, 6 * w, 16), A: np.linspace(0.0, 6 * w, 16)}

    def run(steps_per_period, **options):
        return mesolve_2D(0.5 * D * sx + 0.5 * Drive * sz, eps + A * sp.sin(w * t),
                          [sp.sqrt(gamma1) * sm, sp.sqrt(gamma2) * sz], sp.diag(0, 1),
                          np.linspace(0.0, 10 * period, 10 * steps_per_period + 1),
                          var_arrays=axes, const_values={D: 1.0}, **options)
    return run, period


def _assert_within(result, reference, rms_limit, max_limit):
    error = np.abs(np.asarray(result) - np.asarray(reference))
    assert np.sqrt(np.mean(error**2)) <= rms_limit
    assert error.max() <= max_limit


LIMITS = dict(rms_atol=1e-3, max_atol=1e-2, sample_stride=2)


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
@pytest.mark.parametrize("solver", ("rk4", "tsit5a"))
@pytest.mark.parametrize("output_mode", ("mean", "final", "final_rho"))
def test_calibration_meets_limits_for_every_lindblad_output(solver, output_mode):
    """calibrate_timesteps (step count, or tsit5a step tolerances) meets its limits for each output."""
    import gqis.solver as core

    run, _ = _small_driven_qubit()
    core._TIMESTEP_CALIBRATION_CACHE.clear()
    reference = run(2048, output_mode=output_mode)
    result = run(8, output_mode=output_mode, solver=solver, calibrate_timesteps=LIMITS)
    _assert_within(result, reference, LIMITS["rms_atol"], LIMITS["max_atol"])


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
@pytest.mark.parametrize("solver", ("rk4", "tsit5a"))
def test_calibration_keeps_trace_sample_times(solver):
    """A calibrated trace is sampled at the times of the supplied grid and meets the limits there."""
    import gqis.solver as core

    run, period = _small_driven_qubit()
    core._TIMESTEP_CALIBRATION_CACHE.clear()
    _, _, plain_times = run(8, output_mode="final", solver=solver, return_time_trace=True, time_trace_every=8,
                            ignore_non_finite_output=True)  # only its sample times are used
    result, trace, times, timing = run(8, output_mode="final", solver=solver, return_time_trace=True,
                                       time_trace_every=8, calibrate_timesteps=LIMITS, return_timing_info=True)
    np.testing.assert_allclose(times, plain_times, rtol=1e-6)
    np.testing.assert_allclose(times, (np.arange(len(times)) * 8 + 1) * period / 8, rtol=1e-6)
    if solver == "rk4":
        assert timing["timestep_calibration"]["steps"] > 80  # the grid was refined
    reference, reference_trace, reference_times = run(2048, output_mode="final", return_time_trace=True)
    picked = np.searchsorted(reference_times, times - 1e-6 * period)
    np.testing.assert_allclose(reference_times[picked], times, rtol=1e-5)
    _assert_within(result, reference, LIMITS["rms_atol"], LIMITS["max_atol"])
    _assert_within(trace, reference_trace[:, :, picked], LIMITS["rms_atol"], LIMITS["max_atol"])


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
@pytest.mark.parametrize("solver", ("rk4", "tsit5a"))
@pytest.mark.parametrize("output", ("final", "mean", "observable", "trace"))
def test_calibration_meets_limits_for_ode_outputs(solver, output):
    """odesolve_2D calibration for final and mean states, an observable and a state trace."""
    from gqis import odesolve_2D
    import gqis.solver as core

    if solver == "tsit5a" and output == "mean":
        pytest.skip("tsit5a averages with the trapezoidal rule over its steps; with a smooth solution "
                    "and a coarse tlist this quadrature error is not controlled by the step tolerance")
    x, v, t, damping, amplitude = sp.symbols("x v t damping amplitude", real=True)
    axes = {damping: np.linspace(0.1, 0.5, 16), amplitude: np.linspace(0.1, 1.0, 16)}

    def run(steps, **options):
        return odesolve_2D([v, -damping * v - x - 0.2 * x**3 + amplitude * sp.cos(1.3 * t)], [x, v], [1.0, 0.0],
                           np.linspace(0.0, 20.0, steps + 1), var_arrays=axes, **options)

    limits = dict(rms_atol=1e-4, max_atol=1e-3, sample_stride=2)
    options = {"observable": dict(observable=x**2 + v**2), "mean": dict(output_mode="mean"),
               "trace": dict(return_time_trace=True, time_trace_every=8), "final": {}}[output]
    core._TIMESTEP_CALIBRATION_CACHE.clear()
    result = run(64, solver=solver, calibrate_timesteps=limits, **options)
    if output == "trace":
        _, trace, times = result
        _, fine_trace, fine_times = run(16384, return_time_trace=True)
        picked = np.searchsorted(fine_times, times - 1e-6)
        np.testing.assert_allclose(fine_times[picked], times, rtol=1e-5)
        _assert_within(trace, fine_trace[:, :, picked], limits["rms_atol"], limits["max_atol"])
    elif output == "mean":  # reference: exact time integral of a fine trace
        _, fine_trace, fine_times = run(16384, return_time_trace=True)
        states = np.concatenate([np.broadcast_to([1.0, 0.0], fine_trace.shape[:2] + (1, 2)), fine_trace], axis=2)
        reference = np.trapezoid(states, np.concatenate([[0.0], fine_times]), axis=2) / 20.0
        _assert_within(result, reference, limits["rms_atol"], limits["max_atol"])
    else:
        _assert_within(result, run(16384, **options), limits["rms_atol"], limits["max_atol"])


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
def test_tsit5a_matches_time_dependent_solution_between_coarse_output_points():
    from gqis import mesolve_2D

    dummy = sp.Symbol("dummy", real=True)
    drive_symbol = sp.Symbol("Drive", real=True)
    t = sp.Symbol("t", real=True)
    final_time = 1.7
    hamiltonian = (1 + drive_symbol) * sp.Matrix([[0, 1], [1, 0]]) / 2
    tlist = np.linspace(0.0, final_time, 5, dtype=np.float64)  # 4 output intervals
    final_rho = mesolve_2D(
        hamiltonian, {drive_symbol: sp.Rational(2, 5) * sp.sin(t)}, [], sp.eye(2), tlist,
        var_arrays={dummy: np.array([0.0], dtype=np.float64)}, output_mode="final_rho",
        solver="tsit5a", fp64=True, adaptive_options={"step_rtol": 1e-12, "step_atol": 1e-12,
                                                      "max_substeps": 10000},
    )
    integrated_frequency = final_time + 0.4 * (1.0 - np.cos(final_time))
    expected_rho00 = np.cos(0.5 * integrated_frequency)**2
    assert abs(float(final_rho[0, 0, 0]) - expected_rho00) < 2.0e-7


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
@pytest.mark.parametrize("fp64_time", (True, False))
def test_tsit5a_time_average_integrates_over_adaptive_steps(fp64_time):
    from gqis import mesolve_2D

    dummy = sp.Symbol("dummy", real=True)
    hamiltonian = sp.Matrix([[0, 1], [1, 0]]) / 2  # rho00(t) = cos(t/2)^2 from |0>
    projector = sp.Matrix([[1, 0], [0, 0]])
    tlist = np.linspace(0.0, 10.0, 6, dtype=np.float64)  # 5 output intervals of length 2
    value, info = mesolve_2D(
        hamiltonian, sp.Integer(0), [], projector, tlist,
        var_arrays={dummy: np.array([0.0], dtype=np.float64)}, rho0=sp.Matrix([[1, 0], [0, 0]]),
        output_mode="mean", warmup_time=0.2, solver="tsit5a", fp64=True, return_timing_info=True,
        adaptive_options={"step_rtol": 1e-10, "step_atol": 1e-10, "fp64_time": fp64_time,
                          "count_steps": True, "max_substeps": 10000},
    )
    expected = 0.5 + (np.sin(10.0) - np.sin(2.0)) / (2.0 * 8.0)  # average over [2, 10]
    # Trapezoid rule over the adaptive steps (second order): about (h^2/12) * |f'(10) - f'(2)| / 8.
    assert abs(float(np.real(value[0, 0])) - expected) < 5.0e-5
    assert info["adaptive_steps"]["accepted"][0, 0] > 5

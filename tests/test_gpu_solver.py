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
@pytest.mark.parametrize("solver", ("rk4", "lserk4", "dp5", "ab5", "anas5", "tsit5",
                                    "alshina6", "dop853"))
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
@pytest.mark.parametrize("solver", ("rk4", "lserk4", "dp5", "ab5", "anas5", "tsit5",
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
    original = solver_module.generate_unrolled_drho
    calls = 0

    def counted_generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_module, "generate_unrolled_drho", counted_generate)
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

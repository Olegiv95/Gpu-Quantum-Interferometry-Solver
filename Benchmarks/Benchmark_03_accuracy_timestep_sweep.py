"""Benchmark 03: two- or four-level time-grid accuracy and performance comparison.

Backends
--------
  gqis_*     : method-specific GQIS targets used together by the accuracy sweep
  python_cpu : pure-Python fixed-step RK4 master-equation solver
  python_ode_cpu : SciPy solve_ivp adaptive CPU solver
  qutip_cpu  : QuTiP mesolve
  julia_gpu_*: precision-specific Julia DiffEqGPU GPUTsit5 targets
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))
MANUAL_CUPY_CACHE = PROJECT_ROOT / ".gqis_manual_test_kernel_cache"
MANUAL_CUPY_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUPY_CACHE_DIR", str(MANUAL_CUPY_CACHE))

# Set the cache environment before these imports can load CuPy.
from Benchmark_full_tools import benchmark_output_path, print_equipment_info  # noqa: E402
from gqis import build_reduced_lindblad_rhs, mesolve_2D  # noqa: E402

JULIA_SOLVER_MODES = {"julia_gpu_fp64": "fp64", "julia_gpu_fp32": "fp32",
                      "julia_gpu_fp32_opt": "fp64_only_step",
                      "julia_gpu_fp32_fopt": "fp32_integer_step"}
OTHER_SOLVERS = ("python_cpu", "python_ode_cpu", "qutip_cpu", *JULIA_SOLVER_MODES)
SOLVER_NAMES = {"python_cpu": "Python CPU", "python_ode_cpu": "SciPy", "qutip_cpu": "QuTiP",
                **{name: "Julia GPU" for name in JULIA_SOLVER_MODES}}
SOLVER_METHODS = {"python_cpu": "fixed-step RK4", "python_ode_cpu": "adaptive RK45",
                  "qutip_cpu": "adaptive Adams",
                  "julia_gpu_fp64": "fixed-step GPUTsit5, FP64 state and time",
                  "julia_gpu_fp32": "fixed-step GPUTsit5, FP32 state and time",
                  "julia_gpu_fp32_opt": "fixed-step GPUTsit5, FP32 state with FP64 time",
                  "julia_gpu_fp32_fopt": "fixed-step GPUTsit5, FP32 integer-step time"}
SOLVER_METHOD_LABELS = {"python_cpu": "RK4", "python_ode_cpu": "RK45",
                        "qutip_cpu": "Adams", "julia_gpu_fp64": "Tsit5 FP64",
                        "julia_gpu_fp32": "Tsit5 FP32",
                        "julia_gpu_fp32_opt": "Tsit5 FP32-opt",
                        "julia_gpu_fp32_fopt": "Tsit5 FP32-fopt"}
GQIS_SOLVER_LABELS = {"rk4": "RK4", "lserk4": "LSRK4", "lserk4f": "LSRK4F", "dp5": "DP5",
                      "ab5": "AB5", "anas5": "Anas5", "tsit5": "Tsit5",
                      "alshina6": "Alshina6", "dop853": "DOP853"}
GQIS_SOLVER_DESCRIPTIONS = {
    "rk4": "fixed-step classical RK4",
    "lserk4": "fixed-step Carpenter-Kennedy 2N54",
    "lserk4f": "fixed-step Carpenter-Kennedy 2N54, RHS fused into the residual",
    "dp5": "fixed-step Dormand-Prince fifth-order formula",
    "ab5": "fixed-step fifth-order Adams-Bashforth",
    "anas5": "fixed-step frequency-fitted Anas5(w)",
    "tsit5": "fixed-step Tsitouras fifth-order formula",
    "alshina6": "fixed-step seven-stage Alshina sixth-order formula",
    "dop853": "fixed-step Dormand-Prince eighth-order DOP853 main formula",
}
GQIS_ACCURACY_SOLVERS = {f"gqis_{method}": method for method in GQIS_SOLVER_LABELS}
ACCURACY_SOLVERS = (*GQIS_ACCURACY_SOLVERS, *OTHER_SOLVERS)
ACCURACY_SOLVER_SET = set(ACCURACY_SOLVERS)
SOLVER_NAMES.update({name: "GQIS" for name in GQIS_ACCURACY_SOLVERS})
SOLVER_METHOD_LABELS.update({name: GQIS_SOLVER_LABELS[method]
                             for name, method in GQIS_ACCURACY_SOLVERS.items()})
SOLVER_METHODS.update({name: GQIS_SOLVER_DESCRIPTIONS[method]
                       for name, method in GQIS_ACCURACY_SOLVERS.items()})
LAST_JULIA_TOTAL_S = np.nan
LAST_JULIA_PRECALC_S = np.nan
LAST_GQIS_RHS_CODEGEN_S = np.nan
LAST_GQIS_RHS_CACHE = ""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchConfig:
    delta: float
    w: float
    gamma1: float
    gamma2: float
    tr: float
    solver_steps_per_period: int
    gpu_precision: str
    cpu_precision: str
    eps_list: np.ndarray
    A_list: np.ndarray
    tlist: np.ndarray
    warmup_time: float
    workers: int
    timings: bool
    progress: bool
    ignore_non_finite_output: bool = False
    qutip_drive_mode: str = "interpolated"
    comparison_output: str = "mean"
    julia_time_precision: str = "fp32"
    gqis_solver: str = "rk4"
    gqis_unroll: bool = False
    adaptive_rtol: float = 1e-8
    adaptive_atol: float = 1e-10
    adaptive_nsteps: int = 100_000
    problem: str = "two_level"
    kappa: float = 0.0
    pump_amplitude: float = 0.0
    coupling: float = 0.0
    resonator_frequency: float = 0.0
    photon_levels: int = 0
    regime_name: str = ""

    @property
    def nx(self) -> int:
        return int(len(self.eps_list))

    @property
    def ny(self) -> int:
        return int(len(self.A_list))

    @property
    def num_t(self) -> int:
        return int(len(self.tlist))

    @property
    def num_steps(self) -> int:
        return max(self.num_t - 1, 0)

    @property
    def dt(self) -> float:
        if self.num_t < 2:
            return 1.0
        return float(self.tlist[1] - self.tlist[0])

    @property
    def warmup_steps(self) -> int:
        return int(np.floor(self.warmup_time * self.num_steps))


# -----------------------------------------------------------------------------
# Problem definitions shared by GQIS, Julia, and generic CPU solvers
# -----------------------------------------------------------------------------

# USER MODEL EXTENSION POINT:
# These builders use the same objects as the ordinary GQIS examples: a SymPy
# Hamiltonian, time-dependent drive expression, collapse operators, observable,
# constant substitutions, and two sweep symbols. Copy one builder and its branch
# in problem_definition() when testing another model. A new QuTiP comparison also
# needs the equivalent construction in solve_one_point_qutip().


def two_level_sympy_model(cfg: BenchConfig):
    """Build the symbolic two-level model used by GQIS and Julia."""
    N = 2
    Delta, eps, Drive = sp.symbols("Delta eps Drive", real=True)
    A, t = sp.symbols("A t", real=True)
    gamma1S, gamma2S = sp.symbols("gamma1S gamma2S", real=True, nonnegative=True)

    sx = sp.Matrix([[0, 1], [1, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    sm = sp.Matrix([[0, 1], [0, 0]])
    mean_op = sp.zeros(N)
    mean_op[1, 1] = 1

    H = 0.5 * Delta * sx + 0.5 * Drive * sz
    drive_expr = eps + A * sp.sin(float(cfg.w) * t)
    col_ops = [sp.sqrt(gamma1S) * sm, sp.sqrt(gamma2S) * sz]
    const_values = {Delta: cfg.delta, gamma1S: cfg.gamma1, gamma2S: cfg.gamma2}
    return N, H, drive_expr, col_ops, mean_op, const_values, (eps, A)


def four_level_sympy_model(cfg: BenchConfig):
    """Build Benchmark 02's qubit-plus-two-state-resonator model."""
    N = cfg.photon_levels + 2
    eps, Drive = sp.symbols("eps Drive", real=True)
    A, t = sp.symbols("A t", real=True)

    sz = sp.Matrix([[1, 0], [0, -1]])
    sp_raise = sp.Matrix([[0, 1], [0, 0]])
    sp_lower = sp.Matrix([[0, 0], [1, 0]])
    qeye = sp.eye(N - 2)
    sm1 = sp.kronecker_product(sp_lower, qeye)
    sz1 = sp.kronecker_product(sz, qeye)
    a1 = sp.kronecker_product(qeye, sp_raise)

    hq11 = sz1 / 2
    hp = cfg.pump_amplitude * (a1.H + a1)
    hc11u = cfg.coupling * cfg.delta * (sm1.H * a1 + sm1 * a1.H)
    splitting = sp.sqrt(cfg.delta**2 + Drive**2)
    H = hp + hc11u / splitting + hq11 * (splitting - cfg.resonator_frequency)
    drive_expr = eps + A * sp.cos(float(cfg.w) * t)
    col_ops = [sp.sqrt(cfg.gamma1) * sm1, sp.sqrt(cfg.gamma2) * sz1,
               sp.sqrt(cfg.kappa) * a1]
    return N, H, drive_expr, col_ops, a1, {}, (eps, A)


def problem_definition(cfg: BenchConfig):
    """Return ``N, H, drive, c_ops, observable, constants, sweep_symbols``."""
    if cfg.problem == "two_level":
        return two_level_sympy_model(cfg)
    if cfg.problem == "four_level":
        return four_level_sympy_model(cfg)
    raise ValueError(f"Unsupported problem definition: {cfg.problem}")


def observable_output(value, problem: str):
    value = np.real(value)
    return np.abs(value) if problem == "four_level" else value


def julia_observable_output(expression: str, problem: str) -> str:
    return f"abs({expression})" if problem == "four_level" else expression


def solver_label(solver: str) -> str:
    return f"{SOLVER_NAMES[solver]}({SOLVER_METHOD_LABELS[solver]})"


def is_gqis_solver(solver: str) -> bool:
    return solver in GQIS_ACCURACY_SOLVERS


def is_julia_solver(solver: str) -> bool:
    return solver in JULIA_SOLVER_MODES


def gqis_method(solver: str, cfg: BenchConfig) -> str:
    return GQIS_ACCURACY_SOLVERS.get(solver, cfg.gqis_solver)


def print_progress(label: str, done: int, total: int, started: float,
                   unit: str = "columns") -> None:
    if total <= 0:
        return
    pct = 100.0 * done / total
    elapsed = max(time.perf_counter() - started, np.finfo(float).eps)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0.0 else np.inf
    width = 24
    filled = min(width, int(width * done / total))
    bar = "#" * filled + "-" * (width - filled)
    line = (f"{label}: [{bar}] {done}/{total} {unit} ({pct:5.1f}%)  "
            f"{rate:.2f} {unit}/s  ETA {eta:.0f}s")
    print(f"\r{line:<120}", end="", flush=True)
    if done >= total:
        print()


# -----------------------------------------------------------------------------
# GPU backend
# -----------------------------------------------------------------------------


def run_gpu_solver(cfg: BenchConfig) -> tuple[np.ndarray, float]:
    global LAST_GQIS_RHS_CODEGEN_S, LAST_GQIS_RHS_CACHE
    start = time.time()
    fp64 = cfg.gpu_precision == "fp64"
    scalar_dtype = np.float64 if fp64 else np.float32
    _, H, drive_expr, col_ops, mean_op, const_values, (eps, A) = problem_definition(cfg)

    # Diagnostic settings are passed as one group; the flat form (timings=..., return_timing_info=True,
    # ignore_non_finite_output=...) remains fully supported and equivalent.
    diagnostics = dict(timings=cfg.timings, return_timing_info=True,
                       ignore_non_finite_output=cfg.ignore_non_finite_output)
    result, timing_info = mesolve_2D(
        H, drive_expr, col_ops, mean_op, np.asarray(cfg.tlist, dtype=scalar_dtype),
        var_arrays={eps: np.asarray(cfg.eps_list, dtype=scalar_dtype),
                    A: np.asarray(cfg.A_list, dtype=scalar_dtype)},
        const_values=const_values, output_mode=cfg.comparison_output, fp64=fp64,
        solver=cfg.gqis_solver, unroll=cfg.gqis_unroll,
        solver_frequency=(cfg.w if cfg.gqis_solver.lower() == "anas5" else None),
        warmup_time=cfg.warmup_time, diagnostics=diagnostics)
    LAST_GQIS_RHS_CODEGEN_S = float(timing_info["rhs_codegen_s"])
    LAST_GQIS_RHS_CACHE = str(timing_info["cached_rhs"])

    # mesolve_2D returns (eps, A) for this var_arrays order; plotting expects (A, eps).
    p_mat = np.asarray(observable_output(result, cfg.problem)).T.astype(
        scalar_dtype, copy=False)
    return p_mat, time.time() - start


# -----------------------------------------------------------------------------
# Python and QuTiP CPU backends
# -----------------------------------------------------------------------------

_WORKER_CFG: BenchConfig | None = None
_WORKER_SOLVER: str | None = None
_WORKER_DRHO_FUN = None
_WORKER_OBS_FUN = None
_WORKER_RHO_LEN: int | None = None


def prepare_generic_cpu_rhs(cfg: BenchConfig):
    N, H, drive_expr, col_ops, mean_op, const_values, (eps, A) = problem_definition(cfg)
    Drive = next(symbol for symbol in H.free_symbols if symbol.name == "Drive")
    t = next(symbol for symbol in drive_expr.free_symbols if symbol.name == "t")
    drho_eqs, obs_expr, _, meta = build_reduced_lindblad_rhs(
        N, H.subs(Drive, drive_expr).subs(const_values),
        [op.subs(const_values) for op in col_ops], mean_op)
    rho_syms = meta["rho_syms"]
    drho_fun = sp.lambdify((t, eps, A, *rho_syms), drho_eqs, "numpy")
    obs_fun = sp.lambdify((t, eps, A, *rho_syms), obs_expr, "numpy")
    return drho_fun, obs_fun, len(rho_syms)


def init_worker(cfg: BenchConfig, solver: str, ready_queue=None) -> None:
    global _WORKER_CFG, _WORKER_SOLVER, _WORKER_DRHO_FUN, _WORKER_OBS_FUN, _WORKER_RHO_LEN
    _WORKER_CFG = cfg
    _WORKER_SOLVER = solver
    if cfg.problem != "two_level" and solver in {"python_cpu", "python_ode_cpu"}:
        _WORKER_DRHO_FUN, _WORKER_OBS_FUN, _WORKER_RHO_LEN = prepare_generic_cpu_rhs(cfg)
    if solver == "qutip_cpu":
        import qutip  # noqa: F401
    if ready_queue is not None:
        ready_queue.put(True)


def drho_two_level(rho: np.ndarray, t: float, A: float, eps0: float, cfg: BenchConfig, scalar_dtype,
                   complex_dtype) -> np.ndarray:
    rho00, rho01, rho10, rho11 = rho
    drive = scalar_dtype(eps0 + A * np.sin(scalar_dtype(cfg.w * t)))
    hdelta = scalar_dtype(0.5 * cfg.delta)
    gamma1 = scalar_dtype(cfg.gamma1)
    gamma2 = scalar_dtype(cfg.gamma2)
    gcoh = scalar_dtype(0.5) * gamma1 + scalar_dtype(2.0) * gamma2
    j = complex_dtype(1j)

    d00 = -j * hdelta * (rho10 - rho01) + gamma1 * rho11
    d01 = -j * (drive * rho01 + hdelta * (rho11 - rho00)) - gcoh * rho01
    d10 = -j * (hdelta * (rho00 - rho11) - drive * rho10) - gcoh * rho10
    d11 = -j * hdelta * (rho01 - rho10) - gamma1 * rho11
    return np.array([d00, d01, d10, d11], dtype=complex_dtype)


def solve_one_point_python_rk4(A: float, eps0: float, cfg: BenchConfig) -> float:
    scalar_dtype = np.float32 if cfg.cpu_precision == "fp32" else np.float64
    dt = scalar_dtype(cfg.dt)
    dt2 = scalar_dtype(0.5) * dt
    dt6 = dt / scalar_dtype(6.0)
    t0 = scalar_dtype(cfg.tlist[0]) if cfg.num_t else scalar_dtype(0.0)
    if cfg.problem == "two_level":
        complex_dtype = np.complex64 if cfg.cpu_precision == "fp32" else np.complex128
        rho = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex_dtype)
        def rhs(tt, state):
            return drho_two_level(state, tt, A, eps0, cfg, scalar_dtype, complex_dtype)

        def observe(tt, state):
            return scalar_dtype(np.real(state[3]))
    else:
        if _WORKER_DRHO_FUN is None or _WORKER_OBS_FUN is None or _WORKER_RHO_LEN is None:
            raise RuntimeError("Generic CPU RHS was not initialized.")
        rho = np.zeros(_WORKER_RHO_LEN, dtype=scalar_dtype)
        rho[0] = scalar_dtype(1.0)
        def rhs(tt, state):
            return np.asarray(_WORKER_DRHO_FUN(tt, eps0, A, *state), dtype=scalar_dtype)

        def observe(tt, state):
            return scalar_dtype(_WORKER_OBS_FUN(tt, eps0, A, *state))

    accum = scalar_dtype(0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(cfg.num_steps):
            t = t0 + scalar_dtype(step) * dt
            k1 = rhs(t, rho)
            k2 = rhs(t + dt2, rho + dt2 * k1)
            k3 = rhs(t + dt2, rho + dt2 * k2)
            k4 = rhs(t + dt, rho + dt * k3)
            rho = rho + dt6 * (k1 + scalar_dtype(2.0) * k2
                               + scalar_dtype(2.0) * k3 + k4)
            if not np.all(np.isfinite(rho)):
                return np.nan
            if cfg.comparison_output == "mean" and step >= cfg.warmup_steps:
                accum += observe(t + dt, rho)
                if not np.isfinite(accum):
                    return np.nan
    if cfg.comparison_output == "final":
        final_value = float(observe(t0 + scalar_dtype(cfg.num_steps) * dt, rho))
        return float(observable_output(final_value, cfg.problem))
    mean = float(accum / scalar_dtype(max(cfg.num_steps - cfg.warmup_steps, 1)))
    return float(observable_output(mean, cfg.problem))


def solve_one_point_python_ode(A: float, eps0: float, cfg: BenchConfig) -> float:
    from scipy.integrate import solve_ivp

    scalar_dtype = np.float64
    complex_dtype = np.complex128
    tlist = np.asarray(cfg.tlist, dtype=np.float64)
    if len(tlist) < 2:
        return np.nan

    if cfg.problem != "two_level":
        if _WORKER_DRHO_FUN is None or _WORKER_OBS_FUN is None or _WORKER_RHO_LEN is None:
            raise RuntimeError("Generic CPU RHS was not initialized.")
        rho0 = np.zeros(_WORKER_RHO_LEN, dtype=np.float64)
        rho0[0] = 1.0
        def rhs(t, rho):
            return np.asarray(_WORKER_DRHO_FUN(t, eps0, A, *rho), dtype=np.float64)
    else:
        rho0 = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=complex_dtype)
        def rhs(t, rho):
            return drho_two_level(rho, t, A, eps0, cfg, scalar_dtype, complex_dtype)

    sol = solve_ivp(rhs, (float(tlist[0]), float(tlist[-1])), rho0, method="RK45", t_eval=tlist,
                    rtol=cfg.adaptive_rtol, atol=cfg.adaptive_atol)
    if not sol.success or sol.y.shape[1] == 0:
        return np.nan
    warmup = cfg.warmup_steps
    start = min(warmup + 1, sol.y.shape[1] - 1)
    if cfg.problem != "two_level":
        values = [_WORKER_OBS_FUN(t, eps0, A, *sol.y[:, index])
                  for index, t in enumerate(tlist)]
        if cfg.comparison_output == "final":
            return float(observable_output(values[-1], cfg.problem))
        return float(observable_output(np.mean(values[start:]), cfg.problem))
    if cfg.comparison_output == "final":
        return float(np.real(sol.y[3, -1]))
    return float(np.mean(np.real(sol.y[3, start:])))


def solve_one_point_qutip(A: float, eps0: float, cfg: BenchConfig) -> float:
    if cfg.problem == "four_level":
        return solve_one_point_four_level_qutip(A, eps0, cfg)
    import qutip as qt

    sx = qt.sigmax()
    sz = qt.sigmaz()
    relaxation_op = qt.basis(2, 0) * qt.basis(2, 1).dag()
    proj_exc = qt.basis(2, 1) * qt.basis(2, 1).dag()
    rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()

    tlist = np.asarray(cfg.tlist, dtype=float)
    H0 = 0.5 * cfg.delta * sx + 0.5 * eps0 * sz
    H1 = 0.5 * A * sz
    if cfg.qutip_drive_mode == "analytic":
        H = [H0, [H1, lambda t: np.sin(cfg.w * t)]]
    else:
        H = [H0, [H1, np.sin(cfg.w * tlist)]]

    c_ops = []
    if cfg.gamma1 > 0:
        c_ops.append(np.sqrt(cfg.gamma1) * relaxation_op)
    if cfg.gamma2 > 0:
        c_ops.append(np.sqrt(cfg.gamma2) * sz)

    options = {"method": "adams", "rtol": cfg.adaptive_rtol, "atol": cfg.adaptive_atol,
               "nsteps": cfg.adaptive_nsteps}
    res = qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[proj_exc], options=options)
    p = np.asarray(np.real(res.expect[0]), dtype=np.float64)
    if cfg.comparison_output == "final":
        return float(p[-1])
    warmup = cfg.warmup_steps
    start = min(warmup + 1, len(p) - 1)
    return float(np.mean(p[start:]))


def solve_one_point_four_level_qutip(A: float, eps0: float, cfg: BenchConfig) -> float:
    import qutip as qt

    qeye = np.eye(cfg.photon_levels, dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sp_raise = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sp_lower = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    sm1 = np.kron(sp_lower, qeye)
    sz1 = np.kron(sz, qeye)
    a1 = np.kron(qeye, sp_raise)
    hq11 = sz1 / 2.0
    hp = cfg.pump_amplitude * (a1.conj().T + a1)
    hc11u = cfg.coupling * cfg.delta * (sm1.conj().T @ a1 + sm1 @ a1.conj().T)
    tlist = np.asarray(cfg.tlist, dtype=float)

    if cfg.qutip_drive_mode == "analytic":
        def splitting(t):
            return np.sqrt(cfg.delta**2 + (eps0 + A * np.cos(cfg.w * t))**2)
        H = [qt.Qobj(hp), [qt.Qobj(hc11u), lambda t: 1.0 / splitting(t)],
             [qt.Qobj(hq11), lambda t: splitting(t) - cfg.resonator_frequency]]
    else:
        drive = eps0 + A * np.cos(cfg.w * tlist)
        splitting = np.sqrt(cfg.delta**2 + drive**2)
        H = [qt.Qobj(hp), [qt.Qobj(hc11u), 1.0 / splitting],
             [qt.Qobj(hq11), splitting - cfg.resonator_frequency]]

    c_ops = [np.sqrt(cfg.gamma1) * qt.Qobj(sm1),
             np.sqrt(cfg.gamma2) * qt.Qobj(sz1),
             np.sqrt(cfg.kappa) * qt.Qobj(a1)]
    rho0 = qt.basis(cfg.photon_levels + 2, 0).proj()
    options = {"method": "adams", "rtol": cfg.adaptive_rtol, "atol": cfg.adaptive_atol,
               "nsteps": cfg.adaptive_nsteps}
    res = qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[qt.Qobj(a1)], options=options)
    values = np.asarray(np.real(res.expect[0]), dtype=np.float64)
    if cfg.comparison_output == "final":
        return float(observable_output(values[-1], cfg.problem))
    start = min(cfg.warmup_steps + 1, len(values) - 1)
    return float(observable_output(np.mean(values[start:]), cfg.problem))


def solve_cpu_column(eps_index: int) -> tuple[int, np.ndarray]:
    if _WORKER_CFG is None or _WORKER_SOLVER is None:
        raise RuntimeError("CPU worker was not initialized.")

    cfg = _WORKER_CFG
    eps0 = float(cfg.eps_list[eps_index])
    dtype = (np.float32 if
             (_WORKER_SOLVER == "python_cpu" and cfg.cpu_precision == "fp32") else np.float64)
    col = np.empty(cfg.ny, dtype=dtype)

    if _WORKER_SOLVER == "python_cpu":
        solve_point = solve_one_point_python_rk4
    elif _WORKER_SOLVER == "python_ode_cpu":
        solve_point = solve_one_point_python_ode
    elif _WORKER_SOLVER == "qutip_cpu":
        solve_point = solve_one_point_qutip
    else:
        raise ValueError(f"Unsupported CPU solver: {_WORKER_SOLVER}")

    for i, A in enumerate(cfg.A_list):
        col[i] = solve_point(float(A), eps0, cfg)
    return eps_index, col


def run_cpu_solver(solver: str, cfg: BenchConfig,
                   start_message: str | None = None) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    dtype = np.float32 if (solver == "python_cpu" and cfg.cpu_precision == "fp32") else np.float64
    out = np.empty((cfg.ny, cfg.nx), dtype=dtype)

    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue() if start_message else None
    with ctx.Pool(processes=cfg.workers, initializer=init_worker,
                  initargs=(cfg, solver, ready_queue)) as pool:
        progress_start = start
        if ready_queue is not None:
            for _ in range(cfg.workers):
                ready_queue.get(timeout=300.0)
            print(start_message, flush=True)
            progress_start = time.perf_counter()
        done = 0
        last_update = 0.0
        update_every = max(1, cfg.nx // 50)
        for eps_index, col in pool.imap_unordered(solve_cpu_column, range(cfg.nx)):
            out[:, eps_index] = col
            done += 1
            now = time.perf_counter()
            update = (done == 1 or done == cfg.nx or
                      (solver == "qutip_cpu" and now - last_update >= 0.5) or
                      (solver != "qutip_cpu" and done % update_every == 0))
            if cfg.progress and update:
                print_progress(solver, done, cfg.nx, progress_start)
                last_update = now

    return out, time.perf_counter() - start


# -----------------------------------------------------------------------------
# Julia GPU backend
# -----------------------------------------------------------------------------


def run_julia_gpu_solver(cfg: BenchConfig, *, julia_cmd: str) -> tuple[np.ndarray, float]:
    """Julia DiffEqGPU GPUTsit5 on the GQIS-compiled RHS in ``cfg.julia_time_precision``."""
    global LAST_JULIA_TOTAL_S, LAST_JULIA_PRECALC_S
    from julia_diffeqgpu import julia_map_script, julia_model, map_arguments, run_julia_script

    LAST_JULIA_TOTAL_S = np.nan
    LAST_JULIA_PRECALC_S = np.nan
    _, H, drive_expr, col_ops, mean_op, const_values, sweep = problem_definition(cfg)
    mean = cfg.comparison_output == "mean"
    prep_start = time.time()
    model = julia_model(H, drive_expr, col_ops, mean_op, sweep, const_values=const_values,
                        precision=cfg.julia_time_precision,
                        average="after_warmup" if mean else None, observe=not mean)
    source = julia_map_script(model, output_abs=cfg.problem == "four_level")
    prep_elapsed = time.time() - prep_start
    arguments = map_arguments(cfg.nx, cfg.ny, cfg.num_t, cfg.dt, cfg.eps_list, cfg.A_list,
                              cfg.warmup_steps, float(cfg.tlist[0]) if cfg.num_t else 0.0)
    values, p_mat, process_elapsed, _ = run_julia_script(
        source, arguments, julia_cmd=julia_cmd,
        output_dtype=np.float64 if cfg.julia_time_precision == "fp64" else np.float32)
    julia_elapsed = float(values["SOLVE_TIME"])
    LAST_JULIA_TOTAL_S = prep_elapsed + process_elapsed
    LAST_JULIA_PRECALC_S = max(0.0, LAST_JULIA_TOTAL_S - julia_elapsed)
    if cfg.timings:
        print(f"julia_gpu timings: prep={prep_elapsed:.3f}s "
              f"julia_solve={julia_elapsed:.3f}s total={LAST_JULIA_TOTAL_S:.3f}s "
              f"precalc={LAST_JULIA_PRECALC_S:.3f}s")
    return p_mat, julia_elapsed


# -----------------------------------------------------------------------------
# Accuracy-sweep machinery
# -----------------------------------------------------------------------------


def _accuracy_config(settings: dict, steps_per_period: int, *, reference: bool) -> BenchConfig:
    grid_side = int(settings["grid_side_dimension"])
    steps_per_period = int(steps_per_period)
    simulation_periods = float(settings["simulation_periods"])
    problem = str(settings["problem"]).lower()
    if problem == "two_level":
        model = settings["two_level_parameters"]
        delta = float(model["delta"])
        w = float(model["w_over_delta"] * delta)
        period = 2.0 * np.pi / w
        gamma1 = float(model["gamma1_per_period"] / period)
        gamma2 = float(gamma1 / 2.0 + model["gamma_phi_per_period"] / period)
        eps_limit = float(model["eps_max_over_w"] * w)
        A_limit = float(model["amplitude_max_over_w"] * w)
        extra = {}
    elif problem == "four_level":
        model = settings["four_level_parameters"]
        regime_name = str(settings["four_level_regime"]).lower()
        if regime_name not in model["regimes"]:
            raise ValueError("four_level_regime must name an entry in four_level_parameters['regimes'].")
        regime = model["regimes"][regime_name]
        delta, w = float(model["qubit_frequency"]), float(regime["drive_frequency"])
        gamma1 = float(model["gamma1"])
        gamma2 = float(gamma1 / 2.0 + model["gamma_phi"])
        period = 2.0 * np.pi / w
        eps_limit = float(model["eps_max_over_qubit_frequency"] * delta)
        A_limit = float(model["amplitude_max_over_qubit_frequency"]
                        * regime["amplitude_factor"] * delta)
        extra = {"kappa": float(model["kappa"]),
                 "pump_amplitude": float(model["probe_amplitude"] * delta),
                 "coupling": float(model["coupling_over_qubit_frequency"] * delta),
                 "resonator_frequency": float(model["resonator_frequency"]),
                 "photon_levels": int(model["photon_levels"]),
                 "regime_name": regime_name}
    else:
        # A custom branch must define the drive frequency/period, sweep limits,
        # decay values, and any extra fields consumed by its model builder.
        raise ValueError("problem must be 'two_level' or 'four_level'.")
    num_steps = max(1, int(round(simulation_periods * steps_per_period)))
    eps_list = np.linspace(-eps_limit, eps_limit, grid_side, dtype=np.float32)
    A_list = np.linspace(0.0, A_limit, grid_side, dtype=np.float32)
    tlist = np.linspace(0.0, simulation_periods * period, num_steps + 1, dtype=np.float32)
    qutip_drive_mode = str(settings["qutip_drive_mode"]).lower()
    comparison_output = str(settings["comparison_output"]).lower()
    gqis_unroll = settings["GQIS_unroll"]
    if qutip_drive_mode not in {"analytic", "interpolated"}:
        raise ValueError("qutip_drive_mode must be 'analytic' or 'interpolated'.")
    if comparison_output not in {"mean", "final"}:
        raise ValueError("comparison_output must be 'mean' or 'final'.")
    if not isinstance(gqis_unroll, (bool, np.bool_)):
        raise ValueError("GQIS_unroll must be True or False.")
    prefix = "reference" if reference else "target"
    return BenchConfig(
        delta=delta, w=w, gamma1=gamma1, gamma2=gamma2, tr=simulation_periods,
        solver_steps_per_period=steps_per_period, gpu_precision="fp32", cpu_precision="fp64",
        eps_list=eps_list, A_list=A_list, tlist=tlist, warmup_time=0.0,
        workers=max(1, (os.cpu_count() or 2) - 1), timings=False, progress=False,
        ignore_non_finite_output=bool(settings["ignore_non_finite_output"]),
        qutip_drive_mode=qutip_drive_mode, comparison_output=comparison_output,
        julia_time_precision="fp32",
        gqis_solver="rk4", gqis_unroll=bool(gqis_unroll),
        adaptive_rtol=float(settings[f"{prefix}_rtol"]),
        adaptive_atol=float(settings[f"{prefix}_atol"]),
        adaptive_nsteps=int(settings["adaptive_nsteps"]),
        problem=problem, **extra,
    )


def _reduced_time_grid(cfg: BenchConfig, requested_divider: float) -> BenchConfig:
    steps_per_period = max(1, int(round(cfg.solver_steps_per_period / requested_divider)))
    num_steps = max(1, int(round(cfg.tr * steps_per_period)))
    period = 2.0 * np.pi / cfg.w
    tlist = np.linspace(float(cfg.tlist[0]), cfg.tr * period, num_steps + 1,
                        dtype=cfg.tlist.dtype)
    return replace(cfg, solver_steps_per_period=steps_per_period, tlist=tlist, progress=False)


def _solver_names(value) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",")) if isinstance(value, str) else tuple(value)
    unknown = [name for name in names if name not in ACCURACY_SOLVER_SET]
    if unknown:
        raise ValueError(f"Unknown target solvers: {', '.join(unknown)}")
    return names


def _run_accuracy_solver(name: str, cfg: BenchConfig, settings: dict,
                         start_message: str | None = None) -> tuple[np.ndarray, float]:
    """Run directly on the accuracy-test grid without benchmark CPU dividers."""
    if is_julia_solver(name):
        return run_julia_gpu_solver(
            replace(cfg, julia_time_precision=JULIA_SOLVER_MODES[name]),
            julia_cmd=settings["julia_cmd"])
    if name == "qutip_cpu":
        cfg = replace(cfg, progress=bool(settings["show_worker_progress"]))
        return run_cpu_solver(name, cfg, start_message=start_message)
    if is_gqis_solver(name):
        return run_gpu_solver(replace(cfg, gqis_solver=gqis_method(name, cfg)))
    return run_cpu_solver(name, cfg)


def _warm_gqis(cfg: BenchConfig, solver: str, *, first: bool) -> float:
    side = min(16, cfg.nx, cfg.ny)
    warm_cfg = replace(
        cfg,
        eps_list=np.linspace(float(cfg.eps_list[0]), float(cfg.eps_list[-1]), side,
                             dtype=cfg.eps_list.dtype),
        A_list=np.linspace(float(cfg.A_list[0]), float(cfg.A_list[-1]), side,
                           dtype=cfg.A_list.dtype), progress=False,
        gqis_solver=gqis_method(solver, cfg))
    if first:
        print(f"Preparing shared GQIS symbolic RHS and compiling/warming "
              f"{solver_label(solver)} CUDA kernel...")
    else:
        print(f"Compiling/warming {solver_label(solver)} CUDA kernel with shared RHS...")
    run_gpu_solver(warm_cfg)
    if first:
        if LAST_GQIS_RHS_CACHE == "miss":
            print(f"Shared GQIS symbolic RHS preparation time: {LAST_GQIS_RHS_CODEGEN_S:.3f}s")
            return float(LAST_GQIS_RHS_CODEGEN_S)
        else:
            print("Shared GQIS symbolic RHS already cached; preparation time: 0.000s")
            return 0.0
    return np.nan


def _write_accuracy_rows(rows: list[dict], path: Path, *, append: bool = False) -> None:
    if not rows:
        return
    with path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def _error_metrics(candidate: np.ndarray, reference: np.ndarray,
                   reference_range: float) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    absolute_difference = np.abs(difference)
    mse = float(np.mean(absolute_difference**2))
    rms = float(np.sqrt(mse))
    return {
        "mse": mse,
        "rms": rms,
        "normalized_rms": rms / reference_range if reference_range > 0.0 else np.nan,
        "max_abs": float(np.max(absolute_difference)),
        "p95_abs": float(np.percentile(absolute_difference, 95.0)),
        "p99_abs": float(np.percentile(absolute_difference, 99.0)),
    }


def _reference_metadata(settings: dict, cfg: BenchConfig) -> dict:
    metadata = {
        "relaxation_operator": ("basis_1_to_basis_0" if cfg.problem == "two_level"
                                else "benchmark_02_qubit_resonator_collapse_set"),
        "problem": cfg.problem,
        "comparison_output": cfg.comparison_output,
        "reference_solver": settings["reference_solver"],
        "qutip_drive_mode": (settings["qutip_drive_mode"]
                             if settings["reference_solver"] == "qutip_cpu" else None),
        "grid_side_dimension": cfg.nx,
        "steps_per_period": cfg.solver_steps_per_period,
        "simulation_periods": cfg.tr,
        "rtol": cfg.adaptive_rtol,
        "atol": cfg.adaptive_atol,
        "Delta": cfg.delta,
        "w": cfg.w,
        "gamma1": cfg.gamma1,
        "gamma2": cfg.gamma2,
        "eps_min": float(cfg.eps_list[0]),
        "eps_max": float(cfg.eps_list[-1]),
        "A_min": float(cfg.A_list[0]),
        "A_max": float(cfg.A_list[-1]),
    }
    if cfg.problem == "four_level":
        metadata.update({
            "four_level_regime": cfg.regime_name,
            "kappa": cfg.kappa,
            "pump_amplitude": cfg.pump_amplitude,
            "coupling": cfg.coupling,
            "resonator_frequency": cfg.resonator_frequency,
            "photon_levels": cfg.photon_levels,
        })
    if is_julia_solver(settings["reference_solver"]):
        metadata["julia_time_precision"] = JULIA_SOLVER_MODES[settings["reference_solver"]]
    if is_gqis_solver(settings["reference_solver"]):
        metadata["gqis_solver"] = gqis_method(settings["reference_solver"], cfg)
        metadata["gqis_unroll"] = cfg.gqis_unroll
        if metadata["gqis_solver"].lower() == "anas5":
            metadata["gqis_solver_frequency"] = cfg.w
    return metadata


def _load_reference(path: Path, expected_metadata: dict,
                    expected_shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Saved reference not found: {path}")
    with np.load(path, allow_pickle=False) as saved:
        required = {"result", "calculation_time_s", "metadata"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"Saved reference is missing {', '.join(sorted(missing))}: {path}")
        metadata = json.loads(str(saved["metadata"].item()))
        metadata.setdefault("problem", "two_level")
        metadata.setdefault("comparison_output", "mean")
        if "qutip_drive_mode" not in metadata:
            metadata["qutip_drive_mode"] = (
                "interpolated" if metadata.get("reference_solver") == "qutip_cpu" else None)
        if (metadata.get("reference_solver") == "gpu"
                and expected_metadata.get("reference_solver") == "gqis_rk4"):
            metadata["reference_solver"] = "gqis_rk4"
        if is_gqis_solver(metadata.get("reference_solver", "")):
            metadata.setdefault("gqis_solver", "rk4")
            metadata.setdefault("gqis_unroll", False)
        if metadata != expected_metadata:
            raise ValueError(f"Saved reference metadata does not match current settings: {path}")
        result = np.asarray(saved["result"])
        calculation_time = float(saved["calculation_time_s"].item())
    if result.shape != expected_shape or not np.all(np.isfinite(result)):
        raise ValueError(f"Saved reference has invalid result data: {path}")
    if not np.isfinite(calculation_time) or calculation_time < 0.0:
        raise ValueError(f"Saved reference has invalid calculation time: {path}")
    return result, calculation_time


def _save_reference(path: Path, result: np.ndarray, calculation_time: float,
                    metadata: dict) -> None:
    result = np.asarray(result)
    if not np.all(np.isfinite(result)) or not np.isfinite(calculation_time):
        raise ValueError("Reference result or calculation time is not finite; cache not saved.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp.npz")
    try:
        np.savez_compressed(temporary_path, result=result, calculation_time_s=calculation_time,
                            metadata=json.dumps(metadata, sort_keys=True))
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _divider_axis_options(settings: dict) -> tuple[str, str]:
    scale = str(settings.get("divider_axis_scale", "log2")).lower()
    labels = str(settings.get("divider_axis_labels", "powers_of_two")).lower()
    if scale not in {"log2", "equidistant"}:
        raise ValueError("divider_axis_scale must be 'log2' or 'equidistant'.")
    if labels not in {"powers_of_two", "all"}:
        raise ValueError("divider_axis_labels must be 'powers_of_two' or 'all'.")
    return scale, labels


def _plot_accuracy_sweep(rows: list[dict], settings: dict, reference_time: float,
                         gqis_preparation_time: float, output_path: Path) -> None:
    solvers = _solver_names(settings["target_solvers"])
    show_table = bool(settings.get("show_results_table", True))
    times_only = bool(settings.get("table_times_only", False))
    show_best_row = bool(settings.get("show_best_summary_row", False))
    def format_time(seconds: float) -> str:
        if not np.isfinite(seconds):
            return "--"
        if seconds >= 3600:
            minutes = round(seconds / 60)
            return f"{minutes // 60} h {minutes % 60} min"
        if seconds >= 60:
            return f"{int(seconds // 60)} min {seconds % 60:.3g} s"
        return f"{seconds:.3g}s"

    table_fields = [("target_calculation_time_s", "time", ".3g")]
    if not times_only:
        table_fields += [("rms", "RMS", ".3e"), ("max_abs", "max error", ".3e")]
    table_rows = 1 + len(table_fields) * len(solvers)
    has_table_area = show_table or show_best_row
    if show_table:
        table_height = max(2.5, 0.235 * table_rows) + (0.8 if show_best_row else 0.0)
    else:
        table_height = 1.3 if show_best_row else 0.0
    figure_height = 5.6 + table_height if has_table_area else 6.2
    table_fraction = ((table_height + 0.25) / figure_height if has_table_area else 0.1)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, figure_height))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=table_fraction, wspace=0.28)
    ax_error, ax_time, ax_work = axes
    colors = plt.get_cmap("tab20" if len(solvers) > 10 else "tab10")
    error_cap = 1.0
    requested_dividers = tuple(float(value) for value in settings["step_count_dividers"])
    axis_scale, axis_labels = _divider_axis_options(settings)
    if any(value <= 0.0 for value in requested_dividers):
        raise ValueError("Divider-axis values must be positive.")
    axis_quantity = settings.get("time_grid_axis", "steps_per_period")
    if axis_quantity not in {"steps_per_period", "divider"}:
        raise ValueError("time_grid_axis must be 'steps_per_period' or 'divider'.")
    steps_axis = axis_quantity == "steps_per_period"
    axis_values = {value: (next(int(float(row["target_steps_per_period"])) for row in rows
                               if float(row["requested_step_count_divider"]) == value)
                           if steps_axis else value) for value in requested_dividers}
    divider_positions = (axis_values
                         if axis_scale == "log2" else
                         {value: index for index, value in enumerate(requested_dividers)})
    tick_dividers = requested_dividers
    if axis_labels == "powers_of_two":
        tick_dividers = tuple(value for index, value in enumerate(requested_dividers)
                              if np.isclose(np.log2(axis_values[value]), round(np.log2(axis_values[value])))
                              or index in {0, len(requested_dividers) - 1})
    x_ticks = [divider_positions[value] for value in tick_dividers]

    def format_divider(value: float, _position=None) -> str:
        if axis_scale == "log2":
            return f"{value:g}"
        index = int(round(value))
        return (f"{axis_values[requested_dividers[index]]:g}"
                if 0 <= index < len(requested_dividers) else "")

    def configure_divider_axis(axis) -> None:
        if axis_scale == "log2":
            axis.set_xscale("log", base=2)
            margin = 2.0 ** 0.15
            axis.set_xlim(min(axis_values.values()) / margin,
                          max(axis_values.values()) * margin)
        else:
            axis.set_xlim(-0.5, len(requested_dividers) - 0.5)
        axis.set_xticks(x_ticks)
        axis.xaxis.set_major_formatter(FuncFormatter(format_divider))
        if steps_axis and axis_scale == "log2":
            axis.invert_xaxis()

    for index, solver in enumerate(solvers):
        solver_rows = [row for row in rows if row["target_solver"] == solver
                       and np.isfinite(float(row["target_calculation_time_s"]))]
        if not solver_rows:
            continue
        color = colors(index)
        label = solver_label(solver)
        x_positions = np.asarray([
            divider_positions[float(row["requested_step_count_divider"])]
            for row in solver_rows])
        rms = np.asarray([float(row["rms"]) for row in solver_rows])
        max_abs = np.asarray([float(row["max_abs"]) for row in solver_rows])
        calculation_time = np.asarray([float(row["target_calculation_time_s"])
                                       for row in solver_rows])
        accepted = np.asarray([row["acceptable"] == "yes" for row in solver_rows])
        self_reference = np.asarray([
            solver == settings["reference_solver"]
            and int(row["target_steps_per_period"]) == int(row["reference_steps_per_period"])
            and float(row["rms"]) == 0.0 and float(row["max_abs"]) == 0.0
            for row in solver_rows])
        finite_rms = np.isfinite(rms)
        finite_max = np.isfinite(max_abs)
        plotted_rms = finite_rms & ~self_reference
        plotted_max = finite_max & ~self_reference

        displayed_rms = np.clip(rms[plotted_rms], np.finfo(float).tiny, error_cap)
        ax_error.plot(x_positions[plotted_rms], displayed_rms, "o-", color=color,
                      label=label)
        displayed_max = np.clip(max_abs[plotted_max], np.finfo(float).tiny, error_cap)
        ax_error.plot(x_positions[plotted_max], displayed_max, "s--", color=color, alpha=0.75,
                      label="_nolegend_")
        finite_error = finite_rms & finite_max
        ax_error.scatter(x_positions[~finite_error], np.full(np.sum(~finite_error), error_cap),
                         marker="X", s=65, color=color, zorder=5, label="_nolegend_")
        ax_time.plot(x_positions, calculation_time, "-", color=color, label=label)
        ax_time.scatter(x_positions[accepted], calculation_time[accepted], marker="o",
                        color=color, zorder=4)
        ax_time.scatter(x_positions[~accepted], calculation_time[~accepted], marker="X", s=45,
                        color=color, edgecolor="black", linewidth=0.4, zorder=5)
        if is_julia_solver(solver):
            precalc_time = np.asarray([float(row["julia_precalculation_time_s"])
                                       for row in solver_rows])
            finite_precalc = np.isfinite(precalc_time) & (precalc_time > 0.0)
            if np.any(finite_precalc):
                mean_precalc = float(np.mean(precalc_time[finite_precalc]))
                ax_time.axhline(mean_precalc, color=color, linestyle=":", alpha=0.8,
                                label=f"{label} precalc mean={mean_precalc:.3g}s")
        finite_accepted = accepted[plotted_rms]
        ax_work.plot(displayed_rms, calculation_time[plotted_rms], "-", color=color, label=label)
        ax_work.scatter(displayed_rms[finite_accepted],
                        calculation_time[plotted_rms][finite_accepted], marker="o",
                        color=color, zorder=4)
        ax_work.scatter(displayed_rms[~finite_accepted],
                        calculation_time[plotted_rms][~finite_accepted], marker="X", s=45,
                        color=color, edgecolor="black", linewidth=0.4, zorder=5)
        ax_work.scatter(np.full(np.sum(~finite_rms), error_cap),
                        calculation_time[~finite_rms], marker="X", s=45, color=color, zorder=5)
        work_x = np.where(finite_rms,
                          np.clip(rms, np.finfo(float).tiny, error_cap), error_cap)
        label_offset = -6 if index % 2 == 0 else 7
        step_counts = np.asarray([float(row["target_steps_per_period"]) for row in solver_rows])
        for x, y, steps in zip(work_x[~self_reference], calculation_time[~self_reference],
                               step_counts[~self_reference]):
            ax_work.annotate(f"{steps:g}", (x, y), xytext=(-4, label_offset),
                             textcoords="offset points", ha="right", fontsize=7,
                             color=color)

    ax_error.plot([], [], "o-", color="0.3", label="solid line with circles: RMS")
    ax_error.plot([], [], "s--", color="0.3", label="dashed line with squares: maximum error")
    ax_error.axhline(settings["acceptable_rms"], color="0.45", linestyle="-",
                     label="RMS acceptance limit")
    ax_error.axhline(settings["acceptable_max_abs"], color="0.45", linestyle="--",
                     label="maximum error acceptance limit")
    reference_divider = (float(settings["target_base_steps_per_period"])
                         / float(settings["reference_steps_per_period"]))
    matching_divider = next((value for value in requested_dividers
                             if np.isclose(value, reference_divider)), None)
    if steps_axis and axis_scale == "log2":
        ax_error.axvline(settings["reference_steps_per_period"], color="0.4", linestyle="-.",
                         label=f"reference grid: {settings['reference_steps_per_period']:g} per period")
    elif matching_divider is not None:
        ax_error.axvline(divider_positions[matching_divider], color="0.4", linestyle="-.",
                         label=f"reference grid divider={matching_divider:g}")
    divider_xlabel = "Integration steps per drive period" if steps_axis else "Step-count divider"
    if steps_axis and any(s in {"qutip_cpu", "python_ode_cpu"} for s in solvers):
        divider_xlabel += "\n(adaptive CPU solvers: output samples per period)"
    if axis_scale == "log2":
        divider_xlabel += " (log2 scale)"
    ax_error.set(xlabel=divider_xlabel,
                 ylabel="RMS and maximum absolute deviation from reference",
                 title=f"RMS and maximum error relative to "
                       f"{solver_label(settings['reference_solver'])}")
    ax_error.set_yscale("log")
    ax_error.set_ylim(top=error_cap)
    configure_divider_axis(ax_error)
    ax_error.grid(True, which="both", alpha=0.3)
    legend_columns = 2 if len(solvers) > 4 else 1
    ax_error.legend(fontsize=7 if len(solvers) > 4 else 8, ncol=legend_columns)

    if np.isfinite(gqis_preparation_time) and gqis_preparation_time > 0.0:
        ax_time.axhline(gqis_preparation_time, color="tab:blue", linestyle=":", alpha=0.8,
                       label=f"GQIS shared RHS preparation={gqis_preparation_time:.3g}s")
    ax_time.axhline(reference_time, color="0.25", linestyle=":",
                    label=f"reference: {solver_label(settings['reference_solver'])}")
    ax_time.set(xlabel=divider_xlabel, ylabel="Calculation time [s]",
                title="Calculation time versus time-grid resolution")
    ax_time.set_yscale("log")
    configure_divider_axis(ax_time)
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.scatter([], [], marker="X", s=45, color="0.4", edgecolor="black",
                    linewidth=0.4, label="exceeds accuracy limits")
    ax_time.legend(fontsize=8 if len(solvers) > 4 else None, ncol=legend_columns)

    ax_work.axvline(settings["acceptable_rms"], color="0.25", linestyle=":")
    ax_work.set(xlabel="RMS deviation from reference", ylabel="Calculation time [s]",
                title="Work-precision comparison (labels: steps/period)")
    ax_work.set_xscale("log")
    ax_work.set_yscale("log")
    ax_work.set_xlim(right=error_cap)
    ax_work.grid(True, which="both", alpha=0.3)
    ax_work.legend(fontsize=8 if len(solvers) > 4 else None, ncol=legend_columns)

    problem_title = "Two-level system" if settings["problem"] == "two_level" else "Four-level system"
    output_title = ("time-averaged observable" if settings["comparison_output"] == "mean"
                    else "final-state observable")
    fig.suptitle(f"{problem_title}, {settings['simulation_periods']:g} drive periods, "
                 f"{settings['grid_side_dimension']} x "
                 f"{settings['grid_side_dimension']} parameter grid, {output_title}")

    divider_keys = list(requested_dividers)
    ordered_rows = {}
    for solver in solvers:
        by_divider = {float(row["requested_step_count_divider"]): row
                      for row in rows if row["target_solver"] == solver}
        ordered_rows[solver] = [by_divider.get(key) for key in divider_keys]

    # Use the recorded resolution, including rounding, rather than reconstructing it from a divider.
    table_labels = ["steps/period"]
    table_values = [[str(int(float(next(row["target_steps_per_period"] for row in rows
                                      if float(row["requested_step_count_divider"]) == divider))))
                     for divider in divider_keys]]
    for field, label, fmt in table_fields:
        for solver in solvers:
            values = []
            for row in ordered_rows[solver]:
                reference_point = bool(
                    row and solver == settings["reference_solver"]
                    and int(row["target_steps_per_period"]) == int(row["reference_steps_per_period"])
                    and float(row["rms"]) == 0.0 and float(row["max_abs"]) == 0.0)
                if reference_point and field in {"rms", "max_abs"}:
                    value = "reference"
                elif not row or not np.isfinite(float(row[field])):
                    value = "--"
                else:
                    value = format(float(row[field]), fmt)
                if field == "target_calculation_time_s" and row:
                    value = f"{format_time(float(row[field]))} {'V' if row['acceptable'] == 'yes' else 'X'}"
                values.append(value)
            table_labels.append(f"{solver_label(solver)} {label}")
            table_values.append(values)

    table_bottom = 0.13 if show_best_row else 0.02
    table_margin = 0.17 if show_best_row else 0.06
    table_ax = fig.add_axes((0.15, table_bottom, 0.835,
                             max(0.01, table_fraction - table_margin)))
    table_ax.axis("off")
    table = table_ax.table(cellText=table_values, rowLabels=table_labels,
                           cellLoc="center", rowLoc="right", loc="center",
                           bbox=None)
    table.auto_set_font_size(False)
    table.set_fontsize(7.5 if len(solvers) > 6 else 8.5)
    table.scale(1.0, 1.18 if len(solvers) > 6 else 1.35)
    if not show_table:
        table.set_visible(False)
    best_rows = {}
    for solver_index, solver in enumerate(solvers):
        accepted_rows = [row for row in ordered_rows[solver]
                         if row and row["acceptable"] == "yes"
                         and np.isfinite(float(row["target_calculation_time_s"]))]
        if not accepted_rows:
            continue
        best = max(accepted_rows, key=lambda row: float(row["requested_step_count_divider"]))
        best_rows[solver] = best
        best_divider = float(best["requested_step_count_divider"])
        column = next((index for index, value in enumerate(divider_keys)
                       if np.isclose(value, best_divider)), None)
        if column is None:
            continue
        for field_index in range(len(table_fields)):
            row_index = 1 + field_index * len(solvers) + solver_index
            cell = table[row_index, column]
            cell.set_facecolor("darkolivegreen")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
    if show_best_row:
        summary_values = []
        for solver in solvers:
            best = best_rows.get(solver)
            if best is None:
                summary_values.append(f"{solver_label(solver)}\nno accepted point")
            else:
                summary_values.append(
                    f"{solver_label(solver)}\n{int(float(best['target_steps_per_period']))} | "
                    f"{format_time(float(best['target_calculation_time_s']))}")
        summary_ax = fig.add_axes((0.055, 0.015, 0.93, 0.105))
        summary_ax.axis("off")
        summary_title = (
            "Coarsest accepted grids: steps/period | time (adaptive solvers: output intervals/period); "
            f"RMS <= {float(settings['acceptable_rms']):.1e} and "
            f"maximum error <= {float(settings['acceptable_max_abs']):.1e}")
        summary_ax.text(0.5, 0.94, summary_title, ha="center", va="top",
                        fontsize=9, weight="bold")
        summary_table = summary_ax.table(
            cellText=[summary_values], cellLoc="center", bbox=(0.0, 0.0, 1.0, 0.62))
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(7.5 if len(solvers) > 6 else 8.5)
        for column in range(len(solvers)):
            cell = summary_table[0, column]
            cell.set_facecolor("darkolivegreen")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
    if show_table:
        # Draw section separators as independent lines so ordinary cell borders
        # remain thin and only the requested boundaries are emphasized.
        fig.canvas.draw()
        cells = table.get_celld()
        left = cells[0, -1].get_x()
        right_cell = cells[0, len(divider_keys) - 1]
        right = right_cell.get_x() + right_cell.get_width()
        section_boundaries = [0] + [
            field_index * len(solvers)
            for field_index in range(1, len(table_fields))]
        for row_index in section_boundaries:
            y = cells[row_index, 0].get_y()
            table_ax.plot((left, right), (y, y), color="black", linewidth=1.8,
                          transform=table_ax.transAxes, clip_on=False, zorder=10)
        label_separator = cells[0, 0].get_x()
        top = cells[0, 0].get_y() + cells[0, 0].get_height()
        bottom = cells[len(table_values) - 1, 0].get_y()
        table_ax.plot((label_separator, label_separator), (bottom, top), color="black",
                      linewidth=1.8, transform=table_ax.transAxes,
                      clip_on=False, zorder=10)
    fig.savefig(output_path, dpi=180)
    if settings["show_plot"]:
        plt.show()
    else:
        plt.close(fig)


def _find_coarsest_acceptable(rows: list[dict], settings: dict) -> list[dict]:
    summary = []
    for row in rows:
        row["coarsest_acceptable"] = "no"
    for solver in _solver_names(settings["target_solvers"]):
        accepted = [row for row in rows if row["target_solver"] == solver
                    and row["acceptable"] == "yes" and row["status"] == "measured"]
        if not accepted:
            summary.append({"target_solver": solver, "status": "no tested configuration accepted"})
            continue
        best = max(accepted, key=lambda row: float(row["requested_step_count_divider"]))
        best["coarsest_acceptable"] = "yes"
        summary.append({
            "target_solver": solver,
            "status": "accepted",
            "step_count_divider": best["actual_step_count_divider"],
            "steps_per_period": best["target_steps_per_period"],
            "total_steps": best["target_total_steps"],
            "calculation_time_s": best["target_calculation_time_s"],
            "rms": best["rms"],
            "max_abs": best["max_abs"],
        })
    return summary


def _save_optimal_dividers(rows: list[dict], settings: dict, path: Path) -> None:
    """Save the coarsest tested grid satisfying both accuracy limits."""
    solver_dividers = {}
    details = {}
    for solver in _solver_names(settings["target_solvers"]):
        accepted = [row for row in rows if row["target_solver"] == solver
                    and row["acceptable"] == "yes"
                    and np.isfinite(float(row["requested_step_count_divider"]))
                    and np.isfinite(float(row["target_calculation_time_s"]))
                    and float(row["target_calculation_time_s"]) > 0.0]
        if not accepted:
            continue
        selected = max(accepted, key=lambda row: float(row["requested_step_count_divider"]))
        divider = float(selected["requested_step_count_divider"])
        solver_dividers[solver] = divider
        details[solver] = {
            "divider": divider,
            "steps_per_period": int(selected["target_steps_per_period"]),
            "time_s": float(selected["target_calculation_time_s"]),
            "rms": float(selected["rms"]),
            "max_abs": float(selected["max_abs"]),
        }
    payload = {
        "format": "gqis_accuracy_dividers_v1",
        "problem": settings["problem"],
        "comparison_output": settings["comparison_output"],
        "selection": "largest tested divider satisfying both accuracy limits",
        "benchmark_settings": settings,
        "reference_solver": settings["reference_solver"],
        "acceptable_rms": float(settings["acceptable_rms"]),
        "acceptable_max_abs": float(settings["acceptable_max_abs"]),
        "target_base_steps_per_period": int(settings["target_base_steps_per_period"]),
        "solver_dividers": solver_dividers,
        "details": details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_settings_file(filename: str | Path) -> Path:
    requested = Path(filename).expanduser()
    script_dir = Path(__file__).resolve().parent
    candidates = ([requested] if requested.is_absolute() else
                  [Path.cwd() / requested, script_dir / "results" / requested,
                   script_dir / requested, script_dir.parent / requested])
    path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Benchmark 03 settings file not found: {filename}")
    return path


def _load_run_settings(filename: str | Path) -> dict:
    path = _resolve_settings_file(filename)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "gqis_benchmark_03_settings_v1":
        raise ValueError(f"Unsupported Benchmark 03 settings format: {path}")
    overrides = payload.get("settings")
    if not isinstance(overrides, dict):
        raise ValueError(f"Benchmark 03 settings are missing from: {path}")
    settings = user_settings()
    unknown = set(overrides).difference(settings)
    if unknown:
        raise ValueError(f"Unknown Benchmark 03 settings: {', '.join(sorted(unknown))}")
    for key, value in overrides.items():
        settings[key] = ({**settings[key], **value}
                         if isinstance(settings[key], dict) and isinstance(value, dict)
                         else value)
    print(f"Loaded Benchmark 03 settings: {path}")
    return settings


def _save_run_settings(settings: dict, path: Path) -> None:
    payload = {"format": "gqis_benchmark_03_settings_v1", "settings": settings}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               default=lambda value: (value.item() if isinstance(value, np.generic)
                                                      else str(value))) + "\n",
                    encoding="utf-8")
    print(f"Saved settings:  {path}")


def run_accuracy_timestep_sweep(settings: dict | None = None) -> list[dict]:
    settings = dict(user_settings() if settings is None else settings)
    settings["problem"] = str(settings["problem"]).lower()
    settings["four_level_regime"] = str(settings["four_level_regime"]).lower()
    settings["comparison_output"] = str(settings["comparison_output"]).lower()
    settings["divider_axis_scale"], settings["divider_axis_labels"] = \
        _divider_axis_options(settings)
    reference_solver = str(settings["reference_solver"])
    target_solvers = _solver_names(settings["target_solvers"])
    dividers = tuple(float(value) for value in settings["step_count_dividers"])
    if reference_solver not in ACCURACY_SOLVER_SET:
        raise ValueError(f"Unknown reference solver: {reference_solver}")
    if not dividers or dividers[0] != 1.0 or any(value <= 0.0 for value in dividers):
        raise ValueError("step_count_dividers must start with 1 and contain positive values.")

    output_name = str(settings["output_stem"])
    if settings["problem"] == "four_level":
        output_name = re.sub("two_level", "four_level", output_name, flags=re.IGNORECASE)
        if "four_level" not in output_name.lower():
            output_name += "_four_level"
    if settings["comparison_output"] == "final" and not output_name.lower().endswith("_final"):
        output_name += "_final"
    script_dir = Path(__file__).resolve().parent
    output_stem = benchmark_output_path(output_name, script_dir=script_dir)
    metrics_path = output_stem.with_name(f"{output_stem.name}_metrics.csv")
    figure_path = output_stem.with_name(f"{output_stem.name}_metrics.png")
    settings_path = output_stem.with_name(f"{output_stem.name}_settings.json")
    if settings.get("save_settings_manifest", True):
        _save_run_settings(settings, settings_path)
    divider_file = settings["optimal_dividers_file"]
    divider_path = (benchmark_output_path(divider_file, script_dir=script_dir) if divider_file else
                    output_stem.with_name(f"{output_stem.name}_optimal_dividers.json"))
    reference_file = settings["reference_file"]
    reference_path = (benchmark_output_path(reference_file, script_dir=script_dir)
                      if reference_file else
                      output_stem.with_name(f"{output_stem.name}_reference.npz"))

    reference_cfg = _accuracy_config(
        settings, int(settings["reference_steps_per_period"]), reference=True)
    target_base_cfg = _accuracy_config(
        settings, int(settings["target_base_steps_per_period"]), reference=False)
    selected_gqis = tuple(dict.fromkeys(
        solver for solver in (reference_solver, *target_solvers) if is_gqis_solver(solver)))
    gqis_preparation_time = np.nan
    for index, solver in enumerate(selected_gqis):
        preparation_time = _warm_gqis(reference_cfg, solver, first=index == 0)
        if index == 0:
            gqis_preparation_time = preparation_time

    print_equipment_info()
    print(f"Automated {settings['problem'].replace('_', '-') } time-grid accuracy sweep")
    if settings["problem"] == "four_level":
        print(f"Benchmark 02 regime: {settings['four_level_regime']}")
    print(f"reference={reference_solver}, targets={','.join(target_solvers)}, "
          f"grid={reference_cfg.nx}x{reference_cfg.ny}")
    print(f"Comparison output: {reference_cfg.comparison_output}")
    displayed_solvers = tuple(dict.fromkeys((reference_solver, *target_solvers)))
    print("methods: " + "; ".join(
        f"{solver}={SOLVER_METHODS[solver]}" for solver in displayed_solvers))
    if "qutip_cpu" in {reference_solver, *target_solvers}:
        print(f"QuTiP drive mode: {settings['qutip_drive_mode']}")
    selected_julia = tuple(s for s in displayed_solvers if is_julia_solver(s))
    if selected_julia:
        print("Julia time modes: " + ", ".join(
            f"{solver}={JULIA_SOLVER_MODES[solver]}" for solver in selected_julia))
    if selected_gqis:
        unroll_mode = "forced" if target_base_cfg.gqis_unroll else "NVRTC-managed"
        print(f"GQIS solver loop unrolling: {unroll_mode}")
    print(f"reference_steps_per_period={reference_cfg.solver_steps_per_period}, "
          f"target_base_steps_per_period={target_base_cfg.solver_steps_per_period}")
    print(f"step-count dividers={dividers}")
    qutip_single_point = bool(settings.get("qutip_largest_divider_only", False))
    if qutip_single_point and "qutip_cpu" in target_solvers:
        print(f"QuTiP target: largest divider only ({max(dividers):g})")

    metadata = _reference_metadata(settings, reference_cfg)
    if settings["load_reference_map"]:
        reference_result, reference_time = _load_reference(
            reference_path, metadata, (reference_cfg.ny, reference_cfg.nx))
        print(f"\nLoaded saved reference: {reference_path}")
    else:
        start_message = f"\nCalculating reference with {solver_label(reference_solver)}..."
        if reference_solver != "qutip_cpu":
            print(start_message)
            start_message = None
        reference_result, reference_time = _run_accuracy_solver(
            reference_solver, reference_cfg, settings, start_message=start_message)
        if not np.all(np.isfinite(reference_result)):
            raise FloatingPointError("Reference solver returned non-finite values.")
        if settings["save_reference_map"]:
            _save_reference(reference_path, reference_result, reference_time, metadata)
            print(f"Saved reference: {reference_path}")
    reference_range = float(np.ptp(np.asarray(reference_result, dtype=np.float64)))
    print(f"Reference calculation time: {reference_time:.6g}s; range={reference_range:.6g}")

    rows: list[dict] = []
    saved_rows = 0
    baseline_results: dict[str, np.ndarray] = {}
    baseline_times: dict[str, float] = {}

    for solver in target_solvers:
        if rows and (is_julia_solver(solver) or solver == "qutip_cpu"):
            print(f"\nCompleted results are saved and ready to plot: {metrics_path}", flush=True)
        if solver == "qutip_cpu" and target_base_cfg.nx >= 1024:
            print("QuTiP may take more than 12 hours at full CPU load on this grid. "
                  "Earlier solver results remain available in the CSV.", flush=True)
        print(f"\nTarget {solver_label(solver)}")
        solver_dividers = ((max(dividers),) if solver == "qutip_cpu"
                           and qutip_single_point else dividers)
        for requested_divider in solver_dividers:
            target_cfg = _reduced_time_grid(target_base_cfg, requested_divider)
            actual_divider = target_base_cfg.num_steps / target_cfg.num_steps
            try:
                result, calculation_time = _run_accuracy_solver(solver, target_cfg, settings)
                julia_total_time = LAST_JULIA_TOTAL_S if is_julia_solver(solver) else np.nan
                julia_precalc_time = LAST_JULIA_PRECALC_S if is_julia_solver(solver) else np.nan
                finite_result = bool(np.all(np.isfinite(result)))
                if not finite_result and not target_cfg.ignore_non_finite_output:
                    raise FloatingPointError("solver returned non-finite values")
                if (finite_result and np.isclose(requested_divider, 1.0)
                        and solver not in baseline_results):
                    baseline_results[solver] = np.asarray(result)
                    baseline_times[solver] = float(calculation_time)
                metrics = _error_metrics(result, reference_result, reference_range)
                if solver in baseline_results:
                    own_metrics = _error_metrics(
                        result, baseline_results[solver],
                        float(np.ptp(np.asarray(baseline_results[solver], dtype=np.float64))))
                else:
                    own_metrics = {"rms": np.nan, "max_abs": np.nan, "p99_abs": np.nan}
                accepted = (metrics["rms"] <= float(settings["acceptable_rms"])
                            and metrics["max_abs"] <= float(settings["acceptable_max_abs"]))
                status = "measured" if finite_result else "non-finite"
            except Exception as exc:
                calculation_time = np.nan
                julia_total_time = np.nan
                julia_precalc_time = np.nan
                metrics = {name: np.nan for name in
                           ("mse", "rms", "normalized_rms", "max_abs", "p95_abs", "p99_abs")}
                own_metrics = {"rms": np.nan, "max_abs": np.nan, "p99_abs": np.nan}
                accepted = False
                status = f"failed: {type(exc).__name__}: {exc}"

            row = {
                "relaxation_operator": (
                    "basis_1_to_basis_0" if settings["problem"] == "two_level"
                    else "benchmark_02_qubit_resonator_collapse_set"),
                "problem": settings["problem"],
                "comparison_output": target_cfg.comparison_output,
                "acceptable_rms_limit": float(settings["acceptable_rms"]),
                "acceptable_max_abs_limit": float(settings["acceptable_max_abs"]),
                "gqis_preparation_time_s": gqis_preparation_time,
                "reference_solver": reference_solver,
                "reference_numerical_method": SOLVER_METHODS[reference_solver],
                "target_solver": solver,
                "target_numerical_method": SOLVER_METHODS[solver],
                "gqis_solver": gqis_method(solver, target_cfg) if is_gqis_solver(solver) else "",
                "gqis_solver_frequency": (
                    target_cfg.w if is_gqis_solver(solver)
                    and gqis_method(solver, target_cfg).lower() == "anas5" else ""),
                "gqis_unroll": target_cfg.gqis_unroll if is_gqis_solver(solver) else "",
                "qutip_drive_mode": settings["qutip_drive_mode"],
                "julia_time_precision": (
                    JULIA_SOLVER_MODES[solver] if is_julia_solver(solver) else ""),
                "grid_side_dimension": reference_cfg.nx,
                "requested_step_count_divider": requested_divider,
                "actual_step_count_divider": actual_divider,
                "reference_steps_per_period": reference_cfg.solver_steps_per_period,
                "reference_total_steps": reference_cfg.num_steps,
                "target_steps_per_period": target_cfg.solver_steps_per_period,
                "target_total_steps": target_cfg.num_steps,
                "simulation_periods": target_cfg.tr,
                "target_dt": target_cfg.dt,
                "reference_calculation_time_s": reference_time,
                "target_calculation_time_s": calculation_time,
                "julia_total_time_s": julia_total_time,
                "julia_gpu_calculation_time_s": (
                    calculation_time if is_julia_solver(solver) else np.nan),
                "julia_precalculation_time_s": julia_precalc_time,
                "target_speedup_vs_divider_1": (
                    baseline_times.get(solver, np.nan) / calculation_time
                    if np.isfinite(calculation_time) else np.nan),
                "target_time_over_reference": (
                    calculation_time / reference_time if np.isfinite(calculation_time) else np.nan),
                "mse": metrics["mse"],
                "rms": metrics["rms"],
                "normalized_rms": metrics["normalized_rms"],
                "max_abs": metrics["max_abs"],
                "p95_abs": metrics["p95_abs"],
                "p99_abs": metrics["p99_abs"],
                "rms_vs_target_divider_1": own_metrics["rms"],
                "max_abs_vs_target_divider_1": own_metrics["max_abs"],
                "p99_abs_vs_target_divider_1": own_metrics["p99_abs"],
                "acceptable": "yes" if accepted else "no",
                "status": status,
            }
            rows.append(row)
            if is_julia_solver(solver) or solver == "qutip_cpu":
                _write_accuracy_rows(rows[saved_rows:], metrics_path, append=saved_rows > 0)
                saved_rows = len(rows)
            time_label = "GPU_time" if is_gqis_solver(solver) or is_julia_solver(solver) else "time"
            calculation_text = f"{time_label}={calculation_time:>9.6g}s"
            total_text = (f"tot_time={julia_total_time:>9.6g}s"
                          if is_julia_solver(solver) else "")
            status_text = f"  status={status}" if status != "measured" else ""
            print(f"divider={requested_divider:>2g}  "
                  f"steps/period={target_cfg.solver_steps_per_period:>3d}  "
                  f"steps={target_cfg.num_steps:>6d}  {calculation_text:<22}  "
                  f"{total_text:<22}  RMS={metrics['rms']:>10.3e}  "
                  f"P99={metrics['p99_abs']:>10.3e}  max={metrics['max_abs']:>10.3e}  "
                  f"accept={row['acceptable']:>3}{status_text}")

        if len(rows) > saved_rows:
            _write_accuracy_rows(rows[saved_rows:], metrics_path, append=saved_rows > 0)
            saved_rows = len(rows)
        if settings["save_optimal_dividers"]:
            _save_optimal_dividers(rows, settings, divider_path)

    summary = _find_coarsest_acceptable(rows, settings)
    _plot_accuracy_sweep(rows, settings, reference_time, gqis_preparation_time, figure_path)
    print("\nFastest acceptable tested configurations:")
    for item in summary:
        print(item)
    print(f"\nSaved metrics:   {metrics_path}")
    if settings["save_optimal_dividers"]:
        print(f"Saved dividers:  {divider_path}")
    print(f"Saved figure:    {figure_path}")
    return rows


# =============================================================================
# USER SETTINGS -- normal benchmark changes should be made below this line
# =============================================================================

def user_settings() -> dict:
    """Return the editable model, solver, accuracy, grid, and output settings."""
    return {
        # Built-in problems: "two_level" or "four_level". To add a model, copy
        # a builder in the problem-definition section and add its dispatch branch.
        "problem": "four_level",
        "four_level_regime": "wd500",  # "wd500" or "wd1500"

        # Exactly one reference and any number of targets may be selected.
        # GQIS general: gqis_rk4, gqis_lserk4, gqis_dp5, gqis_tsit5,
        #               gqis_alshina6, gqis_dop853
        # GQIS specialized: gqis_anas5 (periodic), gqis_ab5 (multistep)
        # Julia: julia_gpu_fp64 (all FP64), julia_gpu_fp32 (all FP32),
        #        julia_gpu_fp32_opt (FP64 clock, FP32 calculation),
        #        julia_gpu_fp32_fopt (FP32 time reconstructed from integer step)
        #       all julia_gpu solvers are fixed step
        # CPU: qutip_cpu (Adams adaptive time step), python_ode_cpu (SciPy RK45 adaptive), python_cpu (RK4)
        "reference_solver": "gqis_rk4",
        #"target_solvers": ("gqis_rk4", "gqis_dp5",
        #                   "gqis_ab5", "gqis_tsit5",
        #                    "gqis_dop853","julia_gpu_fp32_fopt","julia_gpu_fp32", "qutip_cpu"),
        "target_solvers": (
            "gqis_rk4",             # general-purpose baseline and current speed winner
            "gqis_ab5",             # multistep: one derivative evaluation per step after startup
            "gqis_anas5",           # frequency-fitted method for periodic driving
            "gqis_tsit5",           # general fifth-order method; direct comparison with Julia Tsit5
            "gqis_dop853",          # high-order method for strict accuracy limits
            "julia_gpu_fp32_fopt",  # experimental integer-step-time Julia implementation
            "julia_gpu_fp32",       # stock FP32 Julia behavior
            #"qutip_cpu",            # adaptive CPU timing point
                   ),
        "GQIS_unroll": False,  # True forces solver-loop unrolling
        # Square parameter grid and fixed-step sweep. Each divider reduces the
        # target step count; the reference always uses its independent density.
        "grid_side_dimension": 2048,
        "reference_steps_per_period": 4096,
        "target_base_steps_per_period": 2048,
        "step_count_dividers": (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32,48,64),
        "simulation_periods": 40.0,

        # Adaptive settings affect QuTiP/SciPy only. A target is accepted only
        # when both absolute-error limits are satisfied.
        "reference_rtol": 1e-10, "reference_atol": 1e-12,
        "target_rtol": 1e-7, "target_atol": 1e-9,
        "adaptive_nsteps": 100_000,
        "acceptable_rms": 1e-3, "acceptable_max_abs": 1e-2,
        "qutip_drive_mode": "analytic",  # "analytic" or "interpolated"
        # True runs a target QuTiP calculation only at the largest divider. This
        # keeps one meaningful CPU timing point on expensive high-resolution grids.
        "qutip_largest_divider_only": True,
        # "mean" compares the interferogram's time-averaged observable;
        # "final" compares the same observable at the last calculated state.
        "comparison_output": "mean",

        # Reference cache. None selects <output_stem>_reference.npz. Metadata is
        # checked before reuse, preventing a different model/grid being loaded.
        "load_reference_map": False, "save_reference_map": False,
        "reference_file": None,

        # Save the coarsest tested grid that passes both limits for each
        # solver. None writes <output_stem>_optimal_dividers.json.
        "save_optimal_dividers": True,
        "optimal_dividers_file": None,

        # Output and diagnostics.
        "ignore_non_finite_output": True,  # retain unstable target points
        "show_worker_progress": True,  # one updating line for CPU columns
        "julia_cmd": "julia",  # executable name or full path
        "show_plot": True,
        "show_results_table": True,  # False makes a compact graph; all values remain in CSV
        "table_times_only": False,  # True omits the RMS and maximum error rows
        "show_best_summary_row": False,  # compact dark-olive steps/period | time summary
        # Save a complete JSON settings manifest beside the CSV and figure. It
        # can later be passed back with --settings to reproduce this run.
        "save_settings_manifest": True,
        # Horizontal axes on the accuracy and timing panels.
        "time_grid_axis": "steps_per_period",  # choices: "steps_per_period" or "divider"
        "divider_axis_scale": "log2",  # choices: "log2" or "equidistant"; applies to either quantity
        "divider_axis_labels": "powers_of_two",  # choices: "powers_of_two" or "all"
        # Relative output names are saved in Benchmarks/results/.
        "output_stem": "Benchmark_03_two_level_accuracy_timestep_sweep",
        
        # Two-level demonstration model. Rates are specified per drive period.
        "two_level_parameters": {
            "delta": 1.0, "w_over_delta": 1.14,
            "gamma_phi_per_period": 0.04, "gamma1_per_period": 0.05,
            "eps_max_over_w": 10.0, "amplitude_max_over_w": 10.0,
        },
        # Four-level Benchmark-02 model. Sweep limits and coupling/probe values
        # are normalized by the qubit frequency where stated in their names.
        "four_level_parameters": {
            "qubit_frequency": 4.71 * 1.15, "resonator_frequency": 7.6767,
            "gamma_phi": 0.0, "gamma1": 2.0e-3, "kappa": 5.0e-3,
            "probe_amplitude": 0.0002 / 10.0,
            "coupling_over_qubit_frequency": 0.04033970276 / 1.23,
            "photon_levels": 2, "eps_max_over_qubit_frequency": 2.09916,
            "amplitude_max_over_qubit_frequency": 2.234042553191489,
            "regimes": {
                "wd500": {"drive_frequency": 0.5, "amplitude_factor": 1.15},
                "wd1500": {"drive_frequency": 1.5, "amplitude_factor": 1.15 * 1.3},
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", help="JSON settings manifest from an earlier run")
    parser.add_argument("--save-settings-only", metavar="FILE",
                        help="save the selected settings as JSON and exit without calculating")
    args = parser.parse_args()
    settings = _load_run_settings(args.settings) if args.settings else user_settings()
    if args.save_settings_only:
        path = benchmark_output_path(
            args.save_settings_only, script_dir=Path(__file__).resolve().parent)
        _save_run_settings(settings, path)
        return
    run_accuracy_timestep_sweep(settings)


if __name__ == "__main__":
    main()

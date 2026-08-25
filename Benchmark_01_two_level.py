"""Benchmark 01: two-level interferogram timing/accuracy comparison.

Backends
--------
  gpu        : GQIS mesolve_2D
  python_cpu : pure-Python fixed-step RK4 master-equation solver
  python_ode_cpu : SciPy solve_ivp adaptive CPU solver
  qutip_cpu  : QuTiP mesolve
  julia_gpu  : Julia DiffEqGPU helper generated from the shared GQIS SymPy RHS
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from Benchmark_full_tools import (add_time_reference_marks, benchmark_sides, collect_equipment_info,
                                  extrapolate_loglog, format_equipment_label, parse_solver_list,
                                  print_equipment_info, save_benchmark_csv,
                                  should_extrapolate_next, sympy_to_julia_fp32,
                                  terminate_process_tree,
                                  )
from gqis import build_reduced_lindblad_rhs, mesolve_2D

SOLVERS = ("gpu", "python_cpu", "python_ode_cpu", "qutip_cpu", "julia_gpu")
SOLVER_SET = set(SOLVERS)
JULIA_HELPER_NAME = "Benchmark_01_two_level_basic_julia_gpu.jl"
LAST_GPU_RHS_STAGE_S = np.nan
BENCH_EXTRAPOLATION_POINTS = 2

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


@dataclass(frozen=True)
class PhysicsParams:
    delta: float = 1.0
    w_over_delta: float = 1.14
    gamma_phi_per_T: float = 0.04
    gamma1_per_T: float = 0.05

    def derived(self) -> tuple[float, float, float, float, float]:
        delta = float(self.delta)
        w = float(self.w_over_delta * delta)
        T = float(2.0 * np.pi / w)
        gamma_phi = float(self.gamma_phi_per_T / T)
        gamma1 = float(self.gamma1_per_T / T)
        gamma2 = float(gamma1 / 2.0 + gamma_phi)
        return delta, w, gamma1, gamma2, T


def make_config(args: argparse.Namespace, *, nx: int | None = None,
                ny: int | None = None) -> BenchConfig:
    physics = PhysicsParams(delta=args.delta, w_over_delta=args.w_over_delta,
                            gamma_phi_per_T=args.gamma_phi_per_T,
                            gamma1_per_T=args.gamma1_per_T)
    delta, w, gamma1, gamma2, T = physics.derived()

    nx = int(args.nx if nx is None else nx)
    ny = int((args.ny if args.ny is not None else nx) if ny is None else ny)
    solver_steps_per_period = int(args.solver_steps_per_period)
    num_steps = max(1, int(round(args.tr * solver_steps_per_period)))

    eps_max = float(args.eps_max_factor * w)
    A_max = float(args.A_max_factor * w)

    scalar_dtype = np.float64 if args.gpu_precision == "fp64" else np.float32
    eps_list = np.linspace(-eps_max, eps_max, nx, dtype=scalar_dtype)
    A_list = np.linspace(0.0, A_max, ny, dtype=scalar_dtype)
    tlist = np.linspace(0.0, args.tr * T, num_steps + 1, dtype=scalar_dtype)

    workers = int(args.workers) if args.workers is not None else max(1, (os.cpu_count() or 2) - 1)

    return BenchConfig(delta=delta, w=w, gamma1=gamma1, gamma2=gamma2, tr=float(args.tr),
                       solver_steps_per_period=solver_steps_per_period,
                       gpu_precision=args.gpu_precision,
                       cpu_precision=args.cpu_precision, eps_list=eps_list, A_list=A_list,
                       tlist=tlist, warmup_time=float(args.warmup_time), workers=max(1, workers),
                       timings=bool(args.timings), progress=not bool(args.no_progress))


def config_for_solver(solver: str, cfg: BenchConfig, args: argparse.Namespace) -> BenchConfig:
    """Return the actual config used by one solver.

    CPU backends can optionally use fewer time samples than the GPU backend.  This
    is intentionally centralized here so solver calls remain uniform everywhere.
    """
    if solver not in {"python_cpu", "python_ode_cpu", "qutip_cpu"}:
        return cfg

    if solver == "python_cpu":
        divider = int(args.python_cpu_spp_divider)
    elif solver == "python_ode_cpu":
        divider = int(args.python_ode_cpu_spp_divider)
    else:
        divider = int(args.qutip_cpu_spp_divider)
    if divider <= 1:
        return cfg

    cpu_spp = max(1, int(round(cfg.solver_steps_per_period / divider)))
    if cpu_spp == cfg.solver_steps_per_period:
        return cfg

    T = 2.0 * np.pi / cfg.w
    cpu_num_steps = max(1, int(round(cfg.tr * cpu_spp)))
    cpu_tlist = np.linspace(float(cfg.tlist[0]), cfg.tr * T, cpu_num_steps + 1,
                            dtype=cfg.tlist.dtype)
    return replace(cfg, solver_steps_per_period=cpu_spp, tlist=cpu_tlist)


# -----------------------------------------------------------------------------
# Shared model and small utilities
# -----------------------------------------------------------------------------


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


def print_config(cfg: BenchConfig) -> None:
    total_steps = cfg.nx * cfg.ny * cfg.num_steps
    print(f"grid={cfg.nx}x{cfg.ny} steps={cfg.num_steps} dt={cfg.dt:.4e} "
          f"solver_steps_per_period={cfg.solver_steps_per_period} "
          f"simulation_periods={cfg.tr:.3f} workers={cfg.workers}")
    print(f"precisions: gpu={cfg.gpu_precision} cpu={cfg.cpu_precision}")
    print(f"total trajectory-step updates = {cfg.nx}*{cfg.ny}*{cfg.num_steps} = {total_steps:.4e}")


def print_progress(label: str, done: int, total: int) -> None:
    if total <= 0:
        return
    pct = 100.0 * done / total
    print(f"\r{label}: {done}/{total} columns ({pct:5.1f}%)", end="", flush=True)
    if done >= total:
        print()


def save_timings_txt(timings: dict[str, float], cfg: BenchConfig, mode: str,
                     order: tuple[str, ...]) -> Path:
    path = Path(f"Benchmark_01_two_level_basic_timings_nx{cfg.nx}_{mode}.txt")
    with path.open("w", encoding="ascii", newline="\n") as f:
        for name in order:
            if name in timings:
                f.write(f"{name}\t{timings[name]:.6f}\n")
    print(f"Saved timings: {path}")
    return path


def plot_full_benchmark(rows: list[dict], solvers: tuple[str, ...], out_png: Path, *, title: str,
                        show: bool, metadata: dict[str, str] | None = None,
                        reference_lines: list[dict] | None = None) -> None:
    """Plot measured and extrapolated benchmark points.

    The extrapolated curve is drawn as one dashed continuation from the last
    measured point.  The last measured point is included in the dashed x/y data,
    but it is not marked as an extrapolated square.
    """
    if not rows:
        return

    side_union = sorted({int(row["side_dimension"]) for row in rows})
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.3], hspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Side dimension")
    ax.set_ylabel("Time [s]")
    label = format_equipment_label(metadata)
    ax.set_title(f"{title}\n{label}" if label else title)
    ax.grid(True, which="major", alpha=0.5)

    table_by_solver: dict[str, dict[int, str]] = {}

    for solver in solvers:
        solver_rows = [row for row in rows if row["solver"] == solver]
        if not solver_rows:
            continue
        solver_rows.sort(key=lambda row: int(row["side_dimension"]))
        table_by_solver[solver] = {
            int(row["side_dimension"]): (f"{float(row['time_s']):.3g}"
                                         if np.isfinite(float(row["time_s"])) else "nan")
            for row in solver_rows
        }

        measured = [(int(row["side_dimension"]), float(row["time_s"])) for row in solver_rows
                    if row["status"] == "measured" and np.isfinite(float(row["time_s"]))]
        extrapolated = [(int(row["side_dimension"]), float(row["time_s"])) for row in solver_rows
                        if row["status"] == "extrapolated" and np.isfinite(float(row["time_s"]))]

        color = None
        if measured:
            mx, my = zip(*measured)
            line = ax.plot(mx, my, linestyle="-", marker="o", label=solver)[0]
            color = line.get_color()

        if extrapolated:
            ex, ey = zip(*extrapolated)
            if measured:
                last_measured_x, last_measured_y = measured[-1]
                dashed_x = (last_measured_x, ) + ex
                dashed_y = (last_measured_y, ) + ey
                ax.plot(dashed_x, dashed_y, linestyle="--", color=color, label="_nolegend_")
                ax.plot(ex, ey, linestyle="None", marker="s", color=color, label="_nolegend_")
            else:
                ax.plot(ex, ey, linestyle="--", marker="s", label=solver)

    for ref in reference_lines or []:
        y = float(ref.get("y", np.nan))
        if not np.isfinite(y) or y <= 0.0:
            continue
        ax.axhline(y, color=ref.get("color", "0.25"), linestyle=ref.get("linestyle", ":"),
                   linewidth=float(ref.get("linewidth",
                                           1.6)), label=str(ref.get("label", "reference")))

    add_time_reference_marks(ax)

    ax.set_xticks(side_union)
    ax.set_xticklabels([str(side) for side in side_union])
    ax.legend(loc="upper center", ncol=max(1, len(solvers)))

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    col_labels = ["Simulations"] + [f"{side * side:.2E}" for side in side_union]
    cell_text = []
    for solver in solvers:
        if solver in table_by_solver:
            cell_text.append([solver] +
                             [table_by_solver[solver].get(side, "") for side in side_union])
    if cell_text:
        table = ax_tbl.table(cellText=cell_text, colLabels=col_labels, loc="center",
                             cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.11)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_maps(panels: list[tuple[str, np.ndarray, str]], cfg: BenchConfig, *, layout: str,
              suptitle: str | None = None) -> None:
    if not panels:
        return

    if layout == "panel":
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), sharex=True,
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (title, data, cmap) in zip(axes, panels):
            add_map_to_axis(ax, data, cfg, title, cmap)
        if suptitle:
            fig.suptitle(suptitle)
        plt.tight_layout()
        plt.show()
        return

    for title, data, cmap in panels:
        fig, ax = plt.subplots(figsize=(8, 8))
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        add_map_to_axis(ax, data, cfg, title, cmap)
        plt.tight_layout()
    plt.show()


def add_map_to_axis(ax: plt.Axes, data: np.ndarray, cfg: BenchConfig, title: str,
                    cmap: str) -> None:
    fig = ax.figure
    im = ax.imshow(data, aspect="auto", cmap=cmap, origin="lower",
                   extent=[cfg.eps_list[0], cfg.eps_list[-1], cfg.A_list[0], cfg.A_list[-1]])
    fig.colorbar(im, ax=ax, label="Average excited population")
    ax.set_xlabel("eps")
    ax.set_ylabel("A")
    ax.set_title(title)


# -----------------------------------------------------------------------------
# GPU backend
# -----------------------------------------------------------------------------


def run_gpu_solver(cfg: BenchConfig) -> tuple[np.ndarray, float]:
    global LAST_GPU_RHS_STAGE_S
    start = time.time()
    fp64 = cfg.gpu_precision == "fp64"
    scalar_dtype = np.float64 if fp64 else np.float32
    _, H, drive_expr, col_ops, mean_op, const_values, (eps, A) = two_level_sympy_model(cfg)

    result, timing_info = mesolve_2D(H, drive_expr, col_ops, mean_op,
                                     np.asarray(cfg.tlist, dtype=scalar_dtype),
                                     var_arrays={eps: np.asarray(cfg.eps_list, dtype=scalar_dtype),
                                                 A: np.asarray(cfg.A_list, dtype=scalar_dtype)},
                                     const_values=const_values, output_mode="mean", fp64=fp64,
                                     timings=cfg.timings, warmup_time=cfg.warmup_time,
                                     return_timing_info=True)
    LAST_GPU_RHS_STAGE_S = float((timing_info or {}).get("rhs_stage_s", np.nan))

    # mesolve_2D returns (eps, A) for this var_arrays order; plotting expects (A, eps).
    p_mat = np.abs(np.real(np.asarray(result))).T.astype(scalar_dtype, copy=False)
    return p_mat, time.time() - start


# -----------------------------------------------------------------------------
# Python and QuTiP CPU backends
# -----------------------------------------------------------------------------

_WORKER_CFG: BenchConfig | None = None
_WORKER_SOLVER: str | None = None


def init_worker(cfg: BenchConfig, solver: str) -> None:
    global _WORKER_CFG, _WORKER_SOLVER
    _WORKER_CFG = cfg
    _WORKER_SOLVER = solver


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
    complex_dtype = np.complex64 if cfg.cpu_precision == "fp32" else np.complex128

    dt = scalar_dtype(cfg.dt)
    dt2 = scalar_dtype(0.5) * dt
    dt6 = dt / scalar_dtype(6.0)
    warmup = cfg.warmup_steps
    denom = scalar_dtype(max(cfg.num_steps - warmup, 1))
    t0 = scalar_dtype(cfg.tlist[0]) if cfg.num_t else scalar_dtype(0.0)

    rho = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=complex_dtype)
    accum = scalar_dtype(0.0)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(cfg.num_steps):
            t = t0 + scalar_dtype(step) * dt
            k1 = drho_two_level(rho, t, A, eps0, cfg, scalar_dtype, complex_dtype)
            k2 = drho_two_level(rho + dt2 * k1, t + dt2, A, eps0, cfg, scalar_dtype, complex_dtype)
            k3 = drho_two_level(rho + dt2 * k2, t + dt2, A, eps0, cfg, scalar_dtype, complex_dtype)
            k4 = drho_two_level(rho + dt * k3, t + dt, A, eps0, cfg, scalar_dtype, complex_dtype)
            rho = rho + dt6 * (k1 + scalar_dtype(2.0) * k2 + scalar_dtype(2.0) * k3 + k4)
            if not np.all(np.isfinite(rho)):
                return np.nan
            if step >= warmup:
                accum = accum + scalar_dtype(np.real(rho[3]))
                if not np.isfinite(accum):
                    return np.nan

    return float(accum / denom)


def solve_one_point_python_ode(A: float, eps0: float, cfg: BenchConfig) -> float:
    from scipy.integrate import solve_ivp

    scalar_dtype = np.float64
    complex_dtype = np.complex128
    tlist = np.asarray(cfg.tlist, dtype=np.float64)
    if len(tlist) < 2:
        return np.nan

    rho0 = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=complex_dtype)

    def rhs(t, rho):
        return drho_two_level(rho, t, A, eps0, cfg, scalar_dtype, complex_dtype)

    sol = solve_ivp(rhs, (float(tlist[0]), float(tlist[-1])), rho0, method="RK45", t_eval=tlist,
                    rtol=1e-7, atol=1e-9)
    if not sol.success or sol.y.shape[1] == 0:
        return np.nan
    p = np.real(sol.y[3]).astype(np.float64, copy=False)
    warmup = cfg.warmup_steps
    start = min(warmup + 1, len(p) - 1)
    return float(np.mean(p[start:]))


def solve_one_point_qutip(A: float, eps0: float, cfg: BenchConfig) -> float:
    import qutip as qt

    sx = qt.sigmax()
    sz = qt.sigmaz()
    sm = qt.sigmam()
    proj_exc = qt.basis(2, 1) * qt.basis(2, 1).dag()
    rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()

    tlist = np.asarray(cfg.tlist, dtype=float)
    H0 = 0.5 * cfg.delta * sx + 0.5 * eps0 * sz
    H1 = 0.5 * A * sz
    H = [H0, [H1, np.sin(cfg.w * tlist)]]

    c_ops = []
    if cfg.gamma1 > 0:
        c_ops.append(np.sqrt(cfg.gamma1) * sm)
    if cfg.gamma2 > 0:
        c_ops.append(np.sqrt(cfg.gamma2) * sz)

    res = qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[proj_exc])
    p = np.asarray(np.real(res.expect[0]), dtype=np.float64)
    warmup = cfg.warmup_steps
    start = min(warmup + 1, len(p) - 1)
    return float(np.mean(p[start:]))


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


def run_cpu_solver(solver: str, cfg: BenchConfig) -> tuple[np.ndarray, float]:
    start = time.time()
    dtype = np.float32 if (solver == "python_cpu" and cfg.cpu_precision == "fp32") else np.float64
    out = np.empty((cfg.ny, cfg.nx), dtype=dtype)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=cfg.workers, initializer=init_worker, initargs=(cfg, solver)) as pool:
        done = 0
        update_every = max(1, cfg.nx // 50)
        for eps_index, col in pool.imap_unordered(solve_cpu_column, range(cfg.nx)):
            out[:, eps_index] = col
            done += 1
            if cfg.progress and (done == 1 or done == cfg.nx or done % update_every == 0):
                print_progress(solver, done, cfg.nx)

    return out, time.time() - start


def run_python_cpu_solver(cfg: BenchConfig) -> tuple[np.ndarray, float]:
    return run_cpu_solver("python_cpu", cfg)


def run_python_ode_cpu_solver(cfg: BenchConfig) -> tuple[np.ndarray, float]:
    return run_cpu_solver("python_ode_cpu", cfg)


def run_qutip_cpu_solver(cfg: BenchConfig) -> tuple[np.ndarray, float]:
    return run_cpu_solver("qutip_cpu", cfg)


# -----------------------------------------------------------------------------
# Julia GPU backend
# -----------------------------------------------------------------------------


def write_julia_helper(path: Path, cfg: BenchConfig) -> None:
    N, H, drive_expr, col_ops, mean_op, const_values, _ = two_level_sympy_model(cfg)

    Delta, eps, Drive = sp.symbols("Delta eps Drive", real=True)
    A, t = sp.symbols("A t", real=True)
    gamma1S, gamma2S = sp.symbols("gamma1S gamma2S", real=True, nonnegative=True)

    H_sub = H.subs(Drive, drive_expr).subs(const_values)
    col_ops_sub = [L.subs(const_values) for L in col_ops]
    drho_eqs, obs_expr, _, meta = build_reduced_lindblad_rhs(
        N, H_sub, col_ops_sub, mean_op)
    rho_syms = meta["rho_syms"]

    u_syms = [sp.Symbol(f"u{i}") for i in range(len(rho_syms))]
    repl = {rho_syms[i]: u_syms[i] for i in range(len(rho_syms))}
    drho_local = [sp.simplify(e.subs(repl)) for e in drho_eqs]
    obs_local = sp.simplify(obs_expr.subs(repl))
    common, reduced = sp.cse([*drho_local, obs_local], symbols=sp.numbered_symbols("tmp"),
                             optimizations="basic")

    rhs_lines = [f"u{i} = u[{i + 1}]" for i in range(len(rho_syms))]
    rhs_lines += [f"{symbol} = Float32({sympy_to_julia_fp32(expr)})"
                  for symbol, expr in common]
    rhs_lines += [f"du{i + 1} = Float32({sympy_to_julia_fp32(expr)})"
                  for i, expr in enumerate(reduced[:-1])]
    rhs_lines += [f"obs = Float32({sympy_to_julia_fp32(reduced[-1])})"]
    rhs_lines += ["ds = (t >= warmup_t) ? obs : 0.0f0"]

    u0_vals = ["1.0f0"] + ["0.0f0"] * (len(rho_syms) - 1) + ["0.0f0"]
    rhs_vec = [f"du{i + 1}" for i in range(len(rho_syms))] + ["ds"]
    obs_index = len(rho_syms) + 1

    helper = f"""using DifferentialEquations
using DiffEqGPU
using CUDA
using StaticArrays
using DelimitedFiles

@inline function rhs(u, p, t)
    eps = p[1]
    A = p[2]
    warmup_t = p[3]
    {"; ".join(rhs_lines)}
    return @SVector [{", ".join(rhs_vec)}]
end

function main()
    if length(ARGS) < 12
        println(stderr, "Expected 12 args.")
        exit(2)
    end

    out_csv = ARGS[1]
    nx = parse(Int, ARGS[2])
    ny = parse(Int, ARGS[3])
    num_t = parse(Int, ARGS[4])
    dt = parse(Float64, ARGS[5])
    eps_min = parse(Float64, ARGS[6])
    eps_max = parse(Float64, ARGS[7])
    A_min = parse(Float64, ARGS[8])
    A_max = parse(Float64, ARGS[9])
    warmup_steps = parse(Int, ARGS[10])
    t0 = parse(Float64, ARGS[11])
    timing_txt = ARGS[12]

    eps_list = collect(Float32, range(Float32(eps_min), stop=Float32(eps_max), length=nx))
    A_list = collect(Float32, range(Float32(A_min), stop=Float32(A_max), length=ny))
    warmup_t = Float32(t0 + warmup_steps * dt)
    denom_t = Float32(max(num_t - 1 - warmup_steps, 1)) * Float32(dt)

    params = Vector{{SVector{{3, Float32}}}}(undef, nx * ny)
    k = 1
    for j in 1:ny
        for i in 1:nx
            params[k] = @SVector [eps_list[i], A_list[j], warmup_t]
            k += 1
        end
    end

    u0 = @SVector [{", ".join(u0_vals)}]
    tf = t0 + dt * max(num_t - 1, 0)
    prob = ODEProblem{{false}}(rhs, u0, (Float32(t0), Float32(tf)), params[1])
    prob_func = (pr, i, repeat) -> remake(pr; p=params[i])
    eprob = EnsembleProblem(prob; prob_func=prob_func, safetycopy=false)

    solve_start = time()
    sol = solve(
        eprob,
        GPUTsit5(),
        DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend());
        trajectories=length(params),
        adaptive=false,
        dt=Float32(dt),
        save_everystep=false
    )
    CUDA.synchronize()
    solve_elapsed = time() - solve_start

    out = zeros(Float32, ny, nx)
    for idx in 1:length(params)
        j = Int(fld(idx - 1, nx)) + 1
        i = Int(mod(idx - 1, nx)) + 1
        out[j, i] = sol[idx][end][{obs_index}] / denom_t
    end

    writedlm(out_csv, out, ',')
    open(timing_txt, "w") do io
        write(io, string(solve_elapsed))
    end
    println("JULIA_SOLVE_TIME=", solve_elapsed)
end

main()
"""
    path.write_text(helper, encoding="utf-8")


def run_julia_gpu_solver(cfg: BenchConfig, *, julia_cmd: str) -> tuple[np.ndarray, float]:
    if shutil.which(julia_cmd) is None:
        raise RuntimeError(f"Julia executable '{julia_cmd}' was not found in PATH.")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        helper_path = td_path / JULIA_HELPER_NAME
        out_csv = td_path / "julia_out.csv"
        timing_txt = td_path / "julia_timing.txt"

        prep_start = time.time()
        write_julia_helper(helper_path, cfg)
        prep_elapsed = time.time() - prep_start

        cmd = [julia_cmd,
               str(helper_path),
               str(out_csv),
               str(cfg.nx),
               str(cfg.ny),
               str(cfg.num_t),
               repr(cfg.dt),
               repr(float(cfg.eps_list[0])),
               repr(float(cfg.eps_list[-1])),
               repr(float(cfg.A_list[0])),
               repr(float(cfg.A_list[-1])),
               str(cfg.warmup_steps),
               repr(float(cfg.tlist[0]) if cfg.num_t else 0.0),
               str(timing_txt)]

        process_start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        process_elapsed = time.time() - process_start
        if proc.returncode != 0:
            raise RuntimeError(f"Julia backend failed.\nstdout:\n{proc.stdout}\n"
                               f"stderr:\n{proc.stderr}")

        julia_elapsed = parse_julia_time(proc.stdout, timing_txt) or process_elapsed
        if cfg.timings:
            print(f"julia_gpu timings: prep={prep_elapsed:.3f}s "
                  f"julia_solve={julia_elapsed:.3f}s subprocess_total={process_elapsed:.3f}s")
        p_mat = np.loadtxt(out_csv, delimiter=",", dtype=np.float32)

    return p_mat, julia_elapsed


def parse_julia_time(stdout: str, timing_path: Path) -> float | None:
    match = re.search(r"JULIA_SOLVE_TIME=([0-9eE+\.-]+)", stdout or "")
    if match:
        return float(match.group(1))
    if timing_path.exists():
        txt = timing_path.read_text(encoding="ascii").strip()
        return float(txt) if txt else None
    return None


# -----------------------------------------------------------------------------
# Unified solver execution and modes
# -----------------------------------------------------------------------------


def solver_dispatch(args: argparse.Namespace) -> dict[
        str, Callable[[BenchConfig], tuple[np.ndarray, float]]]:
    return {
        "gpu": run_gpu_solver,
        "python_cpu": run_python_cpu_solver,
        "python_ode_cpu": run_python_ode_cpu_solver,
        "qutip_cpu": run_qutip_cpu_solver,
        "julia_gpu": lambda cfg: run_julia_gpu_solver(cfg, julia_cmd=args.julia_cmd),
    }


def run_solver(name: str, base_cfg: BenchConfig,
               args: argparse.Namespace) -> tuple[np.ndarray, float, BenchConfig]:
    if name not in SOLVER_SET:
        raise ValueError(f"Unknown solver '{name}'. Available solvers: {', '.join(SOLVERS)}")

    cfg = config_for_solver(name, base_cfg, args)
    show_column_progress = cfg.progress and name in {"python_cpu", "python_ode_cpu", "qutip_cpu"}

    running = f"{name}: Running  grid={cfg.nx}x{cfg.ny}  steps={cfg.num_steps}"
    if show_column_progress:
        print(running, flush=True)
    else:
        print(running, end="", flush=True)

    p_mat, elapsed = solver_dispatch(args)[name](cfg)

    result = f"{name}: time={elapsed:8.3f}s  grid={cfg.nx}x{cfg.ny}  steps={cfg.num_steps}"
    if show_column_progress:
        print(result)
    else:
        # Use backspaces instead of only carriage return because some IDE consoles
        # and captured terminals ignore "\r" and otherwise concatenate the two strings.
        print("\b" * len(running) + result, flush=True)

    gc.collect()
    return p_mat, elapsed, cfg


def normalize_mode_and_solver(args: argparse.Namespace) -> tuple[str, str]:
    mode = args.mode
    solver = args.solver
    if args.mode_or_solver is not None:
        token = args.mode_or_solver
        if token in SOLVER_SET:
            mode, solver = "single", token
        else:
            mode = token
    return mode, solver


def selected_solvers(mode: str, args: argparse.Namespace) -> tuple[str, ...]:
    if mode == "single":
        return (args.solver, )
    if mode == "diff":
        return tuple(dict.fromkeys((args.solver, args.solver_b)))
    if mode == "all":
        return SOLVERS
    if mode == "full_benchmark":
        return tuple(parse_solver_list(args.full_solvers, SOLVER_SET))
    raise ValueError("mode must be one of: single, all, diff, full_benchmark")


def warmup_gpu_solver_for_benchmark(base_cfg: BenchConfig, args: argparse.Namespace) -> None:
    """Run one unmeasured GPU solve before full_benchmark timing.

    This removes one-time GQIS overhead such as symbolic RHS preparation,
    code generation/compilation, CUDA context creation, and cache initialization
    from the first measured benchmark point.
    """
    side = int(args.bench_min_side_size)
    warm_cfg = replace(base_cfg,
                       eps_list=np.linspace(float(base_cfg.eps_list[0]),
                                           float(base_cfg.eps_list[-1]), side,
                                           dtype=base_cfg.eps_list.dtype),
                       A_list=np.linspace(float(base_cfg.A_list[0]),
                                         float(base_cfg.A_list[-1]), side,
                                         dtype=base_cfg.A_list.dtype), progress=False)

    print(f"gpu: warmup/precalculation  grid={warm_cfg.nx}x{warm_cfg.ny}  "
          f"steps={warm_cfg.num_steps}")
    _p_mat, _elapsed = run_gpu_solver(warm_cfg)
    gc.collect()


def _full_benchmark_worker(result_queue, name: str, cfg: BenchConfig,
                           args: argparse.Namespace) -> None:
    try:
        _p_mat, elapsed, _actual_cfg = run_solver(name, cfg, args)
        result_queue.put(("ok", elapsed))
    except BaseException as exc:
        result_queue.put(("error", repr(exc)))


def run_full_solver_with_timeout(name: str, cfg: BenchConfig, args: argparse.Namespace,
                                 timeout_s: float) -> float:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_full_benchmark_worker, args=(result_queue, name, cfg, args))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        print(f"{name}: exceeded {timeout_s:.3g}s; terminating current calculation before "
              "the next point.")
        terminate_process_tree(proc)
        raise TimeoutError(f"{name} exceeded {timeout_s:.3g}s")
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(f"{name} finished without returning a benchmark result.") from exc
    if status == "error":
        raise RuntimeError(payload)
    return float(payload)


def run_full_benchmark(base_cfg: BenchConfig, args: argparse.Namespace,
                       solvers: tuple[str, ...]) -> list[dict]:
    sides = benchmark_sides(args.bench_min_side_size, args.bench_max_side_size)
    rows: list[dict] = []
    histories: dict[str, list[tuple[int, float]]] = {name: [] for name in solvers}
    stopped: dict[str, bool] = {name: False for name in solvers}
    gpu_first_rhs_stage_s = np.nan

    if "gpu" in solvers:
        warmup_gpu_solver_for_benchmark(base_cfg, args)
        gpu_first_rhs_stage_s = LAST_GPU_RHS_STAGE_S

    for name in solvers:
        print(f"\nFull benchmark solver: {name}")
        for side in sides:
            num_simulations = int(side * side)
            if not stopped[name] and should_extrapolate_next(
                    histories[name], args.bench_solver_time_limit):
                stopped[name] = True
                print(f"{name}: the last timing ratio is above 3.6x per side doubling and the "
                      f"point exceeds half of the {args.bench_solver_time_limit:.3g}s limit; "
                      f"skipping side={side} and extrapolating.")
            if stopped[name]:
                est = extrapolate_loglog(histories[name], side,
                                         slope_points=BENCH_EXTRAPOLATION_POINTS)
                rows.append(benchmark_row(side, num_simulations, name, est, "extrapolated"))
                print(f"{name:>10s} side={side}: extrapolated {est:.3g}s")
                continue

            side_cfg = replace(base_cfg,
                               eps_list=np.linspace(float(base_cfg.eps_list[0]),
                                                   float(base_cfg.eps_list[-1]), side,
                                                   dtype=base_cfg.eps_list.dtype),
                               A_list=np.linspace(float(base_cfg.A_list[0]),
                                                 float(base_cfg.A_list[-1]), side,
                                                 dtype=base_cfg.A_list.dtype), progress=False)

            try:
                if name == "gpu":
                    _p_mat, elapsed, _actual_cfg = run_solver(name, side_cfg, args)
                else:
                    elapsed = run_full_solver_with_timeout(
                        name, side_cfg, args, args.bench_solver_time_limit)
                histories[name].append((side, elapsed))
                rows.append(benchmark_row(side, num_simulations, name, elapsed, "measured"))
                if elapsed >= args.bench_solver_time_limit:
                    stopped[name] = True
                    print(f"{name}: reached time limit after side={side}; "
                          "larger sizes will be extrapolated.")
            except Exception as exc:
                est = extrapolate_loglog(histories[name], side,
                                         slope_points=BENCH_EXTRAPOLATION_POINTS)
                status = "extrapolated" if np.isfinite(est) else "failed"
                stopped[name] = True
                rows.append(benchmark_row(side, num_simulations, name, est, status))
                print(f"{name:>10s} side={side}: {status.upper()} ({exc})")

    out_csv = Path(f"{args.output_filename}.csv")
    out_png = Path(f"{args.output_filename}.png")
    metadata = collect_equipment_info()
    metadata.update({
        "system_levels": "2", "benchmark_solvers": ",".join(solvers),
        "simulation_periods": f"{base_cfg.tr:.9g}",
        "solver_steps_per_period": str(base_cfg.solver_steps_per_period),
        "solver_steps_per_trajectory": str(base_cfg.num_steps),
        "time_step": f"{base_cfg.dt:.9g}",
        "averaging_skip_fraction": f"{base_cfg.warmup_time:.9g}",
        "python_cpu_step_density_divider": str(args.python_cpu_spp_divider),
        "python_ode_output_density_divider": str(args.python_ode_cpu_spp_divider),
        "qutip_output_density_divider": str(args.qutip_cpu_spp_divider),
        "gpu_precision": base_cfg.gpu_precision,
        "python_cpu_precision": base_cfg.cpu_precision,
        "python_ode_cpu_precision": "fp64", "qutip_cpu_precision": "fp64",
        "julia_gpu_precision": "fp32", "cpu_workers": str(base_cfg.workers),
        "Delta": f"{base_cfg.delta:.9g}", "w": f"{base_cfg.w:.9g}",
        "gamma1": f"{base_cfg.gamma1:.9g}", "gamma2": f"{base_cfg.gamma2:.9g}",
        "eps_min": f"{float(base_cfg.eps_list[0]):.9g}",
        "eps_max": f"{float(base_cfg.eps_list[-1]):.9g}",
        "A_min": f"{float(base_cfg.A_list[0]):.9g}",
        "A_max": f"{float(base_cfg.A_list[-1]):.9g}",
        "benchmark_min_side": str(args.bench_min_side_size),
        "benchmark_max_side": str(args.bench_max_side_size),
        "solver_time_limit_s": f"{args.bench_solver_time_limit:.9g}",
    })
    if np.isfinite(gpu_first_rhs_stage_s):
        metadata["gpu_first_rhs_stage_s"] = f"{gpu_first_rhs_stage_s:.9g}"
    print_equipment_info(metadata)
    save_benchmark_csv(rows, out_csv, metadata=metadata)
    reference_lines = []
    if np.isfinite(gpu_first_rhs_stage_s):
        reference_lines.append({"y": gpu_first_rhs_stage_s,
                                "label": f"GPU first RHS/codegen {gpu_first_rhs_stage_s:.3g}s",
                                "color": "0.25", "linestyle": ":"})
    plot_full_benchmark(rows, solvers, out_png,
                        title="Calculation time scaling for different numerical approaches "
                              "(2-level system)",
                        show=not args.no_plot, metadata=metadata, reference_lines=reference_lines)
    return rows


def benchmark_row(side: int, num_simulations: int, solver: str, elapsed: float,
                  status: str) -> dict:
    return {
        "side_dimension": side,
        "number_of_simulations": num_simulations,
        "solver": solver,
        "time_s": float(elapsed),
        "prep_s": np.nan,
        "calc_s": np.nan,
        "status": status,
    }


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-level interferogram benchmark.")

    parser.add_argument("mode_or_solver", nargs="?",
                        help="single/all/diff/full_benchmark or a solver name")
    parser.add_argument("--mode",
                        choices=["single", "all", "diff", "full_benchmark"], default=None)
    parser.add_argument("--solver", "--solver-a", dest="solver", choices=SOLVERS, default=None,
                        help="solver for single mode or first solver for diff mode")
    parser.add_argument("--solver-b", choices=SOLVERS, default=None)

    parser.add_argument("--detuning-points", "--nx", dest="nx", metavar="DETUNING_POINTS",
                        type=int, default=None)
    parser.add_argument("--amplitude-points", "--ny", dest="ny", type=int, default=None,
                        metavar="AMPLITUDE_POINTS",
                        help="drive-amplitude grid points; default equals detuning points")
    parser.add_argument("--solver-steps-per-period", "--rk4-steps-per-period",
                        "--samples-per-period", dest="solver_steps_per_period",
                        metavar="SOLVER_STEPS_PER_PERIOD", type=int, default=None)
    parser.add_argument("--simulation-periods", "--tr", dest="tr",
                        metavar="SIMULATION_PERIODS", type=float, default=None)
    parser.add_argument("--averaging-skip-fraction", "--warmup-time", dest="warmup_time",
                        metavar="AVERAGING_SKIP_FRACTION", type=float, default=None,
                        help="initial fraction of time excluded from the time average")
    parser.add_argument("--cpu-worker-count", "--workers", dest="workers",
                        metavar="CPU_WORKER_COUNT", type=int, default=None)

    parser.add_argument("--delta", "--energy-gap", dest="delta", type=float, default=None)
    parser.add_argument("--w-over-delta", "--drive-angular-frequency-over-gap",
                        dest="w_over_delta", type=float, default=None)
    parser.add_argument("--gamma-phi-per-T", "--pure-dephasing-per-period",
                        dest="gamma_phi_per_T", type=float, default=None)
    parser.add_argument("--gamma1-per-T", "--qubit-relaxation-per-period",
                        dest="gamma1_per_T", type=float, default=None)
    parser.add_argument("--eps-max-factor", "--detuning-half-range-factor",
                        dest="eps_max_factor", type=float, default=None,
                        help="eps range is +/- factor*w")
    parser.add_argument("--A-max-factor", "--drive-amplitude-max-factor", dest="A_max_factor",
                        type=float, default=None, help="A range is 0..factor*w")

    parser.add_argument("--gpu-precision", choices=["fp32", "fp64"], default=None)
    parser.add_argument("--cpu-precision", choices=["fp32", "fp64"], default=None)
    parser.add_argument("--cpu-density-divider", "--cpu-spp-divider", dest="cpu_spp_divider",
                        type=int, default=None,
                        help="override the divider for every CPU backend")
    parser.add_argument("--python-cpu-step-density-divider", "--python-cpu-spp-divider",
                        dest="python_cpu_spp_divider", metavar="STEP_DENSITY_DIVIDER", type=int,
                        default=None,
                        help="fixed-step RK4 integration-grid divider")
    parser.add_argument("--python-ode-output-density-divider", "--python-ode-cpu-spp-divider",
                        dest="python_ode_cpu_spp_divider", metavar="OUTPUT_DENSITY_DIVIDER",
                        type=int, default=None,
                        help="adaptive SciPy requested-output-grid divider")
    parser.add_argument("--qutip-output-density-divider", "--qutip-cpu-spp-divider",
                        dest="qutip_cpu_spp_divider", metavar="OUTPUT_DENSITY_DIVIDER", type=int,
                        default=None,
                        help="adaptive QuTiP requested-output/coefficient-grid divider")

    parser.add_argument("--timings", action="store_true",
                        help="show backend internal timings when available")
    parser.add_argument("--julia-cmd", default=None)
    parser.add_argument("--plot-layout", choices=["windows", "panel"], default=None)
    parser.add_argument("--save-timings-txt", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--full-solvers", default=None)
    parser.add_argument("--bench-min-side-size", "--full-min-side", dest="bench_min_side_size",
                        type=int, default=None,
                        help="smallest square-grid side dimension for benchmark")
    parser.add_argument("--bench-max-side-size", "--full-max-side", dest="bench_max_side_size",
                        type=int, default=None,
                        help="biggest square-grid side dimension for benchmark")
    parser.add_argument("--bench-solver-time-limit", "--full-time-limit",
                        dest="bench_solver_time_limit", type=float, default=None,
                        help="terminate a measured point after this many seconds, then extrapolate "
                             "larger sizes")
    parser.add_argument("--output-filename", "--full-output-stem", dest="output_filename",
                        default=None, help="base output filename for CSV and PNG")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = user_settings()

    mode = settings["mode"]
    solver = settings.get("solver", "gpu")
    solver_b = settings.get("solver_b", "qutip_cpu")
    julia_cmd = settings["julia_cmd"]
    delta = settings["Delta"]
    w_over_delta = settings["w/Delta"]
    gamma_phi_per_T = settings["gamma_phi_per_T"]
    gamma1_per_T = settings["gamma1_per_T"]
    eps_max_factor = settings["eps_max/w"]
    A_max_factor = settings["A_max/w"]
    nx = settings["detuning_points"]
    ny = settings["amplitude_points"]
    solver_steps_per_period = settings["solver_steps_per_period"]
    tr = settings["simulation_periods"]
    warmup_time = settings["averaging_skip_fraction"]
    workers = settings["cpu_worker_count"]
    python_cpu_spp_divider = settings["python_cpu_step_density_divider"]
    python_ode_cpu_spp_divider = settings["python_ode_output_density_divider"]
    qutip_cpu_spp_divider = settings["qutip_output_density_divider"]
    gpu_precision = settings["gpu_precision"]
    cpu_precision = settings["cpu_precision"]
    plot_layout = settings["plot_layout"]
    full_solvers = settings.get("solvers", "gpu,qutip_cpu,julia_gpu")
    bench_min_side_size = settings["bench_min_side_size"]
    bench_max_side_size = settings["bench_max_side_size"]
    bench_solver_time_limit = settings["bench_solver_time_limit"]
    output_filename = settings["Output_filename"]

    # Optional CLI overrides.
    if args.mode_or_solver is not None:
        mode = args.mode_or_solver
    if args.mode is not None:
        mode = args.mode
    if args.solver is not None:
        solver = args.solver
    if args.solver_b is not None:
        solver_b = args.solver_b
    if args.julia_cmd is not None:
        julia_cmd = args.julia_cmd
    if args.delta is not None:
        delta = float(args.delta)
    if args.w_over_delta is not None:
        w_over_delta = float(args.w_over_delta)
    if args.gamma_phi_per_T is not None:
        gamma_phi_per_T = float(args.gamma_phi_per_T)
    if args.gamma1_per_T is not None:
        gamma1_per_T = float(args.gamma1_per_T)
    if args.eps_max_factor is not None:
        eps_max_factor = float(args.eps_max_factor)
    if args.A_max_factor is not None:
        A_max_factor = float(args.A_max_factor)
    if args.nx is not None:
        nx = int(args.nx)
    if args.ny is not None:
        ny = int(args.ny)
    if args.solver_steps_per_period is not None:
        solver_steps_per_period = int(args.solver_steps_per_period)
    if args.tr is not None:
        tr = float(args.tr)
    if args.warmup_time is not None:
        warmup_time = float(args.warmup_time)
    if args.workers is not None:
        workers = int(args.workers)
    if args.cpu_spp_divider is not None:
        cpu_spp_divider = int(args.cpu_spp_divider)
        python_cpu_spp_divider = cpu_spp_divider
        python_ode_cpu_spp_divider = cpu_spp_divider
        qutip_cpu_spp_divider = cpu_spp_divider
    if args.python_cpu_spp_divider is not None:
        python_cpu_spp_divider = int(args.python_cpu_spp_divider)
    if args.python_ode_cpu_spp_divider is not None:
        python_ode_cpu_spp_divider = int(args.python_ode_cpu_spp_divider)
    if args.qutip_cpu_spp_divider is not None:
        qutip_cpu_spp_divider = int(args.qutip_cpu_spp_divider)
    if args.gpu_precision is not None:
        gpu_precision = args.gpu_precision
    if args.cpu_precision is not None:
        cpu_precision = args.cpu_precision
    if args.plot_layout is not None:
        plot_layout = args.plot_layout
    if args.full_solvers is not None:
        full_solvers = args.full_solvers
    if args.bench_min_side_size is not None:
        bench_min_side_size = int(args.bench_min_side_size)
    if args.bench_max_side_size is not None:
        bench_max_side_size = int(args.bench_max_side_size)
    if args.bench_solver_time_limit is not None:
        bench_solver_time_limit = float(args.bench_solver_time_limit)
    if args.output_filename is not None:
        output_filename = args.output_filename

    if python_cpu_spp_divider <= 0 or python_ode_cpu_spp_divider <= 0 or qutip_cpu_spp_divider <= 0:
        raise ValueError("CPU integration/output-density dividers must be positive integers.")

    args.mode = mode
    args.solver = solver
    args.solver_b = solver_b
    args.julia_cmd = julia_cmd
    args.delta = delta
    args.w_over_delta = w_over_delta
    args.gamma_phi_per_T = gamma_phi_per_T
    args.gamma1_per_T = gamma1_per_T
    args.eps_max_factor = eps_max_factor
    args.A_max_factor = A_max_factor
    args.nx = nx
    args.ny = ny
    args.solver_steps_per_period = solver_steps_per_period
    args.tr = tr
    args.warmup_time = warmup_time
    args.workers = workers
    args.python_cpu_spp_divider = python_cpu_spp_divider
    args.python_ode_cpu_spp_divider = python_ode_cpu_spp_divider
    args.qutip_cpu_spp_divider = qutip_cpu_spp_divider
    args.gpu_precision = gpu_precision
    args.cpu_precision = cpu_precision
    args.plot_layout = plot_layout
    args.full_solvers = full_solvers
    args.bench_min_side_size = bench_min_side_size
    args.bench_max_side_size = bench_max_side_size
    args.bench_solver_time_limit = bench_solver_time_limit
    args.output_filename = output_filename

    mode, solver = normalize_mode_and_solver(args)
    args.mode = mode
    args.solver = solver

    cfg = make_config(args)
    solvers = selected_solvers(mode, args)
    if mode != "full_benchmark":
        print_equipment_info()

    if "python_cpu" in solvers and args.python_cpu_spp_divider > 1:
        print("Warning: python_cpu uses fixed-step RK4. Large step-density dividers can reduce "
              "accuracy; use --python-cpu-step-density-divider 1 for strict validation.")
    if "python_ode_cpu" in solvers and args.python_ode_cpu_spp_divider > 1:
        print("Note: python_ode_cpu uses SciPy solve_ivp adaptively, but output is "
              "sampled on the reduced tlist. "
              "Use --python-ode-output-density-divider 1 for strict validation.")
    if "qutip_cpu" in solvers and args.qutip_cpu_spp_divider > 1:
        print("Note: qutip_cpu is adaptive internally, but the coefficient array is "
              "sampled on the reduced tlist.")

    print_config(cfg)
    print(f"mode={mode} solvers={','.join(solvers)}")

    if mode == "full_benchmark":
        run_full_benchmark(cfg, args, solvers)
        return

    results: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    used_cfgs: dict[str, BenchConfig] = {}

    run_order = solvers
    if mode == "all":
        run_order = ("gpu", "julia_gpu", "python_cpu", "python_ode_cpu", "qutip_cpu")
    if mode == "diff":
        run_order = tuple(sorted(solvers, key=lambda name: name == "julia_gpu"))

    for name in run_order:
        try:
            p_mat, elapsed, used_cfg = run_solver(name, cfg, args)
            results[name] = p_mat
            timings[name] = elapsed
            used_cfgs[name] = used_cfg
        except Exception as exc:
            if mode == "all":
                print(f"{name:>10s}: SKIPPED ({exc})")
                continue
            raise

    if args.save_timings_txt:
        save_timings_txt(timings, cfg, mode, run_order)

    if mode == "single":
        name = args.solver
        if not args.no_plot:
            plot_maps([(f"{name} (time={timings[name]:.2f}s)", results[name], "jet")],
                      used_cfgs[name], layout=args.plot_layout)
        return

    if mode == "all":
        if not args.no_plot and results:
            panels = [(f"{name} (time={timings[name]:.2f}s)", results[name], "jet")
                      for name in run_order if name in results]
            plot_maps(panels, cfg, layout=args.plot_layout, suptitle="All solvers")
        return

    # mode == diff
    a, b = args.solver, args.solver_b
    if results[a].shape != results[b].shape:
        raise ValueError(f"Cannot diff arrays with different shapes: {a}{results[a].shape} "
                         f"vs {b}{results[b].shape}")

    diff = results[a] - results[b]
    mse = float(np.mean(np.abs(diff)**2))
    rms = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(diff)))
    print(f"diff({a} - {b}): MSE={mse:.6e} RMS={rms:.6e} max_abs={max_abs:.6e}")

    if not args.no_plot:
        panels = [(f"Difference: {a} - {b}\nRMS={rms:.3e}, max_abs={max_abs:.3e}", diff,
                   "bwr"),
                  (f"{a} ({timings[a]:.2f}s)", results[a], "jet"),
                  (f"{b} ({timings[b]:.2f}s)", results[b], "jet")]
        plot_maps(panels, cfg, layout=args.plot_layout, suptitle="Diff mode")


def user_settings() -> dict:
    """User-editable defaults. Command-line arguments override these values."""
    adaptive_cpu_output_density_divider = 10

    # Solver options:
    # "gpu" : GQIS fixed-step CUDA solver.
    # "python_cpu" : fixed-step Python RK4 reference solver.
    # "python_ode_cpu" : adaptive SciPy RK45 CPU solver.
    # "qutip_cpu": adaptive QuTiP CPU reference solver.
    # "julia_gpu": external Julia DifferentialEquations/DiffEqGPU solver.
    # Replace any solver below from the list above to change it

    # Select one complete benchmark mode configuration.
    mode_settings = {"mode": "full_benchmark", "solvers": "gpu,qutip_cpu,julia_gpu"}
    # mode_settings = {"mode": "full_benchmark", "solvers": "gpu,python_cpu,python_ode_cpu,qutip_cpu,julia_gpu"}
    # mode_settings = {"mode": "single", "solver": "gpu"}
    # mode_settings = {"mode": "diff", "solver": "gpu", "solver_b": "qutip_cpu"}
    # mode_settings = {"mode": "all"}

    return {
        **mode_settings,
        "julia_cmd": "julia",  # Julia executable name or full path
        # Physics parameters.
        "Delta": 1.0,  # minimum energy gap
        "w/Delta": 1.14,  # drive angular frequency divided by Delta
        "gamma_phi_per_T": 0.04,  # pure dephasing accumulated per drive period
        "gamma1_per_T": 0.05,  # relaxation accumulated per drive period
        "eps_max/w": 10.0,  # maximum absolute detuning divided by w
        "A_max/w": 10.0,  # maximum drive amplitude divided by w
        # Benchmark grid and time grid.
        "detuning_points": 2048, #Amount of points on horizontal side corresponding to detuning
        "amplitude_points": None,  # None uses detuning_points
        # Increase if mesolve_2D reports non-finite output.
        "solver_steps_per_period": 256,
        "simulation_periods": 40.0,  # periods of driving
        "averaging_skip_fraction": 0.0,  # initial time fraction excluded from the average
        "cpu_worker_count": None,  # None uses os.cpu_count() - 1
        # Keep fixed-step CPU RK4 on the same integration grid as GPU RK4.
        # Adaptive solvers may use a reduced requested-output/coefficient grid.
        "python_cpu_step_density_divider": 1,  # same RK4 step density as GPU
        "python_ode_output_density_divider": adaptive_cpu_output_density_divider,
        "qutip_output_density_divider": adaptive_cpu_output_density_divider,
        # Precision and output options.
        "gpu_precision": "fp32",  # "fp32" or "fp64" for GQIS
        "cpu_precision": "fp64",  # "fp32" or "fp64" for python_cpu
        "plot_layout": "windows",  # "windows" or "panel"
        # Full-benchmark sweep limits.
        "bench_min_side_size": 16,  # smallest square-grid side dimension for benchmark
        "bench_max_side_size": 16384 * 1,  # biggest square-grid side dimension for benchmark
        "bench_solver_time_limit": 100.0*2,  # terminate calculation above this duration in seconds
        "Output_filename": "Benchmark_01_full_benchmark",  # base filename for CSV and PNG output
    }


if __name__ == "__main__":
    main()

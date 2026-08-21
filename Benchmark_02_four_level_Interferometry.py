"""Benchmark 02: four-level interferometry benchmark (GPU / Python / QuTiP / Julia).

The Julia backend uses the same reduced SymPy RHS builder as GQIS.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from Benchmark_full_tools import (benchmark_sides, collect_equipment_info, extrapolate_loglog,
                                  parse_solver_list, plot_benchmark, print_equipment_info,
                                  save_benchmark_csv, should_extrapolate_next,
                                  sympy_to_julia_fp32, terminate_process_tree,
                                  )
from gqis import build_reduced_lindblad_rhs, mesolve_2D

SOLVER_NAMES = {"gpu", "python_cpu", "python_ode_cpu", "qutip_cpu", "julia_gpu"}
BENCH_EXTRAPOLATION_POINTS = 2


@dataclass
class SolverTiming:
    total: float
    prep: float = 0.0
    compute: float = 0.0
    rhs_stage: float = np.nan


@dataclass
class FourCfg:
    delta_abs: float
    w_abs: float
    gamma1: float
    gamma2: float
    kappa: float
    Ap: float
    g1: float
    wr2: float
    A_list: np.ndarray
    eps_list: np.ndarray
    tlist: np.ndarray
    warmup_time: float
    workers: int
    timings: bool
    regime_name: str
    progress: bool

    @property
    def dt(self) -> float:
        if len(self.tlist) < 2:
            return 1.0
        return float(self.tlist[1] - self.tlist[0])

    @property
    def num_t(self) -> int:
        return int(len(self.tlist))

    @property
    def num_steps(self) -> int:
        return max(self.num_t - 1, 0)

    @property
    def warmup_steps(self) -> int:
        return int(np.floor(self.warmup_time * self.num_steps))

    @property
    def N(self) -> int:
        return 4


def _print_solver_result(name: str, timing: SolverTiming, p_mat: np.ndarray) -> None:
    print(f"{name:>10s}: total={timing.total:8.3f}s  "
          f"prep={timing.prep:8.3f}s  calc={timing.compute:8.3f}s  "
          f"min={np.min(p_mat):.6e} max={np.max(p_mat):.6e}  "
          f"(max-0.5)={float(np.max(p_mat) - 0.5):+.6e}")


def _print_progress(label: str, done: int, total: int) -> None:
    """Lightweight progress display without adding a tqdm dependency."""
    if total <= 0:
        return
    pct = 100.0 * done / total
    print(f"\r{label}: {done}/{total} columns ({pct:5.1f}%)", end="", flush=True)
    if done >= total:
        print()


def _set_same_window_geometry(fig: plt.Figure, x: int = 80, y: int = 60, w: int = 900,
                              h: int = 900) -> None:
    try:
        manager = fig.canvas.manager
        if hasattr(manager, "window"):
            win = manager.window
            if hasattr(win, "setGeometry"):
                win.setGeometry(x, y, w, h)
            elif hasattr(win, "wm_geometry"):
                win.wm_geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass


def _plot_maps_windows(panels, eps_list: np.ndarray, A_list: np.ndarray, *, same_spot: bool = True,
                       title_suffix: str = "") -> None:
    for title, p_mat, cmap in panels:
        fig, ax = plt.subplots(figsize=(8, 8))
        if same_spot:
            _set_same_window_geometry(fig)
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        im = ax.imshow(10 * np.log(p_mat), aspect="auto", cmap=cmap, origin="lower",
                       extent=[eps_list[0], eps_list[-1], A_list[0], A_list[-1]])
        fig.colorbar(im, ax=ax, label="|Re(<a>)|")
        ax.set_xlabel("eps")
        ax.set_ylabel("A")
        ax.set_title(title + title_suffix)
        plt.tight_layout()
    plt.show()


def _build_four_level_symbolic(cfg: FourCfg):
    N = cfg.N
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
    hp = cfg.Ap * (a1.H + a1)
    hc11u = cfg.g1 * cfg.delta_abs * (sm1.H * a1 + sm1 * a1.H)
    H0 = hp
    Ht = hc11u / sp.sqrt(cfg.delta_abs**2 +
                         Drive**2) + hq11 * (sp.sqrt(cfg.delta_abs**2 + Drive**2) - cfg.wr2)
    H = H0 + Ht
    drive_expr = eps + A * sp.cos(cfg.w_abs * t)

    L_rel = sp.kronecker_product(sp.sqrt(cfg.gamma1) * sp_lower, qeye)
    L_phi = sp.kronecker_product(sp.sqrt(cfg.gamma2) * sz, qeye)
    L_kappa = sp.sqrt(cfg.kappa) * a1
    col_ops = [L_rel, L_phi, L_kappa]
    mean_op = a1
    return H, drive_expr, col_ops, mean_op, eps, A, t, Drive


def run_gpu_solver(cfg: FourCfg) -> Tuple[np.ndarray, SolverTiming]:
    total_start = time.time()
    prep_start = time.time()
    H, drive_expr, col_ops, mean_op, eps_sym, A_sym, _, _ = _build_four_level_symbolic(cfg)
    prep_time = time.time() - prep_start
    compute_start = time.time()
    out, timing_info = mesolve_2D(H, drive_expr, col_ops, mean_op,
                                  np.asarray(cfg.tlist, dtype=np.float32),
                                  var_arrays={eps_sym: np.asarray(cfg.eps_list, dtype=np.float32),
                                              A_sym: np.asarray(cfg.A_list, dtype=np.float32)},
                                  warmup_time=cfg.warmup_time, timings=cfg.timings,
                                  return_timing_info=True)
    compute_time = time.time() - compute_start
    p = np.abs(np.real(np.asarray(out))).T
    rhs_stage = float((timing_info or {}).get("rhs_stage_s", np.nan))
    return p, SolverTiming(total=time.time() - total_start, prep=prep_time, compute=compute_time,
                           rhs_stage=rhs_stage)


_PY_WORKER_CFG: Optional[FourCfg] = None
_PY_DRHO_FUN = None
_PY_OBS_FUN = None
_PY_RHO_LEN = None
_PY_MODE = None
_PY_PREP_TIMING_QUEUE = None


def _prepare_python_rhs(cfg: FourCfg):
    (H, drive_expr, col_ops, mean_op, eps_sym, A_sym, t_sym,
     Drive_sym) = _build_four_level_symbolic(cfg)
    H_sub = H.subs(Drive_sym, drive_expr)
    drho_eqs, obs_expr, _, meta = build_reduced_lindblad_rhs(
        cfg.N, H_sub, col_ops, mean_op)
    rho_syms = meta["rho_syms"]
    u_syms = [sp.Symbol(f"u{i}", real=True) for i in range(len(rho_syms))]
    repl = {rho_syms[i]: u_syms[i] for i in range(len(rho_syms))}
    drho_local = [sp.simplify(e.subs(repl)) for e in drho_eqs]
    obs_local = sp.simplify(obs_expr.subs(repl))
    drho_fun = sp.lambdify((t_sym, eps_sym, A_sym, *u_syms), drho_local, "numpy")
    obs_fun = sp.lambdify((t_sym, eps_sym, A_sym, *u_syms), obs_local, "numpy")
    return drho_fun, obs_fun, len(rho_syms)


def _init_py_worker(cfg: FourCfg, mode: str, prep_timing_queue=None):
    global _PY_WORKER_CFG, _PY_DRHO_FUN, _PY_OBS_FUN, _PY_RHO_LEN, _PY_MODE, _PY_PREP_TIMING_QUEUE
    _PY_WORKER_CFG = cfg
    _PY_MODE = mode
    _PY_PREP_TIMING_QUEUE = prep_timing_queue
    if mode in {"python_cpu", "python_ode_cpu"}:
        prep_start = time.time()
        _PY_DRHO_FUN, _PY_OBS_FUN, _PY_RHO_LEN = _prepare_python_rhs(cfg)
        if _PY_PREP_TIMING_QUEUE is not None:
            _PY_PREP_TIMING_QUEUE.put(time.time() - prep_start)


def _solve_point_python(eps0: float, A: float, cfg: FourCfg) -> float:
    dt = cfg.dt
    dt2 = 0.5 * dt
    dt6 = dt / 6.0
    warmup = cfg.warmup_steps
    denom = max(cfg.num_steps - warmup, 1)
    t0 = float(cfg.tlist[0]) if cfg.num_t > 0 else 0.0

    u = np.zeros(_PY_RHO_LEN, dtype=np.float64)
    u[0] = 1.0
    acc = 0.0

    def du_at(tt, uu):
        vals = _PY_DRHO_FUN(tt, eps0, A, *uu)
        return np.asarray(vals, dtype=np.float64)

    def obs_at(tt, uu):
        return float(_PY_OBS_FUN(tt, eps0, A, *uu))

    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for step in range(cfg.num_steps):
                tt = t0 + step * dt
                k1 = du_at(tt, u)
                k2 = du_at(tt + dt2, u + dt2 * k1)
                k3 = du_at(tt + dt2, u + dt2 * k2)
                k4 = du_at(tt + dt, u + dt * k3)
                u = u + dt6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                if not np.all(np.isfinite(u)):
                    return np.nan
                if step >= warmup:
                    acc += obs_at(tt, u)
                    if not np.isfinite(acc):
                        return np.nan
    except FloatingPointError:
        return np.nan
    return abs(acc / float(denom))


def _solve_point_python_ode(eps0: float, A: float, cfg: FourCfg) -> float:
    from scipy.integrate import solve_ivp

    tlist = np.asarray(cfg.tlist, dtype=np.float64)
    if len(tlist) < 2:
        return np.nan

    u0 = np.zeros(_PY_RHO_LEN, dtype=np.float64)
    u0[0] = 1.0

    def rhs(tt, uu):
        vals = _PY_DRHO_FUN(tt, eps0, A, *uu)
        return np.asarray(vals, dtype=np.float64)

    sol = solve_ivp(rhs, (float(tlist[0]), float(tlist[-1])), u0, method="RK45", t_eval=tlist,
                    rtol=1e-7, atol=1e-9)
    if not sol.success or sol.y.shape[1] == 0:
        return np.nan

    warmup = cfg.warmup_steps
    start = min(warmup + 1, sol.y.shape[1] - 1)
    vals = [float(_PY_OBS_FUN(float(tlist[i]), eps0, A, *sol.y[:, i]))
            for i in range(start, sol.y.shape[1])]
    return abs(float(np.mean(vals))) if vals else np.nan


def _solve_point_qutip(eps0: float, A: float, cfg: FourCfg) -> float:
    import qutip as qt

    N = cfg.N
    qeye = np.eye(N - 2, dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sp_raise = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sp_lower = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    sm1 = np.kron(sp_lower, qeye)
    sz1 = np.kron(sz, qeye)
    a1 = np.kron(qeye, sp_raise)

    hq11 = sz1 / 2.0
    hp = cfg.Ap * (a1.conj().T + a1)
    hc11u = cfg.g1 * cfg.delta_abs * (sm1.conj().T @ a1 + sm1 @ a1.conj().T)

    drive = eps0 + A * np.cos(cfg.w_abs * cfg.tlist)
    srt = np.sqrt(cfg.delta_abs**2 + drive**2)
    c1 = 1.0 / srt
    c2 = srt - cfg.wr2

    H = [qt.Qobj(hp), [qt.Qobj(hc11u), c1], [qt.Qobj(hq11), c2]]
    c_ops = [np.sqrt(cfg.gamma1) * qt.Qobj(sm1),
             np.sqrt(cfg.gamma2) * qt.Qobj(sz1),
             np.sqrt(cfg.kappa) * qt.Qobj(a1)]
    mean_op = qt.Qobj(a1)
    rho0 = qt.basis(N, 0) * qt.basis(N, 0).dag()
    res = qt.mesolve(H, rho0, cfg.tlist, c_ops=c_ops, e_ops=[mean_op])
    vals = np.asarray(np.real(np.asarray(res.expect[0])), dtype=np.float64)
    warmup = cfg.warmup_steps
    if not len(vals):
        return 0.0
    start = min(warmup + 1, len(vals) - 1)
    return abs(float(np.mean(vals[start:])))


def _worker_column(eps_idx: int):
    if _PY_WORKER_CFG is None:
        raise RuntimeError("Worker not initialized")
    cfg = _PY_WORKER_CFG
    eps0 = float(cfg.eps_list[eps_idx])
    col = np.empty(len(cfg.A_list), dtype=np.float64)
    if _PY_MODE == "python_cpu":
        for i, A in enumerate(cfg.A_list):
            col[i] = _solve_point_python(eps0, float(A), cfg)
    elif _PY_MODE == "python_ode_cpu":
        for i, A in enumerate(cfg.A_list):
            col[i] = _solve_point_python_ode(eps0, float(A), cfg)
    elif _PY_MODE == "qutip_cpu":
        for i, A in enumerate(cfg.A_list):
            col[i] = _solve_point_qutip(eps0, float(A), cfg)
    else:
        raise ValueError(f"Unsupported worker mode '{_PY_MODE}'")
    return eps_idx, col


def _run_parallel_columns(mode: str, cfg: FourCfg) -> Tuple[np.ndarray, SolverTiming]:
    start = time.time()
    out = np.empty((len(cfg.A_list), len(cfg.eps_list)), dtype=np.float64)
    ctx = mp.get_context("spawn")
    prep_timing_queue = ctx.Queue()
    with ctx.Pool(processes=max(1, int(cfg.workers)), initializer=_init_py_worker,
                  initargs=(cfg, mode, prep_timing_queue)) as pool:
        done = 0
        total_cols = len(cfg.eps_list)
        update_every = max(1, total_cols // 50)
        for eps_idx, col in pool.imap_unordered(_worker_column, range(len(cfg.eps_list))):
            out[:, eps_idx] = col
            done += 1
            if cfg.progress and (done == 1 or done == total_cols or done % update_every == 0):
                _print_progress(mode, done, total_cols)

    prep_times = []
    while True:
        try:
            prep_times.append(float(prep_timing_queue.get_nowait()))
        except queue.Empty:
            break
    total_time = time.time() - start
    # Worker preparations happen concurrently; max is the closest wall-time estimate.
    prep_time = max(prep_times) if prep_times else 0.0
    compute_time = max(0.0, total_time - prep_time)
    return out.astype(np.float32, copy=False), SolverTiming(total=total_time, prep=prep_time,
                                                            compute=compute_time)


def run_python_cpu_solver(cfg: FourCfg) -> Tuple[np.ndarray, SolverTiming]:
    return _run_parallel_columns("python_cpu", cfg)


def run_python_ode_cpu_solver(cfg: FourCfg) -> Tuple[np.ndarray, SolverTiming]:
    return _run_parallel_columns("python_ode_cpu", cfg)


def run_qutip_cpu_solver(cfg: FourCfg) -> Tuple[np.ndarray, SolverTiming]:
    return _run_parallel_columns("qutip_cpu", cfg)


def _write_julia_helper(path: Path, drho_eqs, obs_expr, rho_syms):
    rho_locals = [sp.Symbol(f"u{i}") for i in range(len(rho_syms))]
    repl = {rho_syms[i]: rho_locals[i] for i in range(len(rho_syms))}
    drho_local = [sp.simplify(e.subs(repl)) for e in drho_eqs]
    obs_local = sp.simplify(obs_expr.subs(repl))
    common, reduced = sp.cse([*drho_local, obs_local], symbols=sp.numbered_symbols("tmp"),
                             optimizations="basic")

    rhs_local_lines = [f"u{i} = u[{i + 1}]" for i in range(len(rho_syms))]
    rhs_local_lines += [f"{symbol} = Float32({sympy_to_julia_fp32(expr)})"
                        for symbol, expr in common]
    rhs_local_lines += [f"du{i + 1} = Float32({sympy_to_julia_fp32(expr)})"
                        for i, expr in enumerate(reduced[:-1])]
    rhs_local_lines += [f"obs = Float32({sympy_to_julia_fp32(reduced[-1])})"]
    rhs_local_lines += ["ds = (t >= warmup_t) ? obs : 0.0f0"]
    rhs_vec = [f"du{i + 1}" for i in range(len(rho_syms))] + ["ds"]

    u0_vals = ["1.0f0"] + ["0.0f0"] * (len(rho_syms) - 1) + ["0.0f0"]
    code = f"""using DifferentialEquations
using DiffEqGPU
using CUDA
using StaticArrays
using DelimitedFiles

@inline function rhs(u, p, t)
    eps = p[1]
    A = p[2]
    warmup_t = p[3]
    {"; ".join(rhs_local_lines)}
    return @SVector [{", ".join(rhs_vec)}]
end

function main()
    out_csv = ARGS[1]
    nx = parse(Int, ARGS[2]); ny = parse(Int, ARGS[3])
    num_t = parse(Int, ARGS[4]); dt = parse(Float64, ARGS[5])
    eps_min = parse(Float64, ARGS[6]); eps_max = parse(Float64, ARGS[7])
    A_min = parse(Float64, ARGS[8]); A_max = parse(Float64, ARGS[9])
    warmup_steps = parse(Int, ARGS[10]); t0 = parse(Float64, ARGS[11])
    timing_txt = ARGS[12]
    eps_list = collect(Float32, range(Float32(eps_min), stop=Float32(eps_max), length=nx))
    A_list = collect(Float32, range(Float32(A_min), stop=Float32(A_max), length=ny))
    warmup_t = Float32(t0 + warmup_steps * dt)
    denom_t = Float32(max(num_t - 1 - warmup_steps, 1)) * Float32(dt)

    params = Vector{{SVector{{3, Float32}}}}(undef, nx * ny)
    k = 1
    for j in 1:ny
        Aj = A_list[j]
        for i in 1:nx
            params[k] = @SVector [eps_list[i], Aj, warmup_t]
            k += 1
        end
    end
    u0 = @SVector [{", ".join(u0_vals)}]
    tf = t0 + dt * max(num_t - 1, 0)
    prob = ODEProblem{{false}}(rhs, u0, (Float32(t0), Float32(tf)), params[1])
    prob_func = (pr, i, repeat) -> remake(pr; p=params[i])
    eprob = EnsembleProblem(prob; prob_func=prob_func, safetycopy=false)
    solve_start_ns = time_ns()
    sol = solve(eprob, GPUTsit5(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend());
                trajectories=length(params), adaptive=false, dt=Float32(dt),
                save_everystep=false)
    CUDA.synchronize()
    solve_elapsed = (time_ns() - solve_start_ns) / 1.0e9
    out = zeros(Float32, ny, nx)
    for idx in 1:length(params)
        j = Int(fld(idx - 1, nx)) + 1
        i = Int(mod(idx - 1, nx)) + 1
        out[j, i] = abs(sol[idx][end][{len(rho_syms) + 1}] / denom_t)
    end
    writedlm(out_csv, out, ',')
    open(timing_txt, "w") do io
        write(io, string(solve_elapsed))
    end
    println("JULIA_SOLVE_TIME=", solve_elapsed)
end
main()
"""
    path.write_text(code, encoding="utf-8")


def _parse_julia_solve_time(stdout: str, timing_path: Path) -> Optional[float]:
    prefix = "JULIA_SOLVE_TIME="
    for line in (stdout or "").splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix):].strip())
    if timing_path.exists():
        text = timing_path.read_text(encoding="ascii").strip()
        return float(text) if text else None
    return None


def run_julia_gpu_solver(cfg: FourCfg, julia_cmd: str = "julia",
                         timeout_s: Optional[float] = None) -> Tuple[np.ndarray, SolverTiming]:
    if shutil.which(julia_cmd) is None:
        raise RuntimeError(f"Julia executable '{julia_cmd}' was not found in PATH.")
    prep_start = time.time()
    H, drive_expr, col_ops, mean_op, _, _, _, Drive_sym = _build_four_level_symbolic(cfg)
    H_sub = H.subs(Drive_sym, drive_expr)
    drho_eqs, obs_expr, _, meta = build_reduced_lindblad_rhs(
        cfg.N, H_sub, col_ops, mean_op)
    rho_syms = meta["rho_syms"]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        helper = td_path / "bench02_four_level_sympy_rhs.jl"
        out_csv = td_path / "julia_out.csv"
        timing_txt = td_path / "julia_timing.txt"
        _write_julia_helper(helper, drho_eqs, obs_expr, rho_syms)
        prep_time = time.time() - prep_start
        cmd = [julia_cmd,
               str(helper),
               str(out_csv),
               str(len(cfg.eps_list)),
               str(len(cfg.A_list)),
               str(cfg.num_t),
               repr(cfg.dt),
               repr(float(cfg.eps_list[0])),
               repr(float(cfg.eps_list[-1])),
               repr(float(cfg.A_list[0])),
               repr(float(cfg.A_list[-1])),
               str(cfg.warmup_steps),
               repr(float(cfg.tlist[0]) if cfg.num_t > 0 else 0.0),
               str(timing_txt)]
        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Julia backend exceeded {timeout_s:.3g}s and was "
                               "terminated.") from exc
        compute_time = time.time() - start
        if proc.returncode != 0:
            raise RuntimeError(f"Julia failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        solve_time = _parse_julia_solve_time(proc.stdout, timing_txt) or compute_time
        if cfg.timings:
            print(f"julia_gpu timings: prep={prep_time:.3f}s julia_solve={solve_time:.3f}s "
                  f"subprocess_total={compute_time:.3f}s")
        out = np.loadtxt(out_csv, delimiter=",", dtype=np.float32)
    return out, SolverTiming(total=prep_time + solve_time, prep=prep_time, compute=solve_time)


def run_solver(name: str, cfg: FourCfg, julia_cmd: str, timeout_s: Optional[float] = None):
    if name == "gpu":
        return run_gpu_solver(cfg)
    if name == "python_cpu":
        return run_python_cpu_solver(cfg)
    if name == "python_ode_cpu":
        return run_python_ode_cpu_solver(cfg)
    if name == "qutip_cpu":
        return run_qutip_cpu_solver(cfg)
    if name == "julia_gpu":
        return run_julia_gpu_solver(cfg, julia_cmd=julia_cmd, timeout_s=timeout_s)
    raise ValueError(f"Unknown solver '{name}'")


def build_cpu_cfg(cfg: FourCfg, num_t_divider: int) -> FourCfg:
    """Use a coarser time grid for CPU reference solvers."""
    if num_t_divider <= 1:
        return cfg
    cpu_num_steps = max(1, int(round(cfg.num_steps / num_t_divider)))
    t0 = float(cfg.tlist[0]) if cfg.num_t > 0 else 0.0
    t1 = float(cfg.tlist[-1]) if cfg.num_t > 1 else t0 + cfg.dt
    cpu_tlist = np.linspace(t0, t1, cpu_num_steps + 1, dtype=np.float32)
    return replace(cfg, tlist=cpu_tlist)


def run_solver_with_status(name: str, cfg: FourCfg, julia_cmd: str,
                           cpu_cfgs: Optional[Dict[str, FourCfg]] = None):
    effective_cfg = cpu_cfgs.get(name, cfg) if cpu_cfgs is not None else cfg
    show_column_progress = bool(getattr(effective_cfg, "progress", False)) and name in {
        "python_cpu", "python_ode_cpu", "qutip_cpu",
    }

    running = (f"{name}: Running  grid={len(effective_cfg.eps_list)}x{len(effective_cfg.A_list)}  "
               f"steps={effective_cfg.num_steps}")
    if show_column_progress:
        print(running, flush=True)
    else:
        print(running, end="", flush=True)

    p, timing = run_solver(name, effective_cfg, julia_cmd)

    result = (f"{name}: total={timing.total:8.3f}s  prep={timing.prep:8.3f}s  "
              f"calc={timing.compute:8.3f}s  "
              f"grid={len(effective_cfg.eps_list)}x{len(effective_cfg.A_list)}  "
              f"steps={effective_cfg.num_steps}")
    if show_column_progress:
        print(result)
    else:
        print("\b" * len(running) + result, flush=True)
    gc.collect()
    return p, timing


def _full_benchmark_worker(result_queue, name: str, cfg: FourCfg, julia_cmd: str,
                           timeout_s: float) -> None:
    running = (f"{name}: Running  grid={len(cfg.eps_list)}x{len(cfg.A_list)}  "
               f"steps={cfg.num_steps}")
    try:
        print(running, end="", flush=True)
        p_mat, timing = run_solver(name, cfg, julia_cmd, timeout_s=timeout_s)
        result = (f"{name}: total={timing.total:8.3f}s  prep={timing.prep:8.3f}s  "
                  f"calc={timing.compute:8.3f}s  grid={len(cfg.eps_list)}x{len(cfg.A_list)}  "
                  f"steps={cfg.num_steps}")
        print("\b" * len(running) + result, flush=True)
        result_queue.put(("ok", timing))
    except BaseException as exc:
        print("\b" * len(running) + f"{name}: ERROR", flush=True)
        result_queue.put(("error", repr(exc)))


def run_full_solver_with_timeout(name: str, cfg: FourCfg, julia_cmd: str,
                                 timeout_s: float) -> SolverTiming:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_full_benchmark_worker,
                       args=(result_queue, name, cfg, julia_cmd, timeout_s))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        running = (f"{name}: Running  grid={len(cfg.eps_list)}x{len(cfg.A_list)}  "
                   f"steps={cfg.num_steps}")
        timeout_msg = (
            f"{name}: exceeded {timeout_s:.3g}s; terminating current calculation before next point."
        )
        print("\b" * len(running) + timeout_msg, flush=True)
        terminate_process_tree(proc)
        raise TimeoutError(f"{name} exceeded {timeout_s:.3g}s")
    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(f"{name} finished without returning a benchmark result.") from exc
    if status == "error":
        raise RuntimeError(payload)
    return payload


def warmup_gpu_solver_for_benchmark(cfg: FourCfg, side: int) -> float:
    """Run one unmeasured GPU solve so measured GPU points reuse compiled RHS."""
    warm_cfg = replace(cfg,
                       eps_list=np.linspace(float(cfg.eps_list[0]), float(cfg.eps_list[-1]), side,
                                           dtype=np.float32),
                       A_list=np.linspace(float(cfg.A_list[0]), float(cfg.A_list[-1]), side,
                                         dtype=np.float32), progress=False)
    print(f"gpu: warmup/precalculation  grid={len(warm_cfg.eps_list)}x{len(warm_cfg.A_list)}  "
          f"steps={warm_cfg.num_steps}")
    _p_mat, timing = run_gpu_solver(warm_cfg)
    gc.collect()
    return float(timing.rhs_stage)


def run_full_benchmark(cfg: FourCfg, *, julia_cmd: str, solvers: tuple[str, ...], min_side: int,
                       max_side: int, time_limit: float, output_filename: str, show_plot: bool,
                       python_cpu_divider: int, python_ode_cpu_divider: int,
                       qutip_cpu_divider: int) -> list[dict]:
    """Run a timing sweep over square grids and extrapolate slow backends."""
    sides = benchmark_sides(min_side, max_side)
    rows = []
    histories: dict[str, list[tuple[int, float]]] = {s: [] for s in solvers}
    stopped: dict[str, bool] = {s: False for s in solvers}
    gpu_first_rhs_stage_s = np.nan

    if "gpu" in solvers:
        gpu_first_rhs_stage_s = warmup_gpu_solver_for_benchmark(cfg, sides[0])

    for solver_name in solvers:
        print(f"\nFull benchmark solver: {solver_name}")
        for side in sides:
            num_simulations = int(side * side)
            if not stopped[solver_name] and should_extrapolate_next(
                    histories[solver_name], time_limit):
                stopped[solver_name] = True
                print(f"{solver_name}: the last timing ratio is above 3.6x per side doubling and "
                      f"the latest point exceeds half of the {time_limit:.3g}s limit; "
                      f"skipping side={side} and extrapolating.")
            if stopped[solver_name]:
                est = extrapolate_loglog(histories[solver_name], side,
                                         slope_points=BENCH_EXTRAPOLATION_POINTS)
                status = "extrapolated" if np.isfinite(est) else "failed"
                rows.append({"side_dimension": side,
                             "number_of_simulations": num_simulations,
                             "solver": solver_name, "time_s": est,
                             "prep_s": np.nan, "calc_s": np.nan, "status": status})
                if status == "extrapolated":
                    print(f"{solver_name:>10s} side={side}: extrapolated {est:.3g}s")
                else:
                    print(f"{solver_name:>10s} side={side}: FAILED "
                          "(no measured point to extrapolate from)")
                continue

            side_cfg = replace(cfg,
                               eps_list=np.linspace(float(cfg.eps_list[0]),
                                                   float(cfg.eps_list[-1]), side,
                                                   dtype=np.float32),
                               A_list=np.linspace(float(cfg.A_list[0]), float(cfg.A_list[-1]), side,
                                                 dtype=np.float32), progress=False)
            side_cpu_cfgs = {"python_cpu": build_cpu_cfg(side_cfg, python_cpu_divider),
                             "python_ode_cpu": build_cpu_cfg(side_cfg, python_ode_cpu_divider),
                             "qutip_cpu": build_cpu_cfg(side_cfg, qutip_cpu_divider)}
            try:
                effective_cfg = side_cpu_cfgs.get(solver_name, side_cfg)
                if solver_name == "gpu":
                    _p_mat, timing = run_solver_with_status(solver_name, effective_cfg, julia_cmd)
                else:
                    timing = run_full_solver_with_timeout(solver_name, effective_cfg, julia_cmd,
                                                          time_limit)
                status = "measured"
                histories[solver_name].append((side, timing.total))
                if (solver_name == "gpu" and not np.isfinite(gpu_first_rhs_stage_s)
                        and np.isfinite(timing.rhs_stage)):
                    gpu_first_rhs_stage_s = float(timing.rhs_stage)
                if timing.total >= time_limit:
                    stopped[solver_name] = True
                    print(f"{solver_name}: reached time limit after side={side}; "
                          "larger sizes will be extrapolated.")
            except Exception as exc:
                est = extrapolate_loglog(histories[solver_name], side,
                                         slope_points=BENCH_EXTRAPOLATION_POINTS)
                timing = SolverTiming(total=est, prep=np.nan, compute=np.nan)
                status = "extrapolated" if np.isfinite(est) else "failed"
                stopped[solver_name] = True
                print(f"{solver_name:>10s} side={side}: {status.upper()} ({exc})")
            rows.append({"side_dimension": side,
                         "number_of_simulations": num_simulations,
                         "solver": solver_name, "time_s": float(timing.total),
                         "prep_s": float(timing.prep), "calc_s": float(timing.compute),
                         "status": status})

    out_csv = Path(f"{output_filename}.csv")
    out_png = Path(f"{output_filename}.png")
    metadata = collect_equipment_info()
    if np.isfinite(gpu_first_rhs_stage_s):
        metadata["gpu_first_rhs_stage_s"] = f"{gpu_first_rhs_stage_s:.9g}"
    print_equipment_info(metadata)
    save_benchmark_csv(rows, out_csv, metadata=metadata)
    reference_lines = []
    if np.isfinite(gpu_first_rhs_stage_s):
        reference_lines.append({"y": gpu_first_rhs_stage_s,
                                "label": f"GPU first RHS/codegen {gpu_first_rhs_stage_s:.3g}s",
                                "color": "0.25", "linestyle": ":"})
    plot_benchmark(rows, solvers, out_png,
                   title="Calculation time scaling for different numerical approaches "
                         "(4-level system)",
                   show=show_plot, metadata=metadata, reference_lines=reference_lines)
    return rows


def normalize_mode_solver(mode: str, solver: str) -> tuple[str, str]:
    """Accept common shorthand: mode='qutip_cpu' means single qutip_cpu run."""
    if mode in {"full", "full-benchmark", "full_benchmark"}:
        return "full_benchmark", solver
    if mode in SOLVER_NAMES:
        print(f"Interpreting mode='{mode}' as mode='single', solver='{mode}'.")
        return "single", mode
    return mode, solver


def selected_solvers(mode: str, solver: str, solver_b: str) -> set[str]:
    if mode == "single":
        return {solver}
    if mode == "diff":
        return {solver, solver_b}
    if mode == "all":
        return set(SOLVER_NAMES)
    return set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Four-level interferometry benchmark.")
    parser.add_argument("mode_or_solver", nargs="?",
                        help="optional shorthand: single/all/diff or a solver name")
    parser.add_argument("--mode", default=None, help="single, all, diff, or solver shorthand")
    parser.add_argument("--solver", "--solver-a", dest="solver",
                        choices=sorted(SOLVER_NAMES), default=None,
                        help="solver for single mode or first solver for diff mode")
    parser.add_argument("--solver-b", choices=sorted(SOLVER_NAMES), default=None)
    parser.add_argument("--regime",
                        choices=["wd500", "wd1500", "drive_500_mhz", "drive_1500_mhz"],
                        default=None)
    parser.add_argument("--cpu-worker-count", "--workers", dest="workers",
                        metavar="CPU_WORKER_COUNT", type=int, default=None)
    parser.add_argument("--detuning-points", "--nx", dest="nx", metavar="DETUNING_POINTS",
                        type=int, default=None)
    parser.add_argument("--amplitude-points", "--ny", dest="ny", metavar="AMPLITUDE_POINTS",
                        type=int, default=None)
    parser.add_argument("--solver-steps", "--rk4-steps", "--num-steps", "--num-t",
                        dest="num_steps", type=int, default=None,
                        help="integration steps; --rk4-steps and --num-t are legacy aliases")
    parser.add_argument("--solver-steps-per-period", "--rk4-steps-per-period",
                        "--samples-per-period", dest="solver_steps_per_period",
                        metavar="SOLVER_STEPS_PER_PERIOD", type=int, default=None,
                        help="build total steps as simulation_periods*solver_steps_per_period")
    parser.add_argument("--cpu-density-divider", "--cpu-num-t-divider",
                        dest="cpu_num_t_divider", type=int, default=None,
                        help="common CPU-only integration/output-grid divider")
    parser.add_argument("--python-cpu-step-density-divider", "--python-cpu-num-t-divider",
                        dest="python_cpu_num_t_divider", metavar="STEP_DENSITY_DIVIDER", type=int,
                        default=None,
                        help="python_cpu uses RK4 steps/divider")
    parser.add_argument("--python-ode-output-density-divider", "--python-ode-cpu-num-t-divider",
                        dest="python_ode_cpu_num_t_divider", metavar="OUTPUT_DENSITY_DIVIDER",
                        type=int, default=None,
                        help="python_ode_cpu uses output intervals/divider for t_eval")
    parser.add_argument("--qutip-output-density-divider", "--qutip-cpu-num-t-divider",
                        dest="qutip_cpu_num_t_divider", metavar="OUTPUT_DENSITY_DIVIDER", type=int,
                        default=None,
                        help="qutip_cpu uses output intervals/divider")
    parser.add_argument("--simulation-periods", "--tr", dest="tr",
                        metavar="SIMULATION_PERIODS", type=float, default=None)
    parser.add_argument("--averaging-skip-fraction", "--warmup-time", dest="warmup_time",
                        metavar="AVERAGING_SKIP_FRACTION", type=float, default=None,
                        help="initial fraction of time excluded from the time average")
    parser.add_argument("--timings", action="store_true", help="show mesolve_2D internal timing")
    parser.add_argument("--julia-cmd", default=None, help="Julia executable name/path")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true",
                        help="disable CPU column progress display")
    parser.add_argument("--full-solvers", default=None,
                        help="comma-separated solvers for full-benchmark mode")
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

    # Regimes from Example_02_four_level_interferogram.py
    regime = settings["regime"]

    if args.mode_or_solver is not None:
        mode = args.mode_or_solver
    if args.mode is not None:
        mode = args.mode
    if args.solver is not None:
        solver = args.solver
    if args.solver_b is not None:
        solver_b = args.solver_b
    if args.regime is not None:
        regime = args.regime
    regime = {"drive_500_mhz": "wd500", "drive_1500_mhz": "wd1500"}.get(regime, regime)
    if args.julia_cmd is not None:
        julia_cmd = args.julia_cmd
    if args.full_solvers is None:
        args.full_solvers = settings.get("solvers", "gpu,qutip_cpu,julia_gpu")
    if args.bench_min_side_size is None:
        args.bench_min_side_size = settings["bench_min_side_size"]
    if args.bench_max_side_size is None:
        args.bench_max_side_size = settings["bench_max_side_size"]
    if args.bench_solver_time_limit is None:
        args.bench_solver_time_limit = settings["bench_solver_time_limit"]
    if args.output_filename is None:
        args.output_filename = settings["Output_filename"]

    # Fixed constants and selected physical regime.
    Delta = settings["Delta"]
    wr2 = settings["wr2"]
    wq1 = settings["wq1"]
    gammaph = settings["gammaph"]
    gamma1 = settings["gamma1"]
    kappa = settings["kappa"]
    Ap = settings["Ap"]
    g1 = settings["g1"]
    simulation_periods = settings["simulation_periods"]
    warmup_time = settings["averaging_skip_fraction"]
    if regime not in settings:
        raise ValueError(f"Unsupported regime '{regime}'.")
    regime_settings = settings[regime]
    w_over_Delta = regime_settings["w/Delta"]
    eps_max_over_w = regime_settings["eps_max/w"]
    A_max_over_w = regime_settings["A_max/w"]

    if args.tr is not None:
        simulation_periods = float(args.tr)
    if args.warmup_time is not None:
        warmup_time = float(args.warmup_time)

    delta_abs = wq1 * Delta
    w_abs = w_over_Delta * delta_abs
    wd_mhz = 1000.0 * w_abs / Delta
    drive_period = 2.0 * np.pi / w_abs
    gamma2 = gamma1 / 2.0 + gammaph
    Ap = Ap * wq1
    g1 = g1 * wq1
    A_max_abs = A_max_over_w * w_abs
    eps_max_abs = eps_max_over_w * w_abs

    # Benchmark grid (reduce for CPU/QuTiP practicality)
    amplitude_points = settings["amplitude_points"]
    detuning_points = settings["detuning_points"]
    solver_steps_per_period = settings["solver_steps_per_period"]
    num_steps = int(simulation_periods * solver_steps_per_period)
    python_cpu_num_t_divider = settings["python_cpu_step_density_divider"]
    python_ode_cpu_num_t_divider = settings["python_ode_output_density_divider"]
    qutip_cpu_num_t_divider = settings["qutip_output_density_divider"]

    if args.solver_steps_per_period is not None:
        solver_steps_per_period = int(args.solver_steps_per_period)
        num_steps = int(simulation_periods * solver_steps_per_period)
    if args.num_steps is not None:
        num_steps = int(args.num_steps)
    if args.nx is not None:
        detuning_points = int(args.nx)
    if args.ny is not None:
        amplitude_points = int(args.ny)
    if args.cpu_num_t_divider is not None:
        python_cpu_num_t_divider = int(args.cpu_num_t_divider)
        python_ode_cpu_num_t_divider = int(args.cpu_num_t_divider)
        qutip_cpu_num_t_divider = int(args.cpu_num_t_divider)
    if args.python_cpu_num_t_divider is not None:
        python_cpu_num_t_divider = int(args.python_cpu_num_t_divider)
    if args.python_ode_cpu_num_t_divider is not None:
        python_ode_cpu_num_t_divider = int(args.python_ode_cpu_num_t_divider)
    if args.qutip_cpu_num_t_divider is not None:
        qutip_cpu_num_t_divider = int(args.qutip_cpu_num_t_divider)

    if (python_cpu_num_t_divider <= 0 or python_ode_cpu_num_t_divider <= 0
            or qutip_cpu_num_t_divider <= 0):
        raise ValueError("CPU time-grid divider values must be positive integers.")
    mode, solver = normalize_mode_solver(mode, solver)
    active_solvers = selected_solvers(mode, solver, solver_b)
    if mode == "full_benchmark":
        active_solvers = set(parse_solver_list(args.full_solvers, SOLVER_NAMES))
    else:
        print_equipment_info()
    if "python_cpu" in active_solvers and python_cpu_num_t_divider > 1:
        print("Warning: python_cpu uses fixed-step RK4, not an adaptive solver. "
              "Large --python-cpu-step-density-divider values can reduce accuracy; "
              "use 1-2 for strict validation, or use qutip_cpu for an adaptive CPU reference.")
    if "python_ode_cpu" in active_solvers and python_ode_cpu_num_t_divider > 1:
        print("Note: python_ode_cpu uses SciPy solve_ivp adaptively, but output is "
              "sampled on the reduced tlist. "
              "Use --python-ode-output-density-divider 1 for strict validation.")
    if "qutip_cpu" in active_solvers and qutip_cpu_num_t_divider > 1:
        print("Note: qutip_cpu is adaptive internally, but time-dependent coefficient arrays "
              "are sampled on the reduced tlist. Validate accuracy when using a large divider.")

    worker_setting = settings["cpu_worker_count"]
    workers = (max(1, int(args.workers)) if args.workers is not None else
               max(1, int(worker_setting)) if worker_setting is not None else
               max(1, (os.cpu_count() or 2) - 1))

    eps_list = np.linspace(-eps_max_abs, eps_max_abs, detuning_points, dtype=np.float32)
    A_list = np.linspace(0.0, A_max_abs, amplitude_points, dtype=np.float32)
    tlist = np.linspace(0.0, simulation_periods * drive_period, num_steps + 1,
                        dtype=np.float32)
    cfg = FourCfg(delta_abs=delta_abs, w_abs=w_abs, gamma1=gamma1, gamma2=gamma2, kappa=kappa,
                  Ap=Ap, g1=g1, wr2=wr2, A_list=A_list, eps_list=eps_list, tlist=tlist,
                  warmup_time=warmup_time, workers=workers, timings=bool(args.timings),
                  regime_name=regime, progress=not args.no_progress)
    cpu_cfgs = {"python_cpu": build_cpu_cfg(cfg, python_cpu_num_t_divider),
                "python_ode_cpu": build_cpu_cfg(cfg, python_ode_cpu_num_t_divider),
                "qutip_cpu": build_cpu_cfg(cfg, qutip_cpu_num_t_divider)}

    total_steps = len(A_list) * len(eps_list) * cfg.num_steps
    print(f"regime={regime} wd_mhz={wd_mhz:.1f} "
          f"grid={len(eps_list)}x{len(A_list)} "
          f"steps={cfg.num_steps} dt={cfg.dt:.4e} workers={cfg.workers}")
    print(f"CPU time-grid dividers: python_cpu={python_cpu_num_t_divider} "
          f"(steps={cpu_cfgs['python_cpu'].num_steps}), "
          f"python_ode_cpu={python_ode_cpu_num_t_divider} "
          f"(steps={cpu_cfgs['python_ode_cpu'].num_steps}), "
          f"qutip_cpu={qutip_cpu_num_t_divider} "
          f"(steps={cpu_cfgs['qutip_cpu'].num_steps})")
    print(f"Total trajectory-step updates = {len(eps_list)}*{len(A_list)}*"
          f"{cfg.num_steps} = {total_steps:.4e}")

    results: Dict[str, np.ndarray] = {}
    times: Dict[str, SolverTiming] = {}

    if mode == "full_benchmark":
        run_full_benchmark(cfg, julia_cmd=julia_cmd,
                           solvers=parse_solver_list(args.full_solvers, SOLVER_NAMES),
                           min_side=args.bench_min_side_size,
                           max_side=args.bench_max_side_size,
                           time_limit=float(args.bench_solver_time_limit),
                           output_filename=args.output_filename, show_plot=not args.no_plot,
                           python_cpu_divider=python_cpu_num_t_divider,
                           python_ode_cpu_divider=python_ode_cpu_num_t_divider,
                           qutip_cpu_divider=qutip_cpu_num_t_divider)
        raise SystemExit(0)

    if mode == "single":
        p, t = run_solver_with_status(solver, cfg, julia_cmd, cpu_cfgs=cpu_cfgs)
        results[solver] = p
        times[solver] = t
        if not args.no_plot:
            _plot_maps_windows([(f"{solver} ({t.total:.2f}s)", p, "jet")], cfg.eps_list / wq1,
                               cfg.A_list / wq1)
        raise SystemExit(0)

    if mode == "all":
        solver_order = ("gpu", "python_cpu", "python_ode_cpu", "qutip_cpu", "julia_gpu")
        for s in solver_order:
            try:
                p, t = run_solver_with_status(s, cfg, julia_cmd, cpu_cfgs=cpu_cfgs)
                results[s] = p
                times[s] = t
            except Exception as exc:
                print(f"{s:>10s}: SKIPPED ({exc})")
        panels = [(f"{s} ({times[s].total:.2f}s)", results[s], "jet") for s in solver_order
                  if s in results]
        if panels and not args.no_plot:
            _plot_maps_windows(panels, cfg.eps_list / wq1, cfg.A_list / wq1)
        raise SystemExit(0)

    if mode != "diff":
        raise ValueError(f"Unsupported mode '{mode}'. Use mode='single', 'all', or 'diff'. "
                         "To run QuTiP use mode='single', solver='qutip_cpu'.")

    run_order = [solver, solver_b]
    run_order = sorted(run_order, key=lambda s: s == "julia_gpu")
    for s in run_order:
        p, t = run_solver_with_status(s, cfg, julia_cmd, cpu_cfgs=cpu_cfgs)
        results[s] = p
        times[s] = t

    diff = results[solver] - results[solver_b]
    mse = float(np.mean(np.abs(diff)**2))
    rms = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(diff)))
    print(f"diff({solver} - {solver_b}): MSE={mse:.6e} RMS={rms:.6e} "
          f"max_abs={max_abs:.6e}")
    if not args.no_plot:
        difference_title = (f"Difference: {solver}-{solver_b}\n"
                            f"RMS={rms:.3e}, max_abs={max_abs:.3e}")
        _plot_maps_windows([(difference_title, diff, "bwr"),
                            (f"{solver} ({times[solver].total:.2f}s)",
                             results[solver], "jet"),
                            (f"{solver_b} ({times[solver_b].total:.2f}s)",
                             results[solver_b], "jet")],
                           cfg.eps_list / wq1, cfg.A_list / wq1)


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
        "regime": "wd500",  # "wd500" or "wd1500"
        "Delta": 1.0,  # dimensionless qubit gap multiplier before conversion
        "wr2": 7.6767,  # resonator frequency in model frequency units
        "wq1": 4.71 * 1.15,  # qubit frequency scale used to convert normalized inputs
        "gammaph": 0.0e-3,  # pure-dephasing rate
        "gamma1": 2.0e-3,  # qubit relaxation rate
        "kappa": 5.0e-3,  # resonator relaxation rate
        "Ap": 0.0002 / 10.0,  # probe amplitude
        "g1": 0.04033970276 / 1.23,  # qubit-resonator coupling
        # Drive-frequency regimes.
        "wd500": {"w/Delta": 0.0923105326317733, "eps_max/w": 22.74020028,
                  "A_max/w": 27.8315904255},
        "wd1500": {"w/Delta": 0.27693159789532, "eps_max/w": 7.58006676,
                   "A_max/w": 12.0603558511},
        # Benchmark grid and time grid.
        "detuning_points": 64,
        "amplitude_points": 64,
        # Increase if mesolve_2D reports non-finite output.
        "solver_steps_per_period": 256,
        "simulation_periods": 40,  # periods of driving
        "averaging_skip_fraction": 0.0,  # initial time fraction excluded from the average
        "cpu_worker_count": None,  # None uses os.cpu_count() - 1
        # Keep fixed-step CPU RK4 on the same integration grid as GPU RK4.
        # Adaptive solvers may use a reduced requested-output/coefficient grid.
        "python_cpu_step_density_divider": 1,  # same RK4 step density as GPU
        "python_ode_output_density_divider": adaptive_cpu_output_density_divider,
        "qutip_output_density_divider": adaptive_cpu_output_density_divider,
        # Full-benchmark sweep limits.
        "bench_min_side_size": 16,  # smallest square-grid side dimension for benchmark
        "bench_max_side_size": 8192 * 4,  # biggest square-grid side dimension for benchmark
        "bench_solver_time_limit": 200.0 * 4,  # terminate calculation above this duration in seconds
        "Output_filename": "Benchmark_02_full_benchmark",  # base filename for CSV and PNG output
    }


if __name__ == "__main__":
    main()

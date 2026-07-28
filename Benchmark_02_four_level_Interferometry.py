"""Benchmark 02: four-level interferometry benchmark (GPU / Python / QuTiP / Julia).

Julia backend uses RHS generated from SymPy (same symbolic model source).
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

from Benchmark_full_tools import (
    benchmark_sides,
    collect_equipment_info,
    extrapolate_loglog,
    parse_solver_list,
    plot_benchmark,
    save_benchmark_csv,
)
from gqis import build_independent_rho, mesolve_2D


SOLVER_NAMES = {"gpu", "python_cpu", "python_ode_cpu", "qutip_cpu", "julia_gpu"}


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
    Nph: int
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
        return int(self.Nph + 2)


def _print_solver_result(name: str, timing: SolverTiming, p_mat: np.ndarray) -> None:
    print(
        f"{name:>10s}: total={timing.total:8.3f}s  "
        f"prep={timing.prep:8.3f}s  calc={timing.compute:8.3f}s  "
        f"min={np.min(p_mat):.6e} max={np.max(p_mat):.6e}  "
        f"(max-0.5)={float(np.max(p_mat) - 0.5):+.6e}"
    )


def _print_progress(label: str, done: int, total: int) -> None:
    """Lightweight progress display without adding a tqdm dependency."""
    if total <= 0:
        return
    pct = 100.0 * done / total
    print(f"\r{label}: {done}/{total} columns ({pct:5.1f}%)", end="", flush=True)
    if done >= total:
        print()


def _set_same_window_geometry(fig: plt.Figure, x: int = 80, y: int = 60, w: int = 900, h: int = 900) -> None:
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


def _plot_maps_windows(
    panels,
    eps_list: np.ndarray,
    A_list: np.ndarray,
    *,
    same_spot: bool = True,
    title_suffix: str = "",
) -> None:
    for title, p_mat, cmap in panels:
        fig, ax = plt.subplots(figsize=(8, 8))
        if same_spot:
            _set_same_window_geometry(fig)
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        im = ax.imshow(
            10*np.log(p_mat),
            aspect="auto",
            cmap=cmap,
            origin="lower",
            extent=[eps_list[0], eps_list[-1], A_list[0], A_list[-1]],
        )
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
    Ht = hc11u / sp.sqrt(cfg.delta_abs**2 + Drive**2) + hq11 * (sp.sqrt(cfg.delta_abs**2 + Drive**2) - cfg.wr2)
    H = H0 + Ht
    drive_expr = eps + A * sp.cos(cfg.w_abs * t)

    L_rel = sp.kronecker_product(sp.sqrt(cfg.gamma1) * sp_lower, qeye)
    L_phi = sp.kronecker_product(sp.sqrt(cfg.gamma2) * sz, qeye)
    L_kappa = sp.sqrt(cfg.kappa) * a1
    col_ops = [L_rel, L_phi, L_kappa]
    mean_op = a1
    return H, drive_expr, col_ops, mean_op, eps, A, t, Drive


def _build_sympy_rhs(N: int, H: sp.Matrix, col_ops, mean_op: sp.Matrix):
    rho, meta = build_independent_rho(N)
    comm = -sp.I * (H * rho - rho * H)
    lind = sp.zeros(N)
    for L in col_ops:
        Ld = L.H
        LdL = Ld * L
        lind += (L * rho * Ld) - sp.Float(0.5) * (LdL * rho + rho * LdL)
    drho_full = sp.simplify(comm + lind)
    drho_full = drho_full.subs(rho[N - 1, N - 1], 1 - sum([rho[j, j] for j in range(N - 1)]))

    drho_eqs = []
    for i in range(N - 1):
        drho_eqs.append(sp.simplify(sp.re(drho_full[i, i])))
    for i in range(N):
        for j in range(i + 1, N):
            drho_eqs.append(sp.simplify(sp.re(drho_full[i, j])))
            drho_eqs.append(sp.simplify(sp.im(drho_full[i, j])))

    obs_expr = sp.simplify(sp.re(sp.Trace(mean_op * rho)))
    return drho_eqs, obs_expr, meta["rho_syms"]


def _sympy_to_julia(expr: sp.Expr) -> str:
    try:
        from sympy.printing.julia import julia_code
        return julia_code(expr)
    except Exception:
        txt = sp.ccode(expr).replace("M_PI", "pi")
        return txt


def run_gpu_solver(cfg: FourCfg) -> Tuple[np.ndarray, SolverTiming]:
    total_start = time.time()
    prep_start = time.time()
    H, drive_expr, col_ops, mean_op, eps_sym, A_sym, _, _ = _build_four_level_symbolic(cfg)
    prep_time = time.time() - prep_start
    compute_start = time.time()
    out, timing_info = mesolve_2D(
        H,
        drive_expr,
        col_ops,
        mean_op,
        np.asarray(cfg.tlist, dtype=np.float32),
        var_arrays={
            eps_sym: np.asarray(cfg.eps_list, dtype=np.float32),
            A_sym: np.asarray(cfg.A_list, dtype=np.float32),
        },
        warmup_time=cfg.warmup_time,
        timings=cfg.timings,
        return_timing_info=True,
    )
    compute_time = time.time() - compute_start
    p = np.abs(np.real(np.asarray(out))).T
    rhs_stage = float((timing_info or {}).get("rhs_stage_s", np.nan))
    return p, SolverTiming(total=time.time() - total_start, prep=prep_time, compute=compute_time, rhs_stage=rhs_stage)


_PY_WORKER_CFG: Optional[FourCfg] = None
_PY_DRHO_FUN = None
_PY_OBS_FUN = None
_PY_RHO_LEN = None
_PY_MODE = None
_PY_PREP_TIMING_QUEUE = None


def _prepare_python_rhs(cfg: FourCfg):
    H, drive_expr, col_ops, mean_op, eps_sym, A_sym, t_sym, Drive_sym = _build_four_level_symbolic(cfg)
    H_sub = H.subs(Drive_sym, drive_expr)
    drho_eqs, obs_expr, rho_syms = _build_sympy_rhs(cfg.N, H_sub, col_ops, mean_op)
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

    sol = solve_ivp(
        rhs,
        (float(tlist[0]), float(tlist[-1])),
        u0,
        method="RK45",
        t_eval=tlist,
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success or sol.y.shape[1] == 0:
        return np.nan

    warmup = cfg.warmup_steps
    start = min(warmup + 1, sol.y.shape[1] - 1)
    vals = [
        float(_PY_OBS_FUN(float(tlist[i]), eps0, A, *sol.y[:, i]))
        for i in range(start, sol.y.shape[1])
    ]
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

    H = [
        qt.Qobj(hp),
        [qt.Qobj(hc11u), c1],
        [qt.Qobj(hq11), c2],
    ]
    c_ops = [
        np.sqrt(cfg.gamma1) * qt.Qobj(sm1),
        np.sqrt(cfg.gamma2) * qt.Qobj(sz1),
        np.sqrt(cfg.kappa) * qt.Qobj(a1),
    ]
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
    with ctx.Pool(
        processes=max(1, int(cfg.workers)),
        initializer=_init_py_worker,
        initargs=(cfg, mode, prep_timing_queue),
    ) as pool:
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
    return out.astype(np.float32, copy=False), SolverTiming(total=total_time, prep=prep_time, compute=compute_time)


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

    rhs_local_lines = [f"u{i} = u[{i+1}]" for i in range(len(rho_syms))]
    rhs_local_lines += [f"du{i+1} = Float32({_sympy_to_julia(e)})" for i, e in enumerate(drho_local)]
    rhs_local_lines += [f"obs = Float32({_sympy_to_julia(obs_local)})"]
    rhs_local_lines += ["ds = (t >= warmup_t) ? obs : 0.0f0"]
    rhs_vec = [f"du{i+1}" for i in range(len(rho_syms))] + ["ds"]

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
    {'; '.join(rhs_local_lines)}
    return @SVector [{', '.join(rhs_vec)}]
end

function main()
    out_csv = ARGS[1]
    nx = parse(Int, ARGS[2]); ny = parse(Int, ARGS[3]); num_t = parse(Int, ARGS[4])
    dt = parse(Float64, ARGS[5]); eps_min = parse(Float64, ARGS[6]); eps_max = parse(Float64, ARGS[7])
    A_min = parse(Float64, ARGS[8]); A_max = parse(Float64, ARGS[9]); warmup_steps = parse(Int, ARGS[10]); t0 = parse(Float64, ARGS[11])
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
    u0 = @SVector [{', '.join(u0_vals)}]
    tf = t0 + dt * max(num_t - 1, 0)
    prob = ODEProblem{{false}}(rhs, u0, (Float32(t0), Float32(tf)), params[1])
    prob_func = (pr, i, repeat) -> remake(pr; p=params[i])
    eprob = EnsembleProblem(prob; prob_func=prob_func, safetycopy=false)
    sol = solve(eprob, GPUTsit5(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()); trajectories=length(params), adaptive=false, dt=Float32(dt), save_everystep=false)
    out = zeros(Float32, ny, nx)
    for idx in 1:length(params)
        j = Int(fld(idx - 1, nx)) + 1
        i = Int(mod(idx - 1, nx)) + 1
        out[j, i] = abs(sol[idx][end][{len(rho_syms)+1}] / denom_t)
    end
    writedlm(out_csv, out, ',')
end
main()
"""
    path.write_text(code, encoding="utf-8")


def run_julia_gpu_solver(cfg: FourCfg, julia_cmd: str = "julia", timeout_s: Optional[float] = None) -> Tuple[np.ndarray, SolverTiming]:
    total_start = time.time()
    if shutil.which(julia_cmd) is None:
        raise RuntimeError(f"Julia executable '{julia_cmd}' was not found in PATH.")
    prep_start = time.time()
    H, drive_expr, col_ops, mean_op, _, _, _, Drive_sym = _build_four_level_symbolic(cfg)
    H_sub = H.subs(Drive_sym, drive_expr)
    drho_eqs, obs_expr, rho_syms = _build_sympy_rhs(cfg.N, H_sub, col_ops, mean_op)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        helper = td_path / "bench02_four_level_sympy_rhs.jl"
        out_csv = td_path / "julia_out.csv"
        _write_julia_helper(helper, drho_eqs, obs_expr, rho_syms)
        prep_time = time.time() - prep_start
        cmd = [
            julia_cmd,
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
        ]
        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Julia backend exceeded {timeout_s:.3g}s and was terminated.") from exc
        compute_time = time.time() - start
        if proc.returncode != 0:
            raise RuntimeError(f"Julia failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        out = np.loadtxt(out_csv, delimiter=",", dtype=np.float32)
    return out, SolverTiming(total=time.time() - total_start, prep=prep_time, compute=compute_time)


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


def run_solver_with_status(name: str, cfg: FourCfg, julia_cmd: str, cpu_cfgs: Optional[Dict[str, FourCfg]] = None):
    effective_cfg = cpu_cfgs.get(name, cfg) if cpu_cfgs is not None else cfg
    show_column_progress = bool(getattr(effective_cfg, "progress", False)) and name in {"python_cpu", "python_ode_cpu", "qutip_cpu"}

    running = (
        f"{name}: Running  grid={len(effective_cfg.eps_list)}x{len(effective_cfg.A_list)}  "
        f"samples={effective_cfg.num_t} steps={effective_cfg.num_steps}"
    )
    if show_column_progress:
        print(running, flush=True)
    else:
        print(running, end="", flush=True)

    p, timing = run_solver(name, effective_cfg, julia_cmd)

    result = (
        f"{name}: total={timing.total:8.3f}s  prep={timing.prep:8.3f}s  "
        f"calc={timing.compute:8.3f}s  grid={len(effective_cfg.eps_list)}x{len(effective_cfg.A_list)}  "
        f"samples={effective_cfg.num_t} steps={effective_cfg.num_steps}"
    )
    if show_column_progress:
        print(result)
    else:
        print("\b" * len(running) + result, flush=True)
    gc.collect()
    return p, timing


def _full_benchmark_worker(result_queue, name: str, cfg: FourCfg, julia_cmd: str, timeout_s: float) -> None:
    running = (
        f"{name}: Running  grid={len(cfg.eps_list)}x{len(cfg.A_list)}  "
        f"samples={cfg.num_t} steps={cfg.num_steps}"
    )
    try:
        print(running, end="", flush=True)
        p_mat, timing = run_solver(name, cfg, julia_cmd, timeout_s=timeout_s)
        result = (
            f"{name}: total={timing.total:8.3f}s  prep={timing.prep:8.3f}s  "
            f"calc={timing.compute:8.3f}s  grid={len(cfg.eps_list)}x{len(cfg.A_list)}  "
            f"samples={cfg.num_t} steps={cfg.num_steps}"
        )
        print("\b" * len(running) + result, flush=True)
        result_queue.put(("ok", timing))
    except BaseException as exc:
        print("\b" * len(running) + f"{name}: ERROR", flush=True)
        result_queue.put(("error", repr(exc)))


def terminate_process_tree(proc: mp.Process, timeout_s: float = 5.0) -> None:
    """Terminate a benchmark worker and its subprocesses, especially Julia on Windows."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, text=True, check=False)
    else:
        proc.terminate()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.kill()
        proc.join()


def run_full_solver_with_timeout(name: str, cfg: FourCfg, julia_cmd: str, timeout_s: float) -> SolverTiming:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_full_benchmark_worker, args=(result_queue, name, cfg, julia_cmd, timeout_s))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        running = (
            f"{name}: Running  grid={len(cfg.eps_list)}x{len(cfg.A_list)}  "
            f"samples={cfg.num_t} steps={cfg.num_steps}"
        )
        timeout_msg = f"{name}: exceeded {timeout_s:.3g}s; terminating current calculation before next point."
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
    warm_cfg = replace(
        cfg,
        eps_list=np.linspace(float(cfg.eps_list[0]), float(cfg.eps_list[-1]), side, dtype=np.float32),
        A_list=np.linspace(float(cfg.A_list[0]), float(cfg.A_list[-1]), side, dtype=np.float32),
        progress=False,
    )
    print(
        f"gpu: warmup/precalculation  grid={len(warm_cfg.eps_list)}x{len(warm_cfg.A_list)}  "
        f"samples={warm_cfg.num_t} steps={warm_cfg.num_steps}"
    )
    _p_mat, timing = run_gpu_solver(warm_cfg)
    gc.collect()
    return float(timing.rhs_stage)


def run_full_benchmark(
    cfg: FourCfg,
    *,
    julia_cmd: str,
    solvers: tuple[str, ...],
    min_side: int,
    max_side: int,
    time_limit: float,
    output_stem: str,
    show_plot: bool,
    python_cpu_divider: int,
    python_ode_cpu_divider: int,
    qutip_cpu_divider: int,
) -> list[dict]:
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
            if stopped[solver_name]:
                est = extrapolate_loglog(histories[solver_name], side)
                status = "extrapolated" if np.isfinite(est) else "failed"
                rows.append({
                    "side_dimension": side,
                    "number_of_simulations": num_simulations,
                    "solver": solver_name,
                    "time_s": est,
                    "prep_s": np.nan,
                    "calc_s": np.nan,
                    "status": status,
                })
                if status == "extrapolated":
                    print(f"{solver_name:>10s} side={side}: extrapolated {est:.3g}s")
                else:
                    print(f"{solver_name:>10s} side={side}: FAILED (no measured point to extrapolate from)")
                continue

            side_cfg = replace(
                cfg,
                eps_list=np.linspace(float(cfg.eps_list[0]), float(cfg.eps_list[-1]), side, dtype=np.float32),
                A_list=np.linspace(float(cfg.A_list[0]), float(cfg.A_list[-1]), side, dtype=np.float32),
                progress=False,
            )
            side_cpu_cfgs = {
                "python_cpu": build_cpu_cfg(side_cfg, python_cpu_divider),
                "python_ode_cpu": build_cpu_cfg(side_cfg, python_ode_cpu_divider),
                "qutip_cpu": build_cpu_cfg(side_cfg, qutip_cpu_divider),
            }
            try:
                effective_cfg = side_cpu_cfgs.get(solver_name, side_cfg)
                if solver_name == "gpu":
                    _p_mat, timing = run_solver_with_status(solver_name, effective_cfg, julia_cmd)
                else:
                    timing = run_full_solver_with_timeout(solver_name, effective_cfg, julia_cmd, time_limit)
                status = "measured"
                histories[solver_name].append((side, timing.total))
                if solver_name == "gpu" and not np.isfinite(gpu_first_rhs_stage_s) and np.isfinite(timing.rhs_stage):
                    gpu_first_rhs_stage_s = float(timing.rhs_stage)
                if timing.total >= time_limit:
                    stopped[solver_name] = True
                    print(f"{solver_name}: reached time limit after side={side}; larger sizes will be extrapolated.")
            except Exception as exc:
                est = extrapolate_loglog(histories[solver_name], side)
                timing = SolverTiming(total=est, prep=np.nan, compute=np.nan)
                status = "extrapolated" if np.isfinite(est) else "failed"
                stopped[solver_name] = True
                print(f"{solver_name:>10s} side={side}: {status.upper()} ({exc})")
            rows.append({
                "side_dimension": side,
                "number_of_simulations": num_simulations,
                "solver": solver_name,
                "time_s": float(timing.total),
                "prep_s": float(timing.prep),
                "calc_s": float(timing.compute),
                "status": status,
            })

    out_csv = Path(f"{output_stem}.csv")
    out_png = Path(f"{output_stem}.png")
    metadata = collect_equipment_info()
    if np.isfinite(gpu_first_rhs_stage_s):
        metadata["gpu_first_rhs_stage_s"] = f"{gpu_first_rhs_stage_s:.9g}"
    print(f"Benchmark equipment: CPU={metadata.get('cpu', 'unknown')} | GPU={metadata.get('gpu', 'unknown')}")
    save_benchmark_csv(rows, out_csv, metadata=metadata)
    reference_lines = []
    if np.isfinite(gpu_first_rhs_stage_s):
        reference_lines.append({
            "y": gpu_first_rhs_stage_s,
            "label": f"GPU first RHS/codegen {gpu_first_rhs_stage_s:.3g}s",
            "color": "0.25",
            "linestyle": ":",
        })
    plot_benchmark(
        rows,
        solvers,
        out_png,
        title="Calculation time scaling for different numerical approaches",
        show=show_plot,
        metadata=metadata,
        reference_lines=reference_lines,
    )
    return rows


def normalize_mode_solver(mode: str, solver: str) -> tuple[str, str]:
    """Accept common shorthand: mode='qutip_cpu' means single qutip_cpu run."""
    if mode in {"full", "full-benchmark", "full_benchmark"}:
        return "full_benchmark", solver
    if mode in SOLVER_NAMES:
        print(f"Interpreting mode='{mode}' as mode='single', solver='{mode}'.")
        return "single", mode
    return mode, solver


def selected_solvers(mode: str, solver: str, solver_a: str, solver_b: str) -> set[str]:
    if mode == "single":
        return {solver}
    if mode == "diff":
        return {solver_a, solver_b}
    if mode == "all":
        return set(SOLVER_NAMES)
    return set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Four-level interferometry benchmark.")
    parser.add_argument("mode_or_solver", nargs="?", help="optional shorthand: single/all/diff or a solver name")
    parser.add_argument("--mode", default=None, help="single, all, diff, or solver shorthand")
    parser.add_argument("--solver", choices=sorted(SOLVER_NAMES), default=None)
    parser.add_argument("--solver-a", choices=sorted(SOLVER_NAMES), default=None)
    parser.add_argument("--solver-b", choices=sorted(SOLVER_NAMES), default=None)
    parser.add_argument("--regime", choices=["wd500", "wd1500"], default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None, help="eps grid points")
    parser.add_argument("--ny", type=int, default=None, help="A grid points")
    parser.add_argument("--num-steps", "--num-t", dest="num_steps", type=int, default=None, help="RK4 integration steps; --num-t is a backward-compatible alias")
    parser.add_argument("--samples-per-period", type=int, default=None, help="build RK4 steps as periods*samples_per_period")
    parser.add_argument("--cpu-num-t-divider", type=int, default=None, help="common CPU-only integration/output-grid divider")
    parser.add_argument("--python-cpu-num-t-divider", type=int, default=None, help="python_cpu uses RK4 steps/divider")
    parser.add_argument("--python-ode-cpu-num-t-divider", type=int, default=None, help="python_ode_cpu uses output intervals/divider for t_eval")
    parser.add_argument("--qutip-cpu-num-t-divider", type=int, default=None, help="qutip_cpu uses output intervals/divider")
    parser.add_argument("--tr", type=float, default=None, help="number of drive periods")
    parser.add_argument(
        "--warmup-time",
        type=float,
        default=None,
        help="initial fraction of time excluded from the time average",
    )
    parser.add_argument("--timings", action="store_true", help="show mesolve_2D internal timing")
    parser.add_argument("--julia-cmd", default=None, help="Julia executable name/path")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="disable CPU column progress display")
    parser.add_argument("--full-solvers", default=None, help="comma-separated solvers for full-benchmark mode")
    parser.add_argument("--full-min-side", type=int, default=None, help="minimum square-grid side dimension for full-benchmark mode")
    parser.add_argument("--full-max-side", type=int, default=None, help="maximum square-grid side dimension for full-benchmark mode")
    parser.add_argument("--full-time-limit", type=float, default=None, help="terminate a measured point after this many seconds, then extrapolate larger sizes")
    parser.add_argument("--full-output-stem", default=None, help="output filename stem for CSV and PNG")
    return parser.parse_args()


def main() -> None:
        args = parse_args()
        settings = user_settings()

        # Solver mode: "single", "all", "diff"
        mode = settings["mode"]
        #Solvers list "gpu", "python_cpu", "python_ode_cpu", "qutip_cpu", "julia_gpu"
        solver = settings["solver"]
        solver_a = settings["solver_a"]
        solver_b = settings["solver_b"]
        julia_cmd = settings["julia_cmd"]

        # Regimes from Example_02_four_level_interferogram.py
        regime = settings["regime"]

        if args.mode_or_solver is not None:
            mode = args.mode_or_solver
        if args.mode is not None:
            mode = args.mode
        if args.solver is not None:
            solver = args.solver
        if args.solver_a is not None:
            solver_a = args.solver_a
        if args.solver_b is not None:
            solver_b = args.solver_b
        if args.regime is not None:
            regime = args.regime
        if args.julia_cmd is not None:
            julia_cmd = args.julia_cmd
        if args.full_solvers is None:
            args.full_solvers = settings["full_solvers"]
        if args.full_min_side is None:
            args.full_min_side = settings["full_min_side"]
        if args.full_max_side is None:
            args.full_max_side = settings["full_max_side"]
        if args.full_time_limit is None:
            args.full_time_limit = settings["full_time_limit"]
        if args.full_output_stem is None:
            args.full_output_stem = settings["full_output_stem"]

        # Fixed constants
        delta = settings["delta"]
        wr2 = settings["wr2"]
        wq1 = settings["wq1"]

        if regime == "wd500":
            wd_hz = settings["wd500"]["wd_hz"]; gammaph = settings["wd500"]["gammaph"]; gamma1 = settings["wd500"]["gamma1"]; kappa = settings["wd500"]["kappa"]; Ap = settings["wd500"]["Ap"]; g1 = settings["wd500"]["g1"]; tr = settings["wd500"]["tr"]; epsmimax = settings["wd500"]["epsmimax"]; amax_factor = settings["wd500"]["amax_factor"]; warmup_time = settings["wd500"]["warmup_time"]
        elif regime == "wd1500":
            wd_hz = settings["wd1500"]["wd_hz"]; gammaph = settings["wd1500"]["gammaph"]; gamma1 = settings["wd1500"]["gamma1"]; kappa = settings["wd1500"]["kappa"]; Ap = settings["wd1500"]["Ap"]; g1 = settings["wd1500"]["g1"]; tr = settings["wd1500"]["tr"]; epsmimax = settings["wd1500"]["epsmimax"]; amax_factor = settings["wd1500"]["amax_factor"]; warmup_time = settings["wd1500"]["warmup_time"]
        else:
            raise ValueError(f"Unsupported regime '{regime}'.")

        if args.tr is not None:
            tr = float(args.tr)
        if args.warmup_time is not None:
            warmup_time = float(args.warmup_time)

        delta_abs = wq1 * delta
        w_abs = wd_hz / 1000.0 * delta
        T_abs = 2.0 * np.pi / w_abs
        gamma2 = gamma1 / 2.0 + gammaph
        Ap = Ap * wq1
        g1 = g1 * wq1
        Amax = 2.234042553191489 * amax_factor * wq1
        epsmimax_abs = epsmimax * wq1

        # Benchmark grid (reduce for CPU/QuTiP practicality)
        Razm = settings["ny"]
        Razmeps = settings["nx"]
        samples_per_period = settings["samples_per_period"]
        num_steps = int(tr * samples_per_period)
        Nph = settings["Nph"]
        python_cpu_num_t_divider = settings["python_cpu_num_t_divider"]
        python_ode_cpu_num_t_divider = settings["python_ode_cpu_num_t_divider"]
        qutip_cpu_num_t_divider = settings["qutip_cpu_num_t_divider"]

        if args.samples_per_period is not None:
            samples_per_period = int(args.samples_per_period)
            num_steps = int(tr * samples_per_period)
        if args.num_steps is not None:
            num_steps = int(args.num_steps)
        if args.nx is not None:
            Razmeps = int(args.nx)
        if args.ny is not None:
            Razm = int(args.ny)
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

        if python_cpu_num_t_divider <= 0 or python_ode_cpu_num_t_divider <= 0 or qutip_cpu_num_t_divider <= 0:
            raise ValueError("CPU time-grid divider values must be positive integers.")
        mode, solver = normalize_mode_solver(mode, solver)
        active_solvers = selected_solvers(mode, solver, solver_a, solver_b)
        if mode == "full_benchmark":
            active_solvers = set(parse_solver_list(args.full_solvers, SOLVER_NAMES))
        if "python_cpu" in active_solvers and python_cpu_num_t_divider > 1:
            print(
                "Warning: python_cpu uses fixed-step RK4, not an adaptive solver. "
                "Large --python-cpu-num-t-divider values can cause overflow or poor accuracy; "
                "use 1-2 for strict validation, or use qutip_cpu for an adaptive CPU reference."
            )
        if "python_ode_cpu" in active_solvers and python_ode_cpu_num_t_divider > 1:
            print(
                "Note: python_ode_cpu uses SciPy solve_ivp adaptively, but output is sampled on the reduced tlist. "
                "Use --python-ode-cpu-num-t-divider 1 for strict same-output-grid validation."
            )
        if "qutip_cpu" in active_solvers and qutip_cpu_num_t_divider > 1:
            print(
                "Note: qutip_cpu is adaptive internally, but time-dependent coefficient arrays "
                "are sampled on the reduced tlist. Validate accuracy when using a large divider."
            )

        workers = (
            max(1, int(args.workers))
            if args.workers is not None
            else max(1, (os.cpu_count() or 2) - 1)
        )

        eps_list = np.linspace(-epsmimax_abs, epsmimax_abs, Razmeps, dtype=np.float32)
        A_list = np.linspace(0.0, Amax, Razm, dtype=np.float32)
        tlist = np.linspace(0.0, tr * T_abs, num_steps + 1, dtype=np.float32)
        cfg = FourCfg(
            delta_abs=delta_abs,
            w_abs=w_abs,
            gamma1=gamma1,
            gamma2=gamma2,
            kappa=kappa,
            Ap=Ap,
            g1=g1,
            wr2=wr2,
            Nph=Nph,
            A_list=A_list,
            eps_list=eps_list,
            tlist=tlist,
            warmup_time=warmup_time,
            workers=workers,
            timings=bool(args.timings),
            regime_name=regime,
            progress=not args.no_progress,
        )
        cpu_cfgs = {
            "python_cpu": build_cpu_cfg(cfg, python_cpu_num_t_divider),
            "python_ode_cpu": build_cpu_cfg(cfg, python_ode_cpu_num_t_divider),
            "qutip_cpu": build_cpu_cfg(cfg, qutip_cpu_num_t_divider),
        }

        total_steps = len(A_list) * len(eps_list) * cfg.num_steps
        print(
            f"regime={regime} wd_hz={wd_hz:.1f} grid={len(eps_list)}x{len(A_list)} "
            f"samples={cfg.num_t} steps={cfg.num_steps} dt={cfg.dt:.4e} workers={cfg.workers}"
        )
        print(
            f"CPU time-grid dividers: python_cpu={python_cpu_num_t_divider} "
            f"(samples={cpu_cfgs['python_cpu'].num_t}, steps={cpu_cfgs['python_cpu'].num_steps}), "
            f"python_ode_cpu={python_ode_cpu_num_t_divider} "
            f"(samples={cpu_cfgs['python_ode_cpu'].num_t}, steps={cpu_cfgs['python_ode_cpu'].num_steps}), "
            f"qutip_cpu={qutip_cpu_num_t_divider} "
            f"(samples={cpu_cfgs['qutip_cpu'].num_t}, steps={cpu_cfgs['qutip_cpu'].num_steps})"
        )
        print(f"Total trajectory-step updates = {len(eps_list)}*{len(A_list)}*{cfg.num_steps} = {total_steps:.4e}")

        results: Dict[str, np.ndarray] = {}
        times: Dict[str, SolverTiming] = {}

        if mode == "full_benchmark":
            run_full_benchmark(
                cfg,
                julia_cmd=julia_cmd,
                solvers=parse_solver_list(args.full_solvers, SOLVER_NAMES),
                min_side=args.full_min_side,
                max_side=args.full_max_side,
                time_limit=float(args.full_time_limit),
                output_stem=args.full_output_stem,
                show_plot=not args.no_plot,
                python_cpu_divider=python_cpu_num_t_divider,
                python_ode_cpu_divider=python_ode_cpu_num_t_divider,
                qutip_cpu_divider=qutip_cpu_num_t_divider,
            )
            raise SystemExit(0)

        if mode == "single":
            p, t = run_solver_with_status(solver, cfg, julia_cmd, cpu_cfgs=cpu_cfgs)
            results[solver] = p
            times[solver] = t
            if not args.no_plot:
                _plot_maps_windows([(f"{solver} ({t.total:.2f}s)", p, "jet")], cfg.eps_list / wq1, cfg.A_list / wq1)
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
            panels = [(f"{s} ({times[s].total:.2f}s)", results[s], "jet") for s in solver_order if s in results]
            if panels and not args.no_plot:
                _plot_maps_windows(panels, cfg.eps_list / wq1, cfg.A_list / wq1)
            raise SystemExit(0)

        if mode != "diff":
            raise ValueError(
                f"Unsupported mode '{mode}'. Use mode='single', 'all', or 'diff'. "
                "To run QuTiP use mode='single', solver='qutip_cpu'."
            )

        run_order = [solver_a, solver_b]
        run_order = sorted(run_order, key=lambda s: s == "julia_gpu")
        for s in run_order:
            p, t = run_solver_with_status(s, cfg, julia_cmd, cpu_cfgs=cpu_cfgs)
            results[s] = p
            times[s] = t

        diff = results[solver_a] - results[solver_b]
        l2 = float(np.sqrt(np.mean(np.square(diff))))
        linf = float(np.max(np.abs(diff)))
        print(f"diff({solver_a} - {solver_b}): L2={l2:.6e} Linf={linf:.6e}")
        if not args.no_plot:
            _plot_maps_windows(
                [
                    (f"Difference: {solver_a}-{solver_b}", diff, "bwr"),
                    (f"{solver_a} ({times[solver_a].total:.2f}s)", results[solver_a], "jet"),
                    (f"{solver_b} ({times[solver_b].total:.2f}s)", results[solver_b], "jet"),
                ],
                cfg.eps_list / wq1,
                cfg.A_list / wq1,
            )


def user_settings() -> dict:
    """User-editable defaults. Command-line arguments override these values."""
    adaptive_cpu_num_t_divider = 10
    common_regime = {
        "gammaph": 0.0e-3,          # pure dephasing contribution
        "gamma1": 2.0e-3,           # qubit relaxation rate
        "kappa": 5.0e-3,            # resonator relaxation rate
        "Ap": 0.0002 / 10.0,        # weak probe amplitude before wq1 scaling
        "g1": 0.04033970276 / 1.23, # qubit-resonator coupling before wq1 scaling
        "tr": 40,                   # number of driven periods to integrate
        "epsmimax": 2.09916,        # detuning half-range before wq1 scaling
        "warmup_time": 0.0,         # initial fraction of time excluded from the time average
    }
    return {
        # Solver mode: "single", "all", "diff", "full_benchmark"
        "mode": "full_benchmark", # "single", "all", "diff", or "full_benchmark"
        # Solvers list: "gpu", "python_cpu", "python_ode_cpu", "qutip_cpu", "julia_gpu"
        "solver": "python_cpu", # solver used in single mode
        "solver_a": "gpu",      # first solver used in diff mode
        "solver_b": "qutip_cpu", # second solver used in diff mode
        "julia_cmd": "julia",   # Julia executable name or full path

        # Regime selection. Use "wd500" or "wd1500".
        "regime": "wd500", # selects one regime dictionary below

        # Fixed constants.
        "delta": 1.0,       # qubit gap scale before conversion to model units
        "wr2": 7.6767,      # resonator/reference frequency in model units
        "wq1": 4.71 * 1.15, # conversion factor used for eps/A/delta scaling

        # Regime parameters.
        "wd500": {
            **common_regime,
            "wd_hz": 500.0,      # drive frequency label; w_abs = wd_hz/1000*delta
            "amax_factor": 1.15, # amplitude range multiplier
        },
        "wd1500": {
            **common_regime,
            "wd_hz": 1500.0,          # drive frequency label; w_abs = wd_hz/1000*delta
            "amax_factor": 1.15 * 1.3, # amplitude range multiplier
        },

        # Benchmark grid and time grid.
        "nx": 64, # detuning grid points
        "ny": 64, # amplitude grid points
        # Increase samples_per_period if mesolve_2D reports non-finite output.
        "samples_per_period": 60 * 4, # fixed RK4 steps per drive period
        "Nph": 2,                     # resonator truncation; total Hilbert size = Nph + 2
        # Keep fixed-step CPU RK4 on the same integration grid as GPU RK4.
        # Adaptive solvers may use a reduced requested-output/coefficient grid.
        "python_cpu_num_t_divider": 1, # same RK4 integration-step density as GPU
        "python_ode_cpu_num_t_divider": adaptive_cpu_num_t_divider, # reduced SciPy t_eval grid
        "qutip_cpu_num_t_divider": adaptive_cpu_num_t_divider,      # reduced QuTiP tlist/coefficient grid

        # Full-benchmark sweep.
        "full_solvers": "gpu,qutip_cpu,julia_gpu", # add python_ode_cpu here to include SciPy solve_ivp
        "full_min_side": 16,                       # first square-grid side dimension
        "full_max_side": 8192,                     # last square-grid side dimension
        "full_time_limit": 300.0,                  # seconds before extrapolating larger sizes
        "full_output_stem": "Benchmark_02_full_benchmark", # CSV/PNG filename stem
    }


if __name__ == "__main__":
    main()

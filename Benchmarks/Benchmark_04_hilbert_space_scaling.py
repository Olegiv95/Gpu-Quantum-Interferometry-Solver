"""Benchmark 04: calculation time versus Hilbert-space dimension.

Two model families, saved together in one CSV (``system`` column) and drawn side by side in one figure:

* ``qubit_resonator``: the Benchmark 02 qubit-resonator model with ``Nph`` resonator Fock states
  (Hilbert-space dimension ``2*Nph``).
* ``multi_qubit``: ``n`` Benchmark 01 qubits (Hilbert-space dimension ``2**n``) with nearest-neighbour
  ZZ coupling.

``calc_s`` is the GPU time of one full-grid solve; ``total_s`` is the first-call time: model
construction + symbolic ODE generation + that solver's call. ``prep_s = total_s - calc_s``.

Measured times, accuracy calibration, time limits and run options are described in BENCHMARKS.md
(sections "Benchmark 04: Hilbert-Space Scaling" and "Benchmark 04 Run Options").
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SYSTEMS = ("qubit_resonator", "multi_qubit")
TOPOLOGIES = ("chain", "square")
DRIVES = ("single", "all")
ACTIONS = ("continue", "recalculate", "full")
SYMBOLIC_PHASE = "symbolic ODE"  # worker progress label for the shared symbolic stage
# Julia DiffEqGPU (EnsembleGPUKernel, fixed-step GPUTsit5, FP32 integer-step time as in
# Benchmark 03) with the GQIS reduced RHS. Its own process generates, compiles and calibrates.
JULIA_SOLVERS = ("julia_gpu_fp32_fopt",)
ADAPTIVE_SOLVERS = ("gqis_tsit5a",)  # reuse a calibrated tolerance ("accuracy=" note) instead of steps
STEPS_REUSED_NOTE = "steps calibrated at"  # CSV note of points that reused a smaller size's steps
CALIBRATED_HERE_NOTE = "steps calibrated here (not timed)"  # CSV note of the calibration size
OVER_LIMIT_NOTE = "finished past the time limit"  # time_limit_action "finish"; blocks larger sizes
TIME_LIMIT_ACTIONS = ("abort", "finish")
CSV_COLUMNS = ("system", "size", "hilbert_dim", "ode_variables", "solver", "calc_s", "prep_s", "total_s",
               "rhs_stage_s", "steps_per_trajectory", "steps_per_period", "status", "note",
               "time_limit_s", "measured_at")
# Settings that change the measured workload. A saved CSV is reused only if they all match.
SIGNATURE_KEYS = ("system", "topology", "drive", "nx", "ny", "simulation_periods",
                  "solver_steps_per_period", "calibrate_timesteps", "averaging_skip_fraction",
                  "physics")
# Signature keys stored once per system in the CSV metadata, as "<system> <key>".
SYSTEM_KEYS = {"qubit_resonator": ("physics",), "multi_qubit": ("topology", "drive", "physics")}


# -----------------------------------------------------------------------------
# Symbolic models (built inside the worker process)
# -----------------------------------------------------------------------------

def _annihilation(n_states: int):
    import sympy as sp

    a = sp.zeros(n_states)
    for n in range(1, n_states):
        # Numeric value: an exact sp.sqrt(2) would be printed as the undefined CUDA macro M_SQRT2.
        a[n - 1, n] = float(np.sqrt(n))
    return a


def qubit_resonator_model(n_photon_states: int, physics: dict):
    """Benchmark 02 qubit-resonator model with ``n_photon_states`` Fock states (0..Nph-1)."""
    import sympy as sp

    p = _qubit_resonator_parameters(physics)
    eps, A, t, Drive = sp.symbols("eps A t Drive", real=True)
    sz = sp.Matrix([[1, 0], [0, -1]])
    s_lower = sp.Matrix([[0, 0], [1, 0]])
    cavity_eye = sp.eye(n_photon_states)
    qubit_eye = sp.eye(2)

    sm = sp.kronecker_product(s_lower, cavity_eye)
    szq = sp.kronecker_product(sz, cavity_eye)
    a = sp.kronecker_product(qubit_eye, _annihilation(n_photon_states))

    energy = sp.sqrt(p["delta_abs"]**2 + Drive**2)
    H = (p["Ap"] * (a.H + a)
         + p["g1"] * p["delta_abs"] * (sm.H * a + sm * a.H) / energy
         + szq / 2 * (energy - p["wr2"]))
    drive = eps + A * sp.cos(p["w_abs"] * t)
    col_ops = [sp.sqrt(p["gamma1"]) * sm, sp.sqrt(p["gamma2"]) * szq, sp.sqrt(p["kappa"]) * a]
    return {"H": H, "drive": drive, "col_ops": col_ops, "mean_op": a, "eps": eps, "A": A,
            "eps_max": p["eps_max"], "A_max": p["A_max"], "w": p["w_abs"]}


def _qubit_resonator_parameters(physics: dict) -> dict:
    """Benchmark 02 unit conversion (wq1 scales the normalized inputs)."""
    wq1 = float(physics["wq1"])
    delta_abs = wq1 * float(physics["Delta"])
    w_abs = float(physics["w/Delta"]) * delta_abs
    gamma1 = float(physics["gamma1"])
    return {"delta_abs": delta_abs, "w_abs": w_abs, "wr2": float(physics["wr2"]),
            "gamma1": gamma1, "gamma2": gamma1 / 2.0 + float(physics["gammaph"]),
            "kappa": float(physics["kappa"]), "Ap": float(physics["Ap"]) * wq1,
            "g1": float(physics["g1"]) * wq1,
            "eps_max": float(physics["eps_max/w"]) * w_abs,
            "A_max": float(physics["A_max/w"]) * w_abs}


def qubit_coupling_edges(n_qubits: int, topology: str) -> list[tuple[int, int]]:
    """Nearest-neighbour pairs: a line, or a row-major square mesh with ceil(sqrt(n)) columns."""
    if topology == "chain":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if topology == "square":
        cols = math.ceil(math.sqrt(n_qubits))
        edges = []
        for i in range(n_qubits):
            if (i % cols) + 1 < cols and i + 1 < n_qubits:
                edges.append((i, i + 1))
            if i + cols < n_qubits:
                edges.append((i, i + cols))
        return edges
    raise ValueError(f"Unknown topology {topology!r}; use one of {TOPOLOGIES}.")


def multi_qubit_model(n_qubits: int, physics: dict, topology: str, drive: str = "single"):
    """``n`` Benchmark 01 qubits with ZZ coupling on ``topology`` edges.

    ``drive="single"`` drives qubit 0 only; ``drive="all"`` drives every qubit with the same signal.
    """
    if drive not in DRIVES:
        raise ValueError(f"Unknown drive {drive!r}; use one of {DRIVES}.")
    import sympy as sp

    p = _multi_qubit_parameters(physics)
    eps, A, t, Drive = sp.symbols("eps A t Drive", real=True)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    sm = sp.Matrix([[0, 1], [0, 0]])
    excited = sp.Matrix([[0, 0], [0, 1]])

    def on_qubit(op, index):
        factors = [op if k == index else sp.eye(2) for k in range(n_qubits)]
        return sp.kronecker_product(*factors) if n_qubits > 1 else factors[0]

    H = sp.zeros(2**n_qubits)
    col_ops = []
    for i in range(n_qubits):
        gap = p["delta"] * (1.0 + p["gap_step"] * i)  # distinct gaps avoid exact degeneracy
        H += 0.5 * gap * on_qubit(sx, i)
        if drive == "all" or i == 0:
            H += 0.5 * Drive * on_qubit(sz, i)
        col_ops += [math.sqrt(p["gamma1"]) * on_qubit(sm, i),
                    math.sqrt(p["gamma2"]) * on_qubit(sz, i)]
    for i, j in qubit_coupling_edges(n_qubits, topology):
        H += p["J"] * on_qubit(sz, i) * on_qubit(sz, j)
    drive = eps + A * sp.sin(p["w"] * t)
    return {"H": H, "drive": drive, "col_ops": col_ops, "mean_op": on_qubit(excited, 0),
            "eps": eps, "A": A, "eps_max": p["eps_max"], "A_max": p["A_max"], "w": p["w"]}


def _multi_qubit_parameters(physics: dict) -> dict:
    """Benchmark 01 unit conversion (rates are given per drive period)."""
    delta = float(physics["Delta"])
    w = float(physics["w/Delta"]) * delta
    period = 2.0 * np.pi / w
    gamma1 = float(physics["gamma1_per_T"]) / period
    gamma_phi = float(physics["gamma_phi_per_T"]) / period
    return {"delta": delta, "w": w, "gamma1": gamma1, "gamma2": gamma1 / 2.0 + gamma_phi,
            "gap_step": float(physics["gap_step"]), "J": float(physics["J/Delta"]) * delta,
            "eps_max": float(physics["eps_max/w"]) * w, "A_max": float(physics["A_max/w"]) * w}


def hilbert_dim(system: str, size: int) -> int:
    return 2 * int(size) if system == "qubit_resonator" else 2**int(size)


def drive_frequency(system: str, physics: dict) -> float:
    if system == "qubit_resonator":
        return _qubit_resonator_parameters(physics)["w_abs"]
    return _multi_qubit_parameters(physics)["w"]


# -----------------------------------------------------------------------------
# Worker: all requested solvers of one system size in a fresh process
# -----------------------------------------------------------------------------

def run_worker(job: dict) -> None:
    """Calculate ``job["solvers"]`` one by one; append one JSON line per event to the result file."""
    import cupy as cp

    from gqis import mesolve_2D

    def report(payload: dict) -> None:
        with open(job["result_file"], "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")

    cp.zeros(1).sum()  # create the CUDA context outside the measured preparation
    cp.cuda.Device().synchronize()

    build_start = time.perf_counter()
    if job["system"] == "qubit_resonator":
        model = qubit_resonator_model(job["size"], job["physics"])
    else:
        model = multi_qubit_model(job["size"], job["physics"], job["topology"], job["drive"])
    build_s = time.perf_counter() - build_start
    period = 2.0 * np.pi / model["w"]
    steps_per_period = (job["calibration_start_steps_per_period"] if job["calibrate_timesteps"]
                        else job["solver_steps_per_period"])
    num_steps = int(round(job["simulation_periods"] * steps_per_period))
    tlist = np.linspace(0.0, job["simulation_periods"] * period, num_steps + 1, dtype=np.float32)
    sweep = {model["eps"]: np.linspace(-model["eps_max"], model["eps_max"], job["nx"],
                                       dtype=np.float32),
             model["A"]: np.linspace(0.0, model["A_max"], job["ny"], dtype=np.float32)}
    # Symbolic ODE, generated once for all solvers: a 1-point, 1-step call fills the GQIS RHS cache.
    # output_mode="final" keeps its kernel separate, so each solver below still compiles its own.
    gqis_solvers = [s for s in job["solvers"] if s not in JULIA_SOLVERS]
    symbolic_s = 0.0
    if gqis_solvers:
        report({"solver": SYMBOLIC_PHASE, "event": "start", "counted_s": build_s})
        symbolic_start = time.perf_counter()
        mesolve_2D(model["H"], model["drive"], model["col_ops"], model["mean_op"], tlist[:2],
                   var_arrays={model["eps"]: np.zeros(1, dtype=np.float32),
                               model["A"]: np.zeros(1, dtype=np.float32)},
                   output_mode="final", solver=gqis_solvers[0].removeprefix("gqis_"),
                   solver_frequency=model["w"])
        symbolic_s = time.perf_counter() - symbolic_start

    for solver in job["solvers"]:
        # counted_s: time already in this solver's total_s before its own phase starts.
        report({"solver": solver, "event": "start",
                "counted_s": build_s + (0.0 if solver in JULIA_SOLVERS else symbolic_s)})
        if solver in ADAPTIVE_SOLVERS:
            try:
                result = run_tsit5a_point(job, model, sweep, float(tlist[-1]), build_s, symbolic_s)
            except Exception as exc:  # stored in the CSV note; the next solver still runs
                result = {"error": f"{type(exc).__name__}: {exc}".replace("\n", " ")}
            report({"solver": solver, "event": "result", **result})
            continue
        if solver in JULIA_SOLVERS:
            try:
                result = run_julia_point(job, model, build_s)
            except Exception as exc:  # stored in the CSV note; the next solver still runs
                result = {"error": f"{type(exc).__name__}: {exc}".replace("\n", " ")}
            report({"solver": solver, "event": "result", **result})
            continue
        options = dict(var_arrays=sweep, warmup_time=job["averaging_skip_fraction"],
                       solver=solver.removeprefix("gqis_"), solver_frequency=model["w"],
                       return_timing_info=True)
        fixed_steps = job.get("fixed_steps", {}).get(solver)
        if fixed_steps:  # step count calibrated at this solver's smallest size
            solver_tlist = np.linspace(0.0, float(tlist[-1]), int(fixed_steps) + 1, dtype=np.float32)
        else:
            solver_tlist = tlist
            if job["calibrate_timesteps"]:
                options["calibrate_timesteps"] = job["calibrate_timesteps"]

        def solve(options, solver_tlist=solver_tlist):
            start = time.perf_counter()
            _result, info = mesolve_2D(model["H"], model["drive"], model["col_ops"],
                                       model["mean_op"], solver_tlist, **options)
            return info, time.perf_counter() - start

        recalibrated_from = None
        try:
            try:
                info, call_s = solve(options)
            except RuntimeError as exc:
                # Reused steps can be unstable for a larger system: calibrate at this size once.
                if not (fixed_steps and solver in job.get("recalibrate_on_step_error", ())
                        and "Non-finite values" in str(exc)):
                    raise
                # Reported in the CSV note and the result line (no print: it would break the
                # parent's "calculating..." status line).
                recalibrated_from, solver_tlist = int(fixed_steps), tlist
                options["calibrate_timesteps"] = job["calibrate_timesteps"]
                info, call_s = solve(options, solver_tlist)
        except Exception as exc:  # stored in the CSV note; the next solver still runs
            error = f"{type(exc).__name__}: {exc}".replace("\n", " ")
            report({"solver": solver, "event": "result", "error": error})
            continue
        # One call: calc_s is GQIS's own timing of the full-grid kernel (compiled beforehand).
        # First use with this solver: model + symbolic ODE + this call (compilation, optional
        # calibration and the solve; the symbolic ODE is a cache hit). After a recalibration,
        # the failed attempt with the reused steps is not counted.
        calc_s = float(info["gpu_kernel_s"])
        calibration = (info.get("timestep_calibration") or {}).get("details") or {}
        call_s -= float(calibration.get("calibration_s", 0.0))  # optional: not timed
        total_s = build_s + symbolic_s + call_s
        steps = int((info.get("timestep_calibration") or {}).get(
            "steps", len(solver_tlist) - 1))
        report({"solver": solver, "event": "result", "calc_s": calc_s, "total_s": total_s,
                "prep_s": total_s - calc_s, "rhs_stage_s": symbolic_s,
                "steps_per_trajectory": steps,
                "steps_per_period": steps / float(job["simulation_periods"]),
                "recalibrated_from": recalibrated_from,
                "note": "" if info.get("cached_rhs") == "hit" else "symbolic ODE regenerated"})


def _note_accuracy(row: dict) -> float:
    """Tolerance saved in a point's note as "accuracy=<value>"."""
    return float(re.search(r"accuracy=([0-9.eE+-]+)", str(row.get("note", ""))).group(1))


def run_tsit5a_point(job: dict, model: dict, sweep: dict, duration: float, build_s: float,
                     symbolic_s: float) -> dict:
    """GQIS adaptive Tsit5: the per-step tolerance is calibrated like dynamiqs Tsit5 (same
    limits on the final result) or reused from the calibration size; one timed full-grid call."""
    from gqis import mesolve_2D
    from gqis.solver import TIMESTEP_CALIBRATION_DEFAULTS
    from Benchmark_04_hilbert_space_scaling_dynamiqs import calibrate

    solver = ADAPTIVE_SOLVERS[0]
    periods = float(job["simulation_periods"])
    tlist = np.linspace(0.0, duration, int(round(periods * job["adaptive_output_points_per_period"])) + 1)
    eps, amp = sweep[model["eps"]], sweep[model["A"]]

    def solve(tol, eps_values, amp_values):
        start = time.perf_counter()
        values, info = mesolve_2D(
            model["H"], model["drive"], model["col_ops"], model["mean_op"], tlist,
            var_arrays={model["eps"]: eps_values, model["A"]: amp_values},
            warmup_time=job["averaging_skip_fraction"], solver="tsit5a", solver_frequency=model["w"],
            return_timing_info=True, ignore_non_finite_output=True,
            adaptive_options={"step_rtol": tol, "step_atol": tol, "count_steps": True,
                              "max_substeps": job["adaptive_max_substeps"]})
        return np.real(np.asarray(values)), info, time.perf_counter() - start

    def calibrated_tolerance():
        settings = {**TIMESTEP_CALIBRATION_DEFAULTS, "start_tol": 1e-2, "min_tol": 1e-9,
                    **job["calibrate_timesteps"]}
        return calibrate(model, solver, duration, eps, amp, None, {"calibrate_timesteps": settings},
                         run_grid=lambda tol, e, a: solve(tol, e, a)[0])["accuracy"]

    tol = job.get("fixed_steps", {}).get(solver)
    recalibrated_from = None
    if tol is None:  # calibration pass (not timed by the parent)
        tol = calibrated_tolerance()
    values, info, call_s = solve(tol, eps, amp)
    if (not np.all(np.isfinite(values)) and job.get("fixed_steps", {}).get(solver)
            and solver in job.get("recalibrate_on_step_error", ())):
        recalibrated_from, tol = tol, calibrated_tolerance()  # reused tolerance failed here
        values, info, call_s = solve(tol, eps, amp)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"non-finite output (max_substeps {job['adaptive_max_substeps']} reached?)")
    counts = info["adaptive_steps"]
    accepted, rejected = counts["accepted"], counts["rejected"]
    calc_s = float(info["gpu_kernel_s"])
    total_s = build_s + symbolic_s + call_s
    return {"calc_s": calc_s, "total_s": total_s, "prep_s": total_s - calc_s,
            "rhs_stage_s": symbolic_s, "steps_per_trajectory": float(accepted.mean()),
            "steps_per_period": float(accepted.mean()) / periods, "accuracy": tol,
            "recalibrated_from": recalibrated_from,
            "note": (f"accuracy={tol:.6g}; max steps {accepted.max()}; "
                     f"rejected {rejected.sum() / max(accepted.sum(), 1):.1%}")}


# -----------------------------------------------------------------------------
# Julia DiffEqGPU solver with the GQIS reduced RHS
# -----------------------------------------------------------------------------

def run_julia_point(job: dict, model: dict, build_s: float) -> dict:
    """Compile (GQIS), print, calibrate and time one DiffEqGPU point in a separate Julia process."""
    from gqis.solver import TIMESTEP_CALIBRATION_DEFAULTS
    from julia_diffeqgpu import julia_calibration_script, julia_model, run_julia_script

    solver = JULIA_SOLVERS[0]
    generation_start = time.perf_counter()
    source = julia_calibration_script(julia_model(
        model["H"], model["drive"], model["col_ops"], model["mean_op"], (model["eps"], model["A"]),
        precision="fp32_integer_step", average="complex"))
    generation_s = time.perf_counter() - generation_start

    periods = float(job["simulation_periods"])
    duration = periods * 2.0 * np.pi / model["w"]
    calibration = ({**TIMESTEP_CALIBRATION_DEFAULTS, **job["calibrate_timesteps"]}
                   if job["calibrate_timesteps"] else None)
    if calibration:
        base_steps = int(round(periods * job["calibration_start_steps_per_period"]))
        fixed = int(job.get("fixed_steps", {}).get(solver) or 0)
    else:
        base_steps = fixed = int(round(periods * job["solver_steps_per_period"]))
    settings = calibration or TIMESTEP_CALIBRATION_DEFAULTS

    def launch(fixed_steps):
        arguments = [str(job["nx"]), str(job["ny"]), repr(duration), repr(float(model["eps_max"])),
                     repr(float(model["A_max"])), str(base_steps), str(fixed_steps),
                     repr(float(settings["rms_atol"])), repr(float(settings["max_atol"])),
                     str(int(settings["sample_stride"])), str(int(settings["max_steps"])),
                     str(int(settings["bisection_steps"]))]
        return run_julia_script(source, arguments, julia_cmd=job["julia_cmd"], check=False)

    values, _, wall_s, code = launch(fixed)
    recalibrated_from = None
    if code == 3 and fixed and calibration and solver in job.get("recalibrate_on_step_error", ()):
        recalibrated_from = fixed  # reused steps unstable at this size: calibrate here
        values, _, wall_s, code = launch(0)
    if code != 0:
        raise RuntimeError(f"Julia exit code {code}: {values.get('OUTPUT_TAIL', '')}")
    calc_s, steps = float(values["CALC_S"]), int(values["STEPS"])
    # First use: model + ODE compile and Julia printing + Julia start, compilation, full solve.
    # Calibration (here only after unstable reused steps) is optional and not timed.
    total_s = build_s + generation_s + wall_s - float(values.get("CALIBRATION_S", 0.0))
    status = values.get("STATUS", "")
    return {"calc_s": calc_s, "total_s": total_s, "prep_s": total_s - calc_s,
            "rhs_stage_s": generation_s, "steps_per_trajectory": steps,
            "steps_per_period": steps / periods, "recalibrated_from": recalibrated_from,
            "note": "" if status in {"converged", "fixed"} else f"calibration {status}"}


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                       check=False)
    else:
        proc.kill()
    proc.wait()


def _read_events(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:  # a line still being written
            break
    return events


def run_size(job: dict, solvers: list[str], time_limit_s: float, on_result, on_start=None,
             finish_over_limit: bool = False, limit_on_calc: bool = False,
             prep_limit_s: float | None = None) -> None:
    """Calculate ``solvers`` of one size in one worker process; ``on_result`` stores each result.

    ``on_start`` (optional) is called with the solver name, or ``SYMBOLIC_PHASE``, when it starts.

    ``time_limit_s`` applies to each solver's first-call time ``total_s`` (time counted before its
    phase, from the start event, + its phase) and to the shared symbolic ODE generation; the
    running calculation is aborted when it is exceeded. A symbolic-stage timeout or crash ends every
    remaining solver of this size. After a solver timeout or crash, the remaining solvers continue in
    a new worker, which generates the symbolic ODE again.
    With ``finish_over_limit`` nothing is aborted: a solver whose ``total_s`` exceeds the limit is
    stored as measured with ``OVER_LIMIT_NOTE``.
    With ``limit_on_calc`` the limit applies to ``calc_s`` instead: the abort timer starts with the
    solver's own phase (model build and symbolic ODE not counted; kernel compilation still is).
    ``prep_limit_s`` (optional, also with ``finish_over_limit``) aborts a model build plus symbolic
    ODE stage that takes longer, e.g. a frozen preparation.
    """
    measured_key = "calc_s" if limit_on_calc else "total_s"
    remaining = list(solvers)
    # The worker imports the same gqis/Benchmark modules as this process.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    while remaining:
        with tempfile.TemporaryDirectory(prefix="gqis_b04_") as tmp:
            result_path = Path(tmp) / "result.jsonl"
            job_path = Path(tmp) / "job.json"
            job_path.write_text(json.dumps({**job, "solvers": remaining,
                                            "result_file": str(result_path)}), encoding="utf-8")
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker",
                                     str(job_path)], env=env)
            handled, current, current_start, counted, prep_counted = (
                0, None, time.perf_counter(), 0.0, 0.0)
            while True:
                finished = proc.poll() is not None
                events = _read_events(result_path)
                for event in events[handled:]:
                    if event["event"] == "start":
                        current, current_start = event["solver"], time.perf_counter()
                        prep_counted = float(event.get("counted_s", 0.0))
                        counted = 0.0 if limit_on_calc else prep_counted
                        if on_start is not None:
                            on_start(current)
                        continue
                    remaining.remove(event["solver"])
                    current, current_start, counted, prep_counted = None, time.perf_counter(), 0.0, 0.0
                    if "error" in event:
                        on_result(event["solver"], {"status": "failed", "note": event["error"][:300]})
                    else:
                        result = {k: v for k, v in event.items() if k not in {"solver", "event"}}
                        if float(result.get(measured_key, 0.0)) > time_limit_s:
                            result["note"] = "; ".join(text for text in (
                                f"{OVER_LIMIT_NOTE} ({time_limit_s:g} s)", result.get("note", "")) if text)
                        on_result(event["solver"], {**result, "status": "measured"})
                handled = len(events)
                if finished:
                    break
                elapsed = time.perf_counter() - current_start
                preparing = current in (None, SYMBOLIC_PHASE)  # model build or symbolic ODE
                over_prep = bool(prep_limit_s) and preparing and prep_counted + elapsed > prep_limit_s
                over_point = (not finish_over_limit and current is not None
                              and not (limit_on_calc and preparing)
                              and counted + elapsed > time_limit_s)
                if over_prep or over_point:
                    _kill_process_tree(proc)
                    if over_prep:
                        note = f"preparation stopped after {prep_limit_s:g} s"
                    else:
                        note = f"stopped after {time_limit_s:g} s"
                        if limit_on_calc:
                            note = f"calculation {note}"
                        elif current == SYMBOLIC_PHASE:
                            note = f"symbolic ODE generation {note}"
                    for solver in _phase_solvers(current, remaining):
                        remaining.remove(solver)
                        on_result(solver, {"status": "timeout", "note": note})
                    break
                time.sleep(0.5)
            if finished and remaining:
                # The worker crashed: the started solver failed; unstarted ones get a new worker.
                note = f"worker exit code {proc.returncode}"
                if current in (None, SYMBOLIC_PHASE):
                    note = f"{note} before the solvers started"
                for solver in _phase_solvers(current, remaining):
                    remaining.remove(solver)
                    on_result(solver, {"status": "failed", "note": note})


def _phase_solvers(current: str | None, remaining: list[str]) -> list[str]:
    """Solvers ended by an abort: all remaining during the shared stages, else the running one."""
    return list(remaining) if current in (None, SYMBOLIC_PHASE) else [current]


def worker_main(job_file: str) -> None:
    run_worker(json.loads(Path(job_file).read_text(encoding="utf-8")))


# -----------------------------------------------------------------------------
# CSV storage and resume logic
# -----------------------------------------------------------------------------

def output_csv_path(settings: dict) -> Path:
    stem = settings["output_name"]
    if settings["calibrate_timesteps"]:
        stem = f"{stem}_calibrated"
    path = Path(stem).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / "results" / path
    return path.with_suffix(".csv")


def _default_bisection_steps() -> int:
    from gqis.solver import TIMESTEP_CALIBRATION_DEFAULTS

    return int(TIMESTEP_CALIBRATION_DEFAULTS.get("bisection_steps", 0))


def settings_signature(settings: dict, system: str) -> dict[str, str]:
    physics = settings["physics"][system]
    values = {"system": system,
              "topology": settings["topology"] if system == "multi_qubit" else "",
              "drive": settings["drive"] if system == "multi_qubit" else "",
              "nx": settings["nx"], "ny": settings["ny"],
              "simulation_periods": settings["simulation_periods"],
              "solver_steps_per_period": ("" if settings["calibrate_timesteps"]
                                          else settings["solver_steps_per_period"]),
              # GQIS's default bisection_steps is included, so a default change is not mixed in.
              "calibrate_timesteps": (json.dumps({"start_steps_per_period":
                                                  settings["calibration_start_steps_per_period"],
                                                  "bisection_steps": _default_bisection_steps(),
                                                  **settings["calibrate_timesteps"]},
                                                 sort_keys=True)
                                      if settings["calibrate_timesteps"] else ""),
              "averaging_skip_fraction": settings["averaging_skip_fraction"],
              "physics": json.dumps(physics, sort_keys=True)}
    return {key: str(values[key]) for key in SIGNATURE_KEYS}


def signature_metadata(signature: dict[str, str]) -> dict[str, str]:
    """CSV metadata of one system's signature: shared keys as they are, own keys as "<system> <key>"."""
    system = signature["system"]
    own = SYSTEM_KEYS[system]
    return {(f"{system} {key}" if key in own else key): value for key, value in signature.items()
            if key != "system" and (key in own or key not in ("topology", "drive", "physics"))}


def system_metadata(metadata: dict[str, str], system: str) -> dict[str, str]:
    """Plain metadata of one system (shared keys + its own keys without the prefix)."""
    if metadata.get("system"):  # older single-system CSV
        return dict(metadata)
    plain = {key: value for key, value in metadata.items()
             if not any(key.startswith(f"{name} ") for name in SYSTEM_KEYS)}
    plain.update({key: metadata.get(f"{system} {key}", "") for key in SYSTEM_KEYS[system]})
    return {**plain, "system": system}


def read_scaling_csv(path: Path) -> tuple[dict[str, dict[tuple[str, int], dict]], dict[str, str]]:
    """Return ({system: {(solver, size): row}}, metadata)."""
    metadata: dict[str, str] = {}
    rows: dict[str, dict[tuple[str, int], dict]] = {}
    header = None
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for values in csv.reader(stream):
            if not values:
                continue
            if header is None and values[0].startswith("# "):
                metadata[values[0][2:]] = values[1] if len(values) > 1 else ""
                continue
            if header is None:
                header = values
                continue
            row = dict(zip(header, values))
            for key in ("size", "hilbert_dim", "ode_variables"):
                row[key] = int(row[key]) if row.get(key) else 0
            steps = float(row["steps_per_trajectory"]) if row.get("steps_per_trajectory") else 0.0
            row["steps_per_trajectory"] = int(steps) if steps.is_integer() else steps  # adaptive: mean
            for key in ("calc_s", "prep_s", "total_s", "rhs_stage_s", "steps_per_period", "time_limit_s"):
                row[key] = float(row[key]) if row.get(key) else np.nan
            row["system"] = row.get("system") or metadata.get("system", "")
            rows.setdefault(row["system"], {})[(row["solver"], row["size"])] = row
    return rows, metadata


def write_scaling_csv(path: Path, rows: dict[str, dict[tuple[str, int], dict]], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for key, value in metadata.items():
            writer.writerow([f"# {key}", value])
        writer.writerow([])
        writer.writerow(CSV_COLUMNS)
        ordered = [row for system in SYSTEMS for row in rows.get(system, {}).values()]
        for row in sorted(ordered, key=lambda r: (SYSTEMS.index(r["system"]), r["solver"], int(r["size"]))):
            cells = []
            for column in CSV_COLUMNS:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = f"{value:.9g}" if np.isfinite(value) else ""
                cells.append(str(value))
            writer.writerow(cells)
    for attempt in range(30):  # cloud sync or a CSV viewer can briefly lock the file on Windows
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(1.0)


def parse_recalculate(entries) -> list[tuple[str, str]]:
    """Entries: "gqis_lserk4:6", "*:6", "gqis_dop853:*", "failed", "timeout" or "all"."""
    parsed = []
    for entry in entries or ():
        text = str(entry).strip()
        if text in {"failed", "timeout", "all"}:
            parsed.append(("status", text))
        elif ":" in text:
            solver, size = (part.strip() for part in text.split(":", 1))
            parsed.append((solver, size))
        else:
            raise ValueError(f"Cannot read recalculate entry {entry!r}; use 'solver:size', "
                             "'*:size', 'solver:*', 'failed', 'timeout' or 'all'.")
    return parsed


def is_requested(rules: list[tuple[str, str]], solver: str, size: int, row: dict | None) -> bool:
    for kind, value in rules:
        if kind == "status":
            if value == "all" or (row is not None and row.get("status") == value):
                return True
        elif kind in {"*", solver} and value in {"*", str(size)}:
            return True
    return False


def explicit_sizes(rules: list[tuple[str, str]]) -> set[int]:
    return {int(value) for kind, value in rules if kind != "status" and value != "*"}


def blocked_by_smaller_point(rows: dict, solver: str, size: int) -> dict | None:
    """A smaller size of this solver that timed out, failed or ran over the limit blocks larger sizes."""
    for (row_solver, row_size), row in sorted(rows.items(), key=lambda item: item[0][1]):
        if row_solver == solver and row_size < size and (row["status"] in {"timeout", "failed"}
                                                         or OVER_LIMIT_NOTE in str(row.get("note", ""))):
            return row
    return None


def calibration_source_point(rows: dict, solver: str, size: int) -> dict | None:
    """Largest measured size of ``solver`` below ``size`` that was calibrated at its own size.

    Normally this is the smallest size; after a recalibration it is the recalibrated size.
    """
    candidates = [row for (row_solver, row_size), row in rows.items()
                  if row_solver == solver and row_size < size and row["status"] == "measured"
                  and float(row.get("steps_per_trajectory") or 0) > 0
                  and STEPS_REUSED_NOTE not in str(row.get("note", ""))]
    return max(candidates, key=lambda row: int(row["size"])) if candidates else None


def saved_time_limit(row: dict) -> float:
    """Time limit a point was calculated with (older rows: read from the timeout note)."""
    limit = row.get("time_limit_s", np.nan)
    if isinstance(limit, float) and np.isfinite(limit):
        return limit
    match = re.search(r"stopped after ([0-9.eE+]+) s", str(row.get("note", "")))
    return float(match.group(1)) if match else np.nan


def retry_with_looser_limit(rows: dict, solver: str, size: int, time_limit_s: float,
                           finish_over_limit: bool = False) -> bool:
    """A timed-out point is retried when the limit is now longer (or the running point is now
    finished past it) and no smaller point blocks it."""
    row = rows.get((solver, size))
    return (row is not None and row["status"] == "timeout"
            and (finish_over_limit or saved_time_limit(row) < time_limit_s)
            and blocked_by_smaller_point(rows, solver, size) is None)


# -----------------------------------------------------------------------------
# Benchmark driver
# -----------------------------------------------------------------------------

def _differing_settings(settings: dict, saved_metadata: dict, system: str, own_keys: bool) -> list[str]:
    saved = system_metadata(saved_metadata, system)
    signature = settings_signature(settings, system)
    keys = [key for key in SIGNATURE_KEYS
            if key != "system" and (own_keys or key not in SYSTEM_KEYS[system])]
    return [key for key in keys if saved.get(key, "") != signature[key]]


def open_table(settings: dict) -> dict:
    """Load the CSV shared by all systems; return {"path", "rows", "metadata"}.

    Saved points of a system are reused only if the shared settings and that system's own settings
    match. "full" restarts the systems being run; other systems' points are kept.
    """
    from Benchmark_full_tools import collect_equipment_info, print_equipment_info

    csv_path = output_csv_path(settings)
    action = settings["action"]
    if action not in ACTIONS:
        raise ValueError(f"Unknown action {action!r}; use one of {ACTIONS}.")
    rows: dict[str, dict[tuple[str, int], dict]] = {}
    saved_metadata: dict[str, str] = {}
    if csv_path.is_file():
        rows, saved_metadata = read_scaling_csv(csv_path)
        if action == "full":
            backup = csv_path.with_name(f"{csv_path.stem}_previous.csv")
            os.replace(csv_path, backup)
            print(f"Full run: previous table kept as {backup}")
            for system in list(rows):
                if system in settings["systems"] or _differing_settings(settings, saved_metadata, system, False):
                    del rows[system]
        for system in rows:
            differing = _differing_settings(settings, saved_metadata, system, system in settings["systems"])
            if differing:
                raise ValueError(f"{csv_path.name}: {system} points were measured with different settings "
                                 f"({', '.join(differing)}). Choose another output name, use action 'full' "
                                 "or restore the saved settings.")
        print(f"Loaded {sum(len(r) for r in rows.values())} saved points from {csv_path}")
    metadata = {key: value for key, value in saved_metadata.items()
                if any(key.startswith(f"{system} ") for system in rows)}
    metadata = {**collect_equipment_info(), **metadata}
    for system in settings["systems"]:
        metadata.update(signature_metadata(settings_signature(settings, system)))
    metadata.update({"solvers": ",".join(settings["solvers"]), "precision": "fp32",
                     "point_time_limit_s": f"{settings['point_time_limit_s']:g}"})
    print_equipment_info(metadata)
    return {"path": csv_path, "rows": rows, "metadata": metadata}


def run_system(settings: dict, system: str, table: dict) -> None:
    csv_path, metadata = table["path"], table["metadata"]
    rows = table["rows"].setdefault(system, {})
    action = settings["action"]

    rules = parse_recalculate(settings["recalculate"])
    sizes = sorted(set(settings["sizes"][system]) | explicit_sizes(rules))
    label = "Nph" if system == "qubit_resonator" else "qubits"
    missing = [f"{solver}:{size}" for size in sizes for solver in settings["solvers"]
               if (solver, size) not in rows]
    listed = [f"{solver}:{size}" for size in sizes for solver in settings["solvers"]
              if (solver, size) in rows and is_requested(rules, solver, size, rows[(solver, size)])]
    limit = float(settings["point_time_limit_s"])
    finish = settings["time_limit_action"] == "finish"
    on_calc = settings.get("time_limit_on", "total") == "calc"
    if settings["time_limit_action"] not in TIME_LIMIT_ACTIONS:
        raise ValueError(f"time_limit_action must be one of {TIME_LIMIT_ACTIONS}.")
    retries = [f"{solver}:{size}" for size in sizes for solver in settings["solvers"]
               if action == "continue" and retry_with_looser_limit(rows, solver, size, limit, finish)]
    print(f"Action '{action}', time limit {limit:g} s on {'calc_s' if on_calc else 'total_s'} "
          f"({settings['time_limit_action']}), preparation limit "
          f"{settings.get('preparation_time_limit_s') or 'off'} s: {len(rows)} saved points; "
          f"not calculated yet: {', '.join(missing) if missing else 'none'}; "
          f"listed for recalculation: {', '.join(listed) if listed else 'none'}; "
          f"timeouts to retry: {', '.join(retries) if retries else 'none'}")
    for size in sizes:
        dim = hilbert_dim(system, size)
        to_run = []
        pending = []  # (solver, message): printed in solver order among this size's results
        for solver in settings["solvers"]:
            saved = rows.get((solver, size))
            requested = is_requested(rules, solver, size, saved)
            tag = f"{system} {label}={size} (dim {dim}) {solver}"
            if (not requested and action == "continue"
                    and retry_with_looser_limit(rows, solver, size, limit, finish)):
                pending.append((solver, f"{tag}: retrying; timed out at {saved_time_limit(saved):g} s, "
                                        + ("now finished past the limit" if finish
                                           else f"limit is now {limit:g} s")))
                requested = True
            if saved is not None and not requested:
                pending.append((solver, f"{tag}: kept saved {saved['status']} point"))
                continue
            if saved is None and action == "recalculate" and not requested:
                pending.append((solver, f"{tag}: not calculated yet (skipped by action 'recalculate')"))
                continue
            blocker = blocked_by_smaller_point(rows, solver, size)
            if blocker is not None and not requested:
                reason = ("finished past the time limit" if blocker["status"] == "measured"
                          else f"was {blocker['status']}")
                pending.append((solver, f"{tag}: not run; {label}={blocker['size']} {reason}"))
                continue
            to_run.append(solver)

        def print_pending(up_to=None, pending=pending):
            """Print the messages of solvers listed before ``up_to`` (and its own); None: all."""
            order = list(settings["solvers"])
            last = order.index(up_to) if up_to is not None else len(order)
            while pending and order.index(pending[0][0]) <= last:
                print(pending.pop(0)[1])

        if not to_run:
            if pending:
                print(f"\n{system} {label}={size} (dim {dim}): nothing to calculate")
                print_pending()
            continue
        # Optionally reuse the step count calibrated at a smaller size of each solver.
        fixed_steps, steps_source = {}, {}
        if settings["calibrate_timesteps"] and settings["calibrate_smallest_size_only"]:
            for solver in to_run:
                source = calibration_source_point(rows, solver, size)
                if source is not None:
                    fixed_steps[solver] = (_note_accuracy(source) if solver in ADAPTIVE_SOLVERS
                                           else int(source["steps_per_trajectory"]))
                    steps_source[solver] = f"{STEPS_REUSED_NOTE} {label}={source['size']}"
        print(f"\n{system} {label}={size} (dim {dim}): calculating {', '.join(to_run)} "
              f"({settings['nx']}x{settings['ny']} grid, one symbolic ODE)")
        # Status line of the running stage ("...: calculating..."), replaced by its result.
        live = {"owner": None, "text": "", "start": 0.0}
        reported = set()  # (solver, size) results of this size already stored

        def end_live(text=None, live=live):
            """Replace the status line by ``text``; None keeps it and moves to a new line."""
            if live["owner"] is None:
                return
            print("" if text is None else "\r" + text.ljust(len(live["text"])), flush=True)
            live["owner"] = None

        def start(stage, size=size, dim=dim, live=live, to_run=to_run, reported=reported):
            prefix = f"{system} {label}={size} (dim {dim})"
            if live["owner"] == SYMBOLIC_PHASE:
                end_live(f"{prefix} symbolic ODE: {time.perf_counter() - live['start']:.4g} s")
            if stage == SYMBOLIC_PHASE:
                # Skip messages listed before the next solver of this (possibly restarted) worker.
                print_pending(next((s for s in to_run if (s, size) not in reported), None))
                text = f"{prefix} symbolic ODE: generating..."
            else:
                print_pending(stage)
                text = f"{prefix} {stage}: calculating..."
            print(text, end="", flush=True)
            live.update(owner=stage, text=text, start=time.perf_counter())

        def store(solver, result, size=size, dim=dim, steps_source=steps_source):
            source = steps_source.get(solver, "")
            if result.pop("recalibrated_from", None):
                source = (f"recalibrated: {source.removeprefix(STEPS_REUSED_NOTE + ' ')} steps "
                          "gave non-finite output")
            note = "; ".join(text for text in (source, result.pop("note", "")) if text)
            row = {"system": system, "size": int(size), "hilbert_dim": dim, "ode_variables": dim * dim - 1,
                   "solver": solver, "calc_s": np.nan, "prep_s": np.nan, "total_s": np.nan,
                   "rhs_stage_s": np.nan, "steps_per_trajectory": "", "steps_per_period": np.nan,
                   "note": note, "time_limit_s": limit,
                   "measured_at": datetime.now().isoformat(timespec="seconds"),
                   **result}
            row = {column: row.get(column, "") for column in CSV_COLUMNS}
            rows[(solver, size)] = row
            reported.add((solver, size))
            write_scaling_csv(csv_path, table["rows"], metadata)
            tag = f"{system} {label}={size} (dim {dim}) {solver}"
            if row["status"] == "measured":
                text = (f"{tag}: calc={row['calc_s']:.4g} s  total={row['total_s']:.4g} s  "
                        f"prep={row['prep_s']:.4g} s  symbolic={row['rhs_stage_s']:.4g} s  "
                        f"steps={row['steps_per_trajectory']}")
            else:
                text = f"{tag}: {row['status'].upper()} {row['note']}"
            if live["owner"] == solver:
                end_live(text)
                return
            end_live()  # e.g. a symbolic-stage timeout: keep that line, report below it
            print_pending(solver)
            print(text)

        job = {"system": system, "size": int(size), "topology": settings["topology"],
               "drive": settings["drive"],
               "physics": settings["physics"][system], "nx": settings["nx"], "ny": settings["ny"],
               "simulation_periods": settings["simulation_periods"],
               "solver_steps_per_period": settings["solver_steps_per_period"],
               "calibration_start_steps_per_period": settings["calibration_start_steps_per_period"],
               "calibrate_timesteps": settings["calibrate_timesteps"], "fixed_steps": fixed_steps,
               "recalibrate_on_step_error": list(to_run) if settings["recalibrate_on_step_error"] else [],
               "averaging_skip_fraction": settings["averaging_skip_fraction"],
               "adaptive_output_points_per_period": settings["adaptive_output_points_per_period"],
               "adaptive_max_substeps": settings["adaptive_max_substeps"],
               "julia_cmd": settings["julia_cmd"]}
        # Time-step calibration is optional, so it is not timed: solvers calibrating at this size
        # first run once only to calibrate, then are measured with the calibrated steps like any
        # larger size (fresh worker: compilation and Julia start-up stay counted).
        calibrating = [s for s in to_run if settings["calibrate_timesteps"] and s not in fixed_steps]
        if calibrating:
            print(f"{system} {label}={size} (dim {dim}): calibrating time steps of "
                  f"{', '.join(calibrating)} (not timed)", flush=True)

            def keep_steps(solver, result):
                if result["status"] == "measured":
                    fixed_steps[solver] = (result["accuracy"] if solver in ADAPTIVE_SOLVERS
                                           else int(result["steps_per_trajectory"]))
                    steps_source[solver] = CALIBRATED_HERE_NOTE
                else:
                    store(solver, {**result, "note": f"step calibration: {result.get('note', '')}"})

            run_size(job, calibrating, settings["point_time_limit_s"], keep_steps,
                     finish_over_limit=finish, limit_on_calc=on_calc,
                     prep_limit_s=settings.get("preparation_time_limit_s"))
            to_run = [s for s in to_run if s in fixed_steps]
        run_size(job, to_run, settings["point_time_limit_s"], store, start, finish_over_limit=finish,
                 limit_on_calc=on_calc, prep_limit_s=settings.get("preparation_time_limit_s"))
        end_live()
        print_pending()
    print(f"Saved Hilbert-space scaling table: {csv_path}")


# -----------------------------------------------------------------------------
# Settings and command line
# -----------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS)
    parser.add_argument("--topology", choices=TOPOLOGIES)
    parser.add_argument("--drive", choices=DRIVES, help="multi-qubit drive: qubit 0 only or all")
    parser.add_argument("--solvers", nargs="+", help="e.g. gqis_lserk4 gqis_tsit5 gqis_dop853")
    parser.add_argument("--sizes", nargs="+", type=int,
                        help="Nph values or qubit counts (applied to every selected system)")
    parser.add_argument("--action", choices=ACTIONS,
                        help="continue: missing + listed points; recalculate: listed points only; "
                             "full: new table (previous one kept as *_previous.csv)")
    parser.add_argument("--recalculate", nargs="+",
                        help="solver:size, *:size, solver:*, failed, timeout or all")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--periods", dest="simulation_periods", type=float)
    parser.add_argument("--steps-per-period", dest="solver_steps_per_period", type=int)
    parser.add_argument("--calibrate", action="store_true",
                        help="enable calibrate_timesteps with the default limits")
    parser.add_argument("--time-limit", dest="point_time_limit_s", type=float)
    parser.add_argument("--time-limit-action", dest="time_limit_action", choices=TIME_LIMIT_ACTIONS)
    parser.add_argument("--time-limit-on", dest="time_limit_on", choices=("calc", "total"))
    parser.add_argument("--prep-time-limit", dest="preparation_time_limit_s", type=float)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def apply_args(settings: dict, args: argparse.Namespace) -> dict:
    settings = dict(settings)
    for name in ("action", "topology", "drive", "nx", "ny", "simulation_periods", "solver_steps_per_period",
                 "point_time_limit_s", "time_limit_action", "time_limit_on",
                 "preparation_time_limit_s"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    if args.systems:
        settings["systems"] = tuple(args.systems)
    if args.solvers:
        settings["solvers"] = tuple(args.solvers)
    if args.sizes:
        settings["sizes"] = {system: tuple(args.sizes) for system in SYSTEMS}
    if args.recalculate:
        settings["recalculate"] = tuple(args.recalculate)
    if args.calibrate and not settings["calibrate_timesteps"]:
        settings["calibrate_timesteps"] = {"rms_atol": 1e-3, "max_atol": 1e-2, "sample_stride": 8}
    if args.no_plot:
        settings["plot_at_finish"] = False
    if args.no_show:
        settings["show_plot"] = False
    return settings


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.worker:
        worker_main(args.worker)
        return
    settings = apply_args(user_settings(), args)
    from gqis.cuda_solvers import available_solvers

    valid = {f"gqis_{name}" for name in available_solvers()} | set(JULIA_SOLVERS)
    unknown = [name for name in settings["solvers"] if name not in valid]
    if unknown:
        raise ValueError(f"Unknown solver(s) {unknown}; choose from {sorted(valid)}.")
    table = open_table(settings)
    for system in settings["systems"]:
        run_system(settings, system, table)
    if settings["plot_at_finish"]:
        from Benchmark_04_plot_from_csv import generate_plot, user_settings as plot_settings

        generate_plot({**plot_settings(), "csv_file": str(table["path"]), "show_plot": settings["show_plot"]})


def user_settings() -> dict:
    """User-editable defaults. Command-line arguments override these values."""
    # Solver options (GQIS CUDA solvers):
    # "gqis_rk4", "gqis_lserk4", "gqis_lserk4f", "gqis_dp5", "gqis_tsit5", "gqis_anas5", "gqis_ab5",
    # "gqis_alshina6", "gqis_dop853", and the adaptive "gqis_tsit5a".
    # "julia_gpu_fp32_fopt": Julia DiffEqGPU fixed-step GPUTsit5 (FP32) with the GQIS reduced RHS,
    # calibrated with the same limits; needs Julia with DiffEqGPU, CUDA and StaticArrays.

    # Saved points to calculate again (used by "continue" and "recalculate" actions).
    # Sizes are Nph or qubit counts.
    recalculate = ()
    # recalculate = ("gqis_lserk4:6", "*:8", "gqis_dop853:*", "failed", "timeout", "all")

    # Automatic time-step calibration. None uses solver_steps_per_period.
    #calibrate_timesteps = None
    calibrate_timesteps = {"rms_atol": 1e-3, "max_atol": 1e-2, "sample_stride": 8}

    return {
        # "continue": keep saved points, calculate the missing ones and those in recalculate.
        # "recalculate": calculate only the points in recalculate.
        # "full": start a new table; the previous one is kept as <name>_previous.csv.
        "action": "continue", #"continue", "recalculate", "full"
        "systems": ("qubit_resonator", "multi_qubit"),  # saved in one CSV, one panel each
        "topology": "chain",  # multi-qubit coupling: "chain" or "square" (ceil(sqrt(n)) columns)
        "drive": "single",  # multi-qubit drive: "single" (qubit 0 only) or "all" (every qubit)
        "sizes": {"qubit_resonator": (2, 3, 4, 5, 6, 7, 8, 9, 10),  # Nph; Hilbert-space dimension 2*Nph
                  "multi_qubit": (1, 2, 3, 4, 5)},  # qubit count n; Hilbert-space dimension 2**n
        "solvers": ("gqis_lserk4", "gqis_lserk4f", "gqis_tsit5", "gqis_dop853", "julia_gpu_fp32_fopt"),
        "recalculate": recalculate,
        # Parameter grid (eps x A) and time grid, identical for every system size.
        "nx": 1024,  # detuning points
        "ny": 1024,  # amplitude points
        "simulation_periods": 40,  # periods of driving
        "solver_steps_per_period": 128, 
        "averaging_skip_fraction": 0.0,  # initial time fraction excluded from the average
        "calibrate_timesteps": calibrate_timesteps,
        "calibrate_smallest_size_only": True,  # larger sizes reuse each solver's last calibrated steps
        "recalibrate_on_step_error": True,  # NaN/Inf with reused steps: calibrate at that size, retry once
        "calibration_start_steps_per_period": 32,  # starting grid of the calibration search
        "point_time_limit_s": 900.0,  # larger sizes are then not started
        "time_limit_on": "calc",  # "calc": calc_s (compilation included); "total": total_s
        "preparation_time_limit_s": 1800.0,  # model build + symbolic ODE (frozen-stage guard); None: off
        # "abort": stop the point at the limit; "finish": let it finish (noted in the CSV) and
        # retry saved timeouts with "continue". Either way larger sizes are not started.
        "time_limit_action": "abort",
        # gqis_tsit5a (adaptive): tlist output points per period (steps adapt between them, the
        # average integrates over the steps) and the step cap per output interval.
        "adaptive_output_points_per_period": 4,
        "adaptive_max_substeps": 128,
        "julia_cmd": "julia",  # Julia executable for julia_gpu_fp32_fopt
        # Physics parameters. qubit_resonator follows Benchmark 02 (regime wd500).
        "physics": {
            "qubit_resonator": {
                "Delta": 1.0,  # dimensionless qubit gap multiplier before conversion
                "wq1": 4.71 * 1.15,  # qubit frequency scale used to convert normalized inputs
                "wr2": 7.6767,  # resonator frequency in model frequency units
                "gamma1": 2.0e-3,  # qubit relaxation rate
                "gammaph": 0.0,  # pure-dephasing rate
                "kappa": 5.0e-3,  # resonator relaxation rate
                "Ap": 0.0002 / 10.0,  # probe amplitude
                "g1": 0.04033970276 / 1.23,  # qubit-resonator coupling
                "w/Delta": 0.0923105326317733,  # drive frequency divided by the gap
                "eps_max/w": 22.74020028,  # maximum absolute detuning divided by w
                "A_max/w": 27.8315904255,  # maximum drive amplitude divided by w
            },
            # multi_qubit follows Benchmark 01.
            "multi_qubit": {
                "Delta": 1.0,  # minimum energy gap of qubit 0
                "w/Delta": 1.14,  # drive angular frequency divided by Delta
                "gamma1_per_T": 0.05,  # relaxation accumulated per drive period
                "gamma_phi_per_T": 0.04,  # pure dephasing accumulated per drive period
                "eps_max/w": 10.0,  # maximum absolute detuning divided by w
                "A_max/w": 10.0,  # maximum drive amplitude divided by w
                "gap_step": 0.05,  # qubit i has gap Delta*(1 + gap_step*i)
                "J/Delta": 0.1,  # nearest-neighbour coupling J*sz_i*sz_j
            },
        },
        # One CSV for all systems; relative names are saved in Benchmarks/results/,
        # calibrated runs add "_calibrated".
        "output_name": "Benchmark_04_hilbert_space_scaling",
        "plot_at_finish": True,  # draw the figure with Benchmark_04_plot_from_csv.py
        "show_plot": True,
    }


if __name__ == "__main__":
    main()

"""dynamiqs counterpart of Benchmark 04 (calculation time versus Hilbert-space dimension).

Runs in WSL (JAX with CUDA does not run on native Windows), from the Benchmarks folder, e.g.:

    ~/dynamiqs-venv/bin/python Benchmark_04_hilbert_space_scaling_dynamiqs.py

The models, the parameter grid, the time grid, FP32 precision, the accuracy check and the CSV
format follow Benchmark_04_hilbert_space_scaling.py, but everything is built with dynamiqs alone
(dq.destroy, dq.tensor, dq.modulated, dq.mesolve); GQIS is not used. Both systems are saved in one
CSV (``system`` column), which ``Benchmark_04_plot_from_csv.py`` adds to the Benchmark 04 figure.

Output: the time average of <mean_op> over (0, T] at every (eps, A) grid point, like GQIS
output_mode="mean" (GQIS averages after every step; here the average is taken over tsave).

Solvers:
* ``dynamiqs_tsit5``: adaptive Tsit5 (rtol = atol = tol). The tolerance is calibrated like the
  GQIS time step: tol is divided by 10 until the result changes less than the limits on the
  undersampled grid, then the bracket is narrowed by geometric bisection. The average is taken over
  ``tsave_per_period`` dense-output samples per drive period.
* ``dynamiqs_rouchon3``: fixed-step third-order Rouchon (dynamiqs's fixed-step mesolve method),
  calibrated exactly like the GQIS step count (doubling + bisection, Richardson-scaled change). It
  saves after every step, because its dense output between steps is only linear.

Each point (solver, size) runs in a fresh worker process: JIT compilation on the first chunk of the
grid (preparation), then one timed full sweep (``calc_s``). Tolerance calibration is optional and
not timed: at the calibration size a separate worker calibrates first. A point whose
first-call time ``total_s`` exceeds ``point_time_limit_s`` is stopped and blocks larger sizes of
that solver.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from Benchmark_04_hilbert_space_scaling import (read_scaling_csv, signature_metadata, system_metadata,
                                                write_scaling_csv)

# Allocate GPU memory on demand: Windows already holds part of the VRAM, so JAX's default
# preallocation of 75% fails and falls back with errors.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
SOLVERS = ("dynamiqs_tsit5", "dynamiqs_rouchon3")
SIGNATURE_KEYS = ("system", "topology", "drive", "nx", "ny", "simulation_periods",
                  "calibrate_timesteps", "tsave_per_period", "physics", "layout", "matmul_precision")
REUSED_NOTE = "calibrated at"  # note of points that reused a smaller size's steps or tolerance
CALIBRATED_HERE_NOTE = "tolerance calibrated here (not timed)"  # note of the calibration size
OVER_LIMIT_NOTE = "finished past the time limit"  # time_limit_action "finish"; blocks larger sizes


# -----------------------------------------------------------------------------
# Models (Benchmark 04 physics, built with dynamiqs operators)
# -----------------------------------------------------------------------------

def qubit_resonator_parameters(physics: dict) -> dict:
    """Benchmark 02 unit conversion (wq1 scales the normalized inputs)."""
    wq1 = float(physics["wq1"])
    delta = wq1 * float(physics["Delta"])
    w = float(physics["w/Delta"]) * delta
    gamma1 = float(physics["gamma1"])
    return {"delta": delta, "w": w, "wr2": float(physics["wr2"]), "gamma1": gamma1,
            "gamma2": gamma1 / 2.0 + float(physics["gammaph"]), "kappa": float(physics["kappa"]),
            "Ap": float(physics["Ap"]) * wq1, "g1": float(physics["g1"]) * wq1,
            "eps_max": float(physics["eps_max/w"]) * w, "A_max": float(physics["A_max/w"]) * w}


def multi_qubit_parameters(physics: dict) -> dict:
    """Benchmark 01 unit conversion (rates are given per drive period)."""
    delta = float(physics["Delta"])
    w = float(physics["w/Delta"]) * delta
    period = 2.0 * np.pi / w
    gamma1 = float(physics["gamma1_per_T"]) / period
    return {"delta": delta, "w": w, "gamma1": gamma1,
            "gamma2": gamma1 / 2.0 + float(physics["gamma_phi_per_T"]) / period,
            "gap_step": float(physics["gap_step"]), "J": float(physics["J/Delta"]) * delta,
            "eps_max": float(physics["eps_max/w"]) * w, "A_max": float(physics["A_max/w"]) * w}


def coupling_edges(n_qubits: int, topology: str) -> list[tuple[int, int]]:
    if topology == "chain":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    cols = math.ceil(math.sqrt(n_qubits))
    edges = []
    for i in range(n_qubits):
        if (i % cols) + 1 < cols and i + 1 < n_qubits:
            edges.append((i, i + 1))
        if i + cols < n_qubits:
            edges.append((i, i + cols))
    return edges


def hilbert_dim(system: str, size: int) -> int:
    return 2 * int(size) if system == "qubit_resonator" else 2**int(size)


# Time-dependent prefactors. Arrays are bound with jax.tree_util.Partial, so a new chunk of grid
# points does not trigger a new JIT compilation.
def _qr_inverse_energy(eps, A, w, delta, scale, t):
    import jax.numpy as jnp
    drive = eps + A * jnp.cos(w * t)
    return scale / jnp.sqrt(delta**2 + drive**2)


def _qr_energy(eps, A, w, delta, scale, t):
    import jax.numpy as jnp
    drive = eps + A * jnp.cos(w * t)
    return scale * jnp.sqrt(delta**2 + drive**2)


def _mq_drive(eps, A, w, delta, scale, t):
    import jax.numpy as jnp
    return scale * (eps + A * jnp.sin(w * t))


def build_model(system: str, size: int, physics: dict, topology: str, drive: str, layout) -> dict:
    """Static Hamiltonian, modulated terms, jump operators, initial state and averaged operator."""
    import dynamiqs as dq
    import numpy as np

    def op(matrix):
        return dq.asqarray(np.asarray(matrix, dtype=np.complex64), layout=layout)

    sz, sx = op([[1, 0], [0, -1]]), op([[0, 1], [1, 0]])
    if system == "qubit_resonator":
        # Basis order and operators as in Benchmark 04: qubit (x) cavity, s_lower = [[0,0],[1,0]].
        p = qubit_resonator_parameters(physics)
        nph = int(size)
        s_lower = op([[0, 0], [1, 0]])
        a = dq.tensor(dq.eye(2, layout=layout), dq.destroy(nph, layout=layout))
        sm = dq.tensor(s_lower, dq.eye(nph, layout=layout))
        szq = dq.tensor(sz, dq.eye(nph, layout=layout))
        H0 = p["Ap"] * (a + a.dag()) - 0.5 * p["wr2"] * szq
        # g1*delta/E(t) (sm^+ a + sm a^+) + E(t)/2 sz,  E = sqrt(delta^2 + (eps + A cos(w t))^2)
        terms = [(_qr_inverse_energy, p["g1"] * p["delta"], sm.dag() @ a + sm @ a.dag()),
                 (_qr_energy, 0.5, szq)]
        jumps = [math.sqrt(p["gamma1"]) * sm, math.sqrt(p["gamma2"]) * szq,
                 math.sqrt(p["kappa"]) * a]
        mean_op = a
        rho0 = dq.tensor(dq.basis_dm(2, 0), dq.basis_dm(nph, 0))
    else:
        p = multi_qubit_parameters(physics)
        n = int(size)
        sm, excited = op([[0, 1], [0, 0]]), op([[0, 0], [0, 1]])

        def on_qubit(single, index):
            factors = [single if k == index else dq.eye(2, layout=layout) for k in range(n)]
            return dq.tensor(*factors) if n > 1 else factors[0]

        H0 = sum(0.5 * p["delta"] * (1.0 + p["gap_step"] * i) * on_qubit(sx, i) for i in range(n))
        for i, j in coupling_edges(n, topology):
            H0 = H0 + p["J"] * (on_qubit(sz, i) @ on_qubit(sz, j))
        driven = range(n) if drive == "all" else (0,)
        terms = [(_mq_drive, 0.5, sum(on_qubit(sz, i) for i in driven))]
        jumps = []
        for i in range(n):
            jumps += [math.sqrt(p["gamma1"]) * on_qubit(sm, i),
                      math.sqrt(p["gamma2"]) * on_qubit(sz, i)]
        mean_op = on_qubit(excited, 0)
        rho0 = dq.tensor(*[dq.basis_dm(2, 0) for _ in range(n)]) if n > 1 else dq.basis_dm(2, 0)
    return {"H0": H0, "terms": terms, "jumps": jumps, "mean_op": mean_op, "rho0": rho0,
            "w": p["w"], "delta": p["delta"], "eps_max": p["eps_max"], "A_max": p["A_max"]}


def hamiltonian(model: dict, eps, A):
    """Batched time-dependent Hamiltonian (batch shape of eps and A)."""
    import dynamiqs as dq
    from jax.tree_util import Partial

    H = model["H0"]
    for function, scale, operator in model["terms"]:
        H = H + dq.modulated(Partial(function, eps, A, model["w"], model["delta"], scale), operator)
    return H


# -----------------------------------------------------------------------------
# Worker: one (solver, size) point
# -----------------------------------------------------------------------------

_RUNNERS = {}  # one jitted runner per (model, solver, accuracy): JAX caches compiled code per function


def make_runner(model: dict, solver: str, accuracy: float, duration: float, job: dict):
    """Jitted function: (eps, A) chunk -> (time-averaged <mean_op>, steps per trajectory)."""
    key = (id(model), solver, float(accuracy))
    if key not in _RUNNERS:
        _RUNNERS[key] = _new_runner(model, solver, accuracy, duration, job)
    return _RUNNERS[key]


def _new_runner(model: dict, solver: str, accuracy: float, duration: float, job: dict):
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp

    if solver == "dynamiqs_tsit5":
        method = dq.method.Tsit5(rtol=accuracy, atol=accuracy, max_steps=job["adaptive_max_steps"])
        nsave = int(round(job["simulation_periods"] * job["tsave_per_period"]))
    else:  # fixed step; accuracy is the step count
        method = dq.method.Rouchon3(dt=duration / accuracy)
        nsave = int(accuracy)
    tsave = jnp.asarray(np.arange(1, nsave + 1) * (duration / nsave), dtype=jnp.float32)

    @jax.jit
    def run(eps, A):
        result = dq.mesolve(hamiltonian(model, eps, A), model["jumps"], model["rho0"], tsave,
                            exp_ops=[model["mean_op"]], method=method, save_states=False,
                            cartesian_batching=False, progress_meter=False, t0=0.0,
                            vectorized=job["vectorized"])
        return jnp.mean(result.expects[..., 0, :], axis=-1), result.infos.nsteps

    return run, nsave


def chunk_size(dim: int, nsave: int, job: dict) -> int:
    # complex64 bytes; saved expectation values take ~4x their size in dynamiqs/diffrax buffers
    # (measured: 5 GB requested for 32768 points x 6089 saves).
    per_point = 8.0 * (dim * dim * job["state_copies"] + 4 * nsave)
    size = int(job["memory_budget_gb"] * 1e9 / per_point)
    return max(64, 2 ** int(math.floor(math.log2(max(size, 1)))))


def solve_grid(model, solver, accuracy, duration, eps_values, A_values, dim, job, first_chunk_only=False):
    """Averaged result on the eps x A grid and mean steps per trajectory.

    ``first_chunk_only`` runs one chunk only (compiles the runner for the full-grid chunk shape;
    the returned values are then not used)."""
    import jax.numpy as jnp

    run, nsave = make_runner(model, solver, accuracy, duration, job)
    eps_grid, A_grid = np.meshgrid(eps_values, A_values, indexing="ij")
    eps_flat, A_flat = eps_grid.ravel(), A_grid.ravel()
    total = eps_flat.size
    chunk = min(chunk_size(dim, nsave, job), 2 ** int(math.ceil(math.log2(total))))
    padded = int(math.ceil(total / chunk)) * chunk
    eps_flat = np.concatenate([eps_flat, np.full(padded - total, eps_flat[-1])]).astype(np.float32)
    A_flat = np.concatenate([A_flat, np.full(padded - total, A_flat[-1])]).astype(np.float32)
    results, steps = [], []
    for start in range(0, padded, chunk):
        mean, nsteps = run(jnp.asarray(eps_flat[start:start + chunk]),
                           jnp.asarray(A_flat[start:start + chunk]))
        results.append(np.asarray(mean))
        steps.append(np.broadcast_to(np.asarray(nsteps), (chunk,)))
        if first_chunk_only:
            return None, None
    result = np.concatenate(results)[:total].reshape(eps_grid.shape)
    steps = np.concatenate(steps)[:total].astype(np.float64)
    return result, steps


def test_axis(values: np.ndarray, stride: int) -> np.ndarray:
    picked = list(range(0, len(values), stride))
    return values[picked if picked[-1] == len(values) - 1 else picked + [len(values) - 1]]


def calibrate(model, solver, duration, eps_values, A_values, dim, job, run_grid=None) -> dict:
    """Coarsest step count (fixed step) or loosest tolerance (Tsit5) meeting the limits.

    Mirrors the GQIS calibrate_timesteps search on the undersampled grid with the "resolution"
    x = steps (refined by 2, Richardson order of the method) or x = 1/tol (refined by 10, error
    proportional to tol): refine until the Richardson-scaled change passes, then narrow the bracket
    with ``bisection_steps`` geometric bisections against the finest result computed so far.
    The rounding-floor stop is only accepted once the change is below 2x rms_atol (see below).
    ``run_grid(accuracy, eps, A)`` (optional) replaces the dynamiqs solve, e.g. for GQIS tsit5a.
    """
    settings = job["calibrate_timesteps"]
    adaptive = solver in ("dynamiqs_tsit5", "gqis_tsit5a")
    factor, order = (10.0, 1) if adaptive else (2.0, 3)
    eps_test, A_test = test_axis(eps_values, settings["sample_stride"]), test_axis(A_values, settings["sample_stride"])

    def accuracy(x):
        return 1.0 / x if adaptive else int(round(x))

    def run(x):
        if run_grid is not None:
            return run_grid(accuracy(x), eps_test, A_test)
        return solve_grid(model, solver, accuracy(x), duration, eps_test, A_test, dim, job)[0]

    def check(x, coarse, fine_x, fine):
        record = {"accuracy": accuracy(x), "rms": math.inf, "max": math.inf}
        if not (np.all(np.isfinite(coarse)) and np.all(np.isfinite(fine))):
            return record, False
        difference = np.abs(fine - coarse) / (1.0 - (x / fine_x) ** order)
        record.update(rms=float(np.sqrt(np.mean(difference**2))), max=float(np.max(difference)))
        return record, record["rms"] <= settings["rms_atol"] and record["max"] <= settings["max_atol"]

    if adaptive:
        x = 1.0 / settings["start_tol"]
        x_limit = 1.0 / settings["min_tol"]
    else:
        x = float(settings["start_steps_per_period"] * job["simulation_periods"])
        x_limit = float(job["fixed_max_steps"])
    base, coarse = x, run(x)
    tested, status, accepted = [], None, None
    while status is None:
        if factor * x > x_limit:
            status, accepted = "limit", x
            break
        fine = run(factor * x)
        record, passed = check(x, coarse, factor * x, fine)
        tested.append(record)
        print(f"    calibration {solver}: {record}", flush=True)
        if passed:
            status, accepted = "converged", x
            reference_x, reference = factor * x, fine
        elif (len(tested) > 1 and np.isfinite(record["rms"]) and record["rms"] >= tested[-2]["rms"]
              and record["rms"] < 2.0 * settings["rms_atol"]):
            # Rounding floor. Unlike GQIS, it is only accepted near the limit: Rouchon is
            # unconditionally stable, so on coarse grids its change can still grow before converging.
            best = min(tested, key=lambda item: item["rms"])
            status = "precision_floor"
            accepted = 1.0 / best["accuracy"] if adaptive else float(best["accuracy"])
        x, coarse = factor * x, fine

    if status == "converged" and settings["bisection_steps"] > 0:
        def passes(candidate):
            record, passed = check(candidate, run(candidate), reference_x, reference)
            tested.append(record)
            print(f"    calibration {solver}: {record}", flush=True)
            return passed

        failing = accepted / factor if accepted > base else None
        halvings = 0
        while failing is None and halvings < 6 and (adaptive or accepted >= 2):
            halvings += 1  # the starting value passed: look for a failing coarser one
            candidate = accepted / factor
            if passes(candidate):
                accepted = candidate
            else:
                failing = candidate
        for _ in range(settings["bisection_steps"] if failing is not None else 0):
            candidate = math.sqrt(failing * accepted)
            if not adaptive:
                candidate = float(int(round(candidate)))
                if candidate <= failing or candidate >= accepted:
                    break
            if passes(candidate):
                accepted = candidate
            else:
                failing = candidate
    if status != "converged":
        raise RuntimeError(f"calibration did not converge ({status}); tested {tested}")
    return {"accuracy": accuracy(accepted), "tested": tested}


def run_worker(job: dict) -> None:
    def report(payload):
        with open(job["result_file"], "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")

    import dynamiqs as dq
    import jax
    import jax.numpy as jnp

    dq.set_precision("single")
    dq.set_matmul_precision(job["matmul_precision"])
    jnp.zeros(1).block_until_ready()  # CUDA context outside the measured preparation
    report({"event": "start"})
    solver, size, dim = job["solver"], job["size"], hilbert_dim(job["system"], job["size"])
    try:
        build_start = time.perf_counter()
        layout = dq.dia if job["layout"] == "dia" else dq.dense
        model = build_model(job["system"], size, job["physics"], job["topology"], job["drive"], layout)
        build_s = time.perf_counter() - build_start
        duration = job["simulation_periods"] * 2.0 * np.pi / model["w"]
        eps_values = np.linspace(-model["eps_max"], model["eps_max"], job["nx"], dtype=np.float32)
        A_values = np.linspace(0.0, model["A_max"], job["ny"], dtype=np.float32)

        def full(accuracy):
            """Compile on the first chunk (preparation), then one timed full-grid sweep (calc)."""
            solve_grid(model, solver, accuracy, duration, eps_values, A_values, dim, job,
                       first_chunk_only=True)
            start = time.perf_counter()
            result, steps = solve_grid(model, solver, accuracy, duration, eps_values, A_values, dim, job)
            return result, steps, start, time.perf_counter() - start

        prep_start = time.perf_counter()
        accuracy, recalibrated_from, calibrated = job.get("reuse_accuracy"), None, False
        if accuracy is None:
            accuracy = calibrate(model, solver, duration, eps_values, A_values, dim, job)["accuracy"]
            calibrated = True
            if job.get("calibrate_only"):  # calibration pass (not timed); measured in a new worker
                report({"event": "result", "accuracy": accuracy, "calibrated": True})
                return
            prep_start = time.perf_counter()  # calibration is optional: not counted
        result, steps, calc_start, calc_s = full(accuracy)
        if not np.all(np.isfinite(result)) and not calibrated:
            # Reused steps or tolerance can be unstable for a larger system: calibrate here once.
            recalibrated_from = accuracy
            accuracy = calibrate(model, solver, duration, eps_values, A_values, dim, job)["accuracy"]
            calibrated = True
            prep_start = time.perf_counter()  # neither the failed attempt nor calibration is counted
            result, steps, calc_start, calc_s = full(accuracy)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Non-finite values in the result")
        total_s = build_s + (calc_start - prep_start) + calc_s
        report({"event": "result", "calc_s": calc_s, "total_s": total_s, "prep_s": total_s - calc_s,
                "rhs_stage_s": build_s, "steps_per_trajectory": float(np.mean(steps)),
                "steps_max": float(np.max(steps)),
                "steps_per_period": float(np.mean(steps)) / job["simulation_periods"],
                "accuracy": accuracy, "calibrated": calibrated, "recalibrated_from": recalibrated_from})
    except Exception as exc:
        report({"event": "result", "error": f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300]})


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def output_csv_path(settings: dict) -> Path:
    path = Path(settings["output_name"] + "_calibrated.csv")
    return path if path.is_absolute() else SCRIPT_DIR / "results" / path


def settings_signature(settings: dict, system: str) -> dict[str, str]:
    multi = system == "multi_qubit"
    return {"system": system, "topology": settings["topology"] if multi else "",
            "drive": settings["drive"] if multi else "", "nx": str(settings["nx"]),
            "ny": str(settings["ny"]), "simulation_periods": str(settings["simulation_periods"]),
            "calibrate_timesteps": json.dumps(settings["calibrate_timesteps"], sort_keys=True),
            "tsave_per_period": str(settings["tsave_per_period"]),
            "physics": json.dumps(settings["physics"][system], sort_keys=True),
            "layout": settings["layout"], "matmul_precision": settings["matmul_precision"]}


def equipment() -> dict:
    info = {"timestamp_local": datetime.now().isoformat(timespec="seconds")}
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        info["gpu"] = ""
    import dynamiqs
    import jax
    info.update(python=sys.version.split()[0], numpy=np.__version__, jax=jax.__version__,
                dynamiqs=dynamiqs.__version__, precision="fp32")
    return info


def run_point(job: dict, time_limit_s: float, finish_over_limit: bool = False) -> dict:
    """Run one point in a worker; the limit applies to its first-call time ``total_s``."""
    with tempfile.TemporaryDirectory(prefix="dq_b04_") as tmp:
        result_path, job_path = Path(tmp) / "result.jsonl", Path(tmp) / "job.json"
        job_path.write_text(json.dumps({**job, "result_file": str(result_path)}), encoding="utf-8")
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)])
        started = None
        while True:
            events = []
            if result_path.is_file():
                events = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
            if started is None and any(e["event"] == "start" for e in events):
                started = time.perf_counter()
            result = next((e for e in events if e["event"] == "result"), None)
            if result is not None:
                proc.wait()
                return {**result, "over_limit": result.get("total_s", 0.0) > time_limit_s}
            if proc.poll() is not None:
                return {"error": f"worker exit code {proc.returncode}"}
            if (not finish_over_limit and started is not None
                    and time.perf_counter() - started > time_limit_s):
                proc.kill()
                proc.wait()
                return {"timeout": True}
            time.sleep(0.5)


def open_table(settings: dict) -> dict:
    """Load the CSV shared by both systems; saved points are reused only with the same settings."""
    path = output_csv_path(settings)
    rows, saved = read_scaling_csv(path) if path.is_file() else ({}, {})
    for system in rows:
        signature = settings_signature(settings, system)
        stored = system_metadata(saved, system)
        own = system in settings["systems"]
        differing = [key for key in SIGNATURE_KEYS if key != "system"
                     and (own or key not in ("topology", "drive", "physics"))
                     and stored.get(key, "") != signature[key]]
        if differing:
            raise ValueError(f"{path.name}: {system} points were measured with different settings "
                             f"({', '.join(differing)}).")
    metadata = {key: value for key, value in saved.items()
                if any(key.startswith(f"{system} ") for system in rows)}
    metadata = {**equipment(), **metadata}
    for system in settings["systems"]:
        metadata.update(signature_metadata(settings_signature(settings, system)))
    metadata.update({"solvers": ",".join(settings["solvers"]),
                     "point_time_limit_s": f"{settings['point_time_limit_s']:g}"})
    return {"path": path, "rows": rows, "metadata": metadata}


def run_system(settings: dict, system: str, table: dict) -> None:
    path, metadata = table["path"], table["metadata"]
    rows = table["rows"].setdefault(system, {})
    limit = float(settings["point_time_limit_s"])
    finish = settings["time_limit_action"] == "finish"
    label = "Nph" if system == "qubit_resonator" else "qubits"
    print(f"\n=== {system}: {path}", flush=True)
    for size in settings["sizes"][system]:
        dim = hilbert_dim(system, size)
        print(f"\n--- {label}={size} (Hilbert-space dimension {dim})", flush=True)
        for solver in settings["solvers"]:
            smaller = [rows[(solver, s)] for s in settings["sizes"][system]
                       if s < size and (solver, s) in rows]
            blocked = any(row["status"] != "measured" or OVER_LIMIT_NOTE in row["note"] for row in smaller)
            saved = rows.get((solver, size))
            listed = f"{solver}:{size}" in settings["recalculate"]
            # "finish" retries saved timeouts that no smaller point blocks.
            if saved is not None and not listed and not (finish and saved["status"] == "timeout"
                                                         and not blocked):
                print(f"  {solver}: kept ({saved['status']})")
                continue
            if blocked:
                print(f"  {solver}: not run (a smaller size timed out, failed or ran past the limit)")
                continue
            sources = [row for row in smaller if REUSED_NOTE not in row["note"]]
            job = {**{k: settings[k] for k in ("nx", "ny", "simulation_periods", "calibrate_timesteps",
                                               "tsave_per_period", "layout", "matmul_precision",
                                               "vectorized", "memory_budget_gb", "state_copies",
                                               "adaptive_max_steps", "fixed_max_steps",
                                               "topology", "drive")},
                   "system": system, "size": size, "solver": solver,
                   "physics": settings["physics"][system]}
            note_prefix = ""
            if settings["calibrate_smallest_size_only"] and sources:
                source = sources[-1]
                job["reuse_accuracy"] = float(re.search(r"accuracy=([0-9.eE+-]+)", source["note"]).group(1))
                if solver == "dynamiqs_rouchon3":
                    job["reuse_accuracy"] = int(job["reuse_accuracy"])
                note_prefix = f"{REUSED_NOTE} {label}={source['size']}; "
            if "reuse_accuracy" not in job:
                # Calibration is optional, so it is not timed: calibrate in one worker, then measure
                # with that tolerance in a fresh one, like the larger sizes.
                print(f"  {solver}: calibrating (not timed)...", flush=True)
                event = run_point({**job, "calibrate_only": True}, limit, finish)
                if "accuracy" in event:
                    job["reuse_accuracy"] = event["accuracy"]
                    if solver == "dynamiqs_rouchon3":
                        job["reuse_accuracy"] = int(job["reuse_accuracy"])
                    note_prefix = f"{CALIBRATED_HERE_NOTE}; "
            if "reuse_accuracy" in job:  # else the failed calibration event is stored below
                print(f"  {solver}: calculating...", flush=True)
                event = run_point(job, limit, finish)
            if event.get("over_limit"):
                note_prefix = f"{OVER_LIMIT_NOTE} ({limit:g} s); {note_prefix}"
            row = {"system": system, "size": size, "hilbert_dim": dim, "ode_variables": dim * dim - 1, "solver": solver,
                   "time_limit_s": f"{limit:g}", "measured_at": datetime.now().isoformat(timespec="seconds")}
            if event.get("timeout"):
                row.update(status="timeout", note=f"{note_prefix}stopped after {limit:g} s")
            elif "error" in event:
                row.update(status="failed", note=f"{note_prefix}{event['error']}")
            else:
                if event["recalibrated_from"] is not None:
                    note_prefix = f"recalibrated: {label}={size} reused accuracy gave non-finite output; "
                elif event["calibrated"]:
                    note_prefix = ""
                row.update(status="measured", calc_s=f"{event['calc_s']:.7g}",
                           prep_s=f"{event['prep_s']:.7g}", total_s=f"{event['total_s']:.7g}",
                           rhs_stage_s=f"{event['rhs_stage_s']:.7g}",
                           steps_per_trajectory=f"{event['steps_per_trajectory']:.7g}",
                           steps_per_period=f"{event['steps_per_period']:.7g}",
                           note=f"{note_prefix}accuracy={event['accuracy']:.6g}; max steps {event['steps_max']:.0f}")
            rows[(solver, size)] = row
            write_scaling_csv(path, table["rows"], metadata)
            print(f"  {solver}: {row['status']} calc_s={row.get('calc_s', '')} "
                  f"steps={row.get('steps_per_trajectory', '')} {row['note']}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--systems", nargs="+")
    parser.add_argument("--sizes", nargs="+", type=int, help="sizes for every selected system")
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--output-suffix", default="", help="appended to the CSV name (tests)")
    parser.add_argument("--time-limit", dest="point_time_limit_s", type=float)
    parser.add_argument("--time-limit-action", choices=("abort", "finish"))
    parser.add_argument("--recalculate", nargs="+", help="saved points to calculate again, e.g. dynamiqs_tsit5:2")
    args = parser.parse_args(argv)
    if args.worker:
        run_worker(json.loads(Path(args.worker).read_text(encoding="utf-8")))
        return
    settings = user_settings()
    for key in ("systems", "solvers", "nx", "ny", "point_time_limit_s", "time_limit_action", "recalculate"):
        if getattr(args, key):
            settings[key] = getattr(args, key)
    if args.sizes:
        settings["sizes"] = {system: tuple(args.sizes) for system in settings["sizes"]}
    if args.output_suffix:
        settings["output_name"] += args.output_suffix
    table = open_table(settings)
    for system in settings["systems"]:
        run_system(settings, system, table)


def user_settings() -> dict:
    """User-editable defaults. Command-line arguments override these values."""
    return {
        "systems": ("qubit_resonator", "multi_qubit"),  # saved in one CSV
        "topology": "chain",  # multi-qubit coupling: "chain" or "square"
        "drive": "single",  # multi-qubit drive: "single" (qubit 0 only) or "all"
        "sizes": {"qubit_resonator": (2, 3, 4, 5, 6, 7, 8, 9, 10),  # Nph; Hilbert-space dimension 2*Nph
                  "multi_qubit": (1, 2, 3, 4, 5)},  # qubit count n; Hilbert-space dimension 2**n
        "solvers": SOLVERS,
        # Grid, time span and accuracy check as in Benchmark 04.
        "nx": 1024,  # detuning points
        "ny": 1024,  # amplitude points
        "simulation_periods": 40,  # periods of driving
        "calibrate_timesteps": {"rms_atol": 1e-3, "max_atol": 1e-2, "sample_stride": 8,
                                "bisection_steps": 2,
                                "start_steps_per_period": 32,  # fixed step: starting grid
                                "start_tol": 1e-2, "min_tol": 1e-8},  # Tsit5: tolerance range
        "calibrate_smallest_size_only": True,  # larger sizes reuse the last calibrated value
        "point_time_limit_s": 900.0,  # stops the running point; larger sizes are not started
        # "abort": stop the point at the limit; "finish": let it finish (noted in the CSV) and
        # retry saved timeouts. Either way larger sizes are not started.
        "time_limit_action": "abort",
        "recalculate": (),  # saved points to calculate again, e.g. ("dynamiqs_tsit5:2",)
        "tsave_per_period": 32,  # Tsit5: dense-output samples per period for the time average
        # dynamiqs performance options (see dynamiqs Performance docs).
        "layout": "dia",  # operator layout: "dia" (sparse diagonals) or "dense"
        "vectorized": False,  # mesolve vectorized Liouvillian
        "matmul_precision": "highest",  # full FP32 matmuls (no TF32), like GQIS FP32
        "memory_budget_gb": 3.0,  # GPU memory per chunk of grid points
        "state_copies": 32,  # density-matrix-sized buffers assumed per grid point (chunk size)
        "adaptive_max_steps": 1_000_000,  # Tsit5 step limit per trajectory
        "fixed_max_steps": 1 << 20,  # fixed-step calibration limit (total steps)
        # Physics parameters, copied from Benchmark 04.
        "physics": {
            "qubit_resonator": {
                "Delta": 1.0, "wq1": 4.71 * 1.15, "wr2": 7.6767, "gamma1": 2.0e-3,
                "gammaph": 0.0, "kappa": 5.0e-3, "Ap": 0.0002 / 10.0, "g1": 0.04033970276 / 1.23,
                "w/Delta": 0.0923105326317733, "eps_max/w": 22.74020028, "A_max/w": 27.8315904255,
            },
            "multi_qubit": {
                "Delta": 1.0, "w/Delta": 1.14, "gamma1_per_T": 0.05, "gamma_phi_per_T": 0.04,
                "eps_max/w": 10.0, "A_max/w": 10.0, "gap_step": 0.05, "J/Delta": 0.1,
            },
        },
        # One CSV for both systems; relative names are saved in Benchmarks/results/.
        "output_name": "Benchmark_04_hilbert_space_scaling_dynamiqs",
    }


if __name__ == "__main__":
    main()

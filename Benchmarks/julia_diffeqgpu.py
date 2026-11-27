"""Julia DiffEqGPU (fixed-step GPUTsit5) comparison solver shared by the benchmarks.

The ODE comes from GQIS's ``compile_lindblad_rhs``, so Julia integrates exactly the equations GQIS
generates; only the printing differs (Julia with SymPy CSE instead of the CUDA plan). The scripts
run in a separate Julia process with DifferentialEquations, DiffEqGPU, CUDA and StaticArrays.

Precision modes (Benchmark 03): ``fp32`` (stock FP32 state and time), ``fp64``,
``fp64_only_step`` (FP32 state, FP64 time) and ``fp32_integer_step`` (FP32 time rebuilt from the
integer step counter instead of accumulating ``t += dt``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

from Benchmark_full_tools import sympy_to_julia_fp32

PRECISIONS = ("fp32", "fp64", "fp64_only_step", "fp32_integer_step")
TSIT5_ORDER = 5
OUT_CSV = "<out_csv>"  # placeholder in script arguments, replaced by the output map path


@dataclass
class JuliaModel:
    """Julia RHS of one Lindblad model: ``rhs(u, p, t)`` and optional ``observe(u, p, t)``."""

    rhs: str
    observe: str
    states: int  # reduced density-matrix variables (extra average variables follow them)
    averages: int  # extra integrated variables: 0, 1 (real part) or 2 (real, imaginary)
    precision: str
    prepare_s: float  # ODE compile (symbolic preparation and optimization) and Julia printing


def _julia_types(precision):
    """(state type, solver time type, RHS time type) of a precision mode."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}.")
    state = "Float64" if precision == "fp64" else "Float32"
    solver_time = "Float64" if precision in {"fp64", "fp64_only_step"} else "Float32"
    return state, solver_time, state


def _julia_code(expr, precision):
    """Julia text of a scalar expression; exact constants become literals of the state type."""
    constants = {}
    walk = sp.preorder_traversal(expr)
    for sub in walk:
        if sub.is_number and not (sub.is_Rational or sub.is_Float) and sub.is_real:
            constants[sub] = sub.evalf()
            walk.skip()
    code = sympy_to_julia_fp32(expr.xreplace(constants) if constants else expr)
    return re.sub(r"(?<=\d)f(?=[+-]?\d)", "e", code) if precision == "fp64" else code


def julia_model(H, drive, col_ops, mean_op, sweep_symbols, *, const_values=None,
                precision="fp32", average="after_warmup", observe=False) -> JuliaModel:
    """Compile the model with GQIS and print it as a Julia RHS.

    ``H`` contains the placeholder symbol ``Drive``; ``drive`` is its expression of the sweep
    symbols and the time symbol ``t``. ``p[1]``, ``p[2]`` hold the two sweep values.
    ``average``: ``"after_warmup"`` integrates the real part of the observable from
    ``p[3]`` (warmup time) as one extra variable; ``"complex"`` integrates the real and imaginary
    parts from the start as two extra variables; ``None`` adds nothing. ``observe=True`` also
    prints ``observe(u, p, t)``, the real part of the observable of a state.
    """
    from gqis import compile_lindblad_rhs

    start = time.perf_counter()
    state_type, _, rhs_time_type = _julia_types(precision)
    placeholder = next(s for s in H.free_symbols if s.name == "Drive")
    # RHSreuse=False: every benchmark point prepares its ODE (first-call time), like GQIS's first call.
    compiled = compile_lindblad_rhs(H, {placeholder: drive}, col_ops, mean_op, sweep_symbols,
                                    const_values=const_values, RHSreuse=False)
    u = [sp.Symbol(f"u{i}") for i in range(compiled.state_size)]
    to_u = dict(zip(compiled.state_symbols, u))
    t_rhs = sp.Symbol("t_rhs", real=True)
    drive_lines = []
    for symbol, expr in compiled.drive_map.items():
        times = {s: t_rhs for s in expr.free_symbols if s.name == "t"}
        drive_lines.append(f"{symbol} = {state_type}({_julia_code(expr.xreplace(times), precision)})")
    equations = [e.xreplace(to_u) for e in compiled.equations]
    observables = {"after_warmup": [compiled.mean_re], "complex": [compiled.mean_re, compiled.mean_im],
                   None: []}[average]
    common, reduced = sp.cse([*equations, *(o.xreplace(to_u) for o in observables)],
                             symbols=sp.numbered_symbols("tmp"), optimizations="basic")

    def assign(name, expr):
        return f"{name} = {state_type}({_julia_code(expr, precision)})"

    header = ["ParX = p[1]", "ParY = p[2]", f"t_rhs = {rhs_time_type}(t)", *drive_lines,
              *(f"u{i} = u[{i + 1}]" for i in range(len(u)))]
    lines = header + [assign(s, e) for s, e in common]
    lines += [assign(f"du{i + 1}", e) for i, e in enumerate(reduced[:len(u)])]
    outputs = [f"du{i + 1}" for i in range(len(u))]
    if average == "after_warmup":
        lines += [assign("obs", reduced[-1]), f"ds = (t_rhs >= p[3]) ? obs : {state_type}(0)"]
        outputs.append("ds")
    elif average == "complex":
        lines += [assign("obs_re", reduced[-2]), assign("obs_im", reduced[-1])]
        outputs += ["obs_re", "obs_im"]
    rhs = ("@inline function rhs(u, p, t)\n    " + "\n    ".join(lines)
           + f"\n    return @SVector [{', '.join(outputs)}]\nend\n")
    observe_source = ""
    if observe:
        obs_common, obs_reduced = sp.cse([compiled.mean_re.xreplace(to_u)],
                                         symbols=sp.numbered_symbols("obs_tmp"), optimizations="basic")
        body = header + [assign(s, e) for s, e in obs_common] + [assign("obs", obs_reduced[0])]
        observe_source = ("@inline function observe(u, p, t)\n    " + "\n    ".join(body)
                          + "\n    return obs\nend\n")
    return JuliaModel(rhs, observe_source, len(u), {"after_warmup": 1, "complex": 2, None: 0}[average],
                      precision, time.perf_counter() - start)


def julia_preamble(precision, tf_parameter=None):
    """Package imports, number types and the step specialization of a precision mode.

    ``tf_parameter``: index of the final time in ``p`` (``fp32_integer_step`` only).
    """
    state_type, solver_time, rhs_time = _julia_types(precision)
    text = ("using DifferentialEquations\nusing DiffEqGPU\nusing CUDA\nusing StaticArrays\n"
            "using DelimitedFiles\n\n"
            f"const TimeType = {solver_time}\nconst RhsTimeType = {rhs_time}\n"
            f"const StateType = {state_type}\n")
    if precision == "fp64_only_step":
        text += _MIXED_TIME_STEP
    elif precision == "fp32_integer_step":
        if tf_parameter is None:
            raise ValueError("fp32_integer_step needs the index of tf in p (tf_parameter).")
        text += _INTEGER_TIME_STEP.replace("TF_PARAMETER", str(tf_parameter))
    return text


def _initial_state(model):
    """rho = |0><0| (first reduced variable 1) and zero averages."""
    return ", ".join(["StateType(1)"] + ["StateType(0)"] * (model.states - 1 + model.averages))


def julia_map_script(model: JuliaModel, *, output_abs=False) -> str:
    """Script for one (eps, A) map (Benchmarks 01-03).

    Arguments: out_csv nx ny num_t dt eps_min eps_max A_min A_max warmup_steps t0 (see
    ``map_arguments``). Writes the ny x nx map to out_csv and prints ``JULIA_SOLVE_TIME``; the
    kernel is compiled on a small batch first, outside the timed solve. The map is the average of
    the observable after the warmup (``average="after_warmup"``) or the final observable
    (``average=None``, ``observe=True``); ``output_abs`` takes its absolute value.
    """
    integer_step = model.precision == "fp32_integer_step"
    if model.averages == 1:
        value = f"sol[idx][end][{model.states + 1}] / denom_t"
    elif model.observe:
        value = "observe(sol[idx][end], params[idx], TimeType(tf))"
    else:
        raise ValueError("A map needs average='after_warmup', or observe=True for the final value.")
    value = f"abs({value})" if output_abs else value
    return (julia_preamble(model.precision, tf_parameter=4 if integer_step else None) + "\n"
            + model.rhs + "\n" + model.observe + "\n"
            + _MAP_DRIVER.replace("PARAMETER_COUNT", "4" if integer_step else "3")
            .replace("PARAMETER_TAIL", ", StateType(tf)" if integer_step else "")
            .replace("INITIAL_STATE", _initial_state(model)).replace("MAP_VALUE", value))


def map_arguments(nx, ny, num_t, dt, eps_list, A_list, warmup_steps, t0):
    """Arguments of ``julia_map_script``; ``OUT_CSV`` marks the output file."""
    return [OUT_CSV, str(nx), str(ny), str(num_t), repr(float(dt)),
            repr(float(eps_list[0])), repr(float(eps_list[-1])),
            repr(float(A_list[0])), repr(float(A_list[-1])), str(int(warmup_steps)), repr(float(t0))]


def julia_calibration_script(model: JuliaModel) -> str:
    """Script averaging the observable over the full time with GQIS-style step calibration
    (Benchmark 04). Needs ``average="complex"`` and ``precision="fp32_integer_step"``.

    Arguments: nx ny tf eps_max A_max base_steps fixed_steps rms_atol max_atol sample_stride
    max_steps bisection_steps; ``fixed_steps=0`` calibrates (doubling, Richardson-scaled change on
    every sample_stride-th point plus the axis ends, geometric bisection), otherwise uses the
    given steps. Prints ``JULIA_CALIBRATION_S``, ``JULIA_STEPS``, ``JULIA_STATUS`` and
    ``JULIA_CALC_S``; exits with code 3 on a non-finite result.
    """
    if model.averages != 2 or model.precision != "fp32_integer_step":
        raise ValueError("The calibration driver needs average='complex' and fp32_integer_step.")
    return (julia_preamble(model.precision, tf_parameter=3) + "\n"
            + f"const NVARS = {model.states}\nconst ORDER = {TSIT5_ORDER}\n"
            + f"const U0 = @SVector [{_initial_state(model)}]\n\n" + model.rhs + _CALIBRATION_DRIVER)


def run_julia_script(source, arguments, *, julia_cmd="julia", timeout=None, check=True,
                     output_dtype=np.float32):
    """Run a generated Julia script in its own process.

    Returns ``(values, output_map, wall_s, returncode)``: ``JULIA_NAME=value`` lines as a dict of
    strings, the ``OUT_CSV`` map (or ``None``), and the process wall time. ``check`` raises
    on a non-zero exit code; ``timeout`` raises ``TimeoutError``.
    """
    julia = shutil.which(julia_cmd)
    if julia is None:
        raise RuntimeError(f"Julia executable '{julia_cmd}' was not found in PATH.")
    with tempfile.TemporaryDirectory(prefix="gqis_julia_") as tmp:
        script, out_csv = Path(tmp) / "gqis_diffeqgpu.jl", Path(tmp) / "out.csv"
        script.write_text(source, encoding="utf-8")
        command = [julia, str(script)] + [str(out_csv) if a == OUT_CSV else a for a in arguments]
        start = time.perf_counter()
        try:
            proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Julia exceeded {timeout:.3g} s and was terminated.") from exc
        wall_s = time.perf_counter() - start
        if check and proc.returncode != 0:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-8:]
            raise RuntimeError(f"Julia exit code {proc.returncode}: {' | '.join(tail)}")
        output = (np.loadtxt(out_csv, delimiter=",", dtype=output_dtype)
                  if proc.returncode == 0 and out_csv.exists() else None)
    values = dict(re.findall(r"^JULIA_(\w+)=(.*)$", proc.stdout, flags=re.MULTILINE))
    if proc.returncode != 0:
        values["OUTPUT_TAIL"] = " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
    return values, output, wall_s, proc.returncode


_MAP_DRIVER = """
function main()
    out_csv = ARGS[1]
    nx, ny, num_t = parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4])
    dt = parse(Float64, ARGS[5])
    eps_min, eps_max = parse(Float64, ARGS[6]), parse(Float64, ARGS[7])
    A_min, A_max = parse(Float64, ARGS[8]), parse(Float64, ARGS[9])
    warmup_steps, t0 = parse(Int, ARGS[10]), parse(Float64, ARGS[11])

    eps_list = collect(StateType, range(StateType(eps_min), stop=StateType(eps_max), length=nx))
    A_list = collect(StateType, range(StateType(A_min), stop=StateType(A_max), length=ny))
    warmup_t = StateType(t0 + warmup_steps * dt)
    denom_t = StateType(max(num_t - 1 - warmup_steps, 1)) * StateType(dt)
    tf = t0 + dt * max(num_t - 1, 0)
    params = [SVector{PARAMETER_COUNT, StateType}(eps_list[i], A_list[j], warmup_tPARAMETER_TAIL)
              for j in 1:ny for i in 1:nx]

    u0 = @SVector [INITIAL_STATE]
    prob = ODEProblem{false}(rhs, u0, (TimeType(t0), TimeType(tf)), params[1])
    eprob = EnsembleProblem(prob; prob_func=(pr, i, repeat) -> remake(pr; p=params[i]),
                            safetycopy=false)
    solve_map(n) = solve(eprob, GPUTsit5(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend());
                         trajectories=n, adaptive=false, dt=TimeType(dt), save_everystep=false)

    # Compile and initialize on a small batch, outside the timed full-grid solve.
    solve_map(min(length(params), 256))
    CUDA.synchronize()
    GC.gc(true)
    CUDA.reclaim()

    solve_start = time()
    sol = solve_map(length(params))
    CUDA.synchronize()
    solve_elapsed = time() - solve_start

    out = zeros(StateType, ny, nx)
    for idx in 1:length(params)
        j = Int(fld(idx - 1, nx)) + 1
        i = Int(mod(idx - 1, nx)) + 1
        out[j, i] = MAP_VALUE
    end
    writedlm(out_csv, out, ',')
    println("JULIA_SOLVE_TIME=", solve_elapsed)
end

main()
"""

# DiffEqGPU's fixed-step GPUTsit5 with FP64 solver clock and stage times, FP32 state updates.
_MIXED_TIME_STEP = """
@eval DiffEqGPU begin
@inline function step!(integ::GPUT5I{false, S, Float64}, ts, us) where {S}
    c1, c2, c3, c4, c5, c6 = integ.cs
    dt = integ.dt
    t = integ.t
    p = integ.p
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = Float32.(integ.as)
    f = integ.f
    integ.uprev = integ.u
    uprev = integ.u
    integ.tprev = t
    if integ.tstops !== nothing && integ.tstops_idx <= length(integ.tstops) &&
       integ.tstops[integ.tstops_idx] - integ.t - integ.dt - 100 * eps(Float64) < 0
        integ.t = integ.tstops[integ.tstops_idx]
        dt = integ.t - integ.tprev
        integ.tstops_idx += 1
    else
        integ.t += dt
    end
    dt32 = Float32(dt)
    if integ.u_modified
        k1 = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k1 = integ.k7
    end
    tmp = uprev + dt32 * a21 * k1
    k2 = f(tmp, p, t + c1 * dt)
    tmp = uprev + dt32 * (a31 * k1 + a32 * k2)
    k3 = f(tmp, p, t + c2 * dt)
    tmp = uprev + dt32 * (a41 * k1 + a42 * k2 + a43 * k3)
    k4 = f(tmp, p, t + c3 * dt)
    tmp = uprev + dt32 * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    k5 = f(tmp, p, t + c4 * dt)
    tmp = uprev + dt32 * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    k6 = f(tmp, p, t + dt)
    integ.u = uprev + dt32 * ((a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4) +
                              a75 * k5 + a76 * k6)
    k7 = f(integ.u, p, t + dt)
    @inbounds begin
        integ.k1 = k1; integ.k2 = k2; integ.k3 = k3; integ.k4 = k4
        integ.k5 = k5; integ.k6 = k6; integ.k7 = k7
    end
    _, saved_in_cb = handle_callbacks!(integ, ts, us)
    return saved_in_cb
end
end
"""

# DiffEqGPU's fixed-step GPUTsit5 with the FP32 time rebuilt from the integer step counter instead
# of accumulating t += dt (no saveat, callbacks or tstops are used). p[TF_PARAMETER] is tf; only
# the loop marker is clamped near the final grid point, so rounded dt and tf add no extra step.
_INTEGER_TIME_STEP = """
@eval DiffEqGPU begin
@inline function step!(integ::GPUT5I{false, S, Float32}, ts, us) where {S}
    c1, c2, c3, c4, c5, c6 = integ.cs
    dt = integ.dt
    p = integ.p
    a21, a31, a32, a41, a42, a43, a51, a52, a53, a54,
    a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76 = integ.as
    f = integ.f
    step_index = integ.cur_t
    t = muladd(Float32(step_index), dt, integ.t0)
    next_t = muladd(Float32(step_index + 1), dt, integ.t0)
    integ.cur_t = step_index + 1
    integ.tprev = t
    integ.t = next_t >= p[TF_PARAMETER] - 0.5f0 * abs(dt) ? p[TF_PARAMETER] : next_t
    integ.uprev = integ.u
    uprev = integ.u
    if integ.u_modified
        k1 = f(uprev, p, t)
        integ.u_modified = false
    else
        @inbounds k1 = integ.k7
    end
    tmp = uprev + dt * a21 * k1
    k2 = f(tmp, p, muladd(c1, dt, t))
    tmp = uprev + dt * (a31 * k1 + a32 * k2)
    k3 = f(tmp, p, muladd(c2, dt, t))
    tmp = uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3)
    k4 = f(tmp, p, muladd(c3, dt, t))
    tmp = uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    k5 = f(tmp, p, muladd(c4, dt, t))
    tmp = uprev + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    k6 = f(tmp, p, next_t)
    integ.u = uprev + dt * ((a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4) +
                            a75 * k5 + a76 * k6)
    k7 = f(integ.u, p, next_t)
    @inbounds begin
        integ.k1 = k1; integ.k2 = k2; integ.k3 = k3; integ.k4 = k4
        integ.k5 = k5; integ.k6 = k6; integ.k7 = k7
    end
    _, saved_in_cb = handle_callbacks!(integ, ts, us)
    return saved_in_cb
end
end
"""

# Benchmark 04 driver: compile on a small batch, calibrate like GQIS calibrate_timesteps, then one
# timed full solve. The time average of the observable is integrated as two extra ODE variables.
_CALIBRATION_DRIVER = """
grid_params(eps_list, A_list, tf) = [SVector{3, Float32}(e, a, tf) for a in A_list for e in eps_list]

function averaged(params, steps, tf)
    prob = ODEProblem{false}(rhs, U0, (0.0f0, Float32(tf)), params[1])
    eprob = EnsembleProblem(prob; prob_func=(pr, i, repeat) -> remake(pr; p=params[i]), safetycopy=false)
    start = time()
    sol = solve(eprob, GPUTsit5(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend());
                trajectories=length(params), adaptive=false, dt=Float32(tf / steps), save_everystep=false)
    CUDA.synchronize()
    elapsed = time() - start
    values = [ComplexF32(sol[i][end][NVARS + 1], sol[i][end][NVARS + 2]) / Float32(tf)
              for i in eachindex(params)]
    return values, elapsed
end

function change(steps, coarse, fine_steps, fine, rms_atol, max_atol)
    all(isfinite, coarse) && all(isfinite, fine) || return Inf, Inf, false
    scale = 1.0 / (1.0 - (steps / fine_steps)^ORDER)
    difference = abs.(ComplexF64.(fine) .- ComplexF64.(coarse)) .* scale
    rms, largest = sqrt(sum(abs2, difference) / length(difference)), maximum(difference)
    return rms, largest, (rms_atol <= 0 || rms <= rms_atol) && (max_atol <= 0 || largest <= max_atol)
end

function calibrate(params, tf, base, rms_atol, max_atol, max_steps, bisections)
    run(steps) = averaged(params, steps, tf)[1]
    steps, coarse = base, run(base)
    tested = Tuple{Int, Float64}[]
    status, accepted, reference_steps, reference = "", base, 0, coarse
    while status == ""
        if 2 * steps > max_steps
            status, accepted = "max_steps", steps
            break
        end
        fine = run(2 * steps)
        rms, _, passed = change(steps, coarse, 2 * steps, fine, rms_atol, max_atol)
        push!(tested, (steps, rms))
        if passed
            status, accepted, reference_steps, reference = "converged", steps, 2 * steps, fine
        elseif length(tested) > 1 && isfinite(rms) && rms >= tested[end - 1][2]
            status, accepted = "precision_floor", tested[argmin(last.(tested))][1]
        end
        steps, coarse = 2 * steps, fine
    end
    doubling = accepted
    if status == "converged" && bisections > 0
        passes(candidate) = change(candidate, run(candidate), reference_steps, reference,
                                   rms_atol, max_atol)[3]
        failing = accepted > base ? accepted ÷ 2 : 0
        halvings = 0
        while failing == 0 && accepted >= 2 && halvings < 6
            halvings += 1
            candidate = accepted ÷ 2
            passes(candidate) ? (accepted = candidate) : (failing = candidate)
        end
        for _ in 1:(failing > 0 ? bisections : 0)
            candidate = round(Int, sqrt(failing * accepted))
            (candidate <= failing || candidate >= accepted) && break
            passes(candidate) ? (accepted = candidate) : (failing = candidate)
        end
    end
    return accepted, doubling, status
end

function main()
    nx, ny = parse(Int, ARGS[1]), parse(Int, ARGS[2])
    tf, eps_max, A_max = parse(Float64, ARGS[3]), parse(Float64, ARGS[4]), parse(Float64, ARGS[5])
    base, fixed = parse(Int, ARGS[6]), parse(Int, ARGS[7])
    rms_atol, max_atol = parse(Float64, ARGS[8]), parse(Float64, ARGS[9])
    stride, max_steps, bisections = parse(Int, ARGS[10]), parse(Int, ARGS[11]), parse(Int, ARGS[12])
    eps_list = Float32.(range(-eps_max, eps_max; length=nx))
    A_list = Float32.(range(0.0, A_max; length=ny))
    params = grid_params(eps_list, A_list, tf)
    averaged(params[1:min(256, length(params))], base, tf)  # compile outside the measured solves

    status = "fixed"
    steps, doubling = fixed, fixed
    calibration_s = 0.0
    if fixed == 0
        calibration_start = time()
        pick(n) = unique([collect(1:stride:n); n])
        test = grid_params(eps_list[pick(nx)], A_list[pick(ny)], tf)
        steps, doubling, status = calibrate(test, tf, base, rms_atol, max_atol, max_steps, bisections)
        calibration_s = time() - calibration_start
    end
    println("JULIA_CALIBRATION_S=", calibration_s)
    # One full solve; the kernel was compiled above (the trajectory count is not compiled in).
    values, calc_s = averaged(params, steps, tf)
    if !all(isfinite, values) && steps < doubling  # a bisected step unstable on the full grid
        steps = doubling
        values, calc_s = averaged(params, steps, tf)
    end
    if !all(isfinite, values)
        println("JULIA_NONFINITE steps=", steps)
        exit(3)
    end
    println("JULIA_STEPS=", steps)
    println("JULIA_STATUS=", status)
    println("JULIA_CALC_S=", calc_s)
end

main()
"""

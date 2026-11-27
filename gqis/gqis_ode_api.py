"""Direct symbolic ODE input for GQIS's CUDA sweep engine."""

import dataclasses

import numpy as np
import sympy as sp

from ._options import OPTION_GROUPS, apply_option_groups

# cse_simplify stays on for ODEs: unlike Lindblad equations, user expressions are not simplified
# before CSE (e.g. (x + 1)**2 - x*(x + 2) becomes 1 only with it).
_ODE_CODEGEN_DEFAULTS = {"pre_expand": False, "collect_rho": False, "cse_simplify": True}
_ODE_CODEGEN_OPTIONS = ("pre_expand", "collect_rho", "factor_terms", "cse_batch_size",
                        "cse_simplify", "hoist_rho_independent", "RHSreuse")


def odesolve_2D(rhs, state_symbols=None, y0=None, tlist=None, *, var_arrays=None,
                const_values=None, time_symbol=None, observable=None, output_mode="final",
                y0_values=None, runtime_consts=None, keep_symbolic_consts=None, solver="rk4",
                fp64=False, return_time_trace=False, time_trace_every=None, t_in=0.0,
                return_device=False, calibrate_timesteps=None, consts=None, initial=None, output=None,
                solver_options=None, codegen=None, diagnostics=None, **option_set):
    """Solve real first-order ODEs over zero, one or two parameter-sweep axes.

    rhs: Sequence/column matrix of dy/dt expressions in state_symbols order.
        Explicit Eq(Derivative(state, time, evaluate=False), rhs) is also accepted.
        A CompiledRHS from compile_ode_rhs replaces rhs, state_symbols, y0,
        const_values, keep_symbolic_consts, time_symbol, observable and the symbolic
        code-generation options; the call then starts from code emission. Give tlist by
        keyword, var_arrays for the compiled sweep symbols, and optionally y0_values and
        runtime_consts values.
    state_symbols: Distinct SymPy Symbols for real state variables.
    y0: Initial-value sequence in the same order; may depend on sweep parameters.
        Use None with y0_values for explicit per-trajectory initial states.
    tlist: Uniform increasing times beginning at zero; M values mean M-1 steps.
        These are elapsed times; t_in supplies the absolute initial time.
    return_device: Return state/observable arrays on the GPU (CuPy). Trace times
        remain NumPy metadata. Device y0_values are accepted without a host copy.
    observable: Optional scalar real/complex expression. Without it, output is
        the full real state vector. With it, output is the complex observable.
    output_mode: 'final' or 'mean' (right-endpoint average). warmup_time is the
        discarded fraction of steps, from zero to one, as in mesolve_2D.
    return_time_trace: Also return sampled states, or the selected observable.
        Returns (result, trace, trace_times); time_trace_every=k samples at
        t_in + dt, t_in + (k+1)*dt, ... . The initial state is not included in the trace.
    calibrate_timesteps: None, True or a dict; chooses a uniform step (or, for
        tsit5a, the step tolerances) on an undersampled grid before the full run,
        as in mesolve_2D. Default limits
        for ODEs: rms_atol=1e-4, max_atol=1e-3.

    Results have shape (nx, ny, nstate) without an observable, otherwise (nx, ny).
    State traces have shape (nx, ny, nsamples, nstate); observable traces omit
    nstate. Unused sweep axes have length one. y0_values uses (nx, ny, nstate),
    or (nx, nstate) for one axis. FP32/RK4 are defaults; solver selects a GQIS integrator.
    Code-generation, caching, runtime constants, timing and trace options are
    shared with mesolve_2D. No density-matrix or trace constraints are imposed.
    consts, initial, output, solver_options, codegen and diagnostics optionally group
    these options under their usual names, as in mesolve_2D, e.g.
    codegen=dict(RHSreuse=True); unset options keep defaults. Other mesolve_2D
    code-generation and diagnostic keywords are accepted directly and forwarded
    through option_set.
    """
    args = dict(locals())
    option_set = args.pop("option_set")
    groups = {name: args.pop(name) for name in OPTION_GROUPS}
    args, option_set = apply_option_groups(odesolve_2D, args, groups, option_set)
    return _solve_ode_system(**args, **option_set)


def compile_ode_rhs(rhs, state_symbols, y0=None, sweep_symbols=(), *, const_values=None,
                    runtime_consts=None, keep_symbolic_consts=None, time_symbol=None,
                    observable=None, **codegen_options):
    """Prepare and optimize an ODE RHS without generating device code or using the GPU.

    Takes the ODE arguments of ``odesolve_2D`` and returns a ``CompiledRHS``;
    ``odesolve_2D(compiled, tlist=..., var_arrays=...)`` then starts from code emission.

    Args:
        sweep_symbols: Zero, one or two swept parameters, in the order of the later
            ``var_arrays``. A ``var_arrays`` dict also works.
        runtime_consts: Parameters kept symbolic; values are defaults that later calls'
            ``runtime_consts`` override.
        codegen_options: ``pre_expand``, ``collect_rho``, ``factor_terms``,
            ``cse_batch_size``, ``cse_simplify``, ``hoist_rho_independent`` and ``RHSreuse``,
            with the ``odesolve_2D`` defaults.
        Other arguments: as in ``odesolve_2D``.

    The result's ``timings`` holds ``ode_prepare_s`` and ``ode_optimize_s``; ``equations``,
    ``mean_re``/``mean_im`` and ``plan`` use internal names (``rho[i]`` for the states,
    ``ParX``/``ParY`` for the sweeps, ``ode_param_i`` for the other parameters);
    ``ode["symbol_map"]`` maps the given symbols to them.
    """
    unknown = set(codegen_options) - set(_ODE_CODEGEN_OPTIONS)
    if unknown:
        raise TypeError(f"Unknown compile_ode_rhs option(s): {sorted(unknown)}")
    sweep_symbols = tuple(sweep_symbols)
    if len(sweep_symbols) > 2:
        raise ValueError("At most two sweep symbols are supported.")
    model = _prepare_ode_model(rhs, state_symbols, y0, dict.fromkeys(sweep_symbols), const_values,
                               runtime_consts, keep_symbolic_consts, time_symbol, observable)
    from .solver import _compile_rhs
    options = {**_ODE_CODEGEN_DEFAULTS, **codegen_options}
    reuse = options.pop("RHSreuse", True)
    compiled, _ = _compile_rhs(
        model["H"], model["drive"], [], model["mean"], sweep_symbols=tuple(model["sweeps"]),
        const_values=model["constants"], runtime_consts=model["runtime"],
        keep_symbolic_consts=model["keep"], rho0=model["initial"], ode_size=model["size"],
        reuse=reuse, **options)
    # The mapping belongs to this call: equal internal models can come from different names.
    return dataclasses.replace(compiled, ode={"symbol_map": model["replacements"],
                                              "sweep_symbols": sweep_symbols,
                                              "has_observable": observable is not None})


def _prepare_ode_model(rhs, state_symbols, y0, var_arrays, const_values, runtime_consts,
                       keep_symbolic_consts, time_symbol, observable):
    """Validate a direct ODE model and rename its symbols to the internal kernel names."""
    states = tuple(state_symbols)
    if not states or any(not isinstance(s, sp.Symbol) for s in states) or len(set(states)) != len(states):
        raise ValueError("state_symbols must contain distinct SymPy Symbols.")
    if any(s.is_real is False for s in states):
        raise ValueError("State symbols must represent real variables.")
    expressions = [sp.sympify(e) for e in rhs]
    if len(expressions) != len(states):
        raise ValueError("Provide one RHS expression per state symbol.")
    initial = None if y0 is None else [sp.sympify(e) for e in y0]
    if initial is not None and len(initial) != len(states):
        raise ValueError("y0 must contain one initial value per state symbol.")
    obs = sp.sympify(observable) if observable is not None else sp.Integer(0)
    all_symbols = set().union(*(e.free_symbols for e in [*expressions, *(initial or []), obs]))
    time_candidates = [s for s in all_symbols if s.name == "t" and s not in states]
    if time_symbol is None:
        if len(time_candidates) > 1:
            raise ValueError("Specify time_symbol when multiple different symbols are named t.")
        time_symbol = time_candidates[0] if time_candidates else sp.Symbol("t", real=True)
    if not isinstance(time_symbol, sp.Symbol) or time_symbol in states:
        raise ValueError("time_symbol must be a Symbol distinct from state_symbols.")
    for i, expression in enumerate(expressions):
        if isinstance(expression, sp.Equality):
            if expression.lhs != sp.Derivative(states[i], time_symbol, evaluate=False):
                raise ValueError("Each equation must explicitly define its state derivative.")
            expressions[i] = expression.rhs
    if any(e.has(sp.Derivative) for e in expressions):
        raise ValueError("Supply explicit first-order RHS expressions, without derivatives.")

    sweeps = {} if var_arrays is None else dict(var_arrays)
    constants = {} if const_values is None else dict(const_values)
    runtime = {} if runtime_consts is None else dict(runtime_consts)
    parameters = set(sweeps) | set(constants) | set(runtime)
    if any(not isinstance(s, sp.Symbol) for s in parameters) or parameters & (set(states) | {time_symbol}):
        raise ValueError("Parameter keys must be Symbols distinct from the state and time symbols.")
    if set(sweeps) & (set(constants) | set(runtime)):
        raise ValueError("A swept parameter cannot also be a constant.")
    unknown = all_symbols - set(states) - {time_symbol} - parameters
    if unknown:
        raise ValueError(f"Unassigned ODE parameters: {sorted(map(str, unknown))}")
    if initial is not None and any(e.free_symbols & (set(states) | {time_symbol}) for e in initial):
        raise ValueError("Initial values may depend on parameters, not on state or time.")
    # Internal names avoid collisions with kernel locals and CSE temporaries.
    replacements = {s: sp.Symbol(f"rho[{i}]", real=True) for i, s in enumerate(states)}
    stage_time = sp.Symbol("ode_stage_time", real=True)
    replacements[time_symbol] = stage_time
    replacements.update({s: sp.Symbol(f"ode_param_{i}", real=True)
                         for i, s in enumerate(sorted(parameters, key=sp.srepr))})
    expressions = [e.xreplace(replacements) for e in expressions]
    initial = None if initial is None else [e.xreplace(replacements) for e in initial]
    if any(sp.simplify(sp.im(e)) != 0 for e in [*expressions, *(initial or [])]):
        raise ValueError("ODE states and RHS must be real; split complex states into real/imaginary parts.")
    obs = obs.xreplace(replacements)
    def remap(mapping):
        return {replacements[s]: value for s, value in mapping.items()}
    sweeps = remap(sweeps)
    if not sweeps:
        sweeps = {_DUMMY_SWEEP: np.zeros(1)}
    if keep_symbolic_consts is not None and not isinstance(keep_symbolic_consts, str):
        keep_symbolic_consts = [replacements[s] for s in keep_symbolic_consts]
    drive = {stage_time: sp.Symbol("t", real=True)} if time_symbol in all_symbols else {}
    return {"H": sp.Matrix(expressions), "drive": drive, "mean": sp.Matrix([obs]),
            "initial": initial, "sweeps": sweeps, "constants": remap(constants),
            "runtime": remap(runtime), "keep": keep_symbolic_consts, "size": len(states),
            "replacements": replacements}


_DUMMY_SWEEP = sp.Symbol("ode_dummy", real=True)


def _solve_ode_system(rhs, state_symbols, y0, tlist, *, var_arrays, const_values, time_symbol,
                      observable, output_mode, y0_values, runtime_consts, keep_symbolic_consts,
                      solver, fp64, return_time_trace, time_trace_every, t_in, return_device,
                      calibrate_timesteps,
                      **option_set):
    """Validate a direct ODE model and run it through the shared CUDA sweep engine."""
    from .solver import CompiledRHS, _solve_symbolic_system
    if tlist is None:
        raise TypeError("odesolve_2D needs tlist.")
    if output_mode not in {"mean", "final"}:
        raise ValueError("output_mode must be 'mean' or 'final'.")
    if y0 is not None and y0_values is not None:
        raise ValueError("Use either y0 or y0_values, not both.")
    if y0_values is not None and np.iscomplexobj(y0_values):
        raise ValueError("y0_values must be real; split complex states into real/imaginary parts.")
    if any(key.startswith("_") or key in {"rho0", "rho0_values", "rho0_var_arrays", "compiled_rhs"}
           for key in option_set):
        raise ValueError("Use y0/y0_values and var_arrays for ODE initial conditions.")

    if isinstance(rhs, CompiledRHS):
        compiled = rhs
        if compiled.kind != "ode" or compiled.ode is None:
            raise ValueError("Pass a CompiledRHS from compile_ode_rhs (compile_lindblad_rhs "
                             "results go to mesolve_2D).")
        if any(value is not None for value in (state_symbols, y0, const_values, keep_symbolic_consts,
                                               time_symbol, observable)):
            raise ValueError("state_symbols, y0, const_values, keep_symbolic_consts, time_symbol "
                             "and observable are fixed when the RHS is compiled.")
        if y0_values is None and compiled.rho0_exprs is None:
            raise ValueError("Provide y0_values: no y0 was given when the RHS was compiled.")
        if tuple(var_arrays or {}) != compiled.ode["sweep_symbols"]:
            raise ValueError(f"var_arrays must sweep {compiled.ode['sweep_symbols']}, in the order "
                             "given when the RHS was compiled.")
        symbol_map = compiled.ode["symbol_map"]
        def remap(mapping, name):
            unknown = [s for s in mapping if s not in symbol_map]
            if unknown:
                raise ValueError(f"{name} contains symbols unknown to the compiled RHS: {unknown}")
            return {symbol_map[s]: value for s, value in mapping.items()}
        sweeps = remap(var_arrays or {}, "var_arrays") or {_DUMMY_SWEEP: np.zeros(1)}
        runtime = remap(runtime_consts or {}, "runtime_consts")
        has_observable = compiled.ode["has_observable"]
        mode = output_mode if has_observable else {"final": "final_rho", "mean": "mean_state"}[output_mode]
        return _solve_symbolic_system(
            None, None, None, None, tlist, var_arrays=sweeps, runtime_consts=runtime,
            rho0_values=y0_values, output_mode=mode, solver=solver, fp64=fp64,
            return_time_trace=return_time_trace, time_trace_every=time_trace_every,
            t_in=t_in, return_device=return_device, calibrate_timesteps=calibrate_timesteps,
            compiled_rhs=compiled, _ode_size=compiled.state_size, _state_trace=not has_observable,
            **option_set)

    if state_symbols is None:
        raise TypeError("odesolve_2D needs state_symbols (or a CompiledRHS as rhs).")
    if y0 is None and y0_values is None:
        raise ValueError("Provide y0 or y0_values.")
    model = _prepare_ode_model(rhs, state_symbols, y0, var_arrays, const_values, runtime_consts,
                               keep_symbolic_consts, time_symbol, observable)
    for name, value in _ODE_CODEGEN_DEFAULTS.items():
        option_set.setdefault(name, value)
    mode = output_mode if observable is not None else {"final": "final_rho", "mean": "mean_state"}[output_mode]
    return _solve_symbolic_system(
        model["H"], model["drive"], [], model["mean"], tlist,
        var_arrays=model["sweeps"], const_values=model["constants"], runtime_consts=model["runtime"],
        keep_symbolic_consts=model["keep"], rho0=model["initial"], rho0_values=y0_values,
        output_mode=mode, solver=solver, fp64=fp64,
        return_time_trace=return_time_trace, time_trace_every=time_trace_every,
        t_in=t_in, return_device=return_device, calibrate_timesteps=calibrate_timesteps,
        _ode_size=model["size"], _state_trace=observable is None,
        **option_set)

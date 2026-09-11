"""Direct symbolic ODE input for GQIS's fixed-step CUDA sweep engine."""

import numpy as np
import sympy as sp


def odesolve_2D(rhs, state_symbols, y0, tlist, *, var_arrays=None, const_values=None,
                 time_symbol=None, observable=None, output_mode="final", y0_values=None,
                 runtime_consts=None, keep_symbolic_consts=None, solver="rk4", fp64=False,
                 return_time_trace=False, time_trace_every=None, t_in=0.0,
                 return_device=False, **options):
    """Solve real first-order ODEs over zero, one or two parameter-sweep axes.

    rhs: Sequence/column matrix of dy/dt expressions in state_symbols order.
        Explicit Eq(Derivative(state, time, evaluate=False), rhs) is also accepted.
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

    Results have shape (nx, ny, nstate) without an observable, otherwise (nx, ny).
    State traces have shape (nx, ny, nsamples, nstate); observable traces omit
    nstate. Unused sweep axes have length one. y0_values uses (nx, ny, nstate),
    or (nx, nstate) for one axis. FP32/RK4 are defaults; solver selects a GQIS integrator.
    Code-generation, caching, runtime constants, timing and trace options are
    shared with mesolve_2D. No density-matrix or trace constraints are imposed.
    """
    states = tuple(state_symbols)
    if not states or any(not isinstance(s, sp.Symbol) for s in states) or len(set(states)) != len(states):
        raise ValueError("state_symbols must contain distinct SymPy Symbols.")
    if any(s.is_real is False for s in states):
        raise ValueError("State symbols must represent real variables.")
    expressions = [sp.sympify(e) for e in rhs]
    if len(expressions) != len(states):
        raise ValueError("Provide one RHS expression per state symbol.")
    if output_mode not in {"mean", "final"}:
        raise ValueError("output_mode must be 'mean' or 'final'.")
    if y0 is None and y0_values is None:
        raise ValueError("Provide y0 or y0_values.")
    if y0 is not None and y0_values is not None:
        raise ValueError("Use either y0 or y0_values, not both.")
    if y0_values is not None and np.iscomplexobj(y0_values):
        raise ValueError("y0_values must be real; split complex states into real/imaginary parts.")
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
        sweeps = {sp.Symbol("ode_dummy", real=True): np.zeros(1)}
    if keep_symbolic_consts is not None and not isinstance(keep_symbolic_consts, str):
        keep_symbolic_consts = [replacements[s] for s in keep_symbolic_consts]
    if any(key.startswith("_") or key in {"rho0", "rho0_values", "rho0_var_arrays"} for key in options):
        raise ValueError("Use y0/y0_values and var_arrays for ODE initial conditions.")
    from .solver import _solve_symbolic_system
    options.setdefault("pre_expand", False)
    options.setdefault("collect_rho", False)
    mode = output_mode if observable is not None else {"final": "final_rho", "mean": "mean_state"}[output_mode]
    drive = {stage_time: sp.Symbol("t", real=True)} if time_symbol in all_symbols else {}
    return _solve_symbolic_system(
        sp.Matrix(expressions), drive, [], sp.Matrix([obs]), tlist,
        var_arrays=sweeps, const_values=remap(constants), runtime_consts=remap(runtime),
        keep_symbolic_consts=keep_symbolic_consts, rho0=initial, rho0_values=y0_values,
        output_mode=mode, solver=solver, fp64=fp64,
        return_time_trace=return_time_trace, time_trace_every=time_trace_every,
        t_in=t_in, return_device=return_device,
        _ode_size=len(states), _state_trace=observable is None, **options)

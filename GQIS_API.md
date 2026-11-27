# GQIS API Reference

This technical reference documents the application programming interface (API) of the GPU Quantum Interferometry
Solver (GQIS). The Lindblad interface is in `gqis/solver.py`, and the direct ODE interface is in
`gqis/gqis_ode_api.py`. The [README](./README.md#minimal-use) introduces the basic solver calls;
the examples provide working applications of these APIs.

```python
from gqis import mesolve_2D, odesolve_2D
```

There are two solver entry points: `mesolve_2D` for Lindblad problems and `odesolve_2D` for general real
ODEs. The package also exports functions that can be used separately
from the solvers:

- `compile_lindblad_rhs` and `compile_ode_rhs` prepare and optimize the ODE once, without the GPU, and
  return a `CompiledRHS` that both solvers accept; see [compiling the ODE once](#compiling-the-ode-once).
- `build_independent_rho` and `build_reduced_lindblad_rhs` work with the reduced Lindblad equations; see
  [density-matrix helpers](#density-matrix-helpers).

The remaining helpers, documented at the end, expose the symbolic and CUDA code-generation stages.
They are available from `gqis.solver` for inspection and advanced development, but they may change
before version 1.0.

## `mesolve_2D`

Build and solve a finite-dimensional Lindblad master equation over one or two parameter axes. One CUDA thread
integrates one parameter combination with the selected method. GQIS symbolically generates the
right-hand side (RHS), meaning the time derivatives, of the reduced ordinary differential equation (ODE) system and
compiles it as a CUDA kernel.

### Minimal Call

Assuming the symbolic model and numerical arrays have already been defined, the
smallest call with one sweep axis is:

```python
result = mesolve_2D(
    H, Drive, Col_Ops, mean_operator, tlist,
    var_arrays={eps: eps_values},
)
```

The required positional arguments `H`, `Drive`, `Col_Ops`,
`mean_operator`, and `tlist` specify the Hamiltonian, time-dependent drive,
collapse operators, measured operator, and time samples, respectively. At least one sweep axis in
`var_arrays` is required as well. This call returns shape `(len(eps_values), 1)`; a
call with two sweep axes uses `var_arrays={eps: eps_values, A: amplitude_values}` and
returns shape `(len(eps_values), len(amplitude_values))`. With `calibrate_timesteps=True`, GQIS also chooses
the time step from the convergence of the result, and `tlist` then only gives the duration and a starting grid;
see [automatic time-step calibration](#automatic-time-step-calibration).

Notation used below:

- `N` is the Hilbert-space dimension: the number of basis states or quantum
  levels represented by the Hamiltonian. Therefore `H`, every collapse
  operator, `mean_operator`, and a symbolic initial density matrix are `N x N`.
- `num_X` and `num_Y` are the numbers of independent simulations along the two
  axes of the parameter sweep.
- `1D` means a one-dimensional array.

### Physical Model

These five positional arguments are required.

| Parameter | Type | Meaning |
| --- | --- | --- |
| `H` | `N x N` SymPy matrix | Hamiltonian. `H` may contain sweep symbols, constant symbols, and one or more drive placeholder symbols. |
| `Drive` | SymPy expression or `dict[Symbol, Expr]` | Time-dependent signal. A single expression supplies the conventional `Symbol("Drive")` placeholder used by `H`. A dictionary explicitly maps multiple Hamiltonian placeholder symbols to separate signals. Expressions may use `t`, sweep parameters, and constants. |
| `Col_Ops` | sequence of `N x N` SymPy matrices | Lindblad collapse operators. Use `[]` for closed-system evolution. Every operator must have the same shape as `H`. |
| `mean_operator` | `N x N` SymPy matrix | Operator whose expectation value is averaged, returned at the final time, or sampled as a trace. It must match `H`. |
| `tlist` | 1D array | Finite, strictly increasing, uniformly spaced elapsed times beginning at zero. `np.linspace(0, duration, n_steps + 1)` supplies `n_steps` updates ending at absolute time `t_in + duration`. |

### Sweeps And Constants

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `var_arrays` | `dict[Symbol, 1D array]`, required | One or two sweep axes are required. Dictionary insertion order maps the first array to result axis X and the second to Y. For a single parameter point, supply a one-element dummy axis. Array values and lengths can change while reusing the generated ODE. |
| `const_values` | `dict[Symbol, number]`, optional | Constants substituted as numbers while generating the ODE and CUDA code. If one of these values changes on a later call, GQIS normally generates and compiles a new ODE kernel. |

### Initial State And Output

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `rho0` | `N x N` SymPy matrix, reduced expression list, or `None` | Initial density matrix at `t=t_in`. It may contain symbols swept through `rho0_var_arrays` or `var_arrays`, and it may use runtime constants. `None` initializes state `\|0><0\|`. See [initial-state representations](#initial-state-representations). |
| `rho0_var_arrays` | `dict[Symbol, 1D array]` | Sweep values for symbols used only in the symbolic `rho0` matrix. They are merged with `var_arrays`; the total remains limited to two sweep axes. |
| `rho0_values` | numeric array or `None` | Alternative to symbolic `rho0`: one explicit reduced initial state per simulation point. Shape is `(num_X, N*N-1)` for one sweep axis or `(num_X, num_Y, N*N-1)` for two axes. Values can change without recompilation. Cannot be combined with `rho0`. |
| `t_in` | `0.0` | Absolute initial time of the supplied state. Drive evaluations and returned trace times include this offset. |
| `output_mode` | `"mean"` | `"mean"` averages the expectation value over the solver steps after the warmup; `"final"` returns the expectation value at the final time; `"final_rho"` returns the final reduced real density vector in the ordering described under `build_independent_rho`. |
| `warmup_time` | `0.0` | Initial fraction of time excluded from the time average, in `[0, 1]`. This can suppress transient dependence on the initial state without shortening the simulated evolution. A value of `1` leaves no averaging window and returns the final expectation value. |
| `return_time_trace` | `False` | Also return sampled expectation values and their absolute times. Samples begin at `t=t_in+dt`, after the first integration step. |
| `time_trace_every` | `None` | Store one trace value every specified number of solver steps. With stride `k`, stored times are `dt`, `(k+1)*dt`, `(2k+1)*dt`, and so on. Mutually exclusive with `time_trace_samples_per_period`. |
| `time_trace_samples_per_period` | `None` | Number of trace samples to store per drive period (approximate). Requires `solver_samples_per_period`; the integer stride is `max(1, round(solver_samples_per_period / time_trace_samples_per_period))`. |
| `solver_samples_per_period` | `None` | Number of solver integration steps per drive period, used only to convert trace density to a stride. |
| `return_device` | `False` | Return CuPy result/trace arrays instead of copying them to NumPy. Trace times remain NumPy metadata. |

#### Initial-State Representations

The initial state can be given in three ways:

- **Default:** without `rho0` and `rho0_values`, every simulation starts in `|0><0|`.
- **Symbolic `rho0`:** one SymPy density matrix. With numbers only, it is common to all simulations. With
  symbols swept through `rho0_var_arrays` or `var_arrays`, it gives a sweep over initial states. Use it when
  the state has a convenient analytic form.
- **Numerical `rho0_values`:** one reduced vector for every simulation point. Use it when the states have
  been calculated numerically, for example to continue from the result of a previous call.

Each of them specifies one state at `t=t_in` for every independent simulation.

A Hermitian `N x N` density matrix contains `N*N` independent real values, and the unit trace
condition removes one of them. GQIS therefore evolves and stores only `N*N - 1` real values per state:

1. the first `N - 1` diagonal populations
2. the real and imaginary parts of the upper-triangular coherences

The final diagonal population and lower-triangular coherences are reconstructed
from unit trace and Hermiticity. This is why the last dimension of
`rho0_values` is `N*N - 1`, rather than `N*N`.

### Returns

Without optional trace/timing values, the result is a NumPy array:

- `output_mode="mean"` or `"final"`: complex shape `(num_X, num_Y)`.
- `output_mode="final_rho"`: real shape `(num_X, num_Y, N*N-1)`.
- A sweep over one axis retains a singleton Y dimension.

With `return_time_trace=True`, the return is
`(result, trace, trace_t)`, where `trace` has complex shape
`(num_X, num_Y, num_trace)`. With `return_timing_info=True`, `timing_info` is
appended to that tuple; without a trace the return is `(result, timing_info)`.

### Validation And Limitations

- At least one sweep array is required, even for a single simulation; use a
  one-element dummy axis when no physical parameter is swept.
- GQIS validates matrix dimensions, the structure of the time grid, option combinations,
  the shapes of explicit initial-state arrays, and finite output values.
- GQIS does not test whether a supplied symbolic or reduced initial state is
  positive semidefinite. The caller is responsible for providing a physical
  density matrix. For matrix input, the last population and lower triangle are
  reconstructed rather than independently validated.
- The elapsed time grid is uniform and begins at zero.
- Kernel reuse is limited to one Python process; the cache of compiled kernels is not written to
  disk. A new Python process compiles its first model again.
- Practical Hilbert-space size is limited by the cost of symbolic generation, NVRTC
  compilation, CUDA register pressure, and GPU memory.

## `odesolve_2D`: direct SymPy ODE sweeps

This entry point skips Lindblad extraction and uses the same CUDA solvers,
precision options and caches. The number of state variables follows the supplied equations,
without density-matrix constraints.

```python
import numpy as np
import sympy as sp
from gqis import odesolve_2D

x, v, t, gamma, amplitude = sp.symbols("x v t gamma amplitude", real=True)
rhs = [v, -gamma*v - x - x**3 + amplitude*sp.cos(t)]
final, evolution, times = odesolve_2D(
    rhs, [x, v], [0, 0], np.linspace(0, 20, 2049),
    var_arrays={gamma: np.linspace(0.05, 0.5, 32), amplitude: np.linspace(0.1, 1, 32)},
    return_time_trace=True, time_trace_every=16)
```

Supply one `rhs` expression and one initial value in `y0` for each entry of `state_symbols`.
The symbol named `t` is the time by default; `time_symbol` selects another one.
Use SymPy symbols for states, not applied functions. An explicit equation such as
`Eq(Derivative(x, t, evaluate=False), -x)` is also accepted. States and derivatives must
be real; split complex equations into real and imaginary components. Expressions must
be printable as CUDA C (ordinary arithmetic, trigonometric functions and `Piecewise`,
for example); arbitrary Python callbacks and implicit equations are not supported.

`var_arrays` accepts zero, one or two parameter axes in insertion order. Other parameters
belong in `const_values` or `runtime_consts`. Initial expressions may depend on parameters;
alternatively, pass `y0=None` and `y0_values` with shape `(nx, ny, nstate)` or `(nx, nstate)`
for one axis. The uniform `tlist` must start
at zero; its length minus one is the number of integration steps. It describes elapsed time.
Set `t_in` to the absolute time of the initial state; the interval ends at `t_in + tlist[-1]`.
`calibrate_timesteps` chooses the time step automatically by the same procedure as in `mesolve_2D`; see
[automatic time-step calibration](#automatic-time-step-calibration).

| Option | Returned data |
| --- | --- |
| `output_mode="final"` (default) | Final real state, shape `(nx, ny, nstate)` |
| `output_mode="mean"` | Average real state over step endpoints, same shape |
| `observable=expression` | Instead return a scalar observable, shape `(nx, ny)`, complex dtype |
| `return_time_trace=True` | Return `(result, trace, trace_times)`; traces add a sample axis before the state axis |

Unused sweep axes have length one. `time_trace_every=k` samples at `t_in + dt`, `t_in + (k+1)*dt`, etc.;
the trace does not include the initial state, and it includes the final time only if the final step falls on
this sampling pattern.
With an observable, the trace has shape `(nx, ny, nsamples)`. `warmup_time` is the fraction
of integration steps discarded from the mean, not a duration; it does not trim traces.
At `warmup_time=1`, the mean falls back to the final result. `return_timing_info=True`
appends a timing dictionary to the return values, as for `mesolve_2D`.

For solver and precision choices, see [solver options](#solver-and-precision-options).
Optional settings may also be passed as [grouped options](#grouped-options).
Means of the full state need an accumulator vector; storage of sampled evolution grows with trajectory
count, sample count and state dimension. Use sparse trace sampling for large sweeps.

## Solver, Precision And Time Step

### Solver And Precision Options

These options are common to `mesolve_2D` and `odesolve_2D`. Begin with the default RK4 and let
[`calibrate_timesteps`](#automatic-time-step-calibration) choose the time step, or refine the time grid
yourself; compare methods when you need lower runtime at your target accuracy.

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `solver` | `"rk4"` | CUDA integrator: `"rk4"`, `"lserk4"`, `"lserk4f"`, `"dp5"`, `"ab5"`, `"anas5"`, `"tsit5"`, `"alshina6"`, `"dop853"`, or `"tsit5a"` with adaptive step size; see [available solvers](#available-solvers). Only the selected solver body is inserted into the generated CUDA source. |
| `solver_frequency` | `None` | Non-negative angular frequency `w` for fixed-step `"anas5"`. Before launch, its fitted coefficient is calculated from `v = w*dt` and then remains constant throughout the kernel. Anas5 defaults to `w=1` when omitted; supply the dominant frequency for meaningful fitted behavior. Other solvers ignore it. |
| `adaptive_options` | `None` | Settings dictionary of `"tsit5a"`; see [adaptive step size](#adaptive-step-size-tsit5a). |
| `fp64` | `False` | Use 64-bit floating-point (FP64) state, sweep arrays, constants, generated math, and output instead of the faster 32-bit floating-point (FP32) path. |
| `calibrate_timesteps` | `None` | `True` or a settings dictionary chooses one uniform time step before the run; see [automatic time-step calibration](#automatic-time-step-calibration). |

### Available Solvers

The `solver` option selects the same methods in `mesolve_2D` and `odesolve_2D`.
The implementations have the following roles and costs per step after startup. Counts of work vectors
exclude the state,
static terms, and drive values; NVRTC may further scalarize or reuse them.

| Class | Solver | Order | New RHS/step | Work vectors | Main advantage | Main limitation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| General baseline | `rk4` — classical Runge-Kutta | 4 | 4 | 3 | Standard reference method; a good default. | Lower order than the high-accuracy methods. |
| General low-storage | `lserk4` — Carpenter-Kennedy 2N54 | 4 | 5 | 2 | Two work vectors help reduce register pressure for large systems. | One more RHS evaluation than RK4. |
| General low-storage | `lserk4f` — `lserk4` with the RHS fused into the residual update | 4 | 5 | 1 | No separate derivative vector: the lowest register use, for the largest systems. Same results as `lserk4` to rounding; switching between them reuses the generated ODE. | Same RHS cost as `lserk4`. |
| General fifth-order | `dp5` — Dormand-Prince 5(4) main formula | 5 | 6 | 5 | Efficient at tight accuracy limits. | Embedded estimate is unused; more work than RK4. |
| General fifth-order | `tsit5` — Tsitouras 5(4) main formula | 5 | 6 | 5 | Small error coefficients; the same method as Julia GPUTsit5 and dynamiqs Tsit5. | Embedded estimate is unused; more work than RK4. |
| General high-order | `alshina6` — Alshina sixth-order formula | 6 | 7 | 5 | Sixth order with seven stages. | More work per step than the fifth-order methods. |
| General high-order | `dop853` — Dormand-Prince 8(5,3) main formula | 8 | 12 | 6 | Coarser time grids at the same accuracy. | More stages and work vectors; roundoff can limit gains at fine steps. |
| Periodic fitted | `anas5` — frequency-fitted Anas5(w) | 4 generally, 5 for accurate `w` | 6 | 5 | Reduced phase error when the dominant frequency is known. | Loses fifth-order fitting when `w` is inaccurate or the signal is broadband. |
| Multistep | `ab5` — Adams-Bashforth 5 | 5 | 1 after startup | 7 during RK4 startup, then 5 | Very low RHS cost per step after startup on smooth trajectories. | Smaller stability region, startup cost, and storage of previous derivatives. |
| Adaptive step size | `tsit5a` — Tsitouras 5(4) with its embedded error estimate | 5 | 6 (+6 per rejected step) | 7 | Each system takes its own steps between the `tlist` output points. | GPU threads run in groups of 32 (warps), and each group runs as many steps as its slowest parameter point; two more work vectors than `tsit5`. |

All methods except `tsit5a` use the uniform `dt` supplied by `tlist` and reconstruct base time from the
integer step index rather than accumulating `t += dt`. Alshina6 and DOP853 add each stage derivative to the later stages as soon as it is calculated, so its
storage can be reused for the next stage. Derivative and drive values are reused where the method
permits. DOP853 uses the eighth-order update without the embedded error estimates or dense-output stages.
Anas5(w) uses the supplied frequency and fixed step size to set its fitted coefficient.

The [Benchmark 03 accuracy sweeps](BENCHMARKS.md#benchmark-03-accuracy-and-convergence) show how the accuracy
depends on the method and the time step for the two- and four-level examples.

#### Adaptive Step Size: `tsit5a`

With `solver="tsit5a"`, `tlist` gives only the output times: every parameter point chooses its own steps
between them, and `mean` outputs integrate over those steps with the trapezoidal rule. Its settings are passed
as `adaptive_options=dict(...)`. Without `calibrate_timesteps`, the step tolerances below are used as given;
with it, they are calibrated, see [automatic time-step calibration](#automatic-time-step-calibration).

| Setting | Default | Meaning |
| --- | --- | --- |
| `step_rtol`, `step_atol` | `1e-5` | Relative and absolute error tolerances of one step. The error of the final result is larger than these values. |
| `max_substeps` | `64` | Largest number of steps, accepted and rejected, between two output times; exceeding it gives a non-finite result. |
| `fp64_time` | `True` | Keep the time variable in FP64. |
| `count_steps` | `False` | Add the accepted and rejected step counts of every parameter point to the timing dictionary as `adaptive_steps`. |

### Automatic Time-Step Calibration

`calibrate_timesteps` chooses the time step (for `tsit5a`, the step tolerances) from the convergence of the
result. With a fixed step size, `tlist` then sets only the duration and the starting grid:

```python
result = mesolve_2D(H, Drive, Col_Ops, mean_operator, tlist, var_arrays=sweep,
                    calibrate_timesteps=dict(rms_atol=1e-3, max_atol=1e-2, sample_stride=8))
```

GQIS first works on a reduced set of sweep points (every `sample_stride`-th value of each axis). It doubles
the step count until the estimated error of the result is below the limits, then searches for the coarsest
step count that still passes. The full grid is calculated once with that step count. The accepted step is
stored and reused by later calls with the same model and settings.

The exact result is unknown, so the error of a run with `M` steps is estimated from its change against the
run with `2M` steps: `|r_M - r_2M| * 2**p / (2**p - 1)` at every test point, before the RMS and maximum
are taken. The finer run still has `1/2**p` of the error, so the change alone would understate it; the
factor corrects for this. `p` is the order of the method (factor 16/15 for RK4). For the mean output of
fixed-step solvers `p = 1` (factor 2), because the average over the step endpoints converges at first
order. In the search between two step counts, each candidate `M` is compared with the finest run `M_ref`,
with the factor `1 / (1 - (M / M_ref)**p)`.

With `solver="tsit5a"` the same search refines the step tolerances instead of the step count: starting from
`adaptive_options`, it divides `step_rtol` and `step_atol` together by 10, keeping their ratio, until the
estimated error passes, then searches for the loosest tolerance that still passes. `tlist` keeps its output times. The error of the result is taken
as proportional to the tolerance (`p = 1` with a refinement factor of 10, so the factor is 10/9).

| Setting | Default | Meaning |
| --- | --- | --- |
| `rms_atol`, `rms_rtol` | `1e-3`, `0` (`odesolve_2D`: `1e-4`, `0`) | Limit for the RMS change of the result: `rms_atol + rms_rtol * RMS(result)`. |
| `max_atol`, `max_rtol` | `1e-2`, `0` (`odesolve_2D`: `1e-3`, `0`) | Limit for the largest pointwise change: `max_atol + max_rtol * max(abs(result))`. Setting both values of a pair to zero skips that test. |
| `sample_stride` | `8` | Test every `k`-th value on each sweep axis, plus the last value; `1` tests the full grid. |
| `max_steps` | `2**20` | Largest step count tried. |
| `min_step_tol` | `1e-12` | Smallest step tolerance tried with `tsit5a`. |
| `bisection_steps` | `2` | Bisections between the last failing and the first passing step count. The accepted grid has at most `2x` (`0`), `1.41x` (`1`) or `1.19x` (`2`) more steps than the limits require. |
| `full_grid_bisection` | `False` | When the test points are the full grid, the first passing result is returned without bisection. `True` bisects anyway, giving a coarser stored step for repeated calls. |
| `recalibrate_every` | `None` | `None` calibrates once; an integer `k` calibrates again on every `k`-th call. A recalibration never makes the step coarser. |
| `reset` | `False` | Discard the stored step and calibrate again. |
| `on_fail` | `"warning"` | When the limits are not reached within `max_steps` (`min_step_tol` or `max_substeps` with `tsit5a`), or refinement stops reducing the change (for example at the FP32 rounding floor): `"warning"` continues with the best step found, `"ignore"` does the same without a message, `"error"` stops with `RuntimeError`. |
| `range_warning` | `True` | Warn once when a later call reuses the stored step with a sweep range beyond the calibrated one; `False` disables the warning. |

`calibrate_timesteps=True` uses these defaults. For `tsit5a`, "step" in this table means the step tolerance.

- **Repeated calls:** changing the model, fixed constants, solver, precision, output settings or calibration
  settings starts a new calibration. Changing sweep values or runtime constants does not, so for animations
  put the most demanding frame first or use `recalibrate_every`.
- **Short intervals:** the test measures the error of the calculated interval only. For short continuation
  chunks, calibrate once over a longer interval first, as in
  [Example 06](Examples/Example_06_symbolic_ode_sweep.py).
- **Mean output:** with a fixed step size the mean is an average over the step endpoints, which converges
  at first order in the step size; the error estimate therefore uses `p = 1` for it. With `tsit5a` the mean is a trapezoidal
  average over the accepted steps. When the solution is smooth and `tlist` is coarse, the steps reach the
  output interval and this quadrature error no longer decreases with the tolerance; use a finer `tlist`.
- **Time traces:** only the result is tested, not the trace. With a fixed step size the step count stays a
  power-of-two multiple of the `tlist` grid, and the trace keeps the sample times of `tlist`. With `tsit5a`
  the trace is at the `tlist` output times anyway.
- **Reporting:** `timings=True` prints a one-line calibration summary; with `return_timing_info=True` the
  timing dictionary reports the used step count (or, for `tsit5a`, `step_tol`, `step_rtol` and `step_atol`)
  under `"timestep_calibration"`.

## Compiling The ODE Once

| Name | Purpose |
| --- | --- |
| `compile_lindblad_rhs(H, Drive, Col_Ops, mean_operator, sweep_symbols, ...)` | Generate and optimize the reduced Lindblad equations of a model. |
| `compile_ode_rhs(rhs, state_symbols, y0, sweep_symbols, ...)` | Optimize a general SymPy ODE system. |
| `CompiledRHS` | Result of either function; `mesolve_2D` or `odesolve_2D` accepts it in place of the model. |

Preparation has four stages: symbolic ODE generation, its optimization, CUDA code generation and
kernel compilation. The first two need only SymPy and can be run separately, without the GPU:

```python
from gqis import compile_lindblad_rhs, compile_ode_rhs

rhs = compile_lindblad_rhs(H, Drive, Col_Ops, mean_operator, [eps, A],
                           const_values={gamma: 0.01})
print(rhs.timings)  # {'ode_prepare_s': ..., 'ode_optimize_s': ...}
result = mesolve_2D(rhs, tlist=tlist, var_arrays={eps: eps_values, A: amplitude_values})

ode = compile_ode_rhs(rhs_exprs, states, y0, [a], const_values={b: 0.5})
state = odesolve_2D(ode, tlist=tlist, var_arrays={a: a_values})
```

Both functions take the model arguments of their solver, the swept symbols in the order of the later
`var_arrays`, the constants and the symbolic code-generation options. The solver call with the result
starts from CUDA code generation: give `tlist` by keyword, the sweep arrays, and any solver, output
and diagnostic options. Constants, the symbolic initial state, the observable and the code-generation
options are fixed at compilation, so passing them again raises `ValueError`; values of constants kept
symbolic can still be changed through `runtime_consts`.

The returned `CompiledRHS` holds the prepared equations and observable as SymPy expressions and the
stage `timings`, for analysis or for another code generator.

## Summary

Use `mesolve_2D` for a symbolic Lindblad model and `odesolve_2D` for real first-order ODEs.
Supply the equations, a uniform grid of elapsed times and the sweep axes; choose initial values and the
required final, mean or sampled output. `mesolve_2D` and `odesolve_2D` default to fixed-step RK4 and FP32.
Check convergence for the observable you need before increasing the size of the parameter grid.

The following sections describe optional settings: GPU continuation, reuse in repeated calls, grouped options,
code generation and diagnostics, followed by the helper functions.

## Optional Execution And Inspection

### Continuing On The GPU

A long evolution can be calculated in several calls without copying the state to the CPU between them.
Both `mesolve_2D` and `odesolve_2D` have two options for this:

- `return_device=True` returns the result and trace arrays as CuPy arrays on the GPU; the trace times stay
  a NumPy array.
- `t_in` sets the absolute time at which the call starts (default `0.0`).

Request the full state as output, which is `output_mode="final_rho"` in `mesolve_2D` and the default
output in `odesolve_2D`, and pass it to the next call through `rho0_values` or `y0_values`. For example:

```python
state = odesolve_2D(rhs, states, y0, elapsed_times, t_in=10.0, return_device=True, **model_options)
state = odesolve_2D(rhs, states, None, elapsed_times,
                    y0_values=state, t_in=10.0 + elapsed_times[-1],
                    return_device=True, **model_options)
```

Here `states` is the ordered list of state symbols and `model_options` contains the same sweep axes,
constants and solver settings for both calls. Checks for finite values still synchronize a scalar status;
they do not copy the full state.
[Example 06](Examples/Example_06_symbolic_ode_sweep.py) demonstrates this continuation pattern
with a sweep over initial conditions of the Duffing oscillator and a live phase-space display.

Stage times use `t_in + elapsed_stage_time`, with elapsed time derived from the integer step index.
Changing `t_in` does not require another compilation.
FP32 still rounds the
absolute time to its representable precision; use FP64 when small intervals at large times need it.

### Reusing The ODE In Repeated Calls

By default (`RHSreuse=True`), GQIS reuses the generated right-hand side (RHS) of the
ODE and its compiled CUDA kernel across repeated calls, for example in parameter fitting, repeated
sweeps and animations. The symbolic equation structure must remain unchanged; only the numerical values
assigned to selected symbols may change. Constants that change between calls therefore need to
remain symbolic. For example:

```python
result = mesolve_2D(
    H, Drive, Col_Ops, mean_operator, tlist,
    var_arrays={eps: eps_values, A: amplitude_values},
    const_values={gamma: gamma_value},
    keep_symbolic_consts={gamma},
)
```

On the next call, `gamma_value` may change while the generated ODE and
compiled kernel are reused. Without `keep_symbolic_consts={gamma}`, the value is
inserted directly into the generated equations and changing it requires new
symbolic generation and compilation. `runtime_consts={gamma: value}` can also
supply or override the value of a constant kept symbolic this way.

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `runtime_consts` | `dict[Symbol, number]` | Values for constants left symbolic in the generated ODE, uploaded for each call. A value here overrides the same symbol in `const_values`. |
| `keep_symbolic_consts` | iterable, `"all"`, `"auto"`, `"const_values"`, or `None` | Selects which `const_values` symbols remain symbolic instead of being inserted as fixed numbers. `"all"`, `"auto"`, and `"const_values"` select every `const_values` key. |
| `auto_runtime_consts` | `False` | Keep every `const_values` key symbolic. |

### Grouped Options

Optional settings can also be passed in groups by topic. Each group is a dictionary that uses the
usual option names, so the two styles are equivalent:

```python
result = mesolve_2D(
    H, Drive, Col_Ops, mean_operator, tlist,
    var_arrays={eps: eps_values, A: amplitude_values},
    output=dict(warmup_time=0.2, return_time_trace=True, time_trace_every=16),
    diagnostics=dict(timings=True, return_timing_info=True, ignore_non_finite_output=True),
)
```

Options that are not set keep their defaults.

| Group | Options |
| --- | --- |
| `consts` | `const_values`, `runtime_consts`, `keep_symbolic_consts`, `auto_runtime_consts` |
| `initial` | `rho0`, `rho0_var_arrays`, `rho0_values` (Lindblad); `y0_values` (ODE); `t_in` |
| `output` | `warmup_time`, `return_time_trace`, `time_trace_every`, `time_trace_samples_per_period`, `solver_samples_per_period`, `return_device` |
| `solver_options` | `solver`, `fp64`, `solver_frequency`, `adaptive_options` |
| `codegen` | `RHSreuse`, `kernel_template_file`, `unroll`, `pre_expand`, `collect_rho`, `factor_terms`, `cse_batch_size`, `cse_simplify`, `hoist_rho_independent`, `symbolic_workers`, `max_symbolic_workers`, `nvrtc_options` |
| `diagnostics` | `timings`, `return_timing_info`, `Actual_Kernel_Save`, `beep_on_error`, `ignore_non_finite_output` |

`odesolve_2D` accepts the same groups. Give each option either directly or inside its group, not both;
unknown names and conflicting values raise `ValueError`. Model inputs, `var_arrays`, `output_mode`,
`observable` and [`calibrate_timesteps`](#automatic-time-step-calibration) are always passed directly.

### Code Generation And Execution

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `kernel_template_file` | `None` | Uses the canonical CUDA template packaged with GQIS. Supply an explicit path only to test or develop a custom kernel template; relative explicit paths are checked in the current directory and then inside the installed package. |
| `RHSreuse` | `True` | Reuse the generated symbolic RHS independently of the solver-specific compiled CUDA kernel. Different solvers therefore share one symbolic RHS while retaining separate kernels. `False` regenerates both stages on every call. |
| `unroll` | `False` | `True` forces `#pragma unroll` on solver vector loops. `False` leaves the decision to NVRTC, preserving the existing behavior and avoiding mandatory unrolling when larger density vectors would create excessive register pressure. Changing this option produces a separately compiled cached kernel. |
| `pre_expand` | `True` | Expand symbolic RHS expressions before code generation. |
| `collect_rho` | `False` | Collect RHS terms by reduced density variables. Usually not needed: the Lindblad equations are already simplified for each state variable. |
| `factor_terms` | `False` | Factor symbolic terms before common-subexpression elimination (CSE). This can reduce operations but increase preparation time. |
| `cse_batch_size` | `None` | Equations per batch of common-subexpression elimination. `None` performs global CSE; smaller batches can lower CUDA register pressure. |
| `cse_simplify` | `False` (`mesolve_2D`), `True` (`odesolve_2D`) | Simplify expressions during CSE. Lindblad equations are already simplified when they are generated; user-supplied ODE expressions are not, so `odesolve_2D` simplifies them here. `False` shortens the preparation of large ODEs that are already simple. |
| `hoist_rho_independent` | `True` | Evaluate expressions that do not depend on the state once per thread, or once per solver stage together with the drive. |
| `symbolic_workers` | `"auto"` | Number of parallel processes that build the symbolic Lindblad equations. `"auto"` uses one process below Hilbert-space dimension 10 and up to `max_symbolic_workers` from there on; an integer sets the number. For large Hilbert spaces, parallel execution shortens preparation several times. Not used by `odesolve_2D`. |
| `max_symbolic_workers` | `None` | Upper limit for `"auto"`; `None` uses half the logical CPUs. |
| `nvrtc_options` | empty tuple | Additional NVIDIA Runtime Compilation (NVRTC) options passed to `cupy.RawKernel`. A string or iterable of strings is accepted. |
| `Actual_Kernel_Save` | `False` | `True` saves `<caller>_Kernel.cu`; a string saves to that explicit path. Generated kernels are debugging artifacts and are ignored by the repository. |

### ODE Code Generation

The code-generation options above also apply to `odesolve_2D`, with these differences: nonlinear
expressions are not expanded or collected by default, expressions that depend only on time are evaluated
together with the drive at each solver stage, and combinations of runtime constants are precomputed on the CPU.

### Diagnostics

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `timings` | `False` | Print the preparation time with its stages, GPU kernel time, total call time, and whether the generated ODE and kernel were reused. |
| `return_timing_info` | `False` | Append a timing dictionary to the returned values, without console output. |
| `beep_on_error` | `False` | Play a best-effort notification before raising for non-finite output. |
| `ignore_non_finite_output` | `False` | Return not-a-number (NaN) or infinite (Inf) output instead of raising. Useful for accuracy sweeps that also record unstable working points. |

The timing dictionary contains:

| Key | Meaning |
| --- | --- |
| `rhs_stage_s` | All preparation before the kernel launch. |
| `ode_prepare_s`, `ode_optimize_s`, `code_emit_s`, `kernel_build_s` | Its stages: symbolic ODE, optimization, CUDA code generation, kernel compilation. A reused stage reports zero. |
| `rhs_codegen_s` | Sum of the first three stages. |
| `gpu_kernel_s`, `total_s` | GPU kernel time and total call time. |
| `cached_rhs` | `"hit"` or `"miss"` for reuse of the generated ODE; `"provided"` for a [compiled RHS](#compiling-the-ode-once). |
| `cached_kernel` | Reuse of the compiled kernel of the selected solver. |
| `solver`, `solver_frequency`, `unroll` | Settings used by the call. |

### Full Lindblad Call Signature

The complete signature includes the optional controls described above:

```python
mesolve_2D(
    H, Drive, Col_Ops, mean_operator, tlist,
    var_arrays=None, const_values=None,
    kernel_template_file=None, *,
    RHSreuse=True, runtime_consts=None, keep_symbolic_consts=None,
    auto_runtime_consts=False, output_mode="mean", solver="rk4",
    solver_frequency=None, adaptive_options=None, unroll=False, fp64=False,
    pre_expand=True, collect_rho=False, factor_terms=False,
    cse_batch_size=None, cse_simplify=False,
    hoist_rho_independent=True, symbolic_workers="auto",
    max_symbolic_workers=None, nvrtc_options=(), timings=False,
    return_timing_info=False, warmup_time=0.0, rho0=None,
    rho0_var_arrays=None, rho0_values=None, return_time_trace=False,
    time_trace_every=None, time_trace_samples_per_period=None,
    solver_samples_per_period=None, Actual_Kernel_Save=False,
    beep_on_error=False, ignore_non_finite_output=False,
    t_in=0.0, return_device=False, calibrate_timesteps=None,
    consts=None, initial=None, output=None, solver_options=None,
    codegen=None, diagnostics=None,
)
```

A [compiled RHS](#compiling-the-ode-once) can replace the model arguments `H`, `Drive`, `Col_Ops` and
`mean_operator`.

## Density-Matrix Helpers

### `build_independent_rho(N)`

Input: Hilbert-space dimension `N`.

Output: `(rho, metadata)`. `rho` is an `N x N` Hermitian SymPy
matrix with unit trace, built from `N*N-1` real symbols. `metadata` contains `N`, `rho_syms`,
`num_diag`, `num_coherences`, and `vec_len`. Here `num_coherences` is the number
of upper-triangular coherence pairs.

The order of the reduced vector is:

1. Diagonal populations `rho[0,0]` through `rho[N-2,N-2]`.
2. Real and imaginary parts of each upper-triangular coherence in row-major order.

The last population is reconstructed from unit trace.

### `build_reduced_lindblad_rhs(N, H, Col_Ops, mean_operator, *, pre_expand=False, collect_rho=False, factor_terms=False, symbolic_workers=1, max_symbolic_workers=None)`

Inputs: Hilbert-space dimension, symbolic Hamiltonian, collapse operators,
observable operator, and optional symbolic simplification controls. `symbolic_workers` builds the
equations in parallel processes (`1` here by default; `"auto"` as in `mesolve_2D`), with identical results.

Output: `(drho_eqs, mean_re, mean_im, metadata)`. `drho_eqs` contains the
`N*N-1` real Lindblad ODE expressions in the ordering defined above. `mean_re`
and `mean_im` are the components of the expectation value. CUDA generation and the
benchmark Julia and central processing unit (CPU) solvers use this same function so their density-matrix
equations cannot diverge through separately maintained symbolic derivations.

### `rho_matrix_to_independent_exprs(rho0)`

Input: a square SymPy density matrix, or an iterable already containing
`N*N-1` reduced expressions.

Output: a simplified expression list in the exact ordering used by
`build_independent_rho`. Raises `ValueError` for a nonsquare matrix or invalid
length of the reduced vector.

## Symbolic Code-Generation Helpers

These functions are advanced interfaces. Their signatures may evolve before a
stable 1.0 release.

### `generate_unrolled_drho(...)`

Inputs: `N`, `H`, a default `Drive_symbol`, `Col_Ops`, `mean_operator`, optional
`drive_expr`, runtime constant symbols, and the same symbolic optimization
controls exposed by `mesolve_2D`.

Output: `(static_lines, drive_lines, drive_alias_lines, drho_lines, mean_line,
final_line, static_syms, drive_syms, hoisted_syms)`. These CUDA fragments and
symbol lists are inserted into the template by `mesolve_2D`.

### `cse_emit_c_lines(drho_exprs, rho_syms, *, batch_size=None, do_simplify=True, hoist_rho_independent=False, return_hoist=False)`

Inputs: reduced RHS expressions, reduced state symbols, CSE batch size,
simplification switch, and hoisting switches.

Output: CUDA assignment lines. If `return_hoist=True`, returns
`(hoisted_substitutions, lines)`.

### `emit_drive_code(drive_map, *, array_name="Drive_arr", inline_single_use_funcs=True)`

Inputs: a mapping from placeholders to expressions, the target CUDA array name, and a switch
that keeps math functions used only once inline.

Output: `(drive_lines, alias_lines, drive_symbols)`. `alias_lines` is retained
for compatibility and is currently empty.

### `tidy_c_lines(lines)`

Input: generated C/CUDA lines.

Output: cleaned lines using common FP32 CUDA math forms such as `sinf`, `cosf`,
and `sqrtf`, with trivial arithmetic removed.

### `my_ccode(expr)` and `MyCPrinter`

`my_ccode` takes one SymPy expression and returns CUDA-compatible C text.
`MyCPrinter` is the underlying SymPy printer for the 1999 C language standard (C99); its `_print_Float(expr)` method
emits an explicit FP32 literal.

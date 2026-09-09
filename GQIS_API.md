# GQIS API Reference

This technical reference documents the application programming interface (API) of the GPU Quantum Interferometry
Solver (GQIS). The Lindblad interface is in `gqis/solver.py`, and the direct ODE interface is in
`gqis/gqis_ode_api.py`. New users should begin with the [README](./README.md) and the examples.
Use `mesolve_2D` for Lindblad problems or `odesolve_2D` for general real ODEs. The other functions
expose the symbolic and CUDA code-generation stages for inspection or advanced customization.

```python
from gqis import build_independent_rho, build_reduced_lindblad_rhs, mesolve_2D, odesolve_2D
```

These four functions form the public package interface. The remaining helpers
documented below are available from `gqis.solver` for inspection and advanced
development, but they may change before version 1.0.

## `odesolve_2D`: direct SymPy ODE sweeps

This entry point skips Lindblad extraction and uses the same fixed-step CUDA solvers,
precision options and caches. The number of state variables follows the supplied equations,
without density-matrix constraints.
See [Example 06](Examples/Example_06_symbolic_ode_sweep.py) for live Duffing phase-space flow:
a dense cloud stays on the GPU while short solves advance it and a compact raster is sent to Matplotlib.

```python
import numpy as np
import sympy as sp
from gqis import odesolve_2D

x, v, t, gamma, amplitude = sp.symbols("x v t gamma amplitude", real=True)
rhs = [v, -gamma*v - x - x**3 + amplitude*sp.cos(t)]
final, evolution, times = odesolve_2D(
    rhs, [x, v], [0, 0], np.linspace(0, 20, 2049),
    var_arrays={gamma: np.linspace(0.05, 0.5, 32), amplitude: np.linspace(0.1, 1, 32)},
    time_symbol=t, solver="rk4", output_mode="final",
    return_time_trace=True, time_trace_every=16)
```

Supply one `rhs` expression and one initial value in `y0` for each entry of `state_symbols`.
Use SymPy symbols for states, not applied functions. An explicit equation such as
`Eq(Derivative(x, t, evaluate=False), -x)` is also accepted. States and derivatives must
be real; split complex equations into real and imaginary components. Expressions must
be printable as CUDA C (ordinary arithmetic, trigonometric functions and `Piecewise`,
for example); arbitrary Python callbacks and implicit equations are not supported.

`var_arrays` accepts zero, one or two parameter axes in insertion order. Other parameters
belong in `const_values` or `runtime_consts`. Initial expressions may depend on parameters;
alternatively, pass `y0=None` and `y0_values` with shape `(nx, ny, nstate)` or `(nx, nstate)`
for one axis. `time_symbol` defaults to the symbol named `t`. The uniform `tlist` must start
at zero; its length minus one is the number of integration steps. It describes elapsed time.
Set `t_in` to the absolute time of the initial state; the interval ends at `t_in + tlist[-1]`.

| Option | Returned data |
| --- | --- |
| `output_mode="final"` (default) | Final real state, shape `(nx, ny, nstate)` |
| `output_mode="mean"` | Average real state over step endpoints, same shape |
| `observable=expression` | Instead return a scalar observable, shape `(nx, ny)`, complex dtype |
| `return_time_trace=True` | Return `(result, trace, trace_times)`; traces add a sample axis before the state axis |

Unused sweep axes have length one. `time_trace_every=k` samples at `t_in + dt`, `t_in + (k+1)*dt`, etc.;
neither the initial state nor a nonaligned final endpoint is appended automatically.
With an observable, the trace has shape `(nx, ny, nsamples)`. `warmup_time` is the fraction
of integration steps discarded from the mean, not a duration; it does not trim traces.
At `warmup_time=1`, the mean falls back to the final result. `return_timing_info=True`
appends a timing dictionary to the return values, as for `mesolve_2D`.

`solver`, `solver_frequency`, `fp64`, `unroll`, caching, runtime constants and code-generation
options below also apply. Defaults are RK4 and FP32. Nonlinear expressions are not expanded
or collected by default. Common expressions are reused, parameter-only combinations are
hoisted out of integration, and time-only expressions are evaluated in the shared stage-drive
routine. Runtime-constant-only combinations use the existing CPU precomputation path.
Only the selected integrator is emitted. `cse_batch_size` can limit temporary lifetimes and
help reduce register pressure for larger systems.
Full-state means require an extra accumulator vector; evolution storage scales with
`nx * ny * nsamples * nstate`, so use sparse sampling on large grids.

### Continuing On The GPU

Both APIs accept `t_in=0.0` and `return_device=False`. With `return_device=True`, result and
trace arrays are CuPy arrays; trace times remain small NumPy metadata arrays. Pass the returned
state back through `y0_values` (ODE) or `rho0_values` (Lindblad) without downloading it. Finite-value
checks still synchronize a scalar status; they do not copy the full state. For example:

```python
state = odesolve_2D(rhs, states, y0, elapsed_times, t_in=10.0, return_device=True, **model_options)
state = odesolve_2D(rhs, states, None, elapsed_times,
                    y0_values=state, t_in=10.0 + elapsed_times[-1],
                    return_device=True, **model_options)
```

Here `states` is the ordered state-symbol list and `model_options` contains the same sweep axes,
constants and solver settings for both calls. Request full-state output when continuing a trajectory.
Stage times use `t_in + elapsed_stage_time`, with elapsed time derived from the integer step index.
Zero and nonzero offsets share one runtime-parameterized kernel, cached after first use.
Changing `t_in` does not require another compilation.
FP32 still rounds the
absolute time to its representable precision; use FP64 when small intervals at large times need it.

## `mesolve_2D`

```python
mesolve_2D(
    H, Drive, Col_Ops, mean_operator, tlist,
    var_arrays=None, const_values=None,
    kernel_template_file=None, *,
    RHSreuse=True, runtime_consts=None, keep_symbolic_consts=None,
    auto_runtime_consts=False, output_mode="mean", solver="rk4",
    solver_frequency=None, unroll=False, fp64=False,
    pre_expand=True, collect_rho=True, factor_terms=False,
    cse_batch_size=None, cse_simplify=True,
    hoist_rho_independent=True, nvrtc_options=(), timings=False,
    return_timing_info=False, warmup_time=0.0, rho0=None,
    rho0_var_arrays=None, rho0_values=None, return_time_trace=False,
    time_trace_every=None, time_trace_samples_per_period=None,
    solver_samples_per_period=None, Actual_Kernel_Save=False,
    beep_on_error=False, ignore_non_finite_output=False,
    t_in=0.0, return_device=False,
)
```

Build and solve a finite-dimensional Lindblad master equation over one or two parameter axes. One CUDA thread
integrates one parameter combination with the selected fixed-step method. GQIS symbolically generates the
right-hand side (RHS), meaning the time derivatives, of the reduced ordinary differential equation (ODE) system and
compiles it as a CUDA kernel.

Notation used below:

- `N` is the Hilbert-space dimension: the number of basis states or quantum
  levels represented by the Hamiltonian. Therefore `H`, every collapse
  operator, `mean_operator`, and a symbolic initial density matrix are `N x N`.
- `M` is the number of time samples in `tlist`. These samples define `M - 1`
  solver steps from `tlist[0]` through `tlist[-1]`.
- `num_X` and `num_Y` are the numbers of independent simulations along the two
  parameter-sweep axes. They are unrelated to the time samples.
- `1D` means a one-dimensional array.

### Minimal Call

Assuming the symbolic model and numerical arrays have already been defined, the
smallest one-axis call is:

```python
result = mesolve_2D(
    H, Drive, Col_Ops, mean_operator, tlist,
    var_arrays={eps: eps_values},
)
```

The required positional arguments `H`, `Drive`, `Col_Ops`,
`mean_operator`, and `tlist` specify the Hamiltonian, time-dependent drive,
collapse operators, measured operator, and time samples, respectively. At least
one sweep array is required. This call returns shape `(len(eps_values), 1)`; a
two-axis call uses `var_arrays={eps: eps_values, A: amplitude_values}` and
returns shape `(len(eps_values), len(amplitude_values))`.

### Physical Model

| Parameter | Type | Meaning |
| --- | --- | --- |
| `H` | `N x N` SymPy matrix | Hamiltonian. `N` is the number of simulated basis states or quantum levels. `H` may contain sweep symbols, constant symbols, and one or more drive placeholder symbols. |
| `Drive` | SymPy expression or `dict[Symbol, Expr]` | Time-dependent signal. A single expression supplies the conventional `Symbol("Drive")` placeholder used by `H`. A dictionary explicitly maps multiple Hamiltonian placeholder symbols to separate signals. Expressions may use `t`, sweep parameters, and constants. |
| `Col_Ops` | sequence of `N x N` SymPy matrices | Lindblad collapse operators. Use `[]` for closed-system evolution. Every operator must have the same shape as `H`. |
| `mean_operator` | `N x N` SymPy matrix | Operator whose expectation value is averaged, returned at the final time, or sampled as a trace. It must match `H`. |
| `tlist` | 1D array | Finite, strictly increasing, uniformly spaced elapsed times beginning at zero. `M` samples define exactly `M - 1` solver steps through absolute time `t_in + tlist[-1]`. |
| `t_in` | `0.0` | Absolute initial time of the supplied state. Drive evaluations and returned trace times include this offset. |
| `return_device` | `False` | Return CuPy result/trace arrays instead of copying them to NumPy. Trace times remain NumPy metadata. |

### Sweeps And Constants

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `var_arrays` | `dict[Symbol, 1D array]` | One or two sweep axes are required. Dictionary insertion order maps the first array to result axis X and the second to Y. For a single parameter point, supply a one-element dummy axis. Array values and lengths can change while reusing the generated ODE. |
| `const_values` | `dict[Symbol, number]` | Constants substituted as numbers while generating the ODE and CUDA code. If one of these values changes on a later call, GQIS normally generates and compiles a new ODE kernel. |
| `runtime_consts` | `dict[Symbol, number]` | Values for constants intentionally left symbolic in the generated ODE. They are uploaded for each call, so they can change while the previously generated and compiled ODE kernel is reused. A value here overrides the same symbol in `const_values`. This is useful for animations and repeated calculations. |
| `keep_symbolic_consts` | iterable, `"all"`, `"auto"`, `"const_values"`, or `None` | Selects which `const_values` symbols remain symbolic instead of being inserted as fixed numbers. This allows selected constants to change between animation frames while reusing the generated ODE and compiled kernel. `"all"`, `"auto"`, and `"const_values"` select every `const_values` key. |
| `auto_runtime_consts` | `False` | Keep every `const_values` key symbolic so all constant values can change between repeated calls without regenerating the ODE. |
| `rho0_var_arrays` | `dict[Symbol, 1D array]` | Sweep values for symbols used only in the symbolic `rho0` matrix. They are merged with `var_arrays`; the total remains limited to two sweep axes. These arrays select different initial states for independent simulations, not for different time samples. |
| `rho0_values` | numeric array or `None` | Alternative to symbolic `rho0`: one explicit reduced initial state per simulation point. Shape is `(num_X, N*N-1)` for one sweep axis or `(num_X, num_Y, N*N-1)` for two axes. The last dimension stores density-matrix components, not time samples. Values can change without recompilation. Cannot be combined with `rho0`. |

### Initial State And Output

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `rho0` | `N x N` SymPy matrix, reduced expression list, or `None` | Initial density matrix at `t=t_in`. It may contain symbols swept through `rho0_var_arrays` or `var_arrays`, and it may use runtime constants. `None` initializes state `|0><0|`. For matrix input, GQIS reads the first `N-1` diagonal populations and the upper-triangular coherences; unit trace and the lower triangle are reconstructed. |
| `output_mode` | `"mean"` | `"mean"` averages the expectation value after each post-warmup solver step; `"final"` returns the expectation value at the final time; `"final_rho"` returns the final reduced real density vector in the ordering described under `build_independent_rho`. |
| `warmup_time` | `0.0` | Initial fraction of time excluded from the time average, in `[0, 1]`. This can suppress transient dependence on the initial state without shortening the simulated evolution. A value of `1` leaves no averaging window and returns the final expectation value. |
| `return_time_trace` | `False` | Also return sampled expectation values and their absolute times. Samples begin at `t=t_in+dt`, after the first integration step. |
| `time_trace_every` | `None` | Store one trace value every specified number of solver steps. With stride `k`, stored times are `dt`, `(k+1)*dt`, `(2k+1)*dt`, and so on. Mutually exclusive with `time_trace_samples_per_period`. |
| `time_trace_samples_per_period` | `None` | Requested approximate stored trace density per drive period. Requires `solver_samples_per_period`; the integer stride is `max(1, round(solver_samples_per_period / time_trace_samples_per_period))`. |
| `solver_samples_per_period` | `None` | Number of solver integration steps per drive period, used only to convert trace density to a stride. |

### Initial-State Representations

An `N`-level density matrix has shape `N x N`. A Hermitian matrix contains
`N*N` independent real values, and the unit-trace condition removes one of
them. GQIS therefore evolves and stores only `N*N - 1` real values per state:

1. the first `N - 1` diagonal populations
2. the real and imaginary parts of the upper-triangular coherences

The final diagonal population and lower-triangular coherences are reconstructed
from unit trace and Hermiticity. This is why the last dimension of
`rho0_values` is `N*N - 1`, rather than `N*N`.

Use `rho0` with `rho0_var_arrays` when the initial state has a convenient
analytic SymPy form. Use `rho0_values` when the initial states have already been
calculated numerically or are easier to provide as arbitrary reduced vectors.
Both initial-state representations specify one state at `t=t_in` for every independent GPU
simulation; neither contains a state for every time sample.

### Code Generation And Execution

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `kernel_template_file` | `None` | Uses the canonical CUDA template packaged with GQIS. Supply an explicit path only to test or develop a custom kernel template; relative explicit paths are checked in the current directory and then inside the installed package. |
| `RHSreuse` | `True` | Reuse the generated symbolic RHS independently of the solver-specific compiled CUDA kernel. Different fixed-step solvers therefore share one symbolic RHS while retaining separate kernels. Sweep arrays, numerical initial states supplied through `rho0_values`, and numerical values assigned to selected constants kept symbolic may change between calls. `False` regenerates both stages on every call. |
| `solver` | `"rk4"` | Uniform fixed-step CUDA integrator: `"rk4"`, `"lserk4"`, `"dp5"`, `"ab5"`, `"anas5"`, `"tsit5"`, `"alshina6"`, or `"dop853"`. Only the selected solver body is inserted into the generated CUDA source. `"rk4"` remains the default. No solver performs adaptive step acceptance. |
| `solver_frequency` | `None` | Non-negative angular frequency `w` for fixed-step `"anas5"`. Before launch, its fitted coefficient is calculated from `v = w*dt` and then remains constant throughout the kernel. This does not enable adaptive stepping. Anas5 defaults to `w=1` when omitted; supply the dominant frequency for meaningful fitted behavior. Other solvers ignore it. |
| `unroll` | `False` | `True` forces `#pragma unroll` on solver vector loops. `False` leaves the decision to NVRTC, preserving the existing behavior and avoiding mandatory unrolling when larger density vectors would create excessive register pressure. Changing this option produces a separately compiled cached kernel. |
| `fp64` | `False` | Use 64-bit floating-point (FP64) state, sweep arrays, constants, generated math, and output instead of the faster 32-bit floating-point (FP32) path. |
| `pre_expand` | `True` | Expand symbolic RHS expressions before code generation. |
| `collect_rho` | `True` | Collect RHS terms by reduced density variables. |
| `factor_terms` | `False` | Factor symbolic terms before common-subexpression elimination (CSE). This can reduce operations but increase preparation time. |
| `cse_batch_size` | `None` | Equations per common-subexpression-elimination batch. `None` performs global CSE; smaller batches can lower CUDA register pressure. |
| `cse_simplify` | `True` | Simplify expressions during CSE emission. |
| `hoist_rho_independent` | `True` | Move state-independent expressions to per-thread static or per-RK-stage drive calculations. |
| `nvrtc_options` | empty tuple | Additional NVIDIA Runtime Compilation (NVRTC) options passed to `cupy.RawKernel`. A string or iterable of strings is accepted. |
| `Actual_Kernel_Save` | `False` | `True` saves `<caller>_Kernel.cu`; a string saves to that explicit path. Generated kernels are debugging artifacts and are ignored by the repository. |

### Available Fixed-Step Solvers

The `solver` option selects the same methods in `mesolve_2D` and `odesolve_2D`.
The implementations have the following fixed-grid roles and steady-state costs. Work-vector counts exclude the state,
static terms, and drive values; NVRTC may further scalarize or reuse them.

| Class | Solver | Order | New RHS/step | Work vectors | Main advantage | Main limitation |
| --- | --- | ---: | ---: | ---: | --- | --- |
| General baseline | `rk4` — classical Runge-Kutta | 4 | 4 | 3 | Strong general work-precision baseline. | Lower order than the high-accuracy methods. |
| General low-storage | `lserk4` — Carpenter-Kennedy 2N54 | 4 | 5 | 2 | Two work vectors help reduce register pressure for large systems. | One more RHS evaluation than RK4. |
| General fifth-order | `dp5` — Dormand-Prince 5(4) main formula | 5 | 6 | 5 | Broadly useful at tighter accuracy. | Embedded estimate is unused; more work than RK4. |
| General fifth-order | `tsit5` — Tsitouras 5(4) main formula | 5 | 6 | 5 | Good accuracy constants and direct comparison with Julia GPUTsit5. | Embedded estimate is unused; more work than RK4. |
| General high-order | `alshina6` — Alshina sixth-order formula | 6 | 7 | 5 | Sixth order with seven stages. | More work per step than the fifth-order methods. |
| General high-order | `dop853` — Dormand-Prince 8(5,3) main formula | 8 | 12 | 6 | High-order accuracy can permit coarser fixed grids. | More stages and work vectors; roundoff can limit gains at fine steps. |
| Periodic fitted | `anas5` — frequency-fitted Anas5(w) | 4 generally, 5 for accurate `w` | 6 | 5 | Reduced phase error when the dominant frequency is known. | Loses fifth-order fitting when `w` is inaccurate or the signal is broadband. |
| Multistep | `ab5` — Adams-Bashforth 5 | 5 | 1 after startup | 7 during RK4 startup, then 5 | Very low steady RHS cost on smooth trajectories. | Smaller stability region, startup cost, and derivative-history storage. |

All methods use the uniform `dt` supplied by `tlist` and reconstruct base time from the integer step index rather than
accumulating `t += dt`. Alshina6 and DOP853 actively fold early derivative vectors into pending stage combinations;
those buffers are reused as their values become available. Derivative and drive values are reused where the method
permits. DOP853 uses the eighth-order update without the embedded error estimates or dense-output stages.
Anas5(w) uses the supplied frequency and fixed step size to set its fitted coefficient.

Accuracy depends on the chosen method and user-supplied step size. The
[Benchmark 03 accuracy sweeps](BENCHMARKS.md#accuracy-calibrated-dividers) show this dependence
for the two- and four-level examples and provide a workflow for choosing a time grid for your own problem.

### Diagnostics

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `timings` | `False` | Print symbolic-generation/compilation time, GPU kernel time, total call time, and whether the generated ODE and kernel were reused. |
| `return_timing_info` | `False` | Collect timings without requiring console output. The dictionary includes `rhs_codegen_s`, `rhs_stage_s`, `gpu_kernel_s`, `total_s`, `cached_rhs`, `cached_kernel`, the normalized `solver` name, `solver_frequency`, and `unroll`. `cached_rhs` reports reuse of solver-independent symbolic CUDA expressions; `cached_kernel` separately reports reuse of the selected compiled solver kernel. |
| `beep_on_error` | `False` | Play a best-effort notification before raising for non-finite output. |
| `ignore_non_finite_output` | `False` | Return not-a-number (NaN) or infinite (Inf) output instead of raising. Useful for accuracy sweeps that also record unstable working points. |

### Reusing The ODE For Animations

With `RHSreuse=True`, GQIS can reuse the generated right-hand side (RHS) of the
ODE and its compiled CUDA kernel across repeated calls. The symbolic equation
structure must remain unchanged; only the numerical values assigned to selected
symbols may change. Constants that change between frames therefore need to
remain symbolic. For example:

```python
result = mesolve_2D(
    H, Drive, Col_Ops, mean_operator, tlist,
    var_arrays={eps: eps_values, A: amplitude_values},
    const_values={gamma: gamma_for_this_frame},
    keep_symbolic_consts={gamma},
    RHSreuse=True,
)
```

On the next call, `gamma_for_this_frame` may change while the generated ODE and
compiled kernel are reused. Without `keep_symbolic_consts={gamma}`, the value is
inserted directly into the generated equations and changing it requires new
symbolic generation and compilation. `runtime_consts={gamma: value}` can also
supply or override the value of a constant kept symbolic this way.

### Returns

Without optional trace/timing values, the result is a NumPy array:

- `output_mode="mean"` or `"final"`: complex shape `(num_X, num_Y)`.
- `output_mode="final_rho"`: real shape `(num_X, num_Y, N*N-1)`.
- A one-axis sweep retains a singleton Y dimension.

With `return_time_trace=True`, the return is
`(result, trace, trace_t)`, where `trace` has complex shape
`(num_X, num_Y, num_trace)`. With `return_timing_info=True`, `timing_info` is
appended to that tuple; without a trace the return is `(result, timing_info)`.

### Validation And Limitations

- At least one sweep array is required, even for a single simulation; use a
  one-element dummy axis when no physical parameter is swept.
- GQIS validates matrix dimensions, time-grid structure, option combinations,
  explicit initial-state array shapes, and finite output values.
- GQIS does not test whether a supplied symbolic or reduced initial state is
  positive semidefinite. The caller is responsible for providing a physical
  density matrix. For matrix input, the last population and lower triangle are
  reconstructed rather than independently validated.
- The elapsed time grid is uniform and begins at zero. Absolute stage times add `t_in`
  to the elapsed time derived from the integer step index.
- Kernel reuse is process-local; the compiled-kernel cache is not written to
  disk. A new Python process compiles its first model again.
- Practical Hilbert-space size is limited by symbolic-generation cost, NVRTC
  compilation, CUDA register pressure, and GPU memory.

## Density-Matrix Helpers

### `build_independent_rho(N)`

Input: Hilbert-space dimension `N`.

Output: `(rho, metadata)`. `rho` is an `N x N` Hermitian, trace-one SymPy
matrix built from `N*N-1` real symbols. `metadata` contains `N`, `rho_syms`,
`num_diag`, `num_coherences`, and `vec_len`. Here `num_coherences` is the number
of upper-triangular coherence pairs; `M` remains reserved for the number of
time samples.

Reduced-vector order is:

1. Diagonal populations `rho[0,0]` through `rho[N-2,N-2]`.
2. Real and imaginary parts of each upper-triangular coherence in row-major order.

The last population is reconstructed from unit trace.

### `build_reduced_lindblad_rhs(N, H, Col_Ops, mean_operator, *, pre_expand=False, collect_rho=False, factor_terms=False)`

Inputs: Hilbert-space dimension, symbolic Hamiltonian, collapse operators,
observable operator, and optional symbolic simplification controls.

Output: `(drho_eqs, mean_re, mean_im, metadata)`. `drho_eqs` contains the
`N*N-1` real Lindblad ODE expressions in the ordering defined above. `mean_re`
and `mean_im` are the expectation-value components. CUDA generation and the
benchmark Julia and central processing unit (CPU) solvers use this same function so their density-matrix
equations cannot diverge through separately maintained symbolic derivations.

### `rho_matrix_to_independent_exprs(rho0)`

Input: a square SymPy density matrix, or an iterable already containing
`N*N-1` reduced expressions.

Output: a simplified expression list in the exact ordering used by
`build_independent_rho`. Raises `ValueError` for a nonsquare matrix or invalid
reduced-vector length.

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

Inputs: placeholder-to-expression mapping, target CUDA array name, and a switch
that permits one-use math functions to remain inline.

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

## File And Notification Helpers

These underscore-prefixed functions are implementation details, not stable
public API:

- `_resolve_kernel_template_file(kernel_template_file)` returns the canonical packaged template for `None`, resolves an explicit path otherwise, and raises `FileNotFoundError` when no matching template exists.
- `_resolve_generated_kernel_path(actual_kernel_save)` returns the absolute generated-kernel output path.
- `_save_generated_kernel_file(actual_kernel_save, kernel_code)` writes requested CUDA source and returns its path, or returns `None` when disabled.
- `_play_notification_beep(kind)` emits a best-effort error or completion sound and returns `None`.

`mesolve_2D` also uses nested private helpers for expression iteration, runtime
constant derivation, compiled-kernel reuse checks, constant-index compaction, and
host-to-device conversion. They are intentionally local because they depend on
the current solve call and are not callable package APIs.

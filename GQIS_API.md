# GQIS API Reference

This file documents the packaged interface in `gqis/solver.py`. Normal
user code only needs `mesolve_2D`; the other functions expose the symbolic and
CUDA code-generation stages for inspection or advanced customization.

```python
from gqis import build_independent_rho, build_reduced_lindblad_rhs, mesolve_2D
```

These three functions form the public package interface. The remaining helpers
documented below are available from `gqis.solver` for inspection and advanced
development, but they may change before version 1.0.

The compatibility imports `from gpu_int_tool import mesolve_2D` and
`from GPU_Int_Tool import mesolve_2D` are retained for older scripts.

## `mesolve_2D`

```python
mesolve_2D(
    H, Drive, Col_Ops, mean_operator, tlist,
    var_arrays=None, const_values=None,
    kernel_template_file=None, *,
    RHSreuse=True, runtime_consts=None, keep_symbolic_consts=None,
    auto_runtime_consts=False, output_mode="mean", fp64=False,
    pre_expand=True, collect_rho=True, factor_terms=False,
    cse_batch_size=None, cse_simplify=True,
    hoist_rho_independent=True, nvrtc_options=(), timings=False,
    return_timing_info=False, warmup_time=0.0, rho0=None,
    rho0_var_arrays=None, rho0_values=None, return_time_trace=False,
    time_trace_every=None, time_trace_samples_per_period=None,
    solver_samples_per_period=None, Actual_Kernel_Save=False,
    beep_on_error=False, ignore_non_finite_output=False,
)
```

Build and solve a finite-dimensional Lindblad master equation over one or two
parameter axes. One CUDA thread integrates one parameter combination with
fixed-step RK4.

### Physical Model

| Parameter | Type | Meaning |
| --- | --- | --- |
| `H` | square SymPy matrix | Hamiltonian. Its dimension defines the Hilbert-space dimension `N`. It may contain sweep symbols, constant symbols, and one or more drive placeholder symbols. |
| `Drive` | SymPy expression or `dict[Symbol, Expr]` | Time-dependent signal. A single expression supplies the conventional `Symbol("Drive")` placeholder used by `H`. A dictionary explicitly maps multiple Hamiltonian placeholder symbols to separate signals. Expressions may use `t`, sweep parameters, and constants. |
| `Col_Ops` | sequence of square SymPy matrices | Lindblad collapse operators. Use `[]` for closed-system evolution. Every operator must have the same shape as `H`. |
| `mean_operator` | square SymPy matrix | Operator whose expectation value is averaged, returned at the final time, or sampled as a trace. It must match `H`. |
| `tlist` | 1D array | Finite, strictly increasing, uniformly spaced times beginning at zero. `M` samples define exactly `M - 1` solver steps through `tlist[-1]`. Requiring zero avoids an additional kernel time-offset argument. The current CUDA backend uses fixed-step RK4. |

### Sweeps And Constants

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `var_arrays` | `dict[Symbol, 1D array]` | One or two sweep axes are required. Dictionary insertion order maps the first array to result axis X and the second to Y. For a single parameter point, supply a one-element dummy axis. Array values and lengths can change without changing generated RHS structure. |
| `const_values` | `dict[Symbol, number]` | Constant substitutions. By default values are folded into generated CUDA code, so changing them produces a different cache key and kernel. |
| `runtime_consts` | `dict[Symbol, number]` | Explicit constants uploaded through `Const_arr`. A value here overrides the same symbol's value in `const_values`. Values can change while reusing an equivalent compiled RHS. Symbols unused by the model are removed and do not change the generated kernel. |
| `keep_symbolic_consts` | iterable, `"all"`, `"auto"`, `"const_values"`, or `None` | Selects symbols from `const_values` that must remain runtime constants. `"all"`, `"auto"`, and `"const_values"` all select every `const_values` key. |
| `auto_runtime_consts` | `False` | Keep every `const_values` key symbolic at runtime when true. |
| `rho0_var_arrays` | `dict[Symbol, 1D array]` | Initial-condition-only sweep symbols. They are merged with `var_arrays`; the total remains limited to two axes. |
| `rho0_values` | numeric array or `None` | Explicit reduced initial states. Shape is `(num_X, N*N-1)` for a one-axis sweep, or `(num_X, num_Y, N*N-1)` for two axes. Values can change without recompilation. Cannot be combined with `rho0`. |

### Initial State And Output

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `rho0` | SymPy matrix, reduced expression list, or `None` | Initial density matrix. It may contain sweep or runtime-constant symbols. `None` initializes state `|0><0|`. For a matrix, GQIS reads the first `N-1` diagonal populations and the upper-triangular coherences; unit trace and the lower triangle are reconstructed. |
| `output_mode` | `"mean"` | `"mean"` averages the observable after each post-warmup solver step; `"final"` returns its final expectation; `"final_rho"` returns the final reduced real density vector in the ordering described under `build_independent_rho`. |
| `warmup_time` | `0.0` | Initial fraction of time excluded from the time average, in `[0, 1]`. This can suppress transient dependence on the initial state without shortening the simulated trajectory. A value of `1` leaves no averaging window and returns the final expectation value. |
| `return_time_trace` | `False` | Also return sampled expectation values and their times. Samples are recorded from the evolved state after selected integration steps, beginning at `t=dt`, not from the initial state at `t=0`. |
| `time_trace_every` | `None` | Store one trace value every specified number of solver steps. With stride `k`, stored times are `dt`, `(k+1)*dt`, `(2k+1)*dt`, and so on. Mutually exclusive with `time_trace_samples_per_period`. |
| `time_trace_samples_per_period` | `None` | Requested approximate stored trace density per drive period. Requires `solver_samples_per_period`; the integer stride is `max(1, round(solver_samples_per_period / time_trace_samples_per_period))`. |
| `solver_samples_per_period` | `None` | Number of solver integration steps per drive period, used only to convert trace density to a stride. |

### Code Generation And Execution

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `kernel_template_file` | `None` | Uses the canonical CUDA template packaged with GQIS. Supply an explicit path only to test or develop a custom kernel template; relative explicit paths are checked in the current directory and then inside the installed package. |
| `RHSreuse` | `True` | Reuse an in-memory generated and compiled kernel when model structure and code-generation settings match. Sweep values, explicit `rho0_values`, and runtime-constant values are intentionally excluded from the structural cache key. `False` regenerates and recompiles on every call. |
| `fp64` | `False` | Use FP64 state, sweep arrays, constants, generated math, and output instead of the faster FP32 path. |
| `pre_expand` | `True` | Expand symbolic RHS expressions before code generation. |
| `collect_rho` | `True` | Collect RHS terms by reduced density variables. |
| `factor_terms` | `False` | Factor symbolic terms before common-subexpression elimination. This can reduce operations but increase preparation time. |
| `cse_batch_size` | `None` | Equations per common-subexpression-elimination batch. `None` performs global CSE; smaller batches can lower CUDA register pressure. |
| `cse_simplify` | `True` | Simplify expressions during CSE emission. |
| `hoist_rho_independent` | `True` | Move state-independent expressions to per-thread static or per-RK-stage drive calculations. |
| `nvrtc_options` | empty tuple | Additional options passed to `cupy.RawKernel`. A string or iterable of strings is accepted. |
| `Actual_Kernel_Save` | `False` | `True` saves `<caller>_Kernel.cu`; a string saves to that explicit path. Generated kernels are debugging artifacts and are ignored by the repository. |

### Diagnostics

| Parameter | Type/default | Meaning |
| --- | --- | --- |
| `timings` | `False` | Print symbolic/codegen/compile stage time, GPU kernel time, total call time, and cache hit/miss. |
| `return_timing_info` | `False` | Collect timings without requiring console output and include a dictionary with `rhs_stage_s`, `gpu_kernel_s`, `total_s`, and `cached_rhs` (`"hit"` or `"miss"`). On a cache miss, `rhs_stage_s` includes symbolic generation and NVRTC compilation. |
| `beep_on_error` | `False` | Play a best-effort notification before raising for non-finite output. |
| `ignore_non_finite_output` | `False` | Return NaN/Inf output with a warning instead of raising. Intended for diagnosis, not production data. |

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

- At least one sweep array is required, even for a single trajectory; use a
  one-element dummy axis when no physical parameter is swept.
- GQIS validates matrix dimensions, time-grid structure, option combinations,
  explicit initial-state array shapes, and finite output values.
- GQIS does not test whether a supplied symbolic or reduced initial state is
  positive semidefinite. The caller is responsible for providing a physical
  density matrix. For matrix input, the last population and lower triangle are
  reconstructed rather than independently validated.
- The time grid must be uniform and begin at zero because the CUDA kernel uses
  fixed-step RK4 and derives stage times from the integer step index.
- Kernel reuse is process-local; the compiled-kernel cache is not written to
  disk. A new Python process compiles its first model again.
- Practical Hilbert-space size is limited by symbolic-generation cost, NVRTC
  compilation, CUDA register pressure, and GPU memory.

## Density-Matrix Helpers

### `build_independent_rho(N)`

Input: Hilbert-space dimension `N`.

Output: `(rho, metadata)`. `rho` is an `N x N` Hermitian, trace-one SymPy
matrix built from `N*N-1` real symbols. `metadata` contains `N`, `rho_syms`,
`num_diag`, `M`, and `vec_len`.

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
benchmark Julia/CPU backends use this same function so their density-matrix
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
`MyCPrinter` is the underlying SymPy C99 printer; its `_print_Float(expr)`
method emits an explicit FP32 literal.

## File And Notification Helpers

These underscore-prefixed functions are implementation details, not stable
public API:

- `_resolve_kernel_template_file(kernel_template_file)` returns the canonical packaged template for `None`, resolves an explicit path otherwise, and raises `FileNotFoundError` when no matching template exists.
- `_resolve_generated_kernel_path(actual_kernel_save)` returns the absolute generated-kernel output path.
- `_save_generated_kernel_file(actual_kernel_save, kernel_code)` writes requested CUDA source and returns its path, or returns `None` when disabled.
- `_play_notification_beep(kind)` emits a best-effort error or completion sound and returns `None`.

`mesolve_2D` also uses nested private helpers for expression iteration, runtime
constant derivation, cache-key construction, constant-index compaction, and
host-to-device conversion. They are intentionally local because they depend on
the current solve call and are not callable package APIs.

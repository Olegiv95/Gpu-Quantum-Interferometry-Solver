"""GPU Quantum Interferometry Solver (GQIS) symbolic-to-CUDA solver core.

This module converts symbolic Lindblad right-hand side (RHS) expressions into
NVIDIA CUDA code, compiles them with CuPy and NVIDIA Runtime Compilation
(NVRTC), and executes parameter sweeps on a graphics processing unit (GPU).
"""

import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from string import Template

import cupy as cp
import numpy as np
import sympy as sp
from sympy.printing.c import C99CodePrinter
from sympy.printing.precedence import PRECEDENCE

from ._options import OPTION_GROUPS, apply_option_groups
from .cuda_solvers import get_solver_spec, normalize_solver_name

try:
    import winsound
except ImportError:
    winsound = None

# Independent caches let several integrators share one generated symbolic RHS.
_RHS_CODE_CACHE = {}
_KERNEL_CACHE = {}
# Accepted time steps from calibrate_timesteps, kept separately from compiled kernels.
_TIMESTEP_CALIBRATION_CACHE = {}

TIMESTEP_CALIBRATION_DEFAULTS = {
    "rms_atol": 1e-3, "rms_rtol": 0.0,   # limit for the RMS change of the result map
    "max_atol": 1e-2, "max_rtol": 0.0,   # limit for the largest pointwise change
    "sample_stride": 8,                  # test every k-th point on each sweep axis (plus axis ends)
    "max_steps": 2**20,                  # no refinement beyond this step count
    "bisection_steps": 2,                # halvings of the failing/passing bracket; 0: doubling only
    "full_grid_bisection": False,        # True: bisect also when the test points are the full grid
    "recalibrate_every": None,           # None: calibrate once, then reuse; N: every N-th call
    "reset": False,                      # True: discard the stored step and calibrate again
    "on_fail": "warning",                # if limits are not met: "warning", "ignore" (both continue) or "error"
    "range_warning": True,               # warn once when a later sweep range exceeds the calibrated one
}
# With bisection, a supplied grid that already passes is halved at most this many times.
_CALIBRATION_MAX_HALVINGS = 6
# odesolve_2D defaults: general ODE states need tighter limits than interferogram maps.
TIMESTEP_CALIBRATION_ODE_DEFAULTS = {"rms_atol": 1e-4, "max_atol": 1e-3}

def _play_notification_beep(kind):
    """Play a short completion/error beep when available.

    Args:
        kind: ``"error"`` selects an error sound; any other value selects a
            short notification sound.

    Returns:
        None. The function only produces a best-effort audible notification.
    """

    if winsound is None:
        print("\a", end="", flush=True)
        return
    try:
        winsound.MessageBeep(winsound.MB_ICONHAND if kind == "error" else winsound.MB_ICONASTERISK)
    except RuntimeError:
        pass
    try:
        if kind == "error":
            winsound.Beep(880, 180)
            winsound.Beep(660, 260)
        else:
            winsound.Beep(880, 140)
            winsound.Beep(1175, 180)
    except RuntimeError:
        print("\a", end="", flush=True)


def _resolve_generated_kernel_path(actual_kernel_save):
    """Return output path for saving the fully generated CUDA kernel.

    Args:
        actual_kernel_save: ``True`` to derive a filename from the calling
            script, or a non-empty string to use as an explicit output path.

    Returns:
        Absolute path where the generated CUDA source should be written.
    """

    if isinstance(actual_kernel_save, str) and actual_kernel_save.strip():
        return os.path.abspath(actual_kernel_save)

    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    if main_file:
        base_dir = os.path.dirname(os.path.abspath(main_file))
        stem = os.path.splitext(os.path.basename(main_file))[0]
        return os.path.join(base_dir, f"{stem}_Kernel.cu")

    return os.path.abspath("Generated_Kernel.cu")


def _save_generated_kernel_file(actual_kernel_save, kernel_code):
    """Save the generated kernel if requested and return the path.

    Args:
        actual_kernel_save: ``False`` to skip saving, ``True`` to derive a
            filename, or a string path.
        kernel_code: Fully generated CUDA source code.

    Returns:
        Saved file path, or ``None`` when saving is disabled.
    """

    if not actual_kernel_save:
        return None
    out_path = _resolve_generated_kernel_path(actual_kernel_save)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(kernel_code)
    return out_path


def _resolve_kernel_template_file(kernel_template_file):
    """Resolve the CUDA kernel template path.

    Args:
        kernel_template_file: User-provided template path, or ``None`` for the
            canonical template packaged with GQIS. Explicit relative paths are
            checked first against the current working directory, then next to
            this Python module.

    Returns:
        Existing template file path.

    Raises:
        FileNotFoundError: If the template cannot be found.
    """

    packaged_template = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "N_Level_Kernel.cu")
    if kernel_template_file is None:
        if os.path.exists(packaged_template):
            return packaged_template
        raise FileNotFoundError(f"Packaged CUDA kernel template was not found: {packaged_template}")
    if os.path.exists(kernel_template_file):
        return kernel_template_file
    module_candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    kernel_template_file)
    if os.path.exists(module_candidate):
        return module_candidate
    raise FileNotFoundError(f"CUDA kernel template was not found: {kernel_template_file}")


class MyCPrinter(C99CodePrinter):
    """C printer with explicit float literals for generated CUDA code.

    The default SymPy C printer may emit double literals. This subclass emits
    explicit ``f`` suffixes for SymPy floats so the generated FP32 CUDA path does
    not silently promote arithmetic to double precision.
    """

    def _print_Float(self, expr):
        """Return a CUDA FP32 literal for one SymPy Float."""
        # Emit scientific notation and keep explicit float suffix for FP32 codegen.
        val = float(expr)
        return f"{val:.9e}f"

    def _print_Pow(self, expr):
        """Print squares as products; a sum as the base gets parentheses: ``(x + 1)**2``."""
        if expr.exp == 2:
            base = self.parenthesize(expr.base, PRECEDENCE["Mul"])
            return f"({base}*{base})"
        return super()._print_Pow(expr)


def my_ccode(expr):
    """Print a SymPy expression using CUDA-friendly floating literals.

    Args:
        expr: SymPy expression.

    Returns:
        C99/CUDA expression string.
    """
    # Exact irrational constants (pi, E, sqrt(2), log(2), ...) would print as C macros such as
    # M_SQRT2, which NVRTC does not define, or as double constants in FP32 code; print them as
    # floating literals instead. Integers and fractions keep their exact form.
    constants = {}
    if isinstance(expr, sp.Basic):
        walk = sp.preorder_traversal(expr)
        for sub in walk:
            if sub.is_number and not (sub.is_Rational or sub.is_Float) and sub.is_real:
                constants[sub] = sub.evalf()
                walk.skip()
    return MyCPrinter().doprint(expr.xreplace(constants) if constants else expr)


def build_independent_rho(N):
    """Build the reduced real-valued density-vector representation.

    The solver evolves only independent density-matrix components. For an
    ``N x N`` density matrix, Hermiticity and trace conservation reduce the
    real-valued state to ``N*N - 1`` values:
    diagonal elements except the last one, then real/imaginary parts of the
    upper-triangular coherences.

    Args:
        N: Hilbert-space dimension.

    Returns:
        ``(rho, meta)`` where ``rho`` is an ``N x N`` SymPy matrix expressed in
        terms of reduced real symbols, and ``meta`` contains the reduced symbols
        and size bookkeeping.
    """

    I = sp.I
    num_diag = N - 1
    num_coherences = (N * (N - 1)) // 2
    vec_len = num_diag + 2 * num_coherences
    rho_syms = [sp.Symbol(f"rho[{i}]", real=True) for i in range(vec_len)]

    def diag_elem(i):
        """Return diagonal density-matrix element ``rho[i,i]``."""
        return rho_syms[i] if i < num_diag else 1 - sum(rho_syms[:num_diag])

    def upper_index(i, j):
        """Return compact upper-triangular coherence index for ``i < j``."""
        return i * N - (i * (i + 1)) // 2 + (j - i - 1)

    base = num_diag

    def upper_elem(i, j):
        """Return complex upper-triangular element from real/imag symbols."""
        u = upper_index(i, j)
        return rho_syms[base + 2 * u] + I * rho_syms[base + 2 * u + 1]

    def rho_elem(i, j):
        """Return density-matrix element respecting Hermiticity and trace."""
        if i == j:
            return diag_elem(i)
        elif i < j:
            return upper_elem(i, j)
        else:
            return sp.conjugate(upper_elem(j, i))

    rho = sp.Matrix(N, N, rho_elem)
    meta = {"N": N, "rho_syms": rho_syms, "num_diag": num_diag,
            "num_coherences": num_coherences, "vec_len": vec_len}
    return rho, meta


def rho_matrix_to_independent_exprs(rho0):
    """Map an initial density matrix to the reduced real state vector.

    Args:
        rho0: Either an ``N x N`` SymPy density matrix, or an iterable already
            containing the reduced ``N*N - 1`` expressions.

    Returns:
        List of SymPy expressions ordered exactly as ``build_independent_rho``.

    Raises:
        ValueError: If the matrix is not square or the reduced vector length is
            inconsistent with any integer Hilbert dimension.
    """

    if isinstance(rho0, sp.MatrixBase):
        rows, cols = rho0.shape
        if rows != cols:
            raise ValueError("rho0 matrix must be square.")
        N = rows
        exprs = []
        for i in range(N - 1):
            exprs.append(sp.simplify(sp.re(rho0[i, i])))
        for i in range(N):
            for j in range(i + 1, N):
                exprs.append(sp.simplify(sp.re(rho0[i, j])))
                exprs.append(sp.simplify(sp.im(rho0[i, j])))
        return exprs

    exprs = list(rho0)
    vec_len = len(exprs)
    N = int(round(np.sqrt(vec_len + 1)))
    if N * N - 1 != vec_len:
        raise ValueError("rho0 reduced vector length must match N*N-1.")
    return [sp.simplify(sp.sympify(expr)) for expr in exprs]


_DRHO_ASSIGNMENT = re.compile(r"^(\s*)d_rho\[(\d+)\] = (.*);\s*$")


def _insert_fused_lserk_rhs(kernel_code, drho_lines):
    """Add ``compute_drho_lserk``: the RHS accumulated directly into the LSRK4 residual.

    Each generated ``d_rho[i] = f_i;`` becomes
    ``residual[i] = fmaf(lserk_a, residual[i], lserk_dt * (f_i));``. All ``f_i`` read only
    the unchanged state, so no separate derivative vector is needed (2N storage).
    """
    fused = []
    for line in drho_lines:
        match = _DRHO_ASSIGNMENT.match(line)
        if match:
            indent, index, expr = match.groups()
            line = (f"{indent}residual[{index}] = fmaf(lserk_a, residual[{index}], "
                    f"lserk_dt * ({expr}));")
        fused.append(line)
    if any("d_rho" in line for line in fused):
        raise RuntimeError("Unexpected RHS line format; cannot build the fused LSRK4 RHS.")
    function = ("__device__ __forceinline__ void compute_drho_lserk(\n"
                "    const float* __restrict__ rho, float ParX, float ParY,\n"
                "    const float* __restrict__ Drive_arr, const float* __restrict__ Static_arr,\n"
                "    float* __restrict__ residual, const float lserk_a, const float lserk_dt,\n"
                "    const float* __restrict__ Const_arr)\n"
                "{\n    " + "\n    ".join(fused) + "\n}\n\n")
    position = kernel_code.find("__global__")
    if position < 0:
        raise ValueError("Kernel template has no __global__ kernel for the LSRK4 RHS.")
    return kernel_code[:position] + function + kernel_code[position:]


def _simplify_affine(expr, states):
    """Simplify each state coefficient separately when ``expr`` is affine in ``states``.

    Simplifying a whole Lindblad equation puts drive-dependent terms over a common
    denominator, e.g. ``(1/E)*(a*rho_1 + E*b*rho_2)``, which costs an extra multiplication
    per term in every RHS evaluation. Coefficient-wise simplification keeps ``a/E*rho_1 +
    b*rho_2``. Expressions that are not affine in the states are simplified as a whole.
    """
    groups = sp.collect(sp.expand(expr), states, evaluate=False)
    if not all(key == 1 or key in states for key in groups):
        return sp.simplify(expr)
    return sp.Add(*((coeff if coeff.is_number else sp.simplify(coeff)) * key
                    for key, coeff in groups.items()))


def _lindblad_rows(N, H, Col_Ops, rows, pre_expand, collect_rho):
    """Reduced RHS equations of the given density-matrix rows (``symbolic_workers`` worker).

    Repeats the matrix code of ``build_reduced_lindblad_rhs`` element by element, so the
    equations are identical. Returns ``{(i, j): [equations]}``: the real part for ``i == j``,
    the real and imaginary parts for ``i < j``.
    """
    rho, meta = build_independent_rho(N)
    states = meta["rho_syms"]
    last_population = 1 - sum(rho[j, j] for j in range(N - 1))
    terms = [(L, L.H, L.H * L) for L in Col_Ops]
    out = {}
    for i in rows:
        H_row, rho_row = H[i, :], rho[i, :]
        L_rho_rows = [L[i, :] * rho for L, _, _ in terms]
        for j in range(i, N):  # rows 0..N-2: the last population is not a state variable
            expr = -sp.I * ((H_row * rho[:, j])[0] - (rho_row * H[:, j])[0])
            lind = sp.Integer(0)
            for (L, Ld, LdL), L_rho_row in zip(terms, L_rho_rows):
                lind += ((L_rho_row * Ld[:, j])[0]
                         - sp.Float(0.5) * ((LdL[i, :] * rho[:, j])[0] + (rho_row * LdL[:, j])[0]))
            expr = expr + lind
            if pre_expand:
                expr = sp.expand(expr)
            expr = expr.subs(rho[N - 1, N - 1], last_population)
            if pre_expand:
                expr = sp.expand(expr)
            parts = [sp.re(expr)] if i == j else [sp.re(expr), sp.im(expr)]
            parts = [sp.expand(p) if pre_expand else p for p in parts]
            parts = [_simplify_affine(sp.expand(p) if pre_expand else p, states) for p in parts]
            if collect_rho:
                parts = [sp.collect(p, states) for p in parts]
            out[(i, j)] = parts
    return out


AUTO_SYMBOLIC_WORKERS_MIN_DIMENSION = 10  # "auto": parallel from this Hilbert dimension


def _physical_cores():
    """Half the logical CPUs: the physical cores of a CPU with two threads per core."""
    return max(1, (os.cpu_count() or 2) // 2)


def _symbolic_worker_count(N, symbolic_workers, max_symbolic_workers):
    """Worker processes for the Lindblad build, at most one per density-matrix row (N - 1)."""
    if max_symbolic_workers is None:
        max_symbolic_workers = _physical_cores()
    if int(max_symbolic_workers) < 1:
        raise ValueError("max_symbolic_workers must be a positive integer or None.")
    if symbolic_workers in (None, "auto"):
        workers = (int(max_symbolic_workers) if N >= AUTO_SYMBOLIC_WORKERS_MIN_DIMENSION else 1)
    elif isinstance(symbolic_workers, str) or int(symbolic_workers) < 1:
        raise ValueError('symbolic_workers must be "auto" or a positive integer.')
    else:
        workers = int(symbolic_workers)
    return max(1, min(workers, N - 1))


def build_reduced_lindblad_rhs(N, H, Col_Ops, mean_operator, *, pre_expand=False,
                                collect_rho=False, factor_terms=False, symbolic_workers=1,
                                max_symbolic_workers=None):
    """Build the symbolic Lindblad RHS used by every GQIS solver.

    The density matrix is reduced to ``N*N - 1`` real variables by enforcing
    Hermiticity and unit trace. Keeping this derivation in one function ensures
    that CUDA, Julia, and CPU reference solvers solve the same physical ODE.

    Args:
        N: Hilbert-space dimension.
        H: ``N x N`` symbolic Hamiltonian.
        Col_Ops: Iterable of ``N x N`` collapse operators.
        mean_operator: ``N x N`` operator whose expectation value is evaluated.
        pre_expand: Expand matrix and reduced RHS expressions before simplification.
        collect_rho: Collect each reduced RHS expression by state variables.
        factor_terms: Factor symbolic terms before returning the expressions.
        symbolic_workers: Processes that build and simplify the equations, each taking every
            k-th density-matrix row; 1 (default here) works in this process, ``"auto"`` uses
            ``max_symbolic_workers`` from Hilbert dimension 10 (starting the processes costs
            about 1-2 s). At most N - 1 are used. The equations are identical either way.
            Not combined with ``factor_terms``.
        max_symbolic_workers: Limit for ``"auto"``; ``None`` means half the logical CPUs.

    Returns:
        ``(drho_eqs, mean_re, mean_im, meta)`` where ``drho_eqs`` contains the
        ``N*N - 1`` real ODE expressions, ``mean_re`` and ``mean_im`` are the
        real and imaginary parts of the expectation value, and ``meta`` is the
        metadata returned by :func:`build_independent_rho`.
    """
    rho, meta = build_independent_rho(N)

    mean_val = sp.simplify(sp.Trace(mean_operator * rho))
    mean_re_expr = sp.simplify(sp.re(mean_val))
    mean_im_expr = sp.simplify(sp.im(mean_val))

    workers = _symbolic_worker_count(N, symbolic_workers, max_symbolic_workers)
    if workers > 1 and not factor_terms:
        from concurrent.futures import ProcessPoolExecutor
        # Rows have fewer elements further down; dealing them out in turn balances the groups.
        groups = [list(range(w, N - 1, workers)) for w in range(workers)]
        groups = [g for g in groups if g]
        with ProcessPoolExecutor(max_workers=len(groups)) as pool:
            parts = {}
            for result in pool.map(_lindblad_rows, *zip(*[(N, H, list(Col_Ops), g, pre_expand,
                                                           collect_rho) for g in groups])):
                parts.update(result)
        drho_eqs = [parts[(i, i)][0] for i in range(N - 1)]
        for i in range(N - 1):
            for j in range(i + 1, N):
                drho_eqs.extend(parts[(i, j)])
        return drho_eqs, mean_re_expr, mean_im_expr, meta

    comm = -sp.I * (H * rho - rho * H)
    lind = sp.zeros(N)
    for L in Col_Ops:
        Ld = L.H
        LdL = Ld * L
        lind += L * rho * Ld - sp.Float(0.5) * (LdL * rho + rho * LdL)

    drho_full = comm + lind
    if pre_expand:
        drho_full = sp.expand(drho_full)
    if factor_terms:
        drho_full = sp.factor_terms(drho_full, clear=True)

    # build_independent_rho already writes the final diagonal element in terms
    # of the first N - 1 populations. This explicit substitution also handles
    # expressions that SymPy has retained in an equivalent unreduced form.
    last_population = 1 - sum(rho[j, j] for j in range(N - 1))
    drho_full = drho_full.subs(rho[N - 1, N - 1], last_population)
    if pre_expand:
        drho_full = sp.expand(drho_full)
    if factor_terms:
        drho_full = sp.factor_terms(drho_full, clear=True)

    drho_eqs = []
    for i in range(N - 1):
        expr = sp.re(drho_full[i, i])
        drho_eqs.append(sp.expand(expr) if pre_expand else expr)
    for i in range(N):
        for j in range(i + 1, N):
            re_ij = sp.re(drho_full[i, j])
            im_ij = sp.im(drho_full[i, j])
            drho_eqs.append(sp.expand(re_ij) if pre_expand else re_ij)
            drho_eqs.append(sp.expand(im_ij) if pre_expand else im_ij)

    drho_eqs = [_simplify_affine(sp.expand(expr) if pre_expand else expr, meta["rho_syms"])
                for expr in drho_eqs]
    if factor_terms:
        drho_eqs = [sp.factor_terms(expr, clear=True) for expr in drho_eqs]
    if collect_rho:
        drho_eqs = [sp.collect(expr, meta["rho_syms"]) for expr in drho_eqs]

    return drho_eqs, mean_re_expr, mean_im_expr, meta


def generate_unrolled_drho(N, H, Drive_symbol, Col_Ops, mean_operator, drive_expr=None, *,
                           runtime_const_syms=(), pre_expand=False, collect_rho=False,
                           factor_terms=False, cse_batch_size=1, cse_simplify=False,
                           hoist_rho_independent=True):
    """Generate CUDA code fragments for a symbolic Lindblad model.

    Args:
        N: Hilbert-space dimension.
        H: SymPy Hamiltonian matrix. It may contain one or more drive placeholder
            symbols.
        Drive_symbol: Default placeholder symbol used when ``drive_expr`` is a
            single expression rather than a dictionary.
        Col_Ops: List of SymPy collapse-operator matrices.
        mean_operator: SymPy operator used to calculate the scalar expectation value.
        drive_expr: Either one SymPy expression mapped to ``Drive_symbol`` or a
            dictionary ``{drive_placeholder_symbol: expression}``.
        runtime_const_syms: Symbols that must remain runtime constants and be
            read from ``Const_arr`` in the CUDA kernel.
        pre_expand: Expand symbolic expressions before simplification.
        collect_rho: Collect RHS expressions by reduced density-vector symbols.
        factor_terms: Factor symbolic terms before code generation.
        cse_batch_size: Number of RHS equations per CSE batch, or ``None`` for
            one global CSE pass.
        cse_simplify: Simplify CSE expressions before printing.
        hoist_rho_independent: Move terms independent of ``rho`` into static or
            drive code blocks.

    Returns:
        Tuple ``(static_lines, drive_lines, drive_alias_lines, drho_lines,
        mean_line, final_line, static_syms, drive_syms, hoisted_syms)``. These
        strings are inserted into the CUDA template by ``mesolve_2D``.
    """
    drho_eqs, mean_re_expr, mean_im_expr, meta = build_reduced_lindblad_rhs(
        N, H, Col_Ops, mean_operator, pre_expand=pre_expand, collect_rho=collect_rho,
        factor_terms=factor_terms)
    return _generate_ode_code(drho_eqs, mean_re_expr, mean_im_expr, meta["rho_syms"],
                              Drive_symbol, drive_expr, runtime_const_syms=runtime_const_syms,
                              cse_batch_size=cse_batch_size, cse_simplify=cse_simplify,
                              hoist_rho_independent=hoist_rho_independent)


def _generate_ode_code(drho_eqs, mean_re_expr, mean_im_expr, state_symbols,
                       Drive_symbol, drive_expr, *, runtime_const_syms=(),
                       cse_batch_size=None, cse_simplify=True, hoist_rho_independent=True,
                       extract_independent=False):
    """Optimize the RHS symbolically, then print it as CUDA fragments (see ``generate_unrolled_drho``)."""
    plan = _optimize_ode_rhs(drho_eqs, mean_re_expr, mean_im_expr, state_symbols, Drive_symbol,
                             drive_expr, runtime_const_syms=runtime_const_syms,
                             cse_batch_size=cse_batch_size, cse_simplify=cse_simplify,
                             hoist_rho_independent=hoist_rho_independent,
                             extract_independent=extract_independent)
    return _emit_cuda_ode_code(plan)


def _optimize_ode_rhs(drho_eqs, mean_re_expr, mean_im_expr, state_symbols, Drive_symbol,
                      drive_expr, *, runtime_const_syms=(), cse_batch_size=None, cse_simplify=True,
                      hoist_rho_independent=True, extract_independent=False):
    """Device-independent RHS optimization: CSE, hoisting and per-term evaluation level.

    Returns a plan dict of SymPy expressions. Each hoisted temporary gets a level: ``inline``
    (constants only), ``negate`` (minus an earlier one), ``static`` (sweep parameters and
    constants: once per thread) or ``drive`` (time-dependent: once per solver stage).
    ``_emit_cuda_ode_code`` prints the plan; other emitters can print the same plan.
    """
    rho_sym_names = {str(s) for s in state_symbols}
    param_sym_names = {"ParX", "ParY"}

    independent = {}
    if extract_independent and hoist_rho_independent:
        state_set = set(state_symbols)
        def extract(expr):
            if not expr.is_Atom and expr.free_symbols and not expr.free_symbols & state_set:
                if expr not in independent:
                    independent[expr] = sp.Symbol(f"ode_hoist_{len(independent)}", real=True)
                return independent[expr]
            return expr.func(*(extract(arg) for arg in expr.args)) if expr.args else expr
        drho_eqs = [extract(e) for e in drho_eqs]
        mean_re_expr, mean_im_expr = extract(mean_re_expr), extract(mean_im_expr)
    hoisted_subs, blocks = _cse_plan(drho_eqs, state_symbols, batch_size=cse_batch_size,
                                     do_simplify=cse_simplify,
                                     hoist_rho_independent=hoist_rho_independent)
    hoisted_subs = [(symbol, expr) for expr, symbol in independent.items()] + hoisted_subs
    # Drive expressions can be a single Expr (mapped to Drive_symbol) or a dict of Symbol->Expr.
    drive_map = {}
    if drive_expr is not None:
        drive_map = drive_expr if isinstance(drive_expr, dict) else {Drive_symbol: drive_expr}
    drive_plan = _drive_plan(drive_map)

    # Names already bound to a kernel value: drives first, then hoisted temporaries.
    bound_names = {str(s) for s in drive_plan[0]}
    const_local_names = {str(s) for s in (runtime_const_syms or ())}
    static_sym_names = set()
    hoists = []
    if hoist_rho_independent and hoisted_subs:
        for s, expr in hoisted_subs:
            expr_sym_names = {str(sym) for sym in expr.free_symbols}
            coeff, rest = expr.as_coeff_Mul()
            if (not expr_sym_names) or expr_sym_names.issubset(const_local_names):
                level = "inline"  # const-only: inline Const_arr expression, no array slot
                const_local_names.add(str(s))
            elif coeff == -1 and isinstance(rest, sp.Symbol) and str(rest) in bound_names:
                level = "negate"  # negation of an existing value: no new slot
            elif expr_sym_names.issubset(const_local_names | param_sym_names | static_sym_names):
                level = "static"
                static_sym_names.add(str(s))
            else:
                level = "drive"
            hoists.append((s, expr, level))
            bound_names.add(str(s))

    # Hoist static-only subexpressions from the result calculation too, so static basis
    # readout does not recompute sqrt/div terms every time step.
    mean_subs, mean_exprs = sp.cse([mean_re_expr, mean_im_expr],
                                   symbols=sp.numbered_symbols("mean_tmp"))
    mean_inline = {}
    mean_hoists = []
    for s, expr in mean_subs:
        expr = expr.xreplace(mean_inline)
        expr_sym_names = {str(sym) for sym in expr.free_symbols}
        if expr_sym_names & rho_sym_names or not expr_sym_names:
            mean_inline[s] = expr
            mean_exprs = [sp.simplify(e.subs(s, expr)) for e in mean_exprs]
        elif expr_sym_names.issubset(const_local_names | param_sym_names | static_sym_names):
            mean_hoists.append((s, expr, "static"))
            static_sym_names.add(str(s))
        else:
            mean_hoists.append((s, expr, "drive"))
    return {"blocks": blocks, "drive_plan": drive_plan, "hoists": hoists,
            "mean_hoists": mean_hoists, "mean_exprs": mean_exprs,
            "runtime_const_syms": tuple(runtime_const_syms or ())}


def _symbol_pattern(name):
    """Regex matching one symbol name as a whole identifier in generated code."""
    return re.compile(r"(?<![0-9A-Za-z_])" + re.escape(name) + r"(?![0-9A-Za-z_])")


def _emit_cuda_ode_code(plan):
    """Print an ``_optimize_ode_rhs`` plan as CUDA code fragments.

    Returns ``(static_lines, drive_lines, drive_alias_lines, drho_lines, mean_line,
    final_line, static_syms, drive_syms, hoisted_syms)`` for the CUDA kernel template.
    """
    drho_lines = _print_cse_blocks(plan["blocks"])
    drive_syms = list(plan["drive_plan"][0])
    drive_lines = _print_drive_plan(plan["drive_plan"], "Drive_arr")
    static_lines, static_syms, hoisted_syms = [], [], []
    # Replacement map: drives first, then static/drive hoisted temporaries as we emit them.
    name_to_repl = {str(s): f"Drive_arr[{i}]" for i, s in enumerate(drive_syms)}

    def code(expr):
        """Print one expression with the kernel values bound so far."""
        expr_code = my_ccode(expr)
        for name, repl in name_to_repl.items():
            expr_code = _symbol_pattern(name).sub(repl, expr_code)
        return expr_code

    def store(s, expr, level):
        """Assign one static or drive value to its next array slot."""
        if level == "static":
            array, index, kind = "Static_arr", len(static_syms), static_syms
            lines = static_lines
        else:
            # Store directly into Drive_arr to avoid an extra temp register for this symbol.
            array, index, kind = "Drive_arr", len(drive_syms) + len(hoisted_syms), hoisted_syms
            lines = drive_lines
        lines.append(f"{array}[{index}] = {code(expr)};")
        name_to_repl[str(s)] = f"{array}[{index}]"
        kind.append(s)

    for s, expr, level in plan["hoists"]:
        if level == "negate":
            repl = name_to_repl[str(expr.as_coeff_Mul()[1])]
            name_to_repl[str(s)] = repl[1:] if repl.startswith("-") else "-" + repl
        elif level == "inline":
            name_to_repl[str(s)] = f"({code(expr)})"
        else:
            store(s, expr, level)
    for s, expr, level in plan["mean_hoists"]:
        store(s, expr, level)

    mean_re = my_ccode(plan["mean_exprs"][0])
    mean_im = my_ccode(plan["mean_exprs"][1])
    mean_line = f"avg.x += {mean_re}; avg.y += {mean_im};"
    final_line = f"avg.x = {mean_re}; avg.y = {mean_im};"

    # Replace Drive/static symbols and hoisted temporaries, then runtime constants
    # (Const_arr[idx]), in the RHS/mean with array accesses.
    replacements = list(name_to_repl.items()) + [
        (str(sym), f"Const_arr[{idx}]") for idx, sym in enumerate(plan["runtime_const_syms"])]
    for name, repl in replacements:
        pat = _symbol_pattern(name)
        drho_lines = [pat.sub(repl, line) for line in drho_lines]
        static_lines = [pat.sub(repl, line) for line in static_lines]
        drive_lines = [pat.sub(repl, line) for line in drive_lines]
        mean_line = pat.sub(repl, mean_line)
        final_line = pat.sub(repl, final_line)

    # Apply float/sinf/cosf/etc normalizations.
    static_lines = tidy_c_lines(static_lines) if static_lines else []
    drive_lines = tidy_c_lines(drive_lines) if drive_lines else []
    drho_lines = tidy_c_lines(drho_lines) if drho_lines else []
    # compute_drho needs no alias lines (they are merged into compute_drives).
    return (static_lines, drive_lines, [], drho_lines, mean_line, final_line,
            static_syms, drive_syms, hoisted_syms)


def cse_emit_c_lines(drho_exprs, rho_syms, *, batch_size=None, do_simplify=True,
                     hoist_rho_independent=False, return_hoist=False):
    """Emit CUDA C lines for the reduced RHS vector.

    Args:
        drho_exprs: SymPy expressions for ``d rho / dt`` in reduced-vector
            order.
        rho_syms: Reduced state-vector symbols returned by
            ``build_independent_rho``.
        batch_size: ``None`` for one global CSE pass, or a positive integer for
            batched CSE to lower register pressure.
        do_simplify: Simplify expressions before printing.
        hoist_rho_independent: Return CSE substitutions that do not depend on
            the current state vector separately.
        return_hoist: If ``True``, return ``(hoisted_subs, lines)``.

    Returns:
        CUDA C assignment lines, or ``(hoisted_subs, lines)`` when
        ``return_hoist=True``.

    Notes for GPU kernels:
    - Full-vector CSE maximizes reuse but can increase the number of simultaneously-live
      temporaries, driving register pressure/spills.
    - Batched CSE reduces cross-equation reuse but limits live temporaries. With braces,
      this can materially reduce register count on CUDA.
    """
    hoist_subs, blocks = _cse_plan(drho_exprs, rho_syms, batch_size=batch_size,
                                   do_simplify=do_simplify,
                                   hoist_rho_independent=hoist_rho_independent)
    lines = _print_cse_blocks(blocks)
    return (hoist_subs, lines) if return_hoist else lines


def _print_cse_blocks(blocks):
    """Print ``_cse_plan`` blocks as CUDA lines (batched blocks in braces)."""
    lines = []
    for scoped, temps, outputs in blocks:
        if scoped:
            lines.append("{")  # scope temporaries so the compiler can drop them sooner
        lines += [f"float {s} = {my_ccode(expr)};" for s, expr in temps]
        lines += [f"d_rho[{i}] = {my_ccode(expr)};" for i, expr in outputs]
        if scoped:
            lines.append("}")
    return lines


def _cse_plan(drho_exprs, rho_syms, *, batch_size=None, do_simplify=True,
              hoist_rho_independent=False):
    """Symbolic part of ``cse_emit_c_lines``.

    Returns ``(hoist_subs, blocks)``: state-independent CSE temporaries, and blocks
    ``(scoped, temps, outputs)`` with ``temps`` as ``(symbol, expr)`` and ``outputs`` as
    ``(state index, expr)``; ``scoped`` marks batched blocks.
    """

    def _maybe_simplify(x):
        """Simplify one expression only when requested by caller."""
        return sp.simplify(x) if do_simplify else x

    def _partition_subs(cse_subs):
        """Split CSE temporaries into rho-dependent and rho-independent groups."""
        rho_set = set(rho_syms)
        rho_dependent = set()
        hoist = []
        keep = []
        # Keep the original ordering to preserve dependency topological order.
        for s, expr in cse_subs:
            fs = expr.free_symbols
            dep = bool(fs & rho_set) or bool(fs & rho_dependent)
            if dep:
                keep.append((s, expr))
                rho_dependent.add(s)
            else:
                hoist.append((s, expr))
        return hoist, keep

    def _block(exprs, base_i, prefix, scoped):
        """CSE one slice of RHS expressions into a block."""
        cse_subs, cse_exprs = sp.cse(exprs, symbols=sp.numbered_symbols(prefix=prefix),
                                     ignore=set(rho_syms))
        hoist_subs, keep_subs = (_partition_subs(cse_subs) if hoist_rho_independent else
                                 ([], cse_subs))
        temps = [(s, _maybe_simplify(expr)) for s, expr in keep_subs]
        outputs = [(base_i + i, _maybe_simplify(expr)) for i, expr in enumerate(cse_exprs)]
        return hoist_subs, (scoped, temps, outputs)

    if batch_size is None:
        # Preserve historical behavior: one global CSE pass for all equations.
        hoist_subs, block = _block(drho_exprs, 0, "t", False)
        return hoist_subs, [block]

    # Batched CSE: reduces live temporaries and often reduces register pressure.
    bs = int(batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be a positive integer, or None")
    hoist_all, blocks = [], []
    for batch_id, base in enumerate(range(0, len(drho_exprs), bs)):
        # Unique prefix per batch avoids name collisions if caller enables hoisting.
        hoist_subs, block = _block(drho_exprs[base:base + bs], base, f"t{batch_id}_", True)
        hoist_all.extend(hoist_subs)
        blocks.append(block)
    return hoist_all, blocks


def tidy_c_lines(lines):
    """Normalize generated CUDA code lines for the FP32 path.

    Args:
        lines: Iterable of generated C/CUDA source lines.

    Returns:
        List of cleaned lines. The cleanup replaces common math functions with
        single-precision variants and removes trivial multiplications/additions.
    """

    text = "\n    ".join(lines)
    text = re.sub(r"\bpow\(([^,]+),\s*2\)", r"(\1*\1)", text)
    text = re.sub(r"\bsqrt\(", "sqrtf(", text)
    text = re.sub(r"\bsin\(", "sinf(", text)
    text = re.sub(r"\bcos\(", "cosf(", text)
    text = re.sub(r"\bexp\(", "expf(", text)
    text = re.sub(r"\blog\(", "logf(", text)
    text = re.sub(r"\b1\.0+f?\*", "", text)
    text = re.sub(r"\b1(?:\.0+)?e[+\-]?0+f?\*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\+ 0\.0f?", "", text)
    text = re.sub(r"(?<![0-9a-zA-Z_.])([0-9]+\.[0-9]+)(?![0-9a-zA-Z_f])", r"\1f", text)
    return text.splitlines()


def emit_drive_code(drive_map, *, array_name="Drive_arr", inline_single_use_funcs=True):
    """Emit CUDA lines that compute time-dependent drive signals.

    Args:
        drive_map: Dictionary ``{Symbol: Expr}``. Keys are placeholder symbols
            used inside ``H``; values are expressions depending on time ``t``,
            sweep parameters ``ParX``/``ParY``, and runtime constants.
        array_name: CUDA array receiving drive values.
        inline_single_use_funcs: Reserved compatibility knob; currently drive
            expressions are emitted directly into ``array_name``.

    Returns:
        ``(drive_lines, alias_lines, drive_syms)``. ``alias_lines`` is kept for
        API compatibility and is currently always empty.
    """
    plan = _drive_plan(drive_map, inline_single_use_funcs=inline_single_use_funcs)
    return _print_drive_plan(plan, array_name), [], list(plan[0])


def _drive_plan(drive_map, *, inline_single_use_funcs=True):
    """Symbolic part of ``emit_drive_code``: ``(drive_syms, shared_function_temps, exprs)``."""
    if not drive_map:
        return [], [], []

    # Preserve user-provided ordering (dict insertion order) so Drive_arr[i] matches
    # the provided drive list. Prefer "Drive" first if present.
    syms = list(drive_map.keys())
    for i, s in enumerate(syms):
        if str(s) == "Drive":
            if i != 0:
                syms.insert(0, syms.pop(i))
            break

    exprs = [sp.factor(sp.expand(drive_map[s])) for s in syms]

    # Optionally inline one-off function calls (cos/sin/...) to avoid extra temps.
    # Only lift a function call into a temp if it occurs multiple times across all drives.
    func_counts = {}
    for e in exprs:
        for f in e.atoms(sp.Function):
            func_counts[f] = func_counts.get(f, 0) + 1
    funcs = sorted(func_counts.keys(), key=lambda f: str(f))

    func_subs = {}
    for i, f in enumerate(funcs):
        if inline_single_use_funcs and func_counts.get(f, 0) <= 1:
            continue
        func_subs[sp.Symbol(f"d{i}")] = f

    exprs_subbed = []
    for e in exprs:
        for sym, f in func_subs.items():
            e = e.subs(f, sym)
        exprs_subbed.append(e)
    return syms, list(func_subs.items()), exprs_subbed


def _print_drive_plan(plan, array_name="Drive_arr"):
    """Print a ``_drive_plan`` as CUDA lines."""
    _syms, function_temps, exprs = plan
    drive_lines = [f"float {sym} = {my_ccode(func)};" for sym, func in function_temps]
    # Write directly to Drive_arr to avoid duplicate temps (compiler can scalarize indices).
    drive_lines += [f"{array_name}[{i}] = {my_ccode(e)};" for i, e in enumerate(exprs)]
    return drive_lines


def _timestep_calibration_settings(value, ode=False):
    """Return validated calibrate_timesteps settings merged with the defaults."""
    settings = dict(TIMESTEP_CALIBRATION_DEFAULTS)
    if ode:
        settings.update(TIMESTEP_CALIBRATION_ODE_DEFAULTS)
    if value is not True:
        if not isinstance(value, dict):
            raise TypeError("calibrate_timesteps must be True, None or a dict of settings.")
        unknown = set(value) - set(settings)
        if unknown:
            raise ValueError(f"Unknown calibrate_timesteps setting(s) {sorted(unknown)}. "
                             f"Valid settings: {', '.join(settings)}.")
        settings.update(value)
    for name in ("rms_atol", "rms_rtol", "max_atol", "max_rtol"):
        settings[name] = float(settings[name])
        if settings[name] < 0.0:
            raise ValueError(f"calibrate_timesteps {name} must be non-negative.")
    settings["check_rms"] = settings["rms_atol"] > 0.0 or settings["rms_rtol"] > 0.0
    settings["check_max"] = settings["max_atol"] > 0.0 or settings["max_rtol"] > 0.0
    if not (settings["check_rms"] or settings["check_max"]):
        raise ValueError("calibrate_timesteps needs at least one non-zero RMS or max tolerance.")
    settings["sample_stride"] = int(settings["sample_stride"])
    settings["max_steps"] = int(settings["max_steps"])
    if settings["sample_stride"] < 1 or settings["max_steps"] < 1:
        raise ValueError("calibrate_timesteps sample_stride and max_steps must be positive.")
    settings["bisection_steps"] = int(settings["bisection_steps"])
    if settings["bisection_steps"] < 0:
        raise ValueError("calibrate_timesteps bisection_steps must be zero or a positive integer.")
    settings["full_grid_bisection"] = bool(settings["full_grid_bisection"])
    every = settings["recalibrate_every"]
    if every is not None and int(every) < 1:
        raise ValueError("calibrate_timesteps recalibrate_every must be None or a positive integer.")
    settings["recalibrate_every"] = None if every is None else int(every)
    if settings["on_fail"] not in {"warning", "ignore", "error"}:
        raise ValueError("calibrate_timesteps on_fail must be 'warning', 'ignore' or 'error'.")
    return settings


def _timestep_calibration_key(args, settings):
    """Identify a model and accuracy request, independently of code-generation options."""
    def key(expr):
        # SymPy expressions are hashable and compare structurally, so no printing is needed.
        if isinstance(expr, dict):
            return frozenset((k, sp.sympify(v)) for k, v in expr.items())
        if isinstance(expr, (list, tuple)):
            return tuple(sp.sympify(e) for e in expr)
        if isinstance(expr, sp.MatrixBase):
            return expr.shape, tuple(expr)
        return sp.sympify(expr)

    const_values = args["const_values"] or {}
    keep = args["keep_symbolic_consts"]
    kept = set(args["runtime_consts"] or {})
    if args["auto_runtime_consts"] or keep in ("all", "auto", "const_values"):
        kept |= set(const_values)
    elif keep is not None:
        kept |= set(keep)
    fixed_consts = tuple(sorted((str(k), float(v)) for k, v in const_values.items() if k not in kept))
    sweep_symbols = tuple(str(s) for s in (args["var_arrays"] or {})) + tuple(
        str(s) for s in (args["rho0_var_arrays"] or {}))
    if args.get("compiled_rhs") is not None:
        model = (args["compiled_rhs"].cache_key,)
    else:
        model = (key(args["H"]), key(args["Drive"]), tuple(key(L) for L in args["Col_Ops"]),
                 key(args["mean_operator"]),
                 key(args["rho0"]) if args["rho0"] is not None else None, fixed_consts)
    return ("timestep_calibration_v2", *model, sweep_symbols,
            get_solver_spec(args["solver"]).name, bool(args["fp64"]), args["output_mode"],
            float(args["warmup_time"]), args["_ode_size"],
            *(settings[name] for name in ("rms_atol", "rms_rtol", "max_atol", "max_rtol",
                                          "sample_stride", "max_steps", "bisection_steps",
                                          "full_grid_bisection")))


def _sweep_ranges(args):
    """Return (min, max) of every sweep axis, used to notice a changed sweep range."""
    ranges = []
    for name in ("var_arrays", "rho0_var_arrays"):
        for values in (args[name] or {}).values():
            values = values if hasattr(values, "min") else np.asarray(values)
            ranges.append((float(values.min()), float(values.max())))
    return tuple(ranges)


def _undersampled_arguments(args, stride):
    """Return solver arguments restricted to every stride-th point on each sweep axis."""
    test = dict(args)
    test.update(calibrate_timesteps=None, return_time_trace=False, return_timing_info=False,
                timings=False, return_device=False, ignore_non_finite_output=True,
                Actual_Kernel_Save=False, beep_on_error=False)
    def indices(length):
        # Always keep the last point: sweep edges (e.g. the largest amplitude) are often hardest.
        picked = list(range(0, length, stride))
        return picked if picked[-1] == length - 1 else picked + [length - 1]

    def pick(values, picked):
        return [values[i] for i in picked] if isinstance(values, (list, tuple)) else values[picked]

    axes, full_grid = [], True
    for name in ("var_arrays", "rho0_var_arrays"):
        if args[name]:
            test[name] = {}
            for sym, values in args[name].items():
                axes.append(indices(len(values)))
                full_grid = full_grid and len(axes[-1]) == len(values)
                test[name][sym] = pick(values, axes[-1])
    if args["rho0_values"] is not None:
        values = args["rho0_values"]
        values = values if hasattr(values, "ndim") else np.asarray(values)
        values = values[axes[0]] if axes else values
        test["rho0_values"] = values[:, axes[1]] if values.ndim == 3 and len(axes) > 1 else values
    return test, full_grid


def _calibrate_timestep(args, settings, base_steps, duration):
    """Find the coarsest step count whose undersampled result map meets the limits.

    The step count is doubled until the error estimate passes. With ``bisection_steps > 0`` the
    bracket between the last failing and the first passing count is then halved (geometrically)
    that many times; if the supplied grid already passes, it is first halved until a count fails.
    Bisection is skipped when the test grid is the full grid (no undersampling): the passing
    full-grid result is already computed, so coarser candidates would only add full-grid runs.
    ``full_grid_bisection=True`` bisects anyway, giving a coarser stored step for repeated calls.
    """
    start = time.time()
    test, full_grid = _undersampled_arguments(args, settings["sample_stride"])
    order = get_solver_spec(args["solver"]).order
    results = {}  # kept only when the test grid is the full grid, for reuse by the final run

    def run(steps):
        test["tlist"] = np.linspace(0.0, duration, steps + 1)
        result = np.asarray(_solve_symbolic_system(**test))
        if full_grid:
            results[steps] = result
        return result

    def check(steps, coarse, fine_steps, fine):
        """Richardson error estimate of the ``steps`` result from a finer one; returns (record, passed)."""
        record = {"steps": steps, "rms": np.inf, "max": np.inf}
        if not (np.all(np.isfinite(coarse)) and np.all(np.isfinite(fine))):
            return record, False  # an unstable run never passes, also with relative limits
        scale = 1.0 / (1.0 - (steps / fine_steps)**order)  # 2**p/(2**p-1) for an M -> 2M pair
        difference = np.abs(fine - coarse) * scale
        magnitude = np.abs(fine)
        record.update(rms=float(np.sqrt(np.mean(difference**2))), max=float(np.max(difference)))
        rms_limit = settings["rms_atol"] + settings["rms_rtol"] * float(np.sqrt(np.mean(magnitude**2)))
        max_limit = settings["max_atol"] + settings["max_rtol"] * float(np.max(magnitude))
        passed = ((not settings["check_rms"] or record["rms"] <= rms_limit)
                  and (not settings["check_max"] or record["max"] <= max_limit))
        return record, passed

    steps, coarse = base_steps, run(base_steps)
    tested, status, accepted = [], None, None
    while status is None:
        if 2 * steps > settings["max_steps"]:
            status, accepted = "max_steps", steps
            break
        fine = run(2 * steps)
        record, passed = check(steps, coarse, 2 * steps, fine)
        tested.append(record)
        if passed:
            status, accepted = "converged", steps
            reference_steps, reference = 2 * steps, fine
        elif len(tested) > 1 and np.isfinite(record["rms"]) and record["rms"] >= tested[-2]["rms"]:
            # Refinement no longer reduces the change: the rounding floor was reached.
            best = min(tested, key=lambda item: item["rms"])
            status, accepted = "precision_floor", best["steps"]
        steps, coarse = 2 * steps, fine

    doubling_steps = accepted
    if (status == "converged" and settings["bisection_steps"] > 0
            and (settings["full_grid_bisection"] or not full_grid)):
        # Every candidate is compared with the finest result computed so far (2x the passing
        # count), which is far more accurate than any candidate, so the estimate stays reliable.
        def passes(candidate):
            record, passed = check(candidate, run(candidate), reference_steps, reference)
            tested.append({**record, "reference_steps": reference_steps})
            return passed

        failing = accepted // 2 if accepted > base_steps else None
        halvings = 0
        while failing is None and accepted >= 2 and halvings < _CALIBRATION_MAX_HALVINGS:
            halvings += 1  # the supplied grid passed: look for a failing coarser count
            candidate = accepted // 2
            if passes(candidate):
                accepted = candidate
            else:
                failing = candidate
        for _ in range(settings["bisection_steps"] if failing is not None else 0):
            candidate = int(round(np.sqrt(failing * accepted)))
            if candidate <= failing or candidate >= accepted:
                break
            if passes(candidate):
                accepted = candidate
            else:
                failing = candidate

    info = {"status": status, "steps": accepted, "dt": duration / accepted, "tested": tested,
            "doubling_steps": doubling_steps, "test_grid": coarse.shape[:2],
            "calibration_s": time.time() - start, "full_grid_results": results}
    if status != "converged":
        def limit(kind):
            atol, rtol = settings[f"{kind}_atol"], settings[f"{kind}_rtol"]
            return f"{'RMS' if kind == 'rms' else 'max'} {atol:g}" + (f" + {rtol:g} relative" if rtol else "")
        requested = ", ".join(limit(kind) for kind in ("rms", "max") if settings[f"check_{kind}"])
        reason = "does not converge" if status == "precision_floor" else "is not reached within max_steps"
        if tested:
            best = next((item for item in tested if item["steps"] == accepted), tested[-1])
            achieved = f"best achieved RMS {best['rms']:.2e}, max {best['max']:.2e} with {accepted} steps"
        else:
            achieved = f"no refinement beyond {accepted} steps was possible"
        message = f"calibrate_timesteps: requested precision ({requested}) {reason}; {achieved}."
        if status == "precision_floor" and not args["fp64"]:
            message += " If better precision is required, consider switching to the much slower fp64=True."
        if settings["on_fail"] == "error":
            raise RuntimeError(message)
        if settings["on_fail"] == "warning":
            warnings.warn(message, RuntimeWarning, stacklevel=4)
    return info


def _solve_with_timestep_calibration(args):
    """Choose a uniform time step on an undersampled grid, then run the full sweep."""
    settings = _timestep_calibration_settings(args["calibrate_timesteps"], ode=args["_ode_size"] is not None)
    tlist = np.asarray(args["tlist"], dtype=np.float64)
    if tlist.ndim != 1 or len(tlist) < 2:
        raise ValueError("tlist must be a one-dimensional array with at least two samples.")
    base_steps, duration = len(tlist) - 1, float(tlist[-1] - tlist[0])
    key = _timestep_calibration_key(args, settings)
    entry = None if settings["reset"] else _TIMESTEP_CALIBRATION_CACHE.get(key)
    calls = entry["calls"] if entry else 0
    every = settings["recalibrate_every"]
    ranges = _sweep_ranges(args)
    info = None
    if entry is None or (every is not None and calls % every == 0):
        info = _calibrate_timestep(args, settings, base_steps, duration)
        # Only refine: a recalibration never makes later calls coarser than earlier ones.
        dt = info["dt"] if entry is None else min(entry["dt"], info["dt"])
        if entry is not None:  # the stored step now covers both the old and the new range
            ranges = tuple((min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(entry["ranges"], ranges))
        entry = {"dt": dt, "ranges": ranges, "range_warned": False}
    else:
        dt = entry["dt"]
        outside = any(lo < old_lo or hi > old_hi for (lo, hi), (old_lo, old_hi) in zip(ranges, entry["ranges"]))
        if outside and settings["range_warning"] and not entry["range_warned"]:
            entry["range_warned"] = True  # warn once per stored step
            warnings.warn("calibrate_timesteps: the sweep range extends beyond the range used for calibration; "
                          "the stored step is reused without a new test. Use reset=True or recalibrate_every "
                          "to test the new range.", RuntimeWarning, stacklevel=3)
    entry["calls"] = calls + 1
    _TIMESTEP_CALIBRATION_CACHE[key] = entry

    aligned = args["return_time_trace"] or args["solver_samples_per_period"] is not None
    # Without bisection (or with aligned traces) the supplied tlist is the coarsest grid used.
    steps = int(np.ceil(duration / dt * (1.0 - 1e-9)))
    if settings["bisection_steps"] == 0 or aligned:
        steps = max(base_steps, steps)
    factor = 1
    if aligned:
        # Trace samples must stay aligned with the supplied grid: refine it by a power of two.
        while base_steps * factor < steps and base_steps * factor < settings["max_steps"]:
            factor *= 2
        steps = base_steps * factor
    steps = min(steps, max(base_steps, settings["max_steps"]))
    run_args = dict(args, calibrate_timesteps=None, tlist=np.linspace(0.0, duration, steps + 1))
    if args["return_time_trace"] and args["time_trace_samples_per_period"] is None:
        run_args["time_trace_every"] = (args["time_trace_every"] or 1) * factor
    if args["solver_samples_per_period"] is not None:
        run_args["solver_samples_per_period"] = args["solver_samples_per_period"] * factor
    if info is not None and args["timings"]:
        tested = info["tested"]
        last = f"; last RMS/max change {tested[-1]['rms']:.2e}/{tested[-1]['max']:.2e}" if tested else ""
        print(f"calibrate_timesteps: {steps} steps ({info['status']}), tested from "
              f"{base_steps} steps on {info['test_grid'][0]}x{info['test_grid'][1]} points in "
              f"{info['calibration_s']:.2f}s{last}")

    # With a full-grid test, the result for the chosen step count is already available.
    reusable = (info is not None and steps in info["full_grid_results"]
                and not args["return_time_trace"] and not args["return_timing_info"])
    if reusable:
        result = info["full_grid_results"][steps]
        if not np.all(np.isfinite(result)) and not args["ignore_non_finite_output"]:
            raise RuntimeError("Non-finite values detected in mesolve_2D output. Try reducing dt "
                               "(increase time samples) or check model parameters/RHS.")
        return cp.asarray(result) if args["return_device"] else result
    # A step refined below the doubling result is tested only on undersampled points; if the
    # full grid is unstable somewhere, fall back to the doubling result and store that step.
    fallback = (info is not None and info["status"] == "converged" and not aligned
                and info["doubling_steps"] > steps)
    if fallback:
        run_args["ignore_non_finite_output"] = True
    result = _solve_symbolic_system(**run_args)
    if fallback:
        values = result[0] if isinstance(result, tuple) else result
        xp = cp if args["return_device"] else np
        if not bool(xp.all(xp.isfinite(values))):
            warnings.warn(f"calibrate_timesteps: {steps} steps passed on the test points but gave non-finite "
                          f"values on the full grid; using {info['doubling_steps']} steps instead.",
                          RuntimeWarning, stacklevel=3)
            steps = info["doubling_steps"]
            entry["dt"] = duration / steps
            run_args.update(tlist=np.linspace(0.0, duration, steps + 1),
                            ignore_non_finite_output=args["ignore_non_finite_output"])
            result = _solve_symbolic_system(**run_args)
    if args["return_timing_info"]:
        result[-1]["timestep_calibration"] = {
            "steps": steps, "dt": duration / steps, "calibrated_this_call": info is not None,
            "details": {k: v for k, v in info.items() if k != "full_grid_results"} if info else None}
    return result


@dataclass(eq=False)
class CompiledRHS:
    """Prepared and optimized ODE right-hand side, independent of the device code.

    Made by ``compile_lindblad_rhs`` or ``compile_ode_rhs``. Pass it to ``mesolve_2D`` or
    ``odesolve_2D`` in place of the model to start from code emission. The symbolic
    equations (``equations``, ``mean_re``, ``mean_im``) and the optimized ``plan`` (CSE
    temporaries with their evaluation level) can also be analysed or printed for other devices.
    """

    kind: str  # "lindblad" (mesolve_2D) or "ode" (odesolve_2D)
    state_size: int  # real state variables
    sweep_symbols: tuple  # mapped to ParX, ParY in this order
    state_symbols: tuple  # internal state symbols rho[i]
    equations: tuple  # d state / dt in internal symbols (ParX/ParY, Drive placeholders, constants)
    mean_re: object  # observable, real and imaginary parts
    mean_im: object
    drive_map: object  # {placeholder: expression of t, ParX, ParY and constants}
    rho0_exprs: object  # symbolic initial state, or None
    runtime_const_syms: tuple  # Const_arr order: kept constants, then derived expressions
    base_runtime_syms: tuple
    derived_const_exprs: tuple
    const_values: dict  # default values of constants kept symbolic
    plan: dict  # optimized RHS from _optimize_ode_rhs
    cache_key: tuple
    timings: dict  # ode_prepare_s, ode_optimize_s
    ode: object = None  # odesolve_2D: user symbol mapping and observable information
    emitted: dict = field(default_factory=dict, repr=False)  # emitted code per device variant


def _prepare_rhs_model(H, Drive, Col_Ops, mean_operator, *, sweep_symbols, const_values,
                       runtime_consts, keep_symbolic_consts, auto_runtime_consts, rho0, ode_size):
    """Validate the model and substitute sweep symbols, fixed constants and runtime constants."""
    const_values = {} if const_values is None else const_values
    runtime_consts = runtime_consts or {}
    if not isinstance(runtime_consts, dict):
        raise ValueError("runtime_consts must be a dict of {Symbol: value}")
    N = H.shape[0]
    state_size = N * N - 1 if ode_size is None else ode_size
    rho0_exprs = None
    if rho0 is not None:
        rho0_exprs = (rho_matrix_to_independent_exprs(rho0) if ode_size is None
                      else [sp.sympify(value) for value in rho0])
        if len(rho0_exprs) != state_size:
            raise ValueError("rho0 does not match Hamiltonian size.")
    for L in Col_Ops:
        if L.shape != (N, N):
            raise ValueError(f"Collapse operator dimension {L.shape} does not match H {N, N}")
    if ode_size is None and mean_operator.shape != (N, N):
        raise ValueError(f"Mean operator dimension {mean_operator.shape} does not match H {N, N}")

    subs = {sym: sp.Symbol(name, real=True) for sym, name in zip(sweep_symbols, ["ParX", "ParY"])}
    H = H.subs(subs)
    if isinstance(Drive, dict):
        Drive = {sym: sp.sympify(expr).subs(subs) for sym, expr in Drive.items()}
    else:
        Drive = sp.sympify(Drive).subs(subs)
    Col_Ops = [L.subs(subs) for L in Col_Ops]
    mean_operator = mean_operator.subs(subs)
    if rho0_exprs is not None:
        rho0_exprs = [expr.subs(subs) for expr in rho0_exprs]

    if keep_symbolic_consts is None:
        keep_syms = set(const_values.keys()) if auto_runtime_consts else set()
    elif keep_symbolic_consts == "all":
        keep_syms = set(const_values.keys())
    elif keep_symbolic_consts in ("auto", "const_values"):
        keep_syms = set(const_values.keys())
    else:
        keep_syms = set(keep_symbolic_consts)
    keep_syms = keep_syms.union(set(runtime_consts.keys()))
    for s in keep_syms:
        if not isinstance(s, sp.Symbol):
            raise ValueError("keep_symbolic_consts/runtime_consts keys must be SymPy Symbols")

    base_runtime_syms = sorted(list(keep_syms), key=lambda s: str(s))
    base_runtime_set = set(base_runtime_syms)

    # Constants not kept symbolic are folded numerically before CSE/codegen.
    const_values_eff = {k: v for k, v in const_values.items() if k not in base_runtime_set}
    if const_values_eff:
        const_subs = {k: sp.Float(float(v)) for k, v in const_values_eff.items()}
        H = H.xreplace(const_subs)
        if isinstance(Drive, dict):
            Drive = {k: v.xreplace(const_subs) for k, v in Drive.items()}
        else:
            Drive = Drive.xreplace(const_subs)
        Col_Ops = [L.xreplace(const_subs) for L in Col_Ops]
        mean_operator = mean_operator.xreplace(const_subs)
        if rho0_exprs is not None:
            rho0_exprs = [expr.xreplace(const_subs) for expr in rho0_exprs]

    def _iter_all_exprs():
        """Yield every symbolic expression that can affect generated CUDA code."""
        yield from H
        if isinstance(Drive, dict):
            yield from Drive.values()
        else:
            yield Drive
        for L in Col_Ops:
            yield from L
        yield from mean_operator
        if rho0_exprs is not None:
            yield from rho0_exprs

    # Runtime constants that do not appear in H/Drive/collapse/observable/rho0
    # should not change generated CUDA code or the kernel cache key.
    used_base_runtime_set = set()
    for expr in _iter_all_exprs():
        used_base_runtime_set.update(expr.free_symbols & base_runtime_set)
    if used_base_runtime_set != base_runtime_set:
        base_runtime_syms = [s for s in base_runtime_syms if s in used_base_runtime_set]
        base_runtime_set = set(base_runtime_syms)

    def _collect_const_only_exprs(base_syms):
        """Find reusable expressions depending only on runtime constants."""
        if not base_syms:
            return []
        cands = {}
        for expr in _iter_all_exprs():
            for sub in sp.preorder_traversal(expr):
                if sub.is_Number or sub.is_Symbol:
                    continue
                fs = sub.free_symbols
                if fs and fs.issubset(base_syms):
                    cands[sp.srepr(sub)] = sub
        # Keep non-trivial const-only expressions to move work out of the kernel.
        selected = [x for x in cands.values() if x.count_ops() > 0]
        selected.sort(key=lambda x: (x.count_ops(), sp.srepr(x)), reverse=True)
        return selected

    derived_const_exprs = _collect_const_only_exprs(base_runtime_set)

    def _derived_symbol(expr, idx, used):
        """Create a readable unique symbol name for a derived runtime constant."""
        name = None
        if expr.is_Pow and expr.base.is_Symbol:
            if expr.exp == sp.Rational(1, 2):
                name = f"{expr.base}__sqrt"
            elif expr.exp == sp.Rational(-1, 2):
                name = f"{expr.base}__rsqrt"
            elif expr.exp == -1:
                name = f"{expr.base}__inv"
        elif (expr.func in (sp.sin, sp.cos, sp.tan, sp.exp, sp.log) and len(expr.args) == 1
              and expr.args[0].is_Symbol):
            name = f"{expr.args[0]}__{expr.func.__name__}"
        if not name:
            name = f"ConstExpr_{idx}"
        base_name = name
        suffix = 1
        while name in used:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used.add(name)
        return sp.Symbol(name, real=True)

    used_names = {str(s) for s in base_runtime_syms}
    derived_const_syms = [_derived_symbol(expr, i, used_names)
                          for i, expr in enumerate(derived_const_exprs)]
    if derived_const_exprs:
        repl_map = {expr: sym for expr, sym in zip(derived_const_exprs, derived_const_syms)}
        H = H.xreplace(repl_map)
        if isinstance(Drive, dict):
            Drive = {k: v.xreplace(repl_map) for k, v in Drive.items()}
        else:
            Drive = Drive.xreplace(repl_map)
        Col_Ops = [L.xreplace(repl_map) for L in Col_Ops]
        mean_operator = mean_operator.xreplace(repl_map)
        if rho0_exprs is not None:
            rho0_exprs = [expr.xreplace(repl_map) for expr in rho0_exprs]

    defaults = {s: v for s, v in {**const_values, **runtime_consts}.items() if s in base_runtime_set}
    return {"N": N, "state_size": state_size, "H": H, "Drive": Drive, "Col_Ops": Col_Ops,
            "mean_operator": mean_operator, "rho0_exprs": rho0_exprs,
            "base_runtime_syms": tuple(base_runtime_syms),
            "derived_const_exprs": tuple(derived_const_exprs),
            "runtime_const_syms": tuple(base_runtime_syms) + tuple(derived_const_syms),
            "const_values_eff": const_values_eff, "defaults": defaults}


def _compile_rhs(H, Drive, Col_Ops, mean_operator, *, sweep_symbols, const_values=None,
                 runtime_consts=None, keep_symbolic_consts=None, auto_runtime_consts=False,
                 rho0=None, ode_size=None, pre_expand=True, collect_rho=False, factor_terms=False,
                 cse_batch_size=None, cse_simplify=False, hoist_rho_independent=True, reuse=True,
                 symbolic_workers="auto", max_symbolic_workers=None):
    """Prepare and optimize the ODE RHS; returns ``(CompiledRHS, "hit" or "miss")``."""
    start = time.time()
    model = _prepare_rhs_model(H, Drive, Col_Ops, mean_operator, sweep_symbols=sweep_symbols,
                               const_values=const_values, runtime_consts=runtime_consts,
                               keep_symbolic_consts=keep_symbolic_consts,
                               auto_runtime_consts=auto_runtime_consts, rho0=rho0,
                               ode_size=ode_size)
    Drive = model["Drive"]
    drive_key = (tuple((str(k), sp.srepr(v)) for k, v in Drive.items()) if isinstance(Drive, dict)
                 else sp.srepr(Drive))
    # Independent of the device, the selected time integrator and the call's numeric arrays.
    cache_key = (
        "compiled_rhs_v1", model["N"], ode_size, sp.srepr(model["H"]), drive_key,
        tuple(sp.srepr(L) for L in model["Col_Ops"]), sp.srepr(model["mean_operator"]),
        tuple((str(k), float(v)) for k, v in
              sorted(model["const_values_eff"].items(), key=lambda kv: str(kv[0]))),
        tuple(str(s) for s in model["runtime_const_syms"]),
        (tuple(sp.srepr(expr) for expr in model["rho0_exprs"])
         if model["rho0_exprs"] is not None else None),
        pre_expand, collect_rho, factor_terms, cse_batch_size, cse_simplify, hoist_rho_independent,
    )
    cached = _RHS_CODE_CACHE.get(cache_key) if reuse else None
    if cached is not None:
        return cached, "hit"

    state_size = model["state_size"]
    if ode_size is None:
        equations, mean_re, mean_im, meta = build_reduced_lindblad_rhs(
            model["N"], model["H"], model["Col_Ops"], model["mean_operator"],
            pre_expand=pre_expand, collect_rho=collect_rho, factor_terms=factor_terms,
            symbolic_workers=symbolic_workers, max_symbolic_workers=max_symbolic_workers)
        states = meta["rho_syms"]
    else:
        states = [sp.Symbol(f"rho[{i}]", real=True) for i in range(state_size)]
        equations = list(model["H"])
        if pre_expand:
            equations = [sp.expand(e) for e in equations]
        if collect_rho:
            equations = [sp.collect(e, states) for e in equations]
        if factor_terms:
            equations = [sp.factor_terms(e) for e in equations]
        mean_re, mean_im = sp.re(model["mean_operator"][0]), sp.im(model["mean_operator"][0])
    prepared = time.time()
    plan = _optimize_ode_rhs(equations, mean_re, mean_im, states, sp.Symbol("Drive", real=True),
                             Drive, runtime_const_syms=model["runtime_const_syms"],
                             cse_batch_size=cse_batch_size, cse_simplify=cse_simplify,
                             hoist_rho_independent=hoist_rho_independent,
                             extract_independent=ode_size is not None)
    compiled = CompiledRHS(
        kind="lindblad" if ode_size is None else "ode", state_size=state_size,
        sweep_symbols=tuple(sweep_symbols), state_symbols=tuple(states),
        equations=tuple(equations), mean_re=mean_re, mean_im=mean_im, drive_map=Drive,
        rho0_exprs=None if model["rho0_exprs"] is None else tuple(model["rho0_exprs"]),
        runtime_const_syms=model["runtime_const_syms"],
        base_runtime_syms=model["base_runtime_syms"],
        derived_const_exprs=model["derived_const_exprs"], const_values=model["defaults"],
        plan=plan, cache_key=cache_key,
        timings={"ode_prepare_s": prepared - start, "ode_optimize_s": time.time() - prepared})
    if reuse:
        _RHS_CODE_CACHE[cache_key] = compiled
    return compiled, "miss"


def compile_lindblad_rhs(H, Drive, Col_Ops, mean_operator, sweep_symbols, *, const_values=None,
                         runtime_consts=None, keep_symbolic_consts=None, auto_runtime_consts=False,
                         rho0=None, pre_expand=True, collect_rho=False, factor_terms=False,
                         cse_batch_size=None, cse_simplify=False, hoist_rho_independent=True,
                         RHSreuse=True, symbolic_workers="auto", max_symbolic_workers=None):
    """Prepare and optimize a Lindblad RHS without generating device code or using the GPU.

    Takes the model arguments of ``mesolve_2D`` and returns a ``CompiledRHS``.
    ``mesolve_2D(compiled, tlist=..., var_arrays=...)`` then starts from code emission.

    Args:
        sweep_symbols: The one or two swept symbols, in the order of the later ``var_arrays``
            (followed by any ``rho0_var_arrays`` symbols). A ``var_arrays`` dict also works.
        const_values: Constants folded into the RHS; values of constants kept symbolic
            (``keep_symbolic_consts``, ``auto_runtime_consts``) become call defaults.
        runtime_consts: Constants kept symbolic; values are defaults that later calls'
            ``runtime_consts`` override.
        symbolic_workers / max_symbolic_workers: Processes for the symbolic ODE; ``"auto"``
            uses up to ``max_symbolic_workers`` (default: half the logical CPUs) from Hilbert
            dimension 10 (see ``build_reduced_lindblad_rhs``).
        Other arguments: as in ``mesolve_2D``.

    The result's ``timings`` holds ``ode_prepare_s`` (symbolic ODE) and ``ode_optimize_s`` (CSE and
    hoisting); a cached model gives the stored object.
    """
    sweep_symbols = tuple(sweep_symbols)
    if not 1 <= len(sweep_symbols) <= 2:
        raise ValueError("sweep_symbols must contain one or two symbols.")
    compiled, _ = _compile_rhs(
        H, Drive, Col_Ops, mean_operator, sweep_symbols=sweep_symbols, const_values=const_values,
        runtime_consts=runtime_consts, keep_symbolic_consts=keep_symbolic_consts,
        auto_runtime_consts=auto_runtime_consts, rho0=rho0, pre_expand=pre_expand,
        collect_rho=collect_rho, factor_terms=factor_terms, cse_batch_size=cse_batch_size,
        cse_simplify=cse_simplify, hoist_rho_independent=hoist_rho_independent, reuse=RHSreuse,
        symbolic_workers=symbolic_workers, max_symbolic_workers=max_symbolic_workers)
    return compiled


def _collect_used_const_indices(*parts):
    """Return Const_arr indices referenced by generated code lines or strings."""
    pat = re.compile(r"Const_arr\[(\d+)\]")
    used = set()
    for part in parts:
        for s in ([part] if isinstance(part, str) else part):
            if s:
                used.update(int(m.group(1)) for m in pat.finditer(s))
    return sorted(used)


def _remap_const_indices(s, idx_map):
    """Rewrite Const_arr indices after unused constants have been removed."""
    if not s:
        return s
    return re.sub(r"Const_arr\[(\d+)\]",
                  lambda m: f"Const_arr[{idx_map.get(int(m.group(1)), int(m.group(1)))}]", s)


def _emit_cuda_rhs(compiled, uses_rho0_values, reuse=True):
    """CUDA code fragments of a ``CompiledRHS``; returns ``(fragments, emit_s, emitted)``."""
    key = ("cuda", bool(uses_rho0_values))
    if reuse and key in compiled.emitted:
        return compiled.emitted[key], 0.0, False
    start = time.time()
    (static_lines, drive_lines, _alias_lines, drho_lines, mean_line, final_line,
     static_syms, drive_syms, hoisted_syms) = _emit_cuda_ode_code(compiled.plan)
    state_size, rho0_exprs = compiled.state_size, compiled.rho0_exprs
    if uses_rho0_values:
        init_lines = [f"rho[{i}] = Rho0_arr[(size_t)result_idx * N + {i}];"
                      for i in range(state_size)]
    elif rho0_exprs is None:
        init_lines = ["for (int i=0;i<N;++i) rho[i] = 0.0f;", "rho[0] = 1.0f;"]
    else:
        init_lines = [f"rho[{i}] = {my_ccode(expr)};" for i, expr in enumerate(rho0_exprs)]
    used_const_indices = []
    if compiled.runtime_const_syms:
        for idx, sym in enumerate(compiled.runtime_const_syms):
            pat = _symbol_pattern(str(sym))
            init_lines = [pat.sub(f"Const_arr[{idx}]", line) for line in init_lines]
        init_lines = tidy_c_lines(init_lines)
        used_const_indices = _collect_used_const_indices(init_lines, static_lines, drive_lines,
                                                         drho_lines, mean_line, final_line)
        if used_const_indices:
            full_idx = list(range(len(compiled.runtime_const_syms)))
            if used_const_indices != full_idx:
                idx_map = {old: new for new, old in enumerate(used_const_indices)}
                init_lines = [_remap_const_indices(line, idx_map) for line in init_lines]
                static_lines = [_remap_const_indices(line, idx_map) for line in static_lines]
                drive_lines = [_remap_const_indices(line, idx_map) for line in drive_lines]
                drho_lines = [_remap_const_indices(line, idx_map) for line in drho_lines]
                mean_line = _remap_const_indices(mean_line, idx_map)
                final_line = _remap_const_indices(final_line, idx_map)
    else:
        init_lines = tidy_c_lines(init_lines)
    fragments = {"static_lines": tuple(static_lines), "drive_lines": tuple(drive_lines),
                 "drho_lines": tuple(drho_lines), "mean_line": mean_line,
                 "final_line": final_line, "static_syms": tuple(static_syms),
                 "drive_syms": tuple(drive_syms), "hoisted_syms": tuple(hoisted_syms),
                 "init_lines": tuple(init_lines), "used_const_indices": tuple(used_const_indices)}
    if reuse:
        compiled.emitted[key] = fragments
    return fragments, time.time() - start, True


def _runtime_const_values(compiled, runtime_consts, const_values, use_defaults):
    """Values of the runtime constants for this call, in Const_arr order."""
    if not compiled.runtime_const_syms:
        return []
    values = dict(compiled.const_values) if use_defaults else {}
    values.update(const_values or {})
    values.update(runtime_consts or {})
    base_vals = {}
    for s in compiled.base_runtime_syms:
        if s not in values:
            raise ValueError(f"Missing value for runtime constant symbol: {s}")
        base_vals[s] = float(values[s])
    vals = [base_vals[s] for s in compiled.base_runtime_syms]
    vals += [float(expr.evalf(subs=base_vals)) for expr in compiled.derived_const_exprs]
    return vals


def mesolve_2D(H, Drive=None, Col_Ops=None, mean_operator=None, tlist=None,
               var_arrays=None, const_values=None, kernel_template_file=None, *,
               RHSreuse=True, runtime_consts=None, keep_symbolic_consts=None,
               auto_runtime_consts=False, output_mode="mean", solver="rk4",
               solver_frequency=None, adaptive_options=None, unroll=False, fp64=False,
               pre_expand=True, collect_rho=False, factor_terms=False,
               cse_batch_size=None, cse_simplify=False, hoist_rho_independent=True,
               symbolic_workers="auto", max_symbolic_workers=None,
               nvrtc_options=(), timings=False, return_timing_info=False, warmup_time=0.0,
               rho0=None, rho0_var_arrays=None, rho0_values=None,
               return_time_trace=False, time_trace_every=None,
               time_trace_samples_per_period=None, solver_samples_per_period=None,
               Actual_Kernel_Save=False, beep_on_error=False, ignore_non_finite_output=False,
               t_in=0.0, return_device=False, calibrate_timesteps=None, consts=None, initial=None, output=None,
               solver_options=None, codegen=None, diagnostics=None):
    args = dict(locals())
    groups = {name: args.pop(name) for name in OPTION_GROUPS}
    args, _ = apply_option_groups(mesolve_2D, args, groups)
    if args["output_mode"] not in {"mean", "final", "final_rho"}:
        raise ValueError("output_mode must be 'mean', 'final', or 'final_rho'.")
    if isinstance(H, CompiledRHS):
        if H.kind != "lindblad":
            raise ValueError("This CompiledRHS comes from compile_ode_rhs; pass it to odesolve_2D.")
        args.update(H=None, compiled_rhs=H)
    return _solve_symbolic_system(**args)


def _solve_symbolic_system(H, Drive, Col_Ops, mean_operator, tlist,
               var_arrays=None, const_values=None, kernel_template_file=None, *,
               RHSreuse=True, runtime_consts=None, keep_symbolic_consts=None,
               auto_runtime_consts=False, output_mode="mean", solver="rk4",
               solver_frequency=None, adaptive_options=None, unroll=False,
               fp64=False,
               pre_expand=True, collect_rho=False, factor_terms=False,
               cse_batch_size=None, cse_simplify=False, hoist_rho_independent=True,
               symbolic_workers="auto", max_symbolic_workers=None,
               nvrtc_options=(), timings=False, return_timing_info=False, warmup_time=0.0,
               rho0=None, rho0_var_arrays=None, rho0_values=None,
               return_time_trace=False, time_trace_every=None,
               time_trace_samples_per_period=None, solver_samples_per_period=None,
               Actual_Kernel_Save=False, beep_on_error=False, ignore_non_finite_output=False,
               t_in=0.0, return_device=False, calibrate_timesteps=None, compiled_rhs=None,
               _ode_size=None, _state_trace=False):
    """
    Solve a Lindblad master equation over a two-dimensional parameter sweep on GPU.

    Args:
        H: ``N x N`` symbolic Hamiltonian matrix, where ``N`` is the number of
            simulated basis states or quantum levels; or a ``CompiledRHS`` from
            ``compile_lindblad_rhs``, which replaces the model arguments (``Drive``,
            ``Col_Ops``, ``mean_operator``, constants, ``rho0`` and the symbolic
            options) and starts the call from CUDA code generation; give ``tlist``
            by keyword.
        Drive: Symbolic drive expression or dict[Symbol -> expression].
        Col_Ops: List of symbolic collapse operators.
        mean_operator: Symbolic operator whose expectation value is returned.
        t_in: Absolute initial time; drive stages use t_in plus elapsed stage time.
            Zero and nonzero values share the same symbolic RHS and compiled kernel.
        return_device: Return CuPy output arrays instead of NumPy arrays. Device
            rho0_values can be reused directly; finite checks synchronize a scalar.
        tlist: Uniform, strictly increasing elapsed time samples beginning at zero. A
            list of ``M`` samples defines ``M - 1`` fixed solver intervals.
        var_arrays: Dict containing one or two sweep arrays, mapped to ParX/ParY
            in insertion order. A one-element dummy axis may be used when no
            physical parameter is swept.
        const_values: Constants normally substituted as fixed numbers during
            ordinary differential equation (ODE) and CUDA-code generation.
            Changing one on a later call requires new generation and compilation
            unless that symbol is kept symbolic.
        kernel_template_file: Explicit CUDA template path. ``None`` uses the
            canonical template packaged with GQIS.
        RHSreuse: Reuse a process-local generated ODE and compiled kernel for
            equivalent symbolic models. This supports animations and repeated
            calls in which sweep arrays, explicit initial-state values, or
            selected runtime constants change.
        runtime_consts: Values for constants intentionally kept symbolic in the
            generated ODE. They are uploaded on every call and override matching
            entries in const_values, allowing animation parameters to change
            without regenerating the ODE.
        keep_symbolic_consts: Select const_values symbols that remain symbolic
            so their values can change between repeated calls; "all", "auto",
            and "const_values" select every const_values key.
        auto_runtime_consts: If True, keep all const_values symbolic at runtime.
        output_mode: "mean", "final", or "final_rho".
        solver: CUDA integrator. Supported canonical names are "rk4", "lserk4",
            "lserk4f", "dp5", "ab5", "anas5", "tsit5", "alshina6", "dop853" and
            the adaptive "tsit5a". RK4 remains the default. General one-step methods are
            classical RK4, low-storage Carpenter-Kennedy LSRK4 (LSRK4F accumulates
            the RHS directly into its residual, storing one vector fewer), DP5, Tsit5,
            seven-stage sixth-order Alshina6, and 12-stage eighth-order DOP853.
            Anas5(w) is frequency-fitted for a known dominant frequency; AB5 is
            a low-RHS-cost multistep method with a smaller stability region.
            These use the supplied uniform time step. ``tsit5a`` adapts its step
            between the ``tlist`` output times (see ``adaptive_options``).
        solver_frequency: Non-negative angular-frequency estimate used by the
            fixed-step Anas5(w) method. Its coefficient is computed from
            ``solver_frequency * dt`` before launch and remains constant during
            the run. This does not enable adaptive timestepping. It is ignored
            by other solvers.
        adaptive_options: Options of the adaptive ``solver="tsit5a"`` (ignored otherwise; an
            error for other solvers). Keys and defaults in
            ``gqis.cuda_solvers.TSIT5A_DEFAULTS``: ``step_rtol``/``step_atol`` (1e-5, per-step
            local error tolerances, much stricter than the accuracy of the final result),
            ``max_substeps`` (64 accepted + rejected steps per output interval; exceeding it gives
            a non-finite result), ``fp64_time`` (True: time kept in FP64 and converted to FP32
            only for the drive) and ``count_steps`` (False; True adds per-system accepted and
            rejected step counts to the timing info as ``adaptive_steps``). With tsit5a,
            ``tlist`` sets the output points: steps adapt inside each interval and end on them,
            and ``mean`` outputs are averaged over the adaptive steps weighted by their length.
        unroll: If True, add ``#pragma unroll`` to solver vector loops. False
            leaves unrolling to NVRTC, which is the default for safer register
            use as the Hilbert-space dimension grows.
        fp64: If True, generate and run a 64-bit floating-point (FP64) kernel;
            otherwise use 32-bit floating point (FP32).
        pre_expand / collect_rho / factor_terms / cse_batch_size / cse_simplify /
            hoist_rho_independent: SymPy codegen controls.
        symbolic_workers: Processes for the symbolic Lindblad equations. ``"auto"`` uses up to
            ``max_symbolic_workers`` from Hilbert dimension 10 and one process below it
            (starting processes costs about 1-2 s); an integer sets the number. The generated
            equations are the same. ``max_symbolic_workers``: ``None`` means half the logical CPUs.
        nvrtc_options: Additional NVIDIA Runtime Compilation (NVRTC) options.
        timings: If True, print a concise timing breakdown for right-hand-side
            (RHS) generation, GPU-kernel execution, and the total call.
        return_timing_info: If True, collect and return a timing dict together
            with the result, without requiring printed timing output.
        warmup_time: Initial fraction [0, 1] of time excluded from the time
            average. This suppresses transient dependence on the initial state;
            it does not shorten the simulated evolution.
        rho0: Optional ``N x N`` symbolic initial density matrix or reduced rho
            vector with ``N*N - 1`` real components.
            It may contain symbols swept through ``var_arrays`` or
            ``rho0_var_arrays``. This enables initial-condition sweeps without
            changing the Hamiltonian. For matrix input, the last population and
            lower triangle are reconstructed from unit trace and Hermiticity.
        rho0_var_arrays: Optional dict of symbols to 1D arrays used in symbolic
            rho0. These arrays select one initial state for each independent
            simulation, are merged with var_arrays, and share the same two total
            sweep axes. They do not represent time samples.
        rho0_values: Alternative numeric array of explicit reduced initial
            states at t=0, with one state per independent simulation.
            Shape can be ``(num_X, N*N-1)`` for one sweep axis or
            ``(num_X, num_Y, N*N-1)`` for two axes. This is useful for arbitrary
            initial-state lists such as Fibonacci-sphere samples. The final
            dimension contains density-matrix components, not time samples.
        return_time_trace: If True, also return sampled time dependence of
            mean_operator after selected solver steps, beginning at ``t=dt``.
        time_trace_every: Record one trace point every k solver steps. For
            stride k, times are dt, (k+1)*dt, (2k+1)*dt, and so on.
        time_trace_samples_per_period: Desired trace samples per driving period. Requires
            solver_samples_per_period and maps to time_trace_every internally.
        solver_samples_per_period: Number of integration steps per driving period.
        Actual_Kernel_Save: If truthy, save the fully generated CUDA kernel to
            "<calling_python_file>_Kernel.cu". If a non-empty string is given,
            use it as the output path.
        beep_on_error: If True, play a short beep before raising non-finite output errors.
        ignore_non_finite_output: If True, do not raise on not-a-number (NaN) or
            infinite (Inf) output; return it as-is.
        calibrate_timesteps: None, True or a dict. Before the full sweep, double the
            step count of tlist on an undersampled grid until the RMS and maximum
            change of the result meet the tolerances, then narrow the bracket between
            the last failing and the first passing count by bisection_steps (default 2)
            geometric bisections, and run with that uniform step. tlist sets the
            duration and the starting grid; with bisection_steps=0 it is also the
            coarsest grid. The accepted step
            is cached per model and reused on later calls. Settings and defaults
            (rms_atol=1e-3, max_atol=1e-2): ``gqis.solver.TIMESTEP_CALIBRATION_DEFAULTS``.
        consts / initial / output / solver_options / codegen / diagnostics: Optional
            dicts grouping the options above under their usual names, for
            example ``codegen=dict(cse_batch_size=8)``. Give each option either
            directly or in its group, not both. Members are listed in
            ``gqis._options.OPTION_GROUPS``.

    Returns:
        NumPy array with shape:
            - (num_X, num_Y) complex for "mean" / "final"
            - (num_X, num_Y, N*N-1) real for "final_rho"
        If return_time_trace=True, returns (result, trace, trace_t), where trace has
        shape (num_X, num_Y, num_trace) complex.
        If return_timing_info=True, append timing_info to the trace tuple, or
        return (result, timing_info) when no trace is requested.

    Notes:
        Initial-state shape and finite output values are validated, but physical
        positivity of a user-supplied initial density matrix is the caller's
        responsibility. The kernel cache exists only for the current process.
    """
    if calibrate_timesteps is not None and calibrate_timesteps is not False:
        if normalize_solver_name(solver) == "tsit5a":
            raise ValueError("calibrate_timesteps chooses a fixed step; tsit5a adapts its own step "
                             "(set adaptive_options step_rtol/step_atol instead).")
        return _solve_with_timestep_calibration(dict(locals()))
    valid_output_modes = {"mean", "final", "final_rho", "mean_state"}
    if output_mode not in valid_output_modes:
        raise ValueError(f"Unsupported output_mode '{output_mode}'. "
                         f"Use one of: {sorted(valid_output_modes)}")
    if not isinstance(unroll, (bool, np.bool_)):
        raise ValueError("unroll must be True or False.")
    unroll = bool(unroll)
    kernel_template_file = _resolve_kernel_template_file(kernel_template_file)

    t_in = float(t_in)
    if not np.isfinite(t_in):
        raise ValueError("t_in must be finite.")
    scalar_type = "double" if fp64 else "float"
    complex_type = "double2" if fp64 else "float2"
    scalar_dtype = cp.float64 if fp64 else cp.float32
    complex_dtype = cp.complex128 if fp64 else cp.complex64

    var_arrays = {} if var_arrays is None else dict(var_arrays)
    rho0_var_arrays = {} if rho0_var_arrays is None else dict(rho0_var_arrays)
    for sym, arr in rho0_var_arrays.items():
        if sym in var_arrays:
            raise ValueError(f"Symbol {sym} is present in both var_arrays and rho0_var_arrays.")
        var_arrays[sym] = arr
    const_values = {} if const_values is None else const_values

    if rho0 is not None and rho0_values is not None:
        raise ValueError("Use either rho0 or rho0_values, not both.")
    uses_rho0_values = rho0_values is not None
    if len(var_arrays) == 0:
        raise ValueError("var_arrays or rho0_var_arrays must contain at least one sweep array "
                         "(ParX).")
    if len(var_arrays) > 2:
        raise ValueError("Only 2 total sweep arrays are supported across var_arrays and "
                         "rho0_var_arrays.")
    if compiled_rhs is None:
        if H is None or Drive is None or Col_Ops is None or mean_operator is None:
            raise TypeError("Give H, Drive, Col_Ops and mean_operator, or a CompiledRHS.")
    else:
        if any(item is not None for item in (H, Drive, Col_Ops, mean_operator)):
            raise ValueError("With a CompiledRHS, do not pass the model (H, Drive, Col_Ops, "
                             "mean_operator) again.")
        if const_values or keep_symbolic_consts is not None or auto_runtime_consts or rho0 is not None:
            raise ValueError("const_values, keep_symbolic_consts, auto_runtime_consts and rho0 are "
                             "fixed when the RHS is compiled; give values of constants kept "
                             "symbolic through runtime_consts.")
        if tuple(var_arrays) != compiled_rhs.sweep_symbols:
            raise ValueError(f"The sweep arrays must be for {compiled_rhs.sweep_symbols}, in the "
                             "order given when the RHS was compiled.")

    if tlist is None:
        raise ValueError("tlist is required for fixed-step integration.")
    num_t_host = int(len(tlist))
    if num_t_host < 2:
        raise ValueError("tlist must contain at least two time samples.")
    tlist_input = np.asarray(tlist)
    tlist_host = np.asarray(tlist_input, dtype=np.float64)
    if tlist_host.ndim != 1 or not np.all(np.isfinite(tlist_host)):
        raise ValueError("tlist must be a finite one-dimensional array.")
    time_diffs = np.diff(tlist_host)
    if np.any(time_diffs <= 0.0):
        raise ValueError("tlist must be strictly increasing.")
    input_float_dtype = (tlist_input.dtype
                         if np.issubdtype(tlist_input.dtype, np.floating) else np.dtype(np.float64))
    storage_tolerance = (8.0 * np.finfo(input_float_dtype).eps *
                         max(1.0, float(np.max(np.abs(tlist_host)))))
    uniform_dt_host = float((tlist_host[-1] - tlist_host[0]) / (num_t_host - 1))
    solver_spec = get_solver_spec(solver, frequency=solver_frequency,
                                  step_size=uniform_dt_host, adaptive_options=adaptive_options,
                                  fp64=fp64)
    if not np.isclose(tlist_host[0], 0.0, rtol=0.0, atol=storage_tolerance):
        raise ValueError("tlist must begin at zero; the CUDA kernel derives stage times from the "
                         "step index.")
    if not np.allclose(time_diffs, uniform_dt_host, rtol=1.0e-6, atol=storage_tolerance):
        raise ValueError("tlist must be uniformly spaced because the GPU solver uses one "
                         "fixed dt.")
    # A list of M sample times defines M - 1 integration intervals. Passing M
    # to the kernel would advance the state one step beyond tlist[-1].
    num_steps_host = num_t_host - 1
    warmup_time = float(warmup_time)
    if warmup_time < 0.0 or warmup_time > 1.0:
        raise ValueError("warmup_time must be within [0, 1].")
    warmup_steps_host = int(np.floor(warmup_time * num_steps_host))
    if warmup_time >= 1.0:
        warmup_steps_host = num_steps_host
    warmup_steps_host = max(0, min(num_steps_host, warmup_steps_host))
    return_time_trace = bool(return_time_trace)
    if time_trace_every is not None and time_trace_samples_per_period is not None:
        raise ValueError("Use either time_trace_every or time_trace_samples_per_period, not both.")
    if time_trace_samples_per_period is not None:
        if solver_samples_per_period is None:
            raise ValueError("solver_samples_per_period is required when "
                             "time_trace_samples_per_period is used.")
        solver_spp = int(solver_samples_per_period)
        trace_spp = int(time_trace_samples_per_period)
        if solver_spp <= 0 or trace_spp <= 0:
            raise ValueError("solver_samples_per_period and time_trace_samples_per_period must be "
                             "positive.")
        time_trace_stride_host = max(1, int(round(solver_spp / trace_spp)))
    elif time_trace_every is not None:
        time_trace_stride_host = int(time_trace_every)
        if time_trace_stride_host <= 0:
            raise ValueError("time_trace_every must be a positive integer.")
    else:
        time_trace_stride_host = 1
    num_time_trace_host = (int(np.ceil(num_steps_host /
                                       time_trace_stride_host)) if return_time_trace else 0)

    effective_output_mode = output_mode
    if output_mode == "mean" and warmup_steps_host >= num_steps_host:
        # No measurement window left: return the final expectation value instead.
        effective_output_mode = "final"
    if output_mode == "mean_state" and warmup_steps_host >= num_steps_host:
        effective_output_mode = "final_rho"

    collect_timings = bool(timings or return_timing_info)
    total_start = time.time() if collect_timings else 0.0
    rhs_stage_start = time.time() if collect_timings else 0.0
    saved_kernel_path = None
    kernel_cache_status = "miss"

    # 1) ODE preparation and optimization (device-independent), unless a CompiledRHS is given.
    if compiled_rhs is None:
        compiled, rhs_cache_status = _compile_rhs(
            H, Drive, Col_Ops, mean_operator, sweep_symbols=tuple(var_arrays),
            const_values=const_values, runtime_consts=runtime_consts,
            keep_symbolic_consts=keep_symbolic_consts, auto_runtime_consts=auto_runtime_consts,
            rho0=rho0, ode_size=_ode_size, pre_expand=pre_expand, collect_rho=collect_rho,
            factor_terms=factor_terms, cse_batch_size=cse_batch_size, cse_simplify=cse_simplify,
            hoist_rho_independent=hoist_rho_independent, reuse=RHSreuse,
            symbolic_workers=symbolic_workers, max_symbolic_workers=max_symbolic_workers)
    else:
        compiled, rhs_cache_status = compiled_rhs, "provided"
    stage_times = (dict(compiled.timings) if rhs_cache_status == "miss"
                   else {"ode_prepare_s": 0.0, "ode_optimize_s": 0.0})
    state_size = compiled.state_size
    runtime_const_syms = compiled.runtime_const_syms
    runtime_const_vals = _runtime_const_values(compiled, runtime_consts, const_values,
                                               use_defaults=compiled_rhs is not None)

    # 2) CUDA code emission.
    fragments, stage_times["emit_s"], _ = _emit_cuda_rhs(compiled, uses_rho0_values,
                                                         reuse=RHSreuse)
    rhs_codegen_time = stage_times["ode_prepare_s"] + stage_times["ode_optimize_s"] + stage_times["emit_s"]

    # 3) Kernel assembly and compilation additionally depend on solver/output/template options.
    template_mtime = (os.path.getmtime(kernel_template_file)
                      if os.path.exists(kernel_template_file) else None)
    cache_key = (
        "kernel_v9", compiled.cache_key, bool(uses_rho0_values), effective_output_mode,
        warmup_steps_host, solver_spec.name, bool(unroll), bool(fp64),
        solver_spec.source if solver_spec.name == "tsit5a" else "",  # adaptive_options variant
        return_time_trace, _state_trace, kernel_template_file, template_mtime,
        tuple(nvrtc_options) if not isinstance(nvrtc_options, str) else (nvrtc_options,),
    )

    static_lines = list(fragments["static_lines"])
    drive_lines = list(fragments["drive_lines"])
    drho_lines = list(fragments["drho_lines"])
    mean_line, final_line = fragments["mean_line"], fragments["final_line"]
    static_syms = tuple(fragments["static_syms"])
    drive_syms = tuple(fragments["drive_syms"])
    hoisted_syms = tuple(fragments["hoisted_syms"])
    init_lines = list(fragments["init_lines"])
    used_const_indices = list(fragments["used_const_indices"])
    kernel_build_start = time.time()

    cached = _KERNEL_CACHE.get(cache_key) if RHSreuse else None
    if cached is None:
        # Assemble the selected solver around the shared RHS and compile it.
        with open(kernel_template_file) as f:
            template_text = f.read()
            src = Template(template_text)
        has_const_arg_marker = "#CONST_ARG_DECL#" in template_text
        if "#SOLVER_CODE#" not in template_text:
            raise ValueError("Kernel template is missing #SOLVER_CODE#. Use the current "
                             "N_Level_Kernel.cu template.")
        if runtime_const_syms and not has_const_arg_marker:
            raise ValueError("Kernel template is missing #CONST_ARG_DECL# but runtime constants "
                             "are enabled. Use N_Level_Kernel.cu template.")
        static_code = ("\n    ".join(static_lines)
                       if static_lines else "/* no thread-static terms */")
        drho_code = "\n    ".join(drho_lines)
        drives_code = "\n    ".join(drive_lines) if drive_lines else "/* no drives */"
        num_statics = max(1, len(static_syms))
        num_drives = max(1, len(drive_syms) + len(hoisted_syms))  # Drive_arr[0] exists if used
        kernel_code = src.substitute(N_DECL=str(state_size), NUM_STATICS_DECL=str(num_statics),
                                     NUM_DRIVES_DECL=str(num_drives))
        const_arg_decl = (f"    const {scalar_type}* __restrict__ Const_arr,\n"
                          if has_const_arg_marker else "")
        kernel_code = kernel_code.replace("#CONST_ARG_DECL#", const_arg_decl)
        rho0_arg_decl = (f"    const {scalar_type}* __restrict__ Rho0_arr,\n"
                         if uses_rho0_values else "")
        kernel_code = kernel_code.replace("#RHO0_ARG_DECL#", rho0_arg_decl)
        if solver_spec.parameters and "#SOLVER_ARG_DECL#" not in kernel_code:
            raise ValueError("The selected fitted solver requires a kernel template containing "
                             "#SOLVER_ARG_DECL#.")
        solver_arg_decl = "".join(
            f"    const {scalar_type} solver_param{index},\n"
            for index in range(len(solver_spec.parameters)))
        if "adaptive_steps[" in solver_spec.source:  # tsit5a count_steps
            solver_arg_decl += "    int2* __restrict__ adaptive_steps,\n"
        kernel_code = kernel_code.replace("#SOLVER_ARG_DECL#", solver_arg_decl)
        kernel_code = kernel_code.replace("#INIT_RHO#", "\n            ".join(init_lines))
        kernel_code = kernel_code.replace("#INSERT_STATICS#", static_code)
        kernel_code = kernel_code.replace("#INSERT_DRIVES#", drives_code)
        kernel_code = kernel_code.replace("#INSERT_DRHO#", drho_code)
        solver_code = solver_spec.source.replace(
            "#SOLVER_UNROLL#", "#pragma unroll" if unroll else "//#pragma unroll")
        has_time_input = "const float t_in" in kernel_code
        if has_time_input:
            solver_code = solver_code.replace("compute_drives(ParX, ParY, ",
                                               "compute_drives(ParX, ParY, t_in, ")
        kernel_code = kernel_code.replace("#SOLVER_CODE#", solver_code)
        if "compute_drho_lserk(" in solver_code:
            kernel_code = _insert_fused_lserk_rhs(kernel_code, drho_lines)
        mean_line_kernel = mean_line.replace("Drive_arr[", "Drive[")
        mean_line_kernel = mean_line_kernel.replace("Static_arr[", "Static[")
        final_line_kernel = final_line.replace("Drive_arr[", "Drive[")
        final_line_kernel = final_line_kernel.replace("Static_arr[", "Static[")
        if effective_output_mode == "final":
            mean_line_out = ""
            final_line_out = final_line_kernel
            results_line = (
                "results[result_idx].x = avg.x;\n            results[result_idx].y = avg.y;")
        elif effective_output_mode == "mean_state":
            kernel_code = kernel_code.replace("float rho[N];", "float rho[N]; float state_avg[N] = {0};")
            mean_line_out = "if (step >= warmup_steps) { for (int i=0; i<N; ++i) state_avg[i] += rho[i]; }"
            final_line_out = ""
            results_line = ("for (int i=0; i<N; ++i) results[(size_t)result_idx * N + i] = "
                            "state_avg[i] / (num_steps - warmup_steps);")
        elif effective_output_mode == "final_rho":
            mean_line_out = ""
            final_line_out = ""
            results_line = ("for (int i = 0; i < N; ++i) {\n"
                            "                results[(size_t)result_idx * N + i] = rho[i];\n"
                            "            }")
        else:
            if warmup_steps_host > 0:
                mean_line_out = f"if (step >= warmup_steps) {{ {mean_line_kernel} }}"
            else:
                mean_line_out = mean_line_kernel
            final_line_out = ""
            results_line = (f"const {scalar_type} inv_t = ({scalar_type})1.0f / "
                            f"({scalar_type})(num_steps - warmup_steps);\n"
                            "            results[result_idx].x = avg.x * inv_t;\n"
                            "            results[result_idx].y = avg.y * inv_t;")
        adaptive_mean_init = adaptive_mean_line = ""
        if solver_spec.name == "tsit5a" and effective_output_mode in {"mean", "mean_state"}:
            # Average inside the adaptive loop: each accepted step weighted by its length and
            # normalized to the averaging time, so no division by the step count afterwards.
            if effective_output_mode == "mean":
                # obs_prev: observable at the step start (end of the previous accepted step).
                adaptive_mean_init = "float2 obs_prev; " + final_line_kernel.replace("avg.", "obs_prev.")
                weighted = ("avg.x = fmaf(avg_weight, obs.x + obs_prev.x, avg.x);"
                            " avg.y = fmaf(avg_weight, obs.y + obs_prev.y, avg.y);")
                results_line = ("results[result_idx].x = adaptive_failed ? __int_as_float(0x7fc00000) : avg.x;\n"
                                "            results[result_idx].y = adaptive_failed"
                                " ? __int_as_float(0x7fc00000) : avg.y;")
            else:
                weighted = ("for (int i=0; i<N; ++i) "
                            "state_avg[i] = fmaf(avg_weight, rho[i] + y_prev[i], state_avg[i]);")
                results_line = ("for (int i=0; i<N; ++i) results[(size_t)result_idx * N + i] = "
                                "adaptive_failed ? __int_as_float(0x7fc00000) : state_avg[i];")
            adaptive_mean_line = f"if (step >= warmup_steps) {{ {weighted} }}"
            if effective_output_mode == "mean":
                adaptive_mean_line = ("float2 obs; " + final_line_kernel.replace("avg.", "obs.")
                                      + f" {adaptive_mean_line} obs_prev = obs;")
            mean_line_out = ""
        kernel_code = kernel_code.replace("#ADAPTIVE_MEAN_INIT#", adaptive_mean_init)
        kernel_code = kernel_code.replace("#ADAPTIVE_MEAN_LINE#", adaptive_mean_line)
        if return_time_trace:
            trace_obs_line = final_line_kernel.replace("avg.x",
                                                       "time_trace_results[trace_offset].x")
            trace_obs_line = trace_obs_line.replace("avg.y", "time_trace_results[trace_offset].y")
            trace_store = ("for (int i=0; i<N; ++i) time_trace_results[trace_offset * N + i] = rho[i];"
                           if _state_trace else trace_obs_line)
            trace_line_out = (
                "if ((step % time_trace_stride) == 0) {\n"
                "                    const int trace_idx = step / time_trace_stride;\n"
                "                    if (trace_idx < num_time_trace) {\n"
                "                        const size_t trace_offset = (size_t)result_idx * "
                "num_time_trace + trace_idx;\n"
                f"                        {trace_store}\n"
                "                    }\n"
                "                }")
            mean_line_out = (f"{mean_line_out}\n                {trace_line_out}"
                             if mean_line_out else trace_line_out)
        kernel_code = kernel_code.replace("#MEAN_LINE#", mean_line_out)
        kernel_code = kernel_code.replace("#FINAL_LINE#", final_line_out)
        kernel_code = kernel_code.replace("#RESULTS_LINE#", results_line)

        # Match output pointer type to selected output mode.
        if effective_output_mode in {"final_rho", "mean_state"}:
            result_arg_decl = f"    {scalar_type}* __restrict__ results"
            result_arg_comment = " // final reduced rho vector"
        else:
            result_arg_decl = f"    {complex_type}* __restrict__ results"
            result_arg_comment = " // averaged/final expectation value"
        if return_time_trace:
            trace_type = scalar_type if _state_trace else complex_type
            result_arg_decl += (f",\n    {trace_type}* __restrict__ time_trace_results,"
                                "\n    const int time_trace_stride,"
                                "\n    const int num_time_trace")
        else:
            result_arg_decl += result_arg_comment
        kernel_code_new = re.sub(r"^\s*float2\*\s*__restrict__\s*results[^\n]*$", result_arg_decl,
                                 kernel_code, flags=re.MULTILINE)
        if kernel_code_new == kernel_code:
            raise ValueError("Kernel template result argument line was not found.")
        kernel_code = kernel_code_new

        # Promote generated/templated code to FP64 only when requested so FP32 path is untouched.
        if fp64:
            kernel_code = kernel_code.replace("make_float2", "make_double2")
            kernel_code = kernel_code.replace("float2", "double2")
            kernel_code = kernel_code.replace("fmaf(", "fma(")
            kernel_code = kernel_code.replace("sqrtf(", "sqrt(")
            kernel_code = kernel_code.replace("sinf(", "sin(")
            kernel_code = kernel_code.replace("cosf(", "cos(")
            kernel_code = kernel_code.replace("expf(", "exp(")
            kernel_code = kernel_code.replace("logf(", "log(")
            kernel_code = re.sub(r"(?<![A-Za-z0-9_])"
                                 r"((?:\d+\.\d*|\d+|\.\d+)(?:[eE][+\-]?\d+)?)[fF]\b",
                                 r"\1", kernel_code)
            kernel_code = re.sub(r"\bfloat\b", "double", kernel_code)
        saved_kernel_path = _save_generated_kernel_file(Actual_Kernel_Save, kernel_code)
        # Compile kernel
        if isinstance(nvrtc_options, str):
            nvrtc_options = (nvrtc_options, )
        time_evolution_kernel = cp.RawKernel(kernel_code, "time_evolution_kernel",
                                             options=tuple(nvrtc_options))
        # Compile now (CuPy otherwise compiles on the first launch), so gpu_kernel_s is the
        # calculation only, also on the first call.
        time_evolution_kernel.compile()
        cached = {"kernel": time_evolution_kernel,
                  "uses_const_arr_arg": has_const_arg_marker,
                  "uses_rho0_arr_arg": uses_rho0_values,
                  "has_time_input": has_time_input,
                  "solver_parameter_count": len(solver_spec.parameters),
                  "uses_time_trace": return_time_trace,
                  "used_const_indices": tuple(used_const_indices or []),
                  "kernel_code": kernel_code}
        if RHSreuse:
            _KERNEL_CACHE[cache_key] = cached
    else:
        # Reuse the already compiled solver-specific kernel.
        kernel_cache_status = "hit"
        time_evolution_kernel = cached["kernel"]
        if Actual_Kernel_Save and cached.get("kernel_code"):
            saved_kernel_path = _save_generated_kernel_file(Actual_Kernel_Save,
                                                            cached["kernel_code"])
        cached_used = cached.get("used_const_indices", None)
        if cached_used is None:
            used_const_indices = list(range(len(runtime_const_vals)))
        else:
            used_const_indices = list(cached_used)

    stage_times["kernel_build_s"] = time.time() - kernel_build_start
    rhs_stage_time = (time.time() - rhs_stage_start) if collect_timings else 0.0

    # 3) Upload sweep axes and launch kernel.
    def to_device(x):
        """Convert a host array to a CuPy array with the selected scalar dtype."""
        return x.astype(scalar_dtype, copy=False) if isinstance(x, cp.ndarray) else cp.asarray(x, dtype=scalar_dtype)

    ParX_list = to_device(next(iter(var_arrays.values())))
    ParY_list = (to_device(list(var_arrays.values())[1]) if len(var_arrays) > 1 else
                 cp.zeros(1, dtype=scalar_dtype))
    num_X, num_Y = int(len(ParX_list)), int(len(ParY_list))
    rho0_arr = None
    if uses_rho0_values:
        rho0_np = cp.asarray(rho0_values, dtype=scalar_dtype)
        nred = state_size
        if rho0_np.shape == (num_X, nred) and num_Y == 1:
            rho0_np = rho0_np.reshape(num_X, 1, nred)
        if rho0_np.shape != (num_X, num_Y, nred):
            raise ValueError(f"rho0_values must have shape ({num_X}, {num_Y}, {nred}) "
                             f"or ({num_X}, {nred}) when num_Y=1; got {rho0_np.shape}.")
        rho0_arr = cp.asarray(rho0_np.reshape(num_X * num_Y, nred), dtype=scalar_dtype)
    # Allocate output
    if effective_output_mode in {"final_rho", "mean_state"}:
        results = cp.zeros((num_X, num_Y, state_size), dtype=scalar_dtype)
    else:
        results = cp.zeros((num_X, num_Y), dtype=complex_dtype)
    trace_shape = (num_X, num_Y, num_time_trace_host) + ((state_size,) if _state_trace else ())
    time_trace_results = (cp.zeros(trace_shape, dtype=scalar_dtype if _state_trace else complex_dtype)
                          if return_time_trace else None)

    # Grid/block
    block_dim = (16, 8)
    grid_dim = (int(np.ceil(num_Y / block_dim[0])), int(np.ceil(num_X / block_dim[1])))

    Cudt = scalar_dtype(uniform_dt_host)
    num_steps = num_steps_host
    warmup_steps = np.int32(warmup_steps_host)

    # Launch kernel
    runtime_const_vals_eff = runtime_const_vals
    if used_const_indices is not None:
        if used_const_indices:
            runtime_const_vals_eff = [runtime_const_vals[i] for i in used_const_indices]
        else:
            runtime_const_vals_eff = []
    if runtime_const_vals_eff:
        const_arr = cp.asarray(runtime_const_vals_eff, dtype=scalar_dtype)
    else:
        # If kernel expects Const_arr, use a tiny dummy buffer when runtime constants are unused.
        const_arr = cp.zeros(1, dtype=scalar_dtype)
    gpu_kernel_time = 0.0
    if collect_timings:
        ev_start = cp.cuda.Event()
        ev_end = cp.cuda.Event()
        ev_start.record()

    kernel_args = [Cudt, num_X, num_Y, num_steps, warmup_steps, ParX_list, ParY_list]
    if cached.get("has_time_input", False):
        kernel_args.insert(1, scalar_dtype(t_in))
    elif t_in != 0.0:
        raise ValueError("Nonzero t_in requires the current CUDA kernel template.")
    if cached.get("uses_const_arr_arg", False):
        kernel_args.append(const_arr)
    if cached.get("uses_rho0_arr_arg", False):
        kernel_args.append(rho0_arr)
    kernel_args.extend(scalar_dtype(value) for value in solver_spec.parameters)
    adaptive_steps = None
    if "adaptive_steps[" in solver_spec.source:  # tsit5a count_steps: (accepted, rejected)
        adaptive_steps = cp.zeros((int(num_X), int(num_Y), 2), dtype=cp.int32)
        kernel_args.append(adaptive_steps)
    kernel_args.append(results)
    if cached.get("uses_time_trace", False):
        kernel_args.extend([time_trace_results, np.int32(time_trace_stride_host),
                            np.int32(num_time_trace_host)])
    time_evolution_kernel(grid_dim, block_dim, tuple(kernel_args))

    if collect_timings:
        ev_end.record()
        ev_end.synchronize()
        gpu_kernel_time = cp.cuda.get_elapsed_time(ev_start, ev_end) / 1000.0

    results_np = results if return_device else cp.asnumpy(results)
    time_trace_np = ((time_trace_results if return_device else cp.asnumpy(time_trace_results))
                     if return_time_trace else None)
    output_xp = cp if return_device else np
    if not bool(output_xp.all(output_xp.isfinite(results_np))):
        if not ignore_non_finite_output:
            if beep_on_error:
                _play_notification_beep("error")
            raise RuntimeError("Non-finite values detected in mesolve_2D output. Try reducing dt "
                               "(increase time samples) or check model parameters/RHS.")
    if return_time_trace and not bool(output_xp.all(output_xp.isfinite(time_trace_np))):
        if not ignore_non_finite_output:
            if beep_on_error:
                _play_notification_beep("error")
            raise RuntimeError("Non-finite values detected in mesolve_2D time trace. Try reducing "
                               "dt (increase time samples) or check model parameters/RHS.")

    timing_info = None
    if collect_timings:
        total_time = time.time() - total_start
        timing_info = {"rhs_stage_s": rhs_stage_time, "gpu_kernel_s": gpu_kernel_time,
                       "rhs_codegen_s": rhs_codegen_time, "total_s": total_time,
                       "ode_prepare_s": stage_times["ode_prepare_s"],
                       "ode_optimize_s": stage_times["ode_optimize_s"],
                       "code_emit_s": stage_times["emit_s"],
                       "kernel_build_s": stage_times["kernel_build_s"],
                       "cached_rhs": rhs_cache_status, "cached_kernel": kernel_cache_status,
                       "solver": solver_spec.name, "solver_frequency": solver_frequency,
                       "unroll": unroll}
        if adaptive_steps is not None:
            counts = cp.asnumpy(adaptive_steps)
            timing_info["adaptive_steps"] = {"accepted": counts[..., 0], "rejected": counts[..., 1]}
    if timings:
        api_name = "mesolve_2D" if _ode_size is None else "odesolve_2D"
        print(f"{api_name} timings: rhs_stage={rhs_stage_time:.3f}s "
              f"(ode_prepare={stage_times['ode_prepare_s']:.3f}s "
              f"ode_optimize={stage_times['ode_optimize_s']:.3f}s "
              f"code_emit={stage_times['emit_s']:.3f}s "
              f"kernel_build={stage_times['kernel_build_s']:.3f}s) "
              f"rhs_codegen={rhs_codegen_time:.3f}s gpu_kernel={gpu_kernel_time:.3f}s "
              f"total={total_time:.3f}s cachedRHS={rhs_cache_status} "
              f"cachedKernel={kernel_cache_status}")
    if saved_kernel_path:
        print(f"Generated kernel saved to: {saved_kernel_path}")
    if return_time_trace:
        dt_host = uniform_dt_host
        trace_steps = np.arange(num_time_trace_host, dtype=np.float64 if fp64 else np.float32) * time_trace_stride_host
        trace_t = t_in + (trace_steps + 1.0) * dt_host
        output = (results_np, time_trace_np, trace_t)
        return (*output, timing_info) if return_timing_info else output
    if return_timing_info:
        return results_np, timing_info
    return results_np


mesolve_2D.__doc__ = _solve_symbolic_system.__doc__

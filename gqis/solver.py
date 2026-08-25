"""GPU Quantum Interferometry Solver (GQIS) symbolic-to-CUDA solver core.

This module converts symbolic Lindblad right-hand side (RHS) expressions into
NVIDIA CUDA code, compiles them with CuPy and NVIDIA Runtime Compilation
(NVRTC), and executes parameter sweeps on a graphics processing unit (GPU).
"""

import os
import re
import sys
import time
from string import Template

import cupy as cp
import numpy as np
import sympy as sp
from sympy.printing.c import C99CodePrinter

try:
    import winsound
except ImportError:
    winsound = None

# In-memory cache to avoid regenerating/recompiling identical RHS kernels.
_KERNEL_CACHE = {}


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


def my_ccode(expr):
    """Print a SymPy expression using CUDA-friendly floating literals.

    Args:
        expr: SymPy expression.

    Returns:
        C99/CUDA expression string.
    """
    return MyCPrinter().doprint(expr)


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


def build_reduced_lindblad_rhs(N, H, Col_Ops, mean_operator, *, pre_expand=False,
                                collect_rho=False, factor_terms=False):
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

    drho_eqs = [sp.simplify(sp.expand(expr)) if pre_expand else sp.simplify(expr)
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
    rho_sym_names = {str(s) for s in meta["rho_syms"]}
    param_sym_names = {"ParX", "ParY"}
    runtime_const_names = {str(s) for s in (runtime_const_syms or ())}

    hoisted_subs, drho_lines = cse_emit_c_lines(drho_eqs, meta["rho_syms"],
                                                batch_size=cse_batch_size, do_simplify=cse_simplify,
                                                hoist_rho_independent=hoist_rho_independent,
                                                return_hoist=True)
    # Drive expressions can be a single Expr (mapped to Drive_symbol) or a dict of Symbol->Expr.
    drive_map = {}
    if drive_expr is not None:
        if isinstance(drive_expr, dict):
            drive_map = drive_expr
        else:
            drive_map = {Drive_symbol: drive_expr}

    drive_lines, drive_alias_lines, drive_syms = emit_drive_code(drive_map, array_name="Drive_arr")

    # Hoist rho-independent CSE temporaries into either:
    # - a thread-static section (ParX/ParY/runtime-const only), computed once per thread
    # - the drive section (stage-time dependent), computed once per RK stage
    static_lines = []
    static_syms = []
    hoisted_syms = []
    # Replacement map: drives first, then static/drive hoisted temporaries as we emit them.
    name_to_repl = {str(s): f"Drive_arr[{i}]" for i, s in enumerate(drive_syms)}
    runtime_const_set = set(runtime_const_syms or [])
    const_local_syms = set(runtime_const_set)
    const_local_names = {str(s) for s in const_local_syms}
    static_sym_names = set()

    def _negate_repl(repl):
        """Negate a generated replacement string without adding redundant plus signs."""
        return repl[1:] if repl.startswith("-") else "-" + repl

    next_drive_idx = len(drive_syms)
    next_static_idx = 0
    if hoist_rho_independent and hoisted_subs:
        for s, expr in hoisted_subs:
            # Generate code for expr and replace Drive/previous-hoist symbol names.
            expr_code = my_ccode(expr)
            for name, repl in name_to_repl.items():
                pat = re.compile(r"(?<![0-9A-Za-z_])" + re.escape(name) + r"(?![0-9A-Za-z_])")
                expr_code = pat.sub(repl, expr_code)

            expr_sym_names = {str(sym) for sym in expr.free_symbols}
            # Const-only hoists become inline Const_arr substitutions, not Drive_arr entries.
            if (not expr_sym_names) or expr_sym_names.issubset(const_local_names):
                name_to_repl[str(s)] = f"({expr_code})"
                const_local_names.add(str(s))
                continue
            # If expr is just a negation of an existing symbol, avoid a new Drive_arr slot.
            coeff, rest = expr.as_coeff_Mul()
            if coeff == -1 and isinstance(rest, sp.Symbol):
                rest_name = str(rest)
                if rest_name in name_to_repl:
                    name_to_repl[str(s)] = _negate_repl(name_to_repl[rest_name])
                    continue
            if expr_sym_names.issubset(const_local_names | param_sym_names | static_sym_names):
                idx = next_static_idx
                static_lines = (static_lines or []) + [f"Static_arr[{idx}] = {expr_code};"]
                static_syms.append(s)
                static_sym_names.add(str(s))
                name_to_repl[str(s)] = f"Static_arr[{idx}]"
                next_static_idx += 1
            else:
                idx = next_drive_idx
                # Store directly into Drive_arr to avoid an extra temp register for this symbol.
                drive_lines = (drive_lines or []) + [f"Drive_arr[{idx}] = {expr_code};"]
                hoisted_syms.append(s)
                name_to_repl[str(s)] = f"Drive_arr[{idx}]"
                next_drive_idx += 1

    # Hoist static-only subexpressions from the result calculation too, so static basis
    # readout does not recompute sqrt/div terms every time step.
    mean_subs, mean_exprs = sp.cse([mean_re_expr, mean_im_expr],
                                   symbols=sp.numbered_symbols("mean_tmp"))
    next_static_idx = len(static_syms)
    for s, expr in mean_subs:
        expr_sym_names = {str(sym) for sym in expr.free_symbols}
        if expr_sym_names & rho_sym_names:
            mean_exprs = [sp.simplify(e.subs(s, expr)) for e in mean_exprs]
        elif expr_sym_names and expr_sym_names.issubset(const_local_names | param_sym_names
                                                        | static_sym_names):
            expr_code = my_ccode(expr)
            for name, repl in name_to_repl.items():
                pat = re.compile(r"(?<![0-9A-Za-z_])" + re.escape(name) + r"(?![0-9A-Za-z_])")
                expr_code = pat.sub(repl, expr_code)
            static_lines = (static_lines or []) + [f"Static_arr[{next_static_idx}] = {expr_code};"]
            static_syms.append(s)
            static_sym_names.add(str(s))
            name_to_repl[str(s)] = f"Static_arr[{next_static_idx}]"
            next_static_idx += 1
        elif expr_sym_names:
            expr_code = my_ccode(expr)
            for name, repl in name_to_repl.items():
                pat = re.compile(r"(?<![0-9A-Za-z_])" + re.escape(name) + r"(?![0-9A-Za-z_])")
                expr_code = pat.sub(repl, expr_code)
            drive_lines = (drive_lines or []) + [f"Drive_arr[{next_drive_idx}] = {expr_code};"]
            hoisted_syms.append(s)
            name_to_repl[str(s)] = f"Drive_arr[{next_drive_idx}]"
            next_drive_idx += 1
        else:
            mean_exprs = [sp.simplify(e.subs(s, expr)) for e in mean_exprs]

    mean_re = my_ccode(mean_exprs[0])
    mean_im = my_ccode(mean_exprs[1])
    mean_line = f"avg.x += {mean_re}; avg.y += {mean_im};"
    final_line = f"avg.x = {mean_re}; avg.y = {mean_im};"

    # Replace Drive/static symbols and hoisted temporaries in the RHS/mean with array accesses.
    for name, repl in name_to_repl.items():
        pat = re.compile(r"(?<![0-9A-Za-z_])" + re.escape(name) + r"(?![0-9A-Za-z_])")
        drho_lines = [pat.sub(repl, line) for line in drho_lines]
        static_lines = [pat.sub(repl, line) for line in static_lines]
        drive_lines = [pat.sub(repl, line) for line in drive_lines]
        mean_line = pat.sub(repl, mean_line)
        final_line = pat.sub(repl, final_line)

    # Replace runtime constants with Const_arr[idx].
    if runtime_const_syms:
        for idx, sym in enumerate(runtime_const_syms):
            name = str(sym)
            pat = re.compile(r"(?<![0-9A-Za-z_])" + re.escape(name) + r"(?![0-9A-Za-z_])")
            repl = f"Const_arr[{idx}]"
            drho_lines = [pat.sub(repl, line) for line in drho_lines]
            static_lines = [pat.sub(repl, line) for line in static_lines]
            drive_lines = [pat.sub(repl, line) for line in drive_lines]
            mean_line = pat.sub(repl, mean_line)
            final_line = pat.sub(repl, final_line)

    # compute_drho does not need these aliases anymore (they're merged into compute_drives).
    drive_alias_lines = []

    # Apply float/sinf/cosf/etc normalizations.
    static_lines = tidy_c_lines(static_lines) if static_lines else []
    drive_lines = tidy_c_lines(drive_lines) if drive_lines else []
    drho_lines = tidy_c_lines(drho_lines) if drho_lines else []
    return (static_lines, drive_lines, drive_alias_lines, drho_lines, mean_line, final_line,
            static_syms, drive_syms, hoisted_syms,
            )


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

    def _emit_block(exprs, base_i, batch_id):
        """Emit one batched CSE block for a slice of RHS expressions."""
        cse_subs, cse_exprs = sp.cse(exprs,
            # Unique prefix per batch avoids name collisions if caller enables hoisting.
            symbols=sp.numbered_symbols(prefix=f"t{batch_id}_"), ignore=set(rho_syms))
        hoist_subs, keep_subs = (_partition_subs(cse_subs) if hoist_rho_independent else
                                 ([], cse_subs))
        block_lines = []
        # Scope temporaries so the compiler can drop them sooner.
        block_lines.append("{")
        for s, expr in keep_subs:
            block_lines.append(f"float {s} = {my_ccode(_maybe_simplify(expr))};")
        for i, expr in enumerate(cse_exprs):
            block_lines.append(f"d_rho[{base_i + i}] = {my_ccode(_maybe_simplify(expr))};")
        block_lines.append("}")
        return hoist_subs, block_lines

    if batch_size is None:
        # Preserve historical behavior: one global CSE pass for all equations.
        cse_subs, cse_exprs = sp.cse(drho_exprs, symbols=sp.numbered_symbols(prefix="t"),
                                     ignore=set(rho_syms))
        hoist_subs, keep_subs = (_partition_subs(cse_subs) if hoist_rho_independent else
                                 ([], cse_subs))
        lines = []
        for s, expr in keep_subs:
            lines.append(f"float {s} = {my_ccode(_maybe_simplify(expr))};")
        for i, expr in enumerate(cse_exprs):
            lines.append(f"d_rho[{i}] = {my_ccode(_maybe_simplify(expr))};")
        return (hoist_subs, lines) if return_hoist else lines

    # Batched CSE: reduces live temporaries and often reduces register pressure.
    bs = int(batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be a positive integer, or None")
    lines = []
    hoist_all = []
    batch_id = 0
    for base in range(0, len(drho_exprs), bs):
        hoist_subs, block_lines = _emit_block(drho_exprs[base:base + bs], base, batch_id)
        hoist_all.extend(hoist_subs)
        lines.extend(block_lines)
        batch_id += 1
    return (hoist_all, lines) if return_hoist else lines


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

    drive_lines = []
    for sym, orig_func in func_subs.items():
        drive_lines.append(f"float {sym} = {my_ccode(orig_func)};")

    # Write directly to Drive_arr to avoid duplicate temps (compiler can scalarize indices).
    for i, e in enumerate(exprs_subbed):
        drive_lines.append(f"{array_name}[{i}] = {my_ccode(e)};")

    # No alias lines: RHS is rewritten to use Drive_arr[i] directly.
    return drive_lines, [], syms


def mesolve_2D(H, Drive, Col_Ops, mean_operator, tlist,
               var_arrays=None, const_values=None, kernel_template_file=None, *,
               RHSreuse=True, runtime_consts=None, keep_symbolic_consts=None,
               auto_runtime_consts=False, output_mode="mean", fp64=False,
               pre_expand=True, collect_rho=True, factor_terms=False,
               cse_batch_size=None, cse_simplify=True, hoist_rho_independent=True,
               nvrtc_options=(), timings=False, return_timing_info=False, warmup_time=0.0,
               rho0=None, rho0_var_arrays=None, rho0_values=None,
               return_time_trace=False, time_trace_every=None,
               time_trace_samples_per_period=None, solver_samples_per_period=None,
               Actual_Kernel_Save=False, beep_on_error=False, ignore_non_finite_output=False):
    """
    Solve a Lindblad master equation over a two-dimensional parameter sweep on GPU.

    Args:
        H: ``N x N`` symbolic Hamiltonian matrix, where ``N`` is the number of
            simulated basis states or quantum levels.
        Drive: Symbolic drive expression or dict[Symbol -> expression].
        Col_Ops: List of symbolic collapse operators.
        mean_operator: Symbolic operator whose expectation value is returned.
        tlist: Uniform, strictly increasing time samples beginning at zero. A
            list of ``M`` samples defines ``M - 1`` fixed-step fourth-order
            Runge-Kutta (RK4) intervals.
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
        fp64: If True, generate and run a 64-bit floating-point (FP64) kernel;
            otherwise use 32-bit floating point (FP32).
        pre_expand / collect_rho / factor_terms / cse_batch_size / cse_simplify /
            hoist_rho_independent: SymPy codegen controls.
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
        time_trace_every: Record one trace point every k RK4 kernel steps. For
            stride k, times are dt, (k+1)*dt, (2k+1)*dt, and so on.
        time_trace_samples_per_period: Desired trace samples per driving period. Requires
            solver_samples_per_period and maps to time_trace_every internally.
        solver_samples_per_period: Number of RK4 integration steps per driving period.
        Actual_Kernel_Save: If truthy, save the fully generated CUDA kernel to
            "<calling_python_file>_Kernel.cu". If a non-empty string is given,
            use it as the output path.
        beep_on_error: If True, play a short beep before raising non-finite output errors.
        ignore_non_finite_output: If True, do not raise on not-a-number (NaN) or
            infinite (Inf) output; return it as-is.

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
    valid_output_modes = {"mean", "final", "final_rho"}
    if output_mode not in valid_output_modes:
        raise ValueError(f"Unsupported output_mode '{output_mode}'. "
                         f"Use one of: {sorted(valid_output_modes)}")
    kernel_template_file = _resolve_kernel_template_file(kernel_template_file)

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

    # Validate dimensions and map sweep symbols to kernel parameters (ParX/ParY).
    N = H.shape[0]
    rho0_exprs = None
    uses_rho0_values = rho0_values is not None
    if rho0 is not None and uses_rho0_values:
        raise ValueError("Use either rho0 or rho0_values, not both.")
    if rho0 is not None:
        rho0_exprs = rho_matrix_to_independent_exprs(rho0)
        if len(rho0_exprs) != N * N - 1:
            raise ValueError("rho0 does not match Hamiltonian size.")

    for L in Col_Ops:
        if L.shape != (N, N):
            raise ValueError(f"Collapse operator dimension {L.shape} does not match H {N, N}")
    if mean_operator.shape != (N, N):
        raise ValueError(f"Mean operator dimension {mean_operator.shape} does not match H {N, N}")

    if len(var_arrays) == 0:
        raise ValueError("var_arrays or rho0_var_arrays must contain at least one sweep array "
                         "(ParX).")
    if len(var_arrays) > 2:
        raise ValueError("Only 2 total sweep arrays are supported across var_arrays and "
                         "rho0_var_arrays.")

    if tlist is None:
        raise ValueError("tlist is required for RK4 integration.")
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
    if not np.isclose(tlist_host[0], 0.0, rtol=0.0, atol=storage_tolerance):
        raise ValueError("tlist must begin at zero; the CUDA kernel derives stage times from the "
                         "step index.")
    if not np.allclose(time_diffs, uniform_dt_host, rtol=1.0e-6, atol=storage_tolerance):
        raise ValueError("tlist must be uniformly spaced because the GPU RK4 kernel uses one "
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

    subs = {
        sym: sp.Symbol(name, real=True)
        for sym, name in zip(var_arrays.keys(), ["ParX", "ParY"])
    }
    H = H.subs(subs)
    if isinstance(Drive, dict):
        Drive = {sym: sp.sympify(expr).subs(subs) for sym, expr in Drive.items()}
    else:
        Drive = sp.sympify(Drive).subs(subs)
    Col_Ops = [L.subs(subs) for L in Col_Ops]
    mean_operator = mean_operator.subs(subs)
    if rho0_exprs is not None:
        rho0_exprs = [expr.subs(subs) for expr in rho0_exprs]
    # Prepare runtime constants (optional), including auto-derived const-only expressions.
    runtime_consts = runtime_consts or {}
    if not isinstance(runtime_consts, dict):
        raise ValueError("runtime_consts must be a dict of {Symbol: value}")

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
        for e in H:
            yield e
        if isinstance(Drive, dict):
            for e in Drive.values():
                yield e
        else:
            yield Drive
        for L in Col_Ops:
            for e in L:
                yield e
        for e in mean_operator:
            yield e
        if rho0_exprs is not None:
            for e in rho0_exprs:
                yield e

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

    runtime_const_syms = base_runtime_syms + derived_const_syms

    def _compute_runtime_const_vals():
        """Evaluate runtime constants and derived runtime constants for this call."""
        if not runtime_const_syms:
            return []
        base_vals = {}
        for s in base_runtime_syms:
            if s in runtime_consts:
                base_vals[s] = float(runtime_consts[s])
            elif s in const_values:
                base_vals[s] = float(const_values[s])
            else:
                raise ValueError(f"Missing value for runtime constant symbol: {s}")
        vals = [base_vals[s] for s in base_runtime_syms]
        for expr in derived_const_exprs:
            vals.append(float(expr.evalf(subs=base_vals)))
        return vals

    runtime_const_vals = _compute_runtime_const_vals()

    # Generate symbolic drho C code with variable parameters replaced
    Drive_Symb = sp.Symbol("Drive", real=True)

    # Kernel cache key (avoid RHS regeneration/compilation if ODE is unchanged)
    def _expr_key(x):
        """Return a stable symbolic serialization for cache keys."""
        return sp.srepr(x)

    def _drive_key(drv):
        """Return a stable cache-key representation for drive expressions."""
        if isinstance(drv, dict):
            return tuple((str(k), sp.srepr(v)) for k, v in drv.items())
        return sp.srepr(drv)

    def _collect_used_const_indices(init_lines_in, static_lines_in, drive_lines_in, drho_lines_in,
                                    mean_line_in, final_line_in):
        """Return Const_arr indices that are actually referenced by generated code."""
        pat = re.compile(r"Const_arr\[(\d+)\]")
        used = set()
        for s in (list(init_lines_in) + list(static_lines_in) + list(drive_lines_in) +
                  list(drho_lines_in) + [mean_line_in, final_line_in]):
            if not s:
                continue
            for m in pat.finditer(s):
                used.add(int(m.group(1)))
        return sorted(used)

    def _remap_const_indices(s, idx_map):
        """Rewrite Const_arr indices after unused constants have been removed."""
        if not s:
            return s
        return re.sub(r"Const_arr\[(\d+)\]",
                      lambda m: f"Const_arr[{idx_map.get(int(m.group(1)), int(m.group(1)))}]", s)

    # Include template path/mtime and codegen knobs in the cache key so reuse is safe.
    template_mtime = (os.path.getmtime(kernel_template_file)
                      if os.path.exists(kernel_template_file) else None)
    cache_key = (
        "rhs_v3", N,
        _expr_key(H), _drive_key(Drive), tuple(_expr_key(L) for L in Col_Ops),
        _expr_key(mean_operator),
        tuple((str(k), float(v)) for k, v in
              sorted(const_values_eff.items(), key=lambda kv: str(kv[0]))),
        tuple(str(s) for s in runtime_const_syms),
        tuple(sp.srepr(expr) for expr in rho0_exprs) if rho0_exprs is not None else None,
        bool(uses_rho0_values), effective_output_mode, warmup_steps_host, bool(fp64),
        pre_expand, collect_rho, factor_terms, cse_batch_size, cse_simplify,
        hoist_rho_independent, return_time_trace,
        kernel_template_file, template_mtime,
        tuple(nvrtc_options) if not isinstance(nvrtc_options, str) else (nvrtc_options,),
    )

    collect_timings = bool(timings or return_timing_info)
    total_start = time.time() if collect_timings else 0.0
    rhs_stage_start = time.time() if collect_timings else 0.0
    cache_status = "miss"
    saved_kernel_path = None

    cached = _KERNEL_CACHE.get(cache_key) if RHSreuse else None
    used_const_indices = None
    if cached is None:
        # 1) Symbolic codegen -> kernel source patching -> NVRTC compilation.
        (static_lines, drive_lines, _drive_alias_lines, drho_lines, mean_line, final_line,
         static_syms, drive_syms, hoisted_syms,
         ) = generate_unrolled_drho(N, H, Drive_Symb, Col_Ops, mean_operator, Drive,
                                    runtime_const_syms=runtime_const_syms, pre_expand=pre_expand,
                                    collect_rho=collect_rho, factor_terms=factor_terms,
                                    cse_batch_size=cse_batch_size, cse_simplify=cse_simplify,
                                    hoist_rho_independent=hoist_rho_independent)
        if uses_rho0_values:
            init_lines = [f"rho[{i}] = Rho0_arr[result_idx * N + {i}];" for i in range(N * N - 1)]
        elif rho0_exprs is None:
            init_lines = ["for (int i=0;i<N;++i) rho[i] = 0.0f;", "rho[0] = 1.0f;"]
        else:
            init_lines = [f"rho[{i}] = {my_ccode(expr)};" for i, expr in enumerate(rho0_exprs)]
        used_const_indices = []
        if runtime_const_syms:
            for idx, sym in enumerate(runtime_const_syms):
                name = str(sym)
                pat = re.compile(r"(?<![0-9A-Za-z_])" + re.escape(name) + r"(?![0-9A-Za-z_])")
                repl = f"Const_arr[{idx}]"
                init_lines = [pat.sub(repl, line) for line in init_lines]
            init_lines = tidy_c_lines(init_lines)
            used_const_indices = _collect_used_const_indices(init_lines, static_lines, drive_lines,
                                                             drho_lines, mean_line, final_line)
            if used_const_indices:
                full_idx = list(range(len(runtime_const_syms)))
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

        # Read CUDA kernel template
        with open(kernel_template_file) as f:
            template_text = f.read()
            src = Template(template_text)
        has_const_arg_marker = "#CONST_ARG_DECL#" in template_text
        if runtime_const_syms and not has_const_arg_marker:
            raise ValueError("Kernel template is missing #CONST_ARG_DECL# but runtime constants "
                             "are enabled. Use N_Level_Kernel.cu template.")
        static_code = ("\n    ".join(static_lines)
                       if static_lines else "/* no thread-static terms */")
        drho_code = "\n    ".join(drho_lines)
        drives_code = "\n    ".join(drive_lines) if drive_lines else "/* no drives */"
        num_statics = max(1, len(static_syms))
        num_drives = max(1, len(drive_syms) + len(hoisted_syms))  # Drive_arr[0] exists if used
        kernel_code = src.substitute(N_DECL=str(N * N - 1), NUM_STATICS_DECL=str(num_statics),
                                     NUM_DRIVES_DECL=str(num_drives))
        const_arg_decl = (f"    const {scalar_type}* __restrict__ Const_arr,\n"
                          if has_const_arg_marker else "")
        kernel_code = kernel_code.replace("#CONST_ARG_DECL#", const_arg_decl)
        rho0_arg_decl = (f"    const {scalar_type}* __restrict__ Rho0_arr,\n"
                         if uses_rho0_values else "")
        kernel_code = kernel_code.replace("#RHO0_ARG_DECL#", rho0_arg_decl)
        kernel_code = kernel_code.replace("#INIT_RHO#", "\n            ".join(init_lines))
        kernel_code = kernel_code.replace("#INSERT_STATICS#", static_code)
        kernel_code = kernel_code.replace("#INSERT_DRIVES#", drives_code)
        kernel_code = kernel_code.replace("#INSERT_DRHO#", drho_code)
        mean_line_kernel = mean_line.replace("Drive_arr[", "Drive[")
        mean_line_kernel = mean_line_kernel.replace("Static_arr[", "Static[")
        final_line_kernel = final_line.replace("Drive_arr[", "Drive[")
        final_line_kernel = final_line_kernel.replace("Static_arr[", "Static[")
        if effective_output_mode == "final":
            mean_line_out = ""
            final_line_out = final_line_kernel
            results_line = (
                "results[result_idx].x = avg.x;\n            results[result_idx].y = avg.y;")
        elif effective_output_mode == "final_rho":
            mean_line_out = ""
            final_line_out = ""
            results_line = ("for (int i = 0; i < N; ++i) {\n"
                            "                results[result_idx * N + i] = rho[i];\n"
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
        if return_time_trace:
            trace_obs_line = final_line_kernel.replace("avg.x",
                                                       "time_trace_results[trace_offset].x")
            trace_obs_line = trace_obs_line.replace("avg.y", "time_trace_results[trace_offset].y")
            trace_line_out = (
                "if ((step % time_trace_stride) == 0) {\n"
                "                    const int trace_idx = step / time_trace_stride;\n"
                "                    if (trace_idx < num_time_trace) {\n"
                "                        const int trace_offset = result_idx * "
                "num_time_trace + trace_idx;\n"
                f"                        {trace_obs_line}\n"
                "                    }\n"
                "                }")
            mean_line_out = (f"{mean_line_out}\n                {trace_line_out}"
                             if mean_line_out else trace_line_out)
        kernel_code = kernel_code.replace("#MEAN_LINE#", mean_line_out)
        kernel_code = kernel_code.replace("#FINAL_LINE#", final_line_out)
        kernel_code = kernel_code.replace("#RESULTS_LINE#", results_line)

        # Match output pointer type to selected output mode.
        if effective_output_mode == "final_rho":
            result_arg_decl = f"    {scalar_type}* __restrict__ results"
            result_arg_comment = " // final reduced rho vector"
        else:
            result_arg_decl = f"    {complex_type}* __restrict__ results"
            result_arg_comment = " // averaged/final expectation value"
        if return_time_trace:
            result_arg_decl += (f",\n    {complex_type}* __restrict__ time_trace_results,"
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
        cached = {"kernel": time_evolution_kernel,
                  "uses_const_arr_arg": has_const_arg_marker,
                  "uses_rho0_arr_arg": uses_rho0_values,
                  "uses_time_trace": return_time_trace,
                  "used_const_indices": tuple(used_const_indices or []),
                  "kernel_code": kernel_code}
        if RHSreuse:
            _KERNEL_CACHE[cache_key] = cached
    else:
        # 2) Reuse already compiled kernel and constant index mapping.
        cache_status = "hit"
        time_evolution_kernel = cached["kernel"]
        if Actual_Kernel_Save and cached.get("kernel_code"):
            saved_kernel_path = _save_generated_kernel_file(Actual_Kernel_Save,
                                                            cached["kernel_code"])
        cached_used = cached.get("used_const_indices", None)
        if cached_used is None:
            used_const_indices = list(range(len(runtime_const_vals)))
        else:
            used_const_indices = list(cached_used)

    rhs_stage_time = (time.time() - rhs_stage_start) if collect_timings else 0.0

    # 3) Upload sweep axes and launch kernel.
    def to_device(x):
        """Convert a host array to a CuPy array with the selected scalar dtype."""
        return x if isinstance(x, cp.ndarray) else cp.asarray(x, dtype=scalar_dtype)

    ParX_list = to_device(next(iter(var_arrays.values())))
    ParY_list = (to_device(list(var_arrays.values())[1]) if len(var_arrays) > 1 else
                 cp.zeros(1, dtype=scalar_dtype))
    num_X, num_Y = int(len(ParX_list)), int(len(ParY_list))
    rho0_arr = None
    if uses_rho0_values:
        rho0_np = np.asarray(rho0_values, dtype=np.float64 if fp64 else np.float32)
        nred = N * N - 1
        if rho0_np.shape == (num_X, nred) and num_Y == 1:
            rho0_np = rho0_np.reshape(num_X, 1, nred)
        if rho0_np.shape != (num_X, num_Y, nred):
            raise ValueError(f"rho0_values must have shape ({num_X}, {num_Y}, {nred}) "
                             f"or ({num_X}, {nred}) when num_Y=1; got {rho0_np.shape}.")
        rho0_arr = cp.asarray(rho0_np.reshape(num_X * num_Y, nred), dtype=scalar_dtype)
    # Allocate output
    if effective_output_mode == "final_rho":
        results = cp.zeros((num_X, num_Y, N * N - 1), dtype=scalar_dtype)
    else:
        results = cp.zeros((num_X, num_Y), dtype=complex_dtype)
    time_trace_results = (cp.zeros((num_X, num_Y, num_time_trace_host), dtype=complex_dtype)
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
    if cached.get("uses_const_arr_arg", False):
        kernel_args.append(const_arr)
    if cached.get("uses_rho0_arr_arg", False):
        kernel_args.append(rho0_arr)
    kernel_args.append(results)
    if cached.get("uses_time_trace", False):
        kernel_args.extend([time_trace_results, np.int32(time_trace_stride_host),
                            np.int32(num_time_trace_host)])
    time_evolution_kernel(grid_dim, block_dim, tuple(kernel_args))

    if collect_timings:
        ev_end.record()
        ev_end.synchronize()
        gpu_kernel_time = cp.cuda.get_elapsed_time(ev_start, ev_end) / 1000.0

    results_np = cp.asnumpy(results)
    time_trace_np = cp.asnumpy(time_trace_results) if return_time_trace else None
    if not np.all(np.isfinite(results_np)):
        if ignore_non_finite_output:
            print("mesolve_2D warning: non-finite values detected in output; returning raw data "
                  "because ignore_non_finite_output=True.")
        else:
            if beep_on_error:
                _play_notification_beep("error")
            raise RuntimeError("Non-finite values detected in mesolve_2D output. Try reducing dt "
                               "(increase time samples) or check model parameters/RHS.")
    if return_time_trace and not np.all(np.isfinite(time_trace_np)):
        if ignore_non_finite_output:
            print("mesolve_2D warning: non-finite values detected in time trace; returning raw "
                  "trace because ignore_non_finite_output=True.")
        else:
            if beep_on_error:
                _play_notification_beep("error")
            raise RuntimeError("Non-finite values detected in mesolve_2D time trace. Try reducing "
                               "dt (increase time samples) or check model parameters/RHS.")

    timing_info = None
    if collect_timings:
        total_time = time.time() - total_start
        timing_info = {"rhs_stage_s": rhs_stage_time, "gpu_kernel_s": gpu_kernel_time,
                       "total_s": total_time, "cached_rhs": cache_status}
    if timings:
        print(f"mesolve_2D timings: rhs_stage={rhs_stage_time:.3f}s "
              f"gpu_kernel={gpu_kernel_time:.3f}s total={total_time:.3f}s cachedRHS={cache_status}")
    if saved_kernel_path:
        print(f"Generated kernel saved to: {saved_kernel_path}")
    if return_time_trace:
        dt_host = uniform_dt_host
        trace_steps = np.arange(num_time_trace_host, dtype=np.float32) * time_trace_stride_host
        trace_t = (trace_steps + 1.0) * dt_host
        output = (results_np, time_trace_np, trace_t)
        return (*output, timing_info) if return_timing_info else output
    if return_timing_info:
        return results_np, timing_info
    return results_np

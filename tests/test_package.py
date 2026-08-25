import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from gqis.solver import (_resolve_kernel_template_file, build_independent_rho,
                         build_reduced_lindblad_rhs, mesolve_2D)


def test_public_package_import_does_not_require_cupy():
    code = textwrap.dedent("""
        import builtins

        original_import = builtins.__import__

        def import_without_cupy(name, *args, **kwargs):
            if name == "cupy" or name.startswith("cupy."):
                raise ModuleNotFoundError("No module named 'cupy'", name="cupy")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_cupy
        import gqis

        assert "gqis.solver" not in sys.modules
        try:
            gqis.mesolve_2D
        except ImportError as exc:
            assert "gqis[cuda12]" in str(exc)
        else:
            raise AssertionError("mesolve_2D loaded without CuPy")
    """)
    proc = subprocess.run([sys.executable, "-c", "import sys\n" + code], capture_output=True,
                          text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_mesolve_api_reference_documents_every_parameter():
    reference = (Path(__file__).parents[1] / "GQIS_API.md").read_text(encoding="utf-8")
    undocumented = [name for name in inspect.signature(mesolve_2D).parameters
                    if f"`{name}`" not in reference]
    assert undocumented == []


def test_packaged_cuda_template_is_discoverable():
    template = Path(_resolve_kernel_template_file(None))
    assert template.is_file()
    assert template.parent.name == "gqis"


def test_independent_density_matrix_layout():
    rho, metadata = build_independent_rho(3)
    assert rho.shape == (3, 3)
    assert len(metadata["rho_syms"]) == 8
    assert metadata["num_coherences"] == 3
    assert sp.simplify(sp.trace(rho) - 1) == 0


def test_reduced_lindblad_rhs_matches_two_level_bloch_equations():
    delta, drive, gamma1, gamma2 = sp.symbols(
        "delta drive gamma1 gamma2", real=True, nonnegative=True)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    sm = sp.Matrix([[0, 1], [0, 0]])
    population = sp.diag(0, 1)

    equations, mean_re, mean_im, metadata = build_reduced_lindblad_rhs(
        2, delta * sx / 2 + drive * sz / 2,
        [sp.sqrt(gamma1) * sm, sp.sqrt(gamma2) * sz], population)
    rho00, rho01_re, rho01_im = metadata["rho_syms"]
    coherence_decay = gamma1 / 2 + 2 * gamma2
    expected = [gamma1 * (1 - rho00) - delta * rho01_im,
                drive * rho01_im - coherence_decay * rho01_re,
                delta * (rho00 - sp.Rational(1, 2)) - drive * rho01_re
                - coherence_decay * rho01_im]

    assert len(equations) == 3
    assert all(sp.simplify(actual - reference) == 0
               for actual, reference in zip(equations, expected))
    assert sp.simplify(mean_re - (1 - rho00)) == 0
    assert mean_im == 0


def _minimal_model():
    sweep = sp.Symbol("sweep", real=True)
    return sp.zeros(2), sp.Integer(0), [], sp.eye(2), {sweep: np.array([0.0], dtype=np.float32)}


def test_nonuniform_time_grid_is_rejected_before_gpu_launch():
    hamiltonian, drive, collapse, observable, sweep = _minimal_model()
    with pytest.raises(ValueError, match="uniformly spaced"):
        mesolve_2D(hamiltonian, drive, collapse, observable, [0.0, 1.0, 2.5], var_arrays=sweep)


def test_nonzero_time_origin_is_rejected_before_gpu_launch():
    hamiltonian, drive, collapse, observable, sweep = _minimal_model()
    with pytest.raises(ValueError, match="begin at zero"):
        mesolve_2D(hamiltonian, drive, collapse, observable, [1.0, 2.0], var_arrays=sweep)

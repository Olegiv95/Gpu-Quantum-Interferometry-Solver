import inspect
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from gqis.solver import (_resolve_kernel_template_file, build_independent_rho,
                         build_reduced_lindblad_rhs, mesolve_2D)


def test_mesolve_api_reference_documents_every_parameter():
    reference = (Path(__file__).parents[1] / "GQIS_API.md").read_text(encoding="utf-8")
    undocumented = [name for name in inspect.signature(mesolve_2D).parameters
                    if f"`{name}`" not in reference]
    assert undocumented == []


def test_packaged_cuda_template_is_discoverable():
    template = Path(_resolve_kernel_template_file(None))
    assert template.is_file()
    assert template.parent.name == "gqis"


def test_legacy_imports_resolve_to_canonical_api():
    import GPU_Int_Tool
    import gpu_int_tool

    assert gpu_int_tool.mesolve_2D is mesolve_2D
    assert GPU_Int_Tool.mesolve_2D is mesolve_2D
    assert gpu_int_tool.build_reduced_lindblad_rhs is build_reduced_lindblad_rhs
    assert GPU_Int_Tool.build_reduced_lindblad_rhs is build_reduced_lindblad_rhs


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

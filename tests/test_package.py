from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from gqis.solver import _resolve_kernel_template_file, build_independent_rho, mesolve_2D


def test_packaged_cuda_template_is_discoverable():
    template = Path(_resolve_kernel_template_file(None))
    assert template.is_file()
    assert template.parent.name == "gqis"


def test_legacy_imports_resolve_to_canonical_api():
    import GPU_Int_Tool
    import gpu_int_tool

    assert gpu_int_tool.mesolve_2D is mesolve_2D
    assert GPU_Int_Tool.mesolve_2D is mesolve_2D


def test_independent_density_matrix_layout():
    rho, metadata = build_independent_rho(3)
    assert rho.shape == (3, 3)
    assert len(metadata["rho_syms"]) == 8
    assert sp.simplify(sp.trace(rho) - 1) == 0


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

import numpy as np
import pytest
import sympy as sp

cp = pytest.importorskip("cupy")


def _gpu_available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU is unavailable")
def test_pi_rotation_reaches_requested_final_time():
    from gqis import mesolve_2D

    dummy = sp.Symbol("dummy", real=True)
    hamiltonian = sp.Matrix([[0, sp.Rational(1, 2)], [sp.Rational(1, 2), 0]])
    rho0 = sp.Matrix([[1, 0], [0, 0]])
    tlist = np.linspace(0.0, np.pi, 101, dtype=np.float32)

    final_rho = mesolve_2D(hamiltonian, sp.Integer(0), [], sp.eye(2), tlist,
                           var_arrays={dummy: np.array([0.0], dtype=np.float32)}, rho0=rho0,
                           output_mode="final_rho")
    assert abs(float(final_rho[0, 0, 0])) < 2.0e-5

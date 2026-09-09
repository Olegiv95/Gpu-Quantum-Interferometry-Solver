"""CPU-only API/code-generation checks; opt-in tiny CUDA numerical check."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import sympy as sp
from gqis import odesolve_2D
import gqis.solver as core


class OdeApiTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.t, self.a = sp.symbols("x y t a", real=True)
        self.codes = []
        def kernel(code, *args, **kwargs):
            self.codes.append(code)
            return lambda *args: None
        fake = SimpleNamespace(float32=np.float32, float64=np.float64,
            complex64=np.complex64, complex128=np.complex128, ndarray=np.ndarray,
            asarray=np.asarray, asnumpy=np.asarray, zeros=np.zeros, RawKernel=kernel,
            all=np.all, isfinite=np.isfinite)
        self.mock = patch.object(core, "cp", fake)
        self.mock.start()
        self.addCleanup(self.mock.stop)
        core._RHS_CODE_CACHE.clear()
        core._KERNEL_CACHE.clear()
        self.addCleanup(core._RHS_CODE_CACHE.clear)
        self.addCleanup(core._KERNEL_CACHE.clear)

    def solve(self, **kwargs):
        return odesolve_2D([self.y, -self.a*self.x + sp.sin(self.t)],
            [self.x, self.y], [1, 0], np.linspace(0, 1, 9),
            var_arrays={self.a: np.array([1., 2.])}, **kwargs)

    def test_shapes_and_time_dependent_rhs(self):
        final, trace, times = self.solve(return_time_trace=True, time_trace_every=3)
        self.assertEqual(final.shape, (2, 1, 2))
        self.assertEqual(trace.shape, (2, 1, 3, 2))
        np.testing.assert_allclose(times, [0.125, 0.5, 0.875])
        code = self.codes[-1]
        self.assertIn("#define N 2", code)
        rhs = code.split("DRHO_ATTR void compute_drho(")[1].split("__global__")[0]
        self.assertNotIn("sinf(", rhs)  # evaluated in the stage-drive function
        self.assertNotIn("ode_stage_time", code)
        self.assertNotIn("#MEAN_LINE#", code)

    def test_mean_and_shared_rhs(self):
        mean = self.solve(output_mode="mean")
        self.assertEqual(mean.shape, (2, 1, 2))
        self.assertIn("state_avg[i] += rho[i]", self.codes[-1])
        count = len(core._RHS_CODE_CACHE)
        self.solve(solver="tsit5")
        self.assertEqual(len(core._RHS_CODE_CACHE), count)

    def test_time_offset_reuses_kernel_and_shifts_trace_times(self):
        _, _, times = self.solve(t_in=3.0, return_time_trace=True)
        np.testing.assert_allclose(times, 3 + np.arange(1, 9)/8)
        self.assertIn("compute_drives(ParX, ParY, t_in,", self.codes[-1])
        self.assertIn("t += t_in;", self.codes[-1])
        self.solve(t_in=5.0, return_time_trace=True)
        self.assertEqual(len(self.codes), 1)

    def test_zero_and_nonzero_time_share_kernel(self):
        self.solve(t_in=0.0)
        self.assertIn("t += t_in;", self.codes[-1])
        self.solve(t_in=1.0)
        self.assertIn("t += t_in;", self.codes[-1])
        self.assertEqual(len(core._RHS_CODE_CACHE), 1)
        self.solve(t_in=0.0)
        self.assertEqual(len(self.codes), 1)

    def test_device_output_does_not_copy_state_to_host(self):
        with patch.object(core.cp, "asnumpy", side_effect=AssertionError("unexpected host transfer")):
            result = self.solve(return_device=True)
            odesolve_2D([-self.x, -self.y], [self.x, self.y], None, [0, 0.1],
                var_arrays={self.a: np.array([1., 2.])}, y0_values=result,
                t_in=1.0, return_device=True)

    def test_observable_and_fp64(self):
        result, trace, times = self.solve(observable=(self.x+self.y)**2 +
            sp.I*(self.x+self.y)**3 + self.t, fp64=True, return_time_trace=True)
        self.assertEqual(result.dtype, np.complex128)
        self.assertEqual(trace.shape, (2, 1, 8))
        self.assertEqual(times.dtype, np.float64)
        drive = self.codes[-1].split("DRIVES_ATTR void compute_drives(")[1].split("// Force-inline")[0]
        self.assertNotIn("rho[", drive)

    def test_static_precomputation(self):
        odesolve_2D([sp.exp(self.a)*self.x], [self.x], [1], [0, 0.1],
                    var_arrays={self.a: np.array([1.])})
        code = self.codes[-1]
        static = code.split("DRIVES_ATTR void compute_static_terms(")[1].split("// Compute time-dependent")[0]
        self.assertIn("expf(ParX)", static)

    def test_explicit_initial_states_and_equations(self):
        equation = sp.Eq(sp.Derivative(self.x, self.t, evaluate=False), -self.x)
        result = odesolve_2D([equation], [self.x], None, [0, 0.1], y0_values=np.ones((1, 1)))
        self.assertEqual(result.shape, (1, 1, 1))
        self.assertIn("Rho0_arr[", self.codes[-1])

    def test_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "Unassigned"):
            odesolve_2D([self.a*self.x], [self.x], [1], [0, 1])
        with self.assertRaisesRegex(ValueError, "real"):
            odesolve_2D([sp.I*self.x], [self.x], [1], [0, 1])

    def test_lindblad_entry_point_still_generates(self):
        result = core.mesolve_2D(sp.zeros(2), {}, [], sp.diag(1, 0), [0, 0.1],
                                var_arrays={self.a: np.array([0.])})
        self.assertEqual(result.shape, (1, 1))
        self.assertIn("#define N 3", self.codes[-1])


@unittest.skipUnless(os.environ.get("GQIS_RUN_GPU_TESTS") == "1", "opt-in CUDA test")
class OdeGpuTests(unittest.TestCase):
    def test_absolute_time(self):
        x, t = sp.symbols("x t", real=True)
        grid = np.linspace(0, 1, 65)
        for solver in ("rk4", "lserk4", "dp5", "ab5", "anas5", "tsit5", "alshina6", "dop853"):
            result = odesolve_2D([t], [x], [0], grid, t_in=2.0, solver=solver)
            np.testing.assert_allclose(result, 2.5, atol=2e-5)

    def test_small_analytic_problem(self):
        x, y, t = sp.symbols("x y t", real=True)
        grid = np.linspace(0, 1, 129)
        for solver in ("rk4", "lserk4", "dp5", "ab5", "anas5", "tsit5", "alshina6", "dop853"):
            final, trace, times = odesolve_2D([-x, t], [x, y], [1, 0], grid,
                solver=solver, return_time_trace=True, time_trace_every=16)
            np.testing.assert_allclose(final[0, 0], [np.exp(-1), 0.5], atol=2e-5)
            np.testing.assert_allclose(trace[0, 0, :, 0], np.exp(-times), atol=2e-5)
            np.testing.assert_allclose(trace[0, 0, :, 1], times**2/2, atol=2e-5)
        mean = odesolve_2D([-x, t], [x, y], [1, 0], grid, output_mode="mean")
        np.testing.assert_allclose(mean[0, 0], [np.exp(-grid[1:]).mean(),
            (grid[1:]**2/2).mean()], atol=2e-5)


if __name__ == "__main__":
    unittest.main()

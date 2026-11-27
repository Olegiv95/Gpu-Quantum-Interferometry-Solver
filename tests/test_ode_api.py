"""CPU-only API/code-generation checks; opt-in tiny CUDA numerical check."""
import os
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import sympy as sp
from gqis import compile_lindblad_rhs, compile_ode_rhs, odesolve_2D
import gqis.solver as core


class OdeApiTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.t, self.a = sp.symbols("x y t a", real=True)
        self.codes = []
        def kernel(code, *args, **kwargs):
            self.codes.append(code)
            def launch(*args):
                return None
            launch.compile = lambda: None  # the solver compiles before the timed launch
            return launch
        fake = SimpleNamespace(float32=np.float32, float64=np.float64,
            complex64=np.complex64, complex128=np.complex128, ndarray=np.ndarray,
            asarray=np.asarray, asnumpy=np.asarray, zeros=np.zeros, RawKernel=kernel,
            all=np.all, isfinite=np.isfinite)
        self.mock = patch.object(core, "cp", fake)
        self.mock.start()
        self.addCleanup(self.mock.stop)
        for cache in (core._RHS_CODE_CACHE, core._KERNEL_CACHE, core._TIMESTEP_CALIBRATION_CACHE):
            cache.clear()
            self.addCleanup(cache.clear)

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

    def test_square_of_a_sum_keeps_its_parentheses(self):
        odesolve_2D([self.y, -(self.x + 1)**2 - self.a*self.y], [self.x, self.y], [1, 0], [0, 0.1],
                    var_arrays={self.a: np.array([1.])}, cse_simplify=False)
        self.assertIn("((rho[0] + 1)*(rho[0] + 1))", self.codes[-1])
        self.assertNotIn("pow(", self.codes[-1].split("compute_drho(")[1].split("__global__")[0])

    def test_exact_constants_print_as_float_literals(self):
        odesolve_2D([self.y, -sp.sqrt(2)*self.x - sp.pi*self.a*self.y + sp.E*sp.cos(2*sp.pi*self.t)],
                    [self.x, self.y], [1, 0], [0, 0.1], var_arrays={self.a: np.array([1.])})
        self.assertNotRegex(self.codes[-1], r"\bM_\w+")  # C macros such as M_SQRT2: undefined in NVRTC
        self.assertIn("1.414213562e+00f", self.codes[-1])

    def test_compiled_ode_rhs_starts_from_code_emission(self):
        compiled = compile_ode_rhs([self.y, -self.a*self.x + sp.sin(self.t)], [self.x, self.y], [1, 0],
                                   [self.a])
        self.assertEqual(set(compiled.timings), {"ode_prepare_s", "ode_optimize_s"})
        self.assertEqual(self.codes, [])  # compiling the RHS emits and compiles no kernel
        core._RHS_CODE_CACHE.clear()
        with patch.object(core, "_compile_rhs", side_effect=AssertionError("ODE prepared again")):
            via = odesolve_2D(compiled, tlist=np.linspace(0, 1, 9), var_arrays={self.a: np.array([1., 2.])})
        self.assertEqual(via.shape, (2, 1, 2))
        self.solve()
        self.assertEqual(len(self.codes), 1)  # the direct call reuses the same kernel
        core._KERNEL_CACHE.clear()
        self.solve()
        self.assertEqual(self.codes[0], self.codes[1])
        with self.assertRaisesRegex(ValueError, "var_arrays must sweep"):
            odesolve_2D(compiled, tlist=[0, 1], var_arrays={})
        with self.assertRaisesRegex(ValueError, "fixed when the RHS is compiled"):
            odesolve_2D(compiled, tlist=[0, 1], var_arrays={self.a: np.array([1.])}, observable=self.x)
        with self.assertRaisesRegex(ValueError, "mesolve_2D"):
            odesolve_2D(compile_lindblad_rhs(sp.zeros(2), {}, [], sp.diag(1, 0), [self.a]),
                        tlist=[0, 1], var_arrays={self.a: np.array([0.])})

    def test_compiled_lindblad_rhs_matches_direct_call(self):
        model = (sp.Matrix([[self.a, 1], [1, -self.a]]), {}, [sp.Matrix([[0, 1], [0, 0]])], sp.diag(1, 0))
        compiled = compile_lindblad_rhs(*model, [self.a])
        core.mesolve_2D(compiled, tlist=[0, 0.1], var_arrays={self.a: np.array([0.])})
        core.mesolve_2D(*model, [0, 0.1], var_arrays={self.a: np.array([0.])}, RHSreuse=False)
        self.assertEqual(self.codes[0], self.codes[1])
        with self.assertRaisesRegex(ValueError, "order given when the RHS was compiled"):
            core.mesolve_2D(compiled, tlist=[0, 0.1], var_arrays={self.x: np.array([0.])})
        with self.assertRaisesRegex(ValueError, "do not pass the model"):
            core.mesolve_2D(compiled, {}, tlist=[0, 0.1], var_arrays={self.a: np.array([0.])})

    def test_grouped_options_match_flat_options(self):
        flat = self.solve(solver="tsit5", fp64=True, return_time_trace=True, time_trace_every=3, cse_batch_size=1)
        grouped = self.solve(solver_options={"solver": "tsit5", "fp64": True}, codegen={"cse_batch_size": 1},
                             output={"return_time_trace": True, "time_trace_every": 3})
        self.assertEqual(len(self.codes), 1)  # same cached kernel
        for a, b in zip(flat, grouped):
            np.testing.assert_array_equal(a, b)
        result = core.mesolve_2D(sp.zeros(2), {}, [], sp.diag(1, 0), [0, 0.1],
                                 var_arrays={self.a: np.array([0.])},
                                 output={"warmup_time": 0.5}, diagnostics={"timings": False})
        self.assertEqual(result.shape, (1, 1))

    def test_timestep_calibration_reuse_and_recalibration(self):
        with patch.object(core, "_calibrate_timestep", wraps=core._calibrate_timestep) as calibrate:
            settings = dict(rms_atol=1e-4, sample_stride=2)
            self.solve(calibrate_timesteps=settings)
            self.solve(calibrate_timesteps=settings)
            self.assertEqual(calibrate.call_count, 1)  # second call reuses the stored step
            self.solve(calibrate_timesteps=settings, cse_batch_size=1)
            self.assertEqual(calibrate.call_count, 1)  # code generation does not recalibrate
            self.solve(calibrate_timesteps=dict(settings, rms_atol=1e-5))
            self.assertEqual(calibrate.call_count, 2)  # tolerances do
            every = dict(settings, recalibrate_every=2, max_atol=0)
            for _ in range(3):
                self.solve(calibrate_timesteps=every)
            self.assertEqual(calibrate.call_count, 4)  # calls 1 and 3 of this setting
            self.solve(calibrate_timesteps=dict(every, reset=True))
            self.assertEqual(calibrate.call_count, 5)

    def test_timestep_calibration_never_accepts_unstable_runs(self):
        def fake_solve(**kwargs):  # coarse runs (fewer than 4 steps) are non-finite
            steps = len(kwargs["tlist"]) - 1
            return np.full((2, 2), np.nan) if steps < 4 else np.ones((2, 2))
        args = dict(var_arrays={self.a: np.arange(10.)}, rho0_var_arrays=None, rho0_values=None,
                    solver="rk4", fp64=False)
        settings = core._timestep_calibration_settings(dict(rms_atol=0, max_atol=0, rms_rtol=1e-3))
        with patch.object(core, "_solve_symbolic_system", fake_solve):
            info = core._calibrate_timestep(args, settings, base_steps=1, duration=1.0)
        self.assertEqual(info["steps"], 4)  # inf <= rtol*inf must not pass

    def test_timestep_calibration_test_grid_keeps_axis_ends(self):
        args = dict(var_arrays={self.a: np.arange(10.)}, rho0_var_arrays=None,
                    rho0_values=np.zeros((10, 3)), solver="rk4", fp64=False)
        test, full_grid = core._undersampled_arguments(args, 4)
        np.testing.assert_array_equal(test["var_arrays"][self.a], [0, 4, 8, 9])
        self.assertEqual(test["rho0_values"].shape, (4, 3))
        self.assertFalse(full_grid)
        self.assertTrue(core._undersampled_arguments(args, 1)[1])

    def test_timestep_calibration_warns_once_for_wider_sweep_range(self):
        settings = dict(rms_atol=1e-4, sample_stride=2)
        self.solve(calibrate_timesteps=settings)
        wider = dict(var_arrays={self.a: np.array([1., 3.])})
        with self.assertWarnsRegex(RuntimeWarning, "beyond the range used for calibration"):
            odesolve_2D([self.y, -self.a*self.x + sp.sin(self.t)], [self.x, self.y], [1, 0],
                        np.linspace(0, 1, 9), calibrate_timesteps=settings, **wider)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the second call must not warn again
            odesolve_2D([self.y, -self.a*self.x + sp.sin(self.t)], [self.x, self.y], [1, 0],
                        np.linspace(0, 1, 9), calibrate_timesteps=settings, **wider)
            quiet = dict(settings, range_warning=False, rms_atol=2e-4)  # new key, warning disabled
            self.solve(calibrate_timesteps=quiet)
            odesolve_2D([self.y, -self.a*self.x + sp.sin(self.t)], [self.x, self.y], [1, 0],
                        np.linspace(0, 1, 9), calibrate_timesteps=quiet, **wider)

    def test_timestep_calibration_key_includes_max_steps_and_reuses_full_grid(self):
        with patch.object(core, "_calibrate_timestep", wraps=core._calibrate_timestep) as calibrate:
            self.solve(calibrate_timesteps=dict(rms_atol=1e-4, sample_stride=1))
            self.solve(calibrate_timesteps=dict(rms_atol=1e-4, sample_stride=1, max_steps=2**10))
            self.assertEqual(calibrate.call_count, 2)
        # Full-grid test: the final result is reused and bisection is skipped (also with the
        # default bisection_steps), so only the two doubling runs launch the kernel.
        for bisection_steps in (0, 2):
            with patch.object(core, "_solve_symbolic_system", wraps=core._solve_symbolic_system) as solve:
                self.solve(calibrate_timesteps=dict(rms_atol=1e-4, sample_stride=1, reset=True,
                                                    bisection_steps=bisection_steps))
                launches = [c for c in solve.call_args_list if c.kwargs.get("calibrate_timesteps") is None]
            self.assertEqual(len(launches), 2)
        # full_grid_bisection=True restores the bisection candidates on a full-grid test.
        with patch.object(core, "_solve_symbolic_system", wraps=core._solve_symbolic_system) as solve:
            self.solve(calibrate_timesteps=dict(rms_atol=1e-4, sample_stride=1, reset=True,
                                                full_grid_bisection=True))
            launches = [c for c in solve.call_args_list if c.kwargs.get("calibrate_timesteps") is None]
        self.assertGreater(len(launches), 2)

    def test_timestep_calibration_bisection_finds_coarsest_passing_step(self):
        # Fake solver with a known error 3e4/steps**4: the limit 1e-3 needs at least ~74 steps.
        real = core._solve_symbolic_system

        def fake(**kwargs):
            if kwargs.get("calibrate_timesteps"):
                return real(**kwargs)
            steps = len(kwargs["tlist"]) - 1
            shape = tuple(len(v) for v in kwargs["var_arrays"].values()) + (1,)
            result = np.full(shape[:2], 0.3 + 3e4 / steps**4, dtype=complex)
            return (result, {}) if kwargs.get("return_timing_info") else result

        def steps_for(start, bisection_steps):
            core._TIMESTEP_CALIBRATION_CACHE.clear()
            with patch.object(core, "_solve_symbolic_system", side_effect=fake):
                _, info = real(H=sp.eye(2), Drive=sp.Integer(0), Col_Ops=[], mean_operator=sp.eye(2),
                               tlist=np.linspace(0, 1, start + 1), var_arrays={self.a: np.ones(4)},
                               return_timing_info=True, output_mode="final",  # fourth-order fake error
                               calibrate_timesteps=dict(rms_atol=1e-3, max_atol=1e-2,
                                                        bisection_steps=bisection_steps))
            return info["timestep_calibration"]["steps"]

        self.assertEqual(steps_for(16, 0), 128)   # doubling only: 64 fails, 128 passes
        self.assertEqual(steps_for(16, 2), 76)    # bracket [64, 128] -> 91 -> 76
        self.assertEqual(steps_for(512, 2), 76)   # a passing start is halved until 64 fails
        self.assertEqual(steps_for(512, 0), 512)  # without bisection the supplied grid is the coarsest

    def _calibrated(self, fake, **kwargs):
        """Run mesolve_2D calibration with the solver replaced by ``fake`` (uncalibrated calls)."""
        real = core._solve_symbolic_system

        def solve(**call):
            if call.get("calibrate_timesteps"):
                return real(**call)
            shape = tuple(len(v) for v in call["var_arrays"].values()) + (1,)
            result = np.full(shape[:2], fake(call), dtype=complex)
            return (result, {}) if call.get("return_timing_info") else result

        core._TIMESTEP_CALIBRATION_CACHE.clear()
        with patch.object(core, "_solve_symbolic_system", side_effect=solve):
            _, info = real(H=sp.eye(2), Drive=sp.Integer(0), Col_Ops=[], mean_operator=sp.eye(2),
                           tlist=np.linspace(0, 1, 17), var_arrays={self.a: np.ones(4)},
                           return_timing_info=True, **kwargs)
        return info["timestep_calibration"]

    def test_timestep_calibration_mean_uses_first_order_estimate(self):
        # Fake right-endpoint mean with error 1/steps: the limit 1e-3 needs 1000 steps.
        def fake(call):
            return 0.3 + 1.0 / (len(call["tlist"]) - 1)
        limits = dict(rms_atol=1e-3, max_atol=1e-2)
        self.assertEqual(self._calibrated(fake, output_mode="mean", calibrate_timesteps=limits)["steps"], 1024)
        # The same error treated with the RK4 order would accept 724 steps (error 1.4e-3).
        self.assertEqual(self._calibrated(fake, output_mode="final", calibrate_timesteps=limits)["steps"], 724)

    def test_tsit5a_calibration_scales_both_tolerances(self):
        # Fake result error 7*tol: the limit 1e-3 is met for tol <= 1.4e-4; the search steps by 10.
        def fake(call):
            self.assertEqual(len(call["tlist"]), 17)  # output times are never changed
            options = call["adaptive_options"]
            return 0.3 + 7.0 * max(options["step_rtol"], options["step_atol"])
        for start in (1e-5, 1e-2):  # a passing start is coarsened, a failing one refined
            report = self._calibrated(fake, solver="tsit5a", calibrate_timesteps=dict(rms_atol=1e-3, max_atol=1e-2),
                                      adaptive_options=dict(step_rtol=start, step_atol=start / 5))
            self.assertAlmostEqual(report["step_tol"], 1e-4)
            self.assertAlmostEqual(report["step_rtol"], 1e-4)
            self.assertAlmostEqual(report["step_atol"], 2e-5)  # the rtol/atol ratio is kept

    def test_tsit5a_calibration_stops_at_max_substeps(self):
        def fake(call):  # tolerances below 1e-6 exceed max_substeps
            tol = call["adaptive_options"]["step_rtol"]
            return np.nan if tol < 0.99e-6 else 0.3 + 1e3 * tol
        with self.assertWarnsRegex(RuntimeWarning, "max_substeps"):
            report = self._calibrated(fake, solver="tsit5a", calibrate_timesteps=dict(rms_atol=1e-3, max_atol=1e-2))
        self.assertAlmostEqual(report["step_tol"], 1e-6)
        self.assertEqual(report["details"]["status"], "max_substeps")

    def test_timestep_calibration_defaults_differ_for_lindblad_and_ode(self):
        lindblad = core._timestep_calibration_settings(True)
        ode = core._timestep_calibration_settings(True, ode=True)
        self.assertEqual((lindblad["rms_atol"], lindblad["max_atol"]), (1e-3, 1e-2))
        self.assertEqual((ode["rms_atol"], ode["max_atol"]), (1e-4, 1e-3))

    def test_timestep_calibration_rejects_unknown_settings(self):
        with self.assertRaisesRegex(ValueError, "Unknown calibrate_timesteps"):
            self.solve(calibrate_timesteps=dict(rms_tol=1e-4))
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.solve(calibrate_timesteps=dict(rms_atol=0, max_atol=0))

    def test_grouped_options_reject_conflicts_and_unknown_names(self):
        with self.assertRaisesRegex(ValueError, "both directly"):
            self.solve(fp64=True, solver_options={"fp64": False})
        with self.assertRaisesRegex(ValueError, "both directly"):
            self.solve(cse_batch_size=2, codegen={"cse_batch_size": 1})
        with self.assertRaisesRegex(ValueError, "Unknown codegen option"):
            self.solve(codegen={"fp64": True})
        with self.assertRaisesRegex(ValueError, "does not accept"):
            core.mesolve_2D(sp.zeros(2), {}, [], sp.diag(1, 0), [0, 0.1],
                            var_arrays={self.a: np.array([0.])}, initial={"y0_values": None})


@unittest.skipUnless(os.environ.get("GQIS_RUN_GPU_TESTS") == "1", "opt-in CUDA test")
class OdeGpuTests(unittest.TestCase):
    def test_absolute_time(self):
        x, t = sp.symbols("x t", real=True)
        grid = np.linspace(0, 1, 65)
        for solver in ("rk4", "lserk4", "lserk4f", "dp5", "ab5", "anas5", "tsit5", "alshina6", "dop853"):
            result = odesolve_2D([t], [x], [0], grid, t_in=2.0, solver=solver)
            np.testing.assert_allclose(result, 2.5, atol=2e-5)

    def test_small_analytic_problem(self):
        x, y, t = sp.symbols("x y t", real=True)
        grid = np.linspace(0, 1, 129)
        for solver in ("rk4", "lserk4", "lserk4f", "dp5", "ab5", "anas5", "tsit5", "alshina6", "dop853"):
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

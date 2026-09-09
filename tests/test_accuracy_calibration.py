"""Small calibration/CSV checks; no numerical integration or GPU launches."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import Benchmark_01_two_level as two
import Benchmark_02_four_level_Interferometry as four
import Benchmark_03_accuracy_timestep_sweep as accuracy
import Benchmark_accuracy_calibration as calibration
import Benchmark_01_02_plot_from_csv as plotting
from Benchmark_full_tools import save_benchmark_csv


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.settings = accuracy.user_settings()
        self.settings.update(problem="two_level", grid_side_dimension=2,
                             target_solvers=["gqis_rk4", "gqis_dop853", "julia_gpu_fp32_fopt"],
                             simulation_periods=2.0, target_base_steps_per_period=2048)
        self.path = self.root / "test_optimal_dividers.json"
        self.payload = {
            "format": "gqis_accuracy_dividers_v1", "problem": "two_level",
            "comparison_output": "mean", "target_base_steps_per_period": 2048,
            "solver_dividers": {"gqis_rk4": 8, "gqis_dop853": 20},
            "details": {"gqis_dop853": {"steps_per_period": 102}},
            "benchmark_settings": self.settings,
        }
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")

    def test_exact_step_density_and_fallbacks(self):
        loaded = calibration.load_calibration(self.path, "two_level")
        cfg = accuracy._accuracy_config(self.settings, 256, reference=False)
        self.assertEqual(calibration.calibrated_config("gqis_rk4", cfg, loaded).solver_steps_per_period, 256)
        dop = calibration.calibrated_config("gqis_dop853", cfg, loaded)
        self.assertEqual((dop.solver_steps_per_period, dop.num_steps), (102, 204))
        julia = calibration.calibrated_config("julia_gpu_fp32_fopt", cfg, loaded)
        self.assertEqual(julia.solver_steps_per_period, 2048)
        self.assertEqual(calibration.calibrated_config("qutip_cpu", cfg, loaded).solver_steps_per_period, 205)

    def test_legacy_manifest_and_problem_checks(self):
        del self.payload["benchmark_settings"]
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")
        profile = {**two.user_settings(), "Delta": 2.5, "simulation_periods": 30.0}
        with patch.object(two, "user_settings", return_value=profile):
            loaded = calibration.load_calibration(self.path, "two_level")
        self.assertEqual(loaded["settings"]["two_level_parameters"]["delta"], 2.5)
        self.assertEqual(loaded["settings"]["simulation_periods"], 30.0)
        self.assertEqual(loaded["base_steps"], 2048)
        companion = self.root / "test_settings.json"
        companion.write_text(json.dumps({"format": "gqis_benchmark_03_settings_v1",
                                         "settings": self.settings}), encoding="utf-8")
        self.assertEqual(calibration.load_calibration(self.path, "two_level")["base_steps"], 2048)
        self.assertEqual(calibration.load_calibration(self.path, "two_level")["settings"]["simulation_periods"], 2.0)
        with self.assertRaises(ValueError):
            calibration.load_calibration(self.path, "four_level")

    def test_export_selects_fastest_accepted_not_largest_divider(self):
        rows = [dict(target_solver="gqis_rk4", acceptable="yes", requested_step_count_divider=d,
                     target_steps_per_period=spp, target_calculation_time_s=t, rms=1e-5, max_abs=1e-4)
                for d, spp, t in [(4, 512, 0.3), (8, 256, 0.2), (16, 128, 0.25)]]
        accuracy._save_optimal_dividers(rows, self.settings, self.path)
        payload = json.loads(self.path.read_text())
        self.assertEqual(payload["solver_dividers"]["gqis_rk4"], 8)
        self.assertEqual(payload["details"]["gqis_rk4"]["time_s"], 0.2)
        self.assertEqual(payload["benchmark_settings"]["simulation_periods"], 2.0)

    def test_both_entry_points_accept_calibration_and_variants(self):
        for module, problem in [(two, "two_level"), (four, "four_level")]:
            with self.subTest(problem=problem), patch("sys.argv", [module.__name__, "full_benchmark",
                    "--accuracy-dividers-file", str(self.path), "--full-solvers",
                    "gqis_dop853,julia_gpu_fp32_fopt", "--no-plot"]), \
                    patch.object(calibration, "run_calibrated_sweep") as run:
                module.main()
                self.assertEqual(run.call_args.kwargs["problem"], problem)
                self.assertEqual(run.call_args.kwargs["solvers"], "gqis_dop853,julia_gpu_fp32_fopt")

    def test_grid_loop_uses_shared_solvers_without_running_them(self):
        for problem in ["two_level", "four_level"]:
            self.settings["problem"] = self.payload["problem"] = problem
            self.path.write_text(json.dumps(self.payload), encoding="utf-8")
            with patch.object(accuracy, "_warm_gqis", return_value=0.1), \
                    patch.object(two, "run_solver", return_value=(None, 0.2, None)), \
                    patch.object(two, "run_full_solver_with_timeout", return_value=0.3), \
                    patch.object(two, "plot_full_benchmark"), \
                    patch.object(two, "collect_equipment_info", return_value={}), \
                    patch.object(two, "print_equipment_info"):
                rows = calibration.run_calibrated_sweep(
                    self.path, problem=problem, min_side=2, max_side=4, time_limit=10,
                    output_filename=str(self.root / problem), julia_cmd="julia", show_plot=False)
            self.assertEqual(len(rows), 6)
            _, metadata = plotting._read_benchmark_csv(self.root / f"{problem}.csv")
            self.assertEqual(metadata["system_levels"], "4" if problem == "four_level" else "2")
            self.assertIn("gqis_dop853:204", metadata["solver_steps_per_trajectory_by_solver"])

    def test_estimated_gaps_use_measured_neighbors_only(self):
        rows = [plotting._point_row(dict(solver="qutip_cpu", side_dimension=s,
                                         time_s=t, status=status))
                for s, t, status in [(256, 10, "measured"), (512, 1000, "measured"),
                                     (1024, 999999, "extrapolated"), (2048, 16000, "measured"),
                                     (4096, 999999, "extrapolated")]]
        plotting._refresh_extrapolated(rows)
        self.assertAlmostEqual(rows[2]["time_s"], 4000)
        self.assertAlmostEqual(rows[4]["time_s"], 64000)
        self.assertEqual([rows[i]["time_s"] for i in (0, 1, 3)], [10, 1000, 16000])
        self.assertEqual(rows[2]["status"], "extrapolated")

    def test_manual_point_replaces_extrapolation_and_saves_csv(self):
        source = self.root / "input.csv"
        old = plotting._point_row(dict(solver="qutip_cpu", side_dimension=2048,
                                        time_s=60000, status="extrapolated"))
        save_benchmark_csv([old], source, metadata={"system_levels": "2"})
        original = source.read_bytes()
        options = {**plotting.user_settings(), "csv_file": str(source),
                   "save_merged_csv": True, "show_plot": False,
                   "additional_measured_points": [{"solver": "qutip_cpu",
                       "side_dimension": 2048, "time_s": 43747.3}]}
        with patch.object(plotting, "plot_benchmark"):
            plotting.generate_plot(options)
        rows, _ = plotting._read_benchmark_csv(self.root / "input_merged.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["time_s"], 43747.3)
        self.assertEqual(rows[0]["status"], "measured")
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

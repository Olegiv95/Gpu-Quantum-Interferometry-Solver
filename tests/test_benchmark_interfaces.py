from argparse import Namespace
from itertools import combinations
import sys

import numpy as np
import pytest
import sympy as sp

import Benchmark_01_two_level as benchmark_01
import Benchmark_02_four_level_Interferometry as benchmark_02
from Benchmark_full_tools import (collect_equipment_info, extrapolate_loglog,
                                  should_extrapolate_next,
                                  sympy_to_julia_fp32,
                                  )

EXPECTED_SOLVERS = {"gpu", "python_cpu", "python_ode_cpu", "qutip_cpu", "julia_gpu"}


def test_two_level_diff_mode_accepts_every_solver_pair():
    assert set(benchmark_01.SOLVERS) == EXPECTED_SOLVERS
    for solver, solver_b in combinations(benchmark_01.SOLVERS, 2):
        args = Namespace(solver=solver, solver_b=solver_b)
        assert set(benchmark_01.selected_solvers("diff", args)) == {solver, solver_b}


def test_four_level_diff_mode_accepts_every_solver_pair():
    assert benchmark_02.SOLVER_NAMES == EXPECTED_SOLVERS
    for solver, solver_b in combinations(sorted(benchmark_02.SOLVER_NAMES), 2):
        assert benchmark_02.selected_solvers("diff", solver, solver_b) == {solver, solver_b}


def test_two_level_mode_shorthands():
    args = Namespace(mode="single", solver="gpu", mode_or_solver="qutip_cpu")
    assert benchmark_01.normalize_mode_and_solver(args) == ("single", "qutip_cpu")

    args.mode_or_solver = "full_benchmark"
    assert benchmark_01.normalize_mode_and_solver(args) == ("full_benchmark", "gpu")


def test_four_level_mode_shorthands():
    assert benchmark_02.normalize_mode_solver("qutip_cpu", "gpu") == ("single", "qutip_cpu", )
    assert benchmark_02.normalize_mode_solver("full_benchmark", "gpu") == (
        "full_benchmark", "gpu")


def test_benchmark_metadata_records_numerical_package_versions():
    metadata = collect_equipment_info()
    assert metadata["python"]
    assert metadata["gqis"] == "0.1.0"
    assert metadata["numpy"]
    assert metadata["sympy"]
    assert metadata["matplotlib"]


def test_benchmark_settings_use_solver_neutral_step_name():
    for module in (benchmark_01, benchmark_02):
        settings = module.user_settings()
        assert "solver_steps_per_period" in settings
        assert "rk4_steps_per_period" not in settings
        assert "solver_a" not in settings


def test_benchmark_settings_use_public_release_names():
    expected = {"Delta", "w/Delta", "eps_max/w", "A_max/w", "bench_min_side_size",
                "bench_max_side_size", "bench_solver_time_limit", "Output_filename"}
    obsolete = {"delta", "w_over_delta", "eps_max_factor", "A_max_factor", "full_min_side",
                "full_max_side", "full_time_limit", "full_extrapolation_points",
                "full_output_stem"}
    for module in (benchmark_01, benchmark_02):
        settings = module.user_settings()
        regime_settings = settings.get(settings.get("regime", ""), {})
        available = set(settings) | set(regime_settings)
        assert expected <= available
        assert obsolete.isdisjoint(available)


def test_benchmark_limit_cli_names(monkeypatch):
    cli = ["benchmark", "--bench-min-side-size", "32", "--bench-max-side-size", "1024",
           "--bench-solver-time-limit", "60", "--output-filename", "timing_result"]
    for module in (benchmark_01, benchmark_02):
        monkeypatch.setattr(sys, "argv", cli)
        args = module.parse_args()
        assert args.bench_min_side_size == 32
        assert args.bench_max_side_size == 1024
        assert args.bench_solver_time_limit == 60.0
        assert args.output_filename == "timing_result"


def test_four_level_normalized_ranges_preserve_original_regimes():
    settings = benchmark_02.user_settings()
    delta_abs = settings["wq1"] * settings["Delta"]
    for regime, expected_w, amplitude_factor in (("wd500", 0.5, 1.15),
                                                  ("wd1500", 1.5, 1.15 * 1.3)):
        values = settings[regime]
        w = values["w/Delta"] * delta_abs
        assert w == pytest.approx(expected_w)
        assert values["eps_max/w"] * w == pytest.approx(2.09916 * settings["wq1"])
        expected_A_max = 2.234042553191489 * amplitude_factor * settings["wq1"]
        assert values["A_max/w"] * w == pytest.approx(expected_A_max)


def test_legacy_solver_a_cli_alias_maps_to_solver(monkeypatch):
    for module in (benchmark_01, benchmark_02):
        monkeypatch.setattr(sys, "argv", [module.__name__, "--solver-a", "qutip_cpu"])
        args = module.parse_args()
        assert args.solver == "qutip_cpu"
        assert not hasattr(args, "solver_a")


def test_loglog_extrapolation_averages_last_three_measured_points():
    history = [(8, 1.0e6), (16, 10.0), (32, 40.0), (64, 60.0)]
    local_slopes = [np.log10(40.0 / 10.0) / np.log10(4.0),
                    np.log10(60.0 / 40.0) / np.log10(4.0)]
    expected = 60.0 * 4.0**np.mean(local_slopes)

    assert extrapolate_loglog(history, 128, slope_points=3) == pytest.approx(expected)


def test_loglog_extrapolation_defaults_to_last_two_measured_points():
    history = [(16, 10.0), (32, 30.0), (64, 120.0)]
    assert extrapolate_loglog(history, 128) == pytest.approx(480.0)


def test_loglog_extrapolation_fallbacks():
    assert np.isnan(extrapolate_loglog([], 32))
    assert extrapolate_loglog([(16, 2.0)], 32) == pytest.approx(8.0)


def test_predictive_stop_requires_stable_linear_scaling_near_time_limit():
    stable = [(128, 39.0), (256, 155.0)]
    assert should_extrapolate_next(stable, 300.0)
    assert not should_extrapolate_next(stable, 400.0)

    julia_outlier = [(1024, 55.7), (2048, 18.2)]
    assert not should_extrapolate_next(julia_outlier, 30.0)
    assert not should_extrapolate_next(stable[:1], 100.0)
    assert not should_extrapolate_next([(128, 10.0), (256, 35.0)], 60.0)
    assert should_extrapolate_next([(128, 10.0), (256, 100.0)], 150.0)


def test_julia_scalar_codegen_preserves_float32_and_removes_broadcast_operators():
    x = sp.Symbol("x")
    code = sympy_to_julia_fp32((x + sp.Float("0.5"))**2 + sp.Float("1e-5") * x)

    assert "0.5f0" in code
    assert "1.0f-5" in code
    assert not any(operator in code for operator in (".^", ".*", "./"))


def test_four_level_julia_helper_uses_common_subexpressions(tmp_path):
    rho0, rho1, t = sp.symbols("rho0 rho1 t", real=True)
    shared = sp.sqrt((sp.cos(sp.Float("0.5") * t) + 1)**2 + sp.Float("2.0"))
    helper = tmp_path / "compact_rhs.jl"

    benchmark_02._write_julia_helper(helper, [shared * rho0, shared * rho1],
                                     rho0 + rho1, [rho0, rho1])
    code = helper.read_text(encoding="utf-8")

    assert code.count("sqrt(") == 1
    assert code.count("cos(") == 1
    assert "0.5f0" in code
    assert not any(operator in code for operator in (".^", ".*", "./"))

"""Use Benchmark 03's calibrated solvers in Benchmark 01/02 grid sweeps."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np

from Benchmark_full_tools import accuracy_divider_for_solver, load_accuracy_dividers


def load_calibration(filename, problem, settings_file=None):
    from Benchmark_03_accuracy_timestep_sweep import _resolve_settings_file

    path = _resolve_settings_file(filename)
    dividers = load_accuracy_dividers(path, expected_problem=problem)
    payload = json.loads(path.read_text(encoding="utf-8"))
    settings = payload.get("benchmark_settings")
    if settings_file or settings is None:
        companion = (settings_file or
                     path.with_name(path.name.replace("_optimal_dividers.json", "_settings.json")))
        if not settings_file and (companion == path or not companion.is_file()):
            if problem != "two_level":
                raise ValueError("This legacy divider JSON has no problem settings. Supply "
                                 "--accuracy-settings with the matching Benchmark 03 settings JSON.")
            from Benchmark_01_two_level import user_settings as two_level_settings
            from Benchmark_03_accuracy_timestep_sweep import user_settings as accuracy_settings
            profile = two_level_settings()
            settings = accuracy_settings()
            settings.update(problem="two_level", comparison_output="mean",
                            simulation_periods=profile["simulation_periods"],
                            target_base_steps_per_period=int(payload["target_base_steps_per_period"]),
                            target_solvers=list(dividers),
                            averaging_skip_fraction=profile["averaging_skip_fraction"],
                            gpu_precision=profile["gpu_precision"], cpu_precision=profile["cpu_precision"])
            settings["two_level_parameters"] = {
                "delta": profile["Delta"], "w_over_delta": profile["w/Delta"],
                "gamma_phi_per_period": profile["gamma_phi_per_T"],
                "gamma1_per_period": profile["gamma1_per_T"],
                "eps_max_over_w": profile["eps_max/w"],
                "amplitude_max_over_w": profile["A_max/w"],
            }
            print("Legacy calibration: using Benchmark 01's two-level profile "
                  "and Benchmark 03's adaptive settings.")
        else:
            manifest = json.loads(_resolve_settings_file(companion).read_text(encoding="utf-8"))
            if manifest.get("format") != "gqis_benchmark_03_settings_v1":
                raise ValueError("Expected a Benchmark 03 settings manifest.")
            settings = manifest["settings"]
    if settings["problem"] != problem or settings["comparison_output"] != "mean":
        raise ValueError("Calibration settings must match the problem and mean observable.")
    base = int(payload["target_base_steps_per_period"])
    if base <= 0 or base != int(settings["target_base_steps_per_period"]):
        raise ValueError("Calibration and settings have different base step densities.")
    return {"path": str(path), "settings": settings, "dividers": dividers,
            "details": payload.get("details", {}), "base_steps": base,
            "selection": payload.get("selection", "legacy divider selection")}


def calibrated_config(name, cfg, calibration):
    divider = accuracy_divider_for_solver(calibration["dividers"], name)
    details = calibration["details"].get(name, {})
    spp = int(details.get("steps_per_period", max(1, round(calibration["base_steps"] / divider))))
    if spp <= 0:
        raise ValueError(f"Invalid calibrated steps per period for {name}: {spp}")
    steps = max(1, round(cfg.tr * spp))
    tlist = np.linspace(0.0, cfg.tr * 2.0 * np.pi / cfg.w, steps + 1, dtype=cfg.tlist.dtype)
    return replace(cfg, solver_steps_per_period=spp, tlist=tlist)


def run_calibrated_sweep(filename, *, problem, settings_file=None, solvers=None,
                         min_side, max_side, time_limit, output_filename,
                         julia_cmd, show_plot):
    import Benchmark_03_accuracy_timestep_sweep as accuracy
    from Benchmark_01_two_level import run_full_benchmark

    calibration = load_calibration(filename, problem, settings_file)
    settings = calibration["settings"]
    names = solvers or settings["target_solvers"]
    names = names.split(",") if isinstance(names, str) else names
    aliases = {"gpu": "gqis_rk4", "julia_gpu": "julia_gpu_fp32"}
    names = tuple(dict.fromkeys(aliases.get(name.strip(), name.strip()) for name in names))
    accuracy._solver_names(names)
    cfg_settings = {**settings, "grid_side_dimension": min_side}
    cfg = accuracy._accuracy_config(cfg_settings, calibration["base_steps"], reference=False)
    cfg = replace(cfg, warmup_time=float(settings.get("averaging_skip_fraction", 0.0)),
                  gpu_precision=settings.get("gpu_precision", cfg.gpu_precision),
                  cpu_precision=settings.get("cpu_precision", cfg.cpu_precision))
    args = SimpleNamespace(
        calibration=calibration, accuracy_dividers=calibration["dividers"],
        julia_cmd=julia_cmd, bench_min_side_size=min_side, bench_max_side_size=max_side,
        bench_solver_time_limit=time_limit, output_filename=output_filename,
        no_plot=not show_plot,
        python_cpu_spp_divider=accuracy_divider_for_solver(calibration["dividers"], "python_cpu"),
        python_ode_cpu_spp_divider=accuracy_divider_for_solver(calibration["dividers"], "python_ode_cpu"),
        qutip_cpu_spp_divider=accuracy_divider_for_solver(calibration["dividers"], "qutip_cpu"))
    print(f"Loaded calibration: {calibration['path']}")
    print("Using the loaded model, duration, precision and adaptive settings.")
    for name in names:
        actual = calibrated_config(name, cfg, calibration)
        source = "calibrated" if name in calibration["dividers"] else "fallback (not accuracy-validated)"
        print(f"{name}: {actual.solver_steps_per_period} steps/output samples per period; {source}")
    return run_full_benchmark(cfg, args, names)

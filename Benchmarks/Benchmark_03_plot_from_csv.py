"""Rebuild a Benchmark 03 figure from its saved metrics CSV."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import numpy as np

from Benchmark_03_accuracy_timestep_sweep import (ACCURACY_SOLVER_SET,
                                                   _plot_accuracy_sweep)
from Benchmark_full_tools import benchmark_output_path


def _resolve_file(filename) -> Path:
    requested = Path(filename).expanduser()
    if requested.is_absolute() and requested.is_file():
        return requested
    script_dir = Path(__file__).resolve().parent
    candidates = (Path.cwd() / requested, script_dir / "results" / requested,
                  script_dir / requested,
                  script_dir.parent / requested)
    path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Benchmark 03 CSV not found: {filename}")
    return path


def _unique(rows: list[dict], field: str) -> tuple:
    return tuple(dict.fromkeys(row[field] for row in rows))


def _stored_float(rows: list[dict], field: str, fallback=None) -> float:
    values = [row.get(field, "") for row in rows]
    value = next((value for value in values if value not in {"", None}), fallback)
    if value is None:
        raise ValueError(f"CSV does not contain {field}; set its legacy fallback in user_settings().")
    return float(value)


def main() -> None:
    options = user_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="input metrics CSV; overrides csv_file")
    parser.add_argument("--solvers", nargs="+", help="solver names to include, in legend order")
    parser.add_argument("--output", help="output PNG path; overrides output_file")
    parser.add_argument("--no-show", action="store_true", help="save without opening a plot window")
    args = parser.parse_args()
    if args.csv:
        options["csv_file"] = args.csv
    if args.solvers:
        options["include_solvers"] = args.solvers
    if args.output:
        options["output_file"] = str(Path(args.output).expanduser().resolve())
    if args.no_show:
        options["show_plot"] = False
    csv_path = _resolve_file(options["csv_file"])
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Benchmark 03 CSV contains no data: {csv_path}")

    available = _unique(rows, "target_solver")
    selected = tuple(options["include_solvers"]) or available
    unknown = [solver for solver in selected if solver not in ACCURACY_SOLVER_SET]
    missing = [solver for solver in selected if solver not in available]
    if unknown:
        raise ValueError(f"Unknown solver names: {', '.join(unknown)}")
    if missing:
        print(f"Skipping solvers not yet present in CSV: {', '.join(missing)}")
        selected = tuple(solver for solver in selected if solver in available)
    if not selected:
        raise ValueError("None of the selected solvers are present in the CSV yet.")
    rows = [row for row in rows if row["target_solver"] in selected]

    dividers = tuple(float(value) for value in _unique(rows, "requested_step_count_divider"))
    divider_one = next((row for row in rows
                        if np.isclose(float(row["requested_step_count_divider"]), 1.0)), rows[0])
    reference_time = _stored_float(rows, "reference_calculation_time_s")
    preparation_time = _stored_float(rows, "gqis_preparation_time_s", np.nan)
    plot_settings = {
        "target_solvers": selected,
        "step_count_dividers": dividers,
        "reference_solver": rows[0]["reference_solver"],
        "reference_steps_per_period": int(float(rows[0]["reference_steps_per_period"])),
        "target_base_steps_per_period": int(float(divider_one["target_steps_per_period"])),
        "acceptable_rms": _stored_float(
            rows, "acceptable_rms_limit", options["legacy_acceptable_rms"]),
        "acceptable_max_abs": _stored_float(
            rows, "acceptable_max_abs_limit", options["legacy_acceptable_max_abs"]),
        "grid_side_dimension": int(float(rows[0]["grid_side_dimension"])),
        "simulation_periods": _stored_float(
            rows, "simulation_periods", float(rows[0]["reference_total_steps"])
            / float(rows[0]["reference_steps_per_period"])),
        "problem": rows[0].get("problem", "two_level"),
        "comparison_output": rows[0].get("comparison_output", "mean") or "mean",
        "show_results_table": bool(options["show_results_table"]),
        "table_times_only": bool(options["table_times_only"]),
        "show_best_summary_row": bool(options["show_best_summary_row"]),
        "divider_axis_scale": options["divider_axis_scale"],
        "time_grid_axis": options["time_grid_axis"],
        "divider_axis_labels": options["divider_axis_labels"],
        "show_plot": bool(options["show_plot"]),
    }
    output_file = options["output_file"]
    output_path = (benchmark_output_path(output_file, script_dir=Path(__file__).resolve().parent)
                   if output_file else
                   csv_path.with_name(f"{csv_path.stem}_plot.png"))
    _plot_accuracy_sweep(rows, plot_settings, reference_time, preparation_time, output_path)
    print(f"Included solvers: {', '.join(selected)}")
    print(f"Saved figure: {output_path}")


def user_settings() -> dict:
    """User-editable CSV, solver-selection, and figure settings."""
    return {
        "csv_file": "results\Benchmark_03_four_level_accuracy_timestep_sweep_metrics.csv",
        # Empty tuple includes every solver in the CSV. Otherwise list only the
        # curves wanted in the regenerated figure, in the desired legend order.
        #"include_solvers": (),
        # Example:
        "include_solvers": (),#"gqis_rk4", "gqis_tsit5", "gqis_dop853", "qutip_cpu","julia_gpu_fp32_fopt"),
        "show_results_table": False,  # True restores the full table below the graphs
        "table_times_only": False,  # True keeps only step information and solver times
        "show_best_summary_row": True,  # add dark-olive solver: steps/period | time cells
        "time_grid_axis": "steps_per_period",  # choices: "steps_per_period" or "divider"
        "divider_axis_scale": "log2",  # choices: "log2" or "equidistant"; applies to either quantity
        "divider_axis_labels": "powers_of_two",  # choices: "powers_of_two" or "all"
        "show_plot": True,
        "output_file": None,  # None writes <csv filename>_plot.png beside the CSV
        # New CSV files store both limits. These fallbacks are used only for
        # older Benchmark 03 files that do not contain the limit columns.
        "legacy_acceptable_rms": 1e-3,
        "legacy_acceptable_max_abs": 1e-2,
    }


if __name__ == "__main__":
    main()

"""Regenerate Benchmark 01 or 02 scaling figures without running any solver."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from Benchmark_full_tools import (benchmark_output_path, extrapolate_loglog,
                                  plot_benchmark, save_benchmark_csv)


def _resolve_file(filename: str | Path) -> Path:
    requested = Path(filename).expanduser()
    script_dir = Path(__file__).resolve().parent
    candidates = ([requested] if requested.is_absolute() else
                  [Path.cwd() / requested, script_dir / "results" / requested,
                   script_dir / requested, script_dir.parent / requested])
    path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Benchmark CSV not found: {filename}")
    return path


def _read_benchmark_csv(path: Path) -> tuple[list[dict], dict[str, str]]:
    metadata: dict[str, str] = {}
    rows: list[dict] = []
    header = None
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for values in csv.reader(stream):
            if not values:
                continue
            if values[0].startswith("# "):
                metadata[values[0][2:]] = values[1] if len(values) > 1 else ""
                continue
            if header is None:
                header = values
                continue
            raw = dict(zip(header, values))
            side = int(raw["side_dimension"])
            rows.append({
                "side_dimension": side,
                "number_of_simulations": int(raw.get("number_of_simulations") or side * side),
                "solver": raw["solver"],
                "time_s": float(raw["time_s"]) if raw.get("time_s") else np.nan,
                "prep_s": float(raw["prep_s"]) if raw.get("prep_s") else np.nan,
                "calc_s": float(raw["calc_s"]) if raw.get("calc_s") else np.nan,
                "status": raw.get("status") or "measured",
            })
    if header is None or not rows:
        raise ValueError(f"Benchmark CSV contains no timing rows: {path}")
    return rows, metadata


def _point_row(point: dict) -> dict:
    side = int(point["side_dimension"])
    elapsed = float(point["time_s"])
    if side <= 0 or not np.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("Additional points require positive side_dimension and time_s values.")
    return {
        "side_dimension": side, "number_of_simulations": side * side,
        "solver": str(point["solver"]), "time_s": elapsed,
        "prep_s": float(point.get("prep_s", np.nan)),
        "calc_s": float(point.get("calc_s", np.nan)),
        "status": str(point.get("status", "measured")),
    }


def _merge_rows(groups: list[list[dict]]) -> list[dict]:
    merged = {}
    for group in groups:
        for row in group:
            merged[(row["solver"], int(row["side_dimension"]))] = row
    return list(merged.values())


def _refresh_extrapolated(rows: list[dict]) -> None:
    for solver in dict.fromkeys(row["solver"] for row in rows):
        solver_rows = sorted((row for row in rows if row["solver"] == solver),
                             key=lambda row: int(row["side_dimension"]))
        history = [(int(row["side_dimension"]), float(row["time_s"]))
                   for row in solver_rows if row["status"] == "measured"
                   and np.isfinite(float(row["time_s"])) and float(row["time_s"]) > 0.0]
        for row in solver_rows:
            if row["status"] != "extrapolated":
                continue
            preceding = [(side, elapsed) for side, elapsed in history
                         if side < int(row["side_dimension"])]
            following = [(side, elapsed) for side, elapsed in history
                         if side > int(row["side_dimension"])]
            # Interpolate gaps between measured neighbors in log-log space.
            anchors = ([preceding[-1], following[0]] if preceding and following else preceding)
            row["time_s"] = extrapolate_loglog(
                anchors, int(row["side_dimension"]), slope_points=2)


def generate_plot(options: dict | None = None) -> Path:
    options = user_settings() if options is None else options
    csv_path = _resolve_file(options["csv_file"])
    rows, metadata = _read_benchmark_csv(csv_path)
    groups = [rows]
    for filename in options.get("additional_csv_files", ()):
        extra_rows, extra_metadata = _read_benchmark_csv(_resolve_file(filename))
        levels = {metadata.get("system_levels"), extra_metadata.get("system_levels")} - {None, ""}
        if len(levels) > 1:
            raise ValueError(f"Cannot combine benchmark files for different system levels: {levels}")
        groups.append(extra_rows)
    groups.append([_point_row(point) for point in options.get("additional_measured_points", ())])
    rows = _merge_rows(groups)
    if options.get("refresh_extrapolated_points", True):
        _refresh_extrapolated(rows)

    available = tuple(dict.fromkeys(row["solver"] for row in rows))
    selected = tuple(options.get("include_solvers", ())) or available
    unknown = [solver for solver in selected if solver not in available]
    if unknown:
        raise ValueError(f"Solvers not present in the input data: {', '.join(unknown)}")
    rows = [row for row in rows if row["solver"] in selected]

    levels = metadata.get("system_levels", "unknown")
    title = (options.get("title") or
             f"Calculation time scaling for different numerical approaches ({levels}-level system)")
    reference_lines = list(options.get("additional_reference_lines", ()))

    output_file = options.get("output_file")
    output_path = (benchmark_output_path(output_file, script_dir=Path(__file__).resolve().parent)
                   if output_file else csv_path.with_name(f"{csv_path.stem}_plot.png"))
    if options.get("save_merged_csv", False):
        merged_file = options.get("merged_csv_file")
        merged_path = (benchmark_output_path(merged_file, script_dir=Path(__file__).resolve().parent)
                       if merged_file else csv_path.with_name(f"{csv_path.stem}_merged.csv"))
        inputs = [csv_path, *(_resolve_file(p) for p in options.get("additional_csv_files", ()))]
        if merged_path.resolve() in inputs:
            raise ValueError("Choose a merged CSV filename different from the input files.")
        metadata = {**metadata, "benchmark_solvers": ",".join(selected)}
        save_benchmark_csv(rows, merged_path, metadata=metadata)
    plot_benchmark(rows, selected, output_path, title=title,
                   show=bool(options.get("show_plot", True)), metadata=metadata,
                   reference_lines=reference_lines,
                   show_startup_times=bool(options.get("show_startup_times", True)),
                   show_gpu_preparation=bool(options.get("show_gpu_preparation", True)),
                   preparation_overrides_s=options.get("preparation_overrides_s"),
                   solver_labels=options.get("solver_labels"),
                   show_side_dimensions=bool(options.get("show_side_dimensions", True)),
                   show_table=bool(options.get("show_results_table", True)),
                   dpi=int(options.get("dpi", 300)))
    print(f"Included solvers: {', '.join(selected)}")
    return output_path


def user_settings() -> dict:
    """User-editable data-selection and figure settings."""
    return {
        # Select either a Benchmark 01 or Benchmark 02 full-benchmark CSV.
        "csv_file": "Benchmark_02_full_benchmark.csv",
        # Empty includes every solver. Otherwise list only the desired curves.
        "include_solvers": ("gqis_rk4","julia_gpu_fp32_fopt","qutip_cpu"),
        # Display names only; CSV identifiers and solver selection stay unchanged.
        "solver_labels": {"julia_gpu_fp32_fopt": "Julia(GPU FP32)", "qutip_cpu": "QuTiP(CPU)"},
        # Use "julia_gpu_fp32_fopt": "Julia" above for an even shorter label.
        # Later files and inline points replace an existing point with the same
        # solver and side dimension. The original CSV is never modified.
        "additional_csv_files": (),
        "additional_measured_points": (),
        # Add only points measured with the same model and solver time-grid
        # settings as the primary CSV. See BENCHMARKS.md for a manual-point example.
        "save_merged_csv": False,  # True saves the data including added points
        "merged_csv_file": None,  # None writes <CSV stem>_merged.csv; preserves the input
        "refresh_extrapolated_points": True,
        "show_gpu_preparation": True,
        "show_startup_times": True,  # Dotted same-color curves: solve + preparation.
        "preparation_overrides_s": {},  # Optional {solver_name: seconds} from a matching run/log.
        "show_results_table": True,
        "show_side_dimensions": True,  # Side dimension row above Simulations (side squared).
        "show_plot": True,
        "title": None,  # None builds the title from CSV metadata
        "dpi": 300,
        "additional_reference_lines": (),
        "output_file": None,  # None writes <CSV stem>_plot.png beside the CSV
    }


if __name__ == "__main__":
    generate_plot()

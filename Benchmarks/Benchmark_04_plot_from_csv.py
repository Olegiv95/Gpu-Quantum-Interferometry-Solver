"""Draw the Benchmark 04 Hilbert-space scaling figure from its saved CSV without calculating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from Benchmark_04_hilbert_space_scaling import (is_requested, parse_recalculate, read_scaling_csv,
                                                system_metadata)
from Benchmark_full_tools import add_time_reference_marks, format_equipment_label, resolve_benchmark_csv

PANEL_ORDER = ("qubit_resonator", "multi_qubit")
DRIVE_TITLES = {"single": "qubit 0 driven", "all": "all qubits driven"}


DEFAULT_LABELS = {"julia_gpu_fp32_fopt": "DiffEqGPU(Tsit5, GQIS RHS)",
                  "dynamiqs_tsit5": "dynamiqs(Tsit5, WSL)", "dynamiqs_rouchon3": "dynamiqs(Rouchon3, WSL)"}
HEADER_ROW_SCALE = 1.7  # table header height in data rows (two corner labels)
# Settings an additional CSV must share with the main CSV to be drawn in the same panel.
COMPARED_KEYS = ("nx", "ny", "simulation_periods", "topology", "drive", "physics")


def _solver_label(solver: str, labels: dict) -> str:
    return labels.get(solver, DEFAULT_LABELS.get(solver,
                                                 f"GQIS({solver.removeprefix('gqis_').upper()})"))


def _panel_title(system: str, metadata: dict) -> str:
    if system == "qubit_resonator":
        title = "Qubit-resonator with Nph photon states"
    else:
        title = "Coupled qubits"
        details = [f"{metadata['topology']} ZZ coupling" if metadata.get("topology") else "",
                   DRIVE_TITLES.get(metadata.get("drive", ""), metadata.get("drive", ""))]
        details = ", ".join(detail for detail in details if detail)
        title = f"{title} ({details})" if details else title
    nx, ny = metadata.get("nx", "?"), metadata.get("ny", "?")
    grid = f"{nx} x {ny} grid"
    if str(nx).isdigit() and str(ny).isdigit():
        mantissa, exponent = f"{int(nx) * int(ny):.3e}".split("e")
        grid = f"{grid} = ${mantissa}\\times10^{{{int(exponent)}}}$ systems"
    return f"{title}\n{grid}, {metadata.get('simulation_periods', '?')} drive periods"


def _load_panels(csv_path: Path) -> dict:
    """Return {system: (rows, metadata)}, one panel per system saved in the CSV."""
    rows, metadata = read_scaling_csv(csv_path)
    panels = {system: (rows[system], system_metadata(metadata, system))
              for system in PANEL_ORDER if rows.get(system)}
    if not panels:
        raise ValueError(f"No points saved in {csv_path}")
    return panels


def _add_csv(panels: dict, csv_path: Path) -> None:
    """Add another library's points (same CSV format) to the panels of the same systems."""
    rows, metadata = read_scaling_csv(csv_path)
    for system, extra_rows in rows.items():
        if system not in panels:
            continue
        own_rows, own_metadata = panels[system]
        extra_metadata = system_metadata(metadata, system)
        differing = [key for key in COMPARED_KEYS if extra_metadata.get(key, "") != own_metadata.get(key, "")]
        if differing:
            raise ValueError(f"{csv_path.name}: {system} was measured with other settings "
                             f"({', '.join(differing)}) than the main CSV.")
        own_rows.update(extra_rows)
        own_metadata["solvers"] = f"{own_metadata.get('solvers', '')},{extra_metadata.get('solvers', '')}"


def generate_plot(options: dict | None = None) -> Path:
    options = user_settings() if options is None else options
    script_dir = Path(__file__).resolve().parent
    csv_path = resolve_benchmark_csv(options["csv_file"], script_dir=script_dir)
    panels = _load_panels(csv_path)
    for name in options.get("additional_csv_files") or ():
        try:
            _add_csv(panels, resolve_benchmark_csv(name, script_dir=script_dir))
        except FileNotFoundError:
            print(f"Additional CSV not found, skipped: {name}")
    labels = dict(options.get("solver_labels") or {})
    show_total = bool(options.get("show_total_times", True))
    # Point selection: same syntax as the benchmark's recalculate list; applied to both panels.
    excluded = parse_recalculate(options.get("exclude_points", ()))

    solvers = []
    for rows, metadata in panels.values():
        saved = [s for s in metadata.get("solvers", "").split(",") if s]
        solvers = list(dict.fromkeys([*solvers, *saved, *(key[0] for key in rows)]))
    if options.get("include_solvers"):
        solvers = [s for s in options["include_solvers"] if s in solvers]
    # tab10 without its grey (index 7): grey is reserved for the line-style samples in the legend.
    palette = [plt.get_cmap("tab10")(index) for index in range(10) if index != 7]
    # Fixed colors keep each solver's color when the solver selection or run order changes.
    fixed = dict(options.get("solver_colors") or {})
    free = [c for c in palette if not any(np.allclose(c, plt.matplotlib.colors.to_rgba(f)) for f in fixed.values())]
    others = iter(free * (len(solvers) // max(len(free), 1) + 1))
    colors = {solver: fixed.get(solver) or next(others) for solver in solvers}

    show_table = bool(options.get("show_results_table", True))
    # 0.2 inch per one-line data row, a taller header row, 0.5 inch for the table title.
    table_height = 0.2 * (len(solvers) + HEADER_ROW_SCALE) + 0.5 if show_table else 0.0
    fig, axes = plt.subplots(2 if show_table else 1, len(panels), squeeze=False,
                             figsize=(7.5 * len(panels), 6.6 + table_height),
                             gridspec_kw={"height_ratios": [5.0, table_height]} if show_table else None)
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - 1.5 / (6.6 + table_height), bottom=0.02,
                        wspace=0.12, hspace=0.3)
    for column, (system, (rows, metadata)) in enumerate(panels.items()):
        ax = axes[0][column]
        # Only measured points are drawn; timed-out and failed points are skipped.
        measured = {key: row for key, row in rows.items()
                    if key[0] in solvers and row["status"] == "measured" and np.isfinite(row["calc_s"])
                    and not is_requested(excluded, key[0], key[1], row)}
        sizes = sorted({key[1] for key in measured})
        for solver in solvers:
            points = sorted((row for (name, _), row in measured.items() if name == solver),
                            key=lambda row: row["size"])
            if not points:
                continue
            x = [row["size"] for row in points]
            name = _solver_label(solver, labels)
            ax.plot(x, [row["calc_s"] for row in points], marker="o", markersize=5, linewidth=2.0,
                    color=colors[solver], label=name)
            if show_total:
                ax.plot(x, [row["total_s"] for row in points], linestyle=":", linewidth=1.8,
                        color=colors[solver])
        if not sizes:
            ax.text(0.5, 0.5, "no measured points", transform=ax.transAxes, ha="center")
            continue
        size_name = "Photon states Nph" if system == "qubit_resonator" else "Qubits n"
        dim_name = "2 Nph" if system == "qubit_resonator" else "2^n"
        dims = {int(row["size"]): int(row["hilbert_dim"]) for row in measured.values()}
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xlabel(size_name)
        ax.set_ylabel("Time [s]")
        ax.grid(True, which="both", alpha=0.3)
        top = ax.secondary_xaxis("top")
        top.set_xticks(sizes)
        top.set_xticklabels([str(dims[size]) for size in sizes])
        top.set_xlabel(f"Hilbert-space dimension ({dim_name})", fontsize=9)
        ax.set_title(_panel_title(system, metadata), fontsize=11)
        add_time_reference_marks(ax)
        # Grey samples explain the line styles once; solver entries show colors only.
        styles = [Line2D([], [], color="0.5", linestyle=":", linewidth=1.8, label="any color: total time")]
        styles = styles if show_total else []
        styles.append(Line2D([], [], color="0.5", marker="o", markersize=5, linewidth=2.0,
                             label="any color: calculation time"))
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=[*styles, *handles], fontsize=8, loc="lower right")
        if show_table:
            _draw_times_table(axes[1][column], measured, solvers, sizes, labels, colors,
                              "Nph" if system == "qubit_resonator" else "n", dims,
                              metadata.get("calibrate_timesteps", ""))
    # Same time axis on both panels so the systems compare directly.
    plotted = [ax for ax in axes[0] if ax.lines]
    if len(plotted) > 1:
        low = min(ax.get_ylim()[0] for ax in plotted)
        high = max(ax.get_ylim()[1] for ax in plotted)
        for ax in plotted:
            ax.set_ylim(low, high)
    if show_table:
        for ax_table in axes[1]:
            ax_table.axis("off")

    first_metadata = next(iter(panels.values()))[1]
    title = options.get("title") or "Calculation time vs Hilbert-space dimension, FP32"
    equipment = format_equipment_label(first_metadata)
    fig.suptitle(f"{title}\n{equipment}" if equipment else title, fontsize=13)

    output_file = options.get("output_file")
    output_path = (Path(output_file) if output_file
                   else csv_path.with_name(f"{csv_path.stem}_plot.png"))
    fig.savefig(output_path, dpi=int(options.get("dpi", 300)), bbox_inches="tight")
    print(f"Saved Hilbert-space scaling figure: {output_path}")
    if options.get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _draw_times_table(ax, measured: dict, solvers: list, sizes: list, labels: dict, colors: dict,
                      size_symbol: str, dims: dict, calibration: str = "") -> None:
    """First-call total times in seconds; points not measured are left as '--'."""
    used = [solver for solver in solvers if any(key[0] == solver for key in measured)]
    if not used:
        return
    def seconds(value):  # 3 significant digits, whole seconds from 1000 s (no exponent)
        return f"{value:.0f}" if value >= 1000 else f"{value:.3g}"

    cells = []
    for solver in used:
        row = [_solver_label(solver, labels)]
        for size in sizes:
            point = measured.get((solver, size))
            row.append(f"{seconds(point['total_s'])} s" if point else "--")
        cells.append(row)
    table = ax.table(cellText=cells, colLabels=[""] + [f"{size} | {dims[size]}" for size in sizes],
                     cellLoc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    font_size = 9  # same in both tables; time cells shrink below if a value would not fit
    table.set_fontsize(font_size)
    first_width = 0.22
    # Data rows fit one text line; the header row is taller for the two corner labels.
    row_height = 1.0 / (len(used) + HEADER_ROW_SCALE)
    height = HEADER_ROW_SCALE * row_height
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_height(height if row_index == 0 else row_height)
        cell.set_width(first_width if col_index == 0 else (1.0 - first_width) / len(sizes))
        if row_index > 0 and col_index == 0:
            cell.get_text().set_color(colors[used[row_index - 1]])
            cell.get_text().set_fontsize(min(font_size, 7))  # long solver names fit the column
    # Shrink the time cells when the widest time text would cross its cell borders.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    time_cells = [cell for (_, col_index), cell in table.get_celld().items() if col_index > 0]
    ratio = max(cell.get_text().get_window_extent(renderer).width / cell.get_window_extent(renderer).width
                for cell in time_cells)
    if ratio > 0.9:
        for cell in time_cells:
            cell.get_text().set_fontsize(font_size * 0.9 / ratio)
    # Corner cell split by a diagonal: row header "Solver" below it, column header "size | dim" above it.
    ax.plot([0.0, first_width], [1.0, 1.0 - height], color="black", linewidth=0.8,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.012, 1.0 - 0.88 * height, "Solver", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=font_size)
    ax.text(first_width - 0.012, 1.0 - 0.12 * height, f"{size_symbol} | dim", transform=ax.transAxes,
            ha="right", va="top", fontsize=font_size)
    title = "First call total time"
    if calibration:  # JSON settings of the step calibration saved in the CSV metadata
        limits = json.loads(calibration)
        title = (f"First-call total time for an error of the resulting interferogram of "
                 f"{_power(limits['rms_atol'])} rms, {_power(limits['max_atol'])} max")
    ax.set_title(title, fontsize=11)


def _power(value: float, digits: int = 0) -> str:
    """1e-03 -> 10^-3 as a superscript; other mantissas as m x 10^n."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    prefix = "" if float(mantissa) == 1 else rf"{mantissa}\times"
    return rf"${prefix}10^{{{int(exponent)}}}$"


def user_settings() -> dict:
    """User-editable figure settings."""
    return {
        "csv_file": "Benchmark_04_hilbert_space_scaling_calibrated.csv",  # relative: Benchmarks/results/
        # Other libraries' runs drawn in the same panels (same CSV format, grid and physics).
        "additional_csv_files": ("Benchmark_04_hilbert_space_scaling_dynamiqs_calibrated.csv",),
        # Shown solvers in legend and table order (empty: every saved solver).
        "include_solvers": ("gqis_tsit5a", "gqis_tsit5",  # adaptive and fixed-step Tsit5
                            "gqis_lserk4f", "gqis_lserk4",  # fused vs unfused low-storage RK4
                            "dynamiqs_tsit5", "julia_gpu_fp32_fopt"),  # other libraries last
        "solver_colors": {"gqis_tsit5a": "tab:pink", "dynamiqs_tsit5": "tab:orange",
                          "gqis_tsit5": "tab:brown", "julia_gpu_fp32_fopt": "tab:olive",
                          "gqis_lserk4f": "tab:red", "gqis_lserk4": "tab:blue",
                          "gqis_dop853": "tab:green", "gqis_rk4": "tab:purple"},
        "exclude_points": (),  # e.g. ("gqis_tsit5:5", "*:7", "gqis_dop853:*"); both panels
        "solver_labels": {},  # display names only, e.g. {"gqis_lserk4": "GQIS(LSERK4)"}
        "show_total_times": True,  # dotted curves: calculation + preparation (cold first call)
        "show_results_table": True,  # measured times below each panel, first-call total time in seconds
        "show_plot": True,
        "title": None,  # None uses the default title
        "dpi": 300,
        "output_file": None,  # None writes <CSV stem>_plot.png beside the CSV
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", dest="csv_file", help="Benchmark 04 CSV")
    parser.add_argument("--add-csv", dest="additional_csv_files", nargs="*",
                        help="other libraries' CSVs to draw in the same panels (none: only the main CSV)")
    parser.add_argument("--output", dest="output_file", help="Output PNG path")
    parser.add_argument("--solvers", nargs="+", help="Solver identifiers to display")
    parser.add_argument("--exclude", nargs="+", help="solver:size, *:size or solver:*")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a window")
    args = parser.parse_args()
    options = user_settings()
    if args.csv_file:
        options["csv_file"] = args.csv_file
    if args.additional_csv_files is not None:
        options["additional_csv_files"] = tuple(args.additional_csv_files)
    if args.output_file:
        options["output_file"] = args.output_file
    if args.solvers:
        options["include_solvers"] = tuple(args.solvers)
    if args.exclude:
        options["exclude_points"] = tuple(args.exclude)
    if args.no_show:
        options["show_plot"] = False
    generate_plot(options)


if __name__ == "__main__":
    main()

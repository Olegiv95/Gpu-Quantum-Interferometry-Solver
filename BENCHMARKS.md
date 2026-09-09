# Benchmark Validation And Performance

The benchmark scripts validate the GPU Quantum Interferometry Solver (GQIS) against independent numerical solvers and
show how calculation time changes with parameter-grid size. They are supporting evidence for the solver, not the
primary GQIS interface. The runnable examples and [Lindblad and general ODE APIs](./GQIS_API.md)
describe normal use.

## Models

- `Benchmarks/Benchmark_01_two_level.py` evaluates a driven two-level system.
- `Benchmarks/Benchmark_02_four_level_Interferometry.py` evaluates a coupled qubit-resonator model represented by four basis
  states.
- `Benchmarks/Benchmark_03_accuracy_timestep_sweep.py` finds the coarsest tested time grid on which each selected
  solver still satisfies user-defined RMS and maximum-error limits. It can run either built-in model.

Benchmarks 01 and 02 offer the same solver choices. Central processing unit (CPU) solvers run on the computer processor;
graphics processing unit (GPU) solvers run on the NVIDIA GPU.

| Solver | Method |
| --- | --- |
| `gpu` | GQIS fixed-step fourth-order Runge-Kutta (RK4) solver on CUDA. |
| `python_cpu` | Transparent fixed-step Python RK4 reference. |
| `python_ode_cpu` | Adaptive SciPy `solve_ivp` embedded fourth/fifth-order Runge-Kutta (RK45) method on CPU. |
| `qutip_cpu` | Adaptive QuTiP `mesolve` reference on CPU. |
| `julia_gpu` | Julia DifferentialEquations/DiffEqGPU solver using the same reduced density-matrix ordinary differential equation (ODE) system as GQIS. |

These are the default Benchmark 01/02 names. Benchmark 03 and calibrated scaling runs also accept
`gqis_rk4`, `gqis_lserk4`, `gqis_dp5`, `gqis_tsit5`, `gqis_anas5`, `gqis_ab5`, `gqis_alshina6`
and `gqis_dop853`. Their [methods and per-step costs](GQIS_API.md#available-fixed-step-solvers) are
documented in the API reference. Select several in Benchmark 03's `target_solvers` to compare them in one run.

## Running Benchmarks

The user-editable block near the bottom of each script documents the model, grid, solver, and output settings. Run the
default configuration with:

```bash
python Benchmarks/Benchmark_01_two_level.py
python Benchmarks/Benchmark_02_four_level_Interferometry.py
```

The available modes are:

| Mode | Purpose |
| --- | --- |
| `single` | Run one selected solver. |
| `diff` | Run any two solvers and report map differences and timings. |
| `all` | Attempt every available solver. |
| `full_benchmark` | Measure calculation time over powers-of-two square-grid sizes and save comma-separated values (CSV) data and a Portable Network Graphics (PNG) figure. |

For example, compare GQIS with QuTiP using the settings selected in the benchmark file:

```bash
python Benchmarks/Benchmark_02_four_level_Interferometry.py --mode diff --solver gpu --solver-b qutip_cpu
```

`diff` mode prints mean-square deviation (MSE), root-mean-square deviation (RMS), maximum absolute deviation, and both
solver times. Increase `solver_steps_per_period` until the GQIS result is converged. Use divider `1` for QuTiP or SciPy
when validating every solver on the same requested time grid for evaluating time-dependent coefficients and recording
output.

Run either script with `--help` for its complete command-line options. Those options override `user_settings()`.

## Accuracy-Calibrated Dividers

Benchmark 03 compares RMS and maximum observable error against a reference while varying the
user-defined time step. Together with calculation times, this shows which methods and time grids
meet the chosen error limits. The preferred working point can change with the model and tolerances.

Available accuracy sweeps:

- Two-level comparison with QuTiP: [figure](Benchmarks/results/Benchmark_03_two_level_accuracy_timestep_sweep_GQIS_vs_QuTiP_plot.png), [CSV](Benchmarks/results/Benchmark_03_two_level_accuracy_timestep_sweep_GQIS_vs_QuTiP.csv).
- Dense two-level grid: [figure](<Benchmarks/results/Benchmark_03_two_level_accuracy_timestep_dense grid_sweep.png>), [CSV](<Benchmarks/results/Benchmark_03_two_level_accuracy_timestep_dense grid_sweep.csv>).
- Dense four-level grid: [figure](<Benchmarks/results/Benchmark_03_four_level_accuracy_timestep_dense grid_sweep.png>), [CSV](<Benchmarks/results/Benchmark_03_four_level_accuracy_timestep_dense grid_sweep.csv>).

Benchmark 03 can save its accepted time-grid choices by enabling:

```python
"save_optimal_dividers": True,
"optimal_dividers_file": None,
```

Set `"comparison_output": "mean"` for the usual time-averaged interferogram, or use `"final"` to compare the
same observable at the final calculated state. Final-state results and reference caches receive a `_final` filename
suffix, so they do not overwrite the standard mean-output results.

`Benchmarks/Benchmark_03_plot_from_csv.py` regenerates the figure without repeating any calculation. Its `include_solvers`
setting selects which curves are shown, and `show_results_table=False` produces a compact figure for the repository.
The complete table is useful while inspecting a run, but it is not required in a published figure when the CSV is
provided beside it. `table_times_only=True` removes the RMS and maximum-error rows, while
`show_best_summary_row=True` adds a compact dark-olive row containing each solver's largest accepted divider and measured time.
The same coarsest accepted grid is highlighted in the full table, regardless of timing fluctuations.
The first two panels show steps per period on a base-2 logarithmic axis by default; use `time_grid_axis="divider"`
to show divider factors instead. Set `divider_axis_scale="equidistant"` to give
each tested divider equal spacing, and set `divider_axis_labels="all"` to label every tested divider instead of the
default compact power-of-two labels. These settings are available in both Benchmark 03 and its CSV plotting helper.

Every Benchmark 03 run also writes a complete `<output_stem>_settings.json` manifest. Keep one manifest beside each
published CSV and PNG instead of copying the Python benchmark for each figure. For example, the low-grid two-level,
high-resolution two-level, and high-resolution four-level figures can each have a descriptive output stem and its own
small settings manifest while continuing to use the same benchmark implementation. Reproduce one with:

```bash
python Benchmarks/Benchmark_03_accuracy_timestep_sweep.py --settings Benchmarks/results/<name>_settings.json
```

To record the currently edited user settings for an already completed result without starting any calculation, run
`Benchmark_03_accuracy_timestep_sweep.py --save-settings-only <name>_settings.json`.

`Benchmarks/Benchmark_01_02_plot_from_csv.py` rebuilds either scaling figure directly from its CSV. Change `csv_file`,
`include_solvers`, table visibility, title, and output options in its user settings. New measurements can be supplied in
`additional_csv_files` or `additional_measured_points`; a matching solver and side dimension replaces the old point in
the graph without modifying the original CSV or running a solver. With `refresh_extrapolated_points=True`, later dashed
estimates between measured points are interpolated between their nearest measured neighbors in log-log space;
estimates beyond the last measurement use the last two measured points. Both remain marked as estimates.
Combine measurements only when the physical model and the
solver-specific time-grid settings match the primary CSV.

`None` writes `<output_stem>_optimal_dividers.json`. For each tested solver that passes both limits, the file records
the coarsest accepted grid (largest tested divider), its divider, actual steps per period, time, and errors. New files include the
complete Benchmark 03 settings and are refreshed after each solver finishes its sweep.

Benchmark 01 or 02 can load the matching calibration through its user settings:

```python
"accuracy_dividers_file": "Benchmark_03_two_level_accuracy_timestep_sweep_optimal_dividers.json",
```

Relative paths are searched from the working directory, `Benchmarks/results/`, beside the benchmark script, and in
its parent directory.
A calibration file must match the selected two-level or four-level model. If a solver has no accepted entry,
GPU methods fall back to divider `1` and CPU methods to divider `10`. In the current Benchmark 01/02 solver names,
`gpu` uses the `gqis_rk4` entry and `julia_gpu` uses `julia_gpu_fp32`.

In `full_benchmark` mode, loading a calibration uses Benchmark 03's model, duration, solver implementations, Julia
precision variants, and adaptive settings. Only the parameter-grid size changes. The saved steps per period are used
directly, so Benchmark 01/02's default base density cannot change the calibrated timestep. By default, the sweep uses
the calibration's target list; set `accuracy_solvers` or `--full-solvers` to choose a subset using Benchmark 03 names.
Fallback settings are printed explicitly because they have not passed the calibration's accuracy test.

For example, from the repository directory:

```text
python Benchmarks/Benchmark_01_two_level.py full_benchmark --accuracy-dividers-file Benchmark_03_two_level_2048_optimal_dividers.json --full-solvers gqis_rk4,gqis_dop853,julia_gpu_fp32_fopt,qutip_cpu
python Benchmarks/Benchmark_02_four_level_Interferometry.py full_benchmark --accuracy-dividers-file Benchmark_03_four_level_2048_optimal_dividers.json
```

Older divider files remain usable. Their selected points are preserved, and the matching `<stem>_settings.json` is
loaded alongside them. If it is missing for a two-level calibration, Benchmark 01's configured two-level profile is
used automatically, with Benchmark 03's adaptive settings and the calibration's saved step counts. For a four-level
file without a settings snapshot, or to select a specific manifest, pass `--accuracy-settings <matching-settings.json>`.
Existing calibration files retain their recorded selection; regenerate them to apply the current coarsest-accepted-grid rule.

To add a long measurement, edit `additional_measured_points` in `Benchmark_01_02_plot_from_csv.py`, for example:

```python
"additional_measured_points": (
    {"solver": "qutip_cpu", "side_dimension": 2048, "time_s": 43747.3, "status": "measured"},
),
"save_merged_csv": True,
```

This replaces any extrapolated point with the same solver and grid size, regenerates the graph, and saves a separate
`<CSV stem>_merged.csv`. The source CSV is preserved. Use the 43747.3-second measurement only with a matching two-level
model, duration, QuTiP settings and output density (32 output intervals per period in that recorded run).

Generate a separate calibration for each physical model and chosen accuracy limits. The resulting Benchmark 01/02
timings then compare solvers at documented accuracy constraints rather than at an assumed common step density.

> Running `all` or `full_benchmark` can take considerable time, especially with adaptive CPU solvers. Full benchmark
> mode terminates measurements that exceed its configured limit and extrapolates larger grids instead of leaving a
> timed-out process running.

## Reproducing Accuracy Figures

The [preset guide](Benchmarks/presets/README.md) describes the three shared Benchmark 03 profiles:
a small two-level comparison with QuTiP, a dense two-level sweep, and a dense four-level sweep.
Each profile uses the same benchmark script, with its settings stored separately.

Start with the [64×64 two-level preset](Benchmarks/presets/Benchmark_03_two_level_64.json):

```bash
python Benchmarks/Benchmark_03_accuracy_timestep_sweep.py --settings Benchmarks/presets/Benchmark_03_two_level_64.json
```

On Windows, [Run_Benchmark_03_two_level_64.bat](Benchmarks/Run_Benchmark_03_two_level_64.bat)
starts that calculation; [Plot_Benchmark_03_two_level_64.bat](Benchmarks/Plot_Benchmark_03_two_level_64.bat)
rebuilds its figure from CSV without rerunning solvers. Matching launchers exist for both dense-grid profiles.

Enable `save_reference_map` on the first calculation and `load_reference_map` on subsequent runs to reuse
the reference; keep the reference filename and model settings consistent. A dense QuTiP calculation can take
more than 12 hours. GQIS results are saved after each solver sweep, while Julia and QuTiP results are saved
after each completed point, so partial results remain usable.

For publication, retain the exact settings snapshot beside each selected CSV and figure. A preset is a
starting configuration, not a substitute for the settings of an earlier measurement.

## Numerical Comparison Notes

- GQIS defaults to fixed-step RK4; an accuracy calibration can select other GQIS methods. `python_cpu` uses RK4.
  Without a loaded accuracy calibration, GQIS uses divider `1` and the CPU
  methods use their configured fallback dividers.
- `python_ode_cpu` and `qutip_cpu` choose adaptive internal steps. Their default divider of `10` reduces the requested
  number of time samples used to evaluate time-dependent coefficients and record output; it does not change the internal
  adaptive accuracy target. Set the divider to `1` when all solvers must receive the same requested time grid.
- If a time list contains `M` samples, it defines `M - 1` integration intervals. `N` always denotes the number of
  simulated quantum levels, not the time-grid length.
- The Julia solver solves the same trace- and Hermiticity-reduced physical ODE system as GQIS. Its scaling value is the
  synchronized Julia solve time; symbolic and Julia-side single-threaded CPU preparation are excluded. Consequently,
  the plotted value is a synchronized solve-call measurement, including work done inside that call, rather than
  an isolated GPU-kernel measurement. Preparation/overhead is displayed separately when recorded in the CSV.
- The accuracy sweeps complement the hardware- and model-specific timings by showing performance at the chosen error limits.

## Precision And Julia Comparison Notes

FP32 has about seven significant decimal digits, but this is not a bound on the error of an integrated observable.
Step-size error, accumulated roundoff, signal evaluation and averaging all contribute. Reducing the step size
usually improves accuracy until these other contributions or reference error become comparable; higher-order
methods can reach that region with fewer steps. `fp64=True` selects double precision in either GQIS API when needed.

In the [64×64 two-level QuTiP comparison](Benchmarks/results/Benchmark_03_two_level_accuracy_timestep_sweep_GQIS_vs_QuTiP.csv),
at 2048 steps per period, the recorded GQIS methods have absolute RMS differences of approximately `3.1–4.4 × 10⁻⁶`
and maximum differences of `3.5 × 10⁻⁵`. In the [dense four-level sweep](<Benchmarks/results/Benchmark_03_four_level_accuracy_timestep_dense grid_sweep.csv>),
the divider-1 GQIS results differ from the finer RK4 reference by roughly `3–4 × 10⁻⁷` RMS and `4–6 × 10⁻⁶` maximum.
The latter measures agreement with a GQIS reference, not an independent absolute-error bound. These examples
illustrate achievable observable accuracy and its model dependence; the full sweeps show where coarser steps
start to exceed the selected tolerances.

Stock fixed-step Julia `GPUTsit5` with FP32 time showed time-accumulation drift in the long driven calculations
investigated here: repeatedly adding `dt` changes the evaluated drive phase, and finer steps need not improve the
result monotonically. The benchmark's experimental integer-step reconstruction substantially reduced this effect
while retaining FP32 arithmetic. This concerns the tested implementation and time representation, rather than the
order of the Tsit5 formula. Both stock and modified variants remain selectable:

| Benchmark name | Time/state arithmetic |
| --- | --- |
| `julia_gpu_fp32` | Stock FP32 time and state |
| `julia_gpu_fp64` | FP64 time and state |
| `julia_gpu_fp32_opt` | Modified FP64 time accumulator with FP32 stage/state arithmetic |
| `julia_gpu_fp32_fopt` | Modified integer-step reconstruction of FP32 time, FP32 state |

Julia preparation/overhead is measured as total wrapper/subprocess time minus the synchronized solve call.
It includes symbolic preparation, process/package startup, compilation and warm-up, parameter setup, and output
handling outside that call; it is not a separate measurement of `EnsembleProblem` construction alone. The current
wrapper launches a fresh Julia process per point, repeating startup costs. A persistent process with unchanged
function/types could reuse compilation and suitable problem data across calls.

Benchmark 01/02 scaling plots show dotted curves in the corresponding solver color for solve time plus
preparation, rather than horizontal preparation-only lines. Julia uses each point's recorded `prep_s`;
where it is missing, the mean available preparation is used as an estimate, including for extrapolated points.
New GQIS runs record the small-grid startup/warm-up duration and add it to each solve time as a first-use
estimate. It includes the warm-up solve, so it is not a separately measured cold full-grid run. Subsequent
GQIS methods add shared RHS preparation to their own warm-up duration. Older files with only RHS timing
produce a `tot time*` curve; the figure footnote identifies the recorded RHS-only contribution.
If no preparation measurement exists, that solver's dotted curve is omitted. The CSV plotter accepts
`preparation_overrides_s={"julia_gpu_fp32_fopt": seconds}` for a value recovered from a matching run/log;
`show_startup_times=False` hides these additional curves. Stored calculation times remain unchanged.
Use the plotter's `solver_labels` dictionary to set legend/table names, for example
`{"julia_gpu_fp32_fopt": "Julia", "qutip_cpu": "QuTiP(CPU)"}`. These labels do not change the
CSV identifiers; retain the Julia precision/stepping variant in the figure caption when using a shortened name.
Legends use compact `calc time` and `tot time` labels. A note below the table explains the timing boundaries,
startup estimates and QuTiP's CPU solve-call total; the table retains the original timing values.

DiffEqGPU also provides a [lower-level API for reduced overhead](https://docs.sciml.ai/DiffEqGPU/dev/tutorials/lower_level_api/):
prepare GPU-compatible problems, keep them on the device, and call `vectorized_solve` directly. This avoids the
high-level interface's automatic input/output transfers and allows data reuse. Initial compilation and problem
construction are still needed, and changed parameters require corresponding updates. This is a documented route
to reduce overhead, not a measured speedup in the current GQIS comparisons; the benchmark still uses
`EnsembleGPUKernel` through the standard ensemble interface.

## Full Scaling Benchmark

Full benchmark mode measures powers-of-two square grids. Measured plot points use circles. Once a solver exceeds or is
predicted to exceed the time limit, larger values are extrapolated on a graph with logarithmic scales on both axes and
plotted as squares using the same solver color. Extrapolation is intended to show scaling estimates, not substitute for
measured data.

Each generated CSV stores the equipment and software versions, physical and numerical configuration, grid dimensions,
number of simulations, solver, timing components, and measured/extrapolated status. The benchmark also saves its PNG
figure automatically so a long run can be compared with the reference results later.

## Reference Results

Reference desktop system:

- CPU: 11th Gen Intel Core i9-11900K at 3.50 gigahertz (GHz)
- GPU: NVIDIA GeForce RTX 3080 with 10 gigabytes (GB) VRAM
- precision: 32-bit floating point (FP32) for the GQIS scaling runs
- workload: 10,240 RK4 steps per two-level simulation; 5,840 per four-level simulation

The largest measured `32768 x 32768` grids contain 1.07 billion independent simulations.
GQIS(RK4) completed them in **98.43 seconds** for the two-level model and **303.85 seconds** for the four-level model.
The corresponding QuTiP estimates are 3,402,704 seconds (39.4 days) and 8,838,590 seconds (102.3 days);
these largest-grid CPU values are extrapolated, not measured runs.

For the four grid sizes from `4096 x 4096` through `32768 x 32768`, arithmetic means of the
point-by-point timing ratios are approximately **46,900× over QuTiP and 17.9× over Julia** for the two-level
model, and **50,700× over QuTiP and 60.6× over Julia** for the four-level model.
These ratios use the linked CSV calculation times for GQIS(RK4) and Julia's
`julia_gpu_fp32_fopt` variant, versus QuTiP's CPU run times; preparation is not added to the GPU calculation times.
Julia uses experimental integer-step FP32 time reconstruction in this comparison.
QuTiP timings in this range, and Julia timings beyond its measured range, are extrapolated as marked in the CSV.

The calibrated grids differ by solver. Per simulation, the two-level CSV records 10,240 RK4 steps,
1,720 DOP853 steps, 4,080 Julia steps and 1,280 QuTiP output intervals. The four-level CSV records
5,840 RK4 steps, 1,720 DOP853 steps, 5,840 Julia steps and 8,200 QuTiP output intervals.
QuTiP's internal adaptive integration steps are distinct from these output intervals.

### Two-Level Reference

[Timing data (CSV)](./Benchmarks/results/Benchmark_01_full_benchmark.csv) | [Figure file (PNG)](./Benchmarks/results/Benchmark_01_full_benchmark_plot.png)

![Two-level full benchmark](./Benchmarks/results/Benchmark_01_full_benchmark_plot.png)

### Four-Level Reference

[Timing data (CSV)](./Benchmarks/results/Benchmark_02_full_benchmark.csv) | [Figure file (PNG)](./Benchmarks/results/Benchmark_02_full_benchmark_plot.png)

![Four-level full benchmark](./Benchmarks/results/Benchmark_02_full_benchmark_plot.png)

## Reporting New Results

Generated files are written to `Benchmarks/results/`. Keep each selected CSV and PNG together. The CSV is the
authoritative record of hardware, software,
model, time-grid, precision, CPU-divider, sweep-limit, preparation, calculation, and measured/extrapolated metadata.
Regenerate both files after solver or benchmark changes before citing performance.

Benchmark CSV, PNG and settings JSON files are no longer ignored by default. Review the files selected for
each commit and include the results referenced by the documentation. Large reference caches and exploratory
outputs need not be published; keep each published figure with its matching CSV and settings snapshot.

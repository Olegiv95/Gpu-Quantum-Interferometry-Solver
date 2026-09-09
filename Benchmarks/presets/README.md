# Benchmark 03 figure presets

Use the matching `Run_Benchmark_03_*.bat` and `Plot_Benchmark_03_*.bat` files in
the parent `Benchmarks` directory. The plot launchers read saved CSV data;
they do not run solvers.

The high-resolution QuTiP calculation may take more than 12 hours at full CPU
load. Targets run in preset order: GQIS, each Julia variant, then QuTiP where
included. GQIS results are saved after each solver's complete divider sweep;
Julia and QuTiP results are appended after every point. Other CPU solvers save
after their complete sweep. The plot launchers can be used before the test finishes;
selected solvers without saved results yet are skipped.

| Preset | Grid and problem | Reference | QuTiP target |
| --- | --- | --- | --- |
| `two_level_64` | 64 × 64, two levels | QuTiP Adams | Full sweep, last target |
| `two_level_2048` | 2048 × 2048, two levels | GQIS RK4 | Largest divider only, last target |
| `four_level_2048` | 2048 × 2048, four levels, wd500 | GQIS RK4 | Not included |

Each JSON file contains a complete settings snapshot. All three use 40 drive
periods, 4096 reference steps/output intervals per period, a target base of
2048 steps per period, and RMS/max acceptance limits of 1e-3/1e-2.
QuTiP chooses its internal integration steps adaptively. RK4 remains the
large-grid reference to retain the established comparison; DOP853 can be
selected by editing `reference_solver` after validating its reference settings.

Results go to `Benchmarks/results/Benchmark_03_<preset>_metrics.csv` and related
figure/settings files. Running the same preset again replaces its result files.
Reference saving/loading remains disabled in these snapshots; enable
`save_reference_map` for a first run and `load_reference_map` for later reuse
if desired. Cache metadata is checked by the benchmark.

The batch files use the Anaconda Python under your Windows user directory when
available, otherwise `python` from PATH. Set `GQIS_PYTHON` to an executable path
to select another environment. Julia must be available as `julia`, or set its
path in the JSON preset.

Edit the CSV path in a plot launcher to plot an older result with a different
filename. Its `--solvers` list controls the displayed curves. Other plot choices
remain in `Benchmark_03_plot_from_csv.py` under `user_settings()`.

# Benchmark Validation And Performance

The benchmark scripts validate the GPU Quantum Interferometry Solver (GQIS) against independent numerical solvers and
show how calculation time changes with parameter-grid size. They are supporting evidence for the solver, not the
primary GQIS interface. The runnable examples and [`mesolve_2D` application programming interface
(API)](./GQIS_API.md) describe normal use.

## Models

- `Benchmark_01_two_level.py` evaluates a driven two-level system.
- `Benchmark_02_four_level_Interferometry.py` evaluates a coupled qubit-resonator model represented by four basis
  states.

Both benchmarks offer the same solver choices. Central processing unit (CPU) solvers run on the computer processor;
graphics processing unit (GPU) solvers run on the NVIDIA GPU.

| Solver | Method |
| --- | --- |
| `gpu` | GQIS fixed-step fourth-order Runge-Kutta (RK4) solver on CUDA. |
| `python_cpu` | Transparent fixed-step Python RK4 reference. |
| `python_ode_cpu` | Adaptive SciPy `solve_ivp` embedded fourth/fifth-order Runge-Kutta (RK45) method on CPU. |
| `qutip_cpu` | Adaptive QuTiP `mesolve` reference on CPU. |
| `julia_gpu` | Julia DifferentialEquations/DiffEqGPU solver using the same reduced density-matrix ordinary differential equation (ODE) system as GQIS. |

## Running Benchmarks

The user-editable block near the bottom of each script documents the model, grid, solver, and output settings. Run the
default configuration with:

```bash
python Benchmark_01_two_level.py
python Benchmark_02_four_level_Interferometry.py
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
python Benchmark_02_four_level_Interferometry.py --mode diff --solver gpu --solver-b qutip_cpu
```

`diff` mode prints mean-square deviation (MSE), root-mean-square deviation (RMS), maximum absolute deviation, and both
solver times. Increase `solver_steps_per_period` until the GQIS result is converged. Use divider `1` for QuTiP or SciPy
when validating every solver on the same requested time grid for evaluating time-dependent coefficients and recording
output.

Run either script with `--help` for its complete command-line options. Those options override `user_settings()`.

> Running `all` or `full_benchmark` can take considerable time, especially with adaptive CPU solvers. Full benchmark
> mode terminates measurements that exceed its configured limit and extrapolates larger grids instead of leaving a
> timed-out process running.

## Numerical Comparison Notes

- GQIS and `python_cpu` use fixed-step RK4. The `python_cpu` divider defaults to `1`, matching the GQIS step density.
- `python_ode_cpu` and `qutip_cpu` choose adaptive internal steps. Their default divider of `10` reduces the requested
  number of time samples used to evaluate time-dependent coefficients and record output; it does not change the internal
  adaptive accuracy target. Set the divider to `1` when all solvers must receive the same requested time grid.
- If a time list contains `M` samples, it defines `M - 1` integration intervals. `N` always denotes the number of
  simulated quantum levels, not the time-grid length.
- The Julia solver solves the same trace- and Hermiticity-reduced physical ODE system as GQIS. Its scaling value is the
  synchronized Julia solve time; symbolic and Julia-side single-threaded CPU preparation are excluded. Consequently,
  the plotted value does not represent Julia's complete model-preparation workflow or its peak video random-access
  memory (VRAM) requirement.
- The reference timings are hardware- and model-specific. Compare numerical output first, then interpret speed.

## Full Scaling Benchmark

Full benchmark mode measures powers-of-two square grids. Measured plot points use circles. Once a solver exceeds or is
predicted to exceed the time limit, larger values are extrapolated on a graph with logarithmic scales on both axes and
plotted as squares using the same solver color. Extrapolation is intended to show scaling estimates, not substitute for
measured data.

Each generated CSV stores the equipment and software versions, physical and numerical configuration, grid dimensions,
number of simulations, solver, timing components, and measured/extrapolated status. The benchmark also saves its PNG
figure automatically so a long run can be compared with the reference results later.

## Reference Results

Reference workstation:

- CPU: 11th Gen Intel Core i9-11900K at 3.50 gigahertz (GHz)
- GPU: NVIDIA GeForce RTX 3080 with 10 gigabytes (GB) VRAM
- precision: 32-bit floating point (FP32) for the GQIS scaling runs
- workload: 10,240 fixed RK4 steps per GQIS simulation

The largest measured `32768 x 32768` grids contain 1.07 billion independent simulations. GQIS completed them in
about 1 minute 44 seconds for the two-level model and 7 minutes 1 second for the four-level model. Direct QuTiP runs at
this resolution were not practical; extrapolation from measured smaller grids estimates about 69 days and 108 days,
respectively.

Across the linear scaling region from `4096 x 4096` through `32768 x 32768`, where calculation time increases in
proportion to the number of simulations, average point-by-point speedups were about 69,000 times over QuTiP and 22 times
over Julia for the two-level model, and 24,000 times over QuTiP and 38 times over Julia for the four-level model. The
QuTiP and Julia values in this region include extrapolated timings after each reference solver reaches the configured
practical limit.

### Two-Level Reference

[Timing data (CSV)](./Benchmark_01_full_benchmark.csv) | [Figure file (PNG)](./Benchmark_01_full_benchmark.png)

![Two-level full benchmark](./Benchmark_01_full_benchmark.png)

### Four-Level Reference

[Timing data (CSV)](./Benchmark_02_full_benchmark.csv) | [Figure file (PNG)](./Benchmark_02_full_benchmark.png)

![Four-level full benchmark](./Benchmark_02_full_benchmark.png)

## Reporting New Results

Keep the automatically generated CSV and PNG together. The CSV is the authoritative record of hardware, software,
model, time-grid, precision, CPU-divider, sweep-limit, preparation, calculation, and measured/extrapolated metadata.
Regenerate both files after solver or benchmark changes before citing performance.

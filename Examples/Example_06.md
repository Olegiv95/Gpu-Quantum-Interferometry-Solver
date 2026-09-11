# Example 06: Live Duffing Animation

Example 06 evolves many initial conditions of a driven Duffing oscillator and displays the
moving cloud in phase space. The example demonstrates general ODE sweeps and repeated GPU
continuation with `odesolve_2D`. Each display pixel is colored by the mean initial position
of the trajectories currently inside that pixel.

## Run And Configure The Simulation

From the repository root, run:

```bash
python Examples/Example_06_symbolic_ode_sweep.py
```

The [installation guide](../INSTALLATION_TEST.md) covers CUDA and optional example dependencies.
Edit `USER_SETTINGS` in the example to control three independent quantities:

| Setting | Controls |
| --- | --- |
| `attractor_grid_size` | Number of initial conditions/trajectories |
| `attractor_steps_per_period` | Integration accuracy through time resolution |
| `attractor_render_bins` | Resolution of the displayed trajectory histogram |

Choose initial-cloud bounds and viewing bounds separately. The view stays fixed; trajectories
outside the view continue evolving but are not drawn. Reducing display resolution or limiting
playback speed does not change the ODE time step.

The example advances the state in short blocks. `return_device=True` keeps each final state on
the GPU, and `t_in` supplies the next block's absolute starting time. GPU rasterization transfers
only the compact display image to Matplotlib. See [GPU continuation](../GQIS_API.md#continuing-on-the-gpu)
for the corresponding API calls.

## Read The Timings

Console timings distinguish the ODE kernel, the whole solve call, and raster/copy work.
The solve call includes Python setup and GPU finite-value checks. First use may include compilation;
subsequent frames reuse the compiled kernel. Unequal integration-step counts per frame are reported.
Matplotlib drawing, terminal logging and optional video encoding also affect playback speed.

For a per-frame record, set `frame_log_to_ram=True`. After the window closes, the example writes
a timestamped CSV and settings JSON to `Examples/results/`, including after an early user stop.
Set `frame_log_every=0` to avoid console updates during that measurement.

## Summary

Set trajectory count, integration resolution and display resolution independently. The simulation
retains its state on the GPU between frames and downloads a display image. Use the timing CSV to
distinguish numerical work from drawing and scheduling when investigating slow or uneven playback.

## Optional Rendering And Playback Controls

The ODE model, initial conditions and solver call live in the example.
[fast_gpu_render.py](fast_gpu_render.py) contains rendering, animation and logging;
its `RENDER_DEFAULTS` lists advanced options that can be overridden in `USER_SETTINGS`.

### Histogram And Image Rendering

`histogram_partitions=8` spreads atomic updates across reusable buffers, then merges the buffers.
This reduces contention as trajectories concentrate, at the cost of extra memory and a merge pass.
Use `1` to compare the unpartitioned renderer. Partitioning changes rendering, not the simulation state.

With `gpu_display_rgba=True`, the GPU scales the compact histogram to the plotting area's pixel
dimensions and applies the color map. A custom Matplotlib artist draws the finished 8-bit RGBA
image, using two reusable pinned host buffers. `attractor_render_bins=(512, 512)` controls trajectory
projection; `attractor_interpolation="bilinear"` smooths image scaling, while `"nearest"` keeps sharp pixels.

Set `direct_rgba_draw=False` to compare the RGBA `imshow` path, or `gpu_display_rgba=False` to use
Matplotlib scalar-image rendering. Larger windows require larger image transfers and canvas copies.
Resize, linear-axis zoom/pan and video DPI changes rescale the retained raster without another ODE solve.
Those redraws happen outside the frame callback.

### Playback And Video

`animation_blit=True` can reduce redraw work on compatible backends. The changing time label stays
inside the axes; the title and colorbar remain static. `overlap_gpu_with_matplotlib=True` allows
computation of the next frame to overlap drawing; compare timing logs on your system.
Frame-data caching is disabled, and the worker queue holds at most one pending frame.

`animation_max_fps=60` caps interactive playback; `0` removes the cap. GUI timer resolution and
rendering may lower the achieved frame rate. This setting changes neither integration steps nor
MP4 playback speed, which uses `video_fps`. `frame_log_every=10` limits console timing updates.

### Timing CSV Fields

The CSV separates GPU, solve-call, raster/copy, artist-update and callback durations and identifies
interactive playback versus video export. `callback_interval_s` measures successive callback starts;
`outside_callback_s` includes drawing, timer scheduling or encoding between callbacks.
These values do not measure when a frame becomes visible on the monitor.

With GPU RGBA display, `raster_copy_s` measures projection, and `rgba_copy_s` measures scaling,
color conversion and download. `display_width` and `display_height` record image dimensions.
RAM logging stores scalar timing tuples, not images or trajectories; no per-frame file writes or GPU
telemetry queries occur. Forced process termination before saving loses the RAM records.

## Related Duffing Visualization

The [Wikimedia animation by Timeroot](https://commons.wikimedia.org/wiki/File:Duffing_oscillator_strange_attractor_with_color.gif)
uses damping `0.02`, forcing amplitude `3`, angular frequency `1` and a sine drive, displayed over
four periods. Example 06 uses a rectangular initial cloud and a wider fixed view; the Wikimedia
file page does not specify numerical view bounds or the initial point set.

# Example 06: live Duffing animation

Example 06 advances the cloud in short blocks using `t_in` for absolute time and `return_device=True`
to keep the state on the GPU. With GPU rasterization enabled, only the compact display image returns to
Matplotlib. Colors track mean initial position within each pixel. `attractor_grid_size` controls trajectory
count, `attractor_render_bins` controls image resolution, and `attractor_steps_per_period` controls integration
accuracy independently of playback speed. Console timings separate the ODE kernel, whole solve call and
raster/copy work. `histogram_partitions=8` spreads atomic updates over reusable buffers, then merges them,
reducing contention as the cloud concentrates; set it to `1` to compare with the original renderer.
The viewing bounds stay fixed and can be set independently of the initial cloud. Points outside the view
continue evolving but are not drawn. Histogram partitioning trades extra memory and a fixed-cost merge for
less contention; the simulation state is unchanged.

The [Wikimedia animation by Timeroot](https://commons.wikimedia.org/wiki/File:Duffing_oscillator_strange_attractor_with_color.gif)
uses the same equation with damping `0.02`, forcing amplitude `3`, angular frequency `1`, and a sine drive,
displayed over four periods. Its numerical view bounds and initial point set are not published on the file page.
Example 06 uses a rectangular initial cloud and a wider fixed view rather than assuming the
same preparation. See [GPU continuation](../GQIS_API.md#continuing-on-the-gpu) for both APIs.

Frame times also include Python call setup and GPU finite-value checks. First use can produce a compilation
spike; later frames reuse the same kernel for both zero and nonzero `t_in`. Unequal steps per frame are reported explicitly.
Raster atomic contention can still vary within a histogram partition. Matplotlib redraws, terminal output and
optional video encoding affect playback separately from the reported computation timings. `frame_log_every=10`
limits terminal updates; `animation_blit=True` can reduce redraw work on compatible backends. Frame-data caching
is disabled and the optional worker queue holds at most one pending frame, so no trajectory history grows with playback.

The ODE definition, initial conditions and direct `odesolve_2D` call are kept in Example 06. Rendering,
animation and logging are in [fast_gpu_render.py](fast_gpu_render.py); its `RENDER_DEFAULTS`
lists the advanced options, any of which can be overridden in the example's `USER_SETTINGS`.

The changing time label is drawn inside the axes so it remains visible with blitting enabled; the title and
colorbar stay static. By default, `gpu_display_rgba=True` scales the phase raster to the plotting area's
pixel dimensions and applies the colour map on the GPU. A custom Matplotlib artist passes the finished
8-bit RGBA image directly to the renderer, bypassing `imshow`'s CPU conversion, alpha processing and
resampling. Two pinned host buffers are reused during playback. Keep `attractor_render_bins=(512, 512)` for the
trajectory projection and choose `attractor_interpolation="bilinear"` for smooth GPU scaling or `"nearest"`
for sharp pixels. These affect rendering, not integration accuracy or the number of trajectories.
Set `direct_rgba_draw=False` to compare against the previous RGBA `imshow` path, or `gpu_display_rgba=False`
for Matplotlib's scalar-image rendering. Larger windows still require larger image transfers and canvas
copies. Resize, linear-axis zoom/pan and video DPI changes rescale the retained small raster on the GPU,
without repeating the ODE solve. These redraws occur outside the frame callback, so their cost appears in
the outside-callback timing rather than `rgba_copy_s`. The existing
`overlap_gpu_with_matplotlib=True` option can overlap the next
frame's computation with drawing; compare the RAM logs to see whether it helps on your system.
`animation_max_fps=60` caps interactive playback through the GUI timer; set it to `0` to run uncapped.
Timer resolution and rendering time can lower the actual frame rate. This does not change integration
steps or MP4 playback speed, which is controlled separately by `video_fps`.

For stutter analysis, Example 06's `frame_log_to_ram=True` collects scalar timing records for every frame
and writes a timestamped CSV plus settings JSON to `Examples/results/` after the window closes, including
when playback is stopped early. Keep `frame_log_every=0` to avoid terminal updates. The CSV separates GPU,
solve-call, raster/copy, artist-update and callback timings, and identifies video export versus interactive
playback. `callback_interval_s` measures successive callback starts; `outside_callback_s` includes time
between callbacks, such as GUI drawing, timer scheduling or encoding. These are not display-presentation
timestamps. No per-frame files or GPU telemetry queries are made; RAM stores only timing tuples, not images
or trajectories. With GPU RGBA display, `raster_copy_s` measures projection only; `rgba_copy_s` records GPU
scaling, colour conversion and the image download. `display_width` and `display_height` record the image
dimensions, allowing comparisons between window sizes. RAM-only records are lost if the process is
forcibly terminated before saving.

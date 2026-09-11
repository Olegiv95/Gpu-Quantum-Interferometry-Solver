"""GPU phase-space projection, direct Matplotlib drawing and animation support.

Example 06 keeps the ODE and initial conditions; this helper handles display,
video export and timing logs. Advanced settings are collected in RENDER_DEFAULTS.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np

from matplotlib.artist import Artist

try:
    import cupy as cp
except ImportError:
    cp = None


RENDER_DEFAULTS = {
    # Advanced display settings: override any of these in Example 06's USER_SETTINGS.
    "gpu_rasterization": True,        # False downloads the full cloud for CPU projection.
    "gpu_display_rgba": True,         # Scale and colour on GPU; transfer only the finished image.
    "direct_rgba_draw": True,         # Bypass imshow's CPU conversion/resampling.
    "histogram_partitions": 8,        # Reduce atomic contention; 16 MiB at 512 x 512.
    "attractor_empty_color": "white",
    "attractor_interpolation": "bilinear", # GPU: "bilinear" or "nearest".
    "animation_blit": True,
    "animation_interval_ms": 1,       # Minimum GUI timer interval when the FPS cap is off.
    "overlap_gpu_with_matplotlib": True, # One background frame, never a growing queue.
    "gqis_frame_timings": False,
    "frame_log_every": 0,             # 0 disables terminal updates; N prints every N frames.
    "frame_log_to_ram": True,         # Write CSV and settings only after the window closes.
    "frame_log_filename": None,       # None selects a timestamped name in results/.
    "video_filename": "Example_06_Duffing_attractor_realtime.mp4",
    "video_fps": 30,
    "video_dpi": 120,
    "ffmpeg_preset": "medium",
    "ffmpeg_crf": 23,                 # Lower values improve video quality at larger file size.
    "show_figure": True,
}


def _frame_step_ranges(total_steps: int, requested_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Split an exact total integration-step count among animation frames."""
    if total_steps < 1:
        raise ValueError("The attractor integration must contain at least one solver step.")
    frame_count = min(max(1, int(requested_frames)), int(total_steps))
    edges = np.rint(np.linspace(0, total_steps, frame_count + 1)).astype(np.int64)
    starts = edges[:-1]
    stops = edges[1:]
    if np.any(stops <= starts):
        raise RuntimeError("Internal animation-step partition produced an empty frame.")
    return starts, stops


def _normalize_state_output(result, grid_size: int, dtype):
    """Normalize a GQIS final-state result while preserving the 2D sweep grid.

    ``odesolve_2D`` is a 2D CUDA sweep.  Keeping the state as
    ``(grid_size, grid_size, 2)`` is important: flattening 2048**2 trajectories
    into one 4,194,304-element sweep axis can exceed CUDA's 2D launch limits.

    CuPy output remains on the device, including when reshaped for rendering.
    """
    is_gpu = cp is not None and isinstance(result, cp.ndarray)
    xp = cp if is_gpu else np
    array = xp.real(result)

    expected_values = int(grid_size) * int(grid_size) * 2
    if int(array.size) != expected_values:
        raise RuntimeError(
            "Unexpected final-state size from odesolve_2D: "
            f"shape={array.shape}, size={array.size}; expected {expected_values} values."
        )

    if array.ndim >= 1 and array.shape[-1] == 2:
        state = array.reshape(grid_size, grid_size, 2)
    elif array.ndim >= 1 and array.shape[0] == 2:
        state = xp.moveaxis(array, 0, -1).reshape(grid_size, grid_size, 2)
    else:
        state = array.reshape(grid_size, grid_size, 2)

    target_dtype = xp.float64 if dtype == np.float64 else xp.float32
    return xp.ascontiguousarray(state, dtype=target_dtype)


class PhaseRasterizer:
    """Turn millions of phase-space points into one small image.

    CuPy mode uses one custom CUDA binning kernel and one tiny finalize kernel.  It avoids
    the large temporary ``ix``, ``iy``, ``valid`` and ``flat`` arrays produced by a chain
    of high-level histogram operations. The raster can stay on the GPU for display
    scaling or be copied back as a ``ny x nx`` float32 image.
    """

    _BIN_KERNEL_SOURCE = r'''\
extern "C" __global__
void phase_bin(const float* points, const float* colors,
               unsigned int* counts, float* weighted,
               const int n, const int nx, const int ny, const int partitions,
               const float xmin, const float ymin,
               const float sx, const float sy)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    const float x = points[2 * i];
    const float v = points[2 * i + 1];
    if (!isfinite(x) || !isfinite(v) || x < xmin || v < ymin) return;
    const int ix = (int)((x - xmin) * sx);
    const int iy = (int)((v - ymin) * sy);

    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
        const int pixel = (blockIdx.x % partitions) * nx * ny + iy * nx + ix;
        atomicAdd(&counts[pixel], 1u);
        atomicAdd(&weighted[pixel], colors[i]);
    }
}
'''

    _FINAL_KERNEL_SOURCE = r'''\
extern "C" __global__
void phase_finalize(const unsigned int* counts, const float* weighted,
                    float* raster, const int n, const int partitions)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    unsigned int count = 0;
    float sum = 0.0f;
    for (int part = 0; part < partitions; ++part) {
        count += counts[part * n + i];
        sum += weighted[part * n + i];
    }
    raster[i] = count ? sum / (float)count : nanf("");
}
'''

    def __init__(self, color_values: np.ndarray, *, x_range, v_range, bins,
                 use_gpu: bool = True, partitions: int = 8):
        self.nx, self.ny = map(int, bins)
        self.xmin, self.xmax = map(float, x_range)
        self.ymin, self.ymax = map(float, v_range)
        self.colors = np.ascontiguousarray(color_values, dtype=np.float32)
        self.use_gpu = bool(use_gpu and cp is not None)
        self._point_count = int(self.colors.size)
        self.partitions = int(partitions)
        if self.partitions < 1:
            raise ValueError("histogram_partitions must be positive.")

        if self.use_gpu:
            self._bin_kernel = cp.RawKernel(self._BIN_KERNEL_SOURCE, "phase_bin")
            self._final_kernel = cp.RawKernel(self._FINAL_KERNEL_SOURCE, "phase_finalize")
            self._colors_gpu = cp.asarray(self.colors)
            self._points_gpu = None  # Allocate an upload buffer only for host input.
            pixels = self.nx * self.ny
            self._counts_gpu = cp.empty(pixels * self.partitions, dtype=cp.uint32)
            self._weighted_gpu = cp.empty(pixels * self.partitions, dtype=cp.float32)
            self._raster_gpu = cp.empty(pixels, dtype=cp.float32)

    @property
    def backend_name(self) -> str:
        return "CuPy RawKernel" if self.use_gpu else "NumPy"

    def render(self, points, *, return_device=False):
        """Return one ``(ny, nx)`` float32 phase-space image."""
        if self.use_gpu:
            return self._render_gpu(points, return_device=return_device)
        return self._render_cpu(points)

    def _render_gpu(self, points, *, return_device=False):
        if isinstance(points, cp.ndarray):
            # No extra state copy in FP32; FP64 is converted only for display.
            device_points = cp.ascontiguousarray(points, dtype=cp.float32).reshape(-1, 2)
        else:
            host = np.ascontiguousarray(np.asarray(points).reshape(-1, 2), dtype=np.float32)
            if len(host) != self._point_count:
                raise ValueError("Phase-space point count changed between animation frames.")
            if self._points_gpu is None:
                self._points_gpu = cp.empty((self._point_count, 2), dtype=cp.float32)
            self._points_gpu.set(host)
            device_points = self._points_gpu

        self._counts_gpu.fill(0)
        self._weighted_gpu.fill(0.0)
        # Blocks use separate histograms to reduce cross-block atomic contention.
        # Buffers are reused, and the merge cost is independent of cloud concentration.

        threads = 256
        blocks_points = (self._point_count + threads - 1) // threads
        sx = np.float32(self.nx / (self.xmax - self.xmin))
        sy = np.float32(self.ny / (self.ymax - self.ymin))
        self._bin_kernel(
            (blocks_points,), (threads,),
            (
                device_points,
                self._colors_gpu,
                self._counts_gpu,
                self._weighted_gpu,
                np.int32(self._point_count),
                np.int32(self.nx),
                np.int32(self.ny),
                np.int32(self.partitions),
                np.float32(self.xmin),
                np.float32(self.ymin),
                sx,
                sy,
            ),
        )

        pixels = self.nx * self.ny
        blocks_pixels = (pixels + threads - 1) // threads
        self._final_kernel(
            (blocks_pixels,), (threads,),
            (self._counts_gpu, self._weighted_gpu, self._raster_gpu, np.int32(pixels),
             np.int32(self.partitions)),
        )

        if return_device:
            # Scratch view: consume before the next render on this worker/stream.
            return self._raster_gpu.reshape(self.ny, self.nx)
        # This is the only GPU->CPU copy in scalar mode: e.g. 1 MiB
        # for a 512x512 float32 image, independent of the millions of trajectories.
        return cp.asnumpy(self._raster_gpu.reshape(self.ny, self.nx))

    def _render_cpu(self, points) -> np.ndarray:
        # Explicit CPU fallback copies the full cloud; use GPU rasterization for dense grids.
        points = (cp.asnumpy(points) if cp is not None and isinstance(points, cp.ndarray)
                  else np.asarray(points)).reshape(-1, 2)
        ix = ((points[:, 0] - self.xmin) * (self.nx / (self.xmax - self.xmin))).astype(np.int32)
        iy = ((points[:, 1] - self.ymin) * (self.ny / (self.ymax - self.ymin))).astype(np.int32)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        flat = iy[valid] * self.nx + ix[valid]
        counts = np.bincount(flat, minlength=self.nx * self.ny)
        weighted = np.bincount(flat, weights=self.colors[valid], minlength=self.nx * self.ny)
        raster = np.full(self.nx * self.ny, np.nan, dtype=np.float32)
        occupied = counts > 0
        raster[occupied] = (weighted[occupied] / counts[occupied]).astype(np.float32)
        return raster.reshape(self.ny, self.nx)


class GPUDisplayImage:
    """Scale and colour a device raster; download only display-sized uint8 RGBA.

    Colour interpolation includes the empty-bin colour, avoiding NaN borders.
    Two pinned host buffers allow one frame to be displayed while the next is made.
    """

    _SOURCE = r'''
__device__ float4 pixel_color(const float* src, const float4* lut,
                             int x, int y, int nx, int ny, int ncolors,
                             float lo, float inv_range)
{
    x = max(0, min(nx - 1, x)); y = max(0, min(ny - 1, y));
    float value = src[y * nx + x];
    int index = ncolors; // Last LUT entry is the empty-bin colour.
    if (isfinite(value)) {
        float u = fminf(1.0f, fmaxf(0.0f, (value - lo) * inv_range));
        index = min(ncolors - 1, (int)(u * ncolors));
    }
    return lut[index];
}
extern "C" __global__
void display_rgba(const float* src, const float4* lut, uchar4* out,
                  int nx, int ny, int width, int height, int ncolors,
                  float lo, float inv_range, int linear,
                  float u0, float u1, float v0, float v1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    float x = (u0 + (i % width + 0.5f) * ((u1-u0) / width)) * nx - 0.5f;
    float y = (v0 + (i / width + 0.5f) * ((v1-v0) / height)) * ny - 0.5f;
    float4 c;
    if (x < -0.5f || x > nx-0.5f || y < -0.5f || y > ny-0.5f) {
        c = lut[ncolors]; // Panning outside the projected region shows the background.
    } else if (linear) {
        int ix = (int)floorf(x), iy = (int)floorf(y);
        float fx = x - ix, fy = y - iy;
        float4 a = pixel_color(src, lut, ix, iy, nx, ny, ncolors, lo, inv_range);
        float4 b = pixel_color(src, lut, ix+1, iy, nx, ny, ncolors, lo, inv_range);
        float4 d = pixel_color(src, lut, ix, iy+1, nx, ny, ncolors, lo, inv_range);
        float4 e = pixel_color(src, lut, ix+1, iy+1, nx, ny, ncolors, lo, inv_range);
        #define MIX(k) ((a.k + fx*(b.k-a.k))*(1.0f-fy) + (d.k + fx*(e.k-d.k))*fy)
        c = make_float4(MIX(x), MIX(y), MIX(z), MIX(w));
        #undef MIX
    } else {
        c = pixel_color(src, lut, (int)floorf(x+0.5f), (int)floorf(y+0.5f),
                        nx, ny, ncolors, lo, inv_range);
    }
    out[i] = make_uchar4((unsigned char)(255.0f*c.x+0.5f),
                        (unsigned char)(255.0f*c.y+0.5f),
                        (unsigned char)(255.0f*c.z+0.5f),
                        (unsigned char)(255.0f*c.w+0.5f));
}
'''

    def __init__(self, cmap, color_min, color_max, interpolation):
        if interpolation not in ("nearest", "bilinear"):
            raise ValueError("GPU display supports 'nearest' or 'bilinear'; disable "
                             "gpu_display_rgba for other Matplotlib filters.")
        self.kernel = cp.RawKernel(self._SOURCE, "display_rgba")
        self.ncolors = int(cmap.N)
        self.lut = cp.asarray(np.vstack((cmap(np.arange(self.ncolors)), cmap.get_bad())),
                              dtype=cp.float32)
        self.lo = np.float32(color_min)
        self.inv_range = np.float32(1.0/(color_max-color_min) if color_max > color_min else 0.0)
        self.linear = np.int32(interpolation == "bilinear")
        self.shape = None
        self.slot = 0

    def render(self, raster, display_size):
        width, height = display_size[:2]
        view = display_size[2:] or (0.0, 1.0, 0.0, 1.0)
        shape = (height, width, 4)
        if shape != self.shape:
            self.device = cp.empty(shape, dtype=cp.uint8)
            self.host = [np.frombuffer(cp.cuda.alloc_pinned_memory(height*width*4),
                                      dtype=np.uint8, count=height*width*4).reshape(shape)
                         for _ in range(2)]
            self.shape = shape
        raster = cp.asarray(raster, dtype=cp.float32)
        ny, nx = raster.shape
        self.kernel(((width*height+255)//256,), (256,),
                    (raster, self.lut, self.device, np.int32(nx), np.int32(ny),
                     np.int32(width), np.int32(height), np.int32(self.ncolors),
                     self.lo, self.inv_range, self.linear, *(np.float32(v) for v in view)))
        host = self.host[self.slot]
        self.slot = 1-self.slot
        cp.asnumpy(self.device, out=host)  # Blocking: safe for Matplotlib on the GUI thread.
        return host


class DirectRGBAArtist(Artist):
    """Draw finished pixels without AxesImage's CPU conversion or resampling.

    Ordinary frames retain the worker's buffer without copying. A separate GPU
    display buffer handles resize/zoom/export redraws without racing the worker.
    """

    def __init__(self, ax, extent, redraw_display, *, animated=False):
        super().__init__()
        self.extent = extent
        self.display = redraw_display
        self.rgba = self.raster = self.view = None
        self.set_animated(animated)
        self.set_zorder(0)
        ax.add_artist(self)
        self.set_clip_path(ax.patch)
        self.set_in_layout(False)

    def display_view(self):
        ax = self.axes
        xmin, xmax, ymin, ymax = self.extent
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        return (max(1, int(np.ceil(ax.bbox.width))), max(1, int(np.ceil(ax.bbox.height))),
                (left-xmin)/(xmax-xmin), (right-xmin)/(xmax-xmin),
                (bottom-ymin)/(ymax-ymin), (top-ymin)/(ymax-ymin))

    def set_frame(self, rgba, raster, view):
        self.rgba, self.raster, self.view = rgba, raster, view
        self.stale = True

    def draw(self, renderer):
        if not self.get_visible() or self.rgba is None:
            return
        view = self.display_view()
        if view != self.view:
            self.rgba = self.display.render(self.raster, view)
            self.view = view
        gc = renderer.new_gc()
        try:
            self._set_gc_clip(gc)
            # draw_image expects the bottom row first, matching our GPU raster.
            renderer.draw_image(gc, self.axes.bbox.x0, self.axes.bbox.y0, self.rgba)
        finally:
            gc.restore()
        self.stale = False


def animate_phase_space(advance, initial_state, initial_color, settings, *,
                        drive_period, dt, total_steps, output_folder):
    """Animate a device cloud using advance(state, t_in, steps) -> (state, timings)."""
    settings = {**RENDER_DEFAULTS, **settings}
    fps_limit = float(settings.get("animation_max_fps", 0))
    if not np.isfinite(fps_limit) or fps_limit < 0:
        raise ValueError("animation_max_fps must be nonnegative; 0 disables the cap.")
    interval_ms = max(1, int(settings["animation_interval_ms"]))
    if fps_limit > 0:
        interval_ms = max(interval_ms, int(np.ceil(1000.0 / fps_limit)))
    total_start = time.perf_counter()
    grid_size = initial_state.shape[0]
    dtype = np.float64 if settings["fp64"] else np.float32
    point_count = grid_size * grid_size
    frame_starts, frame_stops = _frame_step_ranges(total_steps, settings["animation_frames"])
    frame_count = len(frame_starts)

    rasterizer = PhaseRasterizer(
        initial_color,
        x_range=settings["attractor_plot_x_range"],
        v_range=settings["attractor_plot_v_range"],
        bins=settings["attractor_render_bins"],
        use_gpu=bool(settings.get("gpu_rasterization", True)),
        partitions=settings.get("histogram_partitions", 8),
    )
    initial_phase_image = rasterizer.render(initial_state)
    use_blit = bool(settings.get("animation_blit", False))

    unique_steps = sorted(set((frame_stops - frame_starts).tolist()))
    print(
        f"Realtime attractor: {grid_size}x{grid_size} = {point_count:,} trajectories; "
        f"{frame_count} calculated frames; steps/frame={unique_steps}; "
        f"raster={settings['attractor_render_bins'][0]}x{settings['attractor_render_bins'][1]}; "
        f"renderer={rasterizer.backend_name}"
    )

    # ---------------------------------------------------------------------
    # One animation graph only.  The Matplotlib artist is created once and
    # playback changes only its image data and title, matching Example 03's fast path.
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    use_blit = use_blit and fig.canvas.supports_blit

    color_min = float(np.min(initial_color))
    color_max = float(np.max(initial_color))
    phase_cmap = plt.get_cmap(settings["attractor_cmap"]).copy()
    phase_cmap.set_bad(color=settings["attractor_empty_color"], alpha=1.0)
    phase_norm = mcolors.Normalize(vmin=color_min, vmax=color_max)
    gpu_display = bool(settings.get("gpu_display_rgba", True)) and rasterizer.use_gpu
    display = (GPUDisplayImage(phase_cmap, color_min, color_max,
                              settings["attractor_interpolation"]) if gpu_display else None)
    direct_draw = gpu_display and settings["direct_rgba_draw"]
    extent = [
        rasterizer.xmin, rasterizer.xmax, rasterizer.ymin, rasterizer.ymax,
    ]

    if direct_draw:
        redraw_display = GPUDisplayImage(phase_cmap, color_min, color_max,
                                         settings["attractor_interpolation"])
        phase_img = DirectRGBAArtist(ax, extent, redraw_display, animated=use_blit)
        # Register the full data extent for the toolbar's Home/autoscale behavior.
        ax.update_datalim([(extent[0], extent[2]), (extent[1], extent[3])])
    else:
        phase_img = ax.imshow(initial_phase_image, origin="lower", extent=extent,
                              aspect="auto", cmap=phase_cmap, norm=phase_norm,
                              interpolation="nearest" if gpu_display else settings["attractor_interpolation"],
                              resample=not gpu_display, animated=use_blit)
    ax.set(
        xlabel="Position x",
        ylabel="Velocity dx/dt",
        xlim=extent[:2],
        ylim=extent[2:],
    )
    ax.set_title("Real-time Duffing flow")
    # FuncAnimation blits the axes rectangle, not the title above it. Keep the
    # changing time inside that rectangle so it refreshes without a full redraw.
    phase_title = ax.text(0.02, 0.98, f"t/T = {settings.get('t_in', 0.0)/drive_period:.3f}",
                          transform=ax.transAxes, ha="left", va="top", animated=use_blit,
                          clip_on=True, zorder=3,
                          bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
    fig.colorbar(
        ScalarMappable(norm=phase_norm, cmap=phase_cmap),
        ax=ax,
        label="Initial x(t_in)",
    )
    fig.tight_layout()

    def display_size():
        # Query only on the GUI thread. bbox uses physical renderer pixels, including
        # HiDPI and the DPI temporarily selected during video export.
        if direct_draw:
            return phase_img.display_view()
        return max(1, int(np.ceil(ax.bbox.width))), max(1, int(np.ceil(ax.bbox.height)))

    if display is not None:
        fig.canvas.draw()
        # Warm the display kernel before playback; no ODE solve is needed.
        initial_raster = cp.asarray(initial_phase_image)
        initial_view = display_size()
        initial_rgba = display.render(initial_raster, initial_view)
        if direct_draw:
            phase_img.set_frame(initial_rgba, initial_raster, initial_view)
        else:
            phase_img.set_data(initial_rgba)

    folder = Path(output_folder)
    if settings["save_mp4"]:
        folder.mkdir(exist_ok=True)

    # Mutable live state.  The previous frame's final state is the next frame's y0.
    live = {
        "state": initial_state,
        "future": None,
        "executor": None,
        "last_frame": -1,
        "ema_ms_per_step": None,
        "last_callback_start": None,
        "last_callback_end": None,
        "log_phase": "interactive",
        "playback_pass": 0,
    }
    log_to_ram = bool(settings.get("frame_log_to_ram", False))
    frame_log = []  # Scalar tuples only: no state arrays or raster images retained.
    log_columns = ("phase", "playback_pass", "frame", "steps", "simulation_time_end",
                   "elapsed_wall_s", "callback_interval_s", "outside_callback_s", "callback_s",
                   "wait_or_solve_s", "artist_update_s", "gpu_s", "gpu_s_per_step",
                   "solve_call_s", "raster_copy_s", "compute_total_s", "rhs_stage_s",
                   "cached_rhs", "cached_kernel", "rgba_copy_s", "display_width", "display_height")
    def calculate_frame(frame_pos: int, state_in, image_size):
        frame_start = time.perf_counter()
        start_step = int(frame_starts[frame_pos])
        stop_step = int(frame_stops[frame_pos])
        steps_this_frame = stop_step - start_step

        frame_t_in = settings.get("t_in", 0.0) + start_step * dt
        solve_start = time.perf_counter()
        result, timing_info = advance(state_in, frame_t_in, steps_this_frame)
        state_out = _normalize_state_output(result, grid_size, dtype)
        solve_elapsed = time.perf_counter() - solve_start

        raster_start = time.perf_counter()
        raster = rasterizer.render(state_out, return_device=gpu_display)
        # Retain only the small projected raster for redraws; the worker reuses its
        # histogram while the GUI may rescale the displayed frame after a resize.
        source_raster = raster.copy() if direct_draw else None
        if gpu_display:
            cp.cuda.get_current_stream().synchronize()  # Separate projection from display timing.
        raster_elapsed = time.perf_counter() - raster_start
        rgba_start = time.perf_counter()
        if display is not None:
            raster = display.render(raster, image_size)
        rgba_elapsed = time.perf_counter() - rgba_start if gpu_display else 0.0
        total_elapsed = time.perf_counter() - frame_start

        return {
            "frame_pos": frame_pos,
            "state": state_out,
            "raster": raster,
            "source_raster": source_raster,
            "display_view": image_size,
            "periods": (settings.get("t_in", 0.0) + stop_step*dt) / drive_period,
            "steps": steps_this_frame,
            "solve_s": solve_elapsed,
            "kernel_s": timing_info["gpu_kernel_s"],
            "timing_info": timing_info,
            "raster_s": raster_elapsed,
            "rgba_s": rgba_elapsed,
            "total_s": total_elapsed,
        }

    overlap = bool(settings.get("overlap_gpu_with_matplotlib", False))

    def init_animation():
        return [phase_img, phase_title]

    def update(frame_pos):
        callback_start = time.perf_counter()
        callback_interval = (callback_start - live["last_callback_start"]
                             if live["last_callback_start"] is not None else np.nan)
        outside_callback = (callback_start - live["last_callback_end"]
                            if live["last_callback_end"] is not None else np.nan)
        frame_pos = int(frame_pos)
        # Saving then playing the animation must restart both state and physical time.
        if frame_pos <= live["last_frame"]:
            if live["future"] is not None:
                live["future"].result()
            live.update(state=initial_state, future=None, ema_ms_per_step=None)
            live["playback_pass"] += 1
            callback_interval = outside_callback = np.nan
        live["last_frame"] = frame_pos

        if overlap:
            # Optional one-frame producer/consumer pipeline: frame N+1 can calculate
            # while Matplotlib draws frame N.  There is still no pre-render pass.
            if live["future"] is None:
                live["future"] = live["executor"].submit(
                    calculate_frame, frame_pos, live["state"], display_size()
                )
            frame = live["future"].result()
            live["state"] = frame["state"]

            next_pos = frame_pos + 1
            live["future"] = None
            if next_pos < frame_count:
                live["future"] = live["executor"].submit(
                    calculate_frame, next_pos, live["state"], display_size()
                )
        else:
            frame = calculate_frame(frame_pos, live["state"], display_size())
            live["state"] = frame["state"]

        artist_start = time.perf_counter()
        if direct_draw:
            phase_img.set_frame(frame["raster"], frame["source_raster"], frame["display_view"])
        else:
            phase_img.set_data(frame["raster"])
        phase_title.set_text(
            f"t/T = {frame['periods']:.3f}"
        )
        artist_elapsed = time.perf_counter() - artist_start

        # Normalize solver timing by step count.  Raw frame times can differ simply
        # because 1024 total steps do not divide evenly into an arbitrary frame count.
        ms_per_step = 1000.0 * frame["kernel_s"] / frame["steps"]
        if live["ema_ms_per_step"] is None:
            live["ema_ms_per_step"] = ms_per_step
        else:
            live["ema_ms_per_step"] = 0.9 * live["ema_ms_per_step"] + 0.1 * ms_per_step

        log_every = int(settings.get("frame_log_every", 10))
        if log_every > 0 and (frame_pos % log_every == 0 or frame_pos == frame_count-1):
            line = (f"frame {frame_pos + 1:3d}/{frame_count}  "
                    f"steps={frame['steps']:2d}  GPU={frame['kernel_s']:.4f}s  solve_call={frame['solve_s']:.4f}s  "
                    f"{ms_per_step:.2f} ms/step  EMA={live['ema_ms_per_step']:.2f} ms/step  "
                    f"raster={frame['raster_s']:.4f}s  RGBA+copy={frame['rgba_s']:.4f}s  "
                    f"compute_total={frame['total_s']:.4f}s")
            # Terminal flushing can stall playback; it is not part of compute_total.
            print(f"\r{line:<190}", end="\n" if frame_pos == frame_count-1 else "", flush=True)
        callback_end = time.perf_counter()
        if log_to_ram:
            info = frame["timing_info"]
            frame_log.append((live["log_phase"], live["playback_pass"], frame_pos+1,
                frame["steps"], frame["periods"]*drive_period, callback_start-total_start,
                callback_interval, outside_callback, callback_end-callback_start,
                artist_start-callback_start, artist_elapsed, frame["kernel_s"],
                frame["kernel_s"]/frame["steps"], frame["solve_s"], frame["raster_s"],
                frame["total_s"], info["rhs_stage_s"], info["cached_rhs"], info["cached_kernel"],
                frame["rgba_s"], frame["raster"].shape[1], frame["raster"].shape[0]))
        live["last_callback_start"] = callback_start
        live["last_callback_end"] = time.perf_counter()
        return [phase_img, phase_title]

    animation = None
    if settings["animate"] or settings["save_mp4"]:
        if overlap:
            live["executor"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gqis-animation")
            live["future"] = live["executor"].submit(calculate_frame, 0, live["state"], display_size())

        animation = FuncAnimation(
            fig,
            update,
            init_func=init_animation,
            frames=range(frame_count),
            interval=interval_ms,  # GUI timer pacing; no sleep blocking the window.
            blit=use_blit,
            repeat=False,
            cache_frame_data=False,
        )

        if settings["save_mp4"]:
            live["log_phase"] = "video_export"
            video_path = folder / settings["video_filename"]
            writer = FFMpegWriter(
                fps=settings["video_fps"], codec="libx264", bitrate=-1,
                extra_args=[
                    "-preset", settings["ffmpeg_preset"],
                    "-crf", str(settings["ffmpeg_crf"]),
                    "-pix_fmt", "yuv420p",
                ],
            )
            save_start = time.perf_counter()
            animation.save(video_path, writer=writer, dpi=settings["video_dpi"])
            live.update(log_phase="interactive", last_callback_start=None, last_callback_end=None)
            print(f"\nSaved: {video_path}")
            print(f"Real-time calculation + MP4 export: {time.perf_counter() - save_start:.2f}s")

    print(f"\nInteractive setup time: {time.perf_counter() - total_start:.2f}s")
    print("Frames are calculated by GQIS during playback; there is no frame pre-render pass.")

    try:
        if settings["show_figure"]:
            plt.show(block=True)  # Flush RAM logs only after the animation window closes.
        else:
            plt.close(fig)
    finally:
        if live["executor"] is not None:
            live["executor"].shutdown(wait=True, cancel_futures=True)
        if log_to_ram and frame_log:
            filename = settings.get("frame_log_filename") or (
                f"Example_06_frame_timings_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
            log_path = Path(filename)
            if not log_path.is_absolute():
                log_path = folder / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(log_columns)
                writer.writerows(frame_log)
            log_path.with_suffix(".settings.json").write_text(
                json.dumps(settings, indent=2), encoding="utf-8")
            print(f"Saved {len(frame_log)} frame timing records: {log_path}")

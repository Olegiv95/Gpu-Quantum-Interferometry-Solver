"""Example 06: animate a Duffing oscillator cloud with the general GQIS ODE API.

Change the symbolic RHS and initial conditions below to try another two-variable
ODE. Rendering, video export and timing logs live in fast_gpu_render.py.
"""
from pathlib import Path

import cupy as cp
import numpy as np
import sympy as sp

from gqis import odesolve_2D
try:  # Script launch or import as Examples.Example_06_symbolic_ode_sweep.
    from .fast_gpu_render import animate_phase_space
except ImportError:
    from fast_gpu_render import animate_phase_space


def main(settings):
    # Duffing: x'' = x - x**3 - damping*x' + amplitude*sin(omega*t).
    # State order in rhs, state_symbols and initial_state must agree.
    x, v, t = sp.symbols("x v t", real=True)
    damping, amplitude, omega = sp.symbols("damping amplitude omega", real=True)
    rhs = [v, x - x**3 - damping*v + amplitude*sp.sin(omega*t)]
    constants = {damping: settings["attractor_damping"],
                 amplitude: settings["attractor_amplitude"],
                 omega: settings["attractor_frequency"]}

    # Independent initial conditions on a square grid; upload the state once.
    dtype = np.float64 if settings["fp64"] else np.float32
    side = int(settings["attractor_grid_size"])
    positions = np.linspace(*settings["attractor_position_range"], side, dtype=dtype)
    velocities = np.linspace(*settings["attractor_velocity_range"], side, dtype=dtype)
    X0, V0 = np.meshgrid(positions, velocities, indexing="xy")
    initial_state = cp.asarray(np.stack((X0, V0), axis=-1))
    initial_color = X0.ravel().astype(np.float32)  # One-time colour labels.
    # Index axes preserve the 2D launch geometry; neither appears in this RHS.
    row, col = sp.symbols("trajectory_row trajectory_col", real=True)
    axes = {row: cp.arange(side, dtype=dtype), col: cp.arange(side, dtype=dtype)}
    period = 2*np.pi / settings["attractor_frequency"]
    dt = period / settings["attractor_steps_per_period"]
    total_steps = max(1, int(round(settings["attractor_periods"] *
                                  settings["attractor_steps_per_period"])))

    # Keep the GPU workload identical for every animation frame.  The renderer
    # normally partitions total_steps among animation_frames, which can produce
    # e.g. 1-step and 2-step frames when the numbers are not divisible.  Here the
    # user selects an exact integer number of solver steps per frame instead.
    steps_per_frame = max(1, int(settings["animation_steps_per_frame"]))
    if total_steps % steps_per_frame != 0:
        raise ValueError(
            "For constant solver work per frame, animation_steps_per_frame must "
            f"divide total_steps exactly: total_steps={total_steps}, "
            f"animation_steps_per_frame={steps_per_frame}."
        )
    render_settings = dict(settings)
    render_settings["animation_frames"] = total_steps // steps_per_frame

    def advance(state, t_in, steps):
        # Continue from the previous GPU state. Elapsed tlist starts at zero;
        # t_in sets absolute time for the drive, with no state download.
        tlist = np.arange(steps+1, dtype=np.float64) * dt
        return odesolve_2D(
            rhs, (x, v), None, tlist, var_arrays=axes, y0_values=state,
            time_symbol=t, const_values=constants, solver=settings["solver"],
            solver_frequency=settings["attractor_frequency"], fp64=settings["fp64"],
            t_in=t_in, RHSreuse=True, return_device=True, return_timing_info=True,
            timings=settings.get("gqis_frame_timings", False),
        )

    animate_phase_space(advance, initial_state, initial_color, render_settings,
                        drive_period=period, dt=dt, total_steps=total_steps,
                        output_folder=Path(__file__).resolve().parent / "results")


if __name__ == "__main__":
    USER_SETTINGS = {
        # Duffing equation and initial-condition cloud (the attractor depends on parameters).
        "attractor_damping": 0.02,        # Coefficient of velocity damping.
        "attractor_amplitude": 3.0,       # Sinusoidal forcing amplitude.
        "attractor_frequency": 1.0,       # Angular frequency, radians per unit time.
        "t_in": 0.0,                     # Absolute time of the supplied initial cloud.
        "attractor_periods": 40,           # Additional simulated periods;
        "attractor_steps_per_period": 128, # FDefines solver step size.
        "animation_steps_per_frame": 1, # Exact fixed solver steps per frame; must divide total steps.
        "attractor_grid_size": 1024,       # Trajectories = side squared (1024 squared = 1,048,576).
        "attractor_position_range": (-2.0, 2.0), # Initial positions.
        "attractor_velocity_range": (-2.0, 2.0), # Initial velocities.
        "attractor_plot_x_range": (-3.2, 3.2),   # Positions visible range, horizontal axis.
        "attractor_plot_v_range": (-5.0, 5.0),   # Velocities visible range, vertical axis.

        # Display only: these settings do not change the ODE calculation.
        # Advanced switches and video/log defaults: fast_gpu_render.RENDER_DEFAULTS.
        # Override them here if needed, e.g. "direct_rgba_draw": False.
        "attractor_render_bins": (512, 512), # Raster width, height; unrelated to trajectory count.
        "attractor_cmap": "turbo",        # Pixel color: mean initial x of trajectories in that bin.
        "attractor_interpolation": "bilinear",  # GPU display: "bilinear" (smooth) or "nearest".

        # Solver / real-time animation.
        "solver": "dop853",              # "rk4", "lserk4", "dp5", "tsit5", "anas5", "ab5", "alshina6", "dop853".
        "fp64": False,                    # State/integration precision; raster always FP32.
        "animate": True,                  # False shows initial cloud only, unless saving a video.
        "animation_max_fps": 60,         # Interactive FPS cap; 0 = uncapped. Does not change dt.
        "frame_log_to_ram": False,        # Collect every callback; save once after closing the window.

        "save_mp4": False,                # Calculate/export frames; requires FFmpeg.
        "video_fps": 60,                  # Playback rate; does not change solver step size.
    }
    main(USER_SETTINGS)

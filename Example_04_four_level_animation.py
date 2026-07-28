"""Example 04: four-level animation tutorial using the GQIS backend."""

import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.animation import FFMpegWriter, FuncAnimation

from gqis import mesolve_2D


def tanh_space(start, stop, num, stretch_factor=3.0):
    """Smoothly dense-at-edges sweep, useful for kappa scans."""
    # Linear sweeps can miss sharp edge behavior. This helper gives more points
    # near the endpoints while still covering the full interval.
    x = np.linspace(-stretch_factor, stretch_factor, num, dtype=np.float32)
    y = np.tanh(x)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return start + (stop - start) * y


def build_four_level_model(nph):
    """Build symbolic four-level model once and reuse across frames."""
    N = nph + 2

    # `eps` and `A` are the two sweep axes. The remaining symbols are constants
    # that can be kept runtime-adjustable between animation frames.
    eps, Drive = sp.symbols("eps Drive", real=True)
    A, t = sp.symbols("A t", real=True)
    delta_c, w_c, gamma1_c, gamma2_c, kappa_c, Ap_c, g1_c, wr2_c, wpr_c = sp.symbols(
        "delta_c w_c gamma1_c gamma2_c kappa_c Ap_c g1_c wr2_c wpr_c", real=True
    )

    sz = sp.Matrix([[1, 0], [0, -1]])       # qubit sigma_z
    sp_raise = sp.Matrix([[0, 1], [0, 0]])  # raising operator in this basis convention
    sp_lower = sp.Matrix([[0, 0], [1, 0]])  # lowering operator in this basis convention
    qeye = sp.eye(N - 2)                    # identity in the resonator subspace

    # Lift qubit and resonator operators into the full Hilbert space.
    sm1 = sp.kronecker_product(sp_lower, qeye)
    sz1 = sp.kronecker_product(sz, qeye)
    a1 = sp.kronecker_product(qeye, sp_raise)
    p0 = sp.zeros(N)
    p0[0, 0] = 1  # projector onto the selected qubit basis state |0>

    # Hamiltonian terms. The Drive placeholder is evaluated from drive_expr at
    # each RK4 stage in the generated CUDA kernel.
    hq11 = sz1 / 2
    hr11 = (wr2_c - wpr_c) * a1.H * a1
    hp = Ap_c * (a1.H + a1)
    hc11u = g1_c * delta_c * (sm1.H * a1 + sm1 * a1.H)
    H = hr11 + hp + hc11u / sp.sqrt(delta_c**2 + Drive**2) + hq11 * (sp.sqrt(delta_c**2 + Drive**2) - wr2_c)
    drive_expr = eps + A * sp.cos(w_c * t)

    # Lindblad collapse operators: qubit relaxation, qubit dephasing, and
    # resonator relaxation.
    col_ops = [
        sp.kronecker_product(sp.sqrt(gamma1_c) * sp_lower, qeye),
        sp.kronecker_product(sp.sqrt(gamma2_c) * sz, qeye),
        sp.sqrt(kappa_c) * a1,
    ]

    return {
        "H": H,
        "Drive": drive_expr,
        "Col_Ops": col_ops,
        "Mean_Response": a1,
        "Mean_Population": p0,
        "eps_sym": eps,
        "A_sym": A,
        "const_syms": {
            "delta": delta_c,
            "w": w_c,
            "gamma1": gamma1_c,
            "gamma2": gamma2_c,
            "kappa": kappa_c,
            "Ap": Ap_c,
            "g1": g1_c,
            "wr2": wr2_c,
            "wpr": wpr_c,
        },
    }


def build_time_grid(delta, wd_hz, periods, samples_per_period):
    """Convert drive frequency and build uniform time list."""
    # GQIS uses fixed-step RK4. Include both endpoints by returning one
    # more time sample than integration steps.
    w = wd_hz / 1000.0 * delta
    T = 2.0 * np.pi / w
    num_steps = int(periods * samples_per_period)
    tlist = np.linspace(0.0, periods * T, num_steps + 1, dtype=np.float32)
    return w, T, tlist


def frame_params(mode, value, base, wq1):
    """Apply selected sweep mode to base parameters."""
    # Only one physical parameter is changed per animation frame. Keeping the
    # symbolic model fixed lets GQIS reuse the compiled RHS where possible.
    params = dict(base)
    subtitle = "Static parameters"

    if mode == "wd1":
        params["wd_hz"] = float(value)
        subtitle = f"wd={params['wd_hz']:.1f} MHz"
    elif mode == "gamma1":
        params["gamma1"] = float(value)
        subtitle = f"gamma1={float(value):.4f}*Delta"
    elif mode in ("gamma2", "gammaph"):
        params["gamma2"] = float(value)
        subtitle = f"gammaph={float(value):.4f}*Delta"
    elif mode == "kappa":
        params["kappa"] = float(value)
        subtitle = f"kappa={float(value):.4f}*Delta"
    elif mode == "g1":
        params["g1"] = float(value) * wq1 / 1.23
        subtitle = f"g={float(value):.4f}*Delta"
    elif mode == "Ap":
        params["Ap"] = float(value) * wq1
        subtitle = f"Ap={float(value):.4f}*Delta"
    elif mode == "Same":
        subtitle = "Static parameters"
    else:
        raise ValueError(f"Unsupported mode '{mode}'.")

    return params, subtitle


def gpu_time_evolution(
    A_list,
    eps_list,
    tlist,
    model,
    params,
    output_mode,
    sweep_mode,
    *,
    RHSreuse=True,
    warmup_time=0.0,
    timings=False,
    fp64=False,
):
    """Run one 2D GPU sweep for the four-level model."""
    # Values in const_values either get folded into the generated RHS or, for
    # selected sweep modes below, remain runtime constants so animation frames do
    # not require full symbolic regeneration.
    const_syms = model["const_syms"]
    const_values = {
        const_syms["delta"]: float(params["delta"]),
        const_syms["w"]: float(params["w"]),
        const_syms["gamma1"]: float(params["gamma1"]),
        const_syms["gamma2"]: float(params["gamma2"]),
        const_syms["kappa"]: float(params["kappa"]),
        const_syms["Ap"]: float(params["Ap"]),
        const_syms["g1"]: float(params["g1"]),
        const_syms["wr2"]: float(params["wr2"]),
        const_syms["wpr"]: float(params["wr2"]),
    }

    # Constants listed here are passed at runtime for the corresponding sweep
    # mode. Other constants are substituted before CUDA code generation.
    mode_to_runtime_names = {
        "wd1": {"w"},
        "gamma1": {"gamma1"},
        "gamma2": {"gamma2"},
        "gammaph": {"gamma2"},
        "kappa": {"kappa"},
        "g1": {"g1"},
        "Ap": {"Ap"},
        "Same": set(),
    }
    keep_names = mode_to_runtime_names.get(sweep_mode, set())
    keep_syms = {const_syms[name] for name in keep_names}
    runtime_consts = {s: const_values[s] for s in keep_syms}

    # Choose which observable is averaged over time.
    if output_mode == "Response_Coef":
        mean_op = model["Mean_Response"]
    elif output_mode == "Qubit_occupation":
        mean_op = model["Mean_Population"]
    else:
        raise ValueError("output_mode must be 'Response_Coef' or 'Qubit_occupation'.")

    result = mesolve_2D(
        model["H"],                          # symbolic Hamiltonian
        model["Drive"],                      # time-dependent drive expression
        model["Col_Ops"],                    # Lindblad collapse operators
        mean_op,                             # observable to average
        tlist,                               # fixed RK4 time grid
        var_arrays={model["eps_sym"]: eps_list, model["A_sym"]: A_list},  # 2D sweep axes
        const_values=const_values,           # values for symbolic constants
        RHSreuse=RHSreuse,                   # reuse compiled kernel when possible
        runtime_consts=runtime_consts,       # constants changed between frames
        keep_symbolic_consts=keep_syms if keep_syms else None,
        output_mode="mean",                 # average observable over time
        warmup_time=warmup_time,             # initial fraction of time excluded from the time average
        timings=timings,                     # True prints codegen/kernel timing
        fp64=fp64,                           # True uses double precision
    )
    return np.abs(np.real(np.asarray(result))).T


def main() -> None:
    start_total = time.time()
    settings = user_settings()

    # Fixed model constants ------------------------------------------------
    # These define the base physical scales for the four-level model.
    delta = settings["delta"]
    wq1 = settings["wq1"]
    wr2 = settings["wr2"]
    nph = settings["nph"]

    # Base parameter set (keep one line active) ---------------------------
    # Each line is a different regime. Keep one active to define the animation.
    wd_hz = settings["wd_hz"]; gammaph = settings["gammaph"]; gamma1 = settings["gamma1"]; gamma2 = settings["gamma2"]; kappa = settings["kappa"]; Ap = settings["Ap"]; g1 = settings["g1"]; periods = settings["periods"]; epsmimax = settings["epsmimax"]; amax = settings["amax"]; warmup_time = settings["warmup_time"]

    # Match Example_02 units before entering the symbolic model.
    delta_abs = wq1 * delta
    Ap_abs = Ap * wq1
    g1_abs = g1 * wq1
    epsmimax_abs = epsmimax * wq1
    amax_abs = amax * wq1

    # Sweep and animation settings ----------------------------------------
    # Total work per frame scales as grid_size^2 * periods * samples_per_period.
    frame_count = settings["frame_count"]
    mode = settings["mode"]
    varray = settings["varray"](frame_count)

    output_mode = settings["output_mode"]  # "Response_Coef" or "Qubit_occupation"
    playback_mode = settings["playback_mode"]     # "forward" or "pingpong"
    samples_per_period = settings["samples_per_period"]
    grid_size = settings["grid_size"]
    save_mp4 = settings["save_mp4"]
    video_name = settings["video_name"]
    fps = settings["fps"]
    dpi = settings["dpi"]
    ffmpeg_preset = settings["ffmpeg_preset"]
    ffmpeg_crf = settings["ffmpeg_crf"]

    base = {
        "delta": float(delta_abs),
        "wd_hz": float(wd_hz),
        "gamma1": float(gamma1),
        "gamma2": float(gamma2),
        "kappa": float(kappa),
        "Ap": float(Ap_abs),
        "g1": float(g1_abs),
        "wr2": float(wr2),
    }

    # Build static axes and symbolic model once ---------------------------
    # The symbolic model is reused for all frames; only selected constants change.
    eps_list = np.linspace(-epsmimax_abs, epsmimax_abs, grid_size, dtype=np.float32)
    A_list = np.linspace(0.0, amax_abs, grid_size, dtype=np.float32)
    model = build_four_level_model(nph=nph)

    # Build frame order. Pingpong mode plays forward then backward for smoother loops.
    if playback_mode == "pingpong" and len(varray) > 1:
        frame_indices = np.concatenate(
            [np.arange(len(varray), dtype=np.int32), np.arange(len(varray) - 2, -1, -1, dtype=np.int32)]
        )
    else:
        frame_indices = np.arange(len(varray), dtype=np.int32)

    # Initial frame --------------------------------------------------------
    # This call may include symbolic RHS generation and CUDA compilation. Later
    # frames can reuse the compiled RHS when only runtime constants change.
    params0, subtitle0 = frame_params(mode, float(varray[0]), base, wq1)
    w0, _, tlist0 = build_time_grid(delta=delta, wd_hz=params0["wd_hz"], periods=periods, samples_per_period=samples_per_period)
    params0["w"] = w0
    map0 = gpu_time_evolution(
        A_list,
        eps_list,
        tlist0,
        model,
        params0,
        output_mode,
        mode,
        RHSreuse=True,
        warmup_time=warmup_time,
        timings=False,
        fp64=False,
    )

    if output_mode == "Response_Coef":
        img0 = 10.0 * np.log10(np.clip(map0, 1.0e-30, None))
        cbar_label = r"$|S_{21}|\ (\mathrm{dB})$"
    else:
        img0 = map0
        cbar_label = "Qubit population"

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(
        img0,
        aspect="auto",
        cmap="jet" if output_mode == "Response_Coef" else "viridis",
        origin="lower",
        extent=[eps_list[0] / wq1, eps_list[-1] / wq1, 0.0, amax_abs / wq1],
    )
    fig.colorbar(img, ax=ax, label=cbar_label)
    ax.set_xlabel(r"$\epsilon/\Delta$")
    ax.set_ylabel(r"$A/\Delta$")
    title = ax.set_title(f"Example 04 | mode={mode} | {subtitle0}")
    plt.tight_layout()

    def update(frame_pos):
        # Matplotlib calls this once per frame. Each call computes a full 2D GPU
        # sweep for the current value of the animated parameter.
        frame_start = time.time()
        frame_idx = int(frame_indices[frame_pos])
        value = float(varray[frame_idx])
        params, subtitle = frame_params(mode, value, base, wq1)
        w_now, _, tlist = build_time_grid(
            delta=delta,
            wd_hz=params["wd_hz"],
            periods=periods,
            samples_per_period=samples_per_period,
        )
        params["w"] = w_now
        map_lin = gpu_time_evolution(
            A_list,
            eps_list,
            tlist,
            model,
            params,
            output_mode,
            mode,
            RHSreuse=True,
            warmup_time=warmup_time,
            timings=False,
            fp64=False,
        )

        if output_mode == "Response_Coef":
            img_data = 10.0 * np.log10(np.clip(map_lin, 1.0e-30, None))
        else:
            img_data = map_lin

        img.set_data(img_data)
        img.set_clim(float(np.nanmin(img_data)), float(np.nanmax(img_data)))
        title.set_text(f"Example 04 | mode={mode} | {subtitle}")
        print(
            f"frame {frame_pos + 1}/{len(frame_indices)} calc={time.time() - frame_start:.2f}s",
            end="\r",
            flush=True,
        )
        return [img, title]

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=20, blit=False)

    if save_mp4:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=-1,
            extra_args=["-preset", ffmpeg_preset, "-crf", str(ffmpeg_crf), "-pix_fmt", "yuv420p"],
        )
        ani.save(video_name, writer=writer, dpi=dpi)
        print(f"\nSaved: {video_name}")

    plt.show()
    print(f"Total runtime: {time.time() - start_total:.2f}s")


def user_settings() -> dict:
    """User-editable parameters."""
    wq1 = 4.71 * 1.15
    gamma1 = 2.0e-3
    gammaph = 1.0e-3
    return {
        # Fixed model constants.
        "delta": 1.0,       
        "wq1": wq1,         # Qubit energy gap
        "wr2": 7.6767,      # resonator frequency
        "nph": 2,           # resonator truncation, how many photons in resonator to take into the model

        # Base parameter set. Keep one active by editing this block.
        # These are unscaled input values, matching Example_02; main() converts
        # them to the absolute units used by the symbolic model.
        "wd_hz": 500.0,                    # drive frequency
        "gammaph": gammaph,                # pure dephasing contribution
        "gamma1": gamma1,                  # qubit relaxation rate
        "gamma2": gamma1 / 2.0 + gammaph,  # total dephasing used in collapse operator
        "kappa": 5.0e-3,                   # resonator relaxation rate
        "Ap": 0.0002 / 10.0,               # weak probe amplitude
        "g1": 0.04033970276 / 1.23,        # coupling before
        "periods": 60,                     # number of driven periods per frame
        "epsmimax": 2.09916,               # detuning half-range
        "amax": 2.234042553191489 * 1.15,  # amplitude maximum
        "warmup_time": 0.0,                # initial fraction of time excluded from the time average

        # Alternative regime:
        # "wd_hz": 1500.0, "amax": 2.234042553191489 * 1.15 * 1.3, "warmup_time": 0.2

        # Sweep and animation settings.
        "frame_count": 25, # number of parameter values for simulation
        "mode": "g1",     # animated parameter: wd1, gamma1, gammaph, kappa, g1, Ap
        "varray": lambda frame_count: np.linspace(1.0e-7, 0.1, frame_count, dtype=np.float32), # values for mode
        # "mode": "wd1", "varray": lambda frame_count: np.linspace(500.0, 1500.0, frame_count, dtype=np.float32),
        # "mode": "gamma1", "varray": lambda frame_count: np.linspace(1.0e-3, 0.5, frame_count, dtype=np.float32),
        # "mode": "gammaph", "varray": lambda frame_count: np.linspace(1.0e-3, 0.5, frame_count, dtype=np.float32),
        # "mode": "kappa", "varray": lambda frame_count: tanh_space(5.0e-3, 8.0, frame_count),
        # "mode": "Ap", "varray": lambda frame_count: np.linspace(1.0e-4, 0.3, frame_count, dtype=np.float32),

        # Output settings.
        "output_mode": "Response_Coef", # "Response_Coef" or "Qubit_occupation"
        "playback_mode": "pingpong",    # "forward" or "pingpong"
        # Increase samples_per_period if mesolve_2D reports non-finite output.
        "samples_per_period": 250, # RK4 steps amount per drive period
        "grid_size": 512,          # square grid side per animation frame
        "save_mp4": False,         # False shows interactive animation only
        "video_name": "Example_04_four_level_animation.mp4", # output filename when save_mp4=True
        "fps": 5,                  # saved video frames per second
        "dpi": 120,                # saved video resolution scale
        "ffmpeg_preset": "medium", # x264 speed/compression preset
        "ffmpeg_crf": 23,          # x264 quality; lower is higher quality/larger file
    }


if __name__ == "__main__":
    main()

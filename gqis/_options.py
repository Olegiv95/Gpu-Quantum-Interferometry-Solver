"""Optional-argument groups shared by the GQIS solver entry points."""

from collections.abc import Mapping
import inspect

# Group members keep their flat argument names, so both call styles stay equivalent.
OPTION_GROUPS = {
    "consts": ("const_values", "runtime_consts", "keep_symbolic_consts", "auto_runtime_consts"),
    "initial": ("rho0", "rho0_var_arrays", "rho0_values", "y0_values", "t_in"),
    "output": ("warmup_time", "return_time_trace", "time_trace_every", "time_trace_samples_per_period",
               "solver_samples_per_period", "return_device"),
    "solver_options": ("solver", "fp64", "solver_frequency", "adaptive_options"),
    "codegen": ("RHSreuse", "kernel_template_file", "unroll", "pre_expand", "collect_rho", "factor_terms",
                "cse_batch_size", "cse_simplify", "hoist_rho_independent", "symbolic_workers",
                "max_symbolic_workers", "nvrtc_options"),
    "diagnostics": ("timings", "return_timing_info", "Actual_Kernel_Save", "beep_on_error",
                    "ignore_non_finite_output"),
}


def _is_default(value, default):
    """Return True when a flat argument still has its signature default."""
    if value is default:
        return True
    if isinstance(default, (bool, int, float, str, tuple)) and type(value) is type(default):
        return value == default
    return False


def apply_option_groups(func, args, groups, extra=None):
    """Merge grouped options into flat arguments of ``func``.

    args: flat named arguments of ``func``; groups: group name -> dict or None;
    extra: the ``**option_set`` dict of ``func``, if it forwards further options.
    An option may be given directly or inside its group, not both; unset options keep their defaults.
    """
    params = inspect.signature(func).parameters
    for group, values in groups.items():
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise TypeError(f"{group} must be a dict of option names and values.")
        for key, value in values.items():
            if key not in OPTION_GROUPS[group]:
                raise ValueError(f"Unknown {group} option '{key}'. "
                                 f"Valid options: {', '.join(OPTION_GROUPS[group])}.")
            if key in params:
                if not _is_default(args[key], params[key].default):
                    raise ValueError(f"'{key}' is given both directly and in {group}; use one.")
                args[key] = value
            elif extra is not None:
                if key in extra:
                    raise ValueError(f"'{key}' is given both directly and in {group}; use one.")
                extra[key] = value
            else:
                raise ValueError(f"{func.__name__} does not accept the {group} option '{key}'.")
    return args, extra

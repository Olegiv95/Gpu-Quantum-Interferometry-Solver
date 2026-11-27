"""Public interface for the GPU Quantum Interferometry Solver (GQIS)."""

from importlib import import_module

__all__ = ["build_independent_rho", "build_reduced_lindblad_rhs", "mesolve_2D", "odesolve_2D",
           "compile_lindblad_rhs", "compile_ode_rhs", "CompiledRHS"]
_ODE_API_NAMES = {"odesolve_2D", "compile_ode_rhs"}

__version__ = "0.3.0"


def __getattr__(name: str):
    """Load the CUDA solver only when a public solver function is requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        solver = import_module(".gqis_ode_api" if name in _ODE_API_NAMES else ".solver", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == "cupy":
            raise ImportError(
                "GQIS requires a CuPy package matching the CUDA major version. "
                "Install gqis[cuda12], gqis[cuda11], or gqis[cuda13]."
            ) from exc
        raise

    value = getattr(solver, name)
    globals()[name] = value
    return value

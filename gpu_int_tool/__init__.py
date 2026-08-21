"""Backward-compatible package alias for :mod:`gqis`.

New code should use ``from gqis import mesolve_2D``.
"""

from gqis import __version__, build_independent_rho, build_reduced_lindblad_rhs, mesolve_2D

__all__ = ["build_independent_rho", "build_reduced_lindblad_rhs", "mesolve_2D"]

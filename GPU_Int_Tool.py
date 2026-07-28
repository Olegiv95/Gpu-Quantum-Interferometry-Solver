"""Backward-compatible imports for scripts using the original module name.

New code should use ``from gqis import mesolve_2D``.
"""

from gqis import build_independent_rho, mesolve_2D

__all__ = ["build_independent_rho", "mesolve_2D"]

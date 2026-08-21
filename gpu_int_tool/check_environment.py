"""Backward-compatible entry point for :mod:`gqis.check_environment`."""

from gqis.check_environment import main

if __name__ == "__main__":
    raise SystemExit(main())

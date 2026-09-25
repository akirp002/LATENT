"""Shared, dependency-light portfolio analytics primitives.

The executable scripts remain compatibility entry points; new code should
import from this package instead of re-implementing risk calculations.
"""

from .risk import (
    BenchmarkIdentity,
    CVaRResult,
    TrackingErrorResult,
    active_cvar,
    scenario_matrix,
    tracking_error,
)

__all__ = [
    "BenchmarkIdentity",
    "CVaRResult",
    "TrackingErrorResult",
    "active_cvar",
    "scenario_matrix",
    "tracking_error",
]

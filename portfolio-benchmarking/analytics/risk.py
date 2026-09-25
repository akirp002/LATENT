"""Canonical CVaR, tracking-error, and scenario primitives.

All returns are decimal fractions.  CVaR is reported as a return (negative is
bad), matching the historical optimizer contract.  Keeping benchmark
identity beside results prevents a risk number from being detached from the
policy it measures.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkIdentity:
    """Stable identity for the benchmark used by a risk calculation."""

    name: str
    method: str
    factors: Tuple[str, ...] = ()
    version: str = "1"


@dataclass(frozen=True)
class CVaRResult:
    value: float
    alpha: float
    tail_count: int
    benchmark: BenchmarkIdentity


@dataclass(frozen=True)
class TrackingErrorResult:
    value: float
    variance: float
    benchmark: BenchmarkIdentity
    active_exposure: Tuple[float, ...] = ()


def scenario_matrix(
    returns_wide: pd.DataFrame,
    benchmark_returns: pd.Series,
    tickers: Sequence[str],
    window: int = 12,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build overlapping arithmetic-return scenarios on common history."""
    if window < 1:
        raise ValueError("window must be positive")
    combined = returns_wide.loc[:, list(tickers)].dropna(how="any").join(
        benchmark_returns.rename("Benchmark_Return"), how="inner"
    ).dropna()
    if len(combined) < window:
        raise ValueError(
            f"{len(combined)} common observations cannot form a {window}-period scenario"
        )
    values = np.array(
        [combined.iloc[i : i + window].sum().values for i in range(len(combined) - window + 1)]
    )
    return (
        pd.DataFrame(values[:, :-1].T, index=tickers),
        pd.Series(values[:, -1]),
    )


def active_cvar(
    weights: Sequence[float],
    scenarios: pd.DataFrame,
    benchmark_scenarios: Iterable[float],
    alpha: float = 0.05,
    benchmark: Optional[BenchmarkIdentity] = None,
) -> CVaRResult:
    """Return empirical worst-tail active return (negative means loss)."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    losses = np.asarray(list(benchmark_scenarios), dtype=float) - (
        scenarios.to_numpy(dtype=float).T @ np.asarray(weights, dtype=float)
    )
    count = max(1, int(np.ceil(alpha * len(losses))))
    return CVaRResult(
        value=float(-np.sort(losses)[-count:].mean()),
        alpha=alpha,
        tail_count=count,
        benchmark=benchmark or BenchmarkIdentity("unspecified", "unspecified"),
    )


def tracking_error(
    active_exposure: Sequence[float],
    covariance: np.ndarray,
    benchmark: BenchmarkIdentity,
    annualization: float = 1.0,
) -> TrackingErrorResult:
    """Calculate covariance-based TE with explicit benchmark identity."""
    exposure = np.asarray(active_exposure, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    variance = max(float(exposure @ cov @ exposure), 0.0)
    return TrackingErrorResult(
        value=float(np.sqrt(variance) * annualization),
        variance=variance * annualization**2,
        benchmark=benchmark,
        active_exposure=tuple(exposure),
    )

"""Executable smoke validation for the shared analytics package."""

import numpy as np
import pandas as pd

from .risk import BenchmarkIdentity, active_cvar, scenario_matrix, tracking_error


def main():
    benchmark = BenchmarkIdentity("policy-ff5", "buy-and-hold", ("Mkt-RF", "RMW"))
    idx = pd.date_range("2020-01-01", periods=14, freq="MS")
    returns = pd.DataFrame({"AAA": 0.01, "BBB": -0.005}, index=idx)
    scenarios, bench = scenario_matrix(returns, pd.Series(0.0, index=idx), ["AAA", "BBB"])
    result = active_cvar([0.5, 0.5], scenarios, bench, benchmark=benchmark)
    assert result.tail_count == 1 and result.benchmark == benchmark
    te = tracking_error([1.0, 0.0], np.eye(2), benchmark)
    assert np.isclose(te.value, 1.0) and te.benchmark == benchmark
    print("analytics validation: ok")


if __name__ == "__main__":
    main()

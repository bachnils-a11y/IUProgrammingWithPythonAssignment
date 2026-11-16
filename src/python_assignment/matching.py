"""Select supplied ideal functions by SSE and map test observations by tolerance."""
from dataclasses import dataclass
from collections.abc import Iterable
import math

import numpy as np
import pandas as pd

from .data import CourseData, DataValidationError, TestPoint


@dataclass(frozen=True)
class FunctionFit:
    """Record one training series' selected candidate and absolute error limit."""

    training_function: str
    ideal_function: str
    sse: float
    max_deviation: float
    threshold: float


def select_functions(data: CourseData) -> list[FunctionFit]:
    """Minimize sum of squared y residuals independently for each training series.

    Align rows by x, not file order. Exact SSE ties prefer the lowest numeric
    ideal-function number. Reusing a candidate for different series is allowed.
    Raises DataValidationError if numeric overflow prevents a reliable result.
    """
    # Align by x labels so different CSV row orders cannot change the scores.
    training = data.training.frame.set_index("x")
    ideal = data.ideal.frame.set_index("x").loc[training.index]
    # Numeric sorting places y2 before y10 and defines deterministic exact ties.
    candidates = sorted(ideal.columns, key=lambda name: int(name[1:]))
    values = ideal[candidates].to_numpy(dtype=float)
    fits = []
    for name in training.columns:
        with np.errstate(over="ignore", invalid="ignore"):
            # Broadcast one training column (n, 1) against all candidates (n, m).
            residuals = training[name].to_numpy(dtype=float)[:, None] - values
            # Sum squared residuals down the rows: one SSE score per candidate.
            scores = np.sum(residuals ** 2, axis=0)
        if not np.isfinite(scores).all():
            raise DataValidationError("Squared deviations overflow; rescale input data")
        # argmin returns a column position, not the ideal-function number.
        best = int(np.argmin(scores))
        # The matching limit uses the maximum absolute error, not the SSE.
        maximum = float(np.max(np.abs(residuals[:, best])))
        fits.append(FunctionFit(name, candidates[best], float(scores[best]),
                                maximum, math.sqrt(2) * maximum))
    return fits


def map_test_points(data: CourseData, fits: list[FunctionFit],
                    points: Iterable[TestPoint] | None = None) -> pd.DataFrame:
    """Map each test observation, preserving input order and repeated x values.

    Accept when absolute y deviation <= sqrt(2) * maximum training deviation.
    If several selected functions qualify, choose the smallest absolute deviation,
    then the smallest numeric function ID on exact ties. This tie policy is our
    documented assumption, not an explicit assignment requirement.
    Rejected rows retain x/y with null function/deviation; eligible_count records
    ambiguity. Signed residuals are not stored: delta_y is always absolute.
    """
    ideal = data.ideal.frame.set_index("x")
    if points is None:
        points = data.test.frame.itertuples(index=False)
    rows = []
    for observation, point in enumerate(points, 1):
        if point.x not in ideal.index:
            raise DataValidationError(f"Test x value absent from ideal data: {point.x}")
        # Several training series may select the same ideal function. Count each
        # distinct ideal candidate once, retaining the smallest eligible residual.
        eligible = {}
        for fit in fits:
            delta = abs(float(point.y) - float(ideal.at[point.x, fit.ideal_function]))
            if not math.isfinite(delta):
                raise DataValidationError("Test deviation overflow; rescale input data")
            if delta <= fit.threshold:
                number = int(fit.ideal_function[1:])
                eligible[number] = delta
        # Only eligible candidates compete: nearest first, numeric ID second.
        number = min(eligible, key=lambda n: (eligible[n], n)) if eligible else None
        # Preserve every observation; rejection is a result, not an exception.
        rows.append({
            "test_row": observation, "x": float(point.x), "y": float(point.y),
            "delta_y": eligible[number] if number is not None else None,
            "ideal_function": number, "eligible_count": len(eligible),
            "status": "assigned" if number is not None else "unassigned",
        })
    result = pd.DataFrame(rows, columns=[
        "test_row", "x", "y", "delta_y", "ideal_function", "eligible_count", "status"
    ])
    # Pandas nullable Int64 supports integer IDs and missing rejected assignments.
    result["ideal_function"] = result["ideal_function"].astype("Int64")
    return result

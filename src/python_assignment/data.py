"""Load the supplied CSV files and validate their structure and x coverage."""
from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when a course dataset violates its expected format."""


class TestPoint(NamedTuple):
    """One observation read from the test CSV, without discarding repeated x."""

    x: float
    y: float


def iter_test_points(path: Path) -> Iterator[TestPoint]:
    """Read test observations line by line; validate headers and finite numbers.

    Raises DataValidationError for malformed rows, empty input or file errors.
    """
    try:
        # Accept an optional UTF-8 BOM without including it in the first header.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames != ["x", "y"]:
                raise DataValidationError("test.csv: expected columns x,y")
            count = 0
            for count, row in enumerate(reader, 1):
                # DictReader uses None keys/values for extra/missing CSV fields.
                if None in row or any(value is None for value in row.values()):
                    raise DataValidationError(f"test.csv: malformed row {count}")
                point = TestPoint(float(row["x"]), float(row["y"]))
                if not all(math.isfinite(value) for value in point):
                    raise DataValidationError(f"test.csv: nonfinite row {count}")
                # Pause here until the caller requests the next observation.
                yield point
            if count == 0:
                raise DataValidationError("test.csv: dataset is empty")
    except DataValidationError:
        raise
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        raise DataValidationError(f"Cannot stream {path.name}: {error}") from error


@dataclass
class Dataset:
    """Store a named table; repeated x values are allowed for test points."""

    name: str
    frame: pd.DataFrame

    def validate(self, columns: list[str]) -> None:
        """Require exact columns, nonempty numeric data and finite values.

        Raises:
            DataValidationError: If columns or numeric values are invalid.
        """
        # Require the exact schema, including column order, before calculation.
        if list(self.frame.columns) != columns:
            raise DataValidationError(
                f"{self.name}: expected columns {columns}, "
                f"got {list(self.frame.columns)}"
            )
        if self.frame.empty:
            raise DataValidationError(f"{self.name}: dataset is empty")
        try:
            # Reject invalid values rather than silently replacing or dropping them.
            numeric = self.frame.apply(pd.to_numeric, errors="raise")
        except (ValueError, TypeError) as error:
            raise DataValidationError(
                f"{self.name}: contains nonnumeric values"
            ) from error
        # Numeric conversion alone does not reject NaN or positive/negative infinity.
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise DataValidationError(f"{self.name}: contains missing/infinite values")
        self.frame = numeric


class FunctionDataset(Dataset):
    """A function table has exactly one row for each x value."""

    def validate(self, columns: list[str]) -> None:
        """Apply shared checks and reject ambiguous function values at one x."""
        # Reuse shared validation, then enforce the stricter function-table contract.
        super().validate(columns)
        if self.frame["x"].duplicated().any():
            raise DataValidationError(f"{self.name}: duplicate function x values")


@dataclass
class CourseData:
    """Group the validated training, ideal and test tables."""

    training: FunctionDataset
    ideal: FunctionDataset
    test: Dataset


def read_dataset(path: Path, columns: list[str], *, function: bool) -> Dataset:
    """Read and validate one CSV, preserving the cause of file/parser errors."""
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError,
            UnicodeError) as error:
        raise DataValidationError(f"Cannot read {path.name}: {error}") from error
    # Repeated x values are valid observations, but ambiguous function lookups.
    table_type = FunctionDataset if function else Dataset
    dataset = table_type(path.name, frame)
    dataset.validate(columns)
    return dataset


def load_course_data(directory: Path) -> CourseData:
    """Load all three files and require lookup coverage for every input x.

    Rows need not have matching order: later calculations must join by x.
    Repeated test x values represent separate observations and are retained.
    """
    training = read_dataset(
        directory / "train.csv", ["x", "y1", "y2", "y3", "y4"], function=True
    )
    ideal = read_dataset(
        directory / "ideal.csv", ["x"] + [f"y{i}" for i in range(1, 51)],
        function=True,
    )
    test = read_dataset(directory / "test.csv", ["x", "y"], function=False)
    # Sets remove duplicates only for this coverage check, not from the tables.
    ideal_x = set(ideal.frame["x"])
    for table in (training, test):
        # Set difference leaves required x values that have no ideal-function row.
        missing = set(table.frame["x"]) - ideal_x
        if missing:
            raise DataValidationError(
                f"{table.name}: x values absent from ideal.csv: {sorted(missing)[:5]}"
            )
    return CourseData(training, ideal, test)

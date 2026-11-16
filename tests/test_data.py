"""Test important validation failures and preserve repeated test observations."""
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from python_assignment.data import (
    Dataset, FunctionDataset, DataValidationError, load_course_data, read_dataset,
)


class DataValidationTests(unittest.TestCase):
    """Check input contracts independently of the student's supplied values."""

    def test_repeated_test_x_is_preserved(self):
        """Multiple observations at the same x must not be discarded."""
        data = Dataset("test", pd.DataFrame({"x": [1, 1], "y": [2, 3]}))
        data.validate(["x", "y"])
        self.assertEqual(len(data.frame), 2)

    def test_duplicate_function_x_is_rejected(self):
        """Function lookup must be unambiguous."""
        data = FunctionDataset("ideal", pd.DataFrame({"x": [1, 1], "y": [2, 3]}))
        with self.assertRaises(DataValidationError):
            data.validate(["x", "y"])

    def test_invalid_values_are_rejected(self):
        """Missing, infinite and nonnumeric values must not reach calculations."""
        # Exercise missing, infinite and textual input as separate subcases.
        for value in [None, float("inf"), "not-a-number"]:
            with self.subTest(value=value):
                data = Dataset("test", pd.DataFrame({"x": [1], "y": [value]}))
                with self.assertRaises(DataValidationError):
                    data.validate(["x", "y"])

    def test_wrong_columns_and_empty_data_are_rejected(self):
        """Reject malformed schema and files without observations."""
        for frame in [pd.DataFrame({"x": [1], "wrong": [2]}),
                      pd.DataFrame(columns=["x", "y"])]:
            with self.subTest(columns=list(frame.columns), rows=len(frame)):
                with self.assertRaises(DataValidationError):
                    Dataset("test", frame).validate(["x", "y"])

    def test_missing_file_preserves_original_error(self):
        """Report a meaningful domain error with the original file error."""
        # Isolate test files from the real course dataset.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(DataValidationError) as context:
                read_dataset(Path(directory) / "missing.csv", ["x", "y"],
                             function=False)
            self.assertIsInstance(context.exception.__cause__, FileNotFoundError)

    def test_coverage_and_unsorted_input(self):
        """Align by x rather than row order and reject unavailable test x."""
        # Isolate test files from the real course dataset.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"x": [2, 1], **{f"y{i}": [2, 1] for i in range(1, 5)}}
                         ).to_csv(root / "train.csv", index=False)
            pd.DataFrame({"x": [1, 2], **{f"y{i}": [1, 2] for i in range(1, 51)}}
                         ).to_csv(root / "ideal.csv", index=False)
            pd.DataFrame({"x": [1, 1], "y": [2, 3]}).to_csv(
                root / "test.csv", index=False)
            self.assertEqual(len(load_course_data(root).test.frame), 2)
            pd.DataFrame({"x": [9], "y": [2]}).to_csv(root / "test.csv", index=False)
            with self.assertRaisesRegex(DataValidationError, "absent"):
                load_course_data(root)


if __name__ == "__main__":
    unittest.main()

"""Hand-computable examples for function selection and mapping boundary cases."""
from pathlib import Path
import math
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from python_assignment.data import CourseData, Dataset, FunctionDataset
from python_assignment.matching import FunctionFit, map_test_points, select_functions


def example(training, ideal, test=None):
    """Build small in-memory datasets without depending on supplied course data."""
    return CourseData(FunctionDataset("train", pd.DataFrame(training)),
                      FunctionDataset("ideal", pd.DataFrame(ideal)),
                      Dataset("test", pd.DataFrame(test or {"x": [0], "y": [0]})))


class MatchingTests(unittest.TestCase):
    """Exercise mathematical contracts, exact ties and rejected observations."""

    def test_sse_x_alignment_and_independent_selection(self):
        """A reversed CSV order must not change SSE or chosen function."""
        data = example({"x": [0, 1], "y1": [1, 3], "y2": [9, 9]},
                       {"x": [1, 0], "y1": [2, 0], "y2": [9, 9]})
        fits = select_functions(data)
        self.assertEqual([fit.ideal_function for fit in fits], ["y1", "y2"])
        self.assertEqual(fits[0].sse, 2)
        self.assertEqual(fits[0].max_deviation, 1)
        self.assertAlmostEqual(fits[0].threshold, math.sqrt(2))
        self.assertEqual(fits[1].sse, 0)

    def test_select_by_squares_not_absolute_sum(self):
        """Residuals [2,2] beat [0,3] by SSE despite larger absolute sum."""
        data = example({"x": [0, 1], "y1": [0, 0]},
                       {"x": [0, 1], "y1": [0, 3], "y2": [2, 2]})
        self.assertEqual(select_functions(data)[0].ideal_function, "y2")

    def test_exact_selection_tie_uses_numeric_id(self):
        """Numeric ID ordering must prefer y2 over y10 on identical SSE."""
        data = example({"x": [0], "y1": [1]},
                       {"x": [0], "y10": [0], "y2": [2]})
        self.assertEqual(select_functions(data)[0].ideal_function, "y2")

    def test_inclusive_boundary_and_rejection(self):
        """Accept the exact sqrt(2) limit; reject a point just above it."""
        # Check the inclusive boundary and a clearly distinguishable value above it.
        limit = math.sqrt(2)
        data = example({"x": [0], "y1": [1]}, {"x": [0], "y1": [0]},
                       {"x": [0, 0, 0], "y": [limit, limit + 1e-8, -0.5]})
        result = map_test_points(data, select_functions(data))
        self.assertEqual(result["status"].tolist(),
                         ["assigned", "unassigned", "assigned"])
        self.assertEqual(result["test_row"].tolist(), [1, 2, 3])
        self.assertTrue(pd.isna(result.iloc[1]["ideal_function"]))
        self.assertTrue(pd.isna(result.iloc[1]["delta_y"]))
        self.assertEqual(result.iloc[2]["delta_y"], 0.5)

    def test_multiple_matches_choose_nearest_then_numeric_id(self):
        """Both eligible candidates remain visible via the ambiguity count."""
        data = example({"x": [0], "y1": [0]},
                       {"x": [0], "y10": [0], "y2": [2]},
                       {"x": [0, 0], "y": [0.25, 1]})
        # Supply fixed limits to isolate mapping behavior from function selection.
        fits = [FunctionFit("y1", "y10", 0, 2, 3),
                FunctionFit("y2", "y2", 0, 2, 3)]
        result = map_test_points(data, fits)
        self.assertEqual(result["ideal_function"].tolist(), [10, 2])
        self.assertEqual(result["eligible_count"].tolist(), [2, 2])

    def test_per_function_limits(self):
        """An ineligible nearer function must not displace an eligible one."""
        data = example({"x": [0], "y1": [0]},
                       {"x": [0], "y1": [0], "y2": [1]},
                       {"x": [0], "y": [0.2]})
        # Supply fixed limits to isolate mapping behavior from function selection.
        fits = [FunctionFit("y1", "y1", 0, 0, 0),
                FunctionFit("y2", "y2", 0, 1, math.sqrt(2))]
        self.assertEqual(map_test_points(data, fits).iloc[0]["ideal_function"], 2)

    def test_zero_threshold_and_reused_candidate(self):
        """Exact fits allow only exact points; reused ideals count once."""
        data = example({"x": [0], "y1": [0], "y2": [0]},
                       {"x": [0], "y1": [0]},
                       {"x": [0, 0], "y": [0, 1e-8]})
        result = map_test_points(data, select_functions(data))
        self.assertEqual(result["eligible_count"].tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()

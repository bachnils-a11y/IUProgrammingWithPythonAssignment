"""Check persistence, streaming, visualization data and the complete CLI workflow."""
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from bokeh.models import ColumnDataSource, Plot, Span

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from python_assignment.data import DataValidationError, iter_test_points, load_course_data
from python_assignment.matching import map_test_points, select_functions
from python_assignment.storage import SQLiteRepository
from python_assignment.visualization import build_report, write_report


def make_inputs(root):
    """Create a hand-computable complete 4-series/50-candidate input set."""
    pd.DataFrame({"x": [0, 1], **{f"y{i}": [1, 3] for i in range(1, 5)}}
                 ).to_csv(root / "train.csv", index=False)
    # Reverse ideal rows to ensure matching uses x labels rather than row positions.
    columns = {"x": [1, 0], "y1": [2, 0]}
    columns.update({f"y{i}": [100+i, 100+i] for i in range(2, 51)})
    pd.DataFrame(columns).to_csv(root / "ideal.csv", index=False)
    pd.DataFrame({"x": [0, 0, 1], "y": [0.5, 1, 90]}
                 ).to_csv(root / "test.csv", index=False)


class StorageAndReportTests(unittest.TestCase):
    """Verify database shape, actual stored values, empty output and offline HTML."""

    def setUp(self):
        """Prepare independent temporary input/output paths for each test."""
        self.temporary = TemporaryDirectory()
        # unittest runs registered cleanup even when a test assertion fails.
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        make_inputs(self.root)
        self.data = load_course_data(self.root)
        self.fits = select_functions(self.data)
        self.assignments = map_test_points(self.data, self.fits)

    def test_exact_schema_values_and_repeated_x(self):
        """Persist all inputs and two different accepted observations at x=0."""
        path = self.root / "result.sqlite"
        repository = SQLiteRepository(path)
        try:
            repository.save_inputs(self.data)
            repository.save_assignments(self.assignments)
        finally:
            repository.close()
        # Inspect persistence through an independent SQLite connection.
        with closing(sqlite3.connect(path)) as connection:
            for table, count, width in [("training", 2, 5), ("ideal", 2, 51),
                                         ("test_results", 2, 4)]:
                self.assertEqual(connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"').fetchone()[0], count)
                self.assertEqual(len(connection.execute(
                    f'PRAGMA table_info("{table}")').fetchall()), width)
            self.assertEqual(connection.execute(
                "SELECT x,y,delta_y,ideal_function FROM test_results ORDER BY y"
            ).fetchall(), [(0., 0.5, 0.5, 1), (0., 1., 1., 1)])
            self.assertEqual(connection.execute(
                "SELECT y1 FROM training WHERE x=1").fetchone()[0], 3.)
            self.assertEqual(connection.execute(
                "SELECT y50 FROM ideal WHERE x=0").fetchone()[0], 150.)

    def test_empty_results_and_rerun_replace(self):
        """An all-rejected run clears earlier matches and retains the schema."""
        repository = SQLiteRepository(self.root / "result.sqlite")
        try:
            repository.save_inputs(self.data)
            repository.save_assignments(self.assignments)
            # Simulate a later run in which every observation is rejected.
            rejected = self.assignments.copy()
            rejected["status"] = "unassigned"
            repository.save_assignments(rejected)
            with repository.engine.connect() as connection:
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT COUNT(*) FROM test_results").scalar(), 0)
                self.assertEqual(len(connection.exec_driver_sql(
                    "PRAGMA table_info(test_results)").fetchall()), 4)
        finally:
            repository.close()

    def test_streaming_retains_rows_and_rejects_bad_input(self):
        """The CSV generator retains repeated x and reports invalid numbers."""
        points = iter_test_points(self.root / "test.csv")
        self.assertIs(iter(points), points)
        result = map_test_points(self.data, self.fits, points)
        pd.testing.assert_frame_equal(result, self.assignments)
        (self.root / "bad.csv").write_text("x,y\n0,NaN\n")
        with self.assertRaises(DataValidationError):
            list(iter_test_points(self.root / "bad.csv"))

    def test_unknown_streamed_x_is_rejected(self):
        """Coverage is checked even if a different point stream is supplied."""
        (self.root / "other.csv").write_text("x,y\n9,0\n")
        with self.assertRaises(DataValidationError):
            map_test_points(self.data, self.fits,
                            iter_test_points(self.root / "other.csv"))

    def test_report_data_and_offline_html(self):
        """All observations appear in the overview and BokehJS is embedded."""
        report = build_report(self.data, self.fits, self.assignments)
        # A threshold outside the visible axis range would make the report misleading.
        residual_plots = [plot for plot in report.select({"type": Plot})
                          if list(plot.select({"type": Span}))]
        self.assertEqual(len(residual_plots), 4)
        for plot in residual_plots:
            limit = list(plot.select({"type": Span}))[0].location
            self.assertLessEqual(plot.y_range.start, limit)
            self.assertGreater(plot.y_range.end, limit)
        sources = list(report.select({"type": ColumnDataSource}))
        self.assertTrue(any(list(s.data.get("status", [])) == ["unassigned"]
                            for s in sources))
        self.assertTrue(any(list(s.data.get("test_row", [])) == [1, 2]
                            for s in sources))
        path = self.root / "report.html"
        write_report(path, self.data, self.fits, self.assignments)
        html = path.read_text()
        self.assertIn("Bokeh", html)
        self.assertIn("Programm", html)
        self.assertNotIn('<script src="https://', html)

    def test_cli_success_and_failure_preserves_existing_results(self):
        """Run the real entry point; bad input must not overwrite a valid run."""
        script = Path(__file__).resolve().parents[1] / "main.py"
        output = self.root / "results"
        command = [sys.executable, "-B", str(script), "--data-dir", str(self.root),
                   "--output-dir", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        # Compare file contents to ensure invalid inputs preserve previous outputs.
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        self.assertEqual(set(before), {"assignment.sqlite", "visualizations.html",
                         "data_summary.json", "matching_summary.json",
                         "test_assignments.csv", "matched_points.csv"})
        (self.root / "test.csv").write_text("x,y\n9,0\n")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == "__main__":
    unittest.main()

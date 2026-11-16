"""Run validation, matching, SQLite persistence and an offline Bokeh report."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from sqlalchemy.exc import SQLAlchemyError

# Make the src package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from python_assignment.data import DataValidationError, iter_test_points, load_course_data
from python_assignment.matching import map_test_points, select_functions
from python_assignment.storage import SQLiteRepository
from python_assignment.visualization import write_report


def main() -> int:
    """Build a complete run in a temporary directory before publishing outputs.

    Input validation and generation failures leave previous results untouched.
    Final files are replaced individually, so publication is not a single
    transaction across all files if the operating system fails during replacement.
    """
    # Resolve defaults relative to the project, not the terminal working directory.
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the Python course analysis")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    args = parser.parse_args()
    try:
        data = load_course_data(args.data_dir)
        fits = select_functions(data)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Generate all outputs in staging before replacing previous results.
        with TemporaryDirectory(prefix=".run-", dir=args.output_dir) as directory:
            stage = Path(directory)
            repository = SQLiteRepository(stage / "assignment.sqlite")
            try:
                # The assignment requests persistence before streaming test data.
                repository.save_inputs(data)
                # The generator reads observations lazily during matching.
                points = iter_test_points(args.data_dir / "test.csv")
                assignments = map_test_points(data, fits, points)
                repository.save_assignments(assignments)
            finally:
                # Release SQLite connections even if matching or saving fails.
                repository.close()
            assigned = assignments[assignments["status"] == "assigned"]
            summary = {}
            for table in (data.training, data.ideal, data.test):
                frame = table.frame
                summary[table.name] = {
                    "rows": len(frame), "columns": len(frame.columns),
                    "x_min": float(frame["x"].min()),
                    "x_max": float(frame["x"].max()),
                    "repeated_x_rows": int(frame["x"].duplicated().sum()),
                }
            # Convert dataclasses to plain dictionaries for JSON serialization.
            matching_summary = {
                "selected_functions": [asdict(fit) for fit in fits],
                "test_points": len(assignments), "assigned": len(assigned),
                "unassigned": len(assignments) - len(assigned),
                "multiple_eligible": int((assignments["eligible_count"] > 1).sum()),
                "policy": "absolute delta <= sqrt(2) * max training delta; "
                          "multiple eligible: nearest, then lowest numeric ID",
            }
            for name, report in [("data_summary.json", summary),
                                 ("matching_summary.json", matching_summary)]:
                (stage / name).write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
            assignments.to_csv(stage / "test_assignments.csv", index=False)
            # Keep the required four-column export separate from the full audit CSV.
            assigned[["x", "y", "delta_y", "ideal_function"]].to_csv(
                stage / "matched_points.csv", index=False)
            write_report(stage / "visualizations.html", data, fits, assignments)
            # Each replacement is atomic individually, not across the entire run.
            for output in stage.iterdir():
                output.replace(args.output_dir / output.name)
    except (DataValidationError, OSError, SQLAlchemyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print("Data validation, calculation, SQLite storage and report generation completed successfully.")
    for fit in fits:
        print(f"{fit.training_function} -> {fit.ideal_function}: "
              f"SSE={fit.sse:.6f}, threshold={fit.threshold:.6f}")
    print(f"Test observations: {len(assigned)} assigned, "
          f"{len(assignments) - len(assigned)} unassigned; "
          f"{matching_summary['multiple_eligible']} with multiple eligible candidates.")
    print(f"Database: {args.output_dir / 'assignment.sqlite'}")
    print(f"Report: {args.output_dir / 'visualizations.html'}")
    return 0


# Importing this module must not start the analysis; direct execution should.
if __name__ == "__main__":
    raise SystemExit(main())

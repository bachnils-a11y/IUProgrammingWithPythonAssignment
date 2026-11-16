"""Persist the three required tables in a local SQLite file via SQLAlchemy."""
from pathlib import Path

import pandas as pd
from sqlalchemy import Float, Integer, create_engine
from sqlalchemy.engine import URL

from .data import CourseData


class SQLiteRepository:
    """Manage an assignment database; callers must close it after use."""

    def __init__(self, path: Path):
        """Connect to a SQLite file without requiring an external server."""
        # URL.create handles file paths without manually escaping URL characters.
        self.engine = create_engine(URL.create("sqlite", database=str(path)))

    def save_inputs(self, data: CourseData) -> None:
        """Store training/ideal tables with exactly 5/51 columns, no row index."""
        # Commit successful work and roll back the transaction on failure.
        with self.engine.begin() as connection:
            for name, frame in [("training", data.training.frame),
                                ("ideal", data.ideal.frame)]:
                # Do not export the DataFrame index as an extra database column.
                frame.to_sql(name, connection, if_exists="replace", index=False,
                             dtype={column: Float() for column in frame.columns})

    def save_assignments(self, assignments: pd.DataFrame) -> None:
        """Store only accepted points in exactly four columns; repeated x is valid.

        Rejected observations remain available in the complete CSV audit export.
        Empty results still create the four-column table with explicit SQL types.
        """
        # Rejected points are retained in the audit CSV, not this result table.
        accepted = assignments.loc[assignments["status"] == "assigned",
                                   ["x", "y", "delta_y", "ideal_function"]]
        # Commit successful work and roll back the transaction on failure.
        with self.engine.begin() as connection:
            # Explicit SQL types also define the schema when no points qualify.
            accepted.to_sql("test_results", connection, if_exists="replace",
                            index=False, dtype={"x": Float(), "y": Float(),
                                                "delta_y": Float(),
                                                "ideal_function": Integer()})

    def close(self) -> None:
        """Release pooled connections before moving/replacing a database file."""
        self.engine.dispose()

# Programming with Python - Written Assignment

Python implementation for course **DLMDSPWP01**. The program validates the supplied
CSV data, selects ideal functions by least squares, maps test observations,
stores the required SQLite tables and generates an interactive Bokeh report.

The implementation and written report have been prepared. The reported results
were independently recalculated, confirming the selected functions and assignment
counts. The written report is submitted separately and links to this repository
for the complete application and test code instead of including source listings
in its appendix.

## Project structure

```text
main.py                          Application entry point
src/python_assignment/
    __init__.py                  Package description
    data.py                      CSV loading, validation and test-point streaming
    matching.py                  Function selection and observation mapping
    storage.py                   SQLite persistence through SQLAlchemy
    visualization.py             Standalone Bokeh report
tests/                           Automated tests
requirements.txt                 Recorded package versions
data/raw/                        Local input CSV files (not tracked)
results/                         Generated outputs (not tracked)
```

## Setup

Clone the repository and enter its directory. For an existing local checkout,
start in that project folder and skip these two commands.

```bash
git clone https://github.com/bachnils-a11y/IUProgrammingWithPythonAssignment.git
cd IUProgrammingWithPythonAssignment
```

Create and activate a virtual environment, then install the dependencies.
If `.venv` already exists, skip its creation and activate it directly.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The activation command above is for macOS/Linux. In VS Code, use **Python: Select
Interpreter** to select this project's `.venv`.

The main external dependencies are NumPy, pandas, SQLAlchemy and Bokeh.
`requirements.txt` also records their installed dependencies. Tests use Python's
standard-library `unittest`; SQLite does not require a separate database server.

### Input data

The course dataset is **not included in this repository**. Section 1.4 (page 4)
of the official task specification states: "Your input data is not required."
The local MyCampus submission-rules summary requires the complete source code
in a repository but does not add a requirement to include input CSV files.

Obtain the assigned
`dataset.zip` separately from the course materials and extract it. Create the
local input directory:

```bash
mkdir -p data/raw
```

Alternatively, keep the extracted CSV files in their existing directory and run
`python main.py --data-dir /path/to/csv-files`; no copying into the project is
needed.

For the default setup, place the three extracted CSV files directly in
`data/raw/`, without an extra
nested directory. The expected files are:

| File | Required columns | Rows in the supplied dataset |
|---|---|---:|
| `train.csv` | `x`, `y1`, `y2`, `y3`, `y4` | 400 |
| `ideal.csv` | `x`, `y1` through `y50` | 400 |
| `test.csv` | `x`, `y` | 100 |

The row counts describe the supplied data; they are not hard-coded requirements.
The repository's `.gitignore` excludes `data/raw/`, `dataset.zip`, `results/` and
`Uni Docs/`. A fresh clone therefore contains neither the input data nor the
written report or generated results. The application creates its output directory
and files when run successfully.

## Run the application and tests

```bash
source .venv/bin/activate
python main.py
python -m unittest discover -s tests -v
```

The tests use synthetic fixtures and can run without the course dataset.
Running `python main.py` requires the three input CSV files described above.

Default paths are resolved relative to `main.py`, rather than the terminal's
working directory. Custom locations can be passed explicitly:

```bash
python main.py --data-dir /path/to/csv-files --output-dir /path/to/results
```

The last completed automated test run passed **19 test methods**. They cover input
validation, SSE selection, threshold boundaries, ambiguity, repeated x values,
database contents, streaming, report data and the complete application workflow.
A successful test run verifies those cases, not every possible input or failure.

The application uses English terminal messages, report text, chart labels and
validation messages.

## Validation and calculation

The input checks require exact column names and order, nonempty tables, numeric
finite values, unique x values in the training/ideal tables and ideal-data
coverage for all required x values. Repeated test x values are valid observations
and are preserved. Missing or invalid values are rejected rather than imputed.
Lookup uses exact x values; there is no interpolation or approximate matching.

For each training series, the program compares every ideal candidate using the
sum of squared y deviations:

```text
SSE = sum((training_y - ideal_y_at_the_same_x) ** 2)
```

Data is aligned by x, so differing CSV row orders do not affect the comparison.
The candidate with the smallest SSE is selected independently for each training
series. Candidate reuse is permitted; exact SSE ties prefer the lowest numeric
ideal-function ID.

For each selected function, the acceptance limit is:

```text
threshold = sqrt(2) * max(abs(training_y - selected_ideal_y))
```

A test observation qualifies when its absolute y deviation from a selected
function is **less than or equal to** that function's threshold. If several
functions qualify, the smallest absolute deviation wins; exact ties prefer the
lowest numeric function ID. Reused ideal functions count once as eligible
candidates.

The tie policies and candidate-reuse policy are documented implementation
assumptions, not explicit assessment rules. The tolerance formula follows our
interpretation of the assignment's wording. These points should be confirmed
before submission if the course provides further instructions.

The application validates all tables first, calculates the selected functions,
stores the training/ideal tables, then reads `test.csv` again **line by line** for
mapping. The initial validation already loads the test table into memory, so this
is not a constant-memory streaming pipeline.

## Generated outputs

After a successful run, `results/` contains:

| File | Contents |
|---|---|
| `data_summary.json` | Input dimensions, x ranges and repeated-x counts |
| `matching_summary.json` | Selected functions, SSE, maximum deviations, thresholds, outcome counts and mapping policy |
| `test_assignments.csv` | Every test observation in original order, including status and number of eligible candidates |
| `matched_points.csv` | Successful assignments with `x`, `y`, `delta_y`, `ideal_function` |
| `assignment.sqlite` | Training, ideal-function and successful-assignment tables |
| `visualizations.html` | Interactive, self-contained Bokeh report |

`delta_y` is an absolute deviation. Unassigned observations have empty deviation
and function fields in the complete CSV export; they have not been lost.
`test_row` identifies observations by their sequence, starting at 1.

For the supplied dataset, the saved results are:

| Training series | Selected ideal function |
|---|---|
| y1 | y13 |
| y2 | y24 |
| y3 | y36 |
| y4 | y40 |

There are **34 assigned** and **66 unassigned** test observations. One observation
has two eligible candidates and is resolved using the documented mapping rule.
These are calculated results for the supplied files, not hard-coded targets.

### SQLite database

The database contains exactly these three tables:

| Table | Columns | Current rows |
|---|---:|---:|
| `training` | 5 | 400 |
| `ideal` | 51 | 400 |
| `test_results` | 4 | 34 |

`test_results` contains `x`, `y`, `delta_y` and `ideal_function`. Only successful
assignments are stored there; rejected observations remain in the full CSV.
No DataFrame index is exported as an additional column, and test x values are not
required to be unique. SQLAlchemy manages the connection and transactions; pandas
writes the tables. The implementation does not use a full ORM entity model.

A SQLite file is binary and is not intended to be read as plain text in VS Code.
A separate database viewer is optional, not required to run the application.

### Bokeh visualizations

Open `results/visualizations.html` in a browser, for example by double-clicking it
in Finder. BokehJS is embedded, so no internet connection or web server is needed.

The report includes an overview of all test observations, including rejected
points, and four groups of function/residual plots. Each group shows training
values, the selected ideal function, assigned observations and absolute
deviations. The dashed line marks the acceptance limit. Zoom tools and clickable
legends support closer inspection.

Each function group uses its own y scale. The overview uses a common scale and
is intended for orientation, rather than comparison of small deviations. The
per-function residual plots do not show every rejected point against every
selected function; rejected points appear in the overview and full CSV export.

## Output replacement and error handling

Data-validation, file-system and SQLAlchemy errors are reported with exit code 1.
A successful run returns 0. The program generates outputs in a temporary subfolder
before replacing the final files. Validation or generation failure therefore
leaves previous outputs in place; these may belong to an earlier successful run.
Always check the current terminal result.

Final files are replaced individually, not in a single transaction across all
outputs. A failure during that final publication step can leave mixed results;
rerun the program after addressing the error.

## Written report and verification

The separate written report explains the mathematical method, software design,
tests, results and limitations. It also includes the additional Git exercise as
a hypothetical team workflow; it does not claim that this repository was developed
through that collaborative process. Its appendices provide the repository link,
reproduction instructions and an AI assistance statement.

An independent calculation using CSV records and scalar arithmetic confirmed
the selections y13, y24, y36 and y40, the 34 assigned and 66 unassigned observations,
and the one observation with two eligible candidates. This supports the reported
results for the supplied files, not correctness for every possible input.

The assigned dataset and current course submission requirements must still be
checked against the official course instructions before submission.

## AI assistance

AI assistance was used to develop and document this implementation. The student
remains responsible for understanding, checking and explaining the submitted
work and following applicable course rules.

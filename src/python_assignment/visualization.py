"""Create a standalone, offline Bokeh report with fits, mappings and residuals."""
from pathlib import Path

import pandas as pd
from bokeh.embed import file_html
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Div, HoverTool, Range1d, Span
from bokeh.plotting import figure
from bokeh.resources import INLINE

from .data import CourseData
from .matching import FunctionFit


def build_report(data: CourseData, fits: list[FunctionFit],
                 assignments: pd.DataFrame):
    """Build separate scales per fit and residual views with explicit limits.

    The overview includes rejected points. Per-fit plots use only the test points
    assigned to that ideal function; duplicate observations are kept in sources.
    """
    accepted = assignments[assignments["status"] == "assigned"]
    rejected = assignments[assignments["status"] == "unassigned"]
    ambiguous = int((assignments["eligible_count"] > 1).sum())
    intro = Div(text=f"""<h1>Programming with Python: Results</h1>
        <p>{len(assignments)} test observations: <b>{len(accepted)} assigned</b>,
        {len(rejected)} unassigned; {ambiguous} with multiple eligible candidates.</p>
        <p>Selection: minimum sum of squared y deviations (SSE).
        Assignment: |deviation| ≤ √2 × maximum absolute training deviation.
        Multiple eligible candidates: smallest deviation, then lowest function ID.
        This additional decision rule is a documented implementation assumption.</p>
        <p>The function panels use separate y scales.
        Use the tools to zoom; click legend entries to hide series.</p>""",
                sizing_mode="stretch_width")
    overview = figure(title="All test observations: assigned and unassigned",
                      height=360, sizing_mode="stretch_width",
                      x_axis_label="x", y_axis_label="y",
                      tools="pan,wheel_zoom,box_zoom,reset,save")
    colors = ["#1768ac", "#d14900", "#17815d", "#8d4bbb"]
    # Keep selection order while avoiding duplicate legend entries for reused ideals.
    ids = list(dict.fromkeys(int(fit.ideal_function[1:]) for fit in fits))
    for index, number in enumerate(ids):
        subset = accepted[accepted["ideal_function"] == number]
        overview.scatter("x", "y", source=ColumnDataSource(subset.to_dict(orient="list")), size=8,
                         color=colors[index % len(colors)],
                         legend_label=f"Ideal function y{number}")
    overview.scatter("x", "y", source=ColumnDataSource(rejected.to_dict(orient="list")), marker="x",
                     size=9, color="#666666", legend_label="Unassigned")
    overview.add_tools(HoverTool(tooltips=[("Test row", "@test_row"),
                                         ("x / y", "@x / @y"),
                                         ("Status", "@status"),
                                         ("Eligible candidates", "@eligible_count")]))
    overview.legend.click_policy = "hide"
    panels = []
    ideal = data.ideal.frame.set_index("x")
    # Sort x before drawing lines so the curve does not zigzag through file order.
    training = data.training.frame.sort_values("x")
    for index, fit in enumerate(fits):
        color = colors[index % len(colors)]
        ideal_values = ideal.loc[training["x"], fit.ideal_function].to_numpy()
        number = int(fit.ideal_function[1:])
        subset = accepted[accepted["ideal_function"] == number]
        # Named columns allow Bokeh glyphs and hover tools to share the same data.
        source = ColumnDataSource(subset.to_dict(orient="list"))
        plot = figure(title=f"Training {fit.training_function} → Ideal {fit.ideal_function}",
                      height=320, sizing_mode="stretch_width",
                      x_axis_label="x", y_axis_label="y",
                      tools="pan,wheel_zoom,box_zoom,reset,save")
        plot.line(training["x"], ideal_values, color=color, line_width=2,
                  legend_label="Ideal function")
        plot.scatter(training["x"], training[fit.training_function], size=4,
                     color="#87929e", alpha=0.6, legend_label="Training")
        renderer = plot.scatter("x", "y", source=source, marker="diamond", size=10,
                                color="#c52a34", legend_label="Assigned test observations")
        plot.add_tools(HoverTool(renderers=[renderer], tooltips=[
            ("Test row", "@test_row"), ("|Δy|", "@delta_y{0.000000}")]))
        plot.legend.click_policy = "hide"
        plot.legend.location = "top_left"
        plot.legend.label_text_font_size = "9px"
        residual = figure(title=f"Absolute deviations · Threshold {fit.threshold:.6f}",
                          height=250, sizing_mode="stretch_width",
                          x_axis_label="x", y_axis_label="|Δy|",
                          tools="pan,wheel_zoom,box_zoom,reset,save")
        # Annotation spans do not expand Bokeh's automatic data range. Include
        # the threshold explicitly so the acceptance limit is always visible.
        train_deltas = abs(training[fit.training_function].to_numpy()-ideal_values)
        # Add headroom above the largest value, including the acceptance threshold.
        upper = max(fit.threshold, float(max(train_deltas)),
                    float(subset["delta_y"].max()) if len(subset) else 0.0)
        residual.y_range = Range1d(start=0, end=upper * 1.12 if upper else 1.0)
        residual.scatter(training["x"],
                         train_deltas,
                         size=4, color="#87929e", alpha=0.6, legend_label="Training")
        residual.scatter("x", "delta_y", source=source, size=8, color="#c52a34",
                         legend_label="Assigned test observations")
        residual.add_layout(Span(location=fit.threshold, dimension="width",
                                 line_color="#d14900", line_dash="dashed",
                                 line_width=2))
        residual.legend.click_policy = "hide"
        residual.legend.label_text_font_size = "9px"
        caption = Div(text=f"<p><b>SSE:</b> {fit.sse:.6f} · "
                           f"<b>Max. training deviation:</b> {fit.max_deviation:.6f} · "
                           f"<b>Threshold:</b> {fit.threshold:.6f} (dashed)</p>")
        panels.append(column(plot, residual, caption, sizing_mode="stretch_width"))
    # Separate plots retain their own scales despite the shared two-column layout.
    grid = gridplot([panels[i:i+2] for i in range(0, len(panels), 2)],
                    sizing_mode="stretch_width", merge_tools=False)
    return column(intro, overview, grid, sizing_mode="stretch_width",
                  styles={"max-width": "1400px", "margin": "0 auto"})


def write_report(path: Path, data: CourseData, fits: list[FunctionFit],
                 assignments: pd.DataFrame) -> None:
    """Save HTML with BokehJS embedded, requiring no server or network access."""
    document = build_report(data, fits, assignments)
    # INLINE embeds BokehJS so the report can be opened without internet access.
    path.write_text(file_html(document, INLINE, "Python Assignment: Results"),
                    encoding="utf-8")

"""
Small Altair chart helpers shared across pages.

st.line_chart / st.bar_chart give no control over axis titles or tick-label
rotation, which is why dates and category names end up crowded and vertical.
These wrappers fix both: an explicit, unit-labeled axis on both sides, and
horizontal (unrotated) tick labels.
"""

import altair as alt
import pandas as pd
import streamlit as st


def format_table(df, fmt, na_rep="—"):
    """Format specific columns to display strings, e.g. {"TE": "{:.2%}"}.
    Deliberately does NOT use pandas' df.style.format(): accessing .style
    imports pandas.io.formats.style, which imports matplotlib at module
    level -- in this environment that raises
    `AttributeError: _ARRAY_API not found` (numpy 2.x / matplotlib ABI
    mismatch) and crashes the whole page. This does the same per-column
    string formatting without going anywhere near Styler/matplotlib."""
    out = df.copy()
    for col, spec in fmt.items():
        if col not in out.columns:
            continue
        if callable(spec):
            fn = spec
        else:
            fn = spec.format
        out[col] = out[col].apply(lambda v: na_rep if pd.isna(v) else fn(v))
    return out


def line_chart(
    df,
    y_title,
    value_format=".1f",
    index_name="Date",
    x_title=None,
    zero_line=False,
    ref_line=None,
    key=None,
    log_scale=False,
):
    """Multi-line chart from a wide DataFrame indexed by `index_name`, one
    column per series. Horizontal x-axis labels, unit-labeled y-axis.
    `zero_line=True` overlays a dashed rule at y=0 (for return series,
    where zero is the meaningful reference point, not the axis floor).
    `ref_line`, if given, overlays a dashed rule at that value instead
    (e.g. 1.0 for a growth-of-$1 series, where "no gain/loss" is $1, not $0).
    `log_scale=True` plots the y-axis on a log scale -- essential for a
    multi-decade compounding series, where a linear axis lets the fastest
    compounder flatten every slower line down near zero; on a log axis,
    equal vertical distance means equal percentage move everywhere, so
    every line's compounding path stays legible at once.
    Pass a `key` that changes whenever the selection changes (e.g. include
    the window/metric/chosen-factors in it) -- Vega's pan/zoom state from
    `.interactive()` otherwise sticks to the chart's DOM element and can
    leave a stale zoom applied after you switch to a very different date
    range, making the new lines look like they've vanished."""
    long_df = df.reset_index().melt(id_vars=index_name, var_name="Series", value_name="value")
    long_df = long_df.dropna(subset=["value"])
    y_scale = alt.Scale(type="log") if log_scale else alt.Undefined
    lines = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X(
                f"{index_name}:T",
                title=x_title or index_name,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=value_format), scale=y_scale),
            color=alt.Color("Series:N", title=None),
            tooltip=[
                alt.Tooltip(f"{index_name}:T", title=index_name),
                alt.Tooltip("Series:N"),
                alt.Tooltip("value:Q", title=y_title, format=value_format),
            ],
        )
    )
    chart = lines
    ref_value = 0 if zero_line else ref_line
    if ref_value is not None and not (log_scale and ref_value <= 0):
        ref = (
            alt.Chart(pd.DataFrame({"y": [ref_value]}))
            .mark_rule(color="gray", strokeDash=[4, 4])
            .encode(y=alt.Y("y:Q", scale=y_scale))
        )
        chart = ref + lines
    st.altair_chart(chart.interactive(), use_container_width=True, key=key)


def grouped_bar_chart(long_df, cat_col, value_col, series_col, x_title, y_title, value_format=".2f"):
    """Vertical grouped bar chart from a long-format DataFrame. Horizontal
    (unrotated) category labels, unit-labeled value axis."""
    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{cat_col}:N", title=x_title, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{value_col}:Q", title=y_title, axis=alt.Axis(format=value_format)),
            color=alt.Color(f"{series_col}:N", title=None),
            xOffset=alt.XOffset(f"{series_col}:N"),
            tooltip=[cat_col, series_col, alt.Tooltip(f"{value_col}:Q", format=value_format)],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def bar_chart(df, cat_col, value_col, x_title, y_title, value_format=".2f"):
    """Single-series vertical bar chart. Horizontal category labels,
    unit-labeled value axis."""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{cat_col}:N", title=x_title, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{value_col}:Q", title=y_title, axis=alt.Axis(format=value_format)),
            tooltip=[cat_col, alt.Tooltip(f"{value_col}:Q", format=value_format)],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def scenario_line_chart(df, x_col, y_col, series_col, x_title, y_title, x_format=".0%", y_format=".1%"):
    """Point-and-line chart for a small number of discrete x values (e.g.
    collar coverage levels), one line per `series_col`."""
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_title, axis=alt.Axis(format=x_format, labelAngle=0)),
            y=alt.Y(f"{y_col}:Q", title=y_title, axis=alt.Axis(format=y_format)),
            color=alt.Color(f"{series_col}:N", title=None),
            tooltip=[series_col, alt.Tooltip(f"{x_col}:Q", format=x_format), alt.Tooltip(f"{y_col}:Q", format=y_format)],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

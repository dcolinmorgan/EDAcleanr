"""Shared Hypothesis strategies and fixtures for property-based tests.

Provides reusable strategies for generating messy DataFrames and CSV file
parameters used across all property test modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes


# ---------------------------------------------------------------------------
# messy_dataframes — generates DataFrames with controlled messiness
# ---------------------------------------------------------------------------


@st.composite
def messy_dataframes(
    draw: st.DrawFn,
    min_rows: int = 5,
    max_rows: int = 100,
    min_cols: int = 2,
    max_cols: int = 10,
) -> pd.DataFrame:
    """Generate a DataFrame with controlled missing values, duplicates,
    mixed types, outliers, zero-variance columns, and high-cardinality
    string columns.

    Parameters
    ----------
    draw : hypothesis draw function
    min_rows, max_rows : row count bounds (default 5-100)
    min_cols, max_cols : column count bounds (default 2-10)

    Returns
    -------
    pd.DataFrame with a realistic mix of data quality issues.
    """
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    # Decide how many columns of each kind
    # At least 1 numeric and 1 string column
    n_numeric = draw(st.integers(min_value=1, max_value=max(1, n_cols - 1)))
    n_string = n_cols - n_numeric

    data: dict[str, list] = {}

    # --- Numeric columns ---------------------------------------------------
    for i in range(n_numeric):
        col_name = f"num_{i}"
        values = draw(
            st.lists(
                st.floats(
                    min_value=-1e6,
                    max_value=1e6,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        data[col_name] = values

    # --- String columns ----------------------------------------------------
    for i in range(n_string):
        col_name = f"str_{i}"
        values = draw(
            st.lists(
                st.text(
                    alphabet=st.characters(
                        whitelist_categories=("L", "N", "Z"),
                    ),
                    min_size=1,
                    max_size=20,
                ),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        data[col_name] = values

    df = pd.DataFrame(data)

    # --- Inject missing values (NaN) at random positions -------------------
    inject_missing = draw(st.booleans())
    if inject_missing:
        missing_frac = draw(st.floats(min_value=0.01, max_value=0.5))
        n_missing = max(1, int(n_rows * n_cols * missing_frac))
        for _ in range(n_missing):
            r = draw(st.integers(min_value=0, max_value=n_rows - 1))
            c = draw(st.integers(min_value=0, max_value=n_cols - 1))
            df.iat[r, c] = np.nan

    # --- Inject duplicate rows ---------------------------------------------
    inject_dupes = draw(st.booleans())
    if inject_dupes and n_rows >= 2:
        n_dupes = draw(st.integers(min_value=1, max_value=max(1, n_rows // 4)))
        source_indices = draw(
            st.lists(
                st.integers(min_value=0, max_value=n_rows - 1),
                min_size=n_dupes,
                max_size=n_dupes,
            )
        )
        dup_rows = df.iloc[source_indices]
        df = pd.concat([df, dup_rows], ignore_index=True)

    # --- Inject mixed types in some columns --------------------------------
    inject_mixed = draw(st.booleans())
    if inject_mixed and n_numeric >= 1:
        # Pick a numeric column and sprinkle in some non-numeric strings
        target_col = f"num_{draw(st.integers(min_value=0, max_value=n_numeric - 1))}"
        n_mixed = draw(st.integers(min_value=1, max_value=max(1, len(df) // 5)))
        # Convert column to object so we can mix types
        df[target_col] = df[target_col].astype(object)
        for _ in range(n_mixed):
            r = draw(st.integers(min_value=0, max_value=len(df) - 1))
            df.at[r, target_col] = draw(
                st.text(
                    alphabet=st.characters(whitelist_categories=("L",)),
                    min_size=1,
                    max_size=8,
                )
            )

    # --- Inject outliers in numeric columns --------------------------------
    inject_outliers = draw(st.booleans())
    if inject_outliers and n_numeric >= 1:
        target_col = f"num_{draw(st.integers(min_value=0, max_value=n_numeric - 1))}"
        if pd.api.types.is_numeric_dtype(df[target_col]):
            n_outliers = draw(st.integers(min_value=1, max_value=max(1, len(df) // 10)))
            for _ in range(n_outliers):
                r = draw(st.integers(min_value=0, max_value=len(df) - 1))
                # Value far outside normal range
                extreme = draw(
                    st.floats(min_value=1e7, max_value=1e9, allow_nan=False, allow_infinity=False)
                )
                sign = draw(st.sampled_from([-1.0, 1.0]))
                df.at[r, target_col] = sign * extreme

    # --- Inject a zero-variance column -------------------------------------
    inject_zero_var = draw(st.booleans())
    if inject_zero_var:
        const_val = draw(st.one_of(st.integers(-100, 100), st.text(min_size=1, max_size=5)))
        df["zero_var"] = const_val

    # --- Inject a high-cardinality string column ---------------------------
    inject_high_card = draw(st.booleans())
    if inject_high_card:
        unique_vals = [f"id_{j}" for j in range(len(df))]
        df["high_card"] = unique_vals

    return df



# ---------------------------------------------------------------------------
# csv_file_params — generates encoding/delimiter/data combinations
# ---------------------------------------------------------------------------


@st.composite
def csv_file_params(draw: st.DrawFn) -> dict:
    """Generate a combination of CSV encoding, delimiter, and a small
    DataFrame suitable for round-trip CSV testing.

    Returns
    -------
    dict with keys:
        - "encoding": one of 'utf-8', 'latin-1', 'cp1252'
        - "delimiter": one of ',', '\\t', ';', '|'
        - "data": a small pd.DataFrame (3-20 rows, 2-5 columns)
    """
    encoding = draw(st.sampled_from(["utf-8", "latin-1", "cp1252"]))
    delimiter = draw(st.sampled_from([",", "\t", ";", "|"]))

    n_rows = draw(st.integers(min_value=3, max_value=20))
    n_cols = draw(st.integers(min_value=2, max_value=5))

    # Build columns with simple data that survives encoding round-trips.
    # Use ASCII-safe characters to avoid encoding edge cases that would
    # obscure the property under test.
    data: dict[str, list] = {}
    for i in range(n_cols):
        col_name = f"col_{i}"
        if draw(st.booleans()):
            # Numeric column
            values = draw(
                st.lists(
                    st.floats(
                        min_value=-1e4,
                        max_value=1e4,
                        allow_nan=False,
                        allow_infinity=False,
                        allow_subnormal=False,
                    ),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        else:
            # String column — ASCII-safe so all encodings handle it
            values = draw(
                st.lists(
                    st.text(
                        alphabet=st.characters(
                            whitelist_categories=("L", "N"),
                            whitelist_characters="_",
                            max_codepoint=127,
                        ),
                        min_size=1,
                        max_size=12,
                    ),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        data[col_name] = values

    df = pd.DataFrame(data)

    return {"encoding": encoding, "delimiter": delimiter, "data": df}

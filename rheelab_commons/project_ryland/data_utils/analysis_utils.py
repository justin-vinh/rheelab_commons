"""
------------------------------------------------------------------------------
Author:         Justin Vinh
Collaborators:  John Rhee, MD
Parent Package: Project Ryland
Creation Date:  2025.10.16
Last Modified:  2025.10.16

Purpose:
This module contains functions designed to aid in statistical analyses of
data processed by LLMs
------------------------------------------------------------------------------
"""
import ast
import json
from typing import Iterable, List, Optional, Any

import pandas as pd


# FUNCTIONS TO PROCESS LABEL STUDIO OUTPUTS
# -----------------------------------------------------------------------------
def extract_label_studio_label_status(
    df: pd.DataFrame,
    label_col: str,
    target_labels: List[str],
    col_name_suffix: Optional[str] = "_status",
) -> pd.DataFrame:
    """
    Extract binary indicator columns for specified labels from a Label Studio
    annotation column.

    This function parses JSON-formatted Label Studio outputs stored in
    `label_col` and creates one binary column per target label indicating
    whether that label was assigned to the row.

    Supported input formats:
    1) Flat span dictionaries:
        {"start": ..., "end": ..., "labels": [...]}

    2) Standard Label Studio annotation format:
        {"result": [{"value": {"labels": [...]}}]}

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing Label Studio annotations.
    label_col : str
        Column name containing JSON annotations (string or object).
    target_labels : List[str]
        Labels to extract.
    col_name_suffix : Optional[str], default="_status"
        Suffix added to generated column names.
        If None, original label names are used as column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional binary columns for each target label.
        1 = label present
        0 = label absent
    """

    # Create output column names
    col_names = (
        [f"{lab.lower()}{col_name_suffix}" for lab in target_labels]
        if col_name_suffix else target_labels
    )

    # Initialize all label columns to 0
    for col in col_names:
        df[col] = 0

    # Convert stored value into Python object (list/dict)
    def coerce_to_obj(x: Any):
        if pd.isna(x):
            return None
        if isinstance(x, (list, dict)):
            return x
        if isinstance(x, str):
            return json.loads(x)
        return None

    # Iterate row-by-row through annotation column
    for idx, x in df[label_col].items():
        try:
            obj = coerce_to_obj(x)
            if not obj:
                continue

            # Ensure iterable structure
            items = obj if isinstance(obj, list) else [obj]
            found = set()

            for item in items:
                # Format A: flat span dict
                if "labels" in item and isinstance(item.get("labels"), list):
                    found.update(item["labels"])

                # Format B: nested Label Studio annotation structure
                for r in (item.get("result") or []):
                    labels = (r.get("value") or {}).get("labels") or []
                    found.update(labels)

            # Set indicator columns
            for lab, col in zip(target_labels, col_names):
                if lab in found:
                    df.at[idx, col] = 1

        except Exception:
            # Skip malformed rows silently
            continue

    return df


def expand_label_studio_data_to_col(
    df: pd.DataFrame,
    data_col: str,
    keys: Optional[Iterable[str]] = None,
    prefix: str = "",
    drop: bool = False,
) -> pd.DataFrame:
    """
    Expand a DataFrame column containing per-row dictionaries (or dictionary-like
    strings) into separate columns.

    This function parses a column that may contain:
      - Python dict objects
      - JSON-formatted strings (double quotes)
      - Python-literal strings (single quotes), via ast.literal_eval

    It extracts either:
      - All top-level keys (if keys=None), or
      - A specified subset of keys.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    data_col : str
        Column containing dict or dict-like string data.
    keys : Iterable[str], optional
        Specific keys to extract. Missing keys are filled with pd.NA.
        If None, all top-level keys found across rows are extracted.
    prefix : str, default=""
        Optional prefix added to new column names.
    drop : bool, default=False
        If True, drops the original `data_col`.

    Returns
    -------
    pd.DataFrame
        DataFrame with expanded columns appended.
    """

    # Safely convert each value to a dictionary
    def to_dict(x: Any):
        if pd.isna(x):
            return {}
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            try:
                return json.loads(s)  # Attempt JSON parsing
            except Exception:
                try:
                    return ast.literal_eval(s)  # Attempt Python literal parsing
                except Exception:
                    return {}
        return {}

    # Parse the column into dictionaries
    parsed = df[data_col].apply(to_dict)

    # Expand either all keys or selected keys
    if keys is None:
        expanded = pd.json_normalize(parsed)
    else:
        expanded = pd.DataFrame(
            [{k: d.get(k, pd.NA) for k in keys} for d in parsed]
        )

    # Optionally prefix new column names
    if prefix:
        expanded = expanded.add_prefix(prefix)

    # Concatenate expanded columns to original DataFrame
    out = pd.concat(
        [df.drop(columns=[data_col]) if drop else df, expanded],
        axis=1
    )

    return out
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
# def extract_label_studio_label_status(
#     df: pd.DataFrame,
#     label_col: str,
#     target_labels: List[str],
#     col_name_suffix: Optional[str] = "_status",
# ) -> pd.DataFrame:
#     """
#     Extract binary indicator columns for specified labels from a Label Studio
#     annotation column.
#
#     This function parses JSON-formatted Label Studio outputs stored in
#     `label_col` and creates one binary column per target label indicating
#     whether that label was assigned to the row.
#
#     Supported input formats:
#     1) Flat span dictionaries:
#         {"start": ..., "end": ..., "labels": [...]}
#
#     2) Standard Label Studio annotation format:
#         {"result": [{"value": {"labels": [...]}}]}
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame containing Label Studio annotations.
#     label_col : str
#         Column name containing JSON annotations (string or object).
#     target_labels : List[str]
#         Labels to extract.
#     col_name_suffix : Optional[str], default="_status"
#         Suffix added to generated column names.
#         If None, original label names are used as column names.
#
#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with additional binary columns for each target label.
#         1 = label present
#         0 = label absent
#     """
#
#     # Normalize to lowercase
#     target_labels = [lab.lower() for lab in target_labels]
#
#     # Create output column names
#     col_names = (
#         [f"{lab.lower()}{col_name_suffix}" for lab in target_labels]
#         if col_name_suffix else target_labels
#     )
#
#     # Initialize all label columns to 0
#     for col in col_names:
#         df[col] = 0
#
#     # Convert stored value into Python object (list/dict)
#     def coerce_to_obj(x: Any):
#         if pd.isna(x):
#             return None
#         if isinstance(x, (list, dict)):
#             return x
#         if isinstance(x, str):
#             return json.loads(x)
#         return None
#
#     # Iterate row-by-row through annotation column
#     for idx, x in df[label_col].items():
#         try:
#             obj = coerce_to_obj(x)
#             if not obj:
#                 continue
#
#             # Ensure iterable structure
#             items = obj if isinstance(obj, list) else [obj]
#             found = set()
#
#             for item in items:
#                 if not isinstance(item, dict):
#                     continue
#
#                 # Direct Label Studio format
#                 if "value" in item:
#                     labels = item.get("value", {}).get("labels", [])
#                     found.update([l.lower() for l in labels])
#
#                 # Backup: flat format
#                 if "labels" in item:
#                     found.update([l.lower() for l in item["labels"]])
#
#                 # Backup: nested "result" format
#                 if "result" in item:
#                     for r in item.get("result", []):
#                         labels = r.get("value", {}).get("labels", [])
#                         found.update([l.lower() for l in labels])
#
#             # Set indicator columns
#             for lab, col in zip(target_labels, col_names):
#                 if lab in found:
#                     df.at[idx, col] = 1
#
#         except Exception:
#             # Skip malformed rows silently
#             continue
#
#     return df


def extract_label_studio_label_status(
    df: pd.DataFrame,
    label_col: str,
    target_labels: List[str],
    col_name_suffix: Optional[str] = "_status",
) -> pd.DataFrame:

    target_labels = [lab.lower() for lab in target_labels]

    col_names = (
        [f"{lab}{col_name_suffix}" for lab in target_labels]
        if col_name_suffix else target_labels
    )

    for col in col_names:
        df[col] = 0

    def coerce_to_obj(x: Any):
        if pd.isna(x):
            return None
        if isinstance(x, (list, dict)):
            return x
        if not isinstance(x, str):
            return None

        s = x.strip()

        # try a few rounds in case it's encoded as a string-of-a-string
        for _ in range(3):
            # Try JSON
            try:
                v = json.loads(s)
            except Exception:
                v = None

            # Fallback: Python literal (single quotes)
            if v is None:
                try:
                    v = ast.literal_eval(s)
                except Exception:
                    return None

            if isinstance(v, (list, dict)):
                return v
            if isinstance(v, str):
                s = v.strip()
                continue

            return None

        return None

    # Recursive function to find all "labels"
    def find_labels(obj):
        found = set()

        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "labels" and isinstance(v, list):
                    found.update([str(i).lower() for i in v])
                else:
                    found.update(find_labels(v))

        elif isinstance(obj, list):
            for item in obj:
                found.update(find_labels(item))

        return found

    for idx, x in df[label_col].items():
        try:
            obj = coerce_to_obj(x)
            if obj is None:
                continue

            found = find_labels(obj)

            for lab, col in zip(target_labels, col_names):
                if lab in found:
                    df.at[idx, col] = 1

        except Exception:
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
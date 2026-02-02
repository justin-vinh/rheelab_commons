"""
------------------------------------------------------------------------------
Author:         Justin Vinh
Parent Package: Project Ryland
Creation Date:  2025.09.29
Last Modified:  2025.09.29

Purpose:
Contains functions to import/output data and do basic cleaning
------------------------------------------------------------------------------
"""

from pathlib import Path

import orjson
import pandas as pd


def normalize_newlines(text: str) -> str:
    """
    Normalize all line endings of text to use Unix-style \n
    (helps in using regex downstream)
    """
    if not isinstance(text, str):
        return text
    else:
        # Replace \\r\\n in the text to just \n
        return text.replace('\\r\\n', '\n').replace('\r\n', '\n')


def load_oncdrs_json_to_df(path_name):
    """
    1) Loads the given OncDRS-exported json file and changes it to a
    dataframe.
    2) Normalize all line endings of text in the RPT_TEXT and NARRATIVE_TEXT
    columns to use Unix-style \n (helps in using regex downstream)
    3) Has error handling for missing file or decoding issues
    """
    # Handles incorrect path names, raises an error if there is a bad path
    path = Path(path_name)
    if not path.exists():
        raise FileNotFoundError(f"File {path_name} not found")

    # Tries opening the json file, raises an error if unable to read it
    # try:
    #     with path.open('r') as json_file:
    #         data = json.load(json_file)
    try:
        data = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON format in file: {path}') from e

    # Creates a dataframe using the key-value pairs under response>docs
    df = pd.DataFrame(data['response']['docs'])

    # Normalizes line endings with \n in RPT_TEXT and NARRATIVE_TEXT columns
    for col in ['RPT_TEXT', 'NARRATIVE_TEXT']:
        if col in df.columns:
            df[col] = df[col].map(normalize_newlines)

    return df
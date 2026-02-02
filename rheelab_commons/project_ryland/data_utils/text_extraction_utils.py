"""
------------------------------------------------------------------------------
Author:         Justin Vinh
Collaborators:  Zach Tentor
Parent Package: Project Ryland
Creation Date:  2025.09.29
Last Modified:  2025.09.29

Purpose:
Contain the functions to match keywords within clinical notes and then pull
out the relevant subset of those notes (to ease cost of running LLMs)
------------------------------------------------------------------------------
"""

import re
from typing import List, Dict, Any

import pandas as pd

from .keyword_mappings import (pathology_mappings_by_proc_desc,
                               image_mappings_by_proc_desc,
                               progress_note_mappings_by_proc_desc)


# These functions constitute a more modular but 1:1 replacement for the
# extract_text function in the original code by Zach. See extract_text for
# full explanation of function workflow

# HELPER FUNCTIONS FOR EXTRACT_TEXT
# -----------------------------------------------------------------------------
def get_mappings(proc_desc: str,
                 mapping_type: str,
                 match_type: str = "hard") \
        -> List[Dict[str, Any]]:
    """
    Outputs the keyword mappings for a given procedure description. Behavior
    changes depending on whether desired match is hard or soft regex matching
    as well as the given mapping specification
    """
    # Sets which mapping type to use
    if mapping_type == "pathology":
        mapping_dict = pathology_mappings_by_proc_desc
    elif mapping_type == "imaging":
        mapping_dict = image_mappings_by_proc_desc
    elif mapping_type == "progress":
        mapping_dict = progress_note_mappings_by_proc_desc
    else:
        raise ValueError(
            'mapping_type must be "pathology" or "imaging" or "progress"'
        )

    # strictly matches the mappings
    if match_type == "hard":
        if proc_desc not in mapping_dict:
            raise ValueError(f"Unknown PROC_DESC: {proc_desc}")
        return mapping_dict[proc_desc]

    # Checks to see only if the given descriptions are in the mappings
    elif match_type == "soft":
        # Case-insensitive contains check against available PROC_DESC keys
        for key in mapping_dict.keys():
            if key == "*" or re.search(key, proc_desc, flags=re.IGNORECASE):
                return mapping_dict[key]
        if "*" in mapping_dict:
            return mapping_dict["*"]
        raise ValueError(f"No soft match found for PROC_DESC: {proc_desc}")
    else:
        raise ValueError("match_type must be either 'hard' or 'soft'")


def compile_mappings(mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compiles all start and end keywords based on mappings_by_proc_desc to
    avoid having to recompile them repeatedly
    """
    compiled = []
    # For each mapping, append the keywords and associated compiled regex
    for mapping in mappings:
        compiled.append({
            "start_keyword": mapping["start"],
            "start_regex": re.compile(re.escape(mapping["start"])),
            "end_keywords": mapping["end"],
            "end_regex": re.compile("|".join(re.escape(i) for i in mapping["end"])),
            "condition": mapping.get("condition"),
            "exclude_after": mapping.get("exclude_after"),
            "exclude_regex": re.compile(re.escape(mapping["exclude_after"]))
                             if mapping.get("exclude_after") else None
        })
    return compiled


def find_matches(input_text: str,
                 compiled_mappings: List[Dict[str, Any]],
                 mapping_type: str = None) \
    -> List[Dict[str, Any]]:
    """
     1) Finds all start keyword matches in the input text
     2) Keep track of positions of matches
     3) Maps metadata
    """
    matches = []
    match_count = 0
    # Find starting keywords and then create a list item that includes that
    # keyword, its position, and the needed associated keywords and compiled regex
    for mapping in compiled_mappings:
        for match in mapping["start_regex"].finditer(input_text):

            # Special condition to only keep the first match for imaging notes
            if mapping_type == "imaging" and match_count >= 1:
                continue

            matches.append({
                "section": mapping["start_keyword"],
                "start_pos": match.end(),
                "end_regex": mapping["end_regex"],
                "condition": mapping["condition"],
                "exclude_regex": mapping["exclude_regex"]
            })
            match_count += 1

    # Return list of dicts, sorted based on starting position of match
    return sorted(matches, key=lambda match: match["start_pos"])


def extract_segments(input_text: str, matches: List[Dict[str, Any]]) \
    -> List[Dict[str, Any]]:
    """
    Extract the text segments between the matching start and end keywords,
    applying exclusion rules where applicable
    """
    extracted = []
    current_pos = 0
    text_length = len(input_text)

    for match in matches:
        start_pos = int(match["start_pos"])
        # If a text segment overlaps with a prior segment, skip it
        if start_pos < current_pos:
            continue

        end_regex = match["end_regex"]
        condition = match["condition"]
        exclude_regex = match["exclude_regex"]

        # Default end of text segment is the end of the text unless an end
        # keyword match is found
        end_pos = text_length
        if end_regex:
            end_match = end_regex.search(input_text, start_pos)
            if end_match:
                end_pos = end_match.start()

        # Apply the exclusion criteria check.
        # Exclude note from being extracted if there is match here
        if condition == "exclude" and exclude_regex:
            exclude_match = exclude_regex.search(input_text, start_pos, end_pos)
            if exclude_match:
                continue

        # Cleans extracted text
        # Remove leading whitespace + any colon (ASCII or Unicode variants)
        extracted_text = input_text[start_pos:end_pos]
        extracted_text = re.sub(r"^[:\s]+", "", extracted_text)
        extracted_text = extracted_text.strip()
        # Appends extracted text to the extracted list as a key-value pair
        extracted.append({
            "SECTION": match["section"],
            "SECTION_TEXT": extracted_text
        })
        current_pos = end_pos  # Update current_pos to end of current segment

    # If no matches are found, extract the entire note
    if not extracted:
        extracted.append({
            "SECTION": "ENTIRE NOTE",
            "SECTION_TEXT": input_text.strip()
        })

    return extracted

# -----------------------------------------------------------------------------


def extract_text(input_text: str,
                 proc_desc: str,
                 mapping_type: str,
                 match_type: str = "hard",
                 fallback: str = None) \
        -> List[Dict[str, Any]]:
    """
    Coordinates text extraction using the following pipeline:
    1. Load the keyword mappings for proc_desc
    2. Compile the needed regex from the keyword mappings
    3. Find the matches
    4. Extract the text segments based on the matching keywords
    5. If no segments found, optionally return a fallback text from another col

    Note: match_type (hard vs soft) specify the type of regex matching
    Note: mapping_type specify the mapping dict to use (see keyword_mappings.py)
    """

    # If no text is found in input_text, use fallback text (if provided).
    if pd.isna(input_text) and fallback is not None:
        segments = [{"SECTION": "FALLBACK TEXT", "SECTION_TEXT": fallback}]
        return segments
    # Else put in some random blurb
    elif pd.isna(input_text) and fallback is None:
        segments = [{"SECTION": "FALLBACK TEXT", "SECTION_TEXT": "NO INPUT TEXT"}]

    mappings = get_mappings(proc_desc, mapping_type, match_type=match_type)
    compiled = compile_mappings(mappings)
    matches = find_matches(input_text, compiled, mapping_type=mapping_type)
    segments = extract_segments(input_text, matches)

    return segments


def explode_extracted_text(df,
                           extracted_text_col: str,
                           first_only: bool = False) \
    -> pd.DataFrame:
    """
    Explode or extract text column into new columns.
    If first_only=True, keeps only the first key-value pair instead of exploding all.
    """
    # Keep only first key-value pair
    if first_only:
        df[extracted_text_col] = df[extracted_text_col].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0
            else {"SECTION": None, "SECTION_TEXT": None}
        )
        expanded = df[extracted_text_col].apply(pd.Series)
    # Explode everying in the list of dicts
    else:
        df = df.explode(extracted_text_col).reset_index(drop=True)
        expanded = df[extracted_text_col].apply(pd.Series)

    df = pd.concat([df, expanded], axis=1)

    return df


# MISC TEXT EXTRACTION UTILS
# -----------------------------------------------------------------------------
def extract_kps (text_col):
    """
    Extract the KPS score from text, handling cases like:
    - 'KPS: 90'
    - 'KPS/ECOG: 90/0'
    - 'KPS/ECOG 100/1'
    - 'KPS score = 80'
    """

    text = str(text_col)

    # 1. Handle the combined "KPS/ECOG: 90/0" case first
    match_combo = re.search(
        r'KPS\s*/\s*ECOG\s*[:=\-]?\s*([0-9]{1,3})',
        text,
        flags=re.IGNORECASE
    )
    if match_combo:
        return int(match_combo.group(1))

    # 2. General KPS extractor
    match_general = re.search(
        r'(?:KPS(?:\s*score)?)\s*[:=\- ]*\s*([0-9]{1,3})',
        text,
        flags=re.IGNORECASE
    )
    if match_general:
        return int(match_general.group(1))

    return None
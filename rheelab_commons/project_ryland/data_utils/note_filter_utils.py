"""
------------------------------------------------------------------------------
Author:         Justin Vinh
Collaborators:  Zach Tentor
Parent Package: Project Ryland
Creation Date:  2025.09.30
Last Modified:  2025.10.07

Purpose:
This package contains functions designed to filter and process progress notes
for LLM analysis.
------------------------------------------------------------------------------
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def ensure_naive_datetime(obj):
    """
    Convert datetime-like objects (Series, Index, single str/Timestamp)
    to tz-naive (strip timezone).
    """
    # Handle scalars (str, datetime, Timestamp)
    if isinstance(obj, (str, pd.Timestamp)):
        ts = pd.to_datetime(obj, errors="coerce")
        return ts.tz_localize(None) if ts.tzinfo else ts

    # Handle Series/Index
    series = pd.to_datetime(obj, errors="coerce")
    if hasattr(series.dt, "tz") and series.dt.tz is not None:
        return series.dt.tz_convert(None)
    return series.dt.tz_localize(None)


# HELPER FUNCTIONS FOR FILTER_PROGRESS_NOTES
# -----------------------------------------------------------------------------
def prep_treatment_data(
        df_tx: pd.DataFrame,
        df_info: Optional[pd.DataFrame] = None,
        tx_keywords: List[str] = []) -> pd.DataFrame:
    """
    1) Merge the treatment data with demographics data (mainly for death
    info) - this is optional
    2) clean the resultant df
    3) filter the df by 1+ treatment types
    """

    if df_info is not None:
        # Merge treatment with demographics data
        df_tx_demo = pd.merge(df_tx, df_info, on='DFCI_MRN', how='left')
    else:
        df_tx_demo = df_tx.copy()

    # Collapse the different treatment columns into one representative tx col
    # backfills with tx to the right if there is an NA
    df_tx_demo['TREATMENT'] = df_tx_demo[
        ['STD_CHEMO_PLAN', 'RESEARCH_CHEMO_PLAN', 'OTHER_TREATMENT_PLAN']
    ].bfill(axis=1).iloc[:, 0]
    # If there are still any NAs in TREATMENT, drop them like hot potatoes
    df_tx_demo = df_tx_demo.dropna(subset=['TREATMENT'])

    # Filter the df for the treatments of interest (case-insensitive)
    if tx_keywords:
        mask = df_tx_demo['TREATMENT'].str.lower().apply(
            lambda x: any(tx.lower() in x for tx in tx_keywords)
        )
        df_tx_demo = df_tx_demo[mask].reset_index(drop=True)

    return df_tx_demo


def prep_notes(
        df_notes: pd.DataFrame,
        tx_mrns: np.ndarray,
        progress_note_text_filters: List[str]) -> pd.DataFrame:
    """
    Filter notes so that only MRNs with the desired treatment(s) are kept
    """
    # Keep only progress notes with MRNs found in the treatment list
    df_notes = df_notes[df_notes['DFCI_MRN'].isin(tx_mrns)].copy()
    # Set the time format right (if not already)
    df_notes['EVENT_DATE'] = ensure_naive_datetime(df_notes['EVENT_DATE'])

    # Filter progress notes by keywords in their RPT_TEXT fields
    pattern = "|".join(progress_note_text_filters)
    df_notes = df_notes[df_notes['RPT_TEXT'].str.contains(pattern, na=False)].copy()
    print(f'# of filtered progress notes: \t\t\t{len(df_notes)}')

    return df_notes.reset_index(drop=True)


def prep_diagnosis_data(df_dx: pd.DataFrame) -> pd.DataFrame:
    """Format the diagnosis data"""
    # Drop NAs and force MRNs to integers
    df_dx = df_dx.dropna(subset=['DFCI_MRN']).reset_index(drop=True)
    df_dx['DFCI_MRN'] = df_dx['DFCI_MRN'].astype("Int64")

    # Convert DIAGNOSIS_DT to DIAGNOSIS_DATE in the proper format
    df_dx["DIAGNOSIS_DATE"] = ensure_naive_datetime(df_dx['DIAGNOSIS_DT'])

    return df_dx


def find_notes_after_diagnosis(
        df_notes: pd.DataFrame,
        df_tx: pd.DataFrame,
        df_dx: pd.DataFrame,
        earliest_data_date) -> pd.DataFrame:
    """
    Find the first progress note after a desired diagnosis for each patient.

    Returns a DataFrame that includes:
    - Treatment details (from df_tx)
    - Diagnosis information (from df_dx)
    - The first progress note after the diagnosis (from df_notes)
    """

    # Merge treatment and diagnosis data, remove MRNs that are NA
    df_merged = (pd.merge(df_tx, df_dx, on="DFCI_MRN", how="left").dropna(
        subset=["DFCI_MRN"]
    ))

    # Ensure datetime formatting is correct
    df_merged["TPLAN_START_DT"] = ensure_naive_datetime(df_merged["TPLAN_START_DT"])
    df_notes["EVENT_DATE"] = ensure_naive_datetime(df_notes["EVENT_DATE"])

    # --------------------------------------------
    # Find the age (in years) at time of diagnosis
    # Note: finds delta in days then divide by 365.25 to get years (rounded)
    df_merged["DX_AGE"] = (
        (df_merged["DIAGNOSIS_DATE"] - df_merged["BIRTH_DT"]).dt.days / 365.25
    )
    df_merged["DX_AGE"] = pd.to_numeric(
        df_merged["DX_AGE"],
        errors="coerce").round().astype("Int64")
    # --------------------------------------------

    results = []
    # For each patient, get the diagnosis and notes
    for mrn, dx_group in df_merged.groupby("DFCI_MRN"):
        if mrn not in df_notes["DFCI_MRN"].values:
            continue
        notes_group = df_notes[df_notes["DFCI_MRN"] == mrn]

        # For each diagnosis date, find the first progress note afterward
        for dx_date in dx_group["DIAGNOSIS_DATE"]:
            notes_after_dx = notes_group[notes_group["EVENT_DATE"] >= dx_date]
            if not notes_after_dx.empty:
                first_note = notes_after_dx.iloc[0]

                # Merge diagnosis info + first note info into a single dict
                combined = {**dx_group.iloc[0].to_dict(),
                            **first_note.to_dict()}
                results.append(combined)

    # Build a DataFrame from the results
    final = pd.DataFrame(results)

    # Find the time between dx and progress note
    earliest_data_date = ensure_naive_datetime(earliest_data_date)
    final["DAYS_AFTER_DX"] = (
        (final["EVENT_DATE"] - final["DIAGNOSIS_DATE"]).dt.days
    ).astype("Int64")

    # Only keep notes with diagnoses after the cutoff date
    final = final[final["DIAGNOSIS_DATE"] >= earliest_data_date].drop_duplicates()

    # Create a column marking these as "diagnosis" notes
    final["NOTE_TYPE"] = "first after dx"

    return final.reset_index(drop=True)


def find_notes_after_treatment(
        df_notes: pd.DataFrame,
        df_tx: pd.DataFrame,
        df_dx: pd.DataFrame,
        days_diff_wobble: int = 7,
        days_post_tx: int = None) -> pd.DataFrame:
    """
    For each patient, find...
    1) closest progress note to start of treatment +/- user-defined num of days
    2) closest progress note to a time point set at X days post-treatment.
    If none provided, will find notes closest to 60 days post-treatment.
    """
    # --------------------------------------------
    # Set parameters
    days_diff_wobble = days_diff_wobble  # This variable sets +/- time from tx time
    print(f'Note inclusion range: \t\t\t\t\t{days_diff_wobble} days')

    # Set up the custom time post-treatment, default 2 mo (60 days) if None provided
    if days_post_tx is None:
        days_post_tx = 60  # Days
    print(f'Analysis time point post-treatment: \t{days_post_tx} days')
    # --------------------------------------------

    # Merge the treatment and progress note dfs and drop MRNs that are NA
    df_tx_notes = (
        df_tx
        .merge(df_dx, on='DFCI_MRN', how='left')
        .merge(df_notes, on='DFCI_MRN', how='left')
        .dropna(subset=['DFCI_MRN'])
    )

    # Ensure dates are in proper format
    df_tx_notes['TPLAN_START_DT'] = ensure_naive_datetime(df_tx_notes['TPLAN_START_DT'])
    df_tx_notes['EVENT_DATE'] = ensure_naive_datetime(df_tx_notes['EVENT_DATE'])

    # --------------------------------------------
    # Find the age (in years) at time of treatment start
    # Note: finds delta in days then divide by 365.25 to get years (rounded)
    df_tx_notes["TX_START_AGE"] = (
        (df_tx_notes["TPLAN_START_DT"] - df_tx_notes["BIRTH_DT"]).dt.days / 365.25
    )
    df_tx_notes["TX_START_AGE"] = pd.to_numeric(
        df_tx_notes["TX_START_AGE"],
        errors="coerce").round().astype("Int64")
    # --------------------------------------------

    # Find notes closest to treatment start
    # ------------------------------------------
    # Calculate the date differential in days
    # Find days between progress note and treatment start
    df_tx_notes['DAYS_NOTE_TO_TX_START'] = (
        (df_tx_notes['TPLAN_START_DT'] - df_tx_notes['EVENT_DATE']).abs().dt.days
    ).astype("Int64")

    # Find the progress note closest to start of treatment (+/- X days)
    closest_start = df_tx_notes.sort_values(
        ['DFCI_MRN','TREATMENT', 'TPLAN_START_DT', 'DAYS_NOTE_TO_TX_START']
    ).drop_duplicates(subset=['DFCI_MRN', 'TREATMENT', 'TPLAN_START_DT'],
                      keep='first')
    closest_start = closest_start[
        closest_start['DAYS_NOTE_TO_TX_START'] <= days_diff_wobble
    ].copy()
    closest_start['NOTE_TYPE'] = f'within +/- {days_diff_wobble} days of tx'
    # ------------------------------------------

    # Find notes closest to X days after treatment start
    # ------------------------------------------
    # Calculate the date X days after treatment (X days time offset)
    x_days_later_date_col = f'AFTER_{days_post_tx}_DAYS_DATE'
    df_tx_notes[x_days_later_date_col] = (
        df_tx_notes['TPLAN_START_DT'] + pd.DateOffset(days=days_post_tx)
    )

    # Calculate the days between the X days time offset and each progress note
    date_diff_x_days = f"DATE_DIFF_{days_post_tx}_DAYS_OFFSET"
    df_tx_notes[date_diff_x_days] = (
        (df_tx_notes[x_days_later_date_col] - df_tx_notes['EVENT_DATE']).abs().dt.days
    ).astype("Int64")

    # Find the progress note closest to X days since start of
    # treatment within the wobble range
    closest_to_x_days = df_tx_notes.sort_values(
        ['DFCI_MRN', 'TREATMENT', 'TPLAN_START_DT', date_diff_x_days]
    ).drop_duplicates(subset=['DFCI_MRN', 'TREATMENT', 'TPLAN_START_DT'],
                      keep='first')


    closest_to_x_days['NOTE_TYPE'] = f'closest to {days_post_tx} days since tx'
    # ------------------------------------------

    # Combine them
    df_combined = pd.concat([closest_start, closest_to_x_days],
                            ignore_index=True)
    # Drop columns that don't have survival data
    # df_combined = df_combined.dropna(subset=['HYBRID_DEATH_DT']).drop_duplicates(
    #     subset=['DFCI_MRN', 'TREATMENT', 'TPLAN_START_DT', 'EVENT_DATE']
    # )

    return df_combined.reset_index(drop=True)
# -----------------------------------------------------------------------------


def filter_progress_notes(
        df_progress_notes: pd.DataFrame,
        df_treatments: pd.DataFrame,
        df_patient_info: pd.DataFrame,
        df_diagnosis: pd.DataFrame,
        treatment_keywords: List[str],
        progress_note_text_filters: List[str],
        earliest_data_date: str,
        days_diff_wobble: int,
        days_post_tx: int = None) -> pd.DataFrame:
    """
    Full workflow for filtering progress notes prior to use in LLMs
    1. Ensure MRN and date info are formatted correctly for treatment,
    demographics, and progress note data
    2. Filter the treatment and progress note data
    3. Analyze the progress note data that occur either around time of diagnosis,
    time of treatment, or at a timepoint at specified date post-treatment
    4. Combine the dataframes from (4) into one dataframe
    """

    # 1. Clean up input data -----------------------
    df_progress_notes = df_progress_notes.copy()
    df_treatments = df_treatments.copy()
    df_patient_info = df_patient_info.copy()
    df_diagnosis = df_diagnosis.copy()

    # Ensure time is in the correct format
    df_treatments['TPLAN_START_DT'] = ensure_naive_datetime(
        df_treatments['TPLAN_START_DT'])
    df_patient_info['HYBRID_DEATH_DT'] = ensure_naive_datetime(
        df_patient_info['HYBRID_DEATH_DT'])
    df_patient_info['BIRTH_DT'] = ensure_naive_datetime(
        df_patient_info['BIRTH_DT'])
    # Ensure MRNs are integers
    df_treatments['DFCI_MRN'] = df_treatments['DFCI_MRN'].astype("Int64")
    df_progress_notes['DFCI_MRN'] = df_progress_notes['DFCI_MRN'].astype("Int64")
    df_patient_info['DFCI_MRN'] = df_patient_info['DFCI_MRN'].astype("Int64")

    # 2. Prep ---------------------------------------
    df_tx_filt = prep_treatment_data(
        df_treatments,
        df_patient_info,
        treatment_keywords
    )
    df_notes_filt = prep_notes(
        df_progress_notes,
        df_tx_filt['DFCI_MRN'].unique(),
        progress_note_text_filters
    )
    df_dx = prep_diagnosis_data(df_diagnosis)

    # 3. Analyze ------------------------------------
    notes_after_dx = find_notes_after_diagnosis(
        df_notes_filt,
        df_tx_filt,
        df_dx,
        earliest_data_date
    )
    notes_after_tx = find_notes_after_treatment(
        df_notes_filt,
        df_tx_filt,
        df_dx,
        days_diff_wobble=days_diff_wobble,
        days_post_tx=days_post_tx
    )

    # 4. Combine ------------------------------------
    df_final = pd.concat([notes_after_dx, notes_after_tx], ignore_index=True)
    df_final = df_final.dropna(subset=['RPT_TEXT']).reset_index(drop=True)

    return df_final

# MISC ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------

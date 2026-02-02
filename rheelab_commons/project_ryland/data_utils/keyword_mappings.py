"""
------------------------------------------------------------------------------
Author:         Justin Vinh
Collaborators:  Zach Tentor
Parent Package: Project Ryland
Creation Date:  2025.09.29
Last Modified:  2025.10.07

Purpose:
Contain the keyword mappings for the Project Ryland Utils. This module
also contains keywords for several scripts that are project-specific.
------------------------------------------------------------------------------
"""


# Symptoms of Interest
# ----------------------------------------------------------------------------
gwas_prompt_variables_v1 = {
    'symptoms': [
        'headache',
        'hair loss',
        'fatigue',
        'nausea',
        'anxiety',
        'difficulty sleeping',
        'numbness and tingling',
        'joint pain',
        'rash',
        'diarrhea',
        'constipation',
        'other'
    ]
}


# Input text keywords
progress_note_text_filters = ['CENTER FOR NEURO-ONCOLOGY',
                              'NEURO-ONCOLOGY PROGRESS NOTE',
                              'Subjective: Patient ID',
                              'HISTORY OF PRESENT ILLNESS',
                              'INTERVAL HISTORY'
]

neuro_onc_tumor_keywords = ["glioblastoma",
                            "astrocytoma",
                            "oligodendroglioma",
                            "glioma"
]

# Note types (process description) of interest
# (Used to pre-filter in the main script)
# ----------------------------------------------------------------------------
pathology_proc_desc_of_interest = [
    'SURGICAL PATHOLOGY',
    'ANATOMIC PATHOLOGY',
    'OTHER PATHOLOGY RESULTS',
    'OUTSIDE PATHOLOGY REVIEW',
    'FLOW CYTOMETRY']

image_proc_desc_of_interest = [
    'CT CHEST',
    'CT PET CHEST'
]

# Mappings of keywords by process descriptions
# This dict contains rules for keywords and conditions for each process
# description in RPT_TEXT. It will be used to extract specific text segments
# to be fed to the LLM.
# ----------------------------------------------------------------------------
image_mappings_by_proc_desc =   {
    "*": [
        {
            "start": r"IMPRESSION:",
            "end": ["END IMPRESSION",
                    "This report was electronically signed"],
            "condition": None
        }
    ]
}

progress_note_mappings_by_proc_desc = {
    "Progress Note": [
        {
            "start": r"EXAM:",
            "end": ["DATA:",
                    "MRI"],
            "condition": None
        },
        {
            "start": r"EXAMINATION",
            "end": ["LABORATORY",
                    "ASSESSMENT AND PLAN"],
            "condition": None
        },
        {
            "start": r"PHYSICAL EXAM",
            "end": ["LABORATORY",
                    "IMAGING",
                    "IMPRESSION AND PLAN","LABS",
                    "RADIOGRAPHIC EXAMINATION",
                    "Radiology",
                    "PLAN",
                    "LAB RESULT",
                    "IMPRESSION",
                    "ASSESSMENT/PLAN"],
            "condition": None
        },
        {
            "start": r"Physical Exam",
            "end": ["TEST",
                    "Results",
                    "Labs",
                    "Blood Draw",
                    "LABORATORY"],
            "condition": None
        },
        {
            "start": r"Physical exam",
            "end": ["Lab Results"],
            "condition": None
        },
        {
            "start": r"Examination",
            "end": ["Data Review",
                    "Prior Work up",
                    "Assessment and Plan"],
            "condition": None
    }
    ]
}

pathology_mappings_by_proc_desc = {
    "SURGICAL PATHOLOGY": [
        {
            "start": r"INTERPRETATION",
            "end": ["TEST INFORMATION",
                    "Final Diagnosis",
                    "Prior Results"],
            "condition": None
        },
        {
            "start": r"FINAL PATHOLOGIC DIAGNOSIS",
            "end": ["Electronically Signed Out"],
            "condition": None
        },
        {
            "start": r"PATHOLOGIC DIAGNOSIS",
            "end": ["CLINICAL DATA"],
            "condition": None
        },
        {
            "start": r"FINAL DIAGNOSIS",
            "end": ["GROSS DESCRIPTION",
                    "Final Diagnosis"],
            "condition": None
        },
        {
            "start": r"DIAGNOSIS",
            "end": ["Gross Description",
                    "Final Diagnosis"],
            "condition": None
        },
        {
            "start": r"Diagnosis",
            "end": ["Gross Description"],
            "condition": "exclude",
            "exclude_after": "CLINICAL DATA"
        },
        {
            "start": r"Diagnosis",
            "end": ["Electronically Signed Out"],
            "condition": None
        },
        {
            "start": r"Final Diagnosis",
            "end": ["Electronically Signed Out"],
            "condition": None
        }
    ],
    "ANATOMIC PATHOLOGY": [
        {
            "start": r"PATHOLOGIC DIAGNOSIS",
            "end": ["CLINICAL DATA"],
            "condition": None
        },
        {
            "start": r"FINAL PATHOLOGIC DIAGNOSIS",
            "end": ["Electronically Signed Out"],
            "condition": None
        },
        {
            "start": r"FINAL DIAGNOSIS",
            "end": ["Clinical History"],
            "condition": None
        },
        {
            "start": r"RESULTS",
            "end": ["REFERENCES"],
            "condition": None
        },
        {
            "start": r"INTERPRETATION",
            "end": ["CLINICAL DATA"],
            "condition": None
        },
        {
            "start": r"Diagnosis",
            "end": ["Electronically Signed Out"],
            "condition": None
        },
        {
            "start": r"DIAGNOSIS",
            "end": ["Gross Description"],
            "condition": "exclude",
            "exclude_after": "CLINICAL DATA"
        },
        {
            "start": r"Final Diagnosis",
            "end": ["Clinical History:"],
            "condition": None
        }
    ],
    "FLOW CYTOMETRY": [
        {
            "start": r"INTERPRETATION",
            "end": ["By his/her signature",
                    "These tests were developed"],
            "condition": None
        }
        # Additional mappings can be added if necessary
    ],
    "OUTSIDE PATHOLOGY REVIEW": [
        {
            "start": r"PATHOLOGIC DIAGNOSIS",
            "end": ["CLINICAL DATA"],
            "condition": None
        },
        {
            "start": r"INTEGRATED DIAGNOSIS",
            "end": ["Electronically Signed"],
            "condition": None
        }
        # Clarification needed on how to handle overlapping diagnostics
    ],
    "OTHER PATHOLOGY RESULTS": [
        {
            "start": r"DIAGNOSIS",
            "end": ["CLINICAL DATA"],
            "condition": None
        },
        {
            "start": r"PATHOLOGIC DIAGNOSIS",
            "end": ["CLINICAL DATA"],
            "condition": None
        },
        {
            "start": r"FINAL DIAGNOSIS",
            "end": ["GROSS DESCRIPTION"],
            "condition": None
        },
        {
            "start": r"CYTOLOGIC DIAGNOSIS",
            "end": [],  # No specific end keywords provided
            "condition": None
        },
        {
            "start": r"Final Diagnosis",
            "end": ["Clinical History"],
            "condition": None
        },
        {
            "start": r"INTERPRETATION",
            "end": ["Final Diagnosis"],
            "condition": None
        },
        {
            "start": r"RESULT",
            "end": ["Final Diagnosis"],
            "condition": None
        },
        {
            "start": r"Result",
            "end": ["Final Diagnosis"],
            "condition": None
        }
    ],
    "PROGRESS NOTES": [
        {
            "start": r"EXAMINATION",
            "end": ["LABORATORY",
                    "ASSESSMENT AND PLAN"],
            "condition": None
        },
        {
            "start": r"PHYSICAL EXAM",
            "end": ["LABORATORY",
                    "IMAGING",
                    "IMPRESSION AND PLAN","LABS",
                    "RADIOGRAPHIC EXAMINATION",
                    "Radiology",
                    "PLAN",
                    "LAB RESULT",
                    "IMPRESSION",
                    "ASSESSMENT/PLAN"],
            "condition": None
        },
        {
            "start": r"Physical Exam",
            "end": ["TEST",
                    "Results",
                    "Labs",
                    "Blood Draw",
                    "LABORATORY"],
            "condition": None
        },
        {
            "start": r"Physical exam",
            "end": ["Lab Results"],
            "condition": None
        },
        {
            "start": r"Examination",
            "end": ["Data Review",
                    "Prior Work up",
                    "Assessment and Plan"],
            "condition": None
    }
    ]
}


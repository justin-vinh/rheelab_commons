"""
------------------------------------------------------------------------------
Author:         Justin Vinh
Collaborators:  Thomas Sounack
Parent Package  Project Ryland
Created:        2025.10.10

LLM Prompt Structures File
Purpose: Provides output structures to various prompts
------------------------------------------------------------------------------
"""


from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, create_model

from project_ryland.data_utils.keyword_mappings import gwas_prompt_variables_v1


class NANO_Scale_Score_Three(int, Enum):
    score_0 = 0
    score_1 = 1
    score_2 = 2
    score_3 = 3

class NANO_Scale_Score_Two(int, Enum):
    score_0 = 0
    score_1 = 1
    score_2 = 2

class AssessNANO(BaseModel):
    Gait: Optional[NANO_Scale_Score_Three]
    Strength: Optional[NANO_Scale_Score_Three]
    Ataxia: Optional[NANO_Scale_Score_Two]
    Sensation: Optional[NANO_Scale_Score_Two]
    Visual_Fields: Optional[NANO_Scale_Score_Three]
    Facial_Strength: Optional[NANO_Scale_Score_Two]
    Language: Optional[NANO_Scale_Score_Three]
    Level_of_Consciousness: Optional[NANO_Scale_Score_Three]
    Behavior: Optional[NANO_Scale_Score_Two]

class AssessNanoImaging(BaseModel):
    """
    Structured schema for extracting key diagnostic findings
    from radiology report text.
    """
    diagnosis: Optional[str] = Field(
        None,
        description="Primary diagnosis inferred from the imaging report "
                    "(e.g., 'pulmonary embolism', pancreatic cancer with "
                    "metastases to the bones, 'no acute findings')."
    )
    supporting_text: Optional[str] = Field(
        None,
        description="Exact span of text from the imaging report that supports the diagnosis."
    )
    pe_status: Optional[bool] = Field(
        None,
        description="True if pulmonary embolism (PE) is mentioned or diagnosed, "
                    "False if explicitly ruled out, None if not mentioned."
    )
    dvt_status: Optional[bool] = Field(
        None,
        description="True if deep vein thrombosis (DVT) is mentioned or diagnosed, "
                    "False if explicitly ruled out, None if not mentioned."
    )

class AssessNanoPathology(BaseModel):
    """
    Structured schema for extracting the main diagnostic elements
    from a pathology report, including molecular markers.
    """
    diagnosis: Optional[str] = Field(
        None,
        description="Primary pathologic diagnosis inferred from the pathology report"
                    "(e.g., 'glioblastoma', 'astrocytoma, IDH-mutant')."
    )
    supporting_text: Optional[str] = Field(
        None,
        description="Exact text span from the pathology report that supports the diagnosis."
    )
    idh_status: Optional[str] = Field(
        None,
        description="Reported IDH mutation status "
                    "(e.g., 'mutant', 'wildtype', 'not assessed')."
    )
    who_grade: Optional[str] = Field(
        None,
        description="WHO tumor grade if mentioned "
                    "(e.g., 'Grade II', 'Grade IV')."
    )
    mgmt_status: Optional[str] = Field(
        None,
        description="MGMT promoter methylation status "
                    "(e.g., 'methylated', 'unmethylated', 'not assessed')."
    )


# Prompt Structure for GWAS symptom assessment
# ----------------------------------------------------------------------------
class AssessSymptomDetail(BaseModel):
    """
    Structured schema for extracting symptom info from progress notes
    """
    status: str = Field(
        None,
        description="The status of a symptom based on the progress note. "
                    "The status must be 'Affirmed', 'Negated', or 'Absent'"
    )
    text: str = Field(
        None,
        description="Exact text span from the pathology report that supports the "
                    "assigned status for the symptom."
    )


symptom_list = gwas_prompt_variables_v1['symptoms']
fields = {
    symptom.replace(' ', '_'): (AssessSymptomDetail, ...)
    for symptom in symptom_list
}
AssessSymptoms = create_model(
    'AssessSymptoms',
    **fields
)

# ----------------------------------------------------------------------------
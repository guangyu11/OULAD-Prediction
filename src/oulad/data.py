"""Loading the raw OULAD tables and building the enrolment-level base table."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import KEY, NA_VALUES

# studentVle has 10.7M rows; explicit dtypes cut memory from ~1.5 GB to ~250 MB.
VLE_DTYPES = {
    "code_module": "category",
    "code_presentation": "category",
    "id_student": "int32",
    "id_site": "int32",
    "date": "int16",
    "sum_click": "int32",
}

EDUCATION_ORDER = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
                   "HE Qualification": 3, "Post Graduate Qualification": 4}
AGE_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}


@dataclass
class OULADData:
    """The seven raw OULAD tables."""

    student_info: pd.DataFrame
    student_reg: pd.DataFrame
    student_asm: pd.DataFrame
    assessments: pd.DataFrame
    vle_meta: pd.DataFrame
    courses: pd.DataFrame
    student_vle: pd.DataFrame

    @classmethod
    def load(cls, data_dir="data"):
        """Read the CSVs from `data_dir`, treating "?" as missing."""
        d = Path(data_dir)
        read = lambda name, **kw: pd.read_csv(d / f"{name}.csv", na_values=NA_VALUES, **kw)
        return cls(
            student_info=read("studentInfo"),
            student_reg=read("studentRegistration"),
            student_asm=read("studentAssessment"),
            assessments=read("assessments"),
            vle_meta=read("vle"),
            courses=read("courses"),
            student_vle=read("studentVle", dtype=VLE_DTYPES),
        )


def build_enrolments(data):
    """One cleaned row per enrolment, with the target and encoded background fields."""
    info = data.student_info.copy()

    # Positive class is non-completion, so precision and recall map onto the two
    # costs of an early-warning system.
    info["at_risk"] = (~info["final_result"].isin(["Pass", "Distinction"])).astype("int8")

    # "10-20" is missing its "%" in 3,516 records; without the fix it is a separate category.
    info["imd_band"] = info["imd_band"].replace({"10-20": "10-20%"}).astype("string")

    # Ordered categories become ranks rather than one-hot columns.
    info["imd_band_num"] = info["imd_band"].map(
        {f"{i}-{i+10}%": i + 5 for i in range(0, 100, 10)}).astype("float32")
    info["highest_education_ord"] = info["highest_education"].map(EDUCATION_ORDER).astype("float32")
    info["age_band_ord"] = info["age_band"].map(AGE_ORDER).astype("float32")

    info = info.merge(data.student_reg, on=KEY, how="left", validate="one_to_one")

    # Negative date_registration means days before the start; flipped so larger = earlier.
    info["reg_lead_days"] = -info["date_registration"]
    info["reg_date_missing"] = info["date_registration"].isna().astype("int8")
    # 234 records (0.72%) register after the module starts; flagged rather than dropped.
    info["registered_after_start"] = (info["date_registration"] > 0).fillna(False).astype("int8")

    info = info.merge(data.courses, on=["code_module", "code_presentation"], how="left")

    assert len(info) == len(data.student_info)
    return info


def prepare_vle(data):
    """Attach each click's activity type and return (clicks, list of all activity types)."""
    assert data.vle_meta["id_site"].is_unique
    site2type = data.vle_meta.set_index("id_site")["activity_type"]

    # Anchored to the full type list from vle.csv so every cutoff has the same columns,
    # even when a type is not clicked before an early cutoff.
    activity_types = sorted(data.vle_meta["activity_type"].dropna().unique().tolist())

    clicks = data.student_vle
    clicks["activity_type"] = clicks["id_site"].map(site2type).astype("category")
    assert clicks["activity_type"].isna().sum() == 0
    return clicks, activity_types


def prepare_assessments(data):
    """Submissions with module, presentation, due date and weight attached."""
    # studentAssessment carries no module or presentation; both arrive via assessments.
    asm = data.student_asm.merge(
        data.assessments[["id_assessment", "code_module", "code_presentation",
                          "assessment_type", "date", "weight"]],
        on="id_assessment", how="left", validate="many_to_one")
    asm = asm.rename(columns={"date": "due_date"})
    asm["days_early"] = asm["due_date"] - asm["date_submitted"]
    return asm

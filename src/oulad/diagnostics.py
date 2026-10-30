"""Data audit: non-obvious properties of OULAD that shape the modelling choices."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import KEY, NA_VALUES


def hidden_missing_values(data_dir="data"):
    """Columns where missing values are the string "?" and so look complete on a default read."""
    rows = []
    for name in ["studentInfo", "studentRegistration", "studentAssessment", "vle"]:
        path = Path(data_dir) / f"{name}.csv"
        raw = pd.read_csv(path)
        fixed = pd.read_csv(path, na_values=NA_VALUES)
        for col in raw.columns:
            n = (raw[col].astype(str).str.strip() == "?").sum()
            if n:
                rows.append({"table": name, "column": col, "n_question_marks": n,
                             "share_%": round(n / len(raw) * 100, 2),
                             "missing_rate_default_%": round(raw[col].isna().mean() * 100, 2),
                             "dtype_default": str(raw[col].dtype),
                             "dtype_corrected": str(fixed[col].dtype)})
    return pd.DataFrame(rows)


def unregistration_leakage(data):
    """date_unregistration is filled almost only for withdrawals, so it encodes the label."""
    chk = data.student_info.merge(data.student_reg, on=KEY, how="left")
    out = pd.crosstab(chk["final_result"], chk["date_unregistration"].isna(),
                      normalize="index").round(4) * 100
    out.columns = ["has_unregistration_date_%", "no_unregistration_date_%"]
    return out


def grouped_structure(data):
    """How strongly a student's repeated enrolments depend on each other."""
    info = data.student_info
    vc = info["id_student"].value_counts()
    si = info.copy()
    si["completed"] = si["final_result"].isin(["Pass", "Distinction"])

    order = {"2013B": 0, "2013J": 1, "2014B": 2, "2014J": 3}
    multi = (si[si["id_student"].isin(vc[vc > 1].index)]
             .assign(_o=lambda d: d["code_presentation"].map(order))
             .sort_values(["id_student", "_o"]))
    pairs = multi.groupby("id_student")["completed"].apply(list)
    pairs = pairs[pairs.apply(len) >= 2]
    first, second = pairs.apply(lambda x: x[0]), pairs.apply(lambda x: x[1])

    return pd.DataFrame([
        {"quantity": "enrolments", "value": len(info)},
        {"quantity": "unique students", "value": info["id_student"].nunique()},
        {"quantity": "students with >1 enrolment", "value": int((vc > 1).sum())},
        {"quantity": "max enrolments per student", "value": int(vc.max())},
        {"quantity": "completion rate, overall %", "value": round(si["completed"].mean() * 100, 1)},
        {"quantity": "completion rate | first attempt completed %",
         "value": round(second[first].mean() * 100, 1)},
        {"quantity": "completion rate | first attempt failed %",
         "value": round(second[~first].mean() * 100, 1)},
    ])


def skew_report(table, numeric_cols):
    """Skew before/after log1p for each numeric column (diagnostic only)."""
    rows = []
    for col in numeric_cols:
        s = table[col].dropna()
        if s.nunique() <= 2 or s.min() < 0:
            continue
        sk, skl = s.skew(), np.log1p(s).skew()
        rows.append({"column": col, "skew": round(sk, 2), "skew_after_log": round(skl, 2),
                     "reduction_%": round((abs(sk) - abs(skl)) / abs(sk) * 100, 1) if sk else 0.0,
                     "max": round(s.max(), 2)})
    return pd.DataFrame(rows).sort_values("skew", ascending=False)

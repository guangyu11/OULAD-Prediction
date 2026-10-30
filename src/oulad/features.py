"""Point-in-time feature construction: one wide table per prediction point."""

import numpy as np
import pandas as pd

from .config import KEY, TARGET, VLE_WINDOW_START

# Known at registration (static).
BACKGROUND_CAT = ["code_module", "code_presentation", "gender", "region", "disability"]
BACKGROUND_NUM = ["highest_education_ord", "imd_band_num", "age_band_ord",
                  "num_of_prev_attempts", "studied_credits",
                  "reg_lead_days", "reg_date_missing", "registered_after_start",
                  "module_presentation_length"]

# Never usable as features: the target, the leaking column, raw forms of encoded
# variables, the identifier and the cutoff marker.
FORBIDDEN = ["final_result", "at_risk", "date_unregistration", "date_registration",
             "imd_band", "highest_education", "age_band", "id_student", "cutoff_day"]


VLE_COLUMNS = ["vle_total_clicks", "vle_active_days", "vle_n_sites", "vle_max_clicks_day",
               "vle_mean_clicks_per_active_day", "vle_clicks_h1", "vle_clicks_h2",
               "vle_trend_ratio", "vle_clicks_before_start"]
ASM_COLUMNS = ["asm_n_submitted", "asm_mean_score", "asm_min_score", "asm_n_banked",
               "asm_mean_days_early", "asm_weighted_score"]


def _empty(columns):
    """A no-rows frame with the expected columns, so a quiet cutoff keeps the same schema."""
    return pd.DataFrame(columns=columns, index=pd.MultiIndex.from_tuples([], names=KEY), dtype=float)


def _normalise_key(f):
    """Cast the key to str/int64 so category-dtype indexes join correctly."""
    f = f.reset_index()
    f["code_module"] = f["code_module"].astype(str)
    f["code_presentation"] = f["code_presentation"].astype(str)
    f["id_student"] = f["id_student"].astype("int64")
    return f.set_index(KEY)


def vle_features(clicks, cutoff, activity_types):
    """Aggregate the clickstream to one row per enrolment, using days <= cutoff only."""
    w = clicks[clicks["date"] <= cutoff]
    if len(w) == 0:
        return _empty(VLE_COLUMNS + ["vle_prop_" + str(t) for t in activity_types])

    g = w.groupby(KEY, observed=True)
    f = g.agg(
        vle_total_clicks=("sum_click", "sum"),
        vle_active_days=("date", "nunique"),
        vle_n_sites=("id_site", "nunique"),
        vle_max_clicks_day=("sum_click", "max"),
    )
    f["vle_mean_clicks_per_active_day"] = f["vle_total_clicks"] / f["vle_active_days"]

    # Trend: a declining trajectory should signal disengagement more than a flat low one.
    mid = (VLE_WINDOW_START + cutoff) / 2
    h1 = w[w["date"] <= mid].groupby(KEY, observed=True)["sum_click"].sum()
    h2 = w[w["date"] > mid].groupby(KEY, observed=True)["sum_click"].sum()
    f["vle_clicks_h1"] = h1.reindex(f.index).fillna(0)
    f["vle_clicks_h2"] = h2.reindex(f.index).fillna(0)
    f["vle_trend_ratio"] = (f["vle_clicks_h2"] + 1) / (f["vle_clicks_h1"] + 1)

    f["vle_clicks_before_start"] = (w[w["date"] <= 0].groupby(KEY, observed=True)["sum_click"]
                                    .sum().reindex(f.index).fillna(0))

    # Composition rather than volume: share of clicks per resource type.
    at = (w.groupby(KEY + ["activity_type"], observed=True)["sum_click"]
            .sum().unstack(fill_value=0))
    at = at.div(at.sum(axis=1).replace(0, np.nan), axis=0)
    at = at.reindex(columns=activity_types, fill_value=0.0)
    at.columns = ["vle_prop_" + str(col) for col in at.columns]

    return _normalise_key(f.join(at))


def assessment_features(asm, cutoff):
    """Aggregate submissions to one row per enrolment, using submissions <= cutoff only."""
    w = asm[asm["date_submitted"] <= cutoff].copy()
    if len(w) == 0:
        return _empty(ASM_COLUMNS)

    w["_wscore"] = w["score"] * w["weight"]
    f = w.groupby(KEY, observed=True).agg(
        asm_n_submitted=("id_assessment", "nunique"),
        asm_mean_score=("score", "mean"),
        asm_min_score=("score", "min"),
        asm_n_banked=("is_banked", "sum"),
        asm_mean_days_early=("days_early", "mean"),
        _wsum=("_wscore", "sum"),
        _wtot=("weight", "sum"),
    )
    f["asm_weighted_score"] = f["_wsum"] / f["_wtot"].replace(0, np.nan)
    return _normalise_key(f.drop(columns=["_wsum", "_wtot"]))


class FeatureBuilder:
    """Builds the wide table for a prediction point.

    Two rules make every table a fair early-warning snapshot:
    - Point-in-time features: only clicks and submissions dated on or before the cutoff.
    - Risk set: students who already withdrew by the cutoff are excluded, since their
      outcome is already known.
    """

    def __init__(self, enrolments, clicks, asm, activity_types):
        self.enrolments = enrolments
        self.clicks = clicks
        self.asm = asm
        self.activity_types = activity_types

    def build(self, cutoff):
        """Return (wide table, summary dict) for one cutoff day."""
        base = self.enrolments.copy()

        already_withdrawn = (base["date_unregistration"].notna()
                             & (base["date_unregistration"] <= cutoff))
        n_excluded = int(already_withdrawn.sum())
        base = base.loc[~already_withdrawn].copy()

        vf = vle_features(self.clicks, cutoff, self.activity_types)
        af = assessment_features(self.asm, cutoff)
        out = base.set_index(KEY).join(vf, how="left").join(af, how="left").reset_index()

        vle_cols = [c for c in out.columns if c.startswith("vle_")]

        # Membership of the aggregate index, so the flags do not depend on column order.
        idx = pd.MultiIndex.from_frame(out[KEY])
        out["has_vle_activity"] = idx.isin(vf.index).astype("int8")
        out["has_asm_activity"] = idx.isin(af.index).astype("int8")
        n_zero_vle = int((out["has_vle_activity"] == 0).sum())

        # No activity genuinely means zero for counts; proportions and means stay NaN
        # so the modelling imputer handles them.
        count_like = [c for c in vle_cols if not any(k in c for k in ("prop", "trend", "mean"))]
        out[count_like] = out[count_like].fillna(0)
        out[["asm_n_submitted", "asm_n_banked"]] = out[["asm_n_submitted", "asm_n_banked"]].fillna(0)

        # Active days as a ratio, since the window length varies by cutoff.
        out["vle_active_days_ratio"] = out["vle_active_days"] / (cutoff - VLE_WINDOW_START + 1)
        out["cutoff_day"] = cutoff

        summary = dict(cutoff=cutoff, n_excluded=n_excluded, n=len(out),
                       n_students=out["id_student"].nunique(),
                       zero_vle=n_zero_vle, pos_rate=round(out[TARGET].mean() * 100, 2))
        return out, summary

    def build_all(self, cutoffs):
        """Return ({cutoff: wide table}, summary DataFrame)."""
        tables, summary = {}, []
        for cut in cutoffs:
            tables[cut], s = self.build(cut)
            summary.append(s)
        return tables, pd.DataFrame(summary)


def feature_sets(table):
    """The three feature groups compared in RQ2: background, behaviour and all."""
    behaviour = ([c for c in table.columns if c.startswith(("vle_", "asm_"))]
                 + ["has_vle_activity", "has_asm_activity"])
    sets = {
        "background": BACKGROUND_CAT + BACKGROUND_NUM,
        "behaviour": behaviour,
        "all": BACKGROUND_CAT + BACKGROUND_NUM + behaviour,
    }
    # Guards so the three sets cannot drift apart or leak.
    assert set(sets["all"]) == set(sets["background"]) | set(sets["behaviour"])
    assert not (set(sets["background"]) & set(sets["behaviour"]))
    for cols in sets.values():
        assert all(c in table.columns for c in cols)
        assert not any(c in FORBIDDEN for c in cols)
    return sets

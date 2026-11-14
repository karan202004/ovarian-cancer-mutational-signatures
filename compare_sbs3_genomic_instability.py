
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

INPUT = (
    BASE / "results" / "all_diagnostics" / "survival_preparation"
    / "survival_analysis_dataset_preliminary.csv"
)

OUT = BASE / "results" / "survival_analysis" / "sbs3_genomic_instability"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("SBS3 BEYOND AGE AND GENOMIC INSTABILITY")
print("=" * 68)

df = pd.read_csv(INPUT, sep=None, engine="python")
df.columns = df.columns.str.strip()

print("\nAvailable BRCA-related candidate columns:")
for col in df.columns:
    if "BRCA" in col.upper() or "SOMATIC_STATUS" in col.upper():
        print(" ", col)

if "SOMATIC_STATUS" in df.columns:
    print("\nSOMATIC_STATUS counts (not assumed to mean BRCA status):")
    print(df["SOMATIC_STATUS"].value_counts(dropna=False).to_string())

required = [
    "OS_MONTHS",
    "OS_STATUS",
    "AGE",
    "SBS3_MAIN_PERCENT",
    "ANEUPLOIDY_SCORE",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Prepare columns.
d = df[
    ["OS_MONTHS", "AGE", "SBS3_MAIN_PERCENT", "ANEUPLOIDY_SCORE"]
].copy()

d.columns = ["time", "age", "sbs3", "aneuploidy"]

for col in ["time", "age", "sbs3", "aneuploidy"]:
    d[col] = pd.to_numeric(d[col], errors="coerce")

# Correctly convert statuses such as 1:DECEASED and 0:LIVING.
status = df["OS_STATUS"].astype("string").str.strip().str.upper()
event_text = status.str.split(":").str[0]
d["event"] = pd.to_numeric(event_text, errors="coerce")

print("\nOriginal OS_STATUS values:")
print(status.value_counts(dropna=False).to_string())

print("\nConverted event counts before filtering:")
print(d["event"].value_counts(dropna=False).to_string())

# Keep patients with complete data for all four models.
d = d.replace([np.inf, -np.inf], np.nan).dropna()
d = d[
    (d["time"] > 0)
    & d["event"].isin([0, 1])
].copy()

if d.empty or d["event"].nunique() != 2:
    raise ValueError(
        "Both deaths and censored observations are required. "
        "Check OS_STATUS and missing values."
    )

d["event"] = d["event"].astype(int)
d["age_per_10"] = d["age"] / 10
d["sbs3_per_10"] = d["sbs3"] / 10

# Standardize aneuploidy using this common analysis cohort.
sd = d["aneuploidy"].std()
if not np.isfinite(sd) or sd == 0:
    raise ValueError("ANEUPLOIDY_SCORE has no usable variation.")

d["aneuploidy_per_sd"] = (
    d["aneuploidy"] - d["aneuploidy"].mean()
) / sd

print("\nCOMMON COMPLETE-CASE COHORT")
print("Patients:", len(d))
print("Deaths:", int(d["event"].sum()))
print("Censored:", int((d["event"] == 0).sum()))

models = {
    "Age only": ["age_per_10"],
    "Age + SBS3": ["age_per_10", "sbs3_per_10"],
    "Age + aneuploidy": ["age_per_10", "aneuploidy_per_sd"],
    "Age + SBS3 + aneuploidy": [
        "age_per_10",
        "sbs3_per_10",
        "aneuploidy_per_sd",
    ],
}

fitted = {}
rows = []

for name, predictors in models.items():
    fit_data = d[["time", "event"] + predictors]

    cph = CoxPHFitter()
    cph.fit(
        fit_data,
        duration_col="time",
        event_col="event",
    )
    fitted[name] = cph

    risk = cph.predict_partial_hazard(d[predictors]).to_numpy().ravel()
    cindex = concordance_index(d["time"], -risk, d["event"])

    rows.append({
        "model": name,
        "patients": len(d),
        "deaths": int(d["event"].sum()),
        "partial_AIC": cph.AIC_partial_,
        "in_sample_C_index": cindex,
        "log_likelihood": cph.log_likelihood_,
    })

    print(f"\n{name}")
    print(
        cph.summary[
            [
                "exp(coef)",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
                "p",
            ]
        ].to_string()
    )

comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "model_comparison.csv", index=False)

# Nested likelihood-ratio tests.
lr_comparisons = [
    ("Age only", "Age + SBS3"),
    ("Age only", "Age + aneuploidy"),
    ("Age + SBS3", "Age + SBS3 + aneuploidy"),
    ("Age + aneuploidy", "Age + SBS3 + aneuploidy"),
]

lr_rows = []

for smaller, larger in lr_comparisons:
    lr = max(
        0,
        2 * (
            fitted[larger].log_likelihood_
            - fitted[smaller].log_likelihood_
        ),
    )

    df_diff = len(models[larger]) - len(models[smaller])
    p_value = chi2.sf(lr, df_diff)

    lr_rows.append({
        "comparison": f"{smaller} vs {larger}",
        "LR_statistic": lr,
        "degrees_freedom": df_diff,
        "p_value": p_value,
    })

lr_table = pd.DataFrame(lr_rows)
lr_table.to_csv(OUT / "likelihood_ratio_tests.csv", index=False)

# Five-fold cross-validation using identical folds for every model.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

for fold, (train_idx, test_idx) in enumerate(kf.split(d), start=1):
    train = d.iloc[train_idx]
    test = d.iloc[test_idx]

    for name, predictors in models.items():
        cph = CoxPHFitter()
        cph.fit(
            train[["time", "event"] + predictors],
            duration_col="time",
            event_col="event",
        )

        risk = cph.predict_partial_hazard(
            test[predictors]
        ).to_numpy().ravel()

        try:
            cidx = concordance_index(
                test["time"],
                -risk,
                test["event"],
            )
        except (ZeroDivisionError, ValueError):
            cidx = np.nan

        cv_rows.append({
            "fold": fold,
            "model": name,
            "test_patients": len(test),
            "test_deaths": int(test["event"].sum()),
            "C_index": cidx,
        })

cv = pd.DataFrame(cv_rows)
cv.to_csv(OUT / "cross_validated_c_index_by_fold.csv", index=False)

cv_summary = (
    cv.groupby("model")["C_index"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

cv_summary.to_csv(
    OUT / "cross_validated_c_index_summary.csv",
    index=False,
)

print("\n" + "=" * 68)
print("MODEL COMPARISON")
print("=" * 68)
print(comparison.round(4).to_string(index=False))

print("\nLIKELIHOOD-RATIO TESTS")
print(lr_table.round(5).to_string(index=False))

print("\n5-FOLD CROSS-VALIDATED C-INDEX")
print(cv_summary.round(4).to_string(index=False))

print("\nResults saved to:")
print(OUT)

print("\nBRCA1/2 status has not been tested.")
print("SOMATIC_STATUS is not being treated as BRCA status.")

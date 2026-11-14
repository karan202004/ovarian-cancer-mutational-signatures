
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

INPUT_FILE = (
    PROJECT
    / "results"
    / "all_diagnostics"
    / "survival_preparation"
    / "survival_analysis_dataset_preliminary.csv"
)

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "survival_analysis"
    / "sbs3_beyond_age"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n"
        "Check the file location before running again."
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("DOES SBS3 ADD PROGNOSTIC INFORMATION BEYOND AGE?")
print("=" * 65)
print(f"Input file: {INPUT_FILE}")
print(f"Original rows: {len(df)}")
print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# 2. IDENTIFY THE REQUIRED COLUMNS
# ============================================================

def normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(candidates, description):
    normalized = {normalize(col): col for col in df.columns}

    for candidate in candidates:
        if normalize(candidate) in normalized:
            return normalized[normalize(candidate)]

    raise ValueError(
        f"\nCould not identify the {description} column.\n"
        f"Available columns: {df.columns.tolist()}\n"
        "Check the column names and add the correct name "
        "to the candidate list in the script."
    )


time_col = find_column(
    ["OS_MONTHS", "overall_survival_months", "OS_TIME"],
    "overall-survival time"
)

status_col = find_column(
    ["OS_STATUS", "overall_survival_status", "OS_EVENT"],
    "overall-survival status"
)

age_col = find_column(
    ["AGE", "AGE_AT_DIAGNOSIS", "AGE_AT_INITIAL_PATHOLOGIC_DIAGNOSIS"],
    "age"
)

sbs3_col = find_column(
    [
        "SBS3_MAIN_PERCENT",
        "SBS3",
        "SBS3_PERCENT",
        "SBS3 contribution"
    ],
    "SBS3 contribution"
)

print("\nColumns selected:")
print("Survival time:", time_col)
print("Survival status:", status_col)
print("Age:", age_col)
print("SBS3:", sbs3_col)


# ============================================================
# 3. PREPARE ONE COMMON COHORT FOR BOTH MODELS
# ============================================================

def convert_event(value):
    """Convert TCGA survival status into 1=death, 0=censored."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return int(value)

    text = str(value).strip().upper()

    if "DECEASED" in text or text in {"DEAD", "1"}:
        return 1

    if "LIVING" in text or text in {"ALIVE", "0"}:
        return 0

    return np.nan


analysis = pd.DataFrame({
    "OS_MONTHS": pd.to_numeric(df[time_col], errors="coerce"),
    "EVENT": df[status_col].apply(convert_event),
    "AGE_PER_10": pd.to_numeric(df[age_col], errors="coerce") / 10,
    "SBS3_PER_10": pd.to_numeric(df[sbs3_col], errors="coerce") / 10,
})

analysis = analysis.replace([np.inf, -np.inf], np.nan).dropna()

analysis = analysis[
    (analysis["OS_MONTHS"] > 0)
    & (analysis["AGE_PER_10"] > 0)
    & (analysis["EVENT"].isin([0, 1]))
].copy()

analysis["EVENT"] = analysis["EVENT"].astype(int)

# Both models must use exactly the same patients.
n = len(analysis)
deaths = int(analysis["EVENT"].sum())

if n < 30 or analysis["EVENT"].nunique() < 2:
    raise ValueError(
        f"Insufficient usable survival data: n={n}, deaths={deaths}"
    )

print("\nCOMMON ANALYSIS COHORT")
print(f"Patients: {n}")
print(f"Deaths: {deaths}")
print(f"Censored: {n - deaths}")
print(f"Median age: {analysis['AGE_PER_10'].median() * 10:.1f} years")
print(f"Median SBS3: {analysis['SBS3_PER_10'].median() * 10:.2f}%")


# ============================================================
# 4. FIT THE TWO COX MODELS
# ============================================================

age_data = analysis[["OS_MONTHS", "EVENT", "AGE_PER_10"]]
full_data = analysis[
    ["OS_MONTHS", "EVENT", "AGE_PER_10", "SBS3_PER_10"]
]

age_model = CoxPHFitter()
age_model.fit(
    age_data,
    duration_col="OS_MONTHS",
    event_col="EVENT"
)

full_model = CoxPHFitter()
full_model.fit(
    full_data,
    duration_col="OS_MONTHS",
    event_col="EVENT"
)

print("\nMODEL 1: AGE ONLY")
print(age_model.summary)

print("\nMODEL 2: AGE + SBS3")
print(full_model.summary)


# ============================================================
# 5. FORMAL LIKELIHOOD-RATIO TEST
# ============================================================

# The models are nested and use the same patients.
lr_stat = 2 * (full_model.log_likelihood_ - age_model.log_likelihood_)
lr_stat = max(0.0, float(lr_stat))
lr_p = chi2.sf(lr_stat, df=1)

sbs3_row = full_model.summary.loc["SBS3_PER_10"]

sbs3_hr = float(sbs3_row["exp(coef)"])
sbs3_ci_low = float(sbs3_row["exp(coef) lower 95%"])
sbs3_ci_high = float(sbs3_row["exp(coef) upper 95%"])
sbs3_p = float(sbs3_row["p"])

# AIC: lower values indicate better relative fit.
age_aic = float(age_model.AIC_partial_)
full_aic = float(full_model.AIC_partial_)

# Positive value means the age + SBS3 model has lower AIC.
aic_improvement = age_aic - full_aic


# ============================================================
# 6. IN-SAMPLE CONCORDANCE (DESCRIPTIVE)
# ============================================================

age_risk = age_model.predict_partial_hazard(age_data).to_numpy().ravel()
full_risk = full_model.predict_partial_hazard(full_data).to_numpy().ravel()

# Negate risk scores because higher concordance scores mean
# longer predicted survival in lifelines.
c_age = concordance_index(
    analysis["OS_MONTHS"], -age_risk, analysis["EVENT"]
)

c_full = concordance_index(
    analysis["OS_MONTHS"], -full_risk, analysis["EVENT"]
)


# ============================================================
# 7. FIVE-FOLD CROSS-VALIDATED C-INDEX
# ============================================================

# Evaluate both models on held-out patients.
# Folds are stratified by death/censoring status.

if analysis["EVENT"].value_counts().min() < 5:
    raise ValueError(
        "Too few patients in one event category for 5-fold CV."
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []

X = analysis.reset_index(drop=True)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, X["EVENT"]), start=1
):
    train = X.iloc[train_idx].copy()
    test = X.iloc[test_idx].copy()

    fold_age = CoxPHFitter()
    fold_age.fit(
        train[["OS_MONTHS", "EVENT", "AGE_PER_10"]],
        duration_col="OS_MONTHS",
        event_col="EVENT"
    )

    fold_full = CoxPHFitter()
    fold_full.fit(
        train[["OS_MONTHS", "EVENT", "AGE_PER_10", "SBS3_PER_10"]],
        duration_col="OS_MONTHS",
        event_col="EVENT"
    )

    test_age_risk = fold_age.predict_partial_hazard(
        test[["AGE_PER_10"]]
    ).to_numpy().ravel()

    test_full_risk = fold_full.predict_partial_hazard(
        test[["AGE_PER_10", "SBS3_PER_10"]]
    ).to_numpy().ravel()

    test_events = test["EVENT"].to_numpy()
    test_times = test["OS_MONTHS"].to_numpy()

    cv_c_age = concordance_index(
        test_times, -test_age_risk, test_events
    )

    cv_c_full = concordance_index(
        test_times, -test_full_risk, test_events
    )

    cv_rows.append({
        "fold": fold,
        "test_patients": len(test),
        "test_deaths": int(test["EVENT"].sum()),
        "age_only_c_index": cv_c_age,
        "age_plus_SBS3_c_index": cv_c_full,
        "change_in_c_index": cv_c_full - cv_c_age
    })

cv_results = pd.DataFrame(cv_rows)
cv_results.to_csv(
    OUTPUT_DIR / "cross_validated_c_index_by_fold.csv",
    index=False
)

mean_cv_age = cv_results["age_only_c_index"].mean()
mean_cv_full = cv_results["age_plus_SBS3_c_index"].mean()
mean_cv_change = cv_results["change_in_c_index"].mean()


# ============================================================
# 8. SAVE FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame([
    {
        "model": "Age only",
        "patients": n,
        "deaths": deaths,
        "partial_AIC": age_aic,
        "in_sample_C_index": c_age,
        "mean_5fold_C_index": mean_cv_age
    },
    {
        "model": "Age + SBS3",
        "patients": n,
        "deaths": deaths,
        "partial_AIC": full_aic,
        "in_sample_C_index": c_full,
        "mean_5fold_C_index": mean_cv_full
    }
])

comparison.to_csv(
    OUTPUT_DIR / "age_vs_age_plus_SBS3.csv",
    index=False
)

details = pd.DataFrame([{
    "patients": n,
    "deaths": deaths,
    "SBS3_HR_per_10_percentage_points": sbs3_hr,
    "SBS3_CI95_lower": sbs3_ci_low,
    "SBS3_CI95_upper": sbs3_ci_high,
    "SBS3_p_value": sbs3_p,
    "likelihood_ratio_statistic": lr_stat,
    "likelihood_ratio_p_value": lr_p,
    "age_only_partial_AIC": age_aic,
    "age_plus_SBS3_partial_AIC": full_aic,
    "AIC_improvement": aic_improvement,
    "age_only_in_sample_C_index": c_age,
    "age_plus_SBS3_in_sample_C_index": c_full,
    "in_sample_C_index_change": c_full - c_age,
    "age_only_mean_5fold_C_index": mean_cv_age,
    "age_plus_SBS3_mean_5fold_C_index": mean_cv_full,
    "mean_5fold_C_index_change": mean_cv_change
}])

details.to_csv(
    OUTPUT_DIR / "model_comparison_statistics.csv",
    index=False
)


# ============================================================
# 9. WRITE A PLAIN-LANGUAGE SUMMARY
# ============================================================

if lr_p < 0.05:
    fit_result = (
        "Adding SBS3 significantly improved model fit by the "
        "likelihood-ratio test (p < 0.05)."
    )
else:
    fit_result = (
        "Adding SBS3 did not significantly improve model fit "
        "by the likelihood-ratio test (p >= 0.05)."
    )

if aic_improvement > 0:
    aic_result = "AIC favoured the age + SBS3 model."
elif aic_improvement < 0:
    aic_result = "AIC favoured the age-only model."
else:
    aic_result = "Both models had equal AIC."

if mean_cv_change > 0:
    cv_result = (
        "The age + SBS3 model had a higher mean 5-fold "
        "cross-validated C-index."
    )
elif mean_cv_change < 0:
    cv_result = (
        "The age-only model had a higher mean 5-fold "
        "cross-validated C-index."
    )
else:
    cv_result = (
        "Both models had the same mean 5-fold "
        "cross-validated C-index."
    )

summary = f"""
SBS3 BEYOND AGE: MODEL COMPARISON

Patients included: {n}
Deaths: {deaths}
Censored: {n - deaths}

SBS3 hazard ratio per 10-percentage-point increase:
HR = {sbs3_hr:.3f}
95% CI = {sbs3_ci_low:.3f} to {sbs3_ci_high:.3f}
P-value = {sbs3_p:.5g}

Likelihood-ratio test:
Statistic = {lr_stat:.3f}
P-value = {lr_p:.5g}
{fit_result}

AIC:
Age only = {age_aic:.3f}
Age + SBS3 = {full_aic:.3f}
AIC improvement (age-only minus age+SBS3) = {aic_improvement:.3f}
{aic_result}

Mean five-fold cross-validated C-index:
Age only = {mean_cv_age:.3f}
Age + SBS3 = {mean_cv_full:.3f}
Change = {mean_cv_change:.3f}
{cv_result}

INTERPRETATION CAUTION:
Cross-validated C-index estimates can vary with the sample and folds.
This analysis evaluates incremental prognostic information within
this TCGA cohort; it does not establish clinical usefulness or
independent external validation. The models use complete cases
with both age and SBS3 available.
"""

(OUTPUT_DIR / "interpretation.txt").write_text(
    summary.strip(), encoding="utf-8"
)

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)
print(comparison.to_string(index=False))
print("\nLikelihood-ratio test p-value:", f"{lr_p:.5g}")
print("AIC improvement:", f"{aic_improvement:.3f}")
print("Mean cross-validated C-index change:", f"{mean_cv_change:.3f}")
print("\nInterpretation:")
print(fit_result)
print(aic_result)
print(cv_result)

print("\nAll results saved to:")
print(OUTPUT_DIR)
print("\nScript completed.")

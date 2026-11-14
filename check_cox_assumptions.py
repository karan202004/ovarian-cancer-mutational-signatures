
from pathlib import Path
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# Project paths
PROJECT = Path(__file__).resolve().parent.parent
INPUT = (
    PROJECT
    / "results"
    / "all_diagnostics"
    / "survival_preparation"
    / "survival_analysis_dataset_preliminary.csv"
)

OUT = PROJECT / "results" / "survival_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(INPUT)

# Use patients with valid survival information and SBS3 estimates
df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
df["OS_EVENT"] = pd.to_numeric(df["OS_EVENT"], errors="coerce")
df["SBS3_MAIN_PERCENT"] = pd.to_numeric(
    df["SBS3_MAIN_PERCENT"], errors="coerce"
)

data = df[
    ["OS_MONTHS", "OS_EVENT", "SBS3_MAIN_PERCENT"]
].dropna().copy()

data = data[
    (data["OS_MONTHS"] > 0)
    & data["OS_EVENT"].isin([0, 1])
].copy()

# Express SBS3 in 10-percentage-point increments
data["SBS3_PER_10"] = data["SBS3_MAIN_PERCENT"] / 10

# Fit Cox model
cph = CoxPHFitter()
cph.fit(
    data[["OS_MONTHS", "OS_EVENT", "SBS3_PER_10"]],
    duration_col="OS_MONTHS",
    event_col="OS_EVENT"
)

print("COX MODEL")
cph.print_summary()

# Test proportional hazards assumption
ph_results = proportional_hazard_test(
    cph,
    data[["OS_MONTHS", "OS_EVENT", "SBS3_PER_10"]],
    time_transform="rank"
)

print("\nPROPORTIONAL HAZARDS TEST")
print(ph_results.summary)

ph_results.summary.to_csv(
    OUT / "sbs3_ph_assumption_test.csv"
)

print("\nSaved results to:", OUT)
print("\nInterpretation:")
print("PH-test p >= 0.05: no statistically significant evidence")
print("of a proportional-hazards violation.")
print("PH-test p < 0.05: potential violation; investigate further.")

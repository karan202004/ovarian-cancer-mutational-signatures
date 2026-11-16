from pathlib import Path
import warnings
import pandas as pd
import numpy as np

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

PROJECT = Path(r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures")
INPUT = (
    PROJECT / "results" / "all_diagnostics"
    / "survival_preparation" / "survival_analysis_dataset_preliminary.csv"
)
BRCA_FILE = (
    PROJECT / "results" / "all_diagnostics"
    / "brca1_brca2_mutation_records.csv"
)
OUT = PROJECT / "results" / "survival_analysis"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)
df["SAMPLE_ID"] = df["SAMPLE_ID"].astype(str).str.strip()

# Verify survival fields
df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
df["OS_EVENT"] = pd.to_numeric(df["OS_EVENT"], errors="coerce")
df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
df["SBS3_MAIN_PERCENT"] = pd.to_numeric(
    df["SBS3_MAIN_PERCENT"], errors="coerce"
)

# Keep records with valid survival data and SBS3 estimate
surv = df[
    df["OS_MONTHS"].notna()
    & (df["OS_MONTHS"] >= 0)
    & df["OS_EVENT"].isin([0, 1])
    & df["SBS3_MAIN_PERCENT"].notna()
].copy()

surv["OS_EVENT"] = surv["OS_EVENT"].astype(int)
surv["SBS3_PER_10"] = surv["SBS3_MAIN_PERCENT"] / 10
surv["AGE_PER_10"] = surv["AGE"] / 10


print("SURVIVAL ANALYSIS")
print(f"Patients included: {len(surv)}")
print(f"Deaths: {(surv['OS_EVENT'] == 1).sum()}")
print(f"Censored: {(surv['OS_EVENT'] == 0).sum()}")
print(f"Median follow-up time in dataset (months): "
      f"{surv['OS_MONTHS'].median():.2f}")
print(f"Median SBS3 contribution (%): "
      f"{surv['SBS3_MAIN_PERCENT'].median():.2f}")


#Kaplan-Meier: Lower VS Higher SBS3
# Exploratory split at the cohort median
cutoff = surv["SBS3_MAIN_PERCENT"].median()
surv["SBS3_GROUP"] = np.where(
    surv["SBS3_MAIN_PERCENT"] >= cutoff,
    "Higher SBS3",
    "Lower SBS3"
)

kmf = KaplanMeierFitter()
km_summary = []

for group in ["Lower SBS3", "Higher SBS3"]:
    sub = surv[surv["SBS3_GROUP"] == group]
    kmf.fit(
        durations=sub["OS_MONTHS"],
        event_observed=sub["OS_EVENT"],
        label=group
    )

    km_summary.append({
        "group": group,
        "n": len(sub),
        "deaths": int(sub["OS_EVENT"].sum()),
        "median_survival_months": kmf.median_survival_time_
    })

km_table = pd.DataFrame(km_summary)
km_table.to_csv(OUT / "kaplan_meier_group_summary.csv", index=False)

lower = surv[surv["SBS3_GROUP"] == "Lower SBS3"]
higher = surv[surv["SBS3_GROUP"] == "Higher SBS3"]

lr = logrank_test(
    lower["OS_MONTHS"],
    higher["OS_MONTHS"],
    event_observed_A=lower["OS_EVENT"],
    event_observed_B=higher["OS_EVENT"]
)

print("\nKAPLAN-MEIER GROUPS")
print(km_table.to_string(index=False))
print(f"Median cutoff: {cutoff:.2f}% SBS3")
print(f"Log-rank p-value: {lr.p_value:.6g}")

# Plot survival curves
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
for group in ["Lower SBS3", "Higher SBS3"]:
    sub = surv[surv["SBS3_GROUP"] == group]
    kmf.fit(sub["OS_MONTHS"], sub["OS_EVENT"], label=group)
    kmf.plot_survival_function(ci_show=True)

plt.title("Overall Survival by SBS3 Contribution")
plt.xlabel("Overall survival (months)")
plt.ylabel("Survival probability")
plt.tight_layout()
plt.savefig(OUT / "kaplan_meier_sbs3.png", dpi=300)
plt.close()


#COX MODEL 1: SBS3 Alone
# HR is per 10-percentage-point increase in SBS3 contribution
cox_rows = []

def fit_cox(data, covariates, model_name):
    model_data = data[
        ["OS_MONTHS", "OS_EVENT"] + covariates
    ].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if len(model_data) == 0:
        print(f"\nSkipping {model_name}: no complete cases.")
        return

    if model_data["OS_EVENT"].sum() == 0:
        print(f"\nSkipping {model_name}: no observed deaths.")
        return

    model = CoxPHFitter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                model_data,
                duration_col="OS_MONTHS",
                event_col="OS_EVENT"
            )

        print(f"\n{model_name}: n={len(model_data)}, "
              f"deaths={int(model_data['OS_EVENT'].sum())}")
        print(model.summary[
            ["coef", "exp(coef)", "se(coef)", "p",
             "exp(coef) lower 95%", "exp(coef) upper 95%"]
        ].round(4).to_string())

        for variable, row in model.summary.iterrows():
            cox_rows.append({
                "model": model_name,
                "variable": variable,
                "n": len(model_data),
                "deaths": int(model_data["OS_EVENT"].sum()),
                "hazard_ratio": row["exp(coef)"],
                "ci_lower_95": row["exp(coef) lower 95%"],
                "ci_upper_95": row["exp(coef) upper 95%"],
                "p_value": row["p"]
            })

        return model

    except Exception as e:
        print(f"\nModel failed ({model_name}): {e}")
        return

model1 = fit_cox(
    surv,
    ["SBS3_PER_10"],
    "Model 1: SBS3 alone"
)


# COX Model 2: SBS3 + Age
# Complete-case subset; report the smaller sample size
age_data = surv[surv["AGE"].notna()].copy()

model2 = fit_cox(
    age_data,
    ["SBS3_PER_10", "AGE_PER_10"],
    "Model 2: SBS3 + age"
)


#Provisional BRAC mutatinal indicators

if BRCA_FILE.exists():
    brca = pd.read_csv(BRCA_FILE, low_memory=False)

    if "Tumor_Sample_Barcode" in brca.columns:
        brca["SAMPLE_ID"] = (
            brca["Tumor_Sample_Barcode"].astype(str).str.strip()
        )

        # Any recorded BRCA1/2 variant, regardless of filter
        any_variant = set(brca["SAMPLE_ID"].dropna())

        # Variants marked PASS only; use as a sensitivity analysis
        if "FILTER" in brca.columns:
            pass_variant = set(
                brca.loc[
                    brca["FILTER"].astype(str).str.upper().eq("PASS"),
                    "SAMPLE_ID"
                ].dropna()
            )
        else:
            pass_variant = set()

        surv["BRCA_any_record"] = surv["SAMPLE_ID"].isin(
            any_variant
        ).astype(int)
        surv["BRCA_PASS_record"] = surv["SAMPLE_ID"].isin(
            pass_variant
        ).astype(int)

        print("\nPROVISIONAL BRCA INDICATORS")
        print("Any-record positive samples:",
              int(surv["BRCA_any_record"].sum()))
        print("PASS-record positive samples:",
              int(surv["BRCA_PASS_record"].sum()))

        # Exploratory only; do not interpret as clinical BRCA status.
        model3_data = surv[
            ["OS_MONTHS", "OS_EVENT", "SBS3_PER_10",
             "BRCA_any_record"]
        ].dropna()

        if model3_data["BRCA_any_record"].nunique() == 2:
            fit_cox(
                surv,
                ["SBS3_PER_10", "BRCA_any_record"],
                "Exploratory: SBS3 + any BRCA record"
            )
        else:
            print("Skipping BRCA model: only one BRCA group is present.")

        # Save the joined exploratory indicators
        surv.to_csv(OUT / "survival_data_with_provisional_BRCA.csv",
                    index=False)

    else:
        print("\nBRCA model skipped: sample barcode column not found.")

#Save
if cox_rows:
    pd.DataFrame(cox_rows).to_csv(
        OUT / "cox_model_results.csv", index=False
    )

surv.to_csv(OUT / "survival_analysis_cohort.csv", index=False)

print("ANALYSIS FINISHED")
print("Output folder:", OUT)
print("Files created:")
for file in sorted(OUT.glob("*")):
    print(" -", file.name)

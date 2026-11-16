from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore", category=RuntimeWarning)


PROJECT_DIR = Path(__file__).resolve().parent.parent

SURVIVAL_FILE = (
    PROJECT_DIR
    / "results"
    / "all_diagnostics"
    / "survival_preparation"
    / "survival_analysis_dataset_preliminary.csv"
)

BRCA_DIR = PROJECT_DIR / "results" / "final_brca_validation"
OUTPUT_DIR = PROJECT_DIR / "results" / "final_project_outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

for folder in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("FINAL OVARIAN CANCER MUTATIONAL-SIGNATURE PROJECT")
print("=" * 65)

if not SURVIVAL_FILE.exists():
    raise FileNotFoundError(f"Survival dataset not found:\n{SURVIVAL_FILE}")

df = pd.read_csv(SURVIVAL_FILE, low_memory=False)
print(f"\nLoaded survival dataset: {df.shape[0]} rows, {df.shape[1]} columns")

#Identify columns

def normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(aliases, required=True):
    normalized = {normalize(c): c for c in df.columns}

    for alias in aliases:
        if normalize(alias) in normalized:
            return normalized[normalize(alias)]

    if required:
        print("\nAvailable dataset columns:")
        print("\n".join(map(str, df.columns)))
        raise ValueError(
            f"Could not identify required column from: {aliases}"
        )

    return None


time_col = find_column([
    "OS_MONTHS", "OS_MONTH", "overall_survival_months",
    "overall_survival_month", "survival_months", "os_time"
])

event_col = find_column([
    "OS_STATUS", "overall_survival_status",
    "death_event", "vital_status", "event"
])

sbs3_col = find_column([
    "SBS3_MAIN_PERCENT",
    "SBS3", "SBS3_percent", "SBS3_percentage",
    "SBS3_proportion", "SBS3_exposure", "SBS3_fraction"
])
age_col = find_column([
    "AGE", "AGE_AT_DIAGNOSIS", "age_at_diagnosis", "age_years"
], required=False)

aneuploidy_col = find_column([
    "aneuploidy_score", "aneuploidy",
    "aneuploidy_score_abs", "fraction_genome_altered",
    "genomic_instability", "genomic_instability_score"
], required=False)

print("\nIdentified columns:")
print(f"Survival time: {time_col}")
print(f"Survival status: {event_col}")
print(f"SBS3: {sbs3_col}")
print(f"Age: {age_col}")
print(f"Genomic instability: {aneuploidy_col}")

#Clean survival data

def parse_event(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return int(value) if value in (0, 1) else np.nan

    value = str(value).strip().upper()

    if value.startswith("1:") or value in {
        "DECEASED", "DEAD", "TRUE", "YES", "EVENT"
    }:
        return 1

    if value.startswith("0:") or value in {
        "LIVING", "ALIVE", "FALSE", "NO", "CENSORED"
    }:
        return 0

    return np.nan


work = pd.DataFrame(index=df.index)
work["time"] = pd.to_numeric(df[time_col], errors="coerce")
work["event"] = df[event_col].apply(parse_event)
work["SBS3_raw"] = pd.to_numeric(df[sbs3_col], errors="coerce")

if age_col:
    work["age"] = pd.to_numeric(df[age_col], errors="coerce")

if aneuploidy_col:
    work["aneuploidy"] = pd.to_numeric(
        df[aneuploidy_col], errors="coerce"
    )

work = work.replace([np.inf, -np.inf], np.nan)
work = work[
    work["time"].notna()
    & (work["time"] > 0)
    & work["event"].isin([0, 1])
].copy()

sbs3_max = work["SBS3_raw"].max()

if pd.notna(sbs3_max) and sbs3_max <= 1:
    work["SBS3"] = work["SBS3_raw"] * 100
else:
    work["SBS3"] = work["SBS3_raw"]

if work["SBS3"].notna().sum() < 20:
    raise ValueError("Too few usable SBS3 values for final analysis.")

print(f"\nUsable survival records: {len(work)}")
print(f"Deaths: {int(work['event'].sum())}")
print(f"Censored: {int((work['event'] == 0).sum())}")


summary_rows = [
    ("Patients with usable survival data", len(work)),
    ("Deaths", int(work["event"].sum())),
    ("Censored observations", int((work["event"] == 0).sum())),
    ("Median survival time (months)", work["time"].median()),
    ("Median SBS3", work["SBS3"].median()),
    ("SBS3 25th percentile", work["SBS3"].quantile(0.25)),
    ("SBS3 75th percentile", work["SBS3"].quantile(0.75)),
]

if "age" in work:
    summary_rows += [
        ("Patients with age data", int(work["age"].notna().sum())),
        ("Median age (years)", work["age"].median()),
    ]

pd.DataFrame(
    summary_rows, columns=["Measure", "Value"]
).to_csv(TABLE_DIR / "table1_cohort_summary.csv", index=False)

#Kaplan-Meier Analysis

km_data = work.dropna(subset=["SBS3"]).copy()
cutoff = km_data["SBS3"].median()

km_data["SBS3_group"] = np.where(
    km_data["SBS3"] > cutoff, "High SBS3", "Low SBS3"
)

low = km_data[km_data["SBS3_group"] == "Low SBS3"]
high = km_data[km_data["SBS3_group"] == "High SBS3"]

km_test = logrank_test(
    low["time"], high["time"],
    event_observed_A=low["event"],
    event_observed_B=high["event"]
)

fig, ax = plt.subplots(figsize=(8, 6))

for label, group in [("Low SBS3", low), ("High SBS3", high)]:
    kmf = KaplanMeierFitter()
    kmf.fit(
        group["time"],
        event_observed=group["event"],
        label=f"{label} (n={len(group)})"
    )
    kmf.plot_survival_function(ax=ax, ci_show=True)

ax.set_title("Overall survival by SBS3 group")
ax.set_xlabel("Time (months)")
ax.set_ylabel("Overall survival probability")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure1_kaplan_meier_sbs3.png", dpi=300)
plt.close(fig)

pd.DataFrame([
    {
        "Group": "Low SBS3",
        "Patients": len(low),
        "Deaths": int(low["event"].sum()),
        "Median_SBS3": low["SBS3"].median(),
        "Median_observed_time_months": low["time"].median(),
    },
    {
        "Group": "High SBS3",
        "Patients": len(high),
        "Deaths": int(high["event"].sum()),
        "Median_SBS3": high["SBS3"].median(),
        "Median_observed_time_months": high["time"].median(),
    },
]).to_csv(TABLE_DIR / "table2_kaplan_meier_groups.csv", index=False)

pd.DataFrame([{
    "SBS3_median_cutoff": cutoff,
    "Log_rank_p_value": km_test.p_value,
}]).to_csv(TABLE_DIR / "table3_logrank_test.csv", index=False)


#COX Models: Age, SBS3, and Genomic Instability

model_results = []


def fit_cox(label, columns, data):
    complete = data[columns + ["time", "event"]].dropna().copy()

    if len(complete) < 30 or complete["event"].sum() < 10:
        print(f"Skipping {label}: insufficient complete cases/events.")
        return None

    cph = CoxPHFitter()
    cph.fit(complete, duration_col="time", event_col="event")

    for variable in columns:
        row = cph.summary.loc[variable]

        model_results.append({
            "Model": label,
            "Variable": variable,
            "N": len(complete),
            "Deaths": int(complete["event"].sum()),
            "HR": np.exp(row["coef"]),
            "CI_lower": np.exp(row["coef lower 95%"]),
            "CI_upper": np.exp(row["coef upper 95%"]),
            "p_value": row["p"],
            "AIC_partial": cph.AIC_partial_,
        })

    return cph, complete


if "age" in work:
    model_data = work[["time", "event", "age", "SBS3"]].copy()
    model_data["Age_per_10_years"] = model_data["age"] / 10
    model_data["SBS3_per_10_percentage_points"] = model_data["SBS3"] / 10

    common = model_data.dropna(
        subset=[
            "time", "event",
            "Age_per_10_years",
            "SBS3_per_10_percentage_points"
        ]
    ).copy()

    age_result = fit_cox(
        "Age only", ["Age_per_10_years"], common
    )

    sbs3_result = fit_cox(
        "Age + SBS3",
        ["Age_per_10_years", "SBS3_per_10_percentage_points"],
        common
    )

    if age_result and sbs3_result:
        from scipy.stats import chi2

        age_model = age_result[0]
        combined_model = sbs3_result[0]

        lrt_stat = max(
            0,
            2 * (combined_model.log_likelihood_ - age_model.log_likelihood_)
        )

        pd.DataFrame([{
            "N_common_complete_cases": len(common),
            "Deaths": int(common["event"].sum()),
            "Age_only_AIC_partial": age_model.AIC_partial_,
            "Age_plus_SBS3_AIC_partial": combined_model.AIC_partial_,
            "AIC_improvement": (
                age_model.AIC_partial_ - combined_model.AIC_partial_
            ),
            "Likelihood_ratio_statistic": lrt_stat,
            "Likelihood_ratio_p_value": chi2.sf(lrt_stat, df=1),
        }]).to_csv(
            TABLE_DIR / "table4_age_vs_age_plus_sbs3.csv", index=False
        )

    if aneuploidy_col:
        genomic = work.copy()
        genomic["Age_per_10_years"] = genomic["age"] / 10
        genomic["SBS3_per_10_percentage_points"] = genomic["SBS3"] / 10

        aneuploidy_sd = genomic["aneuploidy"].std()

        if pd.notna(aneuploidy_sd) and aneuploidy_sd > 0:
            genomic["Aneuploidy_per_SD"] = (
                genomic["aneuploidy"] / aneuploidy_sd
            )

            fit_cox(
                "Age + aneuploidy",
                ["Age_per_10_years", "Aneuploidy_per_SD"],
                genomic
            )

            fit_cox(
                "Age + SBS3 + aneuploidy",
                [
                    "Age_per_10_years",
                    "SBS3_per_10_percentage_points",
                    "Aneuploidy_per_SD"
                ],
                genomic
            )

if model_results:
    model_table = pd.DataFrame(model_results)
    model_table.to_csv(
        TABLE_DIR / "table5_cox_model_results.csv", index=False
    )

    plot_data = model_table[
        model_table["HR"].gt(0)
        & model_table["CI_lower"].gt(0)
        & model_table["CI_upper"].gt(0)
    ].copy()

    if not plot_data.empty:
        labels = plot_data["Model"] + ": " + plot_data["Variable"]
        y = np.arange(len(plot_data))

        fig, ax = plt.subplots(
            figsize=(10, max(5, len(plot_data) * 0.45))
        )

        ax.errorbar(
            plot_data["HR"], y,
            xerr=[
                plot_data["HR"] - plot_data["CI_lower"],
                plot_data["CI_upper"] - plot_data["HR"]
            ],
            fmt="o", capsize=3
        )

        ax.axvline(1, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Hazard ratio (log scale)")
        ax.set_title("Cox proportional-hazards model estimates")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "figure2_cox_forest_plot.png", dpi=300)
        plt.close(fig)


#SBS3 distribution and BRAC table
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(work["SBS3"].dropna(), bins=30, edgecolor="black")
ax.set_xlabel("SBS3 exposure (%)")
ax.set_ylabel("Number of samples")
ax.set_title("Distribution of estimated SBS3 exposure")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure3_sbs3_distribution.png", dpi=300)
plt.close(fig)

brca_file = BRCA_DIR / "brca_gene_summary.csv"
brca_records_file = BRCA_DIR / "brca1_brca2_all_records.csv"

if brca_file.exists():
    pd.read_csv(brca_file).to_csv(
        TABLE_DIR / "table6_brca_gene_summary.csv", index=False
    )

if brca_records_file.exists():
    brca_records = pd.read_csv(brca_records_file)

    if "FILTER" in brca_records.columns:
        brca_records["FILTER"].value_counts(dropna=False).rename_axis(
            "FILTER"
        ).reset_index(name="Records").to_csv(
            TABLE_DIR / "table7_brca_filter_summary.csv", index=False
        )


#Copy existing diagnositic table

diagnostics_dir = PROJECT_DIR / "results" / "all_diagnostics"

for csv_file in diagnostics_dir.rglob("*.csv"):
    name = csv_file.name.lower()

    if any(word in name for word in [
        "signature", "cosine", "reconstruction",
        "sensitivity", "robustness", "aneuploidy"
    ]):
        destination = TABLE_DIR / f"existing_{csv_file.name}"

        if not destination.exists():
            try:
                pd.read_csv(csv_file).to_csv(destination, index=False)
            except Exception:
                pass


print("\n" + "=" * 65)
print("Final figures and tables are generated")
print("=" * 65)
print(f"Figures saved in: {FIG_DIR}")
print(f"Tables saved in:  {TABLE_DIR}")
print("\nNo Markdown report was generated.")
print("Review the output tables and figures before submission.")

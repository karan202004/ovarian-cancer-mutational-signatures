from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
DIAG = RESULTS / "all_diagnostics"
SURV_DIR = RESULTS / "survival_analysis"
OUT = SURV_DIR / "final_robustness"

OUT.mkdir(parents=True, exist_ok=True)

SURV_FILE = (
    DIAG / "survival_preparation"
    / "survival_analysis_dataset_preliminary.csv"
)

COUNTS_FILE = RESULTS / "sbs96_counts_by_sample_cosmic_order.csv"

def find_column(df, candidates):
    """Find a column without depending on capitalization."""
    lookup = {str(c).strip().upper(): c for c in df.columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def clean_sample_id(series):
    return series.astype(str).str.strip().str.upper()


def normalize_percent(series):
    """Convert proportions to percentages when values appear fractional."""
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()

    if len(finite) and finite.max() <= 1.01 and finite.min() >= 0:
        values = values * 100

    return values


def fit_cox(data, predictors, label):
    """Fit Cox regression and return coefficient results."""
    cols = ["OS_MONTHS", "OS_EVENT"] + predictors
    d = data[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    d = d[(d["OS_MONTHS"] > 0) & d["OS_EVENT"].isin([0, 1])]

    if len(d) < 30 or d["OS_EVENT"].sum() < 10:
        print(f"Skipped {label}: insufficient complete observations.")
        return [], d, None

    if any(d[c].nunique() < 2 for c in predictors):
        print(f"Skipped {label}: a predictor has no variation.")
        return [], d, None

    try:
        model = CoxPHFitter()
        model.fit(
            d[cols],
            duration_col="OS_MONTHS",
            event_col="OS_EVENT"
        )

        rows = []
        for predictor in predictors:
            s = model.summary.loc[predictor]
            rows.append({
                "model": label,
                "predictor": predictor,
                "n": len(d),
                "deaths": int(d["OS_EVENT"].sum()),
                "HR": s["exp(coef)"],
                "CI95_lower": s["exp(coef) lower 95%"],
                "CI95_upper": s["exp(coef) upper 95%"],
                "p_value": s["p"],
                "concordance": model.concordance_index_,
            })

        print(
            f"{label}: n={len(d)}, deaths={int(d['OS_EVENT'].sum())}, "
            f"concordance={model.concordance_index_:.3f}"
        )
        return rows, d, model

    except Exception as e:
        print(f"Could not fit {label}: {e}")
        return [], d, None


#Load survival data

print("\n" + "=" * 65)
print("OVARIAN CANCER: FINAL ROBUSTNESS ANALYSIS")
print("=" * 65)

if not SURV_FILE.exists():
    raise FileNotFoundError(f"Survival dataset not found: {SURV_FILE}")

surv = pd.read_csv(SURV_FILE)

id_col = find_column(
    surv, ["SAMPLE_ID", "SAMPLE_ID_CLEAN", "Tumor_Sample_Barcode"]
)
time_col = find_column(surv, ["OS_MONTHS"])
event_col = find_column(surv, ["OS_EVENT"])
sbs3_col = find_column(surv, ["SBS3_MAIN_PERCENT", "SBS3_PERCENT", "SBS3"])
age_col = find_column(surv, ["AGE", "AGE_AT_DIAGNOSIS"])

if not all([id_col, time_col, event_col, sbs3_col]):
    raise ValueError(
        "Could not identify required survival columns. "
        f"Found columns: {surv.columns.tolist()}"
    )

surv["SAMPLE_KEY"] = clean_sample_id(surv[id_col])
surv["OS_MONTHS"] = pd.to_numeric(surv[time_col], errors="coerce")
surv["OS_EVENT"] = pd.to_numeric(surv[event_col], errors="coerce")
surv["SBS3_MAIN_PERCENT"] = normalize_percent(surv[sbs3_col])

if age_col:
    surv["AGE"] = pd.to_numeric(surv[age_col], errors="coerce")
    # TCGA age is normally recorded in years. Convert to 10-year units.
    surv["AGE_PER_10"] = surv["AGE"] / 10

surv = surv.drop_duplicates("SAMPLE_KEY", keep="first")

surv = surv[
    (surv["OS_MONTHS"] > 0)
    & surv["OS_EVENT"].isin([0, 1])
    & surv["SBS3_MAIN_PERCENT"].notna()
].copy()

surv["SBS3_PER_10"] = surv["SBS3_MAIN_PERCENT"] / 10

print(f"\nSurvival records available: {len(surv)}")
print(f"Deaths: {int(surv['OS_EVENT'].sum())}")
print(f"Median SBS3: {surv['SBS3_MAIN_PERCENT'].median():.2f}%")

surv.to_csv(OUT / "analysis_cohort.csv", index=False)

#Main COX model and PH assumption

all_model_rows = []
ph_rows = []

rows, main_data, main_model = fit_cox(
    surv, ["SBS3_PER_10"], "Main panel: SBS3"
)
all_model_rows.extend(rows)

if main_model is not None:
    try:
        ph = proportional_hazard_test(
            main_model,
            main_data[["OS_MONTHS", "OS_EVENT", "SBS3_PER_10"]],
            time_transform="rank"
        )
        for predictor, row in ph.summary.iterrows():
            ph_rows.append({
                "model": "Main panel: SBS3",
                "predictor": predictor,
                "test_statistic": row["test_statistic"],
                "p_value": row["p"],
            })
    except Exception as e:
        print("PH test could not be completed:", e)

if "AGE_PER_10" in surv.columns:
    rows, _, age_model = fit_cox(
        surv.dropna(subset=["AGE_PER_10"]),
        ["SBS3_PER_10", "AGE_PER_10"],
        "Main panel: SBS3 + age"
    )
    all_model_rows.extend(rows)

#Find exisiting alternative signature-panel files

print("\nSearching for existing signature contribution files")

candidate_files = []
for path in DIAG.rglob("*.csv"):
    if path.name in {
        SURV_FILE.name,
        "matched_clinical_data.csv",
        "clinical_matching_audit.csv",
    }:
        continue

    try:
        header = pd.read_csv(path, nrows=2)
    except Exception:
        continue

    sbs3_candidate = find_column(
        header, ["SBS3", "SBS3_PERCENT", "SBS3_CONTRIBUTION",
                 "SBS3_MAIN_PERCENT"]
    )
    sample_candidate = find_column(
        header, ["SAMPLE_ID", "SAMPLE_ID_CLEAN",
                 "Tumor_Sample_Barcode", "SAMPLE"]
    )

    if sbs3_candidate and sample_candidate:
        candidate_files.append(path)
        print("  Candidate:", path.relative_to(PROJECT))

# Include the main panel as a named comparison.
panel_data = {
    "main_13_signature": surv[
        ["SAMPLE_KEY", "OS_MONTHS", "OS_EVENT", "SBS3_PER_10"]
    ].copy()
}

for path in candidate_files:
    try:
        df = pd.read_csv(path)
        sid = find_column(
            df, ["SAMPLE_ID", "SAMPLE_ID_CLEAN",
                 "Tumor_Sample_Barcode", "SAMPLE"]
        )
        sig = find_column(
            df, ["SBS3", "SBS3_PERCENT", "SBS3_CONTRIBUTION",
                 "SBS3_MAIN_PERCENT"]
        )

        if not sid or not sig:
            continue

        temp = pd.DataFrame({
            "SAMPLE_KEY": clean_sample_id(df[sid]),
            "SBS3_VALUE": normalize_percent(df[sig]),
        }).dropna()

        temp = temp.drop_duplicates("SAMPLE_KEY")

        # Match only patients with usable survival data.
        temp = temp.merge(
            surv[["SAMPLE_KEY", "OS_MONTHS", "OS_EVENT"]],
            on="SAMPLE_KEY",
            how="inner"
        )

        if len(temp) < 30 or temp["SBS3_VALUE"].nunique() < 2:
            continue

        # Avoid accidentally including non-panel diagnostic files.
        label = path.stem.lower()
        if "core5" in label or "core_5" in label:
            panel_name = "core_5_signature"
        elif "expanded8" in label or "expanded_8" in label:
            panel_name = "expanded_8_signature"
        elif "exploratory" in label or "13" in label:
            panel_name = "main_13_signature_file"
        else:
            panel_name = path.stem

        panel_data[panel_name] = temp.rename(
            columns={"SBS3_VALUE": "SBS3_PERCENT"}
        )
        print(f"  Loaded {panel_name}: {len(temp)} patients")

    except Exception as e:
        print(f"  Skipped {path.name}: {e}")

# Fit each distinct panel, avoiding duplicate main-panel fits.
panel_rows = []
for name, d in panel_data.items():
    if name == "main_13_signature":
        continue

    if "SBS3_PERCENT" not in d.columns:
        continue

    temp = d.copy()
    temp["SBS3_PER_10"] = temp["SBS3_PERCENT"] / 10
    rows, _, _ = fit_cox(
        temp, ["SBS3_PER_10"], f"Panel: {name}"
    )
    panel_rows.extend(rows)

#Mutational burden sensitivity

print("\nChecking mutation-burden sensitivity...")

burden_file = COUNTS_FILE
if not burden_file.exists():
    burden_file = RESULTS / "sbs96_counts_by_sample.csv"

if burden_file.exists():
    counts = pd.read_csv(burden_file)

    count_id = find_column(
        counts, ["SAMPLE_ID", "SAMPLE_ID_CLEAN",
                 "Tumor_Sample_Barcode", "SAMPLE"]
    )

    if count_id:
        counts["SAMPLE_KEY"] = clean_sample_id(counts[count_id])
        channel_cols = [
            c for c in counts.columns
            if c != count_id
            and c != "SAMPLE_KEY"
            and str(c).strip().upper() not in {
                "SAMPLE_ID", "SAMPLE_ID_CLEAN", "TUMOR_SAMPLE_BARCODE"
            }
        ]

        numeric = counts[channel_cols].apply(pd.to_numeric, errors="coerce")
        counts["SBS_TOTAL"] = numeric.sum(axis=1, min_count=1)

        burden = counts[["SAMPLE_KEY", "SBS_TOTAL"]].drop_duplicates(
            "SAMPLE_KEY"
        )
        burden_surv = surv.merge(burden, on="SAMPLE_KEY", how="inner")

        burden_rows = []
        for threshold in [0, 100, 200]:
            subset = burden_surv[
                burden_surv["SBS_TOTAL"] >= threshold
            ].copy()

            label = f"Minimum SBS count {threshold}"
            rows, _, _ = fit_cox(
                subset, ["SBS3_PER_10"], label
            )
            burden_rows.extend(rows)

        pd.DataFrame(burden_rows).to_csv(
            OUT / "mutation_burden_sensitivity.csv", index=False
        )
        print(f"Patients matched to mutation counts: {len(burden_surv)}")
    else:
        print("Could not identify sample ID in mutation-count file.")
else:
    print("Mutation-count matrix not found; burden sensitivity skipped.")

#Save

pd.DataFrame(all_model_rows).to_csv(
    OUT / "main_cox_results.csv", index=False
)

pd.DataFrame(panel_rows).to_csv(
    OUT / "panel_robustness_results.csv", index=False
)

pd.DataFrame(ph_rows).to_csv(
    OUT / "proportional_hazards_test.csv", index=False
)


print("Output folder:", OUT)
print("Files created when data are available:")
print(" - analysis_cohort.csv")
print(" - main_cox_results.csv")
print(" - panel_robustness_results.csv")
print(" - mutation_burden_sensitivity.csv (if counts file found)")
print(" - proportional_hazards_test.csv")

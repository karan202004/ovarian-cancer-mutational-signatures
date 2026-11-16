from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

PROJECT = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

RESULTS = PROJECT / "results"
RAW = PROJECT / "data" / "raw" / "ov_tcga_pan_can_atlas_2018"
if not (RAW / "ov_tcga_pan_can_atlas_2018").exists():
    RAW = RAW / "ov_tcga_pan_can_atlas_2018"

COSMIC_FILE = (
    PROJECT / "data" / "Reference"
    / "COSMIC_catalogue-signatures_SBS96_v3.6"
    / "COSMIC_v3.6_SBS_GRCh37.txt"
)
MATRIX_FILE = RESULTS / "sbs96_counts_by_sample_cosmic_order.csv"
FIT_FILE = RESULTS / "signature_fit_quality_exploratory.csv"
CONTRIB_FILE = RESULTS / "signature_contributions_exploratory.csv"

OUT = RESULTS / "all_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

counts = pd.read_csv(MATRIX_FILE, index_col=0)
cosmic = pd.read_csv(COSMIC_FILE, sep="\t", index_col=0)

counts.index = counts.index.astype(str)
cosmic.index = cosmic.index.astype(str)

if not counts.columns.equals(cosmic.index):
    raise ValueError("SBS96 channel names/order do not match.")

counts = counts.apply(pd.to_numeric, errors="raise")
cosmic = cosmic.apply(pd.to_numeric, errors="raise")

print("=" * 65)
print("OVARIAN CANCER SIGNATURE PROJECT: FULL DIAGNOSTICS")
print("=" * 65)
print(f"Samples: {len(counts)}")
print(f"SBS96 channels: {counts.shape[1]}")
print(f"Total mutations: {int(counts.to_numpy().sum()):,}")
print(f"COSMIC signatures available: {cosmic.shape[1]}")

# Mutational Burden
burden = counts.sum(axis=1).rename("SBS_mutations")
burden_summary = burden.describe(percentiles=[.10, .25, .50, .75, .90])
burden_summary.to_csv(OUT / "mutation_burden_summary.csv")

print("\n MUTATION BURDEN")
print(burden_summary.round(2))
print("Samples with <50 SBS:", int((burden < 50).sum()))
print("Samples with <100 SBS:", int((burden < 100).sum()))
print("Samples with <200 SBS:", int((burden < 200).sum()))

if FIT_FILE.exists():
    fit = pd.read_csv(FIT_FILE, index_col=0)
    fit.index = fit.index.astype(str)
    fit.to_csv(OUT / "existing_fit_quality.csv")

    print("\n[2] EXISTING FIT QUALITY")
    print(fit[[
        "cosine_fit", "relative_reconstruction_error",
        "SBS3_percent_of_attributed"
    ]].describe().round(3))

    # Descriptive groups only; not validated pass/fail cutoffs
    print("\nCosine fit thresholds:")
    for threshold in [0.70, 0.80, 0.90]:
        print(f">={threshold}: {(fit['cosine_fit'] >= threshold).sum()}")
else:
    fit = None
    print("\nExisting fit-quality file not found.")


#Re fit with alternative signature panel
panels = {
    "current13": [
        "SBS1", "SBS2", "SBS3", "SBS5", "SBS8",
        "SBS13", "SBS17a", "SBS17b", "SBS18",
        "SBS35", "SBS40a", "SBS40b", "SBS40c"
    ],
    "core5": ["SBS1", "SBS3", "SBS5", "SBS8", "SBS40c"],
    "expanded8": [
        "SBS1", "SBS2", "SBS3", "SBS5",
        "SBS8", "SBS18", "SBS35", "SBS40c"
    ]
}

panel_rows = []
panel_contributions = {}

for panel_name, signatures in panels.items():
    missing = [s for s in signatures if s not in cosmic.columns]
    if missing:
        print(f"Skipping {panel_name}; missing signatures: {missing}")
        continue

    # 96 channels x selected signatures
    A = cosmic[signatures].to_numpy(dtype=float)
    rows = []

    for sample, row in counts.iterrows():
        y = row.to_numpy(dtype=float)
        coefficients, residual = nnls(A, y)
        reconstructed = A @ coefficients

        norm_y = np.linalg.norm(y)
        norm_reconstructed = np.linalg.norm(reconstructed)
        cosine = (
            np.dot(y, reconstructed) / (norm_y * norm_reconstructed)
            if norm_y > 0 and norm_reconstructed > 0 else np.nan
        )
        relative_error = (
            np.linalg.norm(y - reconstructed) / norm_y
            if norm_y > 0 else np.nan
        )

        total_coeff = coefficients.sum()
        sbs3_index = signatures.index("SBS3")
        sbs3_percent = (
            100 * coefficients[sbs3_index] / total_coeff
            if total_coeff > 0 else np.nan
        )

        rows.append({
            "sample_id": sample,
            "panel": panel_name,
            "SBS_mutations": int(y.sum()),
            "SBS3_percent": sbs3_percent,
            "cosine_fit": cosine,
            "relative_error": relative_error,
            "residual_norm": residual
        })

    result = pd.DataFrame(rows).set_index("sample_id")
    result.to_csv(OUT / f"fit_{panel_name}.csv")
    panel_contributions[panel_name] = result["SBS3_percent"]
    panel_rows.append(result.reset_index())

panel_results = pd.concat(panel_rows, ignore_index=True)
panel_results.to_csv(OUT / "all_panel_fit_results.csv", index=False)

print("\n[3] ALTERNATIVE PANEL COMPARISON")
print(
    panel_results.groupby("panel").agg(
        samples=("sample_id", "count"),
        median_cosine=("cosine_fit", "median"),
        median_error=("relative_error", "median"),
        median_SBS3=("SBS3_percent", "median")
    ).round(3)
)

# Compare estimates across panels for each sample
if len(panel_contributions) >= 2:
    comparison = pd.DataFrame(panel_contributions)
    comparison.to_csv(OUT / "SBS3_panel_stability_by_sample.csv")

    print("\nSBS3 panel stability (sample-level correlation):")
    print(comparison.corr().round(3))

    comparison["panel_range_percentage_points"] = (
        comparison.max(axis=1) - comparison.min(axis=1)
    )
    print("\nRange of SBS3 estimates across panels:")
    print(comparison["panel_range_percentage_points"].describe().round(2))


#Fit quality by mutation burden fit
panel_results["burden_group"] = pd.cut(
    panel_results["SBS_mutations"],
    bins=[0, 50, 100, 200, 500, float("inf")],
    labels=["1-49", "50-99", "100-199", "200-499", "500+"],
    right=False
)

by_burden = panel_results.groupby(
    ["panel", "burden_group"], observed=False
).agg(
    samples=("sample_id", "count"),
    median_cosine=("cosine_fit", "median"),
    median_error=("relative_error", "median"),
    median_SBS3=("SBS3_percent", "median")
).reset_index()

by_burden.to_csv(OUT / "fit_quality_by_burden.csv", index=False)

print("\n[4] FIT QUALITY BY MUTATION BURDEN")
print(by_burden.to_string(index=False))

#Check clinical files and survival data check
patient_file = RAW / "data_clinical_patient.txt"
sample_file = RAW / "data_clinical_sample.txt"

def read_clinical(path):
    if not path.exists():
        print("Not found:", path)
        return None
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)

patient = read_clinical(patient_file)
sample = read_clinical(sample_file)

clinical_summary = []

for label, df in [("patient", patient), ("sample", sample)]:
    if df is None:
        continue

    print(f"\n[5] {label.upper()} CLINICAL DATA")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    for col in df.columns:
        name = col.upper()
        if any(term in name for term in [
            "OS_STATUS", "OS_MONTHS", "AGE", "STAGE",
            "BRCA", "PLATIN", "RESPONSE", "HRD",
            "FGA", "FRACTION"
        ]):
            print(
                f"{col}: missing={df[col].isna().sum()}/{len(df)}, "
                f"unique={df[col].nunique(dropna=True)}"
            )
            clinical_summary.append({
                "file": label,
                "column": col,
                "rows": len(df),
                "missing": int(df[col].isna().sum()),
                "unique_nonmissing": int(df[col].nunique(dropna=True))
            })

if clinical_summary:
    pd.DataFrame(clinical_summary).to_csv(
        OUT / "clinical_variable_availability.csv", index=False
    )


#Sample matching 
if patient is not None:
    patient_id_col = next(
        (c for c in patient.columns
         if c.upper() in ["PATIENT_ID", "PATIENT", "CASE_ID"]),
        None
    )

    if patient_id_col:
        mutation_patients = set(
            counts.index.str.slice(0, 12)
        )
        clinical_patients = set(
            patient[patient_id_col].dropna().astype(str)
        )
        print("\n[6] CLINICAL MATCHING")
        print("Mutation patients:", len(mutation_patients))
        print("Clinical patients:", len(clinical_patients))
        print(
            "Matched patient IDs:",
            len(mutation_patients & clinical_patients)
        )
    else:
        print("\nCould not automatically identify patient ID column.")

summary = {
    "samples": len(counts),
    "SBS96_channels": counts.shape[1],
    "total_SBS_mutations": int(counts.to_numpy().sum()),
    "samples_under_50_SBS": int((burden < 50).sum()),
    "samples_under_100_SBS": int((burden < 100).sum()),
    "samples_under_200_SBS": int((burden < 200).sum()),
    "COSMIC_signatures": cosmic.shape[1]
}

pd.Series(summary, name="value").to_csv(
    OUT / "diagnostic_summary.csv"
)

print("ALL DIAGNOSTICS FINISHED")
print("Results folder:", OUT)

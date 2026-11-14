
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as p

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

RESULTS_DIR = PROJECT_DIR / "results"

MATRIX_FILE = RESULTS_DIR / "sbs96_counts_by_sample.csv"

OUTPUT_COUNTS = RESULTS_DIR / "sbs_feasibility_by_sample.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "sbs_feasibility_summary.csv"
OUTPUT_HISTOGRAM = RESULTS_DIR / "sbs_burden_distribution.png"


# --------------------------------------------------
# 2. Load the SBS96 matrix
# --------------------------------------------------

print("Loading SBS96 matrix...")

matrix = pd.read_csv(
    MATRIX_FILE,
    index_col=0
)

print(f"Number of samples: {len(matrix)}")
print(f"Number of mutation categories: {matrix.shape[1]}")

if matrix.shape[1] != 96:
    raise ValueError(
        f"Expected 96 SBS categories, found {matrix.shape[1]}"
    )

if matrix.isna().any().any():
    raise ValueError("Matrix contains missing values.")

if (matrix < 0).any().any():
    raise ValueError("Matrix contains negative counts.")


# --------------------------------------------------
# 3. Calculate usable SBS count per sample
# --------------------------------------------------

counts = matrix.sum(axis=1).astype(int)

sample_results = pd.DataFrame({
    "Tumor_Sample_Barcode": matrix.index,
    "Usable_SBS_Count": counts.values
})

# These are descriptive burden groups, not validated
# minimum thresholds for SBS3 estimation.
sample_results["Burden_Group"] = pd.cut(
    sample_results["Usable_SBS_Count"],
    bins=[-1, 49, 99, 199, 499, float("inf")],
    labels=[
        "0-49",
        "50-99",
        "100-199",
        "200-499",
        "500+"
    ]
)

sample_results.to_csv(OUTPUT_COUNTS, index=False)


# --------------------------------------------------
# 4. Summarise the mutation burden distribution
# --------------------------------------------------

summary_stats = counts.describe(
    percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
)

summary = pd.DataFrame({
    "Metric": [
        "Number of tumour samples",
        "Total usable SBS mutations",
        "Mean SBS count",
        "Standard deviation",
        "Minimum SBS count",
        "10th percentile",
        "25th percentile",
        "Median SBS count",
        "75th percentile",
        "90th percentile",
        "Maximum SBS count",
        "Samples with fewer than 50 SBS",
        "Samples with fewer than 100 SBS",
        "Samples with fewer than 200 SBS",
        "Samples with at least 200 SBS",
        "Samples with at least 500 SBS"
    ],
    "Value": [
        len(counts),
        int(counts.sum()),
        round(counts.mean(), 2),
        round(counts.std(), 2),
        int(counts.min()),
        round(counts.quantile(0.10), 2),
        round(counts.quantile(0.25), 2),
        round(counts.median(), 2),
        round(counts.quantile(0.75), 2),
        round(counts.quantile(0.90), 2),
        int(counts.max()),
        int((counts < 50).sum()),
        int((counts < 100).sum()),
        int((counts < 200).sum()),
        int((counts >= 200).sum()),
        int((counts >= 500).sum())
    ]
})

summary.to_csv(OUTPUT_SUMMARY, index=False)


# --------------------------------------------------
# 5. Count samples in each descriptive burden group
# --------------------------------------------------

group_counts = (
    sample_results["Burden_Group"]
    .value_counts(sort=False)
    .rename_axis("Burden_Group")
    .reset_index(name="Number_of_Samples")
)

print("\nMutation burden groups:")
print(group_counts.to_string(index=False))


# --------------------------------------------------
# 6. Plot mutation burden distribution
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    counts,
    bins=30,
    edgecolor="black"
)

plt.xlabel("Usable SBS mutations per tumour")
plt.ylabel("Number of tumours")
plt.title("SBS Mutation Burden Across Ovarian Tumours")
plt.tight_layout()

plt.savefig(
    OUTPUT_HISTOGRAM,
    dpi=300
)

plt.show()


# --------------------------------------------------
# 7. Print results
# --------------------------------------------------

print("\n========== SBS FEASIBILITY SUMMARY ==========")
print(summary.to_string(index=False))

print("\nFiles saved:")
print(OUTPUT_COUNTS)
print(OUTPUT_SUMMARY)
print(OUTPUT_HISTOGRAM)

print(
    "\nNote: These burden groups are descriptive only. "
    "They are not validated thresholds for SBS3 estimation."
)

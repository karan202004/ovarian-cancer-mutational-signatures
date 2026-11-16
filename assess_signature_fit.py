from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

FILE = (
    PROJECT_DIR
    / "results"
    / "signature_fit_quality_exploratory.csv"
)

df = pd.read_csv(FILE, index_col=0)

print("Total samples:", len(df))

# Summarize fit quality
print("\nCosine fit distribution:")
print(df["cosine_fit"].describe().round(3))

print("\nRelative reconstruction error:")
print(
    df["relative_reconstruction_error"]
    .describe()
    .round(3)
)

# Descriptive quality groups, not validated scientific cutoffs
for threshold in [0.70, 0.80, 0.90]:
    n = (df["cosine_fit"] >= threshold).sum()
    print(
        f"Samples with cosine fit >= {threshold:.2f}: "
        f"{n} ({100 * n / len(df):.1f}%)"
    )

# Examine fit quality by mutation burden
bins = [0, 50, 100, 200, 500, float("inf")]
labels = ["1-49", "50-99", "100-199", "200-499", "500+"]

df["burden_group"] = pd.cut(
    df["total_SBS_mutations"],
    bins=bins,
    labels=labels,
    right=False
)

print("\nFit quality by mutation burden:")
print(
    df.groupby("burden_group", observed=False)
    .agg(
        samples=("cosine_fit", "size"),
        median_cosine=("cosine_fit", "median"),
        median_error=("relative_reconstruction_error", "median"),
        median_SBS3=("SBS3_percent_of_attributed", "median")
    )
    .round(3)
)

print("\nLowest-quality 10 fits:")
print(
    df.sort_values("cosine_fit")[
        [
            "total_SBS_mutations",
            "cosine_fit",
            "relative_reconstruction_error",
            "SBS3_percent_of_attributed"
        ]
    ].head(10).round(3)
)

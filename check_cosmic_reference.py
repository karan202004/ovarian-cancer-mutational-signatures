from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

REFERENCE_FILE = (
    PROJECT_DIR
    / "data"
    / "Reference"
    / "COSMIC_catalogue-signatures_SBS96_v3.6"
    / "COSMIC_v3.6_SBS_GRCh37.txt"
)

OUTPUT_FILE = PROJECT_DIR / "results" / "cosmic_reference_summary.csv"

print("Loading COSMIC SBS96 reference...")

if not REFERENCE_FILE.exists():
    raise FileNotFoundError(f"File not found: {REFERENCE_FILE}")

# COSMIC files are tab-separated
reference = pd.read_csv(REFERENCE_FILE, sep="\t", index_col=0)

print("Reference shape:", reference.shape)
print("\nFirst 10 channel names:")
print(reference.index[:10].tolist())

print("\nFirst 15 signature names:")
print(reference.columns[:15].tolist())

# Check for SBS3
sbs3_columns = [col for col in reference.columns if col == "SBS3"]

print("\nSBS3 present:", len(sbs3_columns) == 1)

# Check channel count and values
print("Number of mutation channels:", len(reference.index))
print("Missing values:", int(reference.isna().sum().sum()))
print("Negative values:", int((reference < 0).sum().sum()))

if "SBS3" in reference.columns:
    print("SBS3 total:", reference["SBS3"].sum())

#COSMIC signature probabilities should sum approximately to 1
column_sums = reference.sum(axis=0)

print("\nSignature column-sum summary:")
print(column_sums.describe())

#Save
summary = pd.DataFrame({
    "Signature": reference.columns,
    "Column_Sum": column_sums.values,
    "Contains_Missing_Values": [
        reference[col].isna().any() for col in reference.columns
    ]
})

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_FILE, index=False)

print("\nSaved summary to:")
print(OUTPUT_FILE)

print("\nFirst 5 rows of reference:")
print(reference.iloc[:5, :5])

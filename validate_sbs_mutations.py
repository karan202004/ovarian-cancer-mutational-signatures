from pathlib import Path
import pandas as pd


DATA_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
    r"\data\raw\ov_tcga_pan_can_atlas_2018\ov_tcga_pan_can_atlas_2018"
)

OUTPUT_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
    r"\results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mutation_file = DATA_DIR / "data_mutations.txt"

#Load mutation data

mutations = pd.read_csv(
    mutation_file,
    sep="\t",
    comment="#",
    low_memory=False,
    na_values=["NA", "N/A", ""]
)

print("=" * 65)
print("SBS MUTATION VALIDATION")
print("=" * 65)

print("\nOriginal mutation records:", len(mutations))

# Keep single-nucleotide variants

sbs = mutations[
    mutations["Variant_Type"].astype(str).str.upper() == "SNP"
].copy()

print("SNP records:", len(sbs))

# --------------------------------------------------
# 4. Check required fields
# --------------------------------------------------

required_columns = [
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
    "NCBI_Build"
]

print("\nRequired-column check:")

for column in required_columns:
    print(f"{column}: {'FOUND' if column in sbs.columns else 'MISSING'}")

#Check missing values

print("\nMissing values in required fields:")

for column in required_columns:
    if column in sbs.columns:
        missing = sbs[column].isna().sum()
        print(f"{column}: {missing}")

# Check nucleotide alleles

valid_bases = {"A", "C", "G", "T"}

ref = sbs["Reference_Allele"].astype("string").str.upper()
alt = sbs["Tumor_Seq_Allele2"].astype("string").str.upper()

valid_alleles = (
    ref.isin(valid_bases)
    & alt.isin(valid_bases)
    & ref.ne(alt)
)

print("\nAllele validation:")
print("Valid single-base allele records:", int(valid_alleles.sum()))
print("Invalid or ambiguous allele records:", int((~valid_alleles).sum()))


# Check chromosome and coordinate values

sbs["Start_Position"] = pd.to_numeric(
    sbs["Start_Position"],
    errors="coerce"
)

valid_coordinates = (
    sbs["Chromosome"].notna()
    & sbs["Start_Position"].notna()
    & (sbs["Start_Position"] > 0)
)

print("\nCoordinate validation:")
print("Records with valid chromosome and position:",
      int(valid_coordinates.sum()))

# 8. Check genome build

print("\nGenome build:")
print(sbs["NCBI_Build"].value_counts(dropna=False))

#Check duplicate mutation records

duplicate_columns = [
    "Tumor_Sample_Barcode",
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2"
]

duplicates = sbs.duplicated(
    subset=duplicate_columns,
    keep=False
)

print("\nDuplicate-coordinate check:")
print("Records involved in potential duplicates:",
      int(duplicates.sum()))

#Summarise SNP counts by sample

counts = (
    sbs.groupby("Tumor_Sample_Barcode")
    .size()
    .rename("SNP_records")
    .reset_index()
)

print("\nSample count summary:")
print(counts["SNP_records"].describe())


#save
counts.to_csv(
    OUTPUT_DIR / "snp_counts_by_sample.csv",
    index=False
)

print("\nSaved sample counts to:")
print(OUTPUT_DIR / "snp_counts_by_sample.csv")

print("\nValidation complete.")

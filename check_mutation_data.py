from pathlib import Path
import pandas as pd

# Folder containing the extracted cBioPortal study
DATA_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
    r"\data\raw\ov_tcga_pan_can_atlas_2018\ov_tcga_pan_can_atlas_2018"
)

mutation_file = DATA_DIR / "data_mutations.txt"
patient_file = DATA_DIR / "data_clinical_patient.txt"
sample_file = DATA_DIR / "data_clinical_sample.txt"

# --------------------------------------------------
# 1. Load the files
# --------------------------------------------------

mutations = pd.read_csv(
    mutation_file,
    sep="\t",
    comment="#",
    low_memory=False,
    na_values=["NA", "N/A", ""]
)

patients = pd.read_csv(
    patient_file,
    sep="\t",
    comment="#",
    low_memory=False,
    na_values=["NA", "N/A", ""]
)

samples = pd.read_csv(
    sample_file,
    sep="\t",
    comment="#",
    low_memory=False,
    na_values=["NA", "N/A", ""]
)

print("=" * 65)
print("MUTATION DATA QUALITY CHECK")
print("=" * 65)

# --------------------------------------------------
# 2. Check genome build
# --------------------------------------------------

print("\n1. GENOME BUILD")
print(mutations["NCBI_Build"].value_counts(dropna=False))

# --------------------------------------------------
# 3. Check mutation types
# --------------------------------------------------

print("\n2. VARIANT TYPES")
print(mutations["Variant_Type"].value_counts(dropna=False))

print("\n3. VARIANT CLASSIFICATIONS")
print(mutations["Variant_Classification"].value_counts(dropna=False))

# --------------------------------------------------
# 4. Check chromosome names
# --------------------------------------------------

print("\n4. CHROMOSOMES")
print(mutations["Chromosome"].value_counts(dropna=False).head(30))

# --------------------------------------------------
# 5. Check mutation sample IDs
# --------------------------------------------------

mutations["Patient_ID_from_mutation"] = (
    mutations["Tumor_Sample_Barcode"]
    .astype(str)
    .str.slice(0, 12)
)

mutation_samples = mutations["Tumor_Sample_Barcode"].nunique()
mutation_patients = mutations["Patient_ID_from_mutation"].nunique()

print("\n5. SAMPLE COUNTS")
print("Unique tumour sample barcodes:", mutation_samples)
print("Unique patient IDs represented in mutations:", mutation_patients)

# --------------------------------------------------
# 6. Compare mutation and clinical sample IDs
# --------------------------------------------------

clinical_sample_ids = set(samples["SAMPLE_ID"].dropna().astype(str))
mutation_sample_ids = set(
    mutations["Tumor_Sample_Barcode"].dropna().astype(str)
)

matched_samples = mutation_sample_ids & clinical_sample_ids

print("\n6. SAMPLE MATCHING")
print("Samples in mutation file:", len(mutation_sample_ids))
print("Samples in clinical sample file:", len(clinical_sample_ids))
print("Exact sample ID matches:", len(matched_samples))

# --------------------------------------------------
# 7. Count SBS records per sample
# --------------------------------------------------

sbs = mutations[
    mutations["Variant_Type"].astype(str).str.upper().eq("SNP")
].copy()

sbs_counts = (
    sbs.groupby("Tumor_Sample_Barcode")
    .size()
    .sort_values(ascending=False)
)

print("\n7. SINGLE-NUCLEOTIDE VARIANTS")
print("Total SNP records:", len(sbs))
print("Samples with at least one SNP:", len(sbs_counts))

if len(sbs_counts) > 0:
    print("\nSNP count distribution:")
    print(sbs_counts.describe(percentiles=[0.25, 0.50, 0.75, 0.90]))

# --------------------------------------------------
# 8. Show the number of samples in mutation-count bands
# --------------------------------------------------

print("\n8. SAMPLES BY SNP COUNT")

if len(sbs_counts) > 0:
    bands = pd.cut(
        sbs_counts,
        bins=[-1, 49, 99, 199, 499, 999, float("inf")],
        labels=[
            "0-49",
            "50-99",
            "100-199",
            "200-499",
            "500-999",
            "1000+"
        ]
    )

    print(bands.value_counts(sort=False))

print("\n" + "=" * 65)
print("CHECK COMPLETE")
print("=" * 65)
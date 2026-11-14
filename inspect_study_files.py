from pathlib import Path
import pandas as pd

# ============================================================
# 1. SET THE DATA DIRECTORY
# ============================================================

DATA_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
    r"\data\raw\ov_tcga_pan_can_atlas_2018"
    r"\ov_tcga_pan_can_atlas_2018"
)

# ============================================================
# 2. DEFINE THE FIVE FILES
# ============================================================

FILES = {
    "Mutations": DATA_DIR / "data_mutations.txt",
    "Mutation metadata": DATA_DIR / "meta_mutations.txt",
    "Study README": DATA_DIR / "README.md",
    "Patient clinical data": DATA_DIR / "data_clinical_patient.txt",
    "Sample clinical data": DATA_DIR / "data_clinical_sample.txt",
}

# ============================================================
# 3. HELPER FUNCTION TO READ TABULAR FILES
# ============================================================

def read_tsv(file_path):
    """Read a tab-separated file, ignoring comment lines."""

    return pd.read_csv(
        file_path,
        sep="\t",
        comment="#",
        low_memory=False,
        na_values=["NA", "N/A", "NaN", ""]
    )


# ============================================================
# 4. INSPECT FILE EXISTENCE AND SIZE
# ============================================================

print("=" * 85)
print("TCGA-OV PAN-CANCER ATLAS: FILE INSPECTION")
print("=" * 85)

print(f"\nData directory:\n{DATA_DIR}\n")

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data directory not found:\n{DATA_DIR}"
    )

for name, file_path in FILES.items():
    print("-" * 85)

    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)

        print(f"File: {name}")
        print(f"Path: {file_path}")
        print(f"Size: {size_mb:.2f} MB")
        print("Status: FOUND")
    else:
        print(f"File: {name}")
        print("Status: NOT FOUND")


# ============================================================
# 5. INSPECT MUTATION DATA
# ============================================================

print("\n\n" + "=" * 85)
print("1. SOMATIC MUTATION DATA")
print("=" * 85)

mutation_file = FILES["Mutations"]

if mutation_file.exists():

    mutations = read_tsv(mutation_file)

    print(f"\nNumber of mutation records: {len(mutations):,}")
    print(f"Number of columns: {len(mutations.columns)}")

    print("\nCOLUMN NAMES:")
    for column in mutations.columns:
        print(f"  - {column}")

    print("\nFIRST FIVE RECORDS:")
    print(mutations.head().to_string(index=False))

    # Fields needed to reconstruct mutation context
    required_fields = [
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Tumor_Sample_Barcode",
    ]

    print("\nREQUIRED FIELD CHECK:")

    for field in required_fields:
        if field in mutations.columns:
            print(f"  [FOUND]   {field}")
        else:
            print(f"  [MISSING] {field}")

    # Look for equivalent fields if standard names are absent
    print("\nPOSSIBLE COORDINATE, ALLELE AND SAMPLE FIELDS:")

    keywords = [
        "chromosome",
        "position",
        "start",
        "reference",
        "alternate",
        "tumor_seq",
        "sample",
        "allele",
    ]

    for column in mutations.columns:
        if any(word in column.lower() for word in keywords):
            print(f"  - {column}")

    # Number of samples represented in the mutation file
    sample_column = "Tumor_Sample_Barcode"

    if sample_column in mutations.columns:
        print(
            "\nUnique tumour samples represented:",
            mutations[sample_column].nunique()
        )


# ============================================================
# 6. INSPECT MUTATION METADATA
# ============================================================

print("\n\n" + "=" * 85)
print("2. MUTATION METADATA")
print("=" * 85)

metadata_file = FILES["Mutation metadata"]

if metadata_file.exists():

    print(metadata_file.read_text(encoding="utf-8", errors="replace"))


# ============================================================
# 7. INSPECT STUDY README
# ============================================================

print("\n\n" + "=" * 85)
print("3. STUDY README")
print("=" * 85)

readme_file = FILES["Study README"]

if readme_file.exists():

    print(readme_file.read_text(encoding="utf-8", errors="replace"))


# ============================================================
# 8. INSPECT PATIENT CLINICAL DATA
# ============================================================

print("\n\n" + "=" * 85)
print("4. PATIENT CLINICAL DATA")
print("=" * 85)

patient_file = FILES["Patient clinical data"]

if patient_file.exists():

    patients = read_tsv(patient_file)

    print(f"\nNumber of patient records: {len(patients):,}")
    print(f"Number of columns: {len(patients.columns)}")

    print("\nCOLUMN NAMES:")
    for column in patients.columns:
        print(f"  - {column}")

    print("\nFIRST FIVE RECORDS:")
    print(patients.head().to_string(index=False))

    # Search for important clinical variables
    clinical_keywords = [
        "patient",
        "survival",
        "overall_survival",
        "vital_status",
        "age",
        "stage",
        "brca",
        "fraction_genome_altered",
    ]

    print("\nIMPORTANT CLINICAL COLUMNS:")

    for column in patients.columns:
        if any(
            keyword in column.lower().replace(" ", "_")
            for keyword in clinical_keywords
        ):
            print(f"  - {column}")

    # Missing-value summary for likely relevant columns
    relevant_columns = [
        column for column in patients.columns
        if any(
            keyword in column.lower().replace(" ", "_")
            for keyword in [
                "survival",
                "age",
                "stage",
                "brca",
                "fraction_genome_altered",
            ]
        )
    ]

    if relevant_columns:
        print("\nMISSING VALUES IN IMPORTANT CLINICAL COLUMNS:")

        for column in relevant_columns:
            missing = patients[column].isna().sum()
            percent = missing / len(patients) * 100 if len(patients) else 0

            print(
                f"  {column}: {missing:,} missing "
                f"({percent:.1f}%)"
            )


# ============================================================
# 9. INSPECT SAMPLE CLINICAL DATA
# ============================================================

print("\n\n" + "=" * 85)
print("5. SAMPLE CLINICAL DATA")
print("=" * 85)

sample_file = FILES["Sample clinical data"]

if sample_file.exists():

    samples = read_tsv(sample_file)

    print(f"\nNumber of sample records: {len(samples):,}")
    print(f"Number of columns: {len(samples.columns)}")

    print("\nCOLUMN NAMES:")
    for column in samples.columns:
        print(f"  - {column}")

    print("\nFIRST FIVE RECORDS:")
    print(samples.head().to_string(index=False))

    print("\nPOSSIBLE SAMPLE AND PATIENT ID COLUMNS:")

    for column in samples.columns:
        if any(
            keyword in column.lower()
            for keyword in ["sample", "patient", "case"]
        ):
            print(f"  - {column}")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n\n" + "=" * 85)
print("INSPECTION COMPLETE")
print("=" * 85)

print(
    "\nNext step: verify mutation coordinates, reference/alternate "
    "alleles, genome build, and sample matching before generating SBS96."
)
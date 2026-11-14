from pathlib import Path
import pandas as pd

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "results" / "final_brca_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("BRCA1/BRCA2 ANNOTATION VALIDATION")
print("=" * 65)

# Locate the cBioPortal mutation file
mutation_files = list(
    (PROJECT_DIR / "data").rglob("data_mutations.txt")
)

if not mutation_files:
    raise FileNotFoundError(
        "Could not find data_mutations.txt inside the project's data folder."
    )

print("\nMutation files found:")
for i, file in enumerate(mutation_files, start=1):
    print(f"{i}. {file.relative_to(PROJECT_DIR)}")

# Prefer the file inside the TCGA-OV PanCancer Atlas dataset folder
preferred_files = [
    f for f in mutation_files
    if "ov_tcga_pan_can_atlas_2018" in str(f).lower()
]

if len(preferred_files) == 1:
    maf_file = preferred_files[0]
elif len(mutation_files) == 1:
    maf_file = mutation_files[0]
else:
    raise RuntimeError(
        "\nMultiple mutation files were found. Check the paths above "
        "and select the correct TCGA-OV PanCancer Atlas file."
    )

print(f"\nUsing mutation file:\n{maf_file}")

# Read mutation data; ignore comment lines if present
maf = pd.read_csv(
    maf_file,
    sep="\t",
    comment="#",
    low_memory=False
)

print(f"\nTotal mutation records: {len(maf):,}")
print("\nAvailable columns:")
print(maf.columns.tolist())

# Identify relevant columns
gene_col = next(
    (c for c in ["Hugo_Symbol", "Gene", "gene"]
     if c in maf.columns),
    None
)

sample_col = next(
    (c for c in ["Tumor_Sample_Barcode", "Tumor_Sample", "Sample_ID"]
     if c in maf.columns),
    None
)

filter_col = "FILTER" if "FILTER" in maf.columns else None

if gene_col is None or sample_col is None:
    raise ValueError(
        "Could not identify the gene or sample column. "
        "Check the printed column names."
    )

# Extract BRCA1/BRCA2 records
brca = maf[
    maf[gene_col].astype(str).str.upper().isin(["BRCA1", "BRCA2"])
].copy()

print("\n" + "=" * 65)
print("BRCA1/BRCA2 SUMMARY")
print("=" * 65)

print(f"BRCA1/BRCA2 mutation records: {len(brca)}")
print(f"Samples with any BRCA1/BRCA2 record: "
      f"{brca[sample_col].nunique()}")

if not brca.empty:
    print("\nRecords by gene:")
    print(brca[gene_col].value_counts(dropna=False).to_string())

    print("\nUnique samples by gene:")
    print(
        brca.groupby(gene_col)[sample_col]
        .nunique()
        .to_string()
    )

    if filter_col:
        print("\nRecords by FILTER:")
        print(brca[filter_col].value_counts(dropna=False).to_string())

    for col in ["Variant_Classification", "Variant_Type"]:
        if col in brca.columns:
            print(f"\nRecords by {col}:")
            print(brca[col].value_counts(dropna=False).to_string())

    # Save all BRCA records for inspection
    brca.to_csv(
        OUTPUT_DIR / "brca1_brca2_all_records.csv",
        index=False
    )

    # Save summary tables
    gene_summary = (
        brca.groupby(gene_col)
        .agg(
            mutation_records=(sample_col, "size"),
            unique_samples=(sample_col, "nunique")
        )
        .reset_index()
    )
    gene_summary.to_csv(
        OUTPUT_DIR / "brca_gene_summary.csv",
        index=False
    )

    if filter_col:
        filter_summary = (
            brca[filter_col]
            .value_counts(dropna=False)
            .rename_axis("FILTER")
            .reset_index(name="record_count")
        )
        filter_summary.to_csv(
            OUTPUT_DIR / "brca_filter_summary.csv",
            index=False
        )

    # Print a compact preview
    preview_cols = [
        c for c in [
            sample_col, gene_col, "HGVSc", "HGVSp",
            "Variant_Classification", "Variant_Type",
            "FILTER", "Reference_Allele", "Tumor_Seq_Allele2"
        ]
        if c in brca.columns
    ]

    print("\nFirst 15 BRCA1/BRCA2 records:")
    print(brca[preview_cols].head(15).to_string(index=False))

else:
    print("\nNo BRCA1/BRCA2 records were found.")

print("\n" + "=" * 65)
print(f"Output folder:\n{OUTPUT_DIR}")
print("=" * 65)
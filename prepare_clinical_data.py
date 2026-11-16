from pathlib import Path
import pandas as pd

PROJECT = Path(r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures")
DATA = PROJECT / "data" / "raw" / "ov_tcga_pan_can_atlas_2018" / "ov_tcga_pan_can_atlas_2018"
RESULTS = PROJECT / "results" / "all_diagnostics"
RESULTS.mkdir(parents=True, exist_ok=True)

# Load files
patient = pd.read_csv(DATA / "data_clinical_patient.txt", sep="\t", comment="#", low_memory=False)
sample = pd.read_csv(DATA / "data_clinical_sample.txt", sep="\t", comment="#", low_memory=False)
mutations = pd.read_csv(DATA / "data_mutations.txt", sep="\t", comment="#", low_memory=False)
counts = pd.read_csv(
    PROJECT / "results" / "sbs96_counts_by_sample_cosmic_order.csv",
    index_col=0
)

# Standardize IDs
patient["PATIENT_ID_CLEAN"] = patient["PATIENT_ID"].astype(str).str.strip()
sample["SAMPLE_ID_CLEAN"] = sample["SAMPLE_ID"].astype(str).str.strip()
sample["PATIENT_ID_CLEAN"] = sample["PATIENT_ID"].astype(str).str.strip()

mutation_samples = pd.Index(counts.index.astype(str).str.strip())

# Keep only samples with SBS96 counts and match exact sample barcodes
matched_sample = sample[
    sample["SAMPLE_ID_CLEAN"].isin(mutation_samples)
].copy()

# Remove possible duplicate records before merging
matched_sample = matched_sample.drop_duplicates("SAMPLE_ID_CLEAN")
patient_unique = patient.drop_duplicates("PATIENT_ID_CLEAN")

clinical = matched_sample.merge(
    patient_unique,
    on="PATIENT_ID_CLEAN",
    how="left",
    suffixes=("_sample", "_patient"),
    validate="many_to_one"
)

# Report matching and clinical-data availability
print("=" * 60)
print("CLINICAL MATCHING")
print("=" * 60)
print(f"SBS96 samples: {len(mutation_samples)}")
print(f"Matched clinical samples: {clinical['SAMPLE_ID_CLEAN'].nunique()}")
print(f"Rows without patient clinical data: {clinical['PATIENT_ID_CLEAN'].isna().sum()}")

for col in ["AGE", "OS_STATUS", "OS_MONTHS",
            "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"]:
    if col in clinical.columns:
        print(f"{col}: missing {clinical[col].isna().sum()}/{len(clinical)}")

# Extract BRCA1/BRCA2 mutation records for later annotation
gene_col = next(
    (c for c in ["Hugo_Symbol", "HUGO_SYMBOL", "Gene", "Gene_Symbol"]
     if c in mutations.columns),
    None
)

if gene_col is None:
    print("\nCould not identify the gene column in the mutation file.")
    print("Available columns:", mutations.columns.tolist())
else:
    brca = mutations[
        mutations[gene_col].astype(str).str.upper().isin(["BRCA1", "BRCA2"])
    ].copy()

    brca.to_csv(RESULTS / "brca1_brca2_mutation_records.csv", index=False)
    print(f"\nBRCA1/BRCA2 mutation records: {len(brca)}")
    print("Saved BRCA mutation records.")

# Save clinical table and audit
clinical.to_csv(RESULTS / "matched_clinical_data.csv", index=False)

audit = pd.DataFrame({
    "metric": [
        "SBS96 samples",
        "Matched clinical samples",
        "Matched patients",
        "Rows with age",
        "Rows with OS status",
        "Rows with OS months"
    ],
    "value": [
        len(mutation_samples),
        clinical["SAMPLE_ID_CLEAN"].nunique(),
        clinical["PATIENT_ID_CLEAN"].nunique(),
        clinical["AGE"].notna().sum() if "AGE" in clinical else 0,
        clinical["OS_STATUS"].notna().sum() if "OS_STATUS" in clinical else 0,
        clinical["OS_MONTHS"].notna().sum() if "OS_MONTHS" in clinical else 0
    ]
})
audit.to_csv(RESULTS / "clinical_matching_audit.csv", index=False)

print(f"\nSaved files in:\n{RESULTS}")
print("Next: inspect the output before running survival analysis.")

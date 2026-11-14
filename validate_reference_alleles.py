from pathlib import Path

import pandas as pd
from pyfaidx import Fasta


# Project directories
PROJECT_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "ov_tcga_pan_can_atlas_2018"
    / "ov_tcga_pan_can_atlas_2018"
)

REFERENCE_PATH = (
    PROJECT_DIR
    / "data"
    / "Reference"
    / "GRCh37_primary_assembly.fa"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MUTATION_FILE = DATA_DIR / "data_mutations.txt"


# Load mutation data
print("Loading mutation data...")

mutations = pd.read_csv(
    MUTATION_FILE,
    sep="\t",
    comment="#",
    low_memory=False
)

print(f"Total mutation records: {len(mutations):,}")


# Keep single-base substitutions
sbs = mutations[
    (mutations["Variant_Type"] == "SNP")
    & mutations["Reference_Allele"].isin(["A", "C", "G", "T"])
    & mutations["Tumor_Seq_Allele2"].isin(["A", "C", "G", "T"])
    & (mutations["Reference_Allele"] != mutations["Tumor_Seq_Allele2"])
].copy()

sbs["Chromosome"] = (
    sbs["Chromosome"]
    .astype(str)
    .str.replace(r"^chr", "", regex=True)
)

sbs["Start_Position"] = pd.to_numeric(
    sbs["Start_Position"],
    errors="coerce"
)

sbs = sbs.dropna(subset=["Start_Position"]).copy()
sbs["Start_Position"] = sbs["Start_Position"].astype(int)

print(f"Valid single-base substitutions: {len(sbs):,}")


# Open the reference genome
print("\nOpening GRCh37 reference genome...")

genome = Fasta(
    str(REFERENCE_PATH),
    as_raw=True
)

# Check reference alleles
print("Comparing mutation alleles with reference genome...")

matches = 0
mismatches = 0
unavailable = 0

mismatch_examples = []
unavailable_examples = []

for row in sbs[
    [
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Sample_Barcode",
    ]
].itertuples(index=False):

    chrom, position, ref_allele, sample = row

    # Ignore chromosomes absent from the FASTA
    if chrom not in genome:
        unavailable += 1

        if len(unavailable_examples) < 10:
            unavailable_examples.append(
                (chrom, position, ref_allele, sample)
            )

        continue

    # MAF coordinates are 1-based.
    # Python slicing is 0-based.
    observed = str(
        genome[chrom][position - 1:position]
    ).upper()

    if len(observed) != 1 or observed not in "ACGT":
        unavailable += 1

        if len(unavailable_examples) < 10:
            unavailable_examples.append(
                (chrom, position, ref_allele, sample)
            )

    elif observed == ref_allele:
        matches += 1

    else:
        mismatches += 1

        if len(mismatch_examples) < 10:
            mismatch_examples.append(
                (
                    chrom,
                    position,
                    ref_allele,
                    observed,
                    sample,
                )
            )

genome.close()


# Report results
checked = matches + mismatches

print("\n========== VALIDATION RESULTS ==========")
print(f"Matching reference alleles: {matches:,}")
print(f"Mismatching reference alleles: {mismatches:,}")
print(f"Unavailable/ambiguous positions: {unavailable:,}")
print(f"Successfully compared: {checked:,}")

if checked > 0:
    print(f"Match percentage: {matches / checked * 100:.2f}%")

print("\nFirst mismatches:")
for item in mismatch_examples:
    print(item)

print("\nFirst unavailable/ambiguous positions:")
for item in unavailable_examples:
    print(item)


# Save a compact summary
summary = pd.DataFrame(
    {
        "Metric": [
            "Input mutation records",
            "Single-base substitutions",
            "Matching reference alleles",
            "Mismatching reference alleles",
            "Unavailable/ambiguous positions",
            "Successfully compared",
        ],
        "Count": [
            len(mutations),
            len(sbs),
            matches,
            mismatches,
            unavailable,
            checked,
        ],
    }
)

output_path = RESULTS_DIR / "reference_validation_summary.csv"
summary.to_csv(output_path, index=False)

print(f"\nSummary saved to: {output_path}")



from pathlib import Path
from collections import Counter

import pandas as pd
from pyfaidx import Fasta


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

MUTATION_FILE = DATA_DIR / "data_mutations.txt"

REFERENCE_FILE = (
    PROJECT_DIR
    / "data"
    / "Reference"
    / "GRCh37_primary_assembly.fa"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


BASES = ["A", "C", "G", "T"]

SUBSTITUTIONS = [
    "C>A", "C>G", "C>T",
    "T>A", "T>C", "T>G"
]

CHANNELS = [
    f"{left}[{substitution}]{right}"
    for substitution in SUBSTITUTIONS
    for left in BASES
    for right in BASES
]

assert len(CHANNELS) == 96
print(f"Defined mutation categories: {len(CHANNELS)}")

#Load mutation data

print("\nLoading mutation data...")

mutations = pd.read_csv(
    MUTATION_FILE,
    sep="\t",
    comment="#",
    low_memory=False
)

print(f"Total mutation records: {len(mutations):,}")

# Retain single-base substitutions only.
sbs = mutations[
    (mutations["Variant_Type"] == "SNP")
    & mutations["Reference_Allele"].isin(BASES)
    & mutations["Tumor_Seq_Allele2"].isin(BASES)
    & (
        mutations["Reference_Allele"]
        != mutations["Tumor_Seq_Allele2"]
    )
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

sbs = sbs.dropna(
    subset=[
        "Chromosome",
        "Start_Position",
        "Tumor_Sample_Barcode"
    ]
).copy()

sbs["Start_Position"] = sbs["Start_Position"].astype(int)

print(f"Valid SNP records selected: {len(sbs):,}")


#Open the existing GRCh37 reference
print("\nOpening GRCh37 reference genome")

genome = Fasta(
    str(REFERENCE_FILE),
    as_raw=True
)

# Initialise all 96 categories for every sample.
sample_counts = {}
sample_totals = Counter()

processed = 0
reference_mismatches = 0
unavailable_contexts = 0

mismatch_examples = []
unavailable_examples = []

#Retrieve context and count mutations

print("\nBuilding SBS96 matrix...")

for row in sbs[
    [
        "Chromosome",
        "Start_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Tumor_Sample_Barcode"
    ]
].itertuples(index=False, name=None):

    chrom, position, ref, alt, sample = row

    # Keep the exact sample barcode from the mutation file.
    if sample not in sample_counts:
        sample_counts[sample] = Counter()

    # Need one reference base on each side of the mutation.
    if chrom not in genome:
        unavailable_contexts += 1

        if len(unavailable_examples) < 5:
            unavailable_examples.append(
                (chrom, position, sample, "Chromosome not found")
            )
        continue

    chromosome_length = len(genome[chrom])

    if position <= 1 or position >= chromosome_length:
        unavailable_contexts += 1

        if len(unavailable_examples) < 5:
            unavailable_examples.append(
                (chrom, position, sample, "Chromosome boundary")
            )
        continue

    # MAF coordinates are 1-based.
    # Fetch 3 bases: preceding base, mutation, following base.
    context = str(
        genome[chrom][position - 2:position + 1]
    ).upper()

    if len(context) != 3 or any(base not in BASES for base in context):
        unavailable_contexts += 1

        if len(unavailable_examples) < 5:
            unavailable_examples.append(
                (chrom, position, sample, context)
            )
        continue

    left, observed_ref, right = context

    # Confirm the central reference base matches the MAF.
    if observed_ref != ref:
        reference_mismatches += 1

        if len(mismatch_examples) < 5:
            mismatch_examples.append(
                (chrom, position, ref, observed_ref, sample)
            )
        continue

    # Orient substitutions to the pyrimidine reference:
    # C or T. For A/G reference bases, reverse-complement
    # the context and complement both mutation alleles.
    if ref in ["A", "G"]:
        complement = {
            "A": "T",
            "C": "G",
            "G": "C",
            "T": "A"
        }

        left, right = complement[right], complement[left]
        ref = complement[ref]
        alt = complement[alt]

    substitution = f"{ref}>{alt}"
    channel = f"{left}[{substitution}]{right}"

    if channel not in CHANNELS:
        raise ValueError(f"Unexpected SBS96 channel: {channel}")

    sample_counts[sample][channel] += 1
    sample_totals[sample] += 1
    processed += 1


genome.close()

#Save the sample-by-96 matrix

print("\nSaving SBS96 matrix...")

matrix = pd.DataFrame.from_dict(
    sample_counts,
    orient="index"
).reindex(columns=CHANNELS, fill_value=0)

matrix.index.name = "Tumor_Sample_Barcode"

# Categories absent from a sample should have a count of zero.
matrix = matrix.fillna(0).astype(int)

matrix_path = RESULTS_DIR / "sbs96_counts_by_sample.csv"
matrix.to_csv(matrix_path)


# Save the total usable SBS count for each sample.
totals = pd.DataFrame({
    "Tumor_Sample_Barcode": matrix.index,
    "Usable_SBS_Count": matrix.sum(axis=1).values
})

totals_path = RESULTS_DIR / "sbs_counts_by_sample.csv"
totals.to_csv(totals_path, index=False)

#Validate matrix totals

matrix_total = int(matrix.to_numpy().sum())

summary = pd.DataFrame({
    "Metric": [
        "Input mutation records",
        "SNP records selected",
        "Mutations processed into SBS96",
        "Reference mismatches during context extraction",
        "Unavailable or ambiguous contexts",
        "Number of samples in matrix",
        "Number of SBS96 channels",
        "Total counts in matrix",
        "Matrix totals match processed mutations"
    ],
    "Value": [
        len(mutations),
        len(sbs),
        processed,
        reference_mismatches,
        unavailable_contexts,
        len(matrix),
        matrix.shape[1],
        matrix_total,
        matrix_total == processed
    ]
})

summary_path = RESULTS_DIR / "sbs96_validation_summary.csv"
summary.to_csv(summary_path, index=False)

print(f"Samples in matrix: {len(matrix):,}")
print(f"Mutation categories: {matrix.shape[1]}")
print(f"Mutations processed: {processed:,}")
print(f"Reference mismatches: {reference_mismatches:,}")
print(f"Unavailable/ambiguous contexts: {unavailable_contexts:,}")
print(f"Total counts in matrix: {matrix_total:,}")
print(f"Matrix totals match processed mutations: {matrix_total == processed}")

print("\nFirst 5 rows and first 8 categories:")
print(matrix.iloc[:5, :8])

print("\nFiles saved:")
print(matrix_path)
print(totals_path)
print(summary_path)

if mismatch_examples:
    print("\nExample reference mismatches:")
    for example in mismatch_examples:
        print(example)

if unavailable_examples:
    print("\nExample unavailable contexts:")
    for example in unavailable_examples:
        print(example)

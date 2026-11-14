from pyfaidx import Fasta

reference_path = (
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
    r"\data\Reference\GRCh37_primary_assembly.fa"
)

genome = Fasta(reference_path, as_raw=True)

chromosomes = {
    "1": [1, 1001, 1000000, 10000000],
    "2": [1, 1001, 1000000, 10000000],
    "10": [1, 1001, 1000000, 10000000],
    "17": [1, 1001, 1000000, 10000000],
    "X": [1, 1001, 1000000, 10000000],
}

for chrom, positions in chromosomes.items():
    print(f"\nChromosome {chrom}")

    for position in positions:
        # Convert the 1-based genomic position to a Python slice.
        sequence = str(genome[chrom][position - 1:position + 49]).upper()

        acgt_count = sum(base in "ACGT" for base in sequence)
        n_count = sequence.count("N")

        print(
            f"Position {position}: "
            f"{sequence[:20]} | "
            f"ACGT={acgt_count}, N={n_count}"
        )

genome.close()
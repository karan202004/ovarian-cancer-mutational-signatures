from pathlib import Path
from pyfaidx import Fasta

reference_path = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures\data\Reference\GRCh37_primary_assembly.fa"
)

print("Indexing GRCh37 reference genome...")
print("This may take a few minutes.")

genome = Fasta(
    str(reference_path),
    indexname=str(reference_path) + ".fai",
    as_raw=True,
    sequence_always_upper=True,
)

print("\nReference indexed successfully!")
print("Total sequences:", len(genome.keys()))
print("First 15 sequence names:", list(genome.keys())[:15])

if "1" in genome:
    print("\nChromosome 1 length:", len(genome["1"]))
    print("First 20 bases:", genome["1"][:20])
else:
    print("\nChromosome '1' was not found. Check the sequence names.")

genome.close()
import gzip
from pathlib import Path

reference = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures\data\Reference\Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
)

sequence_count = 0
headers = []

print("Scanning reference FASTA headers...")

with gzip.open(reference, "rt") as file:
    for line in file:
        if line.startswith(">"):
            sequence_count += 1

            if len(headers) < 10:
                headers.append(line.strip())

print(f"\nTotal sequences found: {sequence_count}")
print("\nFirst sequence headers:")

for header in headers:
    print(header)

print("\nReference inspection complete.")
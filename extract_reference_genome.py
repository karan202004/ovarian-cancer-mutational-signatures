import gzip
import shutil
from pathlib import Path

reference_dir = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures\data\Reference"
)

compressed_file = reference_dir / "Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
output_file = reference_dir / "Homo_sapiens.GRCh37.dna.primary_assembly.fa"

if not compressed_file.exists():
    raise FileNotFoundError(f"Reference archive not found: {compressed_file}")

if output_file.exists():
    print(f"Output already exists: {output_file}")
    print("Stopping to avoid overwriting an existing file.")
else:
    print("Extracting the complete reference genome...")
    print("This may take several minutes.")

    with gzip.open(compressed_file, "rb") as source:
        with open(output_file, "wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)

    print("\nExtraction complete!")
    print(f"Output: {output_file}")
    print(f"Size: {output_file.stat().st_size / (1024**3):.2f} GB")
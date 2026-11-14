import gzip
import shutil
from pathlib import Path

reference_dir = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures\data\Reference"
)

source = reference_dir / "Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
destination = reference_dir / "GRCh37_primary_assembly.fa"

if not source.is_file():
    raise FileNotFoundError(f"Reference archive not found: {source}")

if destination.exists():
    raise FileExistsError(
        f"Output already exists; not overwriting: {destination}"
    )

print("Decompressing the existing reference archive...")
print("This may take several minutes.")

with gzip.open(source, "rb") as compressed:
    with destination.open("wb") as output:
        shutil.copyfileobj(compressed, output, length=1024 * 1024)

print("\nDecompression complete.")
print(f"Output: {destination}")
print(f"Size: {destination.stat().st_size / (1024**3):.2f} GB")
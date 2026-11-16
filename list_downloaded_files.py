from pathlib import Path

# Folder containing the extracted cBioPortal study
data_dir = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
    r"\data\raw\ov_tcga_pan_can_atlas_2018"
)

print("Files in the TCGA-OV PAN-CANCER ATLAS download")

# Find files recursively, including those in nested folders
files = sorted(path for path in data_dir.rglob("*") if path.is_file())

print(f"\nTotal number of files: {len(files)}\n")

for file in files:
    print(f"{file.name}  |  {file.stat().st_size / 1024:.2f} KB")

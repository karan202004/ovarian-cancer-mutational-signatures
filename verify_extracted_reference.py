from pathlib import Path

PROJECT = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

for path in PROJECT.rglob("data_clinical_patient.txt"):
    print("Patient clinical file:", path)

for path in PROJECT.rglob("data_clinical_sample.txt"):
    print("Sample clinical file:", path)
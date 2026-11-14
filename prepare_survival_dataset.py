
from pathlib import Path
import re
import pandas as pd
import numpy as np

# ============================================================
# 1. PATHS
# ============================================================
PROJECT = Path(r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures")
RESULTS = PROJECT / "results"
DIAG = RESULTS / "all_diagnostics"
OUT = DIAG / "survival_preparation"
OUT.mkdir(parents=True, exist_ok=True)

CLINICAL_FILE = DIAG / "matched_clinical_data.csv"
SBS_MATRIX_FILE = RESULTS / "sbs96_counts_by_sample_cosmic_order.csv"
BRCA_FILE = DIAG / "brca1_brca2_mutation_records.csv"

# ============================================================
# 2. HELPER: FIND SAMPLE ID AND SBS3 COLUMN
# ============================================================
def load_signature_file(path):
    df = pd.read_csv(path)

    # Sample ID may be a column or the first saved index column
    sample_candidates = [
        "SAMPLE_ID_CLEAN", "SAMPLE_ID", "Tumor_Sample_Barcode",
        "sample_id", "sample", "Sample", "sample_barcode"
    ]
    sample_col = next((c for c in sample_candidates if c in df.columns), None)

    if sample_col is None:
        first_col = df.columns[0]
        if first_col.lower().startswith("unnamed"):
            sample_col = first_col
        else:
            raise ValueError(
                f"Cannot identify sample ID in {path.name}. "
                f"Columns: {df.columns.tolist()}"
            )

    df = df.rename(columns={sample_col: "SAMPLE_ID"})
    df["SAMPLE_ID"] = df["SAMPLE_ID"].astype(str).str.strip()

    # Identify SBS3 contribution column
    sbs3_col = next(
        (c for c in df.columns if c.strip().upper() == "SBS3"),
        None
    )
    if sbs3_col is None:
        raise ValueError(
            f"No SBS3 column in {path.name}. Columns: {df.columns.tolist()}"
        )

    df["SBS3_PERCENT"] = pd.to_numeric(df[sbs3_col], errors="coerce")
    return df[["SAMPLE_ID", "SBS3_PERCENT"]].drop_duplicates("SAMPLE_ID")


# ============================================================
# 3. LOAD CLINICAL DATA AND SBS MUTATION BURDEN
# ============================================================
clinical = pd.read_csv(CLINICAL_FILE, low_memory=False)
counts = pd.read_csv(SBS_MATRIX_FILE, index_col=0)

clinical["SAMPLE_ID"] = clinical["SAMPLE_ID_CLEAN"].astype(str).str.strip()
counts.index = counts.index.astype(str).str.strip()

# Counts are 96 SBS channels per sample; row sum = SBS mutation burden
burden = counts.sum(axis=1).rename("SBS_MUTATION_BURDEN").reset_index()
burden.columns = ["SAMPLE_ID", "SBS_MUTATION_BURDEN"]

data = clinical.merge(
    burden, on="SAMPLE_ID", how="inner", validate="one_to_one"
)

# Load main 13-signature SBS3 estimates
main_file = RESULTS / "signature_contributions_exploratory.csv"
main_sbs3 = load_signature_file(main_file).rename(
    columns={"SBS3_PERCENT": "SBS3_MAIN_PERCENT"}
)

data = data.merge(
    main_sbs3, on="SAMPLE_ID", how="left", validate="one_to_one"
)

# ============================================================
# 4. CHECK SURVIVAL CODING
# ============================================================
if "OS_STATUS" not in data.columns or "OS_MONTHS" not in data.columns:
    raise ValueError("OS_STATUS or OS_MONTHS is missing from clinical data.")

print("\n" + "=" * 65)
print("RAW OVERALL-SURVIVAL STATUS VALUES")
print("=" * 65)
print(data["OS_STATUS"].value_counts(dropna=False).to_string())

print("\nOverall-survival time summary:")
print(pd.to_numeric(data["OS_MONTHS"], errors="coerce").describe().to_string())

# Typical cBioPortal coding is '1:DECEASED' and '0:LIVING'.
# Inspect first, then parse conservatively.
def parse_os_event(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()
    if re.match(r"^1\s*:", s) or s in {"DECEASED", "DEAD", "1"}:
        return 1
    if re.match(r"^0\s*:", s) or s in {"LIVING", "ALIVE", "0"}:
        return 0
    return np.nan

data["OS_EVENT"] = data["OS_STATUS"].apply(parse_os_event)
data["OS_MONTHS"] = pd.to_numeric(data["OS_MONTHS"], errors="coerce")
data["AGE"] = pd.to_numeric(data["AGE"], errors="coerce")

# A usable survival record needs event/censor status and nonnegative time
data["OS_VALID"] = (
    data["OS_EVENT"].notna()
    & data["OS_MONTHS"].notna()
    & (data["OS_MONTHS"] >= 0)
)

print("\n" + "=" * 65)
print("SURVIVAL DATA AVAILABILITY")
print("=" * 65)
print(f"Matched samples: {len(data)}")
print(f"Usable OS records: {data['OS_VALID'].sum()}")
print(f"Events/deaths among usable records: "
      f"{int((data.loc[data['OS_VALID'], 'OS_EVENT'] == 1).sum())}")
print(f"Censored/alive among usable records: "
      f"{int((data.loc[data['OS_VALID'], 'OS_EVENT'] == 0).sum())}")
print(f"Age available: {data['AGE'].notna().sum()}")
print(f"Missing age among usable OS records: "
      f"{data.loc[data['OS_VALID'], 'AGE'].isna().sum()}")

# ============================================================
# 5. BRCA1/2 ANNOTATION AUDIT — DO NOT ASSUME PATHOGENICITY
# ============================================================
print("\n" + "=" * 65)
print("BRCA1/BRCA2 MUTATION ANNOTATION")
print("=" * 65)

if BRCA_FILE.exists():
    brca = pd.read_csv(BRCA_FILE, low_memory=False)

    print(f"BRCA mutation records: {len(brca)}")
    print("Available annotation columns:")
    print(brca.columns.tolist())

    for col in [
        "Variant_Classification", "HGVSp_Short", "HGVSc",
        "Consequence", "IMPACT", "FILTER"
    ]:
        if col in brca.columns:
            print(f"\n{col}:")
            print(brca[col].value_counts(dropna=False).head(15).to_string())

    # Save all records for careful annotation later
    brca.to_csv(OUT / "BRCA1_BRCA2_annotation_audit.csv", index=False)
else:
    print("BRCA mutation file not found; skipping annotation audit.")

# ============================================================
# 6. COMPARE SBS3 ESTIMATES ACROSS SIGNATURE PANELS
# ============================================================
panel_files = {
    "core5": DIAG / "core5_signature_contributions.csv",
    "expanded8": DIAG / "expanded8_signature_contributions.csv",
    "current13": DIAG / "signature_contributions_exploratory.csv",
}

# File names vary between scripts, so search the diagnostics folder
# for CSVs that contain an SBS3 column and are not fit-quality tables.
found_panels = {}
for path in DIAG.glob("*.csv"):
    if "contribution" not in path.name.lower():
        continue
    if "quality" in path.name.lower() or "similarity" in path.name.lower():
        continue
    try:
        sbs3_df = load_signature_file(path)
        key = path.stem.replace("signature_contributions_", "")
        found_panels[key] = sbs3_df.rename(
            columns={"SBS3_PERCENT": f"SBS3_{key}_PERCENT"}
        )
    except (ValueError, pd.errors.ParserError):
        continue

for key, path in panel_files.items():
    if path.exists():
        try:
            found_panels[key] = load_signature_file(path).rename(
                columns={"SBS3_PERCENT": f"SBS3_{key}_PERCENT"}
            )
        except ValueError:
            pass

for key, panel_df in found_panels.items():
    col = f"SBS3_{key}_PERCENT"
    if col not in data.columns:
        data = data.merge(
            panel_df, on="SAMPLE_ID", how="left", validate="one_to_one"
        )

panel_cols = [
    c for c in data.columns
    if c.startswith("SBS3_") and c.endswith("_PERCENT")
]

print("\n" + "=" * 65)
print("SBS3 PANEL ROBUSTNESS")
print("=" * 65)
if panel_cols:
    print(data[panel_cols].describe().round(2).to_string())

    if len(panel_cols) >= 2:
        print("\nCorrelation between panel estimates:")
        print(data[panel_cols].corr().round(3).to_string())

        data["SBS3_PANEL_RANGE"] = (
            data[panel_cols].max(axis=1) - data[panel_cols].min(axis=1)
        )
        print("\nPer-sample range across panels (percentage points):")
        print(data["SBS3_PANEL_RANGE"].describe().round(2).to_string())
else:
    print("No SBS3 panel files were identified. Main estimate is retained.")

# ============================================================
# 7. SAVE DATA AND AUDIT REPORT
# ============================================================
data.to_csv(OUT / "survival_analysis_dataset_preliminary.csv", index=False)

audit = pd.DataFrame({
    "metric": [
        "Matched samples",
        "Samples with SBS3 main estimate",
        "Samples with valid OS record",
        "Deaths among valid OS records",
        "Age available among valid OS records",
        "Samples with valid OS and SBS3",
        "Samples with valid OS, SBS3, and age",
        "Samples with missing/invalid OS status or time",
    ],
    "value": [
        len(data),
        data["SBS3_MAIN_PERCENT"].notna().sum(),
        int(data["OS_VALID"].sum()),
        int((data.loc[data["OS_VALID"], "OS_EVENT"] == 1).sum()),
        int(data.loc[data["OS_VALID"], "AGE"].notna().sum()),
        int((data["OS_VALID"] & data["SBS3_MAIN_PERCENT"].notna()).sum()),
        int((data["OS_VALID"] & data["SBS3_MAIN_PERCENT"].notna()
             & data["AGE"].notna()).sum()),
        int((~data["OS_VALID"]).sum()),
    ]
})
audit.to_csv(OUT / "survival_data_audit.csv", index=False)

print("\n" + "=" * 65)
print("SAVED OUTPUTS")
print("=" * 65)
print(OUT / "survival_analysis_dataset_preliminary.csv")
print(OUT / "survival_data_audit.csv")
print(OUT / "BRCA1_BRCA2_annotation_audit.csv")
print("\nDo not run Cox regression until survival coding and BRCA annotation are reviewed.")

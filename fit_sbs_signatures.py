from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

PROJECT_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

MATRIX_FILE = (
    PROJECT_DIR
    / "results"
    / "sbs96_counts_by_sample_cosmic_order.csv"
)

COSMIC_FILE = (
    PROJECT_DIR
    / "data"
    / "Reference"
    / "COSMIC_catalogue-signatures_SBS96_v3.6"
    / "COSMIC_v3.6_SBS_GRCh37.txt"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

counts_df = pd.read_csv(MATRIX_FILE, index_col=0)
cosmic_df = pd.read_csv(COSMIC_FILE, sep="\t", index_col=0)

print("Tumour matrix:", counts_df.shape)
print("COSMIC reference:", cosmic_df.shape)

# Confirm the channels match exactly
if not counts_df.columns.equals(cosmic_df.index):
    raise ValueError(
        "SBS96 channels do not match. Check channel names and order."
    )

# Ensure counts are valid integers
counts_df = counts_df.apply(pd.to_numeric, errors="raise")

if (counts_df < 0).any().any():
    raise ValueError("Negative mutation counts found.")

if not np.allclose(counts_df.to_numpy(),
                   np.round(counts_df.to_numpy())):
    raise ValueError("Mutation counts are not integers.")


#Select a small exploratory signature panel
candidate_signatures = [
    "SBS1",
    "SBS2",
    "SBS3",
    "SBS5",
    "SBS8",
    "SBS13",
    "SBS17a",
    "SBS17b",
    "SBS18",
    "SBS35",
    "SBS40a",
    "SBS40b",
    "SBS40c",
]

missing = [
    sig for sig in candidate_signatures
    if sig not in cosmic_df.columns
]

if missing:
    raise ValueError(
        f"These signatures are missing from COSMIC: {missing}"
    )

# Each column is a signature probability profile.
signature_matrix = cosmic_df[candidate_signatures].to_numpy(
    dtype=float
)


#Fit each tumour using non-negative least squares
contribution_rows = []
quality_rows = []

for sample_id, row in counts_df.iterrows():
    observed = row.to_numpy(dtype=float)

    # NNLS estimates non-negative signature contributions.
    coefficients, residual_norm = nnls(
        signature_matrix,
        observed
    )

    reconstructed = signature_matrix @ coefficients

    total_mutations = observed.sum()
    total_attributed = coefficients.sum()

    # Cosine similarity between observed and reconstructed profiles.
    denominator = (
        np.linalg.norm(observed) * np.linalg.norm(reconstructed)
    )

    cosine_fit = (
        np.dot(observed, reconstructed) / denominator
        if denominator > 0 else np.nan
    )

    # Relative reconstruction error.
    relative_error = (
        np.linalg.norm(observed - reconstructed)
        / np.linalg.norm(observed)
        if np.linalg.norm(observed) > 0 else np.nan
    )

    # Convert coefficients into percentages of attributed mutations.
    if total_attributed > 0:
        percentages = 100 * coefficients / total_attributed
    else:
        percentages = np.zeros_like(coefficients)

    contribution_rows.append(
        pd.Series(percentages, index=candidate_signatures,
                  name=sample_id)
    )

    quality_rows.append({
        "sample_id": sample_id,
        "total_SBS_mutations": int(total_mutations),
        "total_attributed_mutations": total_attributed,
        "SBS3_percent_of_attributed": percentages[
            candidate_signatures.index("SBS3")
        ],
        "cosine_fit": cosine_fit,
        "relative_reconstruction_error": relative_error,
        "nnls_residual_norm": residual_norm,
    })


#Save results
contributions_df = pd.DataFrame(contribution_rows)
quality_df = pd.DataFrame(quality_rows).set_index("sample_id")

contributions_df.to_csv(
    RESULTS_DIR / "signature_contributions_exploratory.csv"
)

quality_df.to_csv(
    RESULTS_DIR / "signature_fit_quality_exploratory.csv"
)

print("\nSignature fitting completed.")
print("Contribution results saved.")
print("Fit quality results saved.")

print("\nFit quality summary:")
print(
    quality_df[
        ["cosine_fit", "relative_reconstruction_error"]
    ].describe()
)

print("\nSBS3 contribution summary (% of attributed mutations):")
print(
    quality_df["SBS3_percent_of_attributed"].describe()
)

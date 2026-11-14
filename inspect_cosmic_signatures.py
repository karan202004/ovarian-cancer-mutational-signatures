
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

COSMIC_FILE = (
    PROJECT_DIR / "data" / "Reference"
    / "COSMIC_catalogue-signatures_SBS96_v3.6"
    / "COSMIC_v3.6_SBS_GRCh37.txt"
)

OUTPUT_FILE = (
    PROJECT_DIR / "results"
    / "cosmic_signature_similarity_to_sbs3.csv"
)

cosmic = pd.read_csv(COSMIC_FILE, sep="\t", index_col=0)

# Each column is a signature probability distribution.
profiles = cosmic.astype(float)

if "SBS3" not in profiles.columns:
    raise ValueError("SBS3 was not found in the COSMIC reference.")

# Cosine similarity: 1 means identical profile; 0 means no directional similarity.
sbs3 = profiles["SBS3"].to_numpy()
sbs3_norm = np.linalg.norm(sbs3)

results = []

for signature in profiles.columns:
    profile = profiles[signature].to_numpy()
    denominator = sbs3_norm * np.linalg.norm(profile)

    similarity = (
        float(np.dot(sbs3, profile) / denominator)
        if denominator > 0 else np.nan
    )

    results.append({
        "Signature": signature,
        "Cosine_Similarity_to_SBS3": similarity
    })

similarity_df = (
    pd.DataFrame(results)
    .sort_values("Cosine_Similarity_to_SBS3", ascending=False)
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
similarity_df.to_csv(OUTPUT_FILE, index=False)

print("Total signatures:", len(profiles.columns))
print("\nTop 20 profiles most similar to SBS3:")
print(similarity_df.head(20).to_string(index=False))

print("\nSaved results to:")
print(OUTPUT_FILE)

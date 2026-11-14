
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(
    r"C:\Users\karan\PycharmProjects\Ovarian_Mutational_Signatures"
)

COSMIC_FILE = (
    PROJECT_DIR / "data" / "Reference"
    / "COSMIC_catalogue-signatures_SBS96_v3.6"
    / "COSMIC_v3.6_SBS_GRCh37.txt"
)

TUMOUR_FILE = PROJECT_DIR / "results" / "sbs96_counts_by_sample.csv"

print("Loading COSMIC reference...")
cosmic = pd.read_csv(COSMIC_FILE, sep="\t", index_col=0)

print("Loading tumour SBS96 matrix...")
tumour = pd.read_csv(TUMOUR_FILE, index_col=0)

cosmic_channels = list(cosmic.index)
tumour_channels = list(tumour.columns)

print("\n========== CHANNEL VALIDATION ==========")
print("COSMIC channels:", len(cosmic_channels))
print("Tumour channels:", len(tumour_channels))

cosmic_set = set(cosmic_channels)
tumour_set = set(tumour_channels)

missing_from_tumour = sorted(cosmic_set - tumour_set)
extra_in_tumour = sorted(tumour_set - cosmic_set)

print("\nChannels in COSMIC but not tumour matrix:")
print(missing_from_tumour)

print("\nChannels in tumour matrix but not COSMIC:")
print(extra_in_tumour)

same_channels = cosmic_set == tumour_set
same_order = cosmic_channels == tumour_channels

print("\nSame 96 channel names:", same_channels)
print("Same channel order:", same_order)

if same_channels:
    # Reorder the tumour matrix to match COSMIC exactly.
    tumour_aligned = tumour.loc[:, cosmic_channels]

    print("\nChannel alignment successful.")
    print("Aligned tumour matrix shape:", tumour_aligned.shape)

    output = (
        PROJECT_DIR / "results"
        / "sbs96_counts_by_sample_cosmic_order.csv"
    )
    tumour_aligned.to_csv(output)

    print("Saved aligned matrix:")
    print(output)

else:
    print("\nSTOP: Channel names differ.")
    print("Do not perform signature fitting until the mismatch is resolved.")

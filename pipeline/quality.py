# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
from pathlib import Path


# ============================================================
# STEP 1: DEFINE PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "normalized_tickets.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "qualified_tickets.csv"
)


# ============================================================
# STEP 2: LOAD NORMALIZED TICKETS
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n--- KNOWLEDGE QUALITY CHECK ---")
print(f"Tickets received: {len(df)}")


# ============================================================
# STEP 3: DEFINE OBVIOUS JUNK RESOLUTIONS
# ============================================================

JUNK_RESOLUTIONS = [
    "done",
    "fixed",
    "resolved",
    "closed",
    "completed",
    "user confirmed",
    "issue resolved"
]


# ============================================================
# STEP 4: DEFINE QUALITY CHECK FUNCTION
# ============================================================

def check_resolution(resolution):

    # Missing resolution
    if pd.isna(resolution):
        return "LOW"

    # Convert to clean lowercase text
    resolution = str(resolution).strip().lower()

    # Empty resolution
    if resolution == "":
        return "LOW"

    # Known junk resolution
    if resolution in JUNK_RESOLUTIONS:
        return "LOW"

    # Very short resolution
    if len(resolution) < 15:
        return "LOW"

    # Passed basic checks
    return "POTENTIAL"


# ============================================================
# STEP 5: APPLY QUALITY CHECK
# ============================================================

df["knowledge_quality"] = (
    df["resolution_comment"]
    .apply(check_resolution)
)


# ============================================================
# STEP 6: SELECT QUALIFIED TICKETS
# ============================================================

qualified_tickets = df[
    df["knowledge_quality"] == "POTENTIAL"
].copy()


# ============================================================
# STEP 7: DISPLAY QUALITY SUMMARY
# ============================================================

print("\n--- QUALITY RESULTS ---")

print(f"Input tickets: {len(df)}")
print(
    f"Qualified tickets: "
    f"{len(qualified_tickets)}"
)
print(
    f"Rejected tickets: "
    f"{len(df) - len(qualified_tickets)}"
)


# ============================================================
# STEP 8: SAVE QUALIFIED TICKETS
# ============================================================

qualified_tickets.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n--- QUALITY FILTER COMPLETED ---")
print("Qualified dataset saved to:")
print(OUTPUT_FILE)
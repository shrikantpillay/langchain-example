# ============================================================
# IMPORTS
# ============================================================

import truststore
truststore.inject_into_ssl()

import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


# ============================================================
# STEP 1: LOAD THE TICKET DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "clustering_tickets.csv"
)

df = pd.read_csv(DATA_FILE)

print("\n--- DATASET LOADED ---")
print(f"Tickets loaded: {len(df)}")


# ============================================================
# STEP 2: LOAD THE EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# STEP 3: GET TICKET DESCRIPTIONS
# ============================================================

descriptions = df["description"].tolist()


# ============================================================
# STEP 4: CONVERT DESCRIPTIONS INTO EMBEDDINGS
# ============================================================

embeddings = embedding_model.encode(
    descriptions
)

print("\n--- EMBEDDINGS CREATED ---")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")


# ============================================================
# STEP 5: TEST SEMANTIC SIMILARITY
# ============================================================

# Compare the first ticket against all tickets
similarities = cosine_similarity(
    [embeddings[0]],
    embeddings
)[0]

# Store similarity scores
df["similarity"] = similarities

# Sort by highest similarity
results = df.sort_values(
    by="similarity",
    ascending=False
)

print("\n--- SIMILARITY TEST ---")
print("Comparing against:")
print(descriptions[0])

print("\nMost similar tickets:")

print(
    results[
        [
            "ticket_id",
            "description",
            "similarity"
        ]
    ].head(10)
)


# ============================================================
# STEP 6: CREATE K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42
)


# ============================================================
# STEP 7: CLUSTER THE TICKETS
# ============================================================

clusters = kmeans.fit_predict(
    embeddings
)

# Add cluster number to original dataset
df["cluster"] = clusters


# ============================================================
# STEP 8: DISPLAY CLUSTER RESULTS
# ============================================================

print("\n--- CLUSTERING RESULTS ---")

for cluster_number in sorted(df["cluster"].unique()):

    print(
        f"\n========== CLUSTER {cluster_number} =========="
    )

    cluster_tickets = df[
        df["cluster"] == cluster_number
    ]

    for _, row in cluster_tickets.iterrows():

        print(
            row["ticket_id"],
            "-",
            row["description"]
        )
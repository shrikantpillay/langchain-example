# ============================================================
# IMPORTS
# ============================================================

import truststore
truststore.inject_into_ssl()

import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# STEP 1: DEFINE PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "qualified_tickets.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "clustered_tickets.csv"
)


# ============================================================
# STEP 2: LOAD QUALIFIED TICKETS
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n--- QUALIFIED DATA LOADED ---")
print(f"Tickets loaded: {len(df)}")


# ============================================================
# STEP 3: LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# STEP 4: GET TICKET DESCRIPTIONS
# ============================================================

descriptions = (
    df["description"]
    .astype(str)
    .tolist()
)


# ============================================================
# STEP 5: CREATE EMBEDDINGS
# ============================================================

embeddings = embedding_model.encode(
    descriptions
)

print("\n--- EMBEDDINGS CREATED ---")
print(
    f"Number of embeddings: "
    f"{len(embeddings)}"
)
print(
    f"Embedding dimensions: "
    f"{len(embeddings[0])}"
)


# ============================================================
# STEP 6: FIND BEST NUMBER OF CLUSTERS
# ============================================================

best_k = None
best_score = -1

print("\n--- TESTING CLUSTER COUNTS ---")


# Maximum K cannot be greater than number of tickets - 1
max_k = min(
    10,
    len(df) - 1
)


for k in range(2, max_k + 1):

    # Create temporary K-Means model
    test_kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    # Create temporary clusters
    test_clusters = test_kmeans.fit_predict(
        embeddings
    )

    # Calculate cluster quality
    score = silhouette_score(
        embeddings,
        test_clusters
    )

    print(
        f"K = {k}, "
        f"Silhouette Score = {score:.3f}"
    )

    # Store best result
    if score > best_score:

        best_score = score
        best_k = k


# ============================================================
# STEP 7: DISPLAY BEST K
# ============================================================

print("\n--- BEST CLUSTER COUNT ---")
print(f"Best K: {best_k}")
print(
    f"Best Silhouette Score: "
    f"{best_score:.3f}"
)


# ============================================================
# STEP 8: CREATE FINAL K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42
)


# ============================================================
# STEP 9: CLUSTER ALL TICKETS
# ============================================================

clusters = kmeans.fit_predict(
    embeddings
)

df["cluster"] = clusters


# ============================================================
# STEP 10: DISPLAY CLUSTER SUMMARY
# ============================================================

print("\n--- CLUSTERING RESULTS ---")

for cluster_number in sorted(
    df["cluster"].unique()
):

    cluster_tickets = df[
        df["cluster"] == cluster_number
    ]

    print(
        f"\nCluster {cluster_number}: "
        f"{len(cluster_tickets)} tickets"
    )

    # Display first few descriptions
    for description in (
        cluster_tickets["description"]
        .head(5)
    ):

        print(
            "-",
            description
        )


# ============================================================
# STEP 11: SAVE CLUSTERED DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n--- CLUSTERING COMPLETED ---")
print("Clustered dataset saved to:")
print(OUTPUT_FILE)
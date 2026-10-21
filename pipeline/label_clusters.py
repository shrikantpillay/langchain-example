# ============================================================
# IMPORTS
# ============================================================

import json
import truststore
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# STEP 1: SETUP
# ============================================================

truststore.inject_into_ssl()
load_dotenv()


# ============================================================
# STEP 2: LOAD CLUSTERED TICKETS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "clustered_tickets.csv"
)

df = pd.read_csv(DATA_FILE)

print("\n--- CLUSTERED DATA LOADED ---")
print(f"Tickets: {len(df)}")
print(f"Clusters: {df['cluster'].nunique()}")


# ============================================================
# STEP 3: INITIALIZE LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ============================================================
# STEP 4: CREATE CLUSTER NAMING PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an IT Service Management knowledge analyst.

        You will receive ticket descriptions belonging to
        the same semantic cluster.

        Determine the common IT issue represented by
        these tickets.

        Give the cluster a short and meaningful
        IT support issue family name.

        Examples:

        VPN Connectivity
        Account Lockout
        Outlook Connectivity
        Laptop Performance
        Password Reset

        Rules:

        1. Use ONLY the descriptions provided.
        2. Do not invent information.
        3. Keep the issue family name concise.
        4. Return ONLY valid JSON.
        5.Consolidate semantically equivalent issue families. Do not merge genuinely different IT issues.

        Return:

        {{
            "issue_family": "..."
        }}
        """
    ),
    (
        "human",
        """
        TICKET DESCRIPTIONS:

        {descriptions}
        """
    )
])


# ============================================================
# STEP 5: CREATE LANGCHAIN
# ============================================================

chain = prompt | llm


# ============================================================
# STEP 6: PROCESS EACH CLUSTER
# ============================================================

cluster_names = {}

for cluster_number in sorted(
    df["cluster"].unique()
):

    # Get tickets belonging to this cluster
    cluster_df = df[
        df["cluster"] == cluster_number
    ]

    # Get descriptions
    descriptions = cluster_df[
        "description"
    ].tolist()

    # Convert descriptions to JSON text
    description_text = json.dumps(
        descriptions,
        indent=2
    )

    # Ask GPT to name the cluster
    response = chain.invoke({
        "descriptions": description_text
    })

    # Convert GPT response to Python dictionary
    result = json.loads(
        response.content
    )

    issue_family = result[
        "issue_family"
    ]

    # Store cluster name
    cluster_names[
        cluster_number
    ] = issue_family

    print(
        f"Cluster {cluster_number}"
        f" -> {issue_family}"
    )


# ============================================================
# STEP 7: ADD ISSUE FAMILY TO DATA
# ============================================================

df["issue_family"] = df[
    "cluster"
].map(
    cluster_names
)


# ============================================================
# STEP 8: SAVE LABELED TICKETS
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "labeled_tickets.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n--- LABELED DATA SAVED ---")
print(OUTPUT_FILE)